# LIBRARIES
import base64
import io
import json
import os
import pathlib
import re

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
import imageio
from flask import request
import numpy as np
import PIL.Image
import plotly.graph_objects as go
from tiled.client import from_uri
from tiled.client.cache import Cache
from skimage import draw
import skimage
import urllib.request
import uuid

import job_dispatcher
from targeted_callbacks import targeted_callback
from kwarg_editor import JSONParameterEditor
import helper_utils
from app_layout_annotator import app


#### GLOBAL PARAMS ####
DEFAULT_LABEL_CLASS = 0
MASK_OUTPUT_DIR = pathlib.Path('data/masks')
IM_OUTPUT_DIR = pathlib.Path('data/images')
USER = 'mlexchange-team'
DATA_DIR = str(os.environ['DATA_DIR'])
TILED_CLIENT = from_uri("http://host.docker.internal:8057", cache=Cache.in_memory(2e9))


def slider_style(n):
    return {0: '0', n: str(n)}


### REACTIVE COMPONENTS FOR DISPLAY FIGURE ###
@app.callback(
    [
        Output("graph", "figure"),
        Output('image-slider', 'max'),
        Output("image-slider", "value"),
        Output("image-slider", "marks")
    ],
    [
        Input("image-slider", "value"),
        Input({'type': "label-class-button", "index": dash.dependencies.ALL},
              "n_clicks_timestamp",
              ),
        Input('image-store', 'data'),
        Input('stroke-width', 'value'),
        Input('dataset-selection', 'value')
    ],
    State('masks', 'data')
)
def update_figure(image_slider_value, any_label_class_button_value, image_store_data,
                  stroke_width, dataset, masks_data):
    # read any shapes stored in browser associated with current slice
    shapes = masks_data.get(str(image_slider_value))
    # find label class value by finding button with the most recent click
    if any_label_class_button_value is None:
        label_class_value = DEFAULT_LABEL_CLASS
    else:
        label_class_value = max(
            enumerate(any_label_class_button_value),
            key=lambda t: 0 if t[1] is None else t[1],
        )[0]
    # plot the new figure
    if len(image_store_data) > 0:
        im_cache = image_store_data[list(image_store_data.keys())[image_slider_value]][1]
        print(image_store_data[list(image_store_data.keys())[image_slider_value]][0])
    else:
        im_cache = None

    # dataset selection
    print('Loading dataset')
    if dataset == 'tiled':
        image_slider_max = len(TILED_CLIENT) - 1
        if image_slider_value > image_slider_max:
            image_slider_value = 0
        im = helper_utils.make_default_figure(image_slider_value, TILED_CLIENT, shapes,
                                              stroke_color=helper_utils.class_to_color(label_class_value),
                                              stroke_width=stroke_width, image_cache=im_cache)
    else:
        np_volume = helper_utils.dcm_to_np(dataset)
        print('Dataset loaded')
        image_slider_max = len(np_volume) - 1
        if image_slider_value > image_slider_max:
            image_slider_value = 0
        im = helper_utils.make_default_figure(image_slider_value, np_volume, shapes,
                                              stroke_color=helper_utils.class_to_color(label_class_value),
                                              stroke_width=stroke_width, image_cache=im_cache)
    
    return [im, image_slider_max, image_slider_value, slider_style(image_slider_max)]


@app.callback(
    Output('masks', 'data'),
    Input("graph", "relayoutData"),
    [   
        State("image-slider", "value"),
        State('masks', 'data'),
     ]
)
def store_masks(graph_relayoutData, image_slice, masks_data):
    """
    Save shapes from figure to json Store
    """
    if graph_relayoutData is not None:
        if 'shapes' in graph_relayoutData.keys():
            masks_data[image_slice] = graph_relayoutData['shapes']
        # change shape path if shape has been updated
        elif any(["shapes" in key for key in graph_relayoutData]):
            for key in graph_relayoutData:
                if 'shapes' in key:
                    # get index of edited shape
                    shape_index_re = re.compile('shapes\[(\d*)\].path')  # fragile, probably break on dash update
                    shape_index = int(shape_index_re.match(key).group(1))
                    masks_data[str(image_slice)][shape_index]['path'] = graph_relayoutData[key]
        else:
            return dash.no_update
    return masks_data


class mask_tasks():
    """
    example use:
    Given a dictionary of masks (key is index of image masked, value is list of svg path objects),
    call:
        f_names = mask_tasks.create_masks(mask_dict), which saves numpy .dat files of the masks and returns
        the file names to you
    """

    @staticmethod  # doesn't need cls as first arg, use staticmethod
    def _path_to_indices(path):  # _ indicates this is a private method
        """From SVG path to numpy array of coordinates, each row being a (row, col) point
        """
        indices_str = [el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")]
        return np.rint(np.array(indices_str, dtype=float)).astype(int)

    @classmethod
    def _path_to_mask(self, path, class_n, shape):
        """From SVG path to a boolean array where all pixels enclosed by the path
        are True, and the other pixels are False.

        Args:
            path, str: svg string
            class_n, int: the labeled class of the path
            shape, (int, int): the shape of the labeled image
        """
        class_n = int(class_n)
        cols, rows = self._path_to_indices(path).T  # a series of vertices, need to translate into points
        # draw.line returns rr, cc
        lines = [draw.line_aa(r0, c0, r1, c1) for r0, c0, r1, c1 in zip(rows[:-1], cols[:-1], rows[1:], cols[1:])]

        # rr, cc = draw.polygon_perimeter(rows, cols) # change to draw.polygon to have filled holes
        mask = np.zeros(shape, dtype=int)
        # the plus one is because zero is taken as an unlabled pixel, so
        # all classes need to be shifted up by one for the labelling.
        for line in lines:
            mask[line[0], line[1]] = class_n + 1
        print(class_n)
        # don't want to have filled paths, just draw mask where the stroke is
        # so commented out below
        #        mask = ndimage.binary_fill_holes(mask)
        return mask

    @classmethod
    def create_save_masks(self, masks_data: dict, mask_output_dir, image_shape_list):
        """
        Create a mask file for each image in an image stack
        Args:
            masks_data: dict{image_index: [plotly_svg_shapes_for_index]}
            mask_output_dir: str or pathlib.Path pointing to where save output
            image_shape_list: list, [n,2] the rows,cols of n images to be masked

        Return:
            filenames of saved mask files (.dat files, with pixels labelled as class)
        """
        mask_output_dir = pathlib.Path(mask_output_dir)
        mask_names = []
        print(masks_data.keys())
        for i, key in enumerate(masks_data):
            print('mask index: {}'.format(key))
            shapes = masks_data[key]
            print('mask shapes: {}'.format(shapes))
            masks = np.zeros(image_shape_list[i])

            masks_image = np.ones((*image_shape_list[i], 3), dtype=np.uint8)  ## assume rgb data

            for s in shapes:
                c_mask = self._path_to_mask(s['path'], helper_utils.color_to_class(s['line']['color']),
                                            image_shape_list[i])

                # update mask to include new shape
                masks[c_mask > 0] = c_mask[c_mask > 0]

            mask_f_name = str(mask_output_dir / 'n-{}'.format(key))
            print(mask_f_name)
            sav_return = np.savetxt(mask_f_name, masks)
            mask_names.append(mask_f_name)
        return mask_names


@app.callback(
    Output('debug-print', 'children'),
    Input('train-seg', 'n_clicks'),
    [
        State('masks', 'data'),
        State('image-store', 'data'),
        State('dataset-selection', 'value')
    ]
)
def train_segmentation(train_seg_n_clicks, masks_data, image_store_data, dataset):
    ### don't fire if no selection is made ###
    if train_seg_n_clicks is None:
        raise PreventUpdate

    # dataset selection
    if dataset == 'tiled':
        np_volume = dataset
    else:
        np_volume = helper_utils.dcm_to_np(dataset)
    # create user directory to store users data/experiments
    data_dir_id = str(uuid.uuid4())  # create unique id for experiment
    USER_NAME = request.authorization['username']  # needs to be run in a callback or we don't have access to 'app'
    io_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER_NAME, data_dir_id))
    io_path.mkdir(parents=True, exist_ok=True)

    MASK_OUTPUT_DIR = io_path / 'masks'
    MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IM_TRAINING_DIR = io_path / 'images'
    IM_TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    #### save images who have a hand drawn mask
    #### save the shape of those saved images for converting mask from path to array
    im_shape_list = []
    image_index_with_mask = list(masks_data.keys())
    for im_index in image_index_with_mask:
        if dataset == 'tiled':
            im = TILED_CLIENT.values_indexer[int(im_index)].read()
        else:
            im = np_volume[int(im_index)]
        imageio.imsave(IM_TRAINING_DIR / '{}_for_training.tif'.format(im_index), im)
        im_shape_list.append(im.shape[0:2])

    #### Create masks from the svg shape data stored in the hidden div ####
    mask_tasks.create_save_masks(masks_data, MASK_OUTPUT_DIR, im_shape_list)

    print('returning')
    return ['']



if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0')
