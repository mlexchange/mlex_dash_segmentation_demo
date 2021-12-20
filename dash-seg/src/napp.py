# LIBRARIES
import os
import io
import pathlib
import re
import json
import base64

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
from kwarg_editor import JSONParameterEditor
import helper_utils
from app_layout import app, MODEL_DATABASE


#### GLOBAL PARAMS ####
DEFAULT_LABEL_CLASS = 0
MASK_OUTPUT_DIR = pathlib.Path('data/masks')
IM_OUTPUT_DIR = pathlib.Path('data/images')
USER = 'mlexchange-team'
DATA_DIR = str(os.environ['DATA_DIR'])
TILED_CLIENT = from_uri("http://host.docker.internal:8057", cache=Cache.on_disk('data/cache'))


def slider_style(n):
    return {0: '0', n: str(n)}

## REACTIVE COMPONENTS FOR DISPLAY FIGURE ###
@app.callback(
    [
        Output("graph", "figure"),
        Output('image-slider', 'max'),
        Output("image-slider", "value"),
        Output("image-slider", "marks"),
        Output("image-length", "data"),
    ],
    [
        Input("image-slider", "value"),
        Input({'type': "label-class-button", "index": dash.dependencies.ALL},
              "n_clicks_timestamp",
              ),
        Input('show-segmentation', 'value'),
        Input('image-store', 'data'),
        Input('stroke-width', 'value'),
        Input('jobs_table', 'selected_rows'),
        Input('dataset-selection', 'value'),
        Input('masks', 'data'),
    ],
    State('jobs_table', 'data')
)
def update_figure(image_slider_value, any_label_class_button_value, show_segmentation_value, image_store_data,
                  stroke_width, row, dataset, masks_data, job_data):
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
    else:
        im_cache = None
    if "Show segmentation" in show_segmentation_value:
        # get selected job id from job list
        if row is not None:
            job_type = job_data[row[0]]["job_type"].split()
            if ' '.join(job_type[0:-1]) == 'deploy':
                model_name  = job_data[row[0]]["model_name"]
                experiment_id = job_data[row[0]]["experiment_id"]
                job_id      = experiment_id

                # needs to be run in a callback or we don't have access to 'app'
                USER_NAME = request.authorization['username']
                # dataset selection - based on selected job
                np_volume = helper_utils.dcm_to_np('data/mlexchange_store/{}/{}/images/segment_series.tif'.format(USER_NAME, job_id))
                image_slider_max = len(np_volume) - 1
                if image_slider_value > image_slider_max:
                    image_slider_value = 0
                im = helper_utils.make_default_figure(image_slider_value, np_volume, shapes,
                                                      stroke_color=helper_utils.class_to_color(label_class_value),
                                                      stroke_width=stroke_width,
                                                      image_cache=im_cache)

                # read in image (too large to store all images in browser cache)
                try:
                    semi = imageio.imread(
                            'data/mlexchange_store/{}/{}/out/{}-classified.tif'.format(USER_NAME, job_id,
                                                                                       image_slider_value))
                except Exception as err:
                    print(err)
                semi = helper_utils.label_to_colors(semi)

                def img_array_to_pil_image(ia):
                    ia = skimage.util.img_as_ubyte(ia)
                    img = PIL.Image.fromarray(ia)
                    return img

                semipng = img_array_to_pil_image(semi)
                semipng.save('data/printcolor.png')

                width, height = (semi.shape[0], semi.shape[1])
                im.add_layout_image(
                    dict(
                        source=semipng,
                        xref="x",
                        yref="y",
                        x=0,
                        y=0,
                        sizex=width,
                        sizey=height,
                        sizing="contain",
                        opacity=0.5,
                        layer="above",
                    )
                )
                im.update_layout(template='plotly_white')

                return [im, image_slider_max, image_slider_value, slider_style(image_slider_max), image_slider_max+1]

    # dataset selection
    if dataset == 'tiled':
        image_slider_max = len(TILED_CLIENT) - 1
        if image_slider_value > image_slider_max:
            image_slider_value = 0
        im = helper_utils.make_default_figure(image_slider_value, TILED_CLIENT, shapes,
                                              stroke_color=helper_utils.class_to_color(label_class_value),
                                              stroke_width=stroke_width, image_cache=im_cache)
    else:
        np_volume = helper_utils.dcm_to_np(dataset)
        image_slider_max = len(np_volume) - 1
        if image_slider_value > image_slider_max:
            image_slider_value = 0
        im = helper_utils.make_default_figure(image_slider_value, np_volume, shapes,
                                              stroke_color=helper_utils.class_to_color(label_class_value),
                                              stroke_width=stroke_width,
                                              image_cache=im_cache)

    return [im, image_slider_max, image_slider_value, slider_style(image_slider_max), image_slider_max+1]


def msg_style(color='black'):
    return {'width': '100%', 'height': '3rem', 'color': color}


def return_msg(job_data, row, jobType, color, message, dataset=None):
    msg = ''
    msg_color = msg_style()
    is_open = False

    if row is None:
        msg_color = msg_style(color)
        is_open = True
        msg = message
    else:
        job_type = job_data[row[0]]["job_type"].split()
        if ' '.join(job_type[0:-1]) != jobType:
            is_open = True
            msg_color = msg_style(color)
            msg = message
        elif dataset == 'tiled':
                msg_color = msg_style(color)
                is_open = True
                msg = 'The data transfer has been initiated. ' \
                      'The job will appear in the table once the transfer has been completed. ' \
                      'This action may take a couple minutes.'

    return msg_color, msg, is_open


@app.callback(
    [
        Output('error-body', 'children'),
        Output('error-msg', 'is_open'),
        Output('error-msg', 'style'),
    ],
    [
        Input('dataset-selection', 'value'),
        Input('show-segmentation', 'value'),
        Input('jobs_table', 'selected_rows'),
        Input("close-error", "n_clicks"),
        Input("compute-seg", "n_clicks")
    ],
    [
        State('jobs_table', 'data')
    ]
)
def show_message(dataset, show_segmentation_value, row, n_clicks1, n_clicks2, job_data):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    msg =''
    msg_color = msg_style()
    is_open = False
    msg1 = "Please select a deploy (segment) from the List of Jobs!"
    msg2 = "Please select a training from the List of Jobs!"
    msg3 = "Please unselect show segmentation (button) before switching to a different dataset!"

    if bool(show_segmentation_value):
        msg_color, msg, is_open = return_msg(job_data, row, 'deploy', 'red', msg1)

    if 'compute-seg' in changed_id:
        msg_color, msg, is_open = return_msg(job_data, row, 'training', 'red', msg2, dataset)

    if 'dataset-selection' in changed_id:
        if row is not None and bool(show_segmentation_value):
            job_type = job_data[row[0]]["job_type"].split()
            if ' '.join(job_type[0:-1]) == 'deploy':
                is_open = True
                msg_color = msg_style('red')
                msg = msg3 

    if 'close-error' in changed_id:
        is_open = False

    return [msg, is_open, msg_color]


@app.callback(
    Output('model-source', 'children'),
    Input('seg-dropdown', 'value'),
)
def update_model_source(seg_dropdown_value):
    model_source = ''
    conditions = {}
    data = model_list_GET_call()

    if seg_dropdown_value == 'Random Forest':
        conditions = {'model_name': 'random_forest'}
    elif seg_dropdown_value == 'pyMSDtorch':
        conditions = {'model_name': 'pyMSDtorch'}
    elif seg_dropdown_value == 'K-Means':
        conditions = {'model_name': 'kmeans'}
    elif seg_dropdown_value == 'TUNet':
        conditions = {'model_name': 'TUNet'}

    if bool(conditions):
        model = [d for d in data if all((k in d and d[k] == v) for k, v in conditions.items())]
        model_source = [dbc.Label('Model Source', className='mr-2'),
                        dcc.Textarea(id='msg-display',
                                     value=model[0]["reference"],
                                     style={'width': '100%', 'height': '3rem'},
                                     className='mb-2')
                        ]

    return model_source


@app.callback(
    Output('masks', 'data'),
    Output('graph', 'relayoutData'),
    Input('del-mask', 'n_clicks'),
    Input('graph', 'relayoutData'),
    Input('masks', 'data'),
    State('image-slider', 'value')
)
def store_masks(n, graph_relayoutData, masks_data, image_slice):
    """
    Save shapes from figure to json Store
    """
    if graph_relayoutData is not None:
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'del-mask' in changed_id:
            return {}, {}
        else:
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
                return dash.no_update, dash.no_update
    return masks_data, dash.no_update


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
        mask = np.zeros(shape, dtype=int) -1
        # output label starts from 0
        for line in lines:
            mask[line[0], line[1]] = class_n
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
        for i, key in enumerate(masks_data):
            shapes = masks_data[key]
            masks = np.zeros(image_shape_list[i])-1

            masks_image = np.ones((*image_shape_list[i], 3), dtype=np.uint8)  ## assume rgb data

            for s in shapes:
                c_mask = self._path_to_mask(s['path'], helper_utils.color_to_class(s['line']['color']),
                                            image_shape_list[i])

                # update mask to include new shape
                masks[c_mask > -1] = c_mask[c_mask > -1]

            mask_f_name = str(mask_output_dir / 'n-{}'.format(key))
            sav_return = np.savetxt(mask_f_name, masks)
            mask_names.append(mask_f_name)
        return mask_names


def model_list_GET_call():
    """
    Get the whole model registry data from the fastapi url.
    """
    url = 'http://model-api:8000/api/v0/model-list'  # current host, could be inside the docker
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    return data


### REACTIVE COMPONENTS FOR TABLE, LOGS, AND PLOT GENERATION ###
@app.callback(
    [
        Output('jobs_table', 'data'),
        Output('show-plot', 'is_open'),
        Output('loss-plot', 'figure'),
        Output('job-logs', 'value'),
        Output('progress-bar', 'children'),
    ],
    Input('update-training-loss', 'n_intervals'),
    Input('jobs_table', 'selected_rows')
)
def update_table(n, row):
    """

    Args:
        n: updates table every interval of time
        row: array, selected row from table, used to display it's logs

    Returns:
        jobs_table: list[dict], contains the details of each submitted job
        show_plot: bool, shows the loss plot if the training model contains one
        loss_plot: plotly, contains the loss plot
        job_logs: str, contains the logs of the selected job
    """
    job_list = helper_utils.get_job(USER, 'seg-demo')
    data_table = []
    if job_list is not None:
        for job in job_list:
            job_type = job['job_type'].split()
            if ' '.join(job_type[0:-1]) == 'deploy':
                param = job['container_kwargs']['training_model']
            else:
                param = str(job['container_kwargs']['parameters'])
            data_table.insert(0,
                              dict(
                                  job_id=job['uid'],
                                  job_type=job['job_type'],
                                  status=job['status'],
                                  dataset=job['container_kwargs']['dataset'],
                                  image_length = job['container_kwargs']['image_length'],
                                  model_name=job['container_kwargs']['model_name'],
                                  parameters=param,
                                  experiment_id=job['container_kwargs']['experiment_id'],
                                  job_logs=job['container_logs'])
                              )

    progress = ''
    log = " "
    if row:
        job_type = data_table[row[0]]['job_type'].split()
        log = data_table[row[0]]["job_logs"]
        if log:
            if ' '.join(job_type[0:-1]) == 'deploy':
                values = (int(float(log.split("classified\t")[-1]))+1)/data_table[row[0]]["image_length"]*100
                labels = 'Deploy progress: ' + str(round(values)) + '%'
                if values < 100 or data_table[row[0]]['status'] == 'running':
                    progress = [dbc.Label(labels), dbc.Progress(value=values)]
                
            start = log.find('loss')
            if start > -1 and len(log) > start-5:
                fig = helper_utils.generate_figure(log,start)
                return data_table, True, fig, log, progress
    return data_table, False, go.Figure(go.Scatter(x=[], y=[])), log, progress


@app.callback(
    [
        Output('debug-print', 'children'),
        Output('train_counter', 'data'),
    ],
    Input('train-seg', 'n_clicks'),
    [
        State('masks', 'data'),
        State('train_counter', 'data'),
        State('seg-dropdown', 'value'),
        State('image-store', 'data'),
        State('additional-seg-params', 'children'),
        State('dataset-selection', 'value'),
        State("image-length", "data")
    ]
)
def train_segmentation(train_seg_n_clicks, masks_data, count, seg_dropdown_value, image_store_data, children, dataset, image_length):
    """

    Args:
        train_seg_n_clicks: dash, triggers the callback when train button is pressed
        masks_data: array, annotations collected from user
        count: int, count of training jobs by this user
        seg_dropdown_value: str, contains the selected segmentation model
        image_store_data: dict[dict] uploaded image?
        children: ??
        dataset: str, selected dataset to use for training

    Returns:
        debug: ??
        count: int, updated training count

    Prepares data for segmentation job.
        1. Convert svg path to nparray of size nxnx3 (rgb image). 0 is unlabelled, user labelled class is
           increased by 1 (ie. user label class 0, this would be represented in the nparray as 1) This is because
           the random forest that this is based on expects that 0 is an unlabelled pixel, so the user labels have to
           be adjusted to take this into account.
        2. Save mask and image files into data directory that the ml_worker node has access to

    References: https://dash.plotly.com/annotations
    """

    ### don't fire if no selection is made ###
    if (train_seg_n_clicks is None) or (seg_dropdown_value is None):
        raise PreventUpdate

    # initializes the counter according to the latest training job in the database
    if count == 0:
        count = helper_utils.init_counters(USER, 'training')

    # dataset selection
    if dataset == 'tiled':
        np_volume = dataset
    else:
        np_volume = helper_utils.dcm_to_np(dataset)
    # create user directory to store users data/experiments
    experiment_id = str(uuid.uuid4())  # create unique id for experiment
    USER_NAME = request.authorization['username']  # needs to be run in a callback or we don't have access to 'app'
    io_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER_NAME, experiment_id))
    io_path.mkdir(parents=True, exist_ok=True)

    MASK_OUTPUT_DIR = io_path / 'masks'
    MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IM_TRAINING_DIR = io_path / 'images'
    IM_TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR = io_path / 'models'
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR = io_path / 'features'
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    #### save images who have a hand drawn mask
    #### save the shape of those saved images for converting mask from path to array
    masks_data = {k:v for (k,v) in masks_data.items() if bool(v)}
    image_index_with_mask = list(masks_data.keys()) #slice index that is labeled
    if seg_dropdown_value == 'K-Means':
         if bool(image_index_with_mask):
             for im_index in image_index_with_mask:
                 if dataset == 'tiled':
                     im = TILED_CLIENT.values_indexer[int(im_index)].read()
                 else:
                     im = np_volume[int(im_index)]
                 imageio.imsave(IM_TRAINING_DIR / '{}_for_training.tif'.format(im_index), im)
         else:
             for index in range(len(np_volume)):
                 if dataset == 'tiled':
                     im = TILED_CLIENT.values_indexer[int(index)].read()
                 else:
                     im = np_volume[int(index)]
                 imageio.imsave(IM_TRAINING_DIR / '{}_for_training.tif'.format(index), im)

    else:
        im_shape_list = []
        if len(image_store_data) > 0:
            for im_index in image_index_with_mask:
                im_str = image_store_data[list(image_store_data.keys())[int(im_index)]][1]
                im_decode = base64.b64decode(im_str)
                im_bytes = io.BytesIO(im_decode)
                im = PIL.Image.open(im_bytes).convert('L')  # convert to grayscale

                im_shape = (im.height, im.width)
                imageio.imsave(IM_TRAINING_DIR / '{}_for_training.tif'.format(im_index), im)
                im_shape_list.append(im_shape)

        else:  # no uploaded data, so use default
            for im_index in image_index_with_mask:
                if dataset == 'tiled':
                    im = TILED_CLIENT.values_indexer[int(im_index)].read()
                else:
                    im = np_volume[int(im_index)]
                imageio.imsave(IM_TRAINING_DIR / '{}_for_training.tif'.format(im_index), im)
                im_shape_list.append(im.shape[0:2])

        #### Create masks from the svg shape data stored in the hidden div ####
        mask_file_names = mask_tasks.create_save_masks(masks_data, MASK_OUTPUT_DIR, im_shape_list)

    mask_dir_docker = str(MASK_OUTPUT_DIR)
    images_dir_docker = str(IM_TRAINING_DIR)
    model_dir_docker = str(MODEL_DIR)
    feature_dir_docker = str(FEATURE_DIR)

    input_params = {}
    if bool(children):
        for child in children['props']['children']:
            key = child["props"]["children"][1]["props"]["id"]["param_key"]
            value = child["props"]["children"][1]["props"]["value"]
            input_params[key] = value

    if seg_dropdown_value == 'Random Forest':
        print('now doing random forest...')

        docker_cmd = "python random_forest.py"
        kw_args = {'model_name':  seg_dropdown_value,
                   'directories': [images_dir_docker, feature_dir_docker, mask_dir_docker, model_dir_docker],
                   'parameters': input_params,
                   'experiment_id': experiment_id,
                   'dataset': dataset,
                   'image_length': image_length
                   }

    elif seg_dropdown_value == "pyMSDtorch":
        docker_cmd = "python src/train.py"
        kw_args = {'model_name': seg_dropdown_value,
                   'directories': [mask_dir_docker, images_dir_docker, model_dir_docker],
                   'parameters':  input_params,
                   'experiment_id': experiment_id,
                   'dataset': dataset,
                   'image_length': image_length
                   }

    elif seg_dropdown_value == "K-Means":
        docker_cmd = "python kmeans.py"
        kw_args = {'model_name':  seg_dropdown_value,
                   'directories': [images_dir_docker, model_dir_docker],
                   'parameters':  input_params,
                   'experiment_id': experiment_id,
                   'dataset': dataset,
                   'image_length': image_length
                   }

    elif seg_dropdown_value == "TUNet":
        docker_cmd = "python src/train.py"
        kw_args = {'model_name': seg_dropdown_value,
                   'directories': [mask_dir_docker, images_dir_docker, model_dir_docker],
                   'parameters':  input_params,
                   'experiment_id': experiment_id,
                   'dataset': dataset,
                   'image_length': image_length
                   }

    train_job = job_dispatcher.SimpleJob(user=USER,
                                         job_type="training " + str(count),
                                         description= " ",
                                         deploy_location="local",
                                         gpu=False,
                                         data_uri=str(DATA_DIR),
                                         container_uri=MODEL_DATABASE[seg_dropdown_value],
                                         container_cmd=docker_cmd,
                                         container_kwargs=kw_args,
                                         )
    train_job.launch_job()
    count += 1
    print('returning')
    return ['', count]


@app.callback(
    [
        Output('classified-image-store', 'data'),
        Output('seg_counter', 'data'),
    ],
        Input('compute-seg', 'n_clicks'),
    [
        State('image-store', 'data'),
        State('seg_counter', 'data'),
        State('jobs_table', 'selected_rows'),
        State('jobs_table', 'data'),
        State('dataset-selection', 'value'),
        State("image-length", "data")
    ],

    prevent_initial_call=True
)
def compute_seg_react(compute_seg_n_clicks, image_store_data, count, row, job_data, dataset, image_length):
    """

    Args:
        compute_seg_n_clicks: dash type, if clicked triggers this func
        image_store_data: dict[dict] ???
        count: int, count of deploy jobs for this user
        row: array, contains the selected row in the list of jobs
        job_data: array, lists of jobs
        dataset: str, contains the selected dataset

    Returns:
        classified_image: array, segmentation mask
        count: int, updated count
    """

    if compute_seg_n_clicks is None or not bool(row):
        raise PreventUpdate
    
    job_type = job_data[row[0]]["job_type"].split()
    if ' '.join(job_type[0:-1]) != 'training':
         raise PreventUpdate

    # initializes the counter according to the latest deploy job in the database
    if count == 0:
        count = helper_utils.init_counters(USER, 'deploy')

    # create user directory to store users data/experiments
    # dataset selection
    if dataset == 'tiled':
        np_volume = []
        for n in range(len(TILED_CLIENT)):
            np_volume.append(TILED_CLIENT.values_indexer[n].read())
        np_volume = np.array(np_volume)
    else:
        np_volume = helper_utils.dcm_to_np(dataset)
    # find most recent job id (current experiment)
    training_experiment_id = job_data[row[0]]["experiment_id"]
    experiment_id = str(uuid.uuid4())    # create unique experiment id
    model_name = job_data[row[0]]["model_name"]
    model_description = job_data[row[0]]["job_type"]
    USER_NAME = request.authorization['username']  # needs to be run in a callback or we don't have access to 'app'
    io_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER_NAME, experiment_id))
    train_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER_NAME, training_experiment_id))
    ab_pwd = pathlib.Path().resolve()
    io_path.mkdir(parents=True, exist_ok=True)

    IM_INPUT_DIR = io_path / 'images'
    IM_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    IM_INPUT_FILE = IM_INPUT_DIR / 'segment_series.tif'
    im_input_dir_dock = IM_INPUT_FILE
    MODEL_INPUT_DIR = train_path / 'models'
    OUT_DIR = io_path / 'out'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir_dock = OUT_DIR  # when mounted in docker container where segmentation code lives, data will be mounted in root, so we need to have the right path

    print('exporting images')
    if len(image_store_data) > 0:
        im_list = []
        for image_filename in image_store_data:
            image_str = image_store_data[image_filename][1]
            image = np.array(PIL.Image.open(io.BytesIO(base64.b64decode(image_str))).convert('L'))
            im_list.append(image)
        im_vol = np.r_[im_list]
        imageio.volwrite(IM_INPUT_FILE, im_vol)
    else:
        imageio.volwrite(IM_INPUT_FILE, np_volume)

    im_input_dir_dock = str(im_input_dir_dock)
    out_dir_dock = str(out_dir_dock)
    print('computing segmentation...')
    meta_params = {"show_progress": 1}  # not be able for empty
    if model_name == "Random Forest":
        model_input_dir_dock = MODEL_INPUT_DIR / 'random-forest.model'
        docker_cmd = "python segment.py"
        kw_args = {'model_name':  model_name,
                   'directories': [im_input_dir_dock, str(model_input_dir_dock), out_dir_dock],
                   'parameters': meta_params,
                   'experiment_id': experiment_id,
                   'dataset': dataset,
                   'image_length': image_length,
                   'training_model': model_description
                   }

    elif model_name == "pyMSDtorch":
        model_input_dir_dock = MODEL_INPUT_DIR / 'state_dict_net.pt'
        docker_cmd = "python src/segment.py"
        kw_args = {'model_name':  model_name,
                   'directories': [im_input_dir_dock, str(model_input_dir_dock), out_dir_dock],
                   'parameters': meta_params,
                   'experiment_id': experiment_id,
                   'dataset': dataset,
                   'image_length': image_length,
                   'training_model': model_description
                   }

    elif model_name == "K-Means":
        model_input_dir_dock = MODEL_INPUT_DIR / 'kmeans.joblib'
        docker_cmd = "python segment.py"
        kw_args = {'model_name':  model_name,
                   'directories': [im_input_dir_dock, str(model_input_dir_dock), out_dir_dock],
                   'parameters': meta_params,
                   'experiment_id': experiment_id,
                   'dataset': dataset,
                   'image_length': image_length,
                   'training_model': model_description
                   }

    elif model_name == "TUNet":
        model_input_dir_dock = MODEL_INPUT_DIR / 'state_dict_net.pt'
        docker_cmd = "python src/segment.py"
        kw_args = {'model_name':  model_name,
                   'directories': [im_input_dir_dock, str(model_input_dir_dock), out_dir_dock],
                   'parameters': meta_params,
                   'experiment_id': experiment_id,
                   'dataset': dataset,
                   'image_length': image_length,
                   'training_model': model_description
                   }

    seg_job = job_dispatcher.SimpleJob(
                    user=USER,
                    job_type="deploy " + str(count),
                    description= " ",
                    deploy_location="local",
                    gpu=False,
                    data_uri=str(DATA_DIR),
                    container_uri=MODEL_DATABASE[model_name],
                    container_cmd=docker_cmd,
                    container_kwargs=kw_args,
                    )

    seg_job.launch_job()
    count += 1
    print('sending images to server to be segmented')
    return ['', count]


@app.callback(
    [   Output('additional-seg-params', 'children'),
        Output('brush-collapse', 'is_open'),
        Output('instructions-collapse', 'is_open')
    ],
    Input('seg-dropdown', 'value')
)
def additional_seg_features(seg_dropdown_value):
    data = model_list_GET_call()
    if seg_dropdown_value == 'Random Forest':
        conditions = {'model_name': 'random_forest'}
    elif seg_dropdown_value == 'pyMSDtorch':
        conditions = {'model_name': 'pyMSDtorch'}
    elif seg_dropdown_value == 'K-Means':
        conditions = {'model_name': 'kmeans'}
    elif seg_dropdown_value == 'TUNet':
        conditions = {'model_name': 'TUNet'}

    model = [d for d in data if all((k in d and d[k] == v) for k, v in conditions.items())]
    gui_item = JSONParameterEditor(_id={'type': 'parameter_editor'},  # pattern match _id (base id), name
                                   json_blob=model[0]["gui_parameters"],
                                   )
    gui_item.init_callbacks(app)
    is_open = True
    if model[0]["type"] == "unsupervised":
        is_open = not is_open
    return [gui_item, is_open, not is_open]


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0')
