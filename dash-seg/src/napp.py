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
from skimage import draw
import skimage
import urllib.request
import uuid

import job_dispatcher
from targeted_callbacks import targeted_callback
from kwarg_editor import JSONParameterEditor
import helper_utils
from app_layout import app, MODEL_DATABASE


#### GLOBAL PARAMS ####
DEFAULT_LABEL_CLASS = 0
MASK_OUTPUT_DIR = pathlib.Path('data/masks')
IM_OUTPUT_DIR = pathlib.Path('data/images')
USER = 'mlexchange-team'
DATA_DIR = str(os.environ['DATA_DIR'])


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
        Input('show-segmentation', 'value'),
        Input('image-store', 'data'),
        Input('stroke-width', 'value'),
        Input('jobs_table', 'selected_rows'),
        Input('dataset-selection', 'value')
    ],
    [
        State('masks', 'data'),
        State('jobs_table', 'data')
     ]
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
        print(image_store_data[list(image_store_data.keys())[image_slider_value]][0])
    else:
        im_cache = None
    if "Show segmentation" in show_segmentation_value:
        # get selected job id from job list
        if row is not None:
            if job_data[row[0]]["job_type"] == 'deploy':
                model_name  = job_data[row[0]]["model_name"]
                data_dir_id = job_data[row[0]]["data_dir_id"]
                job_id      = data_dir_id

                # needs to be run in a callback or we don't have access to 'app'
                USER_NAME = request.authorization['username']
                # dataset selection - based on selected job
                np_volume = helper_utils.dcm_to_np('data/mlexchange_store/{}/{}/images/raw/segment_series.tif'.format(USER_NAME, job_id))
                image_slider_max = len(np_volume) - 1
                if image_slider_value > image_slider_max:
                    image_slider_value = 0
                im = helper_utils.make_default_figure(image_slider_value, np_volume, shapes,
                                                      stroke_color=helper_utils.class_to_color(label_class_value),
                                                      stroke_width=stroke_width,
                                                      image_cache=im_cache)

                # read in image (too large to store all images in browser cache)
                try:
                    if model_name == "Random Forest":
                        semi = imageio.imread(
                            'data/mlexchange_store/{}/{}/out/{}-classified.tif'.format(USER_NAME, job_id,
                                                                                       image_slider_value))
                    elif model_name == "pyMSDtorch":
                        semi = imageio.imread(
                            'data/mlexchange_store/{}/{}/out/{}-classified.tif'.format(USER_NAME, job_id,
                                                                                       image_slider_value))
                    elif model_name == "K-Means":
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
               
                return [im, image_slider_max, image_slider_value, slider_style(image_slider_max)]

    # dataset selection
    print('Loading dataset')
    np_volume = helper_utils.dcm_to_np(dataset)
    print('Dataset loaded')
    image_slider_max = len(np_volume) - 1
    if image_slider_value > image_slider_max:
        image_slider_value = 0
    im = helper_utils.make_default_figure(image_slider_value, np_volume, shapes,
                                          stroke_color=helper_utils.class_to_color(label_class_value),
                                          stroke_width=stroke_width,
                                              image_cache=im_cache)
    
    return [im, image_slider_max, image_slider_value, slider_style(image_slider_max)]



def msg_style(color='black'):
    return {'width':'100%', 'height': '3rem', 'color': color} 

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
        Input("close-error", "n_clicks")
    ],
    [   
        State('jobs_table', 'data')
    ]
)
def show_message(dataset, show_segmentation_value, row, n_clicks, job_data):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    msg = ''
    msg_color = msg_style()
    is_open = False

    if bool(show_segmentation_value):
        if row is None:
            msg_color = msg_style('red')
            is_open = True
            msg = "Please select deploy (segment) from the List of Jobs!"
        else:
            if job_data[row[0]]["job_type"] != 'deploy':
                is_open = True
                msg_color = msg_style('red')
                msg = "Please select deploy (segment) from the List of Jobs!"

    if 'close-error' in changed_id:
        is_open = False

    return [msg, is_open, msg_color]


@app.callback(
    Output('model-source', 'children'),
    Input('seg-dropdown', 'value'),
)
def update_model_source(seg_dropdown_value):
    model_source = ''
    conditions   = {}
    data         = model_list_GET_call()
    
    if seg_dropdown_value == 'Random Forest':
        conditions = {'model_name': 'random_forest'}
    elif seg_dropdown_value == 'pyMSDtorch':
        conditions = {'model_name': 'pyMSDtorch'}
    elif seg_dropdown_value == 'K-Means':
        conditions = {'model_name': 'kmeans'}
    
    if bool(conditions):
        model = [d for d in data if all((k in d and d[k] == v) for k, v in conditions.items())]
        model_source = [dbc.Label('Model Source', className='mr-2'),
                        dcc.Textarea(id='msg-display',
                             value=model[0]["reference"],
                             style={'width':'100%', 'height': '3rem'},
                             className='mb-2')
                        ]

    return model_source


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
    ],
    Input('update-training-loss', 'n_intervals'),
    State('jobs_table', 'selected_rows'),
)
def update_table(n, row):
    job_list = helper_utils.get_job(USER, 'seg-demo')
    data_table = []
    if job_list is not None:
        for job in job_list:
            data_table.insert(0,
                              dict(
                                  job_id        = job['uid'],
                                  job_type      = job['job_type'],
                                  status        = job['status'],
                                  dataset       = job['container_kwargs']['dataset'],
                                  model_name    = job['container_kwargs']['model_name'],
                                  parameters    = str(job['container_kwargs']['parameters']),
                                  data_dir_id   = job['container_kwargs']['data_dir_id'],
                                  job_logs      = job['container_logs'])
                              )
    log = " "
    if row:
        text = 'Type: ' + data_table[row[0]]["job_type"] + '\nStatus: ' + data_table[row[0]]["status"] + \
               '\nDataset: ' + data_table[row[0]]["dataset"] + '\nModel: ' + data_table[row[0]]["model_name"] + \
               '\nParameters: ' + data_table[row[0]]["parameters"]
        log = data_table[row[0]]["job_logs"]
        if log:
            start = log.find('loss')
            if start > -1:
                fig = helper_utils.generate_figure(log,start)
                return data_table, True, fig, log
    return data_table, False, go.Figure(go.Scatter(x=[], y=[])), log


@app.callback(
    Output('debug-print', 'children'),
    Input('train-seg', 'n_clicks'),
    [
        State('masks', 'data'),
        State('seg-dropdown', 'value'),
        State('image-store', 'data'),
        State('additional-seg-params', 'children'),
        State('dataset-selection', 'value')
    ]
)
def train_segmentation(train_seg_n_clicks, masks_data, seg_dropdown_value, image_store_data, children, dataset):
    """
    Args:

    Output:
    debug_print_children: string to print logs
    experiment_store_data: dict storing experimental records
    model_trained_alert_children: update alert showing model is undergoing training

    Prepares data for segmentation job.
    1. Convert svg path to nparray of size nxnx3 (rgb image). 0 is unlabelled, user labelled class is
    increased by 1 (ie. user label class 0, this would be represented in the nparray as 1) This is because
    the random forest that this is based on expects that 0 is an unlabelled pixel, so the user labels have to
    be adjusted to take this into account.

    2. Save mask and image files into data directory that the ml_worker node has access to

    3. Create a list of job messages (uri::segmentation_docker_exec(train/deploy), uri::filename_image, uri::mask)

    4. Send job messages into job dispatcher
    """
    # code from https://dash.plotly.com/annotations

    ### don't fire if no selection is made ###
    if (train_seg_n_clicks is None) or (seg_dropdown_value is None):
        raise PreventUpdate

    # dataset selection
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
    MODEL_DIR = io_path / 'models'
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR = io_path / 'features'
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    #### save images who have a hand drawn mask
    #### save the shape of those saved images for converting mask from path to array
    image_index_with_mask = list(masks_data.keys())
    if seg_dropdown_value == 'K-Means':
        if bool(image_index_with_mask):
            for im_index in image_index_with_mask:
                im = np_volume[int(im_index)]
                imageio.imsave(IM_TRAINING_DIR / '{}_for_training.tif'.format(im_index), im)
        else:
            for index in range(len(np_volume)):
                im = np_volume[index]
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
            key   = child["props"]["children"][1]["props"]["id"]["param_key"]
            value = child["props"]["children"][1]["props"]["value"]
            input_params[key] = value

    if seg_dropdown_value == 'Random Forest':
        print('now doing random forest...')
        kw_args = {'model_name':  seg_dropdown_value,
                   'directories': [images_dir_docker, feature_dir_docker],
                   'parameters':  {},
                   'data_dir_id': data_dir_id,
                   'dataset': dataset
                   }
        feat_job = job_dispatcher.SimpleJob(user=USER,
                                            job_type="feature generation",
                                            description="Random Forest",
                                            deploy_location="local",
                                            gpu=False,
                                            data_uri=DATA_DIR,
                                            container_uri=MODEL_DATABASE[seg_dropdown_value],
                                            container_cmd='python feature_generation.py',
                                            container_kwargs=kw_args,
                                            )

        feat_job.launch_job()
        print('launched feature extraction on ml server')

        docker_cmd = "python random_forest.py"
        kw_args = {'model_name':  seg_dropdown_value,
                   'directories': [mask_dir_docker, feature_dir_docker, model_dir_docker],
                   'parameters': input_params,
                   'data_dir_id': data_dir_id,
                   'dataset': dataset
                   }

    elif seg_dropdown_value == "pyMSDtorch":
        docker_cmd = "python src/train.py"
        kw_args = {'model_name':  seg_dropdown_value,
                   'directories': [mask_dir_docker, images_dir_docker, model_dir_docker],
                   'parameters':  input_params,
                   'data_dir_id': data_dir_id,
                   'dataset': dataset
                   }

    elif seg_dropdown_value == "K-Means":
        docker_cmd = "python kmeans.py"
        kw_args = {'model_name':  seg_dropdown_value,
                   'directories': [images_dir_docker, model_dir_docker],
                   'parameters':  input_params,
                   'data_dir_id': data_dir_id,
                   'dataset': dataset
                   }

    train_job = job_dispatcher.SimpleJob(user=USER,
                                         job_type="training",
                                         description="",
                                         deploy_location="local",
                                         gpu=False,
                                         data_uri=str(DATA_DIR),
                                         container_uri=MODEL_DATABASE[seg_dropdown_value],
                                         container_cmd=docker_cmd,
                                         container_kwargs=kw_args,
                                         )
    train_job.launch_job()

    print('returning')
    return ['']


@app.callback(
    Output('classified-image-store', 'data'),
    Input('compute-seg', 'n_clicks'),
    [
        State('seg-dropdown', 'value'),
        State('image-store', 'data'),
        State('jobs_table', 'selected_rows'),
        State('jobs_table', 'data'),
        State('dataset-selection', 'value')
    ],

    prevent_initial_call=True
)
def compute_seg_react(compute_seg_n_clicks, seg_dropdown_value, image_store_data, row, job_data, dataset):
    '''
    compute_seg_nclicks: dash type, if clicked triggers this func
    seg_dropdown_value: Str, contains the str name of model to run
    experiment_store_data: dict[dict], key: job_id, value: dict{job attributes}
    '''

    if compute_seg_n_clicks is None:
        raise PreventUpdate
    # create user directory to store users data/experiments

    # dataset selection
    np_volume = helper_utils.dcm_to_np(dataset)
    # find most recent job id (current experiment)
    data_dir_id = job_data[row[0]]["data_dir_id"]
    USER_NAME = request.authorization['username']  # needs to be run in a callback or we don't have access to 'app'
    io_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER_NAME, data_dir_id))
    ab_pwd = pathlib.Path().resolve()
    io_path.mkdir(parents=True, exist_ok=True)

    IM_INPUT_DIR = io_path / 'images' / 'raw'
    IM_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    IM_INPUT_FILE = IM_INPUT_DIR / 'segment_series.tif'
    im_input_dir_dock = IM_INPUT_FILE
    MODEL_INPUT_DIR = io_path / 'models'
    model_input_dir_dock = MODEL_INPUT_DIR / 'state_dict_net.pt'
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
    if seg_dropdown_value == "Random Forest":
        model_input_dir_dock = MODEL_INPUT_DIR / 'random-forest.model'
        docker_cmd = "python segment.py"
        kw_args = {'model_name':  seg_dropdown_value,
                   'directories': [im_input_dir_dock, str(model_input_dir_dock), out_dir_dock],
                   'parameters': meta_params,
                   'data_dir_id': data_dir_id,
                   'dataset': dataset
                   }

    elif seg_dropdown_value == "pyMSDtorch":
        model_input_dir_dock = MODEL_INPUT_DIR / 'state_dict_net.pt'
        docker_cmd = "python src/segment.py"
        kw_args = {'model_name':  seg_dropdown_value,
                   'directories': [im_input_dir_dock, str(model_input_dir_dock), out_dir_dock],
                   'parameters': meta_params,
                   'data_dir_id': data_dir_id,
                   'dataset': dataset
                   }

    elif seg_dropdown_value == "K-Means":
        model_input_dir_dock = MODEL_INPUT_DIR / 'kmeans.joblib'
        docker_cmd = "python segment.py"
        kw_args = {'model_name':  seg_dropdown_value,
                   'directories': [im_input_dir_dock, str(model_input_dir_dock), out_dir_dock],
                   'parameters': meta_params,
                   'data_dir_id': data_dir_id,
                   'dataset': dataset
                   }

    seg_job = job_dispatcher.SimpleJob(user=USER,
                                       job_type="deploy",
                                       description="",
                                       deploy_location="local",
                                       gpu=False,
                                       data_uri=str(DATA_DIR),
                                       container_uri=MODEL_DATABASE[seg_dropdown_value],
                                       container_cmd=docker_cmd,
                                       container_kwargs=kw_args,
                                       )
    print(seg_job.data_uri)
    seg_job.launch_job()
    print('sending images to server to be segmented')
    return ['']


@app.callback(
    Output('additional-seg-params', 'children'),
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
    
    model = [d for d in data if all((k in d and d[k] == v) for k, v in conditions.items())]
    gui_item = JSONParameterEditor(_id={'type': 'parameter_editor'},  # pattern match _id (base id), name
                                   json_blob=model[0]["gui_parameters"],
                                   )
    gui_item.init_callbacks(app)
    return gui_item


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0')
