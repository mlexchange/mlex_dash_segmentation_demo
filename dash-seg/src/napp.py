# LIBRARIES
import os
import io
import pathlib
import re
import json
import base64
import copy
import shutil

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
import requests
import uuid

from kwarg_editor import JSONParameterEditor
import helper_utils
from app_layout import app, MODEL_DATABASE


#### GLOBAL PARAMS ####
DEFAULT_LABEL_CLASS = 0
MASK_OUTPUT_DIR = pathlib.Path('data/masks')
IM_OUTPUT_DIR = pathlib.Path('data/images')
USER = 'mlexchange-team'
DATA_DIR = str(os.environ['DATA_DIR'])
UPLOAD_FOLDER_ROOT = "data/upload"

def slider_style(n):
    return {0: '0', n: str(n)}


@app.callback(
    Output('uploader-filename', 'data'),
    Output('dataset-selection', 'options'),
    Output('dataset-selection', 'value'),
    Output('dataset-options', 'data'),
    Input('dash-uploader', 'isCompleted'),
    Input('dataset-selection', 'value'),
    State('dash-uploader', 'fileNames'),
    State('dash-uploader', 'upload_id'),
    State('dataset-options', 'data'),
    State('dataset-selection', 'value'),
)
def image_upload(iscompleted, dataset, upload_filename, upload_id, dataset_options, dataset_value):
    if not iscompleted or dash.callback_context.triggered[0]['prop_id'] == 'dataset-selection.value':
        return [], dataset_options, dataset_value, dataset_options
            
    list_filenames = []
    supported_formats = ['tiff', 'tif', 'jpg', 'jpeg', 'png']
    if upload_filename is not None:
        path_to_zip_file = pathlib.Path(UPLOAD_FOLDER_ROOT) / upload_filename[0]
        if upload_filename[0].split('.')[-1] != 'zip':
            new_file_path = str(path_to_zip_file)
            list_filenames.append(new_file_path)
            new_label = new_file_path.split('/')[-1]
            dataset_options.append({'label': new_label, 'value': new_file_path})

        return list_filenames, dataset_options, new_file_path, dataset_options


### REACTIVE COMPONENTS FOR DISPLAY FIGURE ###
@app.callback(
    [
        Output("graph", "figure"),
        Output('image-slider', 'max'),
        Output("image-slider", "value"),
        Output("image-slider", "marks"),
        Output("image-length", "data"),
    ],
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
    Input('uploader-filename', 'data'),
    State('jobs_table', 'data')
)
def update_figure(image_slider_value, any_label_class_button_value, \
                  show_segmentation_value, image_store_data, stroke_width, row, dataset,\
                  masks_data, uploader_filename, job_data):
    # read any shapes stored in browser associated with current slice
    print(f'masks_data {masks_data}')
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
                #USER_NAME = request.authorization['username']
                USER_NAME = USER
                if bool(uploader_filename):
                    np_volume = helper_utils.dcm_to_np(uploader_filename[0])
                else:
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
                    if model_name == "random_forest":
                        semi = imageio.imread(
                            'data/mlexchange_store/{}/{}/out/{}-classified.tif'.format(USER_NAME, job_id,
                                                                                       image_slider_value))
                    elif model_name == "pyMSDtorch":
                        semi = imageio.imread(
                            'data/mlexchange_store/{}/{}/out/{}-classified.tif'.format(USER_NAME, job_id,
                                                                                       image_slider_value))
                    elif model_name == "kmeans":
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
                im.update_layout(uirevision=dataset)
                
                return [im, image_slider_max, image_slider_value, slider_style(image_slider_max), image_slider_max+1]

    print(f'uploader_filename {uploader_filename}')
    if bool(uploader_filename):
        np_volume = helper_utils.dcm_to_np(uploader_filename[0])
    else:
        # dataset selection
        np_volume = helper_utils.dcm_to_np(dataset)
    image_slider_max = len(np_volume) - 1
    if image_slider_value > image_slider_max:
        image_slider_value = 0
    im = helper_utils.make_default_figure(image_slider_value, np_volume, shapes,
                                          stroke_color=helper_utils.class_to_color(label_class_value),
                                          stroke_width=stroke_width,
                                          image_cache=im_cache)
    im.update_layout(uirevision=dataset)
    
    return [im, image_slider_max, image_slider_value, slider_style(image_slider_max), image_slider_max+1]



def msg_style(color='black'):
    return {'width':'100%', 'height': '3rem', 'color': color} 

def return_msg(job_data, row, jobType, color, message):
    msg =''
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
        msg_color, msg, is_open = return_msg(job_data, row, 'training', 'red', msg2)
        
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
            job_type = job['job_kwargs']['kwargs']['job_type'].split()
            if ' '.join(job_type[0:-1]) == 'deploy':
                param = job['job_kwargs']['kwargs']['training_model']
            else:
                param = str(job['job_kwargs']['kwargs']['parameters'])
            data_table.insert(0,
                              dict(
                                  job_id=job['uid'],
                                  job_type=job['job_kwargs']['kwargs']['job_type'],
                                  status=job['status']['state'],
                                  dataset=job['job_kwargs']['kwargs']['dataset'],
                                  image_length = job['job_kwargs']['kwargs']['image_length'],
                                  model_name=job['job_kwargs']['kwargs']['model_name'],
                                  model_uid=job['job_kwargs']['kwargs']['model_uid'],
                                  parameters=param,
                                  experiment_id=job['job_kwargs']['kwargs']['experiment_id'],
                                  job_logs=job['logs'])
                              )
    
    progress = ''
    log = " "
    if row:
        job_type = data_table[row[0]]['job_type'].split()
        log = data_table[row[0]]["job_logs"]
        if log:
            if ' '.join(job_type[0:-1]) == 'deploy':
                inside = log.split("classified\t")[-1]
                print(f'inside float {inside}')
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
    [   Output('additional-seg-params', 'children'),
        Output('brush-collapse', 'is_open'),
        Output('instructions-collapse', 'is_open'),
        Output('model-source', 'children'),
        Output('model-uid', 'data')
    ],
    Input('seg-dropdown', 'value')
)
def additional_model_features(seg_dropdown_value):
    model_source = ''
    gui_item = []
    conditions   = {'name': 'random_forest'}
    data = requests.get('http://content-api:8000/api/v0/models').json()
    model = [d for d in data if all((k in d and d[k] == v) for k, v in conditions.items())]
    model_uid = model[0]["content_id"]
    
    if seg_dropdown_value == 'Random Forest':
        conditions = {'name': 'random_forest'}
    elif seg_dropdown_value == 'pyMSDtorch':
        conditions = {'name': 'pyMSDtorch'}
    elif seg_dropdown_value == 'K-Means':
        conditions = {'name': 'kmeans'}
    
    if bool(conditions):
        model = [d for d in data if all((k in d and d[k] == v) for k, v in conditions.items())]
        model_uid = model[0]["content_id"]
        model_source = [dbc.Label('Model Source', className='mr-2'),
                        dcc.Textarea(id='msg-display',
                             value=model[0]["reference"],
                             style={'width':'100%', 'height': '3rem'},
                             className='mb-2')
                        ]
                        
        gui_item = JSONParameterEditor(_id={'type': 'parameter_editor'},  # pattern match _id (base id), name
                                   json_blob=model[0]["gui_parameters"],
                                   )
        gui_item.init_callbacks(app)
    
    is_open = True
    if model[0]["type"] == "unsupervised":
        is_open = not is_open
    
    return [gui_item, is_open, not is_open, model_source, model_uid]


def get_content(uid: str):
    url = 'http://content-api:8000/api/v0/contents/{}/content'.format(uid)  # current host, could be inside the docker
    response = requests.get(url).json()
    return response


def job_content_dict(content):
    job_content = {# 'mlex_app': content['name'],
                   'mlex_app': 'seg-demo',
                   'service_type': content['service_type'],
                   'working_directory': DATA_DIR,
                   'job_kwargs': {'uri': content['uri'], 
                                  'cmd': content['cmd'][0]}
    }
    if 'map' in content:
        job_content['job_kwargs']['map'] = content['map']
    
    return job_content


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
        State("image-length", "data"),
        State("model-uid", "data"),
        State('uploader-filename', 'data'),
    ]
)
def train_segmentation(train_seg_n_clicks, masks_data, counts, seg_dropdown_value, \
                       image_store_data, children, dataset, image_length, model_uid, upload_filename):
    """

    Args:
        train_seg_n_clicks: dash, triggers the callback when train button is pressed
        masks_data: array, annotations collected from user
        counts: int, counts of training jobs by this user
        seg_dropdown_value: str, contains the selected segmentation model
        image_store_data: dict[dict] uploaded image?
        children: ??
        dataset: str, selected dataset to use for training

    Returns:
        debug: ??
        counts: int, updated training counts

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
    if counts == 0:
        counts = helper_utils.init_counters(USER, 'training')

    if bool(upload_filename):
        dataset = upload_filename[0]

    # dataset selection
    np_volume = helper_utils.dcm_to_np(dataset)
    # create user directory to store users data/experiments
    experiment_id = str(uuid.uuid4())  # create unique id for experiment
    #USER_NAME = request.authorization['username']  # needs to be run in a callback or we don't have access to 'app'
    USER_NAME = USER
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
         if image_index_with_mask:
             for im_index in image_index_with_mask:
                 im = np_volume[int(im_index)]
                 imageio.imsave(IM_TRAINING_DIR / '{}_for_training.tif'.format(im_index), im)
         else:
             for index in range(len(np_volume)):
                 im = np_volume[index]
                 imageio.imsave(IM_TRAINING_DIR / '{}_for_training.tif'.format(index), im)

    else:
        im_shape_list = []
        if image_store_data:
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
    if children:
        for child in children['props']['children']:
            key   = child["props"]["children"][1]["props"]["id"]["param_key"]
            value = child["props"]["children"][1]["props"]["value"]
            input_params[key] = value

    model_content = get_content(model_uid)
    job_content = job_content_dict(model_content)

    job_content['job_kwargs']['kwargs'] = {}
    job_content['job_kwargs']['kwargs']['image_length'] = image_length
    job_content['job_kwargs']['kwargs']['dataset'] = dataset
    job_content['job_kwargs']['kwargs']['experiment_id'] = experiment_id
    job_content['job_kwargs']['kwargs']['parameters'] = input_params
    job_content['job_kwargs']['kwargs']['model_name'] = model_content['name']
    job_content['job_kwargs']['kwargs']['model_uid'] = model_uid
    job_content['job_kwargs']['kwargs']['job_type'] = "training " + str(counts)
    
    compute_dict = {'user_uid': USER,
                    'host_list': ['mlsandbox.als.lbl.gov', 'local.als.lbl.gov', 'vaughan.als.lbl.gov'],
                    'requirements': {'num_processors': 2,
                                     'num_gpus': 0,
                                     'num_nodes': 2},
                    }
    compute_dict['job_list'] = [job_content]
    compute_dict['dependencies'] = {'0':[]}
    compute_dict['description'] = 'Seg-demo training model: ' + job_content['mlex_app']
    compute_dict['requirements']['num_nodes'] = 1
    
    cmd_list = []
    if seg_dropdown_value == 'Random Forest':
        cmd_list = ["python random_forest.py", images_dir_docker, feature_dir_docker, mask_dir_docker, model_dir_docker]

    elif seg_dropdown_value == "pyMSDtorch":
        cmd_list = ["python src/train.py", mask_dir_docker, images_dir_docker, model_dir_docker]

    elif seg_dropdown_value == "K-Means":
        cmd_list = ["python kmeans.py", images_dir_docker, model_dir_docker]
    
    docker_cmd = " ".join(cmd_list)
    docker_cmd = docker_cmd + ' \'' + json.dumps(input_params) + '\''
    job_content['job_kwargs']['cmd'] = docker_cmd
    
    response = requests.post('http://job-service:8080/api/v0/workflows', json=compute_dict)
    
    counts += 1
    print('returning')
    return ['', counts]


@app.callback(
    [
        Output('classified-image-store', 'data'),
        Output('seg-counter', 'data'),
    ],
        Input('compute-seg', 'n_clicks'),
    [
        State('image-store', 'data'),
        State('seg-counter', 'data'),
        State('jobs_table', 'selected_rows'),
        State('jobs_table', 'data'),
        State('dataset-selection', 'value'),
        State("image-length", "data"),
        State('uploader-filename', 'data'),
    ],

    prevent_initial_call=True
)
def compute_seg_react(compute_seg_n_clicks, image_store_data, counts, row, job_data, \
                      dataset, image_length, upload_filename):
    """

    Args:
        compute_seg_n_clicks: dash type, if clicked triggers this func
        image_store_data: dict[dict] ???
        counts: int, counts of deploy jobs for this user
        row: array, contains the selected row in the list of jobs
        job_data: array, lists of jobs
        dataset: str, contains the selected dataset

    Returns:
        classified_image: array, segmentation mask
        counts: int, updated counts
    """

    if compute_seg_n_clicks is None or not bool(row):
        raise PreventUpdate
    
    job_type = job_data[row[0]]["job_type"].split()
    if ' '.join(job_type[0:-1]) != 'training':
         raise PreventUpdate

    # initializes the counter according to the latest deploy job in the database
    if counts == 0:
        counts = helper_utils.init_counters(USER, 'deploy')

    if upload_filename:
        dataset = upload_filename[0]
    np_volume = helper_utils.dcm_to_np(dataset)
    # find most recent job id (current experiment)
    training_experiment_id = job_data[row[0]]["experiment_id"]
    experiment_id = str(uuid.uuid4())    # create unique experiment id
    model_name = job_data[row[0]]["model_name"]
    model_description = job_data[row[0]]["job_type"]
    #USER_NAME = request.authorization['username']  # needs to be run in a callback or we don't have access to 'app'
    USER_NAME = USER
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
    
    model_uid = job_data[row[0]]["model_uid"]
    model_content = get_content(model_uid)
    job_content = job_content_dict(model_content)
    job_content['job_kwargs']['kwargs'] = {}
    job_content['job_kwargs']['kwargs']['image_length'] = image_length
    job_content['job_kwargs']['kwargs']['dataset'] = dataset
    job_content['job_kwargs']['kwargs']['experiment_id'] = experiment_id
    job_content['job_kwargs']['kwargs']['parameters'] = meta_params
    job_content['job_kwargs']['kwargs']['model_name'] = model_content['name']
    job_content['job_kwargs']['kwargs']['model_uid'] = model_uid
    job_content['job_kwargs']['kwargs']['training_model'] = model_description
    job_content['job_kwargs']['kwargs']['job_type'] = "deploy " + str(counts)
    
    compute_dict = {'user_uid': USER,
                    'host_list': ['mlsandbox.als.lbl.gov', 'local.als.lbl.gov', 'vaughan.als.lbl.gov'],
                    'requirements': {'num_processors': 2,
                                     'num_gpus': 0,
                                     'num_nodes': 2},
                    }
    compute_dict['job_list'] = [job_content]
    compute_dict['dependencies'] = {'0':[]}
    compute_dict['description'] = 'Seg-demo segmenting with model: ' + job_content['mlex_app']
    compute_dict['requirements']['num_nodes'] = 1
    
    cmd_list = []
    if model_name == 'random_forest':
        model_input_dir_dock = MODEL_INPUT_DIR / 'random-forest.model'
        cmd_list = ["python segment.py", im_input_dir_dock, str(model_input_dir_dock), out_dir_dock]

    elif model_name == "pyMSDtorch":
        model_input_dir_dock = MODEL_INPUT_DIR / 'state_dict_net.pt'
        cmd_list = ["python src/segment.py", im_input_dir_dock, str(model_input_dir_dock), out_dir_dock]

    elif model_name == "kmeans":
        model_input_dir_dock = MODEL_INPUT_DIR / 'kmeans.joblib'
        cmd_list = ["python segment.py", im_input_dir_dock, str(model_input_dir_dock), out_dir_dock]
    
    docker_cmd = " ".join(cmd_list)
    docker_cmd = docker_cmd + ' \'' + json.dumps(meta_params) + '\''
    job_content['job_kwargs']['cmd'] = docker_cmd
    
    response = requests.post('http://job-service:8080/api/v0/workflows', json=compute_dict)
    
    counts += 1
    print('sending images to server to be segmented')
    return ['', counts]


@app.callback(
    Output("download-zip", 'data'),
    Input("download-data", "n_clicks"),
    State('jobs_table', 'selected_rows'),
    State('jobs_table', 'data'),
    prevent_initial_call=True
)
def download(n_clicks, row, job_data):
    if row:
        data = job_data[row[0]]
    
    experiment_id = job_data[row[0]]["experiment_id"]
    data_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER, experiment_id))
    shutil.make_archive(data_path, 'zip', data_path)
    zip_file_name = str(data_path)+".zip"
    return dcc.send_file(zip_file_name)


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8052)
