# LIBRARIES
import base64
import io
import json
import os
import pathlib
import re

import dash
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
from app_layout import header, segmentation, sidebar_label, training_results, meta, app, np_volume, MODEL_DATABASE, IMAGES_SHAPE

#### GLOBAL PARAMS ####
DEFAULT_LABEL_CLASS = 0
MASK_OUTPUT_DIR = pathlib.Path('data/masks')
IM_OUTPUT_DIR = pathlib.Path('data/images')
USER = 'mlexchange-team'
DATA_DIR = str(os.environ['DATA_DIR'])


### REACTIVE COMPONENTS FOR UPLOADING FIGURE ###
@app.callback(
    [
        Output('image-store', 'data'),
        Output('image-slider', 'max'),

    ],

    Input('upload-image', 'contents'),
    Input('upload-image', 'filename'),
    State('image-store', 'data'),
)
def image_upload(upload_image_contents, upload_image_filename, image_store_data):
    if upload_image_contents is None:
        raise PreventUpdate
    print('uploading data...')
    if upload_image_contents is not None:
        for c, n in zip(upload_image_contents, upload_image_filename):
            content_type, content_string = c.split(',')
            image_store_data[n] = (content_type, content_string)
            print('storing: {}'.format(n))
        image_slider_max = len(upload_image_filename) - 1
    return [image_store_data, image_slider_max]


### REACTIVE COMPONENTS FOR DISPLAY FIGURE ###
@app.callback(
    [
        Output("graph", "figure"),
        Output('slider-output-container', 'children'),
        Output('msg-display', 'value')
    ],
    [
        Input("image-slider", "value"),
        Input({'type': "label-class-button", "index": dash.dependencies.ALL},
              "n_clicks_timestamp",
              ),
        Input('show-segmentation', 'value'),
        Input("seg-dropdown", "value"),
        Input('image-store', 'data'),
        Input('stroke-width', 'value'),
        Input('jobs_table', 'selected_rows'),
    ],
    [
        State("masks", "data"),
        State('classified-image-store', 'data'),
        State('experiment-store', 'data'),
        State('jobs_table', 'data')
     ],
)
def update_figure(image_slider_value, any_label_class_button_value, show_segmentation_value, seg_dropdown_value,
                  image_store_data, stroke_width, row, masks_data, classified_image_store_data, experiment_store_data, job_data):
    msg = ''
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
    # plot the new figure given:
    # 1. a change in image slice (from slider)
    # 2. a "show masks" button toggled
    if len(image_store_data) > 0:
        im_cache = image_store_data[list(image_store_data.keys())[image_slider_value]][1]
        print(image_store_data[list(image_store_data.keys())[image_slider_value]][0])
    else:
        im_cache = None
    im = helper_utils.make_default_figure(image_slider_value, np_volume, shapes,
                                          stroke_color=helper_utils.class_to_color(label_class_value),
                                          stroke_width=stroke_width,
                                          image_cache=im_cache)
    if ("Show segmentation" in show_segmentation_value):
        # get most recent experiment job id
        if row is not None:
            model_name  = job_data[row[0]]["model_name"]
            data_dir_id = job_data[row[0]]["data_dir_id"]
            job_id      = data_dir_id
            # read in image (too large to store all images in browser cache)
            USER_NAME = request.authorization['username']  # needs to be run in a callback or we don't have access to 'app'
            try:
                if model_name == "Random Forest":
                    semi = imageio.imread(
                        'data/mlexchange_store/{}/{}/out/{}-classified.tif'.format(USER_NAME, job_id, image_slider_value))
                elif model_name == "pyMSDtorch":
                    semi = imageio.mimread('data/mlexchange_store/{}/{}/out/results.tif'.format(USER_NAME, job_id))[
                        image_slider_value]
                elif model_name == "K-Means":
                    semi = imageio.imread(
                        'data/mlexchange_store/{}/{}/out/{}-classified.tif'.format(USER_NAME, job_id, image_slider_value))

            except:
                print('slice not yet segmented')
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
        else:
            msg = 'Please select a deploy (segment) result from List of Jobs!'
    return [im, image_slider_value, msg]


@app.callback(
    [
        Output('masks', 'data'),
    ],

    [
        Input("graph", "relayoutData"),
    ],
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
    return [masks_data]


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
    def create_save_masks(self, masks_data: dict, mask_output_dir=MASK_OUTPUT_DIR,
                          image_shape_list=[IMAGES_SHAPE for i in range(100)]):
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


### REACTIVE COMPONENTS FOR TABLE, LOGS, AND PLOT GENERATION ###
@app.callback(
    [
        Output('jobs_table', 'data'),
        Output('show-plot', 'is_open'),
        Output('loss-plot', 'figure'),
        Output('job-logs', 'value')
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
                                  job_id        =job['uid'],
                                  job_type      =job['job_type'],
                                  status        =job['status'],
                                  model_name    = job['container_kwargs']['model_name'],
                                  parameters    = str(job['container_kwargs']['parameters']),
                                  data_dir_id   = job['container_kwargs']['data_dir_id'],
                                  job_logs        = job['container_logs'])
                              )
    log = " "
    if row:
        log = data_table[row[0]]["job_logs"]
        if log:
            start = log.find('loss')
            if start > -1:
                fig = helper_utils.generate_figure(log,start)
                return data_table, True, fig, log
    return data_table, False, go.Figure(go.Scatter(x=[], y=[])), log


@app.callback(
    [
        Output('debug-print', 'children'),
        Output('experiment-store', 'data'),
    ],
    [
        Input('train-seg', 'n_clicks')
    ],
    [
        State('masks', 'data'),
        State('seg-dropdown', 'value'),
        State('experiment-store', 'data'),
        State('image-store', 'data'),
        State('additional-seg-params', 'children')
    ]
)
def train_segmentation(train_seg_n_clicks, masks_data, seg_dropdown_value, experiment_store_data, image_store_data, children):
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
    if seg_dropdown_value == 'K-Means':
        params = {'n_clusters': kmeans_n_clusters}
        pth = '/app/work/' / MODEL_DIR
        with open(pth / 'params.json', 'w') as f:
            json.dump(params, f)

        # Copy images
        for index in range(len(np_volume)):
            im = np_volume[index]
            imageio.imsave(IM_TRAINING_DIR / '{}_for_training.tif'.format(index), im)

    else:
        im_shape_list = []
        image_index_with_mask = list(masks_data.keys())
        im_list = []
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
                   'directories': [str(images_dir_docker), str(feature_dir_docker)],
                   'parameters':  {},
                   'data_dir_id': data_dir_id
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
                   'data_dir_id': data_dir_id
                   }

    elif seg_dropdown_value == "pyMSDtorch":
        docker_cmd = "python src/train.py"
        input_params = {"num_epochs": 200, "optimizer": "Adam", "criterion": "CrossEntropyLoss", "learning_rate": 0.01, "num_layers": 10, "max_dilation": 10}
        kw_args = {'model_name':  seg_dropdown_value,
                   'directories': [mask_dir_docker, images_dir_docker, model_dir_docker],
                   'parameters':  input_params,
                   'data_dir_id': data_dir_id
                   }

    elif seg_dropdown_value == "K-Means":
        docker_cmd = "python kmeans.py"
        input_params = {}
        kw_args = {'model_name':  seg_dropdown_value,
                   'directories': [images_dir_docker, model_dir_docker],
                   'parameters':  input_params,
                   'data_dir_id': data_dir_id
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

    experiment_store_data = []
    print('returning')
    return ['', experiment_store_data]


@app.callback(
    Output('none', 'data'),
    Input('experiment-store', 'data'),
)
def test_trigger(experiment_store_data):
    print(experiment_store_data)
    return ''


@app.callback(
    [
        Output('classified-image-store', 'data'),
    ],

    [
        Input('compute-seg', 'n_clicks'),
    ],
    [
        State('seg-dropdown', 'value'),
        State('experiment-store', 'data'),
        State('image-store', 'data'),
        State('jobs_table', 'selected_rows'),
        State('jobs_table', 'data')
    ],

    prevent_initial_call=True
)
def compute_seg_react(compute_seg_n_clicks, seg_dropdown_value, experiment_store_data, image_store_data, row, job_data):
    '''
    compute_seg_nclicks: dash type, if clicked triggers this func
    seg_dropdown_value: Str, contains the str name of model to run
    experiment_store_data: dict[dict], key: job_id, value: dict{job attributes}
    '''

    if compute_seg_n_clicks is None:
        raise PreventUpdate
    # create user directory to store users data/experiments

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
    if seg_dropdown_value == "Random Forest":
        meta_params = {"show_progress": 1}  # not be able for empty
        model_input_dir_dock = MODEL_INPUT_DIR / 'random-forest.model'
        docker_cmd = "python segment.py"
        kw_args = {'model_name':  seg_dropdown_value,
                   'directories': [im_input_dir_dock, str(model_input_dir_dock), out_dir_dock],
                   'parameters': meta_params,
                   'data_dir_id': data_dir_id
                   }

    elif (seg_dropdown_value == "pyMSDtorch"):
        model_input_dir_dock = MODEL_INPUT_DIR / 'state_dict_net.pt'
        docker_cmd = "python src/segment.py"
        meta_params= {"show_progress": 1}
        kw_args = {'model_name':  seg_dropdown_value,
                   'directories': [im_input_dir_dock, str(model_input_dir_dock), out_dir_dock],
                   'parameters': meta_params,
                   'data_dir_id': data_dir_id
                   }

    elif (seg_dropdown_value == "K-Means"):
        model_input_dir_dock = MODEL_INPUT_DIR / 'kmeans.joblib'
        docker_cmd = "python segment.py"
        kw_args = {'model_name':  seg_dropdown_value,
                   'directories': [im_input_dir_dock, str(model_input_dir_dock), out_dir_dock],
                   'parameters': meta_params,
                   'data_dir_id': data_dir_id
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

    # now read in deploy results and save them to the
    # classified image store

    return ['']

    # need to compute every image slice. I think we'll just
    # make something that generalizes to the pyMSDtorch, where we have
    # a single worker node-- there is no point scaling up becaue we only have one or two gpus.
    # in the created docker file, we will use job lib to create parallel shit.


def model_list_GET_call():
    """
    Get the whole model registry data from the fastapi url.
    """
    url = 'http://service-api:8000/api/v0/model-list'  # current host, could be inside the docker
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    return data


@app.callback(
    [
        Output('additional-seg-params', 'children'),
    ],
    [
        Input('seg-dropdown', 'value'),
    ]
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
    return [gui_item]


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0')
