# LIBRARIES
import plotly.express as px
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from flask import request
import json
import pathlib
import uuid
from skimage import draw
import skimage
import io
import base64
import PIL.Image
from time import time
import imageio
import numpy as np
from mlex_api import job_dispatcher
import re
import pymongo
##### HELPER UTILS
import helper_utils
##### LAYOUT COMPONENTS
from app_layout import header, segmentation, sidebar_label, random_forest_params, \
    msd_params, training_results, meta, app, np_volume, MODEL_DATABASE, IMAGES_SHAPE

#### GLOBAL PARAMS ####
DEFAULT_LABEL_CLASS = 0
MASK_OUTPUT_DIR = pathlib.Path('data/masks')
IM_OUTPUT_DIR = pathlib.Path('data/images')


### Initialize the MongoDB collection
myclient = pymongo.MongoClient(host="mongodb://mongodb:27017/")
mydb = myclient["segmentation_app"]   # creates the database
MONGOdb = mydb["job_status"]       # creates the collection

### REACTIVE COMPONENTS FOR UPLOADING FIGURE ###
@app.callback(
        [Output('image-store', 'data'),
            Output('image-slider', 'max'),
            ],

        Input('upload-image', 'contents'),
        Input('upload-image', 'filename'),
        State('image-store', 'data'),
        )
def image_upload(upload_image_contents, upload_image_filename,image_store_data):
    if upload_image_contents is None:
        raise PreventUpdate
    print('uploading data...')
    if upload_image_contents is not None:
        for c, n in zip(upload_image_contents, upload_image_filename):
            content_type, content_string = c.split(',')
            image_store_data[n] = (content_type, content_string)
            print('storing: {} \n {}'.format(n,c))
        image_slider_max = len(upload_image_filename)-1
    return [image_store_data, image_slider_max]
        
### REACTIVE COMPONENTS FOR DISPLAY FIGURE ###
@app.callback(
        [
            Output("graph", "figure"),
            Output('slider-output-container', 'children'),
        ],
        [
            Input("image-slider", "value"),
            Input({'type': "label-class-button", "index":dash.dependencies.ALL},
                "n_clicks_timestamp",
                ),
            Input('show-segmentation', 'value'),
            Input("seg-dropdown", "value"),
            Input('image-store', 'data'),
        ],
        [State("masks", "data"),State('classified-image-store', 'data'), 
            State('experiment-store', 'data'),
            ]
        )
def update_figure(image_slider_value, any_label_class_button_value,show_segmentation_value,seg_dropdown_value,image_store_data, masks_data, classified_image_store_data, experiment_store_data):
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
    if len(image_store_data) >0:
        im_cache = image_store_data[list(image_store_data.keys())[image_slider_value]][1]
        print(image_store_data[list(image_store_data.keys())[image_slider_value]][0])
    else:
        im_cache = None
    im = helper_utils.make_default_figure(image_slider_value, np_volume, shapes,
                                          stroke_color=helper_utils.class_to_color(label_class_value),
                                          image_cache=im_cache)
    if ("Show segmentation" in show_segmentation_value):
        print('showing seg')
        # get most recent experiment job id
        job_id = sorted(experiment_store_data, key = lambda k: (experiment_store_data[k]['timestamp']))[0]
        # read in image (too large to store all images in browser cache)
        USER_NAME = request.authorization['username'] # needs to be run in a callback or we don't have access to 'app'
        try:
            if seg_dropdown_value== "Random Forest":
                semi = imageio.imread('data/mlexchange_store/{}/{}/out/{}-classified.tif'.format(USER_NAME, job_id, image_slider_value))
            elif seg_dropdown_value == "MSD":
                semi = imageio.mimread('data/mlexchange_store/{}/{}/out/results.tif'.format(USER_NAME, job_id))[image_slider_value]
        except:
            print('slice not yet segmented')
        semi = helper_utils.label_to_colors(semi)
        print(semi)
        def img_array_to_pil_image(ia):
            ia = skimage.util.img_as_ubyte(ia)
            img = PIL.Image.fromarray(ia)
            return img
        semipng = img_array_to_pil_image(semi)
        semipng.save('data/printcolor.png')
        print(semipng)
        width, height = (semi.shape[0],semi.shape[1])
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
    return [im,image_slider_value]


@app.callback(
    [ 
            Output('masks', 'data'),
        ],
        
        [
            Input("graph", "relayoutData"),
        ],
        [State("image-slider", "value"),
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
            #print(masks_data[image_slice][0].keys())
        # change shape path if shape has been updated
        elif any(["shapes" in key for key in graph_relayoutData]):
            #print(masks_data[str(image_slice)])
            for key in graph_relayoutData:
                if 'shapes' in key:
                    #print('key: {}'.format(key))
                    # get index of edited shape
                    shape_index_re = re.compile('shapes\[(\d*)\].path') # fragile, probably break on dash update
                    shape_index = int(shape_index_re.match(key).group(1))
                    #print(shapes[shape_index])
                    #print('modified...')
                    #print(graph_relayoutData[key])
                    masks_data[str(image_slice)][shape_index]['path'] = graph_relayoutData[key]
                    
            #print(graph_relayoutData[key])
        else:
            return dash.no_update
    return [masks_data]

### REACTIVE FOR MSD PARAMETERS ###

@app.callback(
            Output('learning-rate-slider-output', 'children'),
            
        [
            Input('learning-rate-slider', 'value'),
            ]
        )
def update_learning_rate_slider(learning_rate_slider_value):
    return "Learning Rate: {}".format(learning_rate_slider_value)

### REACTIVE FOR SEGMENTATION PARAMETERS ###
@app.callback(
        [
            Output('additional-seg-params', 'children'),
        ],
        [
            Input('seg-dropdown', 'value'),
        ]
        )
def additional_seg_features(seg_dropdown_value):
    if seg_dropdown_value == 'Random Forest':
        return [random_forest_params]
    elif seg_dropdown_value == 'MSD':
        return [msd_params]
    else:
        return ['']

### REACTIVE FOR SEGMENTATION ###
#### HELPER COLLECTION OF FUNCTIONS PREPARING SEGMENTATION ####
class mask_tasks():
    """
    example use:
    Given a dictionary of masks (key is index of image masked, value is list of svg path objects),
    call:
        f_names = mask_tasks.create_masks(mask_dict), which saves numpy .dat files of the masks and returns
        the file names to you
    """

    @staticmethod #doesn't need cls as first arg, use staticmethod
    def _path_to_indices(path): #_ indicates this is a private method
        """From SVG path to numpy array of coordinates, each row being a (row, col) point
        """
        indices_str = [
	    el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
	]
        return np.rint(np.array(indices_str, dtype=float)).astype(int)


    @classmethod
    def _path_to_mask(self, path,class_n, shape):
        """From SVG path to a boolean array where all pixels enclosed by the path
        are True, and the other pixels are False.

        Args:
            path, str: svg string
            class_n, int: the labeled class of the path
            shape, (int, int): the shape of the labeled image
        """
        class_n = int(class_n)
        cols, rows = self._path_to_indices(path).T # a series of vertices, need to translate into points
        # draw.line returns rr, cc
        lines = [draw.line_aa(r0,c0,r1,c1) for r0,c0,r1,c1 in zip(rows[:-1], cols[:-1], rows[1:], cols[1:])]

        #rr, cc = draw.polygon_perimeter(rows, cols) # change to draw.polygon to have filled holes
        mask = np.zeros(shape, dtype=int)
        # the plus one is because zero is taken as an unlabled pixel, so
        # all classes need to be shifted up by one for the labelling.
        for line in lines:
            mask[line[0], line[1]] = class_n+1
        print(class_n)
        # don't want to have filled paths, just draw mask where the stroke is
        # so commented out below
#        mask = ndimage.binary_fill_holes(mask)
        return mask
    
    @classmethod
    def create_save_masks(self, masks_data: dict, mask_output_dir=MASK_OUTPUT_DIR, image_shape_list=[IMAGES_SHAPE for i in range(100)]):
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
        print (masks_data.keys())
        for i, key in enumerate(masks_data):
            print('mask index: {}'.format(key))
            shapes = masks_data[key]
            print('mask shapes: {}'.format(shapes))
            masks = np.zeros(image_shape_list[i])

            masks_image = np.ones( (*image_shape_list[i], 3), dtype=np.uint8) ## assume rgb data

            for s in shapes:
                c_mask = self._path_to_mask(s['path'], helper_utils.color_to_class(s['line']['color']), image_shape_list[i])

                # update mask to include new shape
                masks[c_mask > 0] = c_mask[c_mask > 0]

            mask_f_name = str(mask_output_dir / 'n-{}'.format(key))
            print(mask_f_name)
            sav_return = np.savetxt(mask_f_name, masks)
            mask_names.append(mask_f_name)
        return mask_names

    
@app.callback(
        [Output('training-results', 'figure'),
            Output('training-visible', 'hidden'),
            Output('model-train-alert', 'children'),
            Output('model-train-alert', 'color'),
            Output('model-seg-alert', 'children'),
            Output('model-seg-alert', 'color'),
            ],
        Input('update-training-loss', 'n_intervals'),
        State('experiment-store', 'data')
        )
def listen_for_results(n, experiment_store_data):
    '''
    TANNY COMPLETE
    monitor job status in Mongo database, updating
    loss function and job status appropriately
    '''
    # current_jobs = MONGOdb.find_one()
    # TANNY: ADD TABLE THAT LISTS THESE JOBS & THE USER CAN PICK ONE
    # LET'S SAY THE USER SELECTS selected_job
    # THE CURRENT JOB IS HARD-CODED
    current_job = MONGOdb.find_one( {}, sort=[('_id', pymongo.DESCENDING)])
    if current_job is not None:
        if current_job["status"] == 'completed':
            job_type = current_job["job_type"]
            if job_type == 'training':
                training_status_display ='Status: Trained'
                training_status_color = 'green'
                segmentation_status_display= dash.no_update
                segmentation_status_color= dash.no_update
            elif job_type == 'deploy':
                segmentation_status_display = 'Status: Segmented'
                segmentation_status_color= 'green'
                training_status_display = dash.no_update
                training_status_color = dash.no_update
            try:
                USER_NAME = request.authorization['username'] # needs to be run in a callback or we don't have access to 'app'

                loss_plot_path = pathlib.Path('data/mlexchange_store/')/USER_NAME/current_job_id / 'models/msd-losses.png'
                print(loss_plot_path)
                loss_plot = imageio.imread(loss_plot_path)
                loss_plot_fig= px.imshow(loss_plot)
                width,height = loss_plot.shape[0:2]
                loss_plot_fig.update_xaxes(
                showgrid=False, showticklabels=False, zeroline=False
                    )
                loss_plot_fig.update_yaxes(
                    showgrid=False,
                    scaleanchor="x",
                    showticklabels=False,
                    zeroline=False,
                )
                return [loss_plot_fig, False, training_status_display, training_status_color, segmentation_status_display, segmentation_status_color]
            except Exception as e:
                print(e)
                loss_plot_fig = px.scatter([[0,0]])
            return [loss_plot_fig,True,training_status_display, training_status_color, segmentation_status_display, segmentation_status_color]
            pass
    return [dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update]

 
@app.callback(
        [
            Output('debug-print','children'),
            Output('experiment-store', 'data'),
            Output('model-alert', 'children'),
        ],
        [
            Input('train-seg','n_clicks')
        ],
        [
            State('masks', 'data'),
            State('seg-dropdown', 'value'),
            State('experiment-store', 'data'),
            State('image-store', 'data')
            ]
        )
def train_segmentation(train_seg_n_clicks, masks_data, seg_dropdown_value, experiment_store_data, image_store_data):
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
    job_id = str(uuid.uuid4()) # create unique id for experiment
    USER_NAME = request.authorization['username'] # needs to be run in a callback or we don't have access to 'app'
    io_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER_NAME, job_id))
    io_path.mkdir(parents=True, exist_ok=True)

    MASK_OUTPUT_DIR = io_path / 'masks'
    MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok =True)
    IM_TRAINING_DIR = io_path / 'images'
    IM_TRAINING_DIR.mkdir(parents=True, exist_ok =True)
    MODEL_DIR = io_path / 'models'
    MODEL_DIR.mkdir(parents=True, exist_ok =True)
    FEATURE_DIR = io_path / 'features'
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    #### save images who have a hand drawn mask
    #### save the shape of those saved images for converting mask from path to array
    im_shape_list = []
    image_index_with_mask = list(masks_data.keys())
    im_list = []
    if len(image_store_data) > 0:
        for im_index in image_index_with_mask:
            im_str = image_store_data[list(image_store_data.keys())[int(im_index)]][1]
            im_decode = base64.b64decode(im_str)
            im_bytes = io.BytesIO(im_decode)
            im = PIL.Image.open(im_bytes).convert('L') #convert to grayscale

            im_shape = (im.height, im.width)
            imageio.imsave(IM_TRAINING_DIR / '{}_for_training.tif'.format(im_index), im)
            im_shape_list.append(im_shape)
    else: # no uploaded data, so use default
        for im_index in image_index_with_mask:
            im = np_volume[int(im_index)]
            imageio.imsave(IM_TRAINING_DIR / '{}_for_training.tif'.format(im_index), im)
            im_shape_list.append(im.shape[0:2])

    #### Create masks from the svg shape data stored in the hidden div ####
    debug = ''
    mask_file_names = mask_tasks.create_save_masks(masks_data, MASK_OUTPUT_DIR, im_shape_list)

    # save tiff series of images (only those that have at least one mask associated with them)
    # call ml_api to dispatch job to workers (blocking for now, future will probably have a dispatcher server to handle setting up the job queue, etc)
    if seg_dropdown_value == 'Random Forest':
        mask_dir_docker = '/' / MASK_OUTPUT_DIR
        images_dir_docker = '/' / IM_TRAINING_DIR
        model_dir_docker = '/' / MODEL_DIR
        feature_dir_docker = '/' / FEATURE_DIR
        feat_job = job_dispatcher.simpleJob('supervised segmentation, feature generation',
                                            job_type='pre-processing',
                                            deploy_location='local-vaughan',
                                            docker_uri=MODEL_DATABASE[seg_dropdown_value],
                                            docker_cmd='python feature_generation.py',
                                            kw_args='{} {}'.format(images_dir_docker,
                                                                   feature_dir_docker),
                                            db_collection=MONGOdb,
                                            GPU=False
                                            )
        feat_job.launchJob()
        print(feat_job.job_id)
        print('launched feature extraction on ml server')
        seg_job = job_dispatcher.simpleJob('supervised segmentation, random forest training',
                                           job_type='training',
                                           deploy_location='local-vaughan',
                                           docker_uri=MODEL_DATABASE[seg_dropdown_value],
                                           docker_cmd='python random_forest.py',
                                           kw_args='{} {} {}'.format(mask_dir_docker, feature_dir_docker,
                                                                     model_dir_docker),
                                           db_collection=MONGOdb,
                                           GPU=False
                                           )
        print(seg_job.job_id)
        seg_job.launchJob()
        print('launched segmentation on ml server')
        experiment_record = {'timestamp': time(),
                             'trained_bool': False,
                             'segmented_bool': False,
                             'epochs_run': 200,
                             'model': seg_dropdown_value,
                             'final_loss': .001,
                             'batch_size': 2,
                             }
        experiment_store_data[job_id] = experiment_record
        print(experiment_store_data)

    elif seg_dropdown_value == "MSD":
        # preface with / as docker executable needs a path specific to its filesyste 
        mask_dir_docker = '/' / MASK_OUTPUT_DIR
        images_dir_docker = '/' / IM_TRAINING_DIR
        model_dir_docker = '/' / MODEL_DIR
        seg_job = job_dispatcher.simpleJob('supervised segmentation, msd training',
                job_type = 'training',
                deploy_location = 'local-vaughan',
                docker_uri = MODEL_DATABASE[seg_dropdown_value],
                docker_cmd = 'python Deploy.py',
                kw_args = '{} {} {}'.format(mask_dir_docker, images_dir_docker, model_dir_docker),
                db_collection=MONGOdb,
                GPU = True
                )
        seg_job.launchJob()
        print('launched segmentation on ml server')

        experiment_record = {'timestamp': time(),
                            'trained_bool':False,
                            'segmented_bool':False,
                            'epochs_run': 200,
                            'model': seg_dropdown_value,
                            'final_loss': .001,
                            'batch_size': 2,
                            } 
        experiment_store_data[job_id] = experiment_record
        print(experiment_store_data)
            
    print('returning')        
    return ['', experiment_store_data, dbc.Alert(id='model-train-alert', children='Status: Training', color='red')]
        
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
        Output('seg-alert', 'children'),
        ],
        
        [
        Input('compute-seg', 'n_clicks'),
        ],
        [
            State('seg-dropdown', 'value'),
            State('experiment-store', 'data'),
            State('image-store', 'data')
            ]
        )
def compute_seg_react(compute_seg_n_clicks, seg_dropdown_value, experiment_store_data, image_store_data):
    '''
    compute_seg_nclicks: dash type, if clicked triggers this func
    seg_dropdown_value: Str, contains the str name of model to run
    experiment_store_data: dict[dict], key: job_id, value: dict{job attributes}
    '''

    if compute_seg_n_clicks is None:
        raise PreventUpdate
    # create user directory to store users data/experiments

    # find most recent job id (current experiment)
    print(experiment_store_data)
    job_id = sorted(experiment_store_data, key = lambda k: (experiment_store_data[k]['timestamp']))[0]
    USER_NAME = request.authorization['username'] # needs to be run in a callback or we don't have access to 'app'
    io_path = pathlib.Path('data/mlexchange_store/{}/{}'.format(USER_NAME, job_id))
    io_path.mkdir(parents=True, exist_ok=True)

    IM_INPUT_DIR = io_path / 'images' / 'raw'
    IM_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    IM_INPUT_FILE = IM_INPUT_DIR/ 'segment_series.tif'
    im_input_dir_dock = '/' / IM_INPUT_FILE
    MODEL_INPUT_DIR = io_path / 'models' 
    model_input_dir_dock = '/' / MODEL_INPUT_DIR/'state_dict_net.pt'
    OUT_DIR = io_path / 'out'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir_dock = '/' / OUT_DIR # when mounted in docker container where segmentation code lives, data will be mounted in root, so we need to have the right path
    
    print('exporting images')
    if len(image_store_data) > 0:
        im_list = []
        for image_filename in image_store_data:
            image_str = image_store_data[image_filename][1]
            image = np.array(PIL.Image.open(io.BytesIO(base64.b64decode(image_str))).convert('L'))
            im_list.append(image)
        im_vol = np.r_[im_list]
        imageio.volwrite(IM_INPUT_FILE, im_vol) 
            #imageio.imsave(IM_INPUT_DIR / '{}_for_segmenting.tif', image)
    else:
        imageio.volwrite(IM_INPUT_FILE, np_volume) 

    print('computing segmentation...')
    if seg_dropdown_value == "Random Forest":
        model_input_dir_dock = '/' / MODEL_INPUT_DIR / 'random-forest.model'
        docker_cmd = "python segment.py"
        kw_args = '{} {} {}'.format(im_input_dir_dock, model_input_dir_dock, out_dir_dock)
        # kw_args = '/' + SAMPLE_DATA+ " /data/models/random-forest.model /data/output"
        GPU = False
    elif (seg_dropdown_value == "MSD"):
        model_input_dir_dock = '/' / MODEL_INPUT_DIR / 'state_dict_net.pt'
        docker_cmd = "python Segment.py"
        kw_args = '{} {} {}'.format(im_input_dir_dock, model_input_dir_dock, out_dir_dock)
        GPU = False
    deploy_job = job_dispatcher.simpleJob('supervised segmentation, random forest deploy',
                                          job_type="deploy",
                                          deploy_location='local-vaughan',
                                          docker_uri=MODEL_DATABASE[seg_dropdown_value],
                                          docker_cmd=docker_cmd,
                                          kw_args=kw_args,
                                          db_collection=MONGOdb,
                                          GPU=GPU
                                          )
    print(deploy_job.kw_args)
    deploy_job.launchJob()
    print('sending images to server to be segmented')

    # now read in deploy results and save them to the
    # classified image store
    data = ''
        
    return [data, dbc.Alert(id='model-seg-alert', children='Status: Segmenting', color='red') ]

    # need to compute every image slice. I think we'll just
    # make something that generalizes to the MSD, where we have
    # a single worker node-- there is no point scaling up becaue we only have one or two gpus. 
    # in the created docker file, we will use job lib to create parallel shit.
                                   

##### DEFINE LAYOUT ####
app.layout = html.Div (
        [
            header,
            dbc.Container(
                [
                    dbc.Row(
                        [dbc.Col(segmentation, width=8), dbc.Col(sidebar_label, width=4)]
                    ),
                    dbc.Row(dbc.Col(html.P(id='debug-print', children=''))),
                    dbc.Row(dbc.Col(training_results)),
                    dbc.Row(dbc.Col(meta)),
                ]
            ),

        ]
        )
if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0')
