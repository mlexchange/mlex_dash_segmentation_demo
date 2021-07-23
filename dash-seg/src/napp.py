import plotly.express as px
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_auth
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from flask import request
import plot_common
import json
import pathlib
import os
import uuid
#from shapes_to_segmentations import (
#    compute_segmentations,
#    blend_image_and_classified_regions_pil,
#)
from skimage import io as skio
from skimage.transform import rescale
from skimage import draw
import skimage
#from trainable_segmentation import multiscale_basic_features
import io
import base64
import matplotlib.pyplot as plt
import PIL.Image
import pickle
from time import time
#from joblib import Memory
import pims
import dash_vtk
import itk
from dash_vtk.utils import to_volume_state
import random
from dash_slicer import VolumeSlicer
import imageio
import numpy as np
from mlex_api.mlex_api import job_dispatcher
##### TEMPLATE MODULES
import templates
import re

text_color = {"dark": "#95969A", "light": "#595959"}
card_color = {"dark": "#2D3038", "light": "#FFFFFF"}
#memory = Memory("./joblib_cache", bytes_limit=3000000000, verbose=3)

#compute_features = memory.cache(multiscale_basic_features)
#### GLOBAL PARAMS ####
SEG_FEATURE_TYPES = ["intensity", "edges", "texture"]
NUM_LABEL_CLASSES = 5
DEFAULT_LABEL_CLASS = 0
DEFAULT_STROKE_WIDTH = 3  # gives line width of 2^3 = 8
class_label_colormap = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"]
class_labels = list(range(NUM_LABEL_CLASSES))
SAMPLE_DATA = 'data/bead_pack.tif'
#SAMPLE_DATA = 'data/arc_data/arc_stack.tiff'
MASK_OUTPUT_DIR = pathlib.Path('data/masks')
IM_OUTPUT_DIR = pathlib.Path('data/images')

features_dict = {}
# run if in docker compose service
AMQP_URL = os.environ['AMQP_URL']

### Initialize Connection to Work Queue
workq = job_dispatcher.workQueue(AMQP_URL)

# hardcoded model database as dict
print('loaded database')
MODEL_DATABASE = {"Random Forest":"aasgreen/random-forest-dc",
                "MSD": "aasgreen/msdnetwork-notebook",
                "":"",
                }

### AUTHENTICAION ###
VALID_USERNAME_PASSWORD_PAIRS = {
        'admin': 'admin'
        }

#### HELPER UTILS
def dcm_to_np(dir_path):
    np_volume = imageio.volread(dir_path)
    #np_volume = imageio.volread("imageio:stent.npz")
    return np_volume

def class_to_color(n):
    return class_label_colormap[n]

def color_to_class(c):
    return class_label_colormap.index(c)

def shapes_to_key(shapes):
    return json.dumps(shapes)

def label_to_colors(
img, colormap=px.colors.qualitative.Light24, alpha=128, color_class_offset=0):
    """
    Take MxN matrix containing integers representing labels and return an MxNx4
    matrix where each label has been replaced by a color looked up in colormap.
    colormap entries must be strings like plotly.express style colormaps.
    alpha is the value of the 4th channel
    color_class_offset allows adding a value to the color class index to force
    use of a particular range of colors in the colormap. This is useful for
    example if 0 means 'no class' but we want the color of class 1 to be
    colormap[0].
    """
    def fromhex(n):
        return int(n, base=16)
    colormap = [
        tuple([fromhex(h[s : s + 2]) for s in range(0, len(h), 2)])
        for h in [c.replace("#", "") for c in colormap]
    ]
    cimg = np.zeros(img.shape[:2] + (3,), dtype="uint8")
    minc = np.min(img)
    maxc = np.max(img)
    for c in range(int(minc), int(maxc) + 1):
        cimg[img == c] = colormap[(c + color_class_offset) % len(colormap)]
    return np.concatenate(
        (cimg, alpha * np.ones(img.shape[:2] + (1,), dtype="uint8")), axis=2
    )

def store_shapes_seg_pair(d, key, seg, remove_old=True):
    """
    Stores shapes and segmentation pair in dict d
    seg is a PIL.Image object
    if remove_old True, deletes all the old keys and values.
    """
    bytes_to_encode = io.BytesIO()
    seg.save(bytes_to_encode, format="png")
    bytes_to_encode.seek(0)
    data = base64.b64encode(bytes_to_encode.read()).decode()
    if remove_old:
        return {key: data}
    d[key] = data
    return d


def look_up_seg(d, key):
    """ Returns a PIL.Image object """
    data = d[key]
    img_bytes = base64.b64decode(data)
    img = PIL.Image.open(io.BytesIO(img_bytes))
    return img

def make_default_figure(image_index, shapes=[],stroke_color = '#ff4f00', image_cache=None):
    if image_cache == None:
        im = np_volume[image_index]
        width, height = im.shape[0:2]
        print('default figure')
    else:
        # image_cache is a dict, keys=filename, value=bin encoding
        img_bytes = base64.b64decode(image_cache)
        im_bbytes = io.BytesIO(img_bytes)
        print('first', im_bbytes)
        im= PIL.Image.open(im_bbytes)
        print('read in uploaded file {}'.format(im))
        width, height = np.array(im).shape[0:2]
        print(width, height)
        
    fig = px.imshow(im, binary_string=True)
    fig.update_layout(
            {
                'dragmode' : 'drawopenpath',
                'shapes' : shapes,
                'newshape.line.color' :stroke_color,
                'margin': {'l':0, 'r':0, 't':0, 'b':0},
                }
        )
    fig.update_xaxes(
        showgrid=False, range=(0, width), showticklabels=False, zeroline=False
        )
    fig.update_yaxes(
            showgrid=False,
            scaleanchor="x",
            range=(height, 0),
            showticklabels=False,
            zeroline=False,
        )
    return fig



#### SETUP DASH APP ####
external_stylesheets = [dbc.themes.BOOTSTRAP, "../assets/segmentation-style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
auth = dash_auth.BasicAuth(
        app,
        VALID_USERNAME_PASSWORD_PAIRS,
        )

server = app.server
app.title = "MLExchange Labeling | Image Segmentation"

#volume_state = dcm_to_volume(SAMPLE_DATA)
np_volume = dcm_to_np(SAMPLE_DATA)
CLASSIFIED_VOLUME = np.zeros(np_volume.shape)
N_IMAGES = np_volume.shape[0]
IMAGES_SHAPE = (np_volume.shape[1], np_volume.shape[2])

### SETUP INPUT/OUTPUT DIRECTORIES ###

### BEGIN DASH CODE ###
header = templates.header()


# Image Segmentation
segmentation = [
    dbc.Card(
        id="segmentation-card",
        children=[
            dbc.CardHeader(
                [
                    dcc.Upload(
                        id='upload-image',
                        children=html.Div([
                            'Drag and Drop',
                            ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': "5px",
                            'textAlign': 'center',
                            'maring':'10px'
                            },
                        multiple=True,
                        )

                ]
            ),
            dbc.CardBody(
                        dcc.Graph(
                        id="graph",
                        figure=make_default_figure(0),
                            config={
                                "modeBarButtonsToAdd": [
                                "drawrect",
                                "drawopenpath",
                                "eraseshape",
                                ]
                            },
                        ),
                    ),
            dbc.CardFooter(
                [
                    # Download links
                    html.A(id="download", download="classifier.json",),
                    html.Div(
                        children=[
                                dbc.Row(
                                    dbc.Col(dbc.FormGroup(
                                            [
                                                dbc.Label(
                                                    "Image Slider",
                                                    html_for='image-slider',
                                                    ),
                                                dcc.Slider(
                                                    id='image-slider',
                                                    min=0,
                                                    max=N_IMAGES,
                                                    value = 0,
                                                    updatemode='drag'
                                                    ),
                                                html.Div(
                                                    id='slider-output-container'
                                                    ),
                                            ],
                                    ), md=10),
                                    ),
                                dbc.Row(
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        "Download classified image",
                                        id="download-image-button",
                                        outline=True,
                                    ),
                                    dbc.Button(
                                        "Download classifier",
                                        id="download-button",
                                        outline=True,
                                    ),
                                ],
                                size="lg",
                                style={"width": "100%"},
                            ),
                            ),
                        ],
                    ),
                    html.A(id="download-image", download="classified-image.png",),
                ]
            ),
        ]
        )
        ]


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
    print(upload_image_contents)
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
    im = make_default_figure(
            image_slider_value, 
            shapes,
            stroke_color=class_to_color(label_class_value),
            image_cache=im_cache,
            )
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
        semi = label_to_colors(semi)
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

# sidebar
sidebar_label = [
    dbc.Card(
        id="sidebar-card",
        children=[
            dbc.CardHeader("Mask Tools"),
            dbc.CardBody(
                [
                    html.H6("Label class", className="card-title"),
                    # Label class chosen with buttons
                    html.Div(
                        id="label-class-buttons",
                        children=[
                            dbc.Button(
                                "%2d" % (n,),
                                id={"type": "label-class-button", "index": n},
                                style={"background-color": class_to_color(c)},
                            )
                            for n, c in enumerate(class_labels)
                        ],
                    ),
                    html.Hr(),
                    dbc.Form(
                        [
                            dbc.FormGroup(
                                [
                                    dbc.Label(
                                        "Width of annotation paintbrush",
                                        html_for="stroke-width",
                                    ),
                                    # Slider for specifying stroke width
                                    dcc.Slider(
                                        id="stroke-width",
                                        min=0,
                                        max=6,
                                        step=0.1,
                                        value=DEFAULT_STROKE_WIDTH,
                                    ),
                                ]
                            ),
                            dbc.FormGroup(
                                [
                                    dbc.Label('Choose Segmentation Model', className = 'mr-2'),
                                    dcc.Dropdown(id='seg-dropdown',
                                        options=[
                                        {"label": entry, "value": entry} for entry in MODEL_DATABASE],

                                        #options=[
                                        #    {'label': 'MSD', 'value': 'msd'},
                                        #    {'label' : 'RandomForest', 'value': 'random_forest'},
                                        #    ],
                                        style={'min-width':'250px'},
                                        value='MSD',
                                        ),
                                    ],
                                ),
                            html.Div(id='additional-seg-params',
                                children=[]
                                ),

                            dbc.FormGroup(
                                [
                                    dbc.Label('Choose Deployment Location', className = 'mr-2'),
                                    dcc.Dropdown(id='deploy-dropdown',
                                        options = [
                                            {'label': 'Vaughan (GPU)', 'value': 'vaughan'},
                                            {'label': 'Local (cpu)', 'value': 'local'},
                                            {'label': 'NERSC', 'value':'nersc'},
                                            {'label': 'Lab IT Cluster', 'value':'labit'},
                                            ],
                                        style={'min-width':'250px'},
                                        ),
                                    ],
                                ),

                            html.Hr(),
                            dbc.Row(
                                    [
                               dbc.Col(
                                    dbc.Button(
                                        "Train Segmenter",
                                        id="train-seg",
                                        outline=True,
                                    )
                                ),
                               dbc.Col(id = 'model-alert',
                                   children= dbc.Alert(id='model-train-alert', children='Status: Untrained', color='dark')
                                ),
                           ]
                           ),
                            dbc.Row([
                                dbc.Col(
                                    dbc.Button(
                                        "Segment Image Stack",
                                        id="compute-seg",
                                        outline=True,
                                    ),
                                ),
                                dbc.Col(id='seg-alert',
                                    children= dbc.Alert(id = 'model-seg-alert', children='Status: Not processed', color='dark')
                                ),

                                ],
                            ),

                            dcc.Checklist(
                                id="show-segmentation",
                                options=[
                                    {
                                        "label": "Show segmentation",
                                        "value": "Show segmentation",
                                    }
                                ],
                                value=[],
                            ),
                        ]
                    ),
                ]
            ),
        ],
    ),
]

random_forest_params = [
                dbc.FormGroup(
                                [
                                    html.H6(
                                        id="stroke-width-display",
                                        className="card-title",
                                    ),
                                    dbc.Label(
                                        "Blurring parameter",
                                        html_for="sigma-range-slider",
                                    ),
                                    dcc.RangeSlider(
                                        id="sigma-range-slider",
                                        min=0.01,
                                        max=20,
                                        step=0.01,
                                        value=[0.5, 16],
                                    ),
                                ]
                            ),
                            dbc.FormGroup(
                                [
                                    dbc.Label(
                                        "Select features",
                                        html_for="segmentation-features",
                                    ),
                                    dcc.Checklist(
                                        id="segmentation-features",
                                        options=[
                                            {"label": l.capitalize(), "value": l}
                                            for l in SEG_FEATURE_TYPES
                                        ],
                                        value=["intensity", "edges"],
                                    ),
                                ]
                            ),
                            ]

msd_params = [
                dbc.FormGroup(
                                [
                                    html.H6(
                                        id="msd-option-1",
                                        className="card-title",
                                    ),
                                    dbc.Label(
                                        id='learning-rate-slider-output',
                                        children="Learning Rate: .02",
                                        html_for="learning-rate",
                                    ),
                                    dcc.Slider(
                                        id="learning-rate-slider",
                                        min=0.01,
                                        max=.1,
                                        step=0.01,
                                        value=.02,
                                    ),
                                ]
                            ),
                           
                            ]


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
        return np.rint(np.array(indices_str, dtype=float)).astype(np.int)


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
                c_mask = self._path_to_mask(s['path'], color_to_class(s['line']['color']), image_shape_list[i])

                # update mask to include new shape
                masks[c_mask > 0] = c_mask[c_mask > 0]

            mask_f_name = str(mask_output_dir / 'n-{}'.format(key))
            print(mask_f_name)
            sav_return = np.savetxt(mask_f_name, masks)
            mask_names.append(mask_f_name)
        return mask_names

    
training_results = [html.Div([
    dcc.Graph(
        id='training-results',
        ),
    dcc.Interval(
        id='update-training-loss',
        interval=1*10000, # milliseconds
        n_intervals=0,
        ),
    ],
    id='training-visible',
    hidden=True,
    ),
    html.Div('')
    ]

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
    monitor message queue with very basic Interval, updating
    loss function and job status appropriately

    '''
    result=workq.get_logs()
    # result will tuple (pika.spec.Basic.GetOK(), pika.spec.Basic.Properties, message body)

    if result is not None:
        print(result) # get job_id
        print(json.loads(result[2])['job_type'])
        response = json.loads(result[2])
        job_type = response['job_type']
        current_job_id = result[1].correlation_id
        print(current_job_id)
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
    else:
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

    4. Send job messages into job queue
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
                                            job_type='training',
                                            deploy_location='local-vaughan',
                                            docker_uri=MODEL_DATABASE[seg_dropdown_value],
                                            docker_cmd='python feature_generation.py',
                                            kw_args='{} {}'.format(images_dir_docker,
                                                                   feature_dir_docker),
                                            work_queue=workq,
                                            GPU=False,
                                            corr_id=job_id
                                            )
        feat_job.launchJob()
        print('launched feature extraction on ml server')
        feature_results = feat_job.monitorJob()
        seg_job = job_dispatcher.simpleJob('supervised segmentation, random forest training',
                                           job_type='training',
                                           deploy_location='local-vaughan',
                                           docker_uri=MODEL_DATABASE[seg_dropdown_value],
                                           docker_cmd='python random_forest.py',
                                           kw_args='{} {} {}'.format(mask_dir_docker, feature_dir_docker,
                                                                     model_dir_docker),
                                           work_queue=workq,
                                           GPU=False,
                                           corr_id=job_id
                                           )
        seg_job.launchJob()
        print('launched segmentation on ml server')
        # feat_job = job_dispatcher.simpleJob('supervised segmentation, feature generation',
        #         deploy_location = 'local-vaughan',
        #         docker_uri = MODEL_DATABASE[seg_dropdown_value],
        #         docker_cmd = 'python3 feature_generation.py',
        #         kw_args = '/data/images /data/features',
        #         amqp_url=AMQP_URL,
        #         corr_id = job_id,
        #         )
        # feat_job.launchJob()
        #
        # seg_job = job_dispatcher.simpleJob('supervised segmentation, random forest training',
        #         job_type = 'training',
        #         deploy_location = 'local-vaughan',
        #         docker_uri = MODEL_DATABASE[seg_dropdown_value],
        #         docker_cmd = 'python3 random_forest.py',
        #         kw_args = '/data/masks /data/features /data/model',
        #         amqp_url=AMQP_URL,
        #         corr_id = job_id,
        #         )
        # feature_results = feat_job.monitorJob()
        # #ml_api.j.createJob()
        # seg_job.launchJob()
        # seg_results = seg_job.monitorJob()
        # #ml_api.job_dispatcher.vaughan.launchJob()
        # print(feature_results)
        # print(seg_results)
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
                work_queue=workq,
                GPU = True,
                corr_id = job_id
                )
        seg_job.launchJob()
        print('launched segmentation on ml server')
        #seg_results = seg_job.monitorJob()
        #print('server has returned value: ')
        #print(str(seg_results))

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
        
    #return [seg_results.decode('utf8')]
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
                                          work_queue=workq,
                                          GPU=GPU,
                                          corr_id=job_id,
                                          )
    # ml_api.j.create
    # if seg_dropdown_value == "Random Forest":
    #     docker_cmd = "python3 segment.py"
    #     kw_args = '/' + SAMPLE_DATA+ " /data/models/random-forest.model /data/output"
    #     GPU = False
    # elif (seg_dropdown_value == "MSD"):
    #
    #     docker_cmd = "python Segment.py"
    #     kw_args = '{} {} {}'.format(im_input_dir_dock, model_input_dir_dock, out_dir_dock)
    #     GPU=True
    # deploy_job = job_dispatcher.simpleJob('supervised segmentation, random forest deploy',
    #         job_type = "deploy",
    #         deploy_location = 'local-vaughan',
    #         docker_uri = MODEL_DATABASE[seg_dropdown_value],
    #         docker_cmd = docker_cmd,
    #         kw_args = kw_args,
    #         work_queue=workq,
    #         GPU = GPU,
    #         corr_id = job_id,
    #         )
   # ml_api.j.createJob()
    print(deploy_job.kw_args)
    deploy_job.launchJob()
    print('sending images to server to be segmented')
    #deploy_results = deploy_job.monitorJob()
    #print('server has returned results: ')
    #print(deploy_results)

    # now read in deploy results and save them to the
    # classified image store
    data = ''
        
    return [data, dbc.Alert(id='model-seg-alert', children='Status: Segmenting', color='red') ]

    # need to compute every image slice. I think we'll just
    # make something that generalizes to the MSD, where we have
    # a single worker node-- there is no point scaling up becaue we only have one or two gpus. 
    # in the created docker file, we will use job lib to create parallel shit.
                                   

meta = [
    html.Div(
        id="no-display",
        children=[
            # Store for user created masks
            # data is a list of dicts describing shapes
            dcc.Store(id='none', data=''),
            dcc.Store(id='username', data=''),
            dcc.Store(id='experiment-store', data={}),
            dcc.Store(id="masks", data={}),
            dcc.Store(id="classifier-store", data={}),
            dcc.Store(id="classifier-store-temp", data={}),
            dcc.Store(id="classified-image-store", data={}),
            dcc.Store(id="features_hash", data=""),
            dcc.Store(id='current-image-num', data=0),
            dcc.Store(id='image-store', data={}),
        ],
    ),
    html.Div(id="download-dummy"),
    html.Div(id="download-image-dummy"),
]

#controls = dbc.Card(
#        body=True,
#        children=dbc.Row(
#            [
#                dbc.Col(dbc.FormGroup([dbc.Label(label), component])) for label, component in sliders.items()
#            ]
#        ),
#    ) # end
#
# Iris bar figure
df = px.data.iris()

def drawFigure():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=px.bar(
                        df, x="sepal_width", y="sepal_length", color="species"
                    ).update_layout(
                        template='plotly_dark',
                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    ),
                    config={
                        'displayModeBar': False
                    }
                ) 
            ])
        ),  
    ])
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
    
