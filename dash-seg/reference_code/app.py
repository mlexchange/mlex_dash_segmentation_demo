import plotly.express as px
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plot_common
import json
from shapes_to_segmentations import (
    compute_segmentations,
    blend_image_and_classified_regions_pil,
)
from skimage import io as skio
from trainable_segmentation import multiscale_basic_features
import io
import base64
import matplotlib.pyplot as plt
import PIL.Image
import pickle
from time import time
from joblib import Memory
import pims
import dash_vtk
import itk
from dash_vtk.utils import to_volume_state
import random
from dash_slicer import VolumeSlicer
import imageio

memory = Memory("./joblib_cache", bytes_limit=3000000000, verbose=3)

compute_features = memory.cache(multiscale_basic_features)

STARTING_IMAGE_SLICE = 0
DEFAULT_STROKE_WIDTH = 3  # gives line width of 2^3 = 8
IMAGE_SKIP = 1 #how far we should skip when we click on 'next'

DEFAULT_IMAGE_PATH = "assets/segmentation_img.jpg"
MLEX_LOGO = "mlex.png"

SEG_FEATURE_TYPES = ["intensity", "edges", "texture"]

image_stack = pims.open('./data/bead_pack.tif')

# the number of different classes for labels
NUM_LABEL_CLASSES = 5
DEFAULT_LABEL_CLASS = 0
class_label_colormap = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"]
class_labels = list(range(NUM_LABEL_CLASSES))
# we can't have less colors than classes
assert NUM_LABEL_CLASSES <= len(class_label_colormap)

# Font and background colors associated with each theme
text_color = {"dark": "#95969A", "light": "#595959"}
card_color = {"dark": "#2D3038", "light": "#FFFFFF"}

def dcm_to_volume(dir_path):
    itk_image = itk.imread(dir_path)
    vtk_image = itk.vtk_image_from_image(itk_image)
    volume_state = to_volume_state(vtk_image)
    return volume_state

def dcm_to_np(dir_path):
    np_volume = imageio.volread(dir_path)
    np_volume = imageio.volread("imageio:stent.npz")
    return np_volume
def class_to_color(n):
    return class_label_colormap[n]


def color_to_class(c):
    return class_label_colormap.index(c)


img = skio.imread(DEFAULT_IMAGE_PATH)
features_dict = {}

external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/segmentation-style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
app.title = "MLExchange Labeling | Image Segmentation"

SAMPLE_DATA = 'data/bead_pack.tif'
volume_state = dcm_to_volume(SAMPLE_DATA)
np_volume = dcm_to_np(SAMPLE_DATA)

def make_default_figure(
    #images=[plot_common.img_array_to_pil_image(image_stack[19])],
    images = [plot_common.img_array_to_pil_image(image_stack[STARTING_IMAGE_SLICE])],
    stroke_color=class_to_color(DEFAULT_LABEL_CLASS),
    stroke_width=DEFAULT_STROKE_WIDTH,
    shapes=[],
):
    fig = plot_common.dummy_fig()
    plot_common.add_layout_images_to_fig(fig, images)
    fig.update_layout(
        {
            "dragmode": "drawopenpath",
            "shapes": shapes,
            "newshape.line.color": stroke_color,
            "newshape.line.width": stroke_width,
            "margin": dict(l=0, r=0, b=0, t=0, pad=4),
        }
    )
    return fig


def shapes_to_key(shapes):
    return json.dumps(shapes)


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


# Modal
with open("explanations.md", "r") as f:
    howto_md = f.read()

modal_overlay = dbc.Modal(
    [
        dbc.ModalBody(html.Div([dcc.Markdown(howto_md)], id="howto-md")),
        dbc.ModalFooter(dbc.Button("Close", id="howto-close", className="howto-bn")),
    ],
    id="modal",
    size="lg",
)

button_howto = dbc.Button(
    "Learn more",
    id="howto-open",
    outline=True,
    color="info",
    # Turn off lowercase transformation for class .button in stylesheet
    style={"textTransform": "none"},
)

button_github = dbc.Button(
    "View Code on github",
    outline=True,
    color="primary",
    href="https://github.com/als-computing/dash-seg",
    id="gh-link",
    style={"text-transform": "none"},
)

#Header
header = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(
                            id="logo",
                            src=app.get_asset_url(MLEX_LOGO),
                            height="100px",
                        ),
                        md="auto",
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H3("MLExchange | Labelling"),
                                    html.P("Image segmentation"),
                                ],
                                id="app-title",
                            )
                        ],
                        md=True,
                        align="center",
                    ),
                ],
                align="center",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.NavbarToggler(id="navbar-toggler"),
                            dbc.Collapse(
                                dbc.Nav(
                                    [
                                        #dbc.NavItem(button_howto),
                                        dbc.NavItem(button_github),
                                    ],
                                    navbar=True,
                                ),
                                id="navbar-collapse",
                                navbar=True,
                            ),
                            modal_overlay,
                        ],
                        md=2,
                    ),
                ],
                align="center",
            ),
        ],
        fluid=True,
    ),
    dark=True,
    color="dark",
    sticky="top",
)

# Description
#description = dbc.Col(
#    [
#        dbc.Card(
#            id="description-card",
#            children=[
#                dbc.CardHeader("Explanation"),
#                dbc.CardBody(
#                    [
#                        dbc.Row(
#                            [
#                                dbc.Col(
#                                    [
#                                        html.Img(
#                                            src="assets/segmentation_img_example_marks.jpg",
#                                            width="200px",
#                                        )
#                                    ],
#                                    md="auto",
#                                ),
#                                dbc.Col(
#                                    html.P(
#                                        "This is an example of interactive machine learning for image classification. "
#                                        "To train the classifier, draw some marks on the picture using different colors for "
#                                        'different parts, like in the example image. Then enable "Show segmentation" to see the '
#                                        "classes a Random Forest Classifier gave to regions of the image, based on the marks you "
#                                        "used as a guide. You may add more marks to clarify parts of the image where the "
#                                        "classifier was not successful and the classification will update."
#                                    ),
#                                    md=True,
#                                ),
#                            ]
#                        ),
#                    ]
#                ),
#            ],
#        )
#    ],
#    md=12,
#)

# vtk view
vtk_view = dash_vtk.View(
        dash_vtk.VolumeRepresentation(
            [dash_vtk.VolumeController(), dash_vtk.ShareDataSet(dash_vtk.Volume(state=volume_state))]
            )
        )

slice_property = {"colorWindow":4095, "colorLevel":1000}
slicerV = VolumeSlicer(app, np_volume)
slicerV.graph.config['scrollZoom'] = False
slicerV.graph.config['modeBarButtonsToAdd']= ['drawopenpath', 'eraseshape']
config = {
        "modeBarButtonsToAdd":  [
            "drawopenpath",
            "eraseshape",
            ]
        }
slice_view_2 = html.Div(id='slice-graph',
        children=[
            html.Div(id='slice-graph-figure', children=slicerV.graph), slicerV.slider, *slicerV.stores
            ]
        )

slice_options = [
        dbc.Card(
            id="slice-sidebar",
            children=[
                dbc.CardHeader("Mask Options"),
                dbc.CardBody(
                    [
                        html.H3("Label Mask Color", className="slice-card-title"),
                        html.Div(
                            id="slice-label-class-buttons",
                            children=[
                                dbc.Button(
                                    "%2d" % (n,),
                                    id={"type": "slice-mask-button", "index": n},
                                    style={'background-color': class_to_color(c)},
                                    )
                                    for n, c in enumerate(class_labels)
                                    ],
                            )
                        ]
                    )
                ]
            )
        ]

slice_view = dash_vtk.View(
                id='slice-view',
                cameraParallelProjection=True,
                cameraPosition=[1,0,0],
                interactorSettings = [],
                children=[
                    dash_vtk.SliceRepresentation(
                        id='slice-repr-i',
                        iSlice=128,
                        property=slice_property,
                        children=dash_vtk.ShareDataSet(),
                    )
                ]
            )

sliders = {
        "Slice i": dcc.Slider(id='slider-i', min=0, max=200, value=128, updatemode='drag'),
        "Color Level": dcc.Slider(id="slider-lvl", min=0, max=4095, value=1000),
        "Color Window": dcc.Slider(id="slider-window", min=0, max=4095, value=4095),
        }
@app.callback(
        [
            Output("slice-view", "triggerRender"),
            Output("slice-repr-i", "iSlice"),
            Output("slice-repr-i", "property"),
            ],
        [
            Input("slider-i", "value"),
            Input("slider-lvl", "value"),
            Input("slider-window", "value"),

            ],
        )
def update_slice_property(i, level, window):
    render_call = random.random() # needs to be different from the previous value to trigger rerender
    slice_prop = {"colorLevel": level, "colorWindow" : window}
    return [render_call, i, slice_prop]
# Image Segmentation
segmentation = [
    dbc.Card(
        id="segmentation-card",
        children=[
            dbc.CardHeader("Viewer"),
            dbc.CardBody(
                [
                    # Wrap dcc.Loading in a div to force transparency when loading
                    html.Div(
                        id="transparent-loader-wrapper",
                        children=[
                            dcc.Loading(
                                id="segmentations-loading",
                                type="circle",
                                     #Graph
                                    children=dcc.Graph(
                                        id="graph",
                                        figure=make_default_figure(),
                                        config={
                                            "modeBarButtonsToAdd": [
                                                "drawrect",
                                                "drawopenpath",
                                                "eraseshape",
                                            ]
                                        },
                                    ),
                                    ),
                                ],
                            )
                        ],
                    ),
                ]
            ),
            dbc.CardFooter(
                [
                    # Download links
                    html.A(id="download", download="classifier.json",),
                    html.Div(
                        children=[
                                dbc.Row(
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button(
                                                "Previous Image",
                                                id='previous-image',
                                                outline=True,
                                                ),
                                            dbc.Button(
                                                'Next Image',
                                                id='next-image',
                                                outline=True,
                                                ),
                                        ],
                                    ),
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

# sidebar
sidebar_label = [
    dbc.Card(
        id="sidebar-card",
        children=[
            dbc.CardHeader("Tools"),
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
                            # Indicate showing most recently computed segmentation
                            dbc.Button(
                                    "Train Segmenter",
                                    id="train-seg",
                                    outline=True,
                                    ),

    #                        dbc.RadioItems(
    #                                id="select-viewing-option",
    #                                options=[
    #                                    {"label": "Show segmentation (train new)", "value": "Show segmentation"},
    #                                    {"label": "Show segmentation (previous classifier)", "value": "Remember training"},
    #                                    {"label": "Show Image", "value": "Show Image"},
    #                                    ],
    #                                value = "Show Image",
    #                                ),
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
    #                         dcc.Checklist(
    #                            id="show-seg-2",
    #                            options=[
    #                                {
    #                                    "label": "Show segmentation 2",
    #                                    "value": "Show segmentation 2",
    #                                }
    #                            ],
    #                            value=[],
    #                        ),
                        ]
                    ),
                ]
            ),
        ],
    ),
]

meta = [
    html.Div(
        id="no-display",
        children=[
            # Store for user created masks
            # data is a list of dicts describing shapes
            dcc.Store(id="masks", data={"shapes": []}),
            dcc.Store(id="classifier-store", data={}),
            dcc.Store(id="classifier-store-temp", data={}),
            dcc.Store(id="classified-image-store", data=""),
            dcc.Store(id="features_hash", data=""),
            dcc.Store(id='current-image-num', data=0),
        ],
    ),
    html.Div(id="download-dummy"),
    html.Div(id="download-image-dummy"),
]

controls = dbc.Card(
        body=True,
        children=dbc.Row(
            [
                dbc.Col(dbc.FormGroup([dbc.Label(label), component])) for label, component in sliders.items()
            ]
        ),
    ) # end

viewer =[
                                dbc.Col(
                                    [
                                        html.Div(
                                            style={"height": "calc(100vh - 50px)", "width": "100%"},
                                            children=[html.Div(style={"height": "100%", "width": "100%"}, children=vtk_view)] 
                                        )
                                   ]
                                ),
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.Div(
                                                            style={"height": "calc(100vh - 50px)", "width": "100%"}, 
                                                            children=[html.Div(style={"height": "100%", "width": "100%"}, children=slice_view)] 
                                                        ) 
                                                    ]
                                                ),
                                                dbc.CardFooter(
                                                    children=dbc.Row(
                                                        [ # list comprehension
                                                            dbc.Col(dbc.FormGroup([dbc.Label(label), component]))
                                                            for label, component in sliders.items()
                                                        ]
                                                    )
                                                ),
                                        ]

                                        )
                                   ]
                               )
            
                            
                        
                    
                
            
]

app.layout = html.Div(
                        [
                            header,
#                            dbc.Row(dbc.Col(controls)),
                            #dbc.Row(dbc.Col(viewer)),
                            dbc.Container(
                                [
                    #                dbc.Row(description),
                                    dbc.Row(
                                        id="app-content",
                                        children=[dbc.Col(segmentation, md=8), dbc.Col(sidebar_label, md=4)],
                                    ),
                                    dbc.Row(id='3d-viewing', children=viewer),
                                    dbc.Row(id='slice2', children = [dbc.Col(slice_view_2, md=8), dbc.Col(slice_options, md=4)]),
                                    dbc.Row(dbc.Col(html.Div(id='debug-print', children='this is test'), md=8)),
                                    dbc.Row(dbc.Col(meta)),
                                ],
                                fluid=True,
                            ),
                        ]
                    ) #end

# Converts image classifier to a JSON compatible encoding and creates a
# dictionary that can be downloaded
# see use_ml_image_segmentation_classifier.py
def save_img_classifier(clf, label_to_colors_args, segmenter_args):
    clfbytes = io.BytesIO()
    pickle.dump(clf, clfbytes)
    clfb64 = base64.b64encode(clfbytes.getvalue()).decode() #why are we encoding/decoding
    return {
        "classifier": clfb64,
        "segmenter_args": segmenter_args,
        "label_to_colors_args": label_to_colors_args,
    }


def show_segmentation(image_path, mask_shapes, features, segmenter_args, clf=None):
    """ adds an image showing segmentations to a figure's layout """

    # add 1 because classifier takes 0 to mean no mask
    shape_layers = [color_to_class(shape["line"]["color"]) + 1 for shape in mask_shapes]
    label_to_colors_args = {
        "colormap": class_label_colormap,
        "color_class_offset": -1,
    }
    segimg, _, clf = compute_segmentations(
        mask_shapes,
        img_path=image_path,
        shape_layers=shape_layers,
        label_to_colors_args=label_to_colors_args,
        features=features,
        clf=clf,
    )
    # get the classifier that we can later store in the Store
    classifier = save_img_classifier(clf, label_to_colors_args, segmenter_args)
    segimgpng = plot_common.img_array_to_pil_image(segimg)
    return (segimgpng, classifier)

def only_show_segmentation(features, clf, label_to_colors_args):
    segimg, _, clf = compute_segmentations(
        features=features,
        clf=clf,
        label_to_colors_args = label_to_colors_args,
    )
    # get the classifier that we can later store in the Store
    segimgpng = plot_common.img_array_to_pil_image(segimg)
    return (segimgpng, clf)

@app.callback(
        [
            #Output("graph-style-annotations", "figure"),
            Output('debug-print', 'children'),
        #    Output('slicer1-graph', 'figure')

        ],
        [
            Input(
            {'type': 'label-class-button', 'index':dash.dependencies.ALL}, 
            "n_clicks_timestamp",
                )
        ],

        [
            State("slicer1-graph", "figure")
        ], prevent_initial_call = True
        )
def choose_mask_color(
    stroke_color_button_value,
    slice_graph_children,
    ):
    print(slice_graph_children)
    # find label class value by finding button with the most recent click
    if stroke_color_button_value is None:
        label_class_value = DEFAULT_LABEL_CLASS
    else:
        label_class_value = max(
            enumerate(stroke_color_button_value),
            key=lambda t: 0 if t[1] is None else t[1],
        )[0]

    slicerV.graph.figure.update_layout(
            {
                'dragmode': 'drawopenpath',
                'newshape.line.color': class_to_color(label_class_value),
            }
            )
    print(slicerV.graph.figure)
    print(class_to_color(label_class_value))
    print('activated!')
    return(['{}, {}'.format(slice_graph_children, class_to_color(label_class_value))])#, slice_graph_children])

@app.callback(
    [
        Output("graph", "figure"),
        Output("masks", "data"),
        Output("stroke-width-display", "children"),
        Output("classifier-store-temp", "data"),
        Output("classifier-store", "data"),
        Output("classified-image-store", "data"),
        Output("current-image-num", "data"),
    ],
    [
        Input("graph", "relayoutData"),
        Input(
            {"type": "label-class-button", "index": dash.dependencies.ALL},
            "n_clicks_timestamp",
        ),
        Input("stroke-width", "value"),
        Input("train-seg", "n_clicks"),
        Input("show-segmentation", "value"),
        Input("download-button", "n_clicks"),
        Input("download-image-button", "n_clicks"),
        Input("segmentation-features", "value"),
        Input("sigma-range-slider", "value"),
        Input("next-image", "n_clicks"),
        Input("previous-image", "n_clicks")
    ],
    [State("masks", "data"), State('current-image-num', 'data'), State('classifier-store-temp', 'data'),],
)
def annotation_react(
    graph_relayoutData,
    any_label_class_button_value,
    stroke_width_value,
    train_seg_n_clicks,
    show_segmentation_value,
    download_button_n_clicks,
    download_image_button_n_clicks,
    segmentation_features_value,
    sigma_range_slider_value,
    next_image_n_clicks,
    previous_image_n_clicks,
    masks_data,
    current_image_num,
    current_classifier,
):
    classified_image_store_data = dash.no_update
    classifier_store_data = dash.no_update
    classifier_store_data_temp = dash.no_update
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    img = plot_common.img_array_to_pil_image(image_stack[current_image_num])

    if cbcontext == 'next-image.n_clicks':
        print('next!')
        current_image_num +=1*IMAGE_SKIP
        masks_data['shapes'] = []
        print('current image: {}'.format(current_image_num))

#        fig = make_default_figure(images = [plot_common.img_array_to_pil_image(image_stack[current_image_num])], 
#            stroke_color=class_to_color(label_class_value),
#            stroke_width=stroke_width,
#            shapes=masks_data["shapes"],
#        )

    if cbcontext == 'previous-image.n_clicks':
        current_image_num -=1*IMAGE_SKIP
        masks_data['shapes'] = []
 
        #fig = make_default_figure(images = [plot_common.img_array_to_pil_image(image_stack[current_image_num])], 
        #stroke_color=class_to_color(label_class_value),
        #stroke_width=stroke_width,
        #shapes=masks_data["shapes"],
        #)       
        print('current image: {}'.format(current_image_num))


#    if (cbcontext in ["segmentation-features.value", "sigma-range-slider.value"] or (
#        ( ("Show segmentation" in show_segmentation_value) or ("Show segmentation" in select_viewing_option )))
#        and (len(masks_data["shapes"]) > 0)
#    ):
#        segmentation_features_dict = {
#            "intensity": False,
#            "edges": False,
#            "texture": False,
#        }
#        for feat in segmentation_features_value:
#            segmentation_features_dict[feat] = True
#        t1 = time()
#        features = compute_features(
#            plot_common.pil_image_to_ndarray(img), #expects a ndarray
#            **segmentation_features_dict,
#            sigma_min=sigma_range_slider_value[0],
#            sigma_max=sigma_range_slider_value[1],
#        )
#        t2 = time()
#        print(t2 - t1)
#
    if cbcontext == "graph.relayoutData":
        if "shapes" in graph_relayoutData.keys():
            masks_data["shapes"] = graph_relayoutData["shapes"]
        else:
            return dash.no_update
    stroke_width = int(round(2 ** (stroke_width_value)))
    # find label class value by finding button with the most recent click
    if any_label_class_button_value is None:
        label_class_value = DEFAULT_LABEL_CLASS
    else:
        label_class_value = max(
            enumerate(any_label_class_button_value),
            key=lambda t: 0 if t[1] is None else t[1],
        )[0]

    fig = make_default_figure(images = [plot_common.img_array_to_pil_image(image_stack[current_image_num])], 
        stroke_color=class_to_color(label_class_value),
        stroke_width=stroke_width,
        shapes=masks_data["shapes"],
    )

    if cbcontext == "train-seg.n_clicks":
        #first do feature extraction
        segmentation_features_dict = {
            "intensity": False,
            "edges": False,
            "texture": False,
        }
        for feat in segmentation_features_value:
            segmentation_features_dict[feat] = True
        t1 = time()
        features = compute_features(
            plot_common.pil_image_to_ndarray(img), #expects a ndarray
            **segmentation_features_dict,
            sigma_min=sigma_range_slider_value[0],
            sigma_max=sigma_range_slider_value[1],
        )
        t2 = time()
        print(t2 - t1)

        #now, train segmenter
        segimgpng = None
        try:
            feature_opts = dict(segmentation_features_dict=segmentation_features_dict)
            feature_opts["sigma_min"] = sigma_range_slider_value[0]
            feature_opts["sigma_max"] = sigma_range_slider_value[1]
            segimgpng, clf = show_segmentation(
            plot_common.img_array_to_pil_image(image_stack[current_image_num]), masks_data["shapes"], features, feature_opts
            )
            classifier_store_data_temp = clf #this will be written to hidden div for future use
            print('seg trained!')
        except ValueError:
            # if segmentation fails, draw nothing
            pass

    if ( "Show segmentation" in show_segmentation_value):
        #first do feature extraction
        segmentation_features_dict = {
            "intensity": False,
            "edges": False,
            "texture": False,
        }
        for feat in segmentation_features_value:
            segmentation_features_dict[feat] = True
        t1 = time()
        features = compute_features(
            plot_common.pil_image_to_ndarray(img), #expects a ndarray
            **segmentation_features_dict,
            sigma_min=sigma_range_slider_value[0],
            sigma_max=sigma_range_slider_value[1],
        )
        t2 = time()
        print(t2 - t1)


        #draw predicted labels on image using cached classifier (classifier_store_data)
        clf = base64.b64decode(current_classifier['classifier'])
        clf = pickle.loads(clf)
        label_to_colors_args = current_classifier['label_to_colors_args']
        segimgpng = None
        try:
            segimgpng, clf = only_show_segmentation(features, clf, label_to_colors_args)

        except ValueError:
            pass
         
        if segimgpng is not None:
            images_to_draw = [segimgpng]
        fig = plot_common.add_layout_images_to_fig(fig, images_to_draw)

        fig.update_layout(uirevision="segmentation")


    if cbcontext == "download-button.n_clicks":
        classifier_store_data = clf
    if cbcontext == "download-image-button.n_clicks":
        classified_image_store_data = plot_common.pil_image_to_uri(
                        blend_image_and_classified_regions_pil(
                            PIL.Image.open(DEFAULT_IMAGE_PATH), segimgpng)
                        )
#    if cbcontext == "select-viewing-option.value":
#        print('selecting option: {}'.format(select_viewing_option))
#
#        if ("Show segmentation" in select_viewing_option) and (len(masks_data["shapes"]) > 0):
#            segimgpng = None
#            try:
#                feature_opts = dict(segmentation_features_dict=segmentation_features_dict)
#                feature_opts["sigma_min"] = sigma_range_slider_value[0]
#                feature_opts["sigma_max"] = sigma_range_slider_value[1]
#                segimgpng, clf = show_segmentation(
#                plot_common.img_array_to_pil_image(image_stack[current_image_num]), masks_data["shapes"], features, feature_opts
#                )
#                if cbcontext == "download-button.n_clicks":
#                    classifier_store_data = clf
#                if cbcontext == "download-image-button.n_clicks":
#                    classified_image_store_data = plot_common.pil_image_to_uri(
#                        blend_image_and_classified_regions_pil(
#                            PIL.Image.open(DEFAULT_IMAGE_PATH), segimgpng
#                        )
#                    )
#            except ValueError:
#                # if segmentation fails, draw nothing
#                pass
#            images_to_draw = []
#            if segimgpng is not None:
#                images_to_draw = [segimgpng]
#            fig = plot_common.add_layout_images_to_fig(fig, images_to_draw)
#
#
#    if ( ( ("Show segmentation" in show_segmentation_value ) or ( "Show segmentation" in select_viewing_option) or 
#            ( "Remember training" in select_viewing_option) )  and (len(masks_data["shapes"]) > 0) ):
#        segimgpng = None
#        print('current classifier: {}'.format(current_classifier) )
#        if ( ( "Remember training" in select_viewing_option) and (len(current_classifier) > 0) ):
#            clf = current_classifier
#        else:
#            clf = None
#        try:
#            feature_opts = dict(segmentation_features_dict=segmentation_features_dict)
#            feature_opts["sigma_min"] = sigma_range_slider_value[0]
#            feature_opts["sigma_max"] = sigma_range_slider_value[1]
#            segimgpng, clf = show_segmentation(
#            plot_common.img_array_to_pil_image(image_stack[current_image_num]), masks_data["shapes"], features, feature_opts, 
#            clf = clf)
#            if cbcontext == "download-button.n_clicks":
#                classifier_store_data = clf
#            if cbcontext == "download-image-button.n_clicks":
#                classified_image_store_data = plot_common.pil_image_to_uri(
#                    blend_image_and_classified_regions_pil(
#                        PIL.Image.open(DEFAULT_IMAGE_PATH), segimgpng
#                    )
#                )
#        except ValueError:
#            # if segmentation fails, draw nothing
#            pass
#        images_to_draw = []
#        if segimgpng is not None:
#            images_to_draw = [segimgpng]
#        fig = plot_common.add_layout_images_to_fig(fig, images_to_draw)
    return (
        fig,
        masks_data,
        "Current paintbrush width: %d" % (stroke_width,),
        classifier_store_data_temp,
        classifier_store_data,
        classified_image_store_data,
        current_image_num,
    )


# set the download url to the contents of the classifier-store (so they can be
# downloaded from the browser's memory)
app.clientside_callback(
    """
function(the_store_data) {
    let s = JSON.stringify(the_store_data);
    let b = new Blob([s],{type: 'text/plain'});
    let url = URL.createObjectURL(b);
    return url;
}
""",
    Output("download", "href"),
    [Input("classifier-store", "data")],
)


# set the download url to the contents of the classified-image-store (so they can be
# downloaded from the browser's memory)
app.clientside_callback(
    """
function(the_image_store_data) {
    return the_image_store_data;
}
""",
    Output("download-image", "href"),
    [Input("classified-image-store", "data")],
)

# simulate a click on the <a> element when download.href is updated
app.clientside_callback(
    """
function (download_href) {
    let elem = document.querySelector('#download');
    elem.click()
    return "";
}
""",
    Output("download-dummy", "children"),
    [Input("download", "href")],
)

# simulate a click on the <a> element when download.href is updated
app.clientside_callback(
    """
function (download_image_href) {
    let elem = document.querySelector('#download-image');
    elem.click()
    return "";
}
""",
    Output("download-image-dummy", "children"),
    [Input("download-image", "href")],
)


# Callback for modal popup
#@app.callback(
#    Output("modal", "is_open"),
#    [Input("howto-open", "n_clicks"), Input("howto-close", "n_clicks")],
#    [State("modal", "is_open")],
#)
#def toggle_modal(n1, n2, is_open):
#    if n1 or n2:
#        return not is_open
#    return is_open
#
#
# we use a callback to toggle the collapse on small screens
#@app.callback(
#    Output("navbar-collapse", "is_open"),
#    [Input("navbar-toggler", "n_clicks")],
#    [State("navbar-collapse", "is_open")],
#)
#def toggle_navbar_collapse(n, is_open):
#    if n:
#        return not is_open
#    return is_open
#

if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0')
