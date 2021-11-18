import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import dash_auth
import dash_table
##### HELPER UTILS
import helper_utils
##### TEMPLATE MODULES
import templates

### GLOBAL VARIABLES
SEG_FEATURE_TYPES = ["intensity", "edges", "texture"]
NUM_LABEL_CLASSES = 5
class_labels = list(range(NUM_LABEL_CLASSES))
np_volume = helper_utils.dcm_to_np('data/bead_pack.tif')
DEFAULT_STROKE_WIDTH = 3  # gives line width of 2^3 = 8
# hardcoded model database as dict


### AUTHENTICAION ###
VALID_USERNAME_PASSWORD_PAIRS = {
        'admin': 'admin'
        }

#### SETUP DASH APP ####
external_stylesheets = [dbc.themes.BOOTSTRAP, "../assets/segmentation-style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
auth = dash_auth.BasicAuth(
        app,
        VALID_USERNAME_PASSWORD_PAIRS,
        )

server = app.server
app.title = "MLExchange Labeling | Image Segmentation"

### BEGIN DASH CODE ###
header = templates.header()


# presents the image with slider
segmentation = [
    dbc.Card(
        id="segmentation-card",
        children=[
            dbc.CardHeader(
                [
                    dbc.Label('Choose Dataset', className='mr-2'),
                    dcc.Dropdown(
                        id='dataset-selection',
                        options=[
                            {'label': 'Tiled', 'value': 'tiled'}
                        ],
                        value = 'tiled',
                        clearable=False,
                        style={'margin-bottom': '1rem'}
                    ),
                ]
            ),
            dbc.CardBody(
                        dcc.Graph(
                        id="graph",
                        figure = helper_utils.make_default_figure(0,np_volume),
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
                                    dbc.Col(
                                        html.Div(
                                            children = [dbc.Label(
                                                            "Image Slider",
                                                            className="mb-3",
                                                            html_for='image-slider',
                                                          ),
                                                          dcc.Slider(
                                                            id='image-slider',
                                                            min=0,
                                                            max=200,
                                                            value = 0,
                                                            updatemode='drag',
                                                            tooltip={"placement": "top", "always_visible": True},
                                                            marks={0: '0', 199: '199'},
                                                          )],
                                            style={'margin-bottom': '1rem', 'align-items': 'center', 'justify-content': 'center'}
                                        ), 
                                    md=10),
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
                                            )
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

# sidebar - labeling tools
sidebar_label = [
    dbc.Card(
        id="sidebar-card",
        style={"width": "100%"},
        children=[
            dbc.CardHeader("Annotation Tools"),
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
                                style={"background-color": helper_utils.class_to_color(c)},
                            )
                            for n, c in enumerate(class_labels)
                        ],
                        className='row',
                        style={'margin-bottom': '1rem', 'align-items': 'center', 'justify-content': 'center'}
                    ),
                    html.Div([
                        dbc.Label(
                            "Width of annotation paintbrush",
                            className="mb-3",
                            html_for="stroke-width"
                        ),
                        # Slider for specifying stroke width
                        dcc.Slider(
                            id="stroke-width",
                            min=0,
                            max=6,
                            step=0.1,
                            value=DEFAULT_STROKE_WIDTH,
                            tooltip={"placement": "top", "always_visible": True},
                        ),
                    ]),
            ]),
            dbc.CardHeader("Model"),
            dbc.CardBody(
                [
                    dbc.Form(
                        [
                            html.Hr(),
                            html.Div(
                                [
                                    dbc.Button(
                                        "SAVE ANNOTATION",
                                        id="train-seg",
                                        outline=True,
                                        size="lg",
                                        className='m-1',
                                        style={'width': '60%'}
                                    )
                                ],
                                className='row',
                                style={'align-items': 'center', 'justify-content': 'center'}
                            ),
                            ]),
                    ],
            )
        ],
    ),
]


# metadata
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
                    dbc.Row(dbc.Col(meta)),
                ]
            )
        ]
        )



