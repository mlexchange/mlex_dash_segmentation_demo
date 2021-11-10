import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import dash_auth
import dash_table
import numpy as np
##### HELPER UTILS
import helper_utils
##### TEMPLATE MODULES
import templates

### GLOBAL VARIABLES
SEG_FEATURE_TYPES = ["intensity", "edges", "texture"]
NUM_LABEL_CLASSES = 5
class_labels = list(range(NUM_LABEL_CLASSES))
SAMPLE_DATA = 'data/bead_pack.tif'
DEFAULT_STROKE_WIDTH = 3  # gives line width of 2^3 = 8
np_volume = helper_utils.dcm_to_np(SAMPLE_DATA)
CLASSIFIED_VOLUME = np.zeros(np_volume.shape)
N_IMAGES = np_volume.shape[0]
IMAGES_SHAPE = (np_volume.shape[1], np_volume.shape[2])
# hardcoded model database as dict
MODEL_DATABASE = {"Random Forest": "mlexchange/random-forest-dc",
                  "MSD": "mlexchange/msdnetwork-notebook",
                  "K-Means": "mlexchange/k-means-dc",
                }


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
                        figure=helper_utils.make_default_figure(0,np_volume),
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
                                    dbc.Button(
                                        "test api",
                                        id="api-button",
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

# sidebar - labeling tools
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
                                style={"background-color": helper_utils.class_to_color(c)},
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
                                    dbc.Label('Choose Segmentation Model', className='mr-2'),
                                    dcc.Dropdown(id='seg-dropdown',
                                                 options=[
                                                     {"label": entry, "value": entry} for entry in MODEL_DATABASE],

                                                 # options=[
                                                 #    {'label': 'MSD', 'value': 'msd'},
                                                 #    {'label' : 'RandomForest', 'value': 'random_forest'},
                                                 #    ],
                                                 style={'min-width': '250px'},
                                                 value='MSD',
                                                 ),
                                ],
                            ),
                            html.Div(id='additional-seg-params',
                                     children=[]
                                     ),

                            dbc.FormGroup(
                                [
                                    dbc.Label('Choose Deployment Location', className='mr-2'),
                                    dcc.Dropdown(id='deploy-dropdown',
                                                 options=[
                                                     {'label': 'Vaughan (GPU)', 'value': 'vaughan'},
                                                     {'label': 'mlsandbox (cpu)', 'value': 'mlsandbox'},
                                                     {'label': 'Local (cpu)', 'value': 'local'},
                                                     {'label': 'NERSC', 'value': 'nersc'},
                                                     {'label': 'Lab IT Cluster', 'value': 'labit'},
                                                 ],
                                                 style={'min-width': '250px'},
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
                                    dbc.Col(id='model-alert',
                                            children=dbc.Alert(id='model-train-alert', children='Status: Untrained',
                                                               color='dark')
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
                                        children=dbc.Alert(id='model-seg-alert', children='Status: Not processed',
                                                           color='dark')
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

# parameters for RF
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



# parameters for MSD
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

# parameters for K-Means
kmeans_params = [
    dbc.FormGroup(
        [
            html.H6(
                id="K-Means",
                className="card-title",
            ),
            dbc.Label(
                id='n_clusters-output',
                children="Number of Clusters: 2",
                html_for="n-clusters",
            ),
            dcc.Slider(
                id="n_clusters",
                min=2,
                max=10,
                step=1,
                value=2,
            ),
        ]
    ),

]


# show table

# job status display
job_status_display = [
    html.Div(
        children=[
            html.H5('List of jobs'),
            dash_table.DataTable(
                id='jobs_table',
                columns=[
                    {'name': 'Job ID', 'id': 'job_id'},
                    {'name': 'Type', 'id': 'job_type'},
                    {'name': 'Status', 'id': 'status'},
                    {'name': 'Data directory', 'id': 'data_dir_id'}],
                data = [],
                hidden_columns = ['job_id', 'data_dir_id'],
                row_selectable='single',
                style_cell={'padding': '1rem'}),
        ],
    )
]


# training results
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
            dcc.Store(id='nothing', data=''),
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
                    dbc.Row(dbc.Col(training_results)),
                    dbc.Row(dbc.Col(job_status_display)),
                    dbc.Row(dbc.Col(meta)),
                ]
            ),

        ]
        )




