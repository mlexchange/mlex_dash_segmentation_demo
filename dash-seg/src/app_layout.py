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
MODEL_DATABASE = {"Random Forest": "mlexchange/random-forest-dc",
                  "pyMSDtorch": "mlexchange/msdnetwork-notebook",
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

# job status display
job_status_display = [
    html.Div(
        children=[
            dbc.Button(
                "SHOW JOB LIST",
                id="job-button",
                outline=True,
            ),
            dbc.Modal(
                id = 'job-modal',
                size='lg',
                is_open=False,
                children=[
                    dash_table.DataTable(
                        id='jobs_table',
                        columns=[
                            {'name': 'Job ID', 'id': 'job_id'},
                            {'name': 'Type', 'id': 'job_type'},
                            {'name': 'Status', 'id': 'status'},
                            {'name': 'Dataset', 'id': 'dataset'},
                            {'name': 'Model', 'id': 'model_name'},
                            {'name': 'Parameters', 'id': 'parameters'},
                            {'name': 'Data directory', 'id': 'data_dir_id'},
                            {'name': 'Logs', 'id': 'job_logs'}
                        ],
                        data = [],
                        hidden_columns = ['job_id', 'data_dir_id', 'job_logs'],
                        row_selectable='single',
                        style_cell={'padding': '1rem', 'maxWidth': '7rem', 'whiteSpace': 'normal'},
                        fixed_rows={'headers': True},
                        css=[{"selector": ".show-hide", "rule": "display: none"}],
                        style_data_conditional=[
                            {'if': {'column_id': 'status', 'filter_query': '{status} = completed'},
                             'backgroundColor': 'green',
                             'color': 'white'},
                            {'if': {'column_id': 'status', 'filter_query': '{status} = failed'},
                             'backgroundColor': 'red',
                             'color': 'white'}
                        ]
                    )
                ]
            ),
            dcc.Textarea(id='job-display',
                         value='No job has been selected',
                         style={'width':'100%', 'height': '10rem'})
        ]
    )
]

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
                            {'label': 'Bead Experimental', 'value': 'data/bead_pack.tif'},
                            {'label': 'Bead Simulated', 'value': 'data/bead_pack_artifacts.tif'},
                            {'label': 'Castle Simulated', 'value': 'data/castle_artifacts.tif'},
                            {'label': 'Gambier Simulated', 'value': 'data/Gambier_artifacts.tif'},
                            {'label': 'LRC32 Simulated', 'value': 'data/lrc32_artifacts.tif'}
                        ],
                        value = 'data/bead_pack.tif',
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
                                    dbc.Col(dbc.FormGroup(
                                            [
                                                dbc.Label(
                                                    "Image Slider",
                                                    html_for='image-slider',
                                                    ),
                                                dcc.Slider(
                                                    id='image-slider',
                                                    min=0,
                                                    max=200,
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
    ),
    dbc.Card(
        id="msg-card",
        children=[
            dbc.CardHeader("Message"),
            dbc.CardBody(
                [
                    dcc.Textarea(id='msg-display',
                                 value='',
                                 style={'width':'100%', 'height': '3rem'})
                ]
            )
        ]),
    dbc.Collapse(
        id = 'show-plot',
        children = dbc.Card(
            id="plot-card",
            children=[
                dbc.CardHeader("Loss Plot"),
                dbc.CardBody(
                    [
                        dcc.Graph(id='loss-plot',
                                  style={'width':'100%', 'height': '20rem'})
                    ]
                )
            ]
        )
    ),
    # dbc.Card(
    #     id="logs-card",
    #     children=[
    #         dbc.CardHeader("Job Logs"),
    #         dbc.CardBody(
    #             [
    #                 dcc.Textarea(id='job-logs',
    #                              value='',
    #                              style={'width':'100%', 'height': '10rem'})
    #             ]
    #         )
    #     ])
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
                        style={'margin-bottom': '1rem', 'align-items': 'center', 'justify-content': 'center'}),
                    html.Div([
                        dbc.Label(
                            "Width of annotation paintbrush",
                            html_for="stroke-width"
                        ),
                        # Slider for specifying stroke width
                        dcc.Slider(
                            id="stroke-width",
                            min=0,
                            max=6,
                            step=0.1,
                            value=DEFAULT_STROKE_WIDTH,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ]),
                ]),
            dbc.CardHeader("Model"),
            dbc.CardBody(
                [
                    dbc.Form(
                        [
                            dbc.FormGroup(
                                [
                                    dbc.Label('Choose Segmentation Model', className='mr-2'),
                                    dcc.Dropdown(id='seg-dropdown',
                                                 options=[
                                                     {"label": entry, "value": entry} for entry in MODEL_DATABASE],
                                                 style={'min-width': '250px'},
                                                 value='Random Forest',
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
                            html.Div(
                                [
                                    dbc.Button(
                                        "TRAIN",
                                        id="train-seg",
                                        outline=True,
                                        size="lg",
                                        style={'width':'40%'}
                                    ),
                                    dbc.Button(
                                        "TEST",
                                        id="compute-seg",
                                        outline=True,
                                        size="lg",
                                        style={'width':'40%'}
                                    ),
                                ],
                                className='row',
                                style={'align-items': 'center', 'justify-content': 'center'}
                            ),

                            dcc.Checklist(
                                id="show-segmentation",
                                options=[
                                    {
                                        "label": "  Show segmentation",
                                        "value": "Show segmentation",
                                    }
                                ],
                                value=[],
                                style={'margin-top': '1rem', 'margin-bottom': '1rem'}
                            )
                            ]),
                    ],
            ),
            dbc.CardHeader("List of Jobs"),
            dbc.CardBody([
                dbc.Row(dbc.Col(job_status_display)),
            ])
        ],
    ),
]



# training results
training_results = [html.Div([
    dcc.Graph(
        id='training-results',
        ),
    dcc.Interval(
        id='update-training-loss',
        interval=1*2000, # milliseconds
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
                    dbc.Row(dbc.Col(meta)),
                ]
            )
        ]
        )



