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
            dash_table.DataTable(
                id='jobs_table',
                columns=[
                    {'name': 'Job ID', 'id': 'job_id'},
                    {'name': 'Type', 'id': 'job_type'},
                    {'name': 'Status', 'id': 'status'},
                    {'name': 'Dataset', 'id': 'dataset'},
                    {'name': 'Image length', 'id': 'image_length'},
                    {'name': 'Model', 'id': 'model_name'},
                    {'name': 'Parameters', 'id': 'parameters'},
                    {'name': 'Experiment ID', 'id': 'experiment_id'},
                    {'name': 'Logs', 'id': 'job_logs'}
                ],
                data = [],
                hidden_columns = ['job_id', 'image_length', 'experiment_id', 'job_logs'],
                row_selectable='single',
                style_cell={'padding': '1rem'}, #, 'maxWidth': '7rem', 'whiteSpace': 'normal'},
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
    )
]


error_message = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalBody(id="error-body"),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", id="close-error", className="ms-auto", n_clicks=0
                    )
                ),
            ],
            id="error-msg",
            centered=True,
            is_open=False,
        ),
    ]
)


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
    ),
    error_message,
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
    dbc.Card(
        id="logs-card",
        children=[
            dbc.CardHeader("Job Logs"),
            dbc.CardBody(
                [
                    dcc.Textarea(id='job-logs',
                                 value='',
                                 style={'width':'100%', 'height': '10rem'})
                ]
            )
        ])
]


label_panel = html.Div(
    [dbc.Card(
        id="brush-card",
        style={"width": "100%"},
        children=[
            dbc.Collapse(children=[
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
                                    style={'margin-left': '0.1rem', 'margin-right': '0.1rem', "background-color": helper_utils.class_to_color(c)},
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
                            )
                        ], style={'align-items': 'center', 'justify-content': 'center', 'margin-bottom': '0rem'}),
                    ])],
                id="brush-collapse",
                is_open=True,
                style = {'margin-bottom': '-2rem'}
            ),
            dbc.Collapse(
                children = [dbc.CardHeader("Instructions"),
                dbc.CardBody(
                    dbc.Label('Please mark the image slice(s) for the selected unsupervised model. \
                    Otherwise, the whole stack will be used.', className='mr-2'))],
                id="instructions-collapse",
                is_open=False,
                style = {'margin-bottom': '-1rem'}
            ),
            dbc.Col(dbc.Button('Delete Annotations',
                       id='del-mask',
                       n_clicks=0,
                       outline=True,
                       className='m-1',
                       style={'width': '95%'}),
                    style = {'margin-top': '0rem', 'margin-bottom': '1rem', 'align-items': 'center', 'justify-content': 'center'})
        ]),
    ]
)

# panel = dbc.Card(
#             style={"width": "100%"},
#             children=[
#                 dbc.CardHeader("Instructions"),
#                 dbc.CardBody(dbc.Label('Please mark the image slice(s) for the selected unsupervised model. Otherwise, the whole stack will be used.', className='mr-2'))
#             ])

# sidebar - labeling tools
sidebar_label = [
    html.Div(id='top-right-panel'),
    label_panel,
    dbc.Card(
        id="sidebar-card",
        style={"width": "100%"},
        children=[
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
                            html.Div(id='model-source'),
                            html.Div(id='additional-seg-params'),
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
                                                 value='local',
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
                                        className='m-1',
                                        style={'width':'40%'}
                                    ),
                                    dbc.Button(
                                        "TEST",
                                        id="compute-seg",
                                        outline=True,
                                        size="lg",
                                        className='m-1',
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
            html.Div(id='progress-bar',
                style={'margin-top': '0.7rem', 'margin-left': '1rem', 'margin-right': '1rem'}
            ),
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
            dcc.Store(id='train_counter', data=0),
            dcc.Store(id='seg_counter', data=0),
            dcc.Store(id='image-length', data=0),
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



