import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc


button_github = dbc.Button(
    "View Code on github",
    outline=True,
    color="primary",
    href="https://github.com/als-computing/dash-seg",
    id="gh-link",
    style={"text-transform": "none"},
)



def header():
    header= dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(
                                id="logo",
                                src='assets/mlex.png',
                                height="60px",
                            ),
                            md="auto",
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.H3("MLExchange | Image segmentation"),
                                        #html.P("Image segmentation"),
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
                                            #dbc.NavItem(button_github),
                                        ],
                                        navbar=True,
                                    ),
                                    id="navbar-collapse",
                                    navbar=True,
                                ),
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
    return header




