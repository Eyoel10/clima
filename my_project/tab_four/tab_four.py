import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
from my_project.global_scheme import config, dropdown_names

def custom():
    """ Return the layout for the custom sunpath and its dropdowns.
    """
    return html.Div(
        className = 'container-col full-width container-center',
        id = "tab-four-custom-sun-container",
        children = [
            html.Div(
                className = "container-row container-center full-width",
                children = [
                    html.H6(
                        className = "text-next-to-input",
                        children = ["Variable: "]
                    ),
                    dcc.Dropdown(
                        id = "custom-sun-var-dropdown", 
                        options = [
                            {'label': i, 'value': dropdown_names[i]} for i in dropdown_names
                        ], 
                        value = 'DBT'
                    ),
                ]
            ),
            html.Div(
                className = "container-row container-center full-width",
                children = [
                    html.H6(
                        className = "text-next-to-input",
                        children = ["View: "]
                    ),
                    dcc.Dropdown(
                        id = "custom-sun-view-dropdown", 
                        options = [
                            {'label': 'Polar', 'value': 'polar'},
                            {'label': 'Latitude/Longitude', 'value': 'lat/long'}
                        ], 
                        value = 'polar'
                    ),
                ]
            ),
            dcc.Graph(
                id = 'custom-sunpath',
                config = config
            )
        ]
    )


def tab_four():
    """ Contents of tab four.
    """
    return html.Div(
        className = "container-col tab-container full-width",
        id = "tab-four-container",
        children = [
            html.Div(
                className = "container-row full-width container-center",
                children = [
                    html.H6(
                        className = "text-next-to-input",
                        children = ["View: "]
                    ),
                    dcc.Dropdown(
                        id = "solar-dropdown", 
                        options = [
                            {'label': 'Polar', 'value': 'polar'},
                            {'label': 'Latitude/Longitude', 'value': 'lat/long'}
                        ], 
                        value = 'polar'
                    ),
                ]
            ),
            dcc.Graph(
                id = 'solar-dropdown-output',
                config = config
            ), 
            html.Div(
                className = 'container-row full-width',
                id = 'tab-four-subcontainer',
                children = [
                    dcc.Graph(
                        id = 'yearly-solar',
                        config = config
                    ),
                    dcc.Graph(
                        id = 'monthly-solar',
                        config = config
                    ), 
                ]
            ),
            dcc.Graph(
                id = 'cloud-cover',
                config = config
            ),
            custom(),
            dcc.Graph(
                id = 'daily-ghrad',
                config = config
            ), 
            dcc.Graph(
                id = 'heatmap-ghrad',
                config = config
            ), 
            dcc.Graph(
                id = 'daily-dnrad',
                config = config
            ), 
            dcc.Graph(
                id = 'heatmap-dnrad',
                config = config
            ),
            dcc.Graph(
                id = 'daily-difhrad',
                config = config
            ), 
            dcc.Graph(
                id = 'heatmap-difhrad',
                config = config
            )
        ]
    )
