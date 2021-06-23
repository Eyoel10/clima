import dash_core_components as dcc
import dash_html_components as html
from my_project.global_scheme import fig_config, month_lst, container_row_center_full
from dash.dependencies import Input, Output, State
from my_project.template_graphs import heatmap, wind_rose
import pandas as pd
from my_project.utils import title_with_tooltip

from app import app, cache, TIMEOUT


def sliders():
    """Returns 2 sliders for the hour"""
    return html.Div(
        className="container-col justify-center",
        id="slider-container",
        children=[
            html.Div(
                children=title_with_tooltip(
                    text="Customizable Wind Rose",
                    tooltip_text="Customizable Wind Rose",
                ),
            ),
            html.Div(
                className="container-row each-slider",
                children=[
                    html.P("Month Range"),
                    dcc.RangeSlider(
                        id="month-slider",
                        min=1,
                        max=12,
                        step=1,
                        value=[1, 12],
                        marks={1: "1", 12: "12"},
                        tooltip={"always_visible": False, "placement": "top"},
                        allowCross=True,
                    ),
                ],
            ),
            html.Div(
                className="container-row each-slider",
                children=[
                    html.P("Hour Range"),
                    dcc.RangeSlider(
                        id="hour-slider",
                        min=1,
                        max=24,
                        step=1,
                        value=[1, 24],
                        marks={1: "1", 24: "24"},
                        tooltip={"always_visible": False, "placement": "topLeft"},
                        allowCross=False,
                    ),
                ],
            ),
        ],
    )


def seasonal_wind_rose():
    """Return the section with the 4 seasonal wind rose graphs."""
    return html.Div(
        className="container-col",
        children=[
            html.Div(
                children=title_with_tooltip(
                    text="Seasonal Wind Rose",
                    tooltip_text="Seasonal Wind Rose",
                ),
            ),
            html.Div(
                className=container_row_center_full,
                children=[
                    html.Div(
                        className="container-col",
                        children=[
                            dcc.Loading(
                                type="circle",
                                children=[
                                    dcc.Graph(
                                        id="winter-wind-rose",
                                        config=fig_config,
                                    ),
                                ],
                            ),
                            html.P(
                                className="seasonal-text", id="winter-wind-rose-text"
                            ),
                        ],
                    ),
                    html.Div(
                        className="container-col",
                        children=[
                            dcc.Loading(
                                type="circle",
                                children=[
                                    dcc.Graph(
                                        id="spring-wind-rose",
                                        config=fig_config,
                                    ),
                                ],
                            ),
                            html.P(
                                className="seasonal-text", id="spring-wind-rose-text"
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className=container_row_center_full,
                children=[
                    html.Div(
                        className="container-col",
                        children=[
                            dcc.Loading(
                                type="circle",
                                children=[
                                    dcc.Graph(
                                        id="summer-wind-rose",
                                        config=fig_config,
                                    ),
                                ],
                            ),
                            html.P(
                                className="seasonal-text", id="summer-wind-rose-text"
                            ),
                        ],
                    ),
                    html.Div(
                        className="container-col",
                        children=[
                            dcc.Loading(
                                type="circle",
                                children=[
                                    dcc.Graph(
                                        id="fall-wind-rose",
                                        config=fig_config,
                                    ),
                                ],
                            ),
                            html.P(className="seasonal-text", id="fall-wind-rose-text"),
                        ],
                    ),
                ],
            ),
        ],
    )


def daily_wind_rose():
    """Return the section for the 3 daily wind rose graphs."""
    return html.Div(
        className="container-col full-width",
        id="tab5-daily-container",
        children=[
            html.Div(
                children=title_with_tooltip(
                    text="Daily Wind Rose",
                    tooltip_text="Daily Wind Rose",
                ),
            ),
            html.Div(
                id="daily-wind-rose-outer-container",
                className="container-row full-width",
                children=[
                    html.Div(
                        className="container-col",
                        children=[
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=[
                                        dcc.Graph(
                                            className="daily-wind-graph",
                                            id="morning-wind-rose",
                                            config=fig_config,
                                        ),
                                    ],
                                ),
                            ),
                            html.P(className="daily-text", id="morning-windrose-text"),
                        ],
                    ),
                    html.Div(
                        className="container-col",
                        children=[
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=[
                                        dcc.Graph(
                                            className="daily-wind-graph",
                                            id="noon-wind-rose",
                                            config=fig_config,
                                        ),
                                    ],
                                ),
                            ),
                            html.P(className="daily-text", id="noon-windrose-text"),
                        ],
                    ),
                    html.Div(
                        className="container-col",
                        children=[
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=[
                                        dcc.Graph(
                                            className="daily-wind-graph",
                                            id="night-wind-rose",
                                            config=fig_config,
                                        ),
                                    ],
                                ),
                            ),
                            html.P(className="daily-text", id="night-windrose-text"),
                        ],
                    ),
                ],
            ),
        ],
    )


def custom_wind_rose():
    """"""
    return html.Div(
        className="container-col justify-center full-width",
        id="custom-windrose-container",
        children=[
            html.Div(
                children=title_with_tooltip(
                    text="Customizable Wind Rose",
                    tooltip_text="Customizable Wind Rose",
                ),
            ),
            html.Div(
                className="container-row full-width justify-center",
                id="tab5-custom-dropdown-container",
                children=[
                    html.Div(
                        className="container-col justify-center tab5-custom-half-container",
                        children=[
                            html.Div(
                                className=container_row_center_full,
                                children=[
                                    html.H6(
                                        className="text-next-to-input",
                                        children=["Start Month:"],
                                    ),
                                    dcc.Dropdown(
                                        id="tab5-custom-start-month",
                                        options=[
                                            {"label": j, "value": i + 1}
                                            for i, j in enumerate(month_lst)
                                        ],
                                        value="1",
                                    ),
                                ],
                            ),
                            html.Div(
                                className=container_row_center_full,
                                children=[
                                    html.H6(
                                        className="text-next-to-input",
                                        children=["Start Hour:"],
                                    ),
                                    dcc.Dropdown(
                                        id="tab5-custom-start-hour",
                                        options=[
                                            {"label": str(i) + ":00", "value": i}
                                            for i in range(1, 25)
                                        ],
                                        value="1",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="container-col tab5-custom-half-container justify-center",
                        children=[
                            html.Div(
                                className=container_row_center_full,
                                children=[
                                    html.H6(
                                        className="text-next-to-input",
                                        children=["End Month:"],
                                    ),
                                    dcc.Dropdown(
                                        id="tab5-custom-end-month",
                                        options=[
                                            {"label": j, "value": i + 1}
                                            for i, j in enumerate(month_lst)
                                        ],
                                        value="12",
                                    ),
                                ],
                            ),
                            html.Div(
                                className=container_row_center_full,
                                children=[
                                    html.H6(
                                        className="text-next-to-input",
                                        children=["End Hour:"],
                                    ),
                                    dcc.Dropdown(
                                        id="tab5-custom-end-hour",
                                        options=[
                                            {"label": str(i) + ":00", "value": i}
                                            for i in range(1, 25)
                                        ],
                                        value="24",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Loading(
                type="circle",
                children=[
                    dcc.Graph(id="custom-wind-rose", config=fig_config),
                ],
            ),
        ],
    )


def layout_wind():
    """Contents in the fifth tab 'Wind'."""
    return html.Div(
        className="container-col justify-center",
        children=[
            html.Div(
                children=title_with_tooltip(
                    text="Annual Wind Rose",
                    tooltip_text="Annual Wind Rose",
                ),
            ),
            dcc.Loading(
                type="circle",
                children=[
                    dcc.Graph(id="wind-rose", config=fig_config),
                ],
            ),
            dcc.Loading(
                type="circle",
                children=[
                    dcc.Graph(id="wind-speed", config=fig_config),
                ],
            ),
            dcc.Loading(
                type="circle",
                children=[
                    dcc.Graph(id="wind-direction", config=fig_config),
                ],
            ),
            seasonal_wind_rose(),
            daily_wind_rose(),
            custom_wind_rose(),
        ],
    )


# wind rose
@app.callback(
    Output("wind-rose", "figure"),
    # General
    [Input("df-store", "modified_timestamp")],
    [Input("global-local-radio-input", "value")],
    [State("df-store", "data")],
    [State("meta-store", "data")],
)
@cache.memoize(timeout=TIMEOUT)
def update_tab_wind_rose(ts, global_local, df, meta):
    """Update the contents of tab five. Passing in the info from the sliders and the general info (df, meta)."""
    df = pd.read_json(df, orient="split")
    annual = wind_rose(df, meta, "", [1, 12], [1, 24], True)
    return annual


# wind speed
@app.callback(
    Output("wind-speed", "figure"),
    # General
    [Input("df-store", "modified_timestamp")],
    [Input("global-local-radio-input", "value")],
    [State("df-store", "data")],
    [State("meta-store", "data")],
)
@cache.memoize(timeout=TIMEOUT)
def update_tab_wind_speed(ts, global_local, df, meta):
    """Update the contents of tab five. Passing in the info from the sliders and the general info (df, meta)."""
    df = pd.read_json(df, orient="split")

    speed = heatmap(df, "Wspeed", global_local)

    return speed


# wind direction
@app.callback(
    Output("wind-direction", "figure"),
    # General
    [Input("df-store", "modified_timestamp")],
    [Input("global-local-radio-input", "value")],
    [State("df-store", "data")],
    [State("meta-store", "data")],
)
@cache.memoize(timeout=TIMEOUT)
def update_tab_wind_direction(ts, global_local, df, meta):
    """Update the contents of tab five. Passing in the info from the sliders and the general info (df, meta)."""
    df = pd.read_json(df, orient="split")
    direction = heatmap(df, "Wdir", global_local)
    return direction


# Custom Wind rose
@app.callback(
    Output("custom-wind-rose", "figure"),
    # Custom Graph Input
    [Input("tab5-custom-start-month", "value")],
    [Input("tab5-custom-start-hour", "value")],
    [Input("tab5-custom-end-month", "value")],
    [Input("tab5-custom-end-hour", "value")],
    # General
    [Input("df-store", "modified_timestamp")],
    [State("df-store", "data")],
    [State("meta-store", "data")],
)
@cache.memoize(timeout=TIMEOUT)
def update_tab_five(start_month, start_hour, end_month, end_hour, ts, df, meta):
    """Update the contents of tab five. Passing in the info from the sliders and the general info (df, meta)."""
    df = pd.read_json(df, orient="split")
    start_hour = int(start_hour)
    end_hour = int(end_hour)
    start_month = int(start_month)
    end_month = int(end_month)

    # Wind Rose Graphs
    if start_month <= end_month:
        df = df.loc[(df["month"] >= start_month) & (df["month"] <= end_month)]
    else:
        df = df.loc[(df["month"] <= end_month) | (df["month"] >= start_month)]
    if start_hour <= end_hour:
        df = df.loc[(df["hour"] >= start_hour) & (df["hour"] <= end_hour)]
    else:
        df = df.loc[(df["hour"] <= end_hour) | (df["hour"] >= start_hour)]
    custom = wind_rose(
        df, meta, "", [start_month, end_month], [start_hour, end_hour], True
    )

    return custom


### Seasonal Graphs ###
@app.callback(
    # Graphs
    Output("winter-wind-rose", "figure"),
    Output("spring-wind-rose", "figure"),
    Output("summer-wind-rose", "figure"),
    Output("fall-wind-rose", "figure"),
    # Text
    Output("winter-wind-rose-text", "children"),
    Output("spring-wind-rose-text", "children"),
    Output("summer-wind-rose-text", "children"),
    Output("fall-wind-rose-text", "children"),
    # General
    [Input("df-store", "modified_timestamp")],
    [Input("global-local-radio-input", "value")],
    [State("df-store", "data")],
    [State("meta-store", "data")],
)
@cache.memoize(timeout=TIMEOUT)
def update_tab_five_seasonal_graphs(ts, global_local, df, meta):
    """Update the contents of tab five. Passing in the info from the sliders and the general info (df, meta)."""
    df = pd.read_json(df, orient="split")

    hours = [1, 24]
    winter_months = [12, 2]
    spring_months = [3, 5]
    summer_months = [6, 8]
    fall_months = [9, 12]

    # Wind Rose Graphs
    winter = wind_rose(df, meta, "", winter_months, hours, False)
    spring = wind_rose(df, meta, "", spring_months, hours, True)
    summer = wind_rose(df, meta, "", summer_months, hours, False)
    fall = wind_rose(df, meta, "", fall_months, hours, False)

    # Text
    winter_df = df.loc[
        (df["month"] <= winter_months[1]) | (df["month"] >= winter_months[0])
    ]
    winter_total_count = winter_df.shape[0]
    winter_calm_count = winter_df.query("Wspeed == 0").shape[0]

    spring_df = df.loc[
        (df["month"] >= spring_months[0]) & (df["month"] <= spring_months[1])
    ]
    spring_total_count = spring_df.shape[0]
    spring_calm_count = spring_df.query("Wspeed == 0").shape[0]

    summer_df = df.loc[
        (df["month"] >= summer_months[0]) & (df["month"] <= summer_months[1])
    ]
    summer_total_count = summer_df.shape[0]
    summer_calm_count = summer_df.query("Wspeed == 0").shape[0]

    fall_df = df.loc[(df["month"] >= fall_months[0]) & (df["month"] <= fall_months[1])]
    fall_total_count = fall_df.shape[0]
    fall_calm_count = fall_df.query("Wspeed == 0").shape[0]

    winter_text = (
        "Observations between the months of "
        + month_lst[winter_months[0] - 1]
        + " and "
        + month_lst[winter_months[1] - 1]
        + " between "
        + str(hours[0])
        + ":00 hours and "
        + str(hours[1])
        + ":00 hours. Selected observations "
        + str(winter_total_count)
        + " of 8760, or "
        + str(int(100 * (winter_total_count / 8760)))
        + "%. "
        + str(winter_calm_count)
        + " observations have calm winds."
    )

    spring_text = (
        "Observations between the months of "
        + month_lst[spring_months[0] - 1]
        + " and "
        + month_lst[spring_months[1] - 1]
        + " between "
        + str(hours[0])
        + ":00 hours and "
        + str(hours[1])
        + ":00 hours. Selected observations "
        + str(spring_total_count)
        + " of 8760, or "
        + str(int(100 * (spring_total_count / 8760)))
        + "%. "
        + str(spring_calm_count)
        + " observations have calm winds."
    )

    summer_text = (
        "Observations between the months of "
        + month_lst[summer_months[0] - 1]
        + " and "
        + month_lst[summer_months[1] - 1]
        + " between "
        + str(hours[0])
        + ":00 hours and "
        + str(hours[1])
        + ":00 hours. Selected observations "
        + str(summer_total_count)
        + " of 8760, or "
        + str(int(100 * (summer_total_count / 8760)))
        + "%. "
        + str(summer_calm_count)
        + " observations have calm winds."
    )

    fall_text = (
        "Observations between the months of "
        + month_lst[fall_months[0] - 1]
        + " and "
        + month_lst[fall_months[1] - 1]
        + " between "
        + str(hours[0])
        + ":00 hours and "
        + str(hours[1])
        + ":00 hours. Selected observations "
        + str(fall_total_count)
        + " of 8760, or "
        + str(int(100 * (fall_total_count / 8760)))
        + "%. "
        + str(fall_calm_count)
        + " observations have calm winds."
    )

    return (
        winter,
        spring,
        summer,
        fall,
        winter_text,
        spring_text,
        summer_text,
        fall_text,
    )


### Daily Graphs ###
@app.callback(
    # Daily Graphs
    Output("morning-wind-rose", "figure"),
    Output("noon-wind-rose", "figure"),
    Output("night-wind-rose", "figure"),
    # Text
    Output("morning-windrose-text", "children"),
    Output("noon-windrose-text", "children"),
    Output("night-windrose-text", "children"),
    # General
    [Input("df-store", "modified_timestamp")],
    [Input("global-local-radio-input", "value")],
    [State("df-store", "data")],
    [State("meta-store", "data")],
)
@cache.memoize(timeout=TIMEOUT)
def update_tab_five_daily_graphs(ts, global_local, df, meta):
    """Update the contents of tab five. Passing in the info from the sliders and the general info (df, meta)."""
    df = pd.read_json(df, orient="split")

    months = [1, 12]
    morning_times = [6, 13]
    noon_times = [14, 21]
    night_times = [22, 5]

    # Wind Rose Graphs
    morning = wind_rose(df, meta, "", months, morning_times, False)
    noon = wind_rose(df, meta, "", months, noon_times, False)
    night = wind_rose(df, meta, "", months, night_times, True)

    # Text
    morning_df = df.loc[
        (df["hour"] >= morning_times[0]) & (df["hour"] <= morning_times[1])
    ]
    morning_total_count = morning_df.shape[0]
    morning_calm_count = morning_df.query("Wspeed == 0").shape[0]

    noon_df = df.loc[
        (df["hour"] >= morning_times[0]) & (df["hour"] <= morning_times[1])
    ]
    noon_total_count = noon_df.shape[0]
    noon_calm_count = noon_df.query("Wspeed == 0").shape[0]

    night_df = df.loc[(df["hour"] <= night_times[1]) | (df["hour"] >= night_times[0])]
    night_total_count = night_df.shape[0]
    night_calm_count = night_df.query("Wspeed == 0").shape[0]

    morning_text = (
        "Observations between the months of "
        + month_lst[months[0] - 1]
        + " and "
        + month_lst[months[1] - 1]
        + " between "
        + str(morning_times[0])
        + ":00 hours and "
        + str(morning_times[1])
        + ":00 hours. Selected observations "
        + str(morning_total_count)
        + " of 8760, or "
        + str(int(100 * (morning_total_count / 8760)))
        + "% "
        + str(morning_calm_count)
        + " observations have calm winds."
    )
    noon_text = (
        "Observations between the months of "
        + month_lst[months[0] - 1]
        + " and "
        + month_lst[months[1] - 1]
        + " between "
        + str(noon_times[0])
        + ":00 hours and "
        + str(noon_times[1])
        + ":00 hours. Selected observations "
        + str(noon_total_count)
        + " of 8760, or "
        + str(int(100 * (noon_total_count / 8760)))
        + "% "
        + str(noon_calm_count)
        + " observations have calm winds."
    )
    night_text = (
        "Observations between the months of "
        + month_lst[months[0] - 1]
        + " and "
        + month_lst[months[1] - 1]
        + " between "
        + str(night_times[0])
        + ":00 hours and "
        + str(night_times[1])
        + ":00 hours. Selected observations "
        + str(night_total_count)
        + " of 8760, or "
        + str(int(100 * (night_total_count / 8760)))
        + "% "
        + str(night_calm_count)
        + " observations have calm winds."
    )

    return morning, noon, night, morning_text, noon_text, night_text
