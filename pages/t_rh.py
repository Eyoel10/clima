import dash
from dash_extensions.enrich import Output, Input, State, dcc, html, callback

from pages.lib.global_scheme import dropdown_names
from pages.lib.template_graphs import (
    heatmap,
    yearly_profile,
    daily_profile
)
from pages.lib.utils import (
    generate_chart_name,
    generate_units,
    generate_units_degree,
    title_with_tooltip,
    summary_table_tmp_rh_tab,
    title_with_link,
    dropdown,
)

dash.register_page(__name__, name= 'Temperature and Humidity', order=2)

var_to_plot = ["Dry bulb temperature", "Relative humidity"]


def layout():
    return html.Div(
        className="container-col full-width",
        children=[
            html.Div(
                className="container-row full-width align-center justify-center",
                children=[
                    html.H4(
                        className="text-next-to-input", children=["Select a variable: "]
                    ),
                    dropdown(
                        id="dropdown",
                        className="dropdown-t-rh",
                        options={var: dropdown_names[var] for var in var_to_plot},
                        value=dropdown_names[var_to_plot[0]],
                    ),
                ],
            ),
            html.Div(
                className="container-col",
                children=[
                    html.Div(
                        children=title_with_link(
                            text="Yearly chart",
                            id_button="yearly-chart-label",
                            doc_link="https://cbe-berkeley.gitbook.io/clima/documentation/tabs-explained/temperature-and-humidity/temperatures-explained",
                        ),
                    ),
                    dcc.Loading(
                        type="circle",
                        children=html.Div(id="yearly-chart"),
                    ),
                    html.Div(
                        children=title_with_link(
                            text="Daily chart",
                            id_button="daily-chart-label",
                            doc_link="https://cbe-berkeley.gitbook.io/clima/documentation/tabs-explained/temperature-and-humidity/temperatures-explained",
                        ),
                    ),
                    dcc.Loading(
                        type="circle",
                        children=html.Div(id="daily"),
                    ),
                    html.Div(
                        children=title_with_link(
                            text="Heatmap chart",
                            id_button="heatmap-chart-label",
                            doc_link="https://cbe-berkeley.gitbook.io/clima/documentation/tabs-explained/temperature-and-humidity/temperatures-explained",
                        ),
                    ),
                    dcc.Loading(
                        type="circle",
                        children=html.Div(id="heatmap"),
                    ),
                    html.Div(
                        children=title_with_tooltip(
                            text="Descriptive statistics",
                            tooltip_text="count, mean, std, min, max, and percentiles",
                            id_button="table-tmp-rh",
                        ),
                    ),
                    html.Div(
                        id="table-tmp-hum",
                    ),
                ],
            ),
        ],
    )


@callback(
    Output("yearly-chart", "children"),
    [
        Input("df-store", "modified_timestamp"),
        Input("global-local-radio-input", "value"),
        Input("dropdown", "value"),
    ],
    [
        State("df-store", "data"),
        State("meta-store", "data"),
        State("si-ip-unit-store", "data"),
    ],
)
def update_yearly_chart(ts, global_local, dd_value, df, meta, si_ip):
    if dd_value == dropdown_names[var_to_plot[0]]:
        dbt_yearly = yearly_profile(df, "DBT", global_local, si_ip)
        dbt_yearly.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
        units = generate_units_degree(si_ip)
        return dcc.Graph(
            config=generate_chart_name("DryBulbTemperature_yearly", meta, units),
            figure=dbt_yearly,
        )
    else:
        rh_yearly = yearly_profile(df, "RH", global_local, si_ip)
        rh_yearly.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
        units = generate_units(si_ip)
        return dcc.Graph(
            config=generate_chart_name("RelativeHumidity_yearly", meta, units),
            figure=rh_yearly,
        )


@callback(
    Output("daily", "children"),
    [
        Input("df-store", "modified_timestamp"),
        Input("global-local-radio-input", "value"),
        Input("dropdown", "value"),
    ],
    [
        State("df-store", "data"),
        State("meta-store", "data"),
        State("si-ip-unit-store", "data"),
    ],
)
def update_daily(ts, global_local, dd_value, df, meta, si_ip):
    if dd_value == dropdown_names[var_to_plot[0]]:
        units = generate_units_degree(si_ip)
        return dcc.Graph(
            config=generate_chart_name("DryBulbTemperature_daily", meta, units),
            figure=daily_profile(
                df[["DBT", "hour", "UTC_time", "month_names", "day", "month"]],
                "DBT",
                global_local,
                si_ip,
            ),
        )
    else:
        units = generate_units(si_ip)
        return dcc.Graph(
            config=generate_chart_name("RelativeHumidity_daily", meta, units),
            figure=daily_profile(
                df[["RH", "hour", "UTC_time", "month_names", "day", "month"]],
                "RH",
                global_local,
                si_ip,
            ),
        )


@callback(
    [Output("heatmap", "children")],
    [
        Input("df-store", "modified_timestamp"),
        Input("global-local-radio-input", "value"),
        Input("dropdown", "value"),
    ],
    [
        State("df-store", "data"),
        State("meta-store", "data"),
        State("si-ip-unit-store", "data"),
    ],
)
def update_heatmap(ts, global_local, dd_value, df, meta, si_ip):
    """Update the contents of tab three. Passing in general info (df, meta)."""
    if dd_value == dropdown_names[var_to_plot[0]]:
        units = generate_units_degree(si_ip)
        return dcc.Graph(
            config=generate_chart_name("DryBulbTemperature_heatmap", meta, units),
            figure=heatmap(
                df[["DBT", "hour", "UTC_time", "month_names", "day"]],
                "DBT",
                global_local,
                si_ip,
            ),
        )
    else:
        units = generate_units(si_ip)
        return dcc.Graph(
            config=generate_chart_name("RelativeHumidity_heatmap", meta, units),
            figure=heatmap(
                df[["RH", "hour", "UTC_time", "month_names", "day"]],
                "RH",
                global_local,
                si_ip,
            ),
        )


@callback(
    Output("table-tmp-hum", "children"),
    [
        Input("df-store", "modified_timestamp"),
        Input("dropdown", "value"),
    ],
    [State("df-store", "data"), State("si-ip-unit-store", "data")],
)
def update_table(ts, dd_value, df, si_ip):
    """Update the contents of tab three. Passing in general info (df, meta)."""
    return summary_table_tmp_rh_tab(
        df[["month", "hour", dd_value, "month_names"]], dd_value, si_ip
    )