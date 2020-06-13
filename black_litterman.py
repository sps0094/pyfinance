import edhec_risk_kit as erk
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin
import plotly.graph_objs as go
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)
from jupyter_dash import JupyterDash
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import dash_table as dt

app = JupyterDash()
checkoptions = [
    {'label': 'Weights', 'value': 0},
    {'label': 'Cov_Mat', 'value': 1},
    {'label': 'Views', 'value': 2},
    {'label': 'Pick_Mat', 'value': 3},
]
app.layout = html.Div([html.Div([dcc.Checklist(id='display_table',
                                               options=checkoptions,
                                               value=[],
                                               labelStyle={'display': 'inline-block'}
                                               )]),
                       html.Div(id='wts_table_container', children=[dt.DataTable(id='wts_table'), 'Hi'], style={'display': 'block'}),
                       html.Div(id='cov_table_container', children=[dt.DataTable(id='cov_table'), 'Hi'], style={'display': 'block'}),
                       html.Div(id='view_table_container', children=[dt.DataTable(id='view_table'), 'Hi'], style={'display': 'block'}),
                       html.Div(id='pick_table_container', children=[dt.DataTable(id='pick_table'), 'Hi'], style={'display': 'block'}),
                       ])
@app.callback([Output('wts_table_container', 'style'),
               Output('cov_table_container', 'style'),
               Output('view_table_container', 'style'),
               Output('pick_table_container', 'style')],
              [Input('display_table', 'value')])
def upd_visibility(value):
    disp_wts = 'block' if 0 in value else 'none'
    disp_cov = 'block' if 1 in value else 'none'
    disp_view = 'block' if 2 in value else 'none'
    disp_pick = 'block' if 3 in value else 'none'
    return {'display':disp_wts}, {'display':disp_cov}, {'display':disp_view}, {'display':disp_pick}


app.run_server()