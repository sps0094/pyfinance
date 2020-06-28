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
    {'label': 'Rhos', 'value': 1},
    {'label': 'Volatility', 'value': 2},
    {'label': 'Cov_Mat', 'value': 3},
    {'label': 'Views', 'value': 4},
    {'label': 'Pick_Mat', 'value': 5},
]
app.layout = html.Div(
    [html.Div([html.Div([dcc.Checklist(id='local_file', options=[{'label': 'Y', 'value': 1}], value=[],
                                       labelStyle={'display': 'inline-block'})]),
               html.Label(children='Enter list of asset names: '),
               dcc.Input(id='asset_list', value='', type='text')], style={'display': 'inline-block'}),
     html.Div([html.Label(children='Enter no of views: '),
               dcc.Input(id='no_views', value='', type='number')], style={'display': 'inline-block'}),
     html.Div([dcc.Checklist(id='display_table', options=checkoptions, value=[],
                             labelStyle={'display': 'inline-block'})]),
     html.Div(id='wts_table_container', children=[dt.DataTable(id='wts_table', editable=True)],
              style={'display': 'block'}),
     html.Div(id='rhos_table_container', children=[dt.DataTable(id='rhos_table', editable=True)],
              style={'display': 'block'}),
     html.Div(id='vol_table_container', children=[dt.DataTable(id='vol_table', editable=True)],
              style={'display': 'block'}),
     html.Div(id='cov_table_container', children=[dt.DataTable(id='cov_table', editable=True)],
              style={'display': 'block'}),
     html.Div(id='view_table_container', children=[dt.DataTable(id='view_table', editable=True)],
              style={'display': 'block'}),
     html.Div(id='pick_table_container', children=[dt.DataTable(id='pick_table', editable=True)],
              style={'display': 'block'}),
     html.Div(id='dump', style={'display': 'none'}),
     html.Div(id='dump_vw_df', style={'display': 'none'}),
     html.Div(id='dump_wt_df', style={'display': 'none'}),
     html.Div(id='dump_no', style={'display': 'none'}),
     html.Button(id='submit', children='SUBMIT', n_clicks=0, style={'display': 'inline-block'}),
     html.Button(id='update', children='UPDATE', n_clicks=0, style={'display': 'none'})])


@app.callback([Output('asset_list', 'value'),
               Output('dump_no', 'children'),
               Output('dump_vw_df', 'children'),
               Output('dump_wt_df', 'children'),
               Output('display_table', 'options')],
              [Input('local_file', 'value')])
def upd_asset_list(localfile):
    if localfile:
        ind_vw_2014 = erk.get_df('data/ind49_m_vw_rets.csv', to_per=True, start_period='2014')
        ind_mkt_wts_2014 = erk.plot_corr_mktret(ind_ret_filename='data/ind49_m_vw_rets.csv',
                                                n_firms_filename='data/ind49_m_nfirms.csv',
                                                size_filename='data/ind49_m_size.csv',
                                                start_period='2014',
                                                end_period=None,
                                                to_per=True,
                                                retrieve_mkt_cap_wts=True,
                                                format='%Y%m')
        asset_list = [' '.join(list(ind_vw_2014.columns))]
        json_vw = ind_vw_2014.to_json(date_format='iso', orient='table')
        json_wts = ind_mkt_wts_2014.to_json(date_format='iso', orient='table')
        checkoptions = [
            {'label': 'Views', 'value': 4},
            {'label': 'Pick_Mat', 'value': 5},
        ]
        return asset_list, len(ind_vw_2014.columns), json_vw, json_wts, checkoptions


@app.callback([Output('no_views', 'max')],
              [Input('dump_no', 'children')])
def max_views(max):
    return [int(max)]


@app.callback([Output('wts_table_container', 'style'),
               Output('rhos_table_container', 'style'),
               Output('vol_table_container', 'style'),
               Output('cov_table_container', 'style'),
               Output('view_table_container', 'style'),
               Output('pick_table_container', 'style'),
               Output('wts_table', 'columns'),
               Output('wts_table', 'data'),
               Output('rhos_table', 'columns'),
               Output('rhos_table', 'data'),
               Output('vol_table', 'columns'),
               Output('vol_table', 'data'),
               Output('cov_table', 'columns'),
               Output('cov_table', 'data'),
               Output('view_table', 'columns'),
               Output('view_table', 'data'),
               Output('pick_table', 'columns'),
               Output('pick_table', 'data'),
               Output('update', 'style')],
              [Input('submit', 'n_clicks')],
              [State('display_table', 'value'),
               State('asset_list', 'value'),
               State('no_views', 'value')])
def upd_visibility(n_clicks, value, asset_list, no_views):
    if isinstance(asset_list, list):
        asset_list = asset_list[0]
    asset_list = list(asset_list.split(' '))
    disp_wts = 'block' if 0 in value else 'none'
    disp_rhos = 'block' if 1 in value else 'none'
    disp_vol = 'block' if 2 in value else 'none'
    disp_cov = 'block' if 3 in value else 'none'
    disp_view = 'block' if 4 in value else 'none'
    disp_pick = 'block' if 5 in value else 'none'
    col_wts = [{'id': 'asset_name_wts', 'name': 'Assets'}] + [{'id': 'wts', 'name': 'Prior_weights'}]
    data_wts = [dict(asset_name_wts=asset) for asset in asset_list]
    col_vol = [{'id': 'asset_name_wts', 'name': 'Assets'}] + [{'id': 'vol', 'name': 'Prior_volatility'}]
    data_vol = [dict(asset_name_wts=asset) for asset in asset_list]
    col_rhos = [{'id': 'rhos', 'name': 'Rhos'}] + [{'id': asset, 'name': asset} for asset in asset_list]
    data_rhos = [dict(rhos=asset) for asset in asset_list]
    col_views = [{'id': 'views_no', 'name': 'Assets'}] + [{'id': 'views', 'name': 'views'}]
    data_views = [dict(views_no=k) for k in range(1, no_views + 1)]
    col_cov = [{'id': 'cov', 'name': 'cov_mat'}] + [{'id': asset, 'name': asset} for asset in asset_list]
    data_cov = [dict(cov=asset) for asset in asset_list]
    col_pick = [{'id': 'pick', 'name': 'pick_mat'}] + [{'id': asset, 'name': asset} for asset in asset_list]
    data_pick = [dict(pick=k) for k in range(1, no_views + 1)]
    return {'display': disp_wts}, {'display': disp_rhos}, {'display': disp_vol}, {'display': disp_cov}, {
        'display': disp_view}, {'display': disp_pick}, \
           col_wts, data_wts, col_rhos, data_rhos, col_vol, data_vol, col_cov, data_cov, col_views, data_views, col_pick, data_pick, {
               'display': 'inline-block'}


@app.callback([Output('dump', 'children')],
              [Input('update', 'n_clicks')],
              [State('wts_table', 'data'),
               State('cov_table', 'data'),
               State('rhos_table', 'data'),
               State('vol_table', 'data'),
               State('view_table', 'data'),
               State('pick_table', 'data'),
               State('asset_list', 'value'),
               State('display_table', 'value'),
               State('dump_vw_df', 'children'),
               State('dump_wt_df', 'children')])
def update_values(n_clicks, wts, cov, rhos, vol, views, pick_mat, asset_list, value, ind_vw_2014, ind_mkt_wts_2014):
    if isinstance(asset_list, list):
        asset_list = asset_list[0]
    asset_list = list(asset_list.split(' '))
    if ind_vw_2014 is not None and ind_mkt_wts_2014 is not None:
        ind_vw_2014 = pd.read_json(ind_vw_2014, orient='table').to_period('M')
        ind_mkt_wts_2014 = pd.read_json(ind_mkt_wts_2014, orient='table').to_period('M').iloc[0].rename('wts_prior')
        wts_prior = ind_mkt_wts_2014
        sigma_prior = erk.get_cov(ind_vw_2014)
    else:
        if 0 in value:
            wts_prior = pd.DataFrame(wts).set_index('asset_name_wts').squeeze().rename('wts_prior').fillna(0)
            wts_prior = wts_prior.astype('float')
        if 1 in value:
            rhos_prior = pd.DataFrame(rhos, columns=[asset for asset in asset_list]).astype('float')
        else:
            rhos_prior = 0
        if 2 in value:
            vol_prior = pd.DataFrame(vol)['vol']
            vol_prior = vol_prior.astype('float')
        else:
            vol_prior = 0
        if 3 in value:
            sigma_prior = pd.DataFrame(cov, columns=[asset for asset in asset_list]).astype('float')
        else:
            sigma_prior = rhos_prior.mul(vol_prior, axis=0)
    if 4 in value:
        views = pd.DataFrame(views)['views']
        views = views.astype('float')
    if 5 in value:
        pick_df = pd.DataFrame(pick_mat, columns=[asset for asset in asset_list], index=views.index).astype('float')
    return ' '


app.run_server()
