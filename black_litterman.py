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


def get_bl_results(wts_prior: pd.Series, sigma_prior: pd.DataFrame, delta, tau, p: pd.DataFrame=None, q: pd.Series=None, omega=None):
    pi = erk.rev_opt_implied_returns(delta, sigma_prior, wts_prior).to_numpy()
    bl_mu, bl_sigma = erk.black_litterman(wts_prior, sigma_prior, p, q, omega, delta, tau)
    wts_msr = erk.w_msr_closed_form(bl_sigma, bl_mu, scale=True)
    return pi, bl_mu, bl_sigma, wts_msr


app.layout = html.Div(
    [html.Div([html.Div([dcc.Checklist(id='local_file', options=[{'label': 'Y', 'value': 1}], value=[],
                                       labelStyle={'display': 'inline-block'})]),
               html.Label(children='Enter list of asset names: '),
               dcc.Input(id='asset_list', value='', type='text')], style={'display': 'inline-block'}),
     html.Div([html.Label(children='Enter no of views: '),
               dcc.Input(id='no_views', value='', type='number'),
               dcc.Input(id='tau', value=0.02, type='number'),
               dcc.Input(id='delta', value=2.5, type='number')], style={'display': 'inline-block'}),
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
     html.Div(id='dump_vw_df', style={'display': 'none'}),
     html.Div(id='dump_wt_df', style={'display': 'none'}),
     html.Div(id='dump_no', style={'display': 'none'}),
     html.Button(id='submit', children='SUBMIT', n_clicks=0, style={'display': 'inline-block'}),
     html.Button(id='update', children='UPDATE', n_clicks=0, style={'display': 'none'}),
     html.Div(dt.DataTable(id='bl_results_table'), style={'display': 'flex', 'justify-content': 'space-evenly', 'padding-top': '25px'}),
     html.Div(html.Pre(id='disp_bl_cov', style={'display': 'block'}), style={'display': 'flex', 'justify-content': 'space-evenly', 'padding-top': '25px'})])


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
    col_views = [{'id': 'views_no', 'name': 'Views'}] + [{'id': 'views', 'name': 'views'}]
    data_views = [dict(views_no=k) for k in range(1, no_views + 1)]
    col_cov = [{'id': 'cov', 'name': 'cov_mat'}] + [{'id': asset, 'name': asset} for asset in asset_list]
    data_cov = [dict(cov=asset) for asset in asset_list]
    col_pick = [{'id': 'pick', 'name': 'pick_mat'}] + [{'id': asset, 'name': asset} for asset in asset_list]
    data_pick = [dict(pick=k) for k in range(1, no_views + 1)]
    return {'display': disp_wts}, {'display': disp_rhos}, {'display': disp_vol}, {'display': disp_cov}, {
        'display': disp_view}, {'display': disp_pick}, \
           col_wts, data_wts, col_rhos, data_rhos, col_vol, data_vol, col_cov, data_cov, col_views, data_views, col_pick, data_pick, {
               'display': 'inline-block'}


@app.callback([Output('disp_bl_cov', 'children'),
               Output('bl_results_table', 'columns'),
               Output('bl_results_table', 'data'),],
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
               State('dump_wt_df', 'children'),
               State('tau', 'value'),
               State('delta', 'value')])
def update_values(n_clicks, wts, cov, rhos, vol, views, pick_mat, asset_list, value, ind_vw_2014, ind_mkt_wts_2014, tau, delta):
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
            vol_prior = vol_prior.astype('float').fillna(0)
        else:
            vol_prior = 0
        if 3 in value:
            sigma_prior = pd.DataFrame(cov, columns=[asset for asset in asset_list]).astype('float').fillna(0)
        else:
            sigma_prior = rhos_prior.mul(vol_prior, axis=0)
    if 4 in value:
        views = pd.DataFrame(views)['views']
        views = views.astype('float').fillna(0)
    if 5 in value:
        pick_df = pd.DataFrame(pick_mat, columns=[asset for asset in asset_list], index=views.index).astype('float').fillna(0)
    pi, bl_mu, bl_sigma, wts_msr = get_bl_results(wts_prior, sigma_prior, delta, tau, pick_df, views)
    bl_results_df = pd.DataFrame({'Asset': asset_list,
                                  'Cur_wts': wts_prior,
                                  'pi': pi,
                                  'bl_mu': bl_mu,
                                  'opt_wts': wts_msr})
    disp_bl_cov = 'Posterior cov: {}'.format(bl_sigma)
    bl_results_columns = [{'name': col, 'id': col} for col in bl_results_df.columns]
    bl_results_data = bl_results_df.to_dict('records')
    return disp_bl_cov, bl_results_columns, bl_results_data


app.run_server()
