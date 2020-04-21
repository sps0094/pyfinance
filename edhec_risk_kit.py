# Various Import Statements

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
import scipy.stats as sp
from scipy.optimize import Bounds, minimize
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output, State
import ast
from ipywidgets import widgets
import json


def get_df_columns(filename):
    df = pd.read_csv(filename, na_values=-99.99, index_col=0, parse_dates=[0])
    df.dropna(how='all', inplace=True, axis=1)
    df.columns = df.columns.str.strip()
    return df.columns


def get_df(filename, start_period, end_period, format, reqd_strategies, mode='return'):
    """

    :param filename:
    :param start_period:
    :param end_period:
    :param format:
    :param reqd_strategies:
    :param mode: return or nos or size
    :return:
    """
    df = pd.read_csv(filename, index_col=0, parse_dates=[0], na_values=-99.99)
    if mode == 'return':
        df = df / 100
    df.dropna(how='all', inplace=True, axis=1)
    df.columns = df.columns.str.strip()
    df = df[reqd_strategies]
    df.index = pd.to_datetime(df.index, format=format)
    return df[start_period:end_period]


def get_ann_vol(df):
    ann_vol = df.std() * np.sqrt(12)
    return ann_vol


def get_ann_return(df):
    ann_factor = 12 / len(df.index)
    ann_ret_np = np.expm1(ann_factor * (np.log1p(df).sum()))  # using log method for eff computation
    return ann_ret_np


def get_sharpe_ratio(ann_ret, ann_vol, rf=0.10):
    return (ann_ret - rf) / ann_vol


def get_semi_std(df):
    semi_std = df[df < 0].std(ddof=0)
    return semi_std


def hist_var(col_series, alpha):
    return np.percentile(col_series, alpha * 100)


def para_var(col_series, alpha):
    z = sp.norm.ppf(alpha)
    return col_series.mean() + z * col_series.std(ddof=0)


def corn_var(col_series, alpha):
    z = sp.norm.ppf(alpha)
    kurtosis = sp.kurtosis(col_series, fisher=True)
    skew = sp.skew(col_series)
    z = (z +
         (z ** 2 - 1) * skew / 6 +
         (z ** 3 - 3 * z) * (kurtosis - 3) / 24 -
         (2 * z ** 3 - 5 * z) * (skew ** 2) / 36)
    return col_series.mean() + z * col_series.std(ddof=0)


def get_VaR(df: pd.DataFrame, var_method, alpha):
    if var_method == 'historic':
        return df.aggregate(hist_var, alpha=alpha)
    elif var_method == 'parametric':
        return df.aggregate(para_var, alpha=alpha)
    elif var_method == 'cornish':
        return df.aggregate(corn_var, alpha=alpha)


def get_CVaR(df, VaR):
    CVaR = pd.Series(
        {df.columns[i]: df[df[df.columns[i]] < VaR[i]][df.columns[i]].mean() for i in range(len(df.columns))})
    return CVaR


def add_1(ddf):
    return ddf + 1


def drawdown(df: pd.DataFrame, retrive_index=False, init_wealth=1000):
    if retrive_index:
        factor = np.exp(np.cumsum(np.log(df)))  # using log instead of cumprod for efficiency
        wealth_index = init_wealth * factor
        return wealth_index
    factor = np.exp(np.cumsum(np.log1p(df)))  # using log instead of cumprod for efficiency
    wealth_index = init_wealth * factor
    prev_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - prev_peaks) / prev_peaks
    return [wealth_index, prev_peaks, drawdowns]


def risk_info(df, risk_plot: list, rf, alpha, var_method='cornish'):
    ann_vol = get_ann_vol(df)
    ann_ret = get_ann_return(df)
    sharpe_ratio = get_sharpe_ratio(ann_ret, ann_vol, rf)
    semi_std = get_semi_std(df)
    kurtosis = sp.kurtosis(df, fisher=True)
    skew = sp.skew(df)
    VaR = get_VaR(df, var_method, alpha)
    CVaR = get_CVaR(df, VaR)
    drawdown_df = drawdown(df)[2]
    drawdown_df = drawdown_df.aggregate(lambda col_series: col_series.min())
    info = pd.DataFrame({'ann_ret': ann_ret,
                         'ann_vol': ann_vol,
                         'sharpe_ratio': sharpe_ratio,
                         'semi_dev': semi_std,
                         'Kurtosis': kurtosis,
                         'Skew': skew,
                         'VaR': VaR,
                         'CVaR': CVaR,
                         'Drawdown': drawdown_df})
    return info.sort_values(by=risk_plot, ascending=True).transpose()


def ren_df(df, rev_name, exis_name='index'):
    return df.reset_index().rename(columns={exis_name: rev_name})


def get_cov(df):
    return df.cov()


def get_pf_ret(wt_array, ret_array):
    return wt_array.T @ ret_array


def get_pf_vol(wt_array, cov_mat):
    return (wt_array.T @ cov_mat @ wt_array) ** 0.5


def annualize_pf_vol(pf_vol, periodicity):
    return pf_vol * np.sqrt(periodicity)


def format_perc(wts):
    return '{:.4%}'.format(wts)


def get_hover_info(n_assets, reqd_strategies, wts_list):
    hoverinfo = []
    pf_alloc_wts_str = [list(map(format_perc, wt_array)) for wt_array in wts_list]
    for pf_alloc in pf_alloc_wts_str:
        hovertext = ''
        for i in range(n_assets):
            hovertext += ('{}: {} \n'.format(reqd_strategies[i], pf_alloc[i]))
        hoverinfo.append(hovertext)
    return hoverinfo


def optimize_wts(ret_series, cov_mat, n_points):
    wts_list = []
    n_assets = ret_series.shape[0]
    ret_array = ret_series.to_numpy()
    init_guess = np.repeat(1 / n_assets, n_assets)
    bounds = Bounds(lb=0.0, ub=1.0)
    is_tgt_met = {
        'type': 'eq',
        'args': (ret_array,),
        'fun': lambda wt_array, ret_array: get_pf_ret(wt_array, ret_array) - tgt_ret
    }
    wts_sum_to_1 = {
        'type': 'eq',
        'fun': lambda wt_array: np.sum(wt_array) - 1
    }
    for tgt_ret in np.linspace(ret_series.min(), ret_series.max(), n_points):
        results = minimize(fun=get_pf_vol,
                           args=(cov_mat,),
                           method='SLSQP',
                           bounds=bounds,
                           constraints=[is_tgt_met, wts_sum_to_1],
                           x0=init_guess,
                           options={'disp': False})
        wts_list.append(results.x)
    return wts_list


def get_mean_var_pts(ret_series, cov_df, n_points, reqd_strategies):
    n_assets = ret_series.shape[0]
    ret_array = ret_series.to_numpy()
    cov_mat = cov_df.to_numpy()
    wts_list = optimize_wts(ret_series, cov_mat, n_points)
    pf_ret = [get_pf_ret(wt_array, ret_array) for wt_array in wts_list]
    pf_vol = [annualize_pf_vol(get_pf_vol(wt_array, cov_mat), 12) for wt_array in wts_list]
    hover_desc = get_hover_info(n_assets, reqd_strategies, wts_list)
    mean_var_df = pd.DataFrame({'Returns': pf_ret,
                                'Volatility': pf_vol,
                                'Hover Description': hover_desc})
    return mean_var_df


def get_msr(ret_series, cov_df, rf, reqd_strategies, gmv=False):
    n_assets = ret_series.shape[0]
    if gmv:
        ret_array = np.repeat(1.0,
                              n_assets)  # for gmv wts to be independent of E(R) and thus minimisation function tries to manipulate volatility to minimioze -ve SR
    else:
        ret_array = ret_series.to_numpy()
    cov_mat = cov_df.to_numpy()
    bounds = Bounds(lb=0.0, ub=1.0)
    init_guess = np.repeat(1 / n_assets, n_assets)
    sum_wts_to_1 = {
        'type': 'eq',
        'fun': lambda wt_array: np.sum(wt_array) - 1
    }

    def neg_msr(wt_array, ret_array, cov_mat, rf):
        return -(get_pf_ret(wt_array, ret_array) - rf) / get_pf_vol(wt_array, cov_mat)

    results = minimize(fun=neg_msr,
                       args=(ret_array, cov_mat, rf,),
                       method='SLSQP',
                       bounds=bounds,
                       constraints=[sum_wts_to_1],
                       options={'disp': False},
                       x0=init_guess)
    msr_wt_array = results.x
    if gmv:
        ret_array = ret_series.to_numpy()  # ret_series restored for calculating mean_var pts using optimized weights (optimized independent of E(R))
    msr_ret = get_pf_ret(msr_wt_array, ret_array)
    msr_vol = annualize_pf_vol(get_pf_vol(msr_wt_array, cov_mat), 12)
    hover_desc = get_hover_info(n_assets, reqd_strategies, [msr_wt_array])[0]
    return [msr_vol, msr_ret, hover_desc, msr_wt_array]


def get_gmv(ret_series, cov_df, rf, reqd_strategies):
    return get_msr(ret_series, cov_df, rf, reqd_strategies, gmv=True)


def get_eq(ret_series, cov_df, reqd_strategies):
    n_assets = ret_series.shape[0]
    ret_array = ret_series.to_numpy()
    cov_mat = cov_df.to_numpy()
    eq_wt_array = np.repeat(1 / n_assets, n_assets)
    eq_ret = get_pf_ret(eq_wt_array, ret_array)
    eq_vol = annualize_pf_vol(get_pf_vol(eq_wt_array, cov_mat), 12)
    hover_desc = get_hover_info(n_assets, reqd_strategies, [eq_wt_array])[0]
    return [eq_vol, eq_ret, hover_desc]


def get_corr_mat(df, window):
    """

    :param df:
    :return: -> gives correlation matrix for each block of window period and mean correlations
    """
    corr_mat = df.rolling(window=window).corr().dropna(how='all', axis=0)
    corr_mat.index.names = ['Date', 'Sector']
    corr_groupings = corr_mat.groupby(level='Date')
    corr_series = corr_groupings.apply(lambda
                                           corr_mat: corr_mat.values.mean())  # getting mean corr for corr_mat for each date (each date being groupedby)
    return [corr_mat, corr_series]


def cipp_algo(risky_ret_df: pd.DataFrame, multiplier, floor: float, reqd_strategies, poi, alpha, var_method, rf=0.03,
              max_draw_mode=False, plot=True, s0=1000, gbm=False):
    def repl_shape(new_df: pd.DataFrame, tgt_df: pd.DataFrame):
        return new_df.reindex_like(tgt_df)

    init_wealth = s0
    pf_value = init_wealth
    prev_peak = init_wealth
    floor_value = init_wealth * floor
    riskfree_df = repl_shape(pd.DataFrame(), risky_ret_df)
    cppi_ret_history = repl_shape(pd.DataFrame(), risky_ret_df)
    cppi_value_history = repl_shape(pd.DataFrame(), risky_ret_df)
    cppi_risky_wt_history = repl_shape(pd.DataFrame(), risky_ret_df)
    cppi_cushion_history = repl_shape(pd.DataFrame(), risky_ret_df)
    cppi_floor_history = repl_shape(pd.DataFrame(), risky_ret_df)
    riskfree_df[:] = rf / 12

    for dt_index in range(len(risky_ret_df.index)):
        if max_draw_mode:
            prev_peak = np.maximum(prev_peak, pf_value)
            floor_value = prev_peak * floor
        cushion = (pf_value - floor_value) / pf_value
        risky_wt = multiplier * cushion
        risky_wt = np.maximum(risky_wt, 0)
        risky_wt = np.minimum(risky_wt, 1)
        rf_wt = 1 - risky_wt
        cppi_pf_rt = (risky_wt * risky_ret_df.iloc[dt_index]) + (rf_wt * riskfree_df.iloc[dt_index])
        pf_value = (cppi_pf_rt + 1) * pf_value

        # create logs
        cppi_ret_history.iloc[dt_index] = cppi_pf_rt.transpose()
        cppi_value_history.iloc[dt_index] = pf_value.transpose()
        cppi_cushion_history.iloc[dt_index] = cushion
        cppi_floor_history.iloc[dt_index] = floor_value
        cppi_risky_wt_history.iloc[dt_index] = risky_wt

    if gbm:
        return cppi_value_history
    # plot wealth index, drawdowns cushions and weights
    app = dash.Dash()
    temp_risky_ret = drawdown(risky_ret_df)
    risky_wealth = temp_risky_ret[0]
    risky_drawdown = temp_risky_ret[2]
    cppi_drawdown = drawdown(cppi_ret_history)[2]
    cppi_wealth_plot = go.Scatter(x=cppi_value_history.index,
                                  y=cppi_value_history[poi],
                                  name='cppi_wealth_index',
                                  text=(cppi_risky_wt_history[poi] * 100).round(decimals=2))
    cppi_drawdown_plot = go.Scatter(x=cppi_drawdown.index,
                                    y=cppi_drawdown[poi],
                                    name='cppi_drawdown')
    cppi_wt_plot = go.Scatter(x=cppi_risky_wt_history.index,
                              y=cppi_risky_wt_history[poi],
                              name='cppi-risky-asset-alloc')
    risky_wealth_plot = go.Scatter(x=risky_wealth.index,
                                   y=risky_wealth[poi],
                                   name='risky_wealth_index')
    risky_drawdown_plot = go.Scatter(x=risky_drawdown.index,
                                     y=risky_drawdown[poi],
                                     name='risky_drawdown')
    floor_plot = go.Scatter(x=cppi_floor_history.index,
                            y=cppi_floor_history[poi],
                            mode='lines',
                            line=dict(dash='dashdot',
                                      width=3),
                            name='Floor')
    lowpt_cppi_drawdown = cppi_drawdown[poi].min()
    lowpt_cppi_drawdown_year = cppi_drawdown[poi].idxmin()
    lowpt_risky_drawdown = risky_drawdown[poi].min()
    lowpt_risky_drawdown_year = risky_drawdown[poi].idxmin()
    lowpts = [[lowpt_cppi_drawdown, lowpt_cppi_drawdown_year], [lowpt_risky_drawdown, lowpt_risky_drawdown_year]]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    fig.add_trace(cppi_wealth_plot, row=1, col=1)
    fig.add_trace(risky_wealth_plot, row=1, col=1)
    fig.add_trace(floor_plot, row=1, col=1)
    fig.add_trace(cppi_drawdown_plot, row=2, col=1)
    fig.add_trace(risky_drawdown_plot, row=2, col=1)
    fig.add_trace(cppi_wt_plot, row=3, col=1)
    fig.update_layout(height=750)
    annotations = [dict(x=i[1],
                        y=i[0],
                        ax=0,
                        ay=50,
                        xref='x',
                        yref='y2',
                        arrowhead=7,
                        showarrow=True,
                        text='Max DrawDown is {} and occurred at {}'.format(i[0], i[1])) for i in lowpts]
    fig.update_layout(annotations=annotations)
    app.layout = html.Div([dcc.Graph(id='drawdowns', figure=fig)])
    if __name__ != '__main__' and plot:
        app.run_server()

    # create a result dataframe
    backtest_results = {'cppi_wealth': cppi_value_history,
                        'cppi_return': cppi_ret_history,
                        'cppi_drawdown': cppi_drawdown,
                        'cppi_risky_wts': cppi_risky_wt_history,
                        'risky_wealth': risky_wealth,
                        'floor': floor_value,
                        'risky_drawdown': risky_drawdown}
    return backtest_results


def plot_eff_frontier(ret_series: pd.Series, cov_df: pd.DataFrame, n_points: int, reqd_strategies, rf, show_msr=True,
                      show_eq=False, show_gmv=True):
    mean_var_df = get_mean_var_pts(ret_series, cov_df, n_points, reqd_strategies)
    msr = get_msr(ret_series, cov_df, rf, reqd_strategies)
    eq = get_eq(ret_series, cov_df, reqd_strategies)
    gmv = get_gmv(ret_series, cov_df, rf, reqd_strategies)
    app = dash.Dash()
    data = [go.Scatter(x=mean_var_df['Volatility'],
                       y=mean_var_df['Returns'],
                       mode='markers+lines',
                       name='efficient_frontier',
                       text=mean_var_df['Hover Description'])]
    if show_msr:
        data.append(go.Scatter(x=[0, msr[0]],
                               y=[rf, msr[1]],
                               mode='markers+lines',
                               name='CML',
                               text=['RF - 100%', msr[2]]))
    if show_eq:
        data.append(go.Scatter(x=[eq[0]],
                               y=[eq[1]],
                               mode='markers',
                               name='EQ',
                               text=[eq[2]]))
    if show_gmv:
        data.append(go.Scatter(go.Scatter(x=[gmv[0]],
                                          y=[gmv[1]],
                                          mode='markers',
                                          name='GMV',
                                          text=[gmv[2]])))

    app.layout = html.Div([html.Div([dcc.Graph(id='eff_frontier', figure=dict(data=data,
                                                                              layout=go.Layout(
                                                                                  title='Efficient Frontier',
                                                                                  xaxis=dict(title='Variance'),
                                                                                  yaxis=dict(title='mean'),
                                                                                  hovermode='closest')))]),
                           html.Div([html.Pre(id='display_info')])])

    @app.callback(Output('display_info', 'children'),
                  [Input('eff_frontier', 'hoverData')])
    def upd_markdown(hover_data):
        hover_data = hover_data['points'][0]
        wts_data = hover_data['text']
        pf_vol_data = hover_data['x']
        pf_ret_data = hover_data['y']
        disp_data = '''
            The weights are \n{}
            PF - Volatility: {:.2%}
            PF - Return    : {:.2%}
        '''.format(wts_data, pf_vol_data, pf_ret_data)
        return disp_data

    if __name__ != '__main__':
        app.run_server()


def plot_corr_mktret(ind_ret_filename, n_firms_filename, size_filename, start_period, end_period, format,
                     reqd_strategies, window=36, retrieve_mcw=False):
    app = dash.Dash()
    # Populate all reqd dataframes
    ind_ret_m_df = get_df(ind_ret_filename, start_period, end_period, format, reqd_strategies, mode='return')
    ind_n_firms_df = get_df(n_firms_filename, start_period, end_period, format, reqd_strategies, mode='nos')
    ind_size_df = get_df(size_filename, start_period, end_period, format, reqd_strategies, mode='size')

    # industry returns --> mkt cap returns for index constructions
    mkt_cap_df = ind_n_firms_df * ind_size_df
    total_mkt_cap_series = mkt_cap_df.sum(axis=1)
    mkt_wts_df = mkt_cap_df.divide(total_mkt_cap_series, axis=0)
    mcw = ind_ret_m_df * mkt_wts_df
    mcw_ret_df = pd.DataFrame({'mkt_cap_wt_ret_monthly': mcw.sum(axis=1)})
    if retrieve_mcw:
        return mcw_ret_df

    # index_generation
    mcw_index = drawdown(mcw_ret_df)[0]
    # mcw_index_36MA = mcw_index.rolling(window=window).mean()

    # rolling returns
    mcw_rolling_returns = mcw_ret_df.rolling(window=window).aggregate(get_ann_return)

    # corr matrix
    corr_results = get_corr_mat(mcw, window=window)
    corr_series = corr_results[1]

    # plots
    # ret_data = go.Scatter(x=mcw_ret_df.index,
    #                       y=mcw_ret_df['mkt_cap_wt_ret_monthly'],
    #                       mode='lines',
    #                       name='mcw_returns')
    roll_ret_data = go.Scatter(x=mcw_rolling_returns.index,
                               y=mcw_rolling_returns['mkt_cap_wt_ret_monthly'],
                               mode='lines',
                               name='roll_returns')
    roll_corr_data = go.Scatter(x=corr_series.index,
                                y=corr_series,
                                mode='lines',
                                name='roll_corr',
                                yaxis='y2')
    # index_data = go.Scatter(x=mcw_index.index,
    #                         y=mcw_index['mkt_cap_wt_ret_monthly'],
    #                         mode='lines',
    #                         name='index')
    # ma_data = go.Scatter(x=mcw_index_36MA.index,
    #                      y=mcw_index_36MA['mkt_cap_wt_ret_monthly'],
    #                      mode='lines',
    #                      name='ma_index',
    #                      yaxis='y2')
    layout = go.Layout(yaxis=dict(title='roll_return'),
                       yaxis2=dict(side='right',
                                   overlaying='y1',
                                   title='roll_corr'),
                       hovermode='closest')
    app.layout = html.Div([dcc.Graph(id='corr', figure=dict(data=[roll_ret_data, roll_corr_data], layout=layout))])

    if __name__ != '__main__':
        app.run_server()


def plot(df, mode, reqd_strategies: list, risk_plot: list, poi, var_method, alpha, rf):
    alpha = alpha / 100
    app = dash.Dash()
    infodf = risk_info(df, risk_plot=risk_plot, rf=rf, alpha=alpha, var_method=var_method)
    idx = reqd_strategies.index(poi)
    if mode == 'returns' or mode == 'downside':
        hist_plot = [df[col] for col in df.columns]
        group_labels = df.columns
        fig = ff.create_distplot(hist_plot, group_labels, show_hist=False)
        if mode == 'downside':
            var_annotation_x = infodf.loc['VaR'][idx]
            cvar_annotation_x = infodf.loc['CVaR'][idx]
            annotations = [dict(x=var_annotation_x,
                                y=0,
                                ax=0,
                                ay=-200,
                                showarrow=True,
                                arrowhead=7,
                                text='Min {} probability for {} % loss'.format(alpha, -(var_annotation_x * 100).round(
                                    decimals=4)),
                                xref='x',
                                yref='y'),
                           dict(x=cvar_annotation_x,
                                y=0,
                                ax=0,
                                ay=-100,
                                showarrow=True,
                                arrowhead=7,
                                text='Expected loss is {} %'.format(-(cvar_annotation_x * 100).round(decimals=4)),
                                xref='x',
                                yref='y')
                           ]
            fig.update_layout(annotations=annotations)
        app.layout = html.Div([dcc.Graph(id='returns', figure=fig)])

    elif mode == 'risk_stats':
        infodf = ren_df(infodf, 'risk_params', 'index')
        app.layout = dt.DataTable(id='risk-stats',
                                  columns=[{'name': col,
                                            'id': col} for col in infodf.columns],
                                  data=infodf.to_dict('records'))

    elif mode == 'risk_plot':
        data = [go.Bar(x=infodf.columns,
                       y=infodf.loc[risk_type],
                       name=risk_type) for risk_type in risk_plot]
        app.layout = html.Div([dcc.Graph(id='risk_plots', figure=dict(data=data))])

    elif mode == 'drawdowns':
        all_index = drawdown(df)
        ddf = all_index[2][reqd_strategies[idx]]
        wdf = all_index[0][reqd_strategies[idx]]
        pdf = all_index[1][reqd_strategies[idx]]
        wealth_plot = go.Scatter(x=df.index,
                                 y=wdf,
                                 name='wealth_index')
        peak_plot = go.Scatter(x=df.index,
                               y=pdf,
                               name='peak_index')
        draw_plot = go.Scatter(x=df.index,
                               y=ddf,
                               name='drawdown')
        lowpt_drawdown = ddf.min()
        lowpt_drawdown_year = ddf.idxmin()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.add_trace(wealth_plot, row=1, col=1)
        fig.add_trace(peak_plot, row=1, col=1)
        fig.add_trace(draw_plot, row=2, col=1)
        annotations = [dict(x=lowpt_drawdown_year,
                            y=lowpt_drawdown,
                            ax=0,
                            ay=150,
                            xref='x',
                            yref='y2',
                            arrowhead=7,
                            showarrow=True,
                            text='Max DrawDown is {} and occurred at {}'.format(lowpt_drawdown, lowpt_drawdown_year))]
        fig.update_layout(annotations=annotations)
        app.layout = html.Div([dcc.Graph(id='drawdowns', figure=fig)])


    else:
        app.layout = html.Div([dcc.Markdown(id='test', children='Hi')])
    if __name__ != '__main__':
        app.run_server()


def gbm(s0=100):
    # plot
    app = dash.Dash()
    app.layout = html.Div(
        [html.Div([html.Label(id='l_sce', children='N-Scenarios: '), dcc.Input(id='i_sce', type='number', value=10)]),
         html.Div([html.Label(id='l_st/yr', children='N-Steps per year: '),
                   dcc.Input(id='i_st/yr', type='number', value=12)]),
         html.Div([html.Label(id='l_yr', children='N-Years: '), dcc.Input(id='i_yr', type='number', value=10)]),
         html.Div([html.Label(id='l_er', children='Expected Return: '),
                   dcc.Input(id='i_er', type='number', value=0.07, step=0.005)]),
         html.Div([html.Label(id='l_vol', children='Expected Volatility: '),
                   dcc.Input(id='i_vol', type='number', value=0.15, step=0.005)]),
         html.Div([html.Label(id='l_floor', children='Floor: '),
                   dcc.Input(id='i_floor', type='number', value=0.8, step=0.1)]),
         html.Div([html.Label(id='l_multi', children='Multiplier: '), dcc.Input(id='i_multi', type='number', value=3)]),
         html.Div([html.Label(id='l_rf', children='Risk Free Rate: '),
                   dcc.Input(id='i_rf', type='number', value=0.03, step=0.005)]),
         html.Div([dcc.RadioItems(id='cppi', options=[{'label': 'CPPI?', 'value': 1}, {'label': 'Risky?', 'value': 0}],
                                  value=1)]),
         html.Div([html.Button(id='gen_gbm', children='Generate', n_clicks=0)]),
         html.Div([dcc.Graph(id='gbm_plot')]),
         html.Div([dcc.Markdown(id='gbm_stats')], style={'fontsize': '40em'})])

    @app.callback(Output('gbm_stats', 'children'),
                  [Input('gen_gbm', 'n_clicks')],
                  [State('i_sce', 'value'),
                   State('i_st/yr', 'value'),
                   State('i_yr', 'value'),
                   State('i_er', 'value'),
                   State('i_vol', 'value'),
                   State('i_floor', 'value'),
                   State('i_multi', 'value'),
                   State('i_rf', 'value'),
                   State('cppi', 'value')])
    def update_gbm(n_clicks, n_scenarios, steps_per_yr, n_years, er, vol, floor, multiplier, rf, cppi):
        floor_value = floor * s0
        dt = 1 / steps_per_yr
        total_time_steps = int(n_years * steps_per_yr)


        # Using refined method
        dz = np.random.normal(loc=(1 + er) ** dt, scale=vol * np.sqrt(dt),
                              size=(total_time_steps, n_scenarios))
        # mu and sigma is annualized.
        # The drift and rw terms require mu and sigma for the infinitesimally small time.
        # Even better to use continuous comp ret.
        # eg dt = 0.25 and mu is 10% per year. So drift term for 1Qtr needs mu for such qtr viz (1.1)**0.25
        gbm_df = pd.DataFrame(dz)
        gbm_df.loc[0] = 1.0
        if cppi:
            gbm_df = gbm_df.apply(lambda gbm_rets: gbm_rets - 1)
            wealth_index = cipp_algo(gbm_df, multiplier=multiplier, floor=floor, reqd_strategies=[''], poi='', alpha='',
                                     var_method='', rf=rf, s0=s0, gbm=True)
        else:
            wealth_index = drawdown(gbm_df, retrive_index=True, init_wealth=s0)
        wealth_index.to_csv('tempfile.csv')
        terminal_wealth = wealth_index.iloc[-1]
        exp_wealth = np.mean(terminal_wealth)
        med_wealth = np.median(terminal_wealth)
        failure_mask = np.less(terminal_wealth, floor_value)
        n_breaches = failure_mask.sum()
        p_breaches = n_breaches / n_scenarios
        # exp_loss_post_breach = np.mean(terminal_wealth[failure_mask]) if n_breaches > 0 else 0.0
        # exp_shortfall1 = floor_value - exp_loss_post_breach if n_breaches > 0 else 0.0
        exp_shortfall = np.dot(floor_value - terminal_wealth, failure_mask) / n_breaches if n_breaches > 0 else 0.0
        best_case = terminal_wealth.max()
        worst_case = terminal_wealth.min()
        return '''
        Mean: ${:.2f}\n
        Median: ${:.2f}\n
        Violations: {} ({:.2%})\n
        Exp Shortfall: ${:.2f}\n
        Diff in worst and best case scenario: {}\n
        Worst Case: {}
        '''.format(exp_wealth, med_wealth, n_breaches, p_breaches, exp_shortfall, best_case - worst_case, worst_case)

        # data point for plotting

    @app.callback(Output('gbm_plot', 'figure'),
                  [Input('gbm_stats', 'children'),
                   Input('i_floor', 'value')])
    def upd_gbm_plot(gbm_stats, floor):
        floor_value = floor * s0
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
        wealth_index = pd.read_csv('tempfile.csv', index_col='Unnamed: 0')
        gbm_motion = wealth_index.aggregate(lambda scenario: go.Scatter(x=scenario.index, y=scenario))
        gbm_motion = gbm_motion.tolist()
        hist_plot = []
        for scenario in wealth_index.columns:
            hist_plot.append(wealth_index[scenario].tolist())
        length = len(hist_plot)
        for i in range(length - 1):
            hist_plot[0].extend(hist_plot[i + 1])
        hist_plot = go.Histogram(y=hist_plot[0],
                                 name='gbm_dist')
        for gbm_data in gbm_motion:
            fig.add_trace(gbm_data, row=1, col=1)
        fig.add_trace(hist_plot, row=1, col=2)  # Hist bin plot
        floor_threshold = [
            dict(type='line', xref='paper', yref='y1', x0=0, x1=1, y0=floor_value, y1=floor_value, name='floor',
                 line=dict(dash='dashdot', width=5))]
        fig.update_layout(showlegend=False,
                          hovermode='y',
                          height=500,
                          shapes=floor_threshold)
        return fig

    if __name__ != '__main__':
        app.run_server()


def get_discount_factor(disc_rate, periods):
    return np.power((1 + disc_rate), -periods)


def get_present_value(liabilities: pd.Series, disc_rate):
    present_value = (liabilities * get_discount_factor(disc_rate, liabilities.index)).sum()
    return present_value


def get_funding_ratio(liabilities: pd.Series, assets, disc_rate):
    pv = get_present_value(liabilities, disc_rate)
    return [np.divide(assets, pv), pv]


def conv_to_short_rate(r):
    """
    price relative = exp(t*sr) => ln(1+r)/t = sr (assumes t = 1)
    :param r: annualised interest rate
    :return: short rates
    """
    return np.log1p(r)

def conv_to_annualised_rate(sr):
    """
    exp(t*sr) - 1 = r (assumes t = 1)
    :param sr: short rate
    :return: annualised rate for a given short rate
    """
    return np.expm1(sr)


def cir():
    app = dash.Dash()

    def upd_label(out_id, inp_id):
        @app.callback(Output(out_id, 'children'),
                      [Input(inp_id, 'value')])
        def upd_(value):
            return value

    app.layout = html.Div([html.Div([html.Div([html.Label(children='Select Initial asset value: '),
                                     dcc.Slider(id='sl_av', min=0.10, max=1.5, step=0.05, value=0.75),
                                     html.Label(id='out_av')], style={'display': 'inline-block'}),
                           html.Div([html.Label(children='Select Initial rf annualized: '),
                                     dcc.Slider(id='sl_rf', min=0.01, max=0.10, step=0.005, value=0.03),
                                     html.Label(id='out_rf')], style={'display': 'inline-block'}),
                           html.Div([html.Label(children='Select expected LT RF: '),
                                    dcc.Slider(id='sl_ltrf', min=0.01, max=0.10, step=0.005, value=0.05),
                                    html.Label(id='out_ltrf')], style={'display': 'inline-block'}),
                           html.Div([html.Label(children='Select speed of MR: '),
                                    dcc.Slider(id='sl_speed', min=0, max=1, step=0.05, value=0.6),
                                    html.Label(id='out_speed')], style={'display': 'inline-block'}),
                           html.Div([html.Label(children='Select volatility: '),
                                     dcc.Slider(id='sl_vola', min=0, max=1, step=0.05, value=0.15),
                                     html.Label(id='out_vola')], style={'display': 'inline-block'}),
                           html.Button(id='sub_cir', children='SUBMIT', n_clicks=0, style={'display': 'inline-block'})], style= {'display': 'flex', 'justify-content': 'space-evenly'}),
                           html.Div([html.Div([html.Label(children='Select N-Periods: '),
                                               dcc.Slider(id='sl_periods', min=1, max=20, step=1, value=10),
                                               html.Label(id='out_periods')], style={'display': 'inline-block'}),
                                     html.Div([html.Label(children='Select steps_per_yr: '),
                                               dcc.Slider(id='sl_stperyr', min=1, max=20, step=1, value=12),
                                               html.Label(id='out_stperyr')], style={'display': 'inline-block'}),
                                     html.Div([html.Label(children='Select N-Scenarios: '),
                                               dcc.Slider(id='sl_scenarios', min=1, max=250, step=1, value=10),
                                               html.Label(id='out_scenarios')], style={'display': 'inline-block'})
                                     ], style= {'display': 'flex', 'justify-content': 'space-evenly', 'padding-top': '25px'}),
                           html.Div([dcc.Graph(id='cir')])])

    upd_label('out_av', 'sl_av')
    upd_label('out_rf', 'sl_rf')
    upd_label('out_ltrf', 'sl_ltrf')
    upd_label('out_speed', 'sl_speed')
    upd_label('out_vola', 'sl_vola')
    upd_label('out_periods', 'sl_periods')
    upd_label('out_stperyr', 'sl_stperyr')
    upd_label('out_scenarios', 'sl_scenarios')

    @app.callback(Output('cir', 'figure'),
                  [Input('sub_cir', 'n_clicks')],
                  [State('sl_rf', 'value'),
                   State('sl_periods', 'value'),
                   State('sl_stperyr', 'value'),
                   State('sl_scenarios', 'value'),
                   State('sl_vola', 'value'),
                   State('sl_speed', 'value'),
                   State('sl_ltrf', 'value')])
    def upd_cir(n_clicks, rf, n_years, steps_per_yr, n_scenarios, volatility, a, b):
        dt = 1/steps_per_yr
        sr = conv_to_short_rate(rf)
        total_time_steps = int(n_years*steps_per_yr)+1
        shock = np.random.normal(loc=0, scale=volatility*np.sqrt(dt), size=(total_time_steps, n_scenarios))
        rates = np.empty_like(shock)
        rates[0] = sr
        for steps in range(1, total_time_steps):
            prev_rate = rates[steps-1]
            drift = a * (b - prev_rate)
            shock[steps] = shock[steps] * np.sqrt(prev_rate)
            dr = drift + shock[steps]
            rates[steps] = prev_rate + dr
        rates_gbm_df = pd.DataFrame(conv_to_annualised_rate(rates), index=range(total_time_steps))

        fig = make_subplots(rows=1, cols=2)
        rates_gbm = rates_gbm_df.aggregate(lambda scenario: go.Scatter(x=scenario.index, y=scenario))
        rates_gbm = rates_gbm.tolist()
        for rates_data in rates_gbm:
            fig.add_trace(rates_data, row=1, col=1)
        mrl = [dict(type='line', xref='paper', yref='y1', x0=0, x1=1, y0=b, y1=b, name='Mean Reverting Level',
                    line=dict(dash='dashdot', width=5))]
        fig.update_layout(showlegend=False,
                          title='Cox Ingresol Roxx Model with GBM model of interest rates',
                          height=500,
                          hovermode='closest',
                          shapes=mrl)
        return fig


    app.run_server()




