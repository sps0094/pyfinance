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
from scipy.optimize import Bounds, minimize, minimize_scalar
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output, State
import math
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
    :param start_period:None if NA
    :param end_period:None if NA
    :param format:
    :param reqd_strategies: None if NA
    :param mode: return or nos or size
    :return:
    """
    df = pd.read_csv(filename, index_col=0, parse_dates=[0], na_values=-99.99)
    if mode == 'return':
        df = df / 100
    df.dropna(how='all', inplace=True, axis=1)
    df.columns = df.columns.str.strip()
    if reqd_strategies is not None:
        df = df[reqd_strategies]
    df.index = pd.to_datetime(df.index, format=format)
    if start_period and end_period is not None:
        return df[start_period:end_period]
    else:
        return df


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


def drawdown(df: pd.DataFrame, retrive_index=False, init_wealth=1000, is1p=True):
    if retrive_index:
        if is1p:
            factor = np.exp(np.cumsum(np.log(df)))  # using log instead of cumprod for efficiency
        else:
            factor = np.exp(np.cumsum(np.log1p(df)))  # using log instead of cumprod for efficiency
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


def terminal_risk_stats(fv, floor_factor, wealth_index, aslst=False, strategy=None):
    floor_value = fv*floor_factor
    if isinstance(wealth_index, pd.DataFrame):
        terminal_wealth = wealth_index.iloc[-1]
    else:
        terminal_wealth = wealth_index # The terminal row
    n_scenarios = terminal_wealth.shape[0]
    exp_wealth = np.mean(terminal_wealth)
    med_wealth = np.median(terminal_wealth)
    vol_wealth = np.std(terminal_wealth)
    failure_mask = np.less(terminal_wealth, floor_value)
    n_breaches = failure_mask.sum()
    p_breaches = n_breaches / n_scenarios
    # exp_loss_post_breach = np.mean(terminal_wealth[failure_mask]) if n_breaches > 0 else 0.0
    # exp_shortfall1 = floor_value - exp_loss_post_breach if n_breaches > 0 else 0.0
    exp_shortfall = np.dot(floor_value - terminal_wealth, failure_mask) / n_breaches if n_breaches > 0 else 0.0
    best_case = terminal_wealth.max()
    worst_case = terminal_wealth.min()
    if aslst:
        stats = [strategy, exp_wealth, vol_wealth, med_wealth, n_breaches, p_breaches, exp_shortfall]
        return stats
    else:
        return '''
                Mean: ${:.2f}\n
                Median: ${:.2f}\n
                Violations: {} ({:.2%})\n
                Exp Shortfall: ${:.2f}\n
                Diff in worst and best case scenario: {}\n
                Worst Case: {}
                '''.format(exp_wealth, med_wealth, n_breaches, p_breaches, exp_shortfall, best_case - worst_case,
                           worst_case)


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


def gbm_stock(s0, n_scenarios, steps_per_yr, n_years, er, vol, floor, multiplier, rf, cppi, ret_series=False):
    floor_value = floor * s0
    dt = 1 / steps_per_yr
    total_time_steps = int(n_years * steps_per_yr)+1

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
    if ret_series:
        gbm_df.drop(0, inplace=True)
        return gbm_df
    return wealth_index


def plot_gbm(s0=100):
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
        wealth_index = gbm_stock(s0, n_scenarios, steps_per_yr, n_years, er, vol, floor, multiplier, rf, cppi)
        wealth_index.to_csv('tempfile.csv')
        result_stats = terminal_risk_stats(s0, floor, wealth_index)
        return result_stats

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


def get_macaulay_duration(pvf): # Nota annualized. Make sure to annualize
    mac_dur = pvf.apply(lambda pvf: np.average(pvf.index+1, weights=pvf))
    return mac_dur


def get_discount_factor(disc_rate:pd.DataFrame, period):
    disc_factors_df = disc_rate.apply(lambda r: np.power((1+r), -period))
    return disc_factors_df


def get_present_value(cash_flow: pd.Series, disc_rate: pd.DataFrame):
    if not isinstance(disc_rate, pd.DataFrame):
        cash_flow.index -= 1 #To correct for cash_flow.index+1 when called from cir()
        disc_rate = pd.DataFrame(data=[disc_rate for t in cash_flow.index], index=cash_flow.index)
        get_present_value(cash_flow, disc_rate)
    if not len(disc_rate.index) == len(cash_flow.index):
        dr_steps = disc_rate.shape[0]
        cf_steps = cash_flow.shape[0]
        shortfall = cf_steps - dr_steps
        dr_last = disc_rate.iloc[-1]
        append_rate_df = pd.DataFrame(data=np.asarray(pd.concat([dr_last] * shortfall, axis=0)).reshape(shortfall, disc_rate.shape[1]),
                          index=range(dr_steps, cf_steps, 1))
        disc_rate = disc_rate.append(append_rate_df)
    disc_factors = get_discount_factor(disc_rate, cash_flow.index+1)
    present_value_factors = disc_factors.apply(lambda disc_factor: disc_factor*cash_flow)
    present_value = present_value_factors.sum()
    mac_dur = get_macaulay_duration(present_value_factors)
    return np.asarray(present_value), mac_dur


def gen_bond_cash_flows(tenor, steps_per_year, cr, fv):
    dt = 1/steps_per_year
    total_time_steps = int(tenor*steps_per_year)
    periodicity_adj_cr = cr * dt
    coupon_cf = fv * periodicity_adj_cr
    bond_cf = pd.Series([coupon_cf for i in range(0, total_time_steps)])
    bond_cf.iloc[-1] += fv
    return bond_cf


def get_bond_prices(n_years, tenor, steps_per_year, disc_rate, cr=0.03, fv=100):
    dt = 1 / steps_per_year
    if isinstance(disc_rate, pd.DataFrame):
        periodicity_adj_disc_rate = disc_rate * dt
        bond_cf = gen_bond_cash_flows(tenor, steps_per_year, cr, fv)
        bond_prices, mac_dur = get_present_value(bond_cf, periodicity_adj_disc_rate)
        return bond_prices, mac_dur, bond_cf
    else:
        total_time_steps = int(n_years*steps_per_year)
        disc_rate = pd.DataFrame(data=np.repeat(disc_rate, total_time_steps).reshape(total_time_steps,1))
        return get_bond_prices(n_years,tenor,steps_per_year, disc_rate, cr, fv)


def get_funding_ratio(pv_liabilities, pv_assets):
    return np.divide(pv_assets, pv_liabilities)


def get_terminal_wealth(rets):
    return np.exp(np.log1p(rets).sum())


def cumulate(rets):
    return np.expm1(np.log1p(rets).sum())


def get_optimal_wts(md_liab, ldb, sdb, av, disc_rate, dt):
    x0 = np.repeat(0.5, 2)
    bounds = Bounds(lb=0.00, ub=1.00)

    def core_check_algo(wts, ldb, sdb, av, disc_rate, dt):
        wt_l = wts[0]
        alloc_long_dur_bond = av * wt_l
        alloc_short_dur_bond = av * (1 - wt_l)
        n_long_dur_bond_match = alloc_long_dur_bond / ldb[0]
        n_short_dur_bond_match = alloc_short_dur_bond / sdb[0]
        dur_matched_bond_cf = pd.DataFrame(data=pd.concat([ldb[2] * n_long_dur_bond_match, sdb[2] * n_short_dur_bond_match]), columns=['cf'])
        dur_matched_bond_cf = dur_matched_bond_cf.groupby(dur_matched_bond_cf.index)['cf'].sum()
        dur_matched_bond_cf.index += 1
        disc_rate = disc_rate * dt
        pv_pf, mac_dur_pf = get_present_value(dur_matched_bond_cf, disc_rate)
        mac_dur_pf = mac_dur_pf[0]
        return mac_dur_pf * dt

    def check_dur_match(wts, md_liab, ldb, sdb, av, disc_rate, dt):
        mac_dur_pf = core_check_algo(wts, ldb, sdb, av, disc_rate, dt)
        return mac_dur_pf - md_liab

    sum_wts_to_1 = {
        'type': 'eq',
        'fun': lambda wts: np.sum(wts) - 1
    }

    is_diff_zero = {
        'type': 'eq',
        'args': (md_liab, ldb, sdb, av, disc_rate, dt),
        'fun': check_dur_match
    }

    result = minimize(fun=check_dur_match,
                      args=(md_liab, ldb, sdb, av, disc_rate, dt),
                      x0=x0,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=[sum_wts_to_1, is_diff_zero],
                      options={'disp': False}
                      )
    wts = result.x
    return wts


# # NEED TO ADAPT FOR BONDS WITH VARYING COUPON PERIODS
def get_duration_matched_pf(liabilities: pd.Series, n_years: list, steps_per_year: list, disc_rate, cr: list, fv: list, av, fr_change_sim=False):
    pv_liabilities, mac_dur_liabilities = get_present_value(liabilities, disc_rate)
    pv_bond_1, mac_dur_bond_1, bond_cf_1 = get_bond_prices(n_years[0], n_years[0], steps_per_year[0], disc_rate, cr[0], fv[0])
    pv_bond_2, mac_dur_bond_2, bond_cf_2 = get_bond_prices(n_years[1], n_years[1], steps_per_year[1], disc_rate, cr[1], fv[1])
    # bond_cf_1.index += 1
    # bond_cf_2.index += 1
    mac_dur_bond_1 = mac_dur_bond_1.loc[0] / steps_per_year[0]
    mac_dur_bond_2 = mac_dur_bond_2.loc[0] / steps_per_year[1]
    mac_dur_liabilities = mac_dur_liabilities.loc[0]
    pv_bond_1 = pv_bond_1[0]
    pv_bond_2 = pv_bond_2[0]
    pv_liabilities = pv_liabilities[0]
    if mac_dur_bond_1 > mac_dur_bond_2:
        long_dur_bond = [pv_bond_1, mac_dur_bond_1, bond_cf_1, steps_per_year[0]]
        short_dur_bond = [pv_bond_2, mac_dur_bond_2, bond_cf_2, steps_per_year[1]]
    else:
        long_dur_bond = [pv_bond_2, mac_dur_bond_2, bond_cf_2, steps_per_year[1]]
        short_dur_bond = [pv_bond_1, mac_dur_bond_1, bond_cf_1, steps_per_year[0]]
    tts_for_pf = steps_per_year[0] if len(bond_cf_1.index) > len(bond_cf_2.index) else steps_per_year[1] # To adj disc_rate periodicity for dur_match pf
    dt = 1 / tts_for_pf

    #computes duration match wtss

    wt_array = get_optimal_wts(mac_dur_liabilities, long_dur_bond, short_dur_bond, av, disc_rate, dt)
    wt_long_dur_bond = wt_array[0]
    wt_short_dur_bond = wt_array[1]
    # wt_short_dur_bond = (long_dur_bond[1] - mac_dur_liabilities) / (long_dur_bond[1] - short_dur_bond[1])
    # wt_long_dur_bond = 1-wt_short_dur_bond
    # wt_long_dur_bond = 1.0
    # wt_short_dur_bond = 1-wt_long_dur_bond
    alloc_long_dur_bond = av*wt_long_dur_bond
    alloc_short_dur_bond = av*wt_short_dur_bond
    n_long_dur_bond_match = alloc_long_dur_bond / long_dur_bond[0]
    n_short_dur_bond_match = alloc_short_dur_bond / short_dur_bond[0]
    n_long_bond_full = av/long_dur_bond[0]
    n_short_bond_full = av/short_dur_bond[0]
    dur_matched_bond_cf = pd.DataFrame(data=pd.concat([long_dur_bond[2]*n_long_dur_bond_match, short_dur_bond[2]*n_short_dur_bond_match]), columns=['cf'])
    dur_matched_bond_cf = dur_matched_bond_cf.groupby(dur_matched_bond_cf.index)['cf'].sum()
    dur_matched_bond_cf.index += 1
    long_bond_cf_full = long_dur_bond[2]*n_long_bond_full
    short_bond_cf_full = short_dur_bond[2]*n_short_bond_full
    disc_rate = disc_rate * dt
    pv_pf, mac_dur_pf = get_present_value(dur_matched_bond_cf, disc_rate)
    pv_pf = pv_pf[0]
    mac_dur_pf = mac_dur_pf[0] * dt
    if fr_change_sim:
        disc_rates = np.linspace(0, 0.1, 50)
        fr_long = []
        fr_short = []
        fr_match = []
        dr_list = []
        for dr in disc_rates:
            dr_list.append(dr)
            liab, dur_li = get_present_value(liabilities, dr)
            l_bond, dur_l = get_present_value(long_bond_cf_full, dr)
            s_bond, dur_s = get_present_value(short_bond_cf_full, dr)
            m_bond, dur_m = get_present_value(dur_matched_bond_cf, dr)
            fr_long.append(get_funding_ratio(liab[0], l_bond[0]))
            fr_short.append(get_funding_ratio(liab[0], s_bond[0]))
            fr_match.append(get_funding_ratio(liab[0], m_bond[0]))
        fr = pd.DataFrame({
            'dr': dr_list,
            'fr_long': fr_long,
            'fr_short': fr_short,
            'fr_match': fr_match,
        }).set_index(keys='dr')
        app = dash.Dash()
        data = [go.Scatter(x=fr.index,
                           y=fr[col],
                           mode='lines',
                           name=col) for col in fr.columns]
        app.layout = html.Div([dcc.Graph(id='cfr', figure=dict(data=data))])
        app.run_server()
    return [wt_long_dur_bond, wt_short_dur_bond, mac_dur_pf, mac_dur_liabilities, long_dur_bond[1], short_dur_bond[1]]


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


def get_rates_gbm(rf, n_years, steps_per_yr, n_scenarios, volatility, a, b):
    dt = 1 / steps_per_yr
    b = conv_to_short_rate(b)  # Since short rates are being modelled
    sr = conv_to_short_rate(rf)
    total_time_steps = int(n_years * steps_per_yr) + 1
    shock = np.random.normal(loc=0, scale=volatility * np.sqrt(dt), size=(total_time_steps, n_scenarios))
    rates = np.empty_like(shock)
    # For ZCB price generation
    # Formula - please refer cir1.png
    h = math.sqrt(a ** 2 + 2 * volatility ** 2)
    zcb = np.empty_like(shock)

    def price(ttm, rf):
        _A = ((2 * h * math.exp((h + a) * ttm / 2)) / (2 * h + (h + a) * (math.exp(h * ttm) - 1))) ** (
                    2 * a * b / volatility ** 2)
        _B = (2 * (math.exp(h * ttm) - 1)) / (2 * h + (h + a) * (math.exp(h * ttm) - 1))
        _P = _A * np.exp(-_B * rf)
        return _P

    zcb[0] = price(n_years, rf)

    rates[0] = sr
    for steps in range(1, total_time_steps):
        prev_rate = rates[steps - 1]
        drift = a * (b - prev_rate) * dt
        shock[steps] = shock[steps] * np.sqrt(prev_rate)
        dr = drift + shock[steps]
        rates[steps] = abs(prev_rate + dr)
        zcb[steps] = price(n_years - steps * dt, rates[steps])
    rates_gbm_df = pd.DataFrame(data=conv_to_annualised_rate(rates), index=range(total_time_steps))
    zcb_gbm_df = pd.DataFrame(data=zcb, index=range(total_time_steps))
    zcb_rets = zcb_gbm_df.pct_change().dropna()
    return rates_gbm_df, zcb_gbm_df, zcb_rets


def get_btr(rates_gbm_df, n_years, steps_per_yr, tenor, cr, fv, n_scenarios):
    cb_df, mac_dur_df, bond_cf = get_bond_gbm(rates_gbm_df, n_years, steps_per_yr, tenor, cr, fv)
    mac_dur_df = mac_dur_df / steps_per_yr
    bond_ann_ret = get_bond_tr(cb_df, bond_cf, n_scenarios)
    return bond_ann_ret, cb_df, mac_dur_df


def reshape_disc_rate(n_years, steps_per_year, n_scenarios, disc_rate):
    rates_df = pd.DataFrame(data=disc_rate, index=range(0, (n_years*steps_per_year+1)), columns=range(0, n_scenarios))
    return rates_df


def get_bond_gbm(rates_gbm_df: pd.DataFrame, n_years, steps_per_yr, tenor=0, cr=0.05, fv=100):
    bond_cf = 0
    dt = 1/steps_per_yr
    total_time_steps = int(n_years*steps_per_yr)
    n_scenarios = len(rates_gbm_df.columns)
    cb = np.repeat(0.0, (total_time_steps) * n_scenarios).reshape(total_time_steps, n_scenarios)
    mac_dur = np.empty_like(cb)
    # CB prices
    for step in range(0, total_time_steps):
        ttm = total_time_steps - step
        disc_rate = rates_gbm_df.loc[step]
        disc_rate = pd.DataFrame(np.asarray(pd.concat([disc_rate] * ttm, axis=0)).reshape(ttm, n_scenarios))
        cb[step], mac_dur[step], temp = get_bond_prices(n_years - step * dt, tenor-step*dt, steps_per_yr, disc_rate, cr, fv)
        if step == 0:
            bond_cf = temp
    cb_df = pd.DataFrame(cb)
    mac_dur_df = pd.DataFrame(mac_dur)
    cb_df = cb_df.append(cb_df.iloc[-1] * (rates_gbm_df.iloc[-2] * dt + 1), ignore_index=True)
    return cb_df, mac_dur_df, bond_cf


def get_bond_tr(cb_df, bond_cf, n_scenarios):
    print()
    if not len(cb_df.index) - len(bond_cf) == 1:
        dr_steps = cb_df.shape[0]
        cf_steps = bond_cf.shape[0]
        shortfall = cf_steps - dr_steps + 2
        bond_cf.drop(bond_cf.tail(shortfall).index, inplace=True)
    else:
        bond_cf.drop(bond_cf.tail(1).index, inplace=True)
    bond_cf.index += 1
    concat_cf = pd.concat([bond_cf] * n_scenarios, axis=1)
    concat_cf.loc[0] = 0
    concat_cf.loc[len(concat_cf.index)] = 0
    tcf_df = (cb_df + concat_cf)
    tr_df = (np.divide(tcf_df, cb_df.shift()) - 1).dropna()
    # bond_ann_ret = get_ann_return(tr_df)
    return tr_df


def plot_cir():
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
                                    dcc.Slider(id='sl_ltrf', min=0.01, max=0.10, step=0.005, value=0.03),
                                    html.Label(id='out_ltrf')], style={'display': 'inline-block'}),
                           html.Div([html.Label(children='Select speed of MR: '),
                                    dcc.Slider(id='sl_speed', min=0.2, max=1, step=0.05, value=0.5),
                                    html.Label(id='out_speed')], style={'display': 'inline-block'}),
                           html.Div([html.Label(children='Select volatility: '),
                                     dcc.Slider(id='sl_vola', min=0, max=1, step=0.05, value=0.15),
                                     html.Label(id='out_vola')], style={'display': 'inline-block'}),
                           html.Button(id='sub_cir', children='SUBMIT', n_clicks=0, style={'display': 'inline-block'})], style= {'display': 'flex', 'justify-content': 'space-evenly'}),
                           html.Div([html.Div([html.Label(children='Select N-Periods: '),
                                               dcc.Slider(id='sl_periods', min=1, max=20, step=1, value=10),
                                               html.Label(id='out_periods')], style={'display': 'inline-block'}),
                                     html.Div([html.Label(children='Select steps_per_yr: '),
                                               dcc.Slider(id='sl_stperyr', min=1, max=10000, step=1, value=12),
                                               html.Label(id='out_stperyr')], style={'display': 'inline-block'}),
                                     html.Div([html.Label(children='Select N-Scenarios: '),
                                               dcc.Slider(id='sl_scenarios', min=2, max=250, step=1, value=10),
                                               html.Label(id='out_scenarios')], style={'display': 'inline-block'})
                                     ], style= {'display': 'flex', 'justify-content': 'space-evenly', 'padding-top': '25px'}),
                           html.Div([dcc.Graph(id='cir')]),
                           html.Div([dcc.Graph(id='hist_tfr')])])

    upd_label('out_av', 'sl_av')
    upd_label('out_rf', 'sl_rf')
    upd_label('out_ltrf', 'sl_ltrf')
    upd_label('out_speed', 'sl_speed')
    upd_label('out_vola', 'sl_vola')
    upd_label('out_periods', 'sl_periods')
    upd_label('out_stperyr', 'sl_stperyr')
    upd_label('out_scenarios', 'sl_scenarios')

    @app.callback([Output('cir', 'figure'),
                   Output('hist_tfr', 'figure')],
                  [Input('sub_cir', 'n_clicks')],
                  [State('sl_rf', 'value'),
                   State('sl_periods', 'value'),
                   State('sl_stperyr', 'value'),
                   State('sl_scenarios', 'value'),
                   State('sl_vola', 'value'),
                   State('sl_speed', 'value'),
                   State('sl_ltrf', 'value'),
                   State('sl_av', 'value')])
    def upd_cir(n_clicks, rf, n_years, steps_per_yr, n_scenarios, volatility, a, b, av):
        def get_scatter_points(df: pd.DataFrame):
            return df.aggregate(lambda scenario: go.Scatter(x=scenario.index, y=scenario)).tolist()
        tenor = n_years
        rates_gbm_df, zcb_gbm_df, zcb_rets = get_rates_gbm(rf, n_years, steps_per_yr, n_scenarios, volatility, a, b)
        liabilities = zcb_gbm_df  # Assuming same liab as that of ZCB
        cb_df, mac_dur_df, bond_cf = get_bond_gbm(rates_gbm_df,n_years=n_years, steps_per_yr=steps_per_yr, tenor=tenor)

        # Investments in ZCB at T0
        n_bonds = av/zcb_gbm_df.loc[0,0]
        av_zcb_df = n_bonds * zcb_gbm_df
        # fr_zcb = (av_zcb_df/liabilities).round(decimals=6)
        fr_zcb = get_funding_ratio(liabilities, av_zcb_df).round(decimals=6)
        fr_zcb_df = fr_zcb.pct_change().dropna()

        # Cash investments cumprod
        fd_rates = rates_gbm_df.apply(lambda x: x/steps_per_yr)
        av_cash_df = drawdown(fd_rates, retrive_index=True, init_wealth=av, is1p=False)
        # fr_cash = av_cash_df/liabilities
        fr_cash = get_funding_ratio(liabilities, av_cash_df)
        fr_cash_df = fr_cash.pct_change().dropna()

        fig = make_subplots(rows=4, cols=2, shared_xaxes=True,specs=[[{}, {}],
                                                                     [{}, {}],
                                                                     [{}, {}],
                                                                     [{}, {}]], subplot_titles=("CIR model of Interest rates","ZCB Prices based on CIR", "CB Prices based on CIR",
                                                                                                "CB Mac Dur", "Cash invested in FD with rolling maturity",
                                                                                                " {:.4f} ZCB investments at T=0".format(n_bonds),"Funding Ratio %ch-Cash",
                                                                                                "Funding Ratio %ch-ZCB"))
        rates_gbm = get_scatter_points(rates_gbm_df)
        zcb_gbm = get_scatter_points(zcb_gbm_df)
        cb_gbm = get_scatter_points(cb_df)
        cb_mac_dur_gbm = get_scatter_points(mac_dur_df)
        av_zcb_gbm = get_scatter_points(av_zcb_df)
        av_cash_gbm = get_scatter_points(av_cash_df)
        fr_cash_gbm = get_scatter_points(fr_cash_df)
        fr_zcb_gbm = get_scatter_points(fr_zcb_df)
        tfr_cash_hist = fr_cash.iloc[-1].tolist()
        tfr_zcb_hist = fr_zcb.iloc[-1].loc[0] # since all are same

        for rates_data in rates_gbm:
            fig.add_trace(rates_data, row=1, col=1)
        for zcb_price in zcb_gbm:
            fig.add_trace(zcb_price, row=1, col=2)
        for cb_price in cb_gbm:
            fig.add_trace(cb_price, row=2, col=1)
        for cb_mac_dur in cb_mac_dur_gbm:
            fig.add_trace(cb_mac_dur, row=2, col=2)
        for av_cash in av_cash_gbm:
            fig.add_trace(av_cash, row=3, col=1)
        for av_zcb in av_zcb_gbm:
            fig.add_trace(av_zcb, row=3, col=2)
        for fr_cash in fr_cash_gbm:
            fig.add_trace(fr_cash, row=4, col=1)
        for fr_zcb in fr_zcb_gbm:
            fig.add_trace(fr_zcb, row=4, col=2)

        b = conv_to_annualised_rate(b)
        mrl = [dict(type='line', xref='x1', yref='y1', x0=0, x1=n_years*steps_per_yr, y0=b, y1=b, name='Mean Reverting Level',
                    line=dict(dash='dashdot', width=5))]
        fig.update_xaxes(matches='x')
        fig.update_layout(showlegend=False,
                          height=1000,
                          hovermode='closest',
                          shapes=mrl)
        tfr_zcb = [dict(type='line', xref='x1', yref='paper', y0=0, y1=1, x0=tfr_zcb_hist, x1=tfr_zcb_hist, name='tfr-zcb',
                        line=dict(dash='dashdot', width=5))]
        tfr_cash_distplot = ff.create_distplot(hist_data=[tfr_cash_hist], group_labels=["tfr_cash"], show_hist=False)
        tfr_cash_distplot.update_layout(shapes=tfr_zcb, hovermode='closest')
        return fig, tfr_cash_distplot

    app.run_server()


def bt_mix(r1: pd.DataFrame, r2: pd.DataFrame, allocator, **kwargs):
    if not r1.shape == r2.shape:
        raise ValueError("Returns need to be of same shape")
    wt_r1 = allocator(r1, r2, **kwargs)
    if not wt_r1.shape == r1.shape:
        raise ValueError("Use a compatible allocator")
    return wt_r1*r1 + (1-wt_r1)*r2


def fixed_mix_allocator(r1: pd.DataFrame, r2: pd.DataFrame, wt_r1):
        return pd.DataFrame(data=wt_r1, index=r1.index, columns=r1.columns)


def glide_path_allocator(r1: pd.DataFrame, r2: pd.DataFrame, wt_start=0.8, wt_end=0.2):
    n_points = r1.shape[0]
    n_scenarios = r1.shape[1]
    wt_r1 = pd.Series(np.linspace(wt_start, wt_end, n_points))
    wt_r1 = pd.concat([wt_r1]*n_scenarios, axis=1)
    return wt_r1


def floor_allocator(r1: pd.DataFrame, r2: pd.DataFrame, floor, zcb_prices: pd.DataFrame, m=3, max_dd_mode=False):
    zcb_prices = zcb_prices.drop(index=0).reindex()
    if not r1.shape == r2.shape:
        raise ValueError("Non-Compatible rets dataframe")
    wt_r1 = pd.DataFrame().reindex_like(r1)
    total_time_steps, n_scenarios = r1.shape
    pf_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    for step in range(0, total_time_steps):
        if max_dd_mode:
            peak_value = np.maximum(peak_value, pf_value)
            floor_value = floor*peak_value
        else:
            floor_value = floor*zcb_prices.iloc[step]
        cushion = (pf_value - floor_value) / pf_value
        wt1 = (cushion*m).clip(0,1)
        pf_ret = wt1*r1.iloc[step] + (1-wt1)*r2.iloc[step]
        pf_value = (1+pf_ret)*pf_value
        wt_r1.iloc[step] = wt1
    return wt_r1


def distplot_terminal_paths(floor_factor,**kwargs):
    app = dash.Dash()
    terminal_paths = []
    pf_type = []
    stats = []
    for key, value in kwargs.items():
        pf_type.append(key)
        terminal_paths.append(value.tolist())
        stats.append(terminal_risk_stats(fv=1, floor_factor=floor_factor, wealth_index=value, aslst=True, strategy=key))
    stats = pd.DataFrame(data=stats, columns=["strategy", 'Exp_wealth', "Exp_Volatility", "Med_Wealth", "#_violations", "p_violations", "CVaR"])
    floor = [dict(type='line', xref='x1', yref='paper', y0=0, y1=1, x0=floor_factor, x1=floor_factor, name='floor',
                    line=dict(dash='dashdot', width=5))]
    fig = ff.create_distplot(hist_data=terminal_paths, group_labels=pf_type, show_hist=False)
    fig.update_layout(shapes=floor)
    app.layout = html.Div([html.Div([dcc.Graph(id='terminal_dist_plot', figure=fig)]),
                           html.Div([dt.DataTable(id='risk-stats',
                                                  columns=[{'name': col,
                                                            'id': col} for col in stats.columns],
                                                  data=stats.to_dict('records'))])
                           ])
    app.run_server()






