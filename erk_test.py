import edhec_risk_kit as erk
import pandas as pd
# risk_plot = ['ann_ret']

# for mcw based pf
# reqd_strategies = erk.get_df_columns('data/ind30_m_vw_rets.csv')
# df = erk.plot_corr_mktret('data/ind30_m_vw_rets.csv','data/ind30_m_nfirms.csv', 'data/ind30_m_size.csv', start_period='2007', end_period='2018', format='%Y%m', reqd_strategies=reqd_strategies,
#                           retrieve_mcw=True)

# for indiv pf
# reqd_strategies1 = ['Steel', 'Fin', 'Beer']
# df1 = erk.get_df('data/ind30_m_vw_rets.csv', start_period='2007', end_period='2018', format='%Y%m', reqd_strategies=reqd_strategies1)


# ann_ret_act = erk.get_ann_return(df)
# ann_ret_exp = pd.Series([0.11, 0.13])
# cov = erk.get_cov(df)
# erk.plot_eff_frontier(ann_ret_act, cov, 25, reqd_strategies,rf=0.02, show_eq=True, show_gmv=True)
# print(ann_ret_act)
# erk.plot(df, mode='drawdowns', reqd_strategies=reqd_strategies, risk_plot=risk_plot, poi='Fin')





# riskdf1 = erk.risk_info(df, risk_plot=['ann_ret'], rf=0.03, alpha=5, var_method='cornish')
# btr = erk.cipp_algo(df1, 3, 0.75, reqd_strategies1, poi='Fin', alpha=5, var_method='cornish', rf=0.03, max_draw_mode=True, plot=False)
# df1 = btr['cppi_return']
# erk.plot(df1, mode='risk_stats', reqd_strategies=reqd_strategies1, risk_plot=risk_plot, poi='Fin', var_method='cornish',alpha=5, rf=0.03)
#
# print(btr)

# erk.plot(df, mode='drawdowns', reqd_strategies=reqd_strategies, risk_plot=risk_plot, poi='Fin', var_method='cornish', alpha=5, rf=0.03)


#gbm
erk.gbm()