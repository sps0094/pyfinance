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
# erk.plot_gbm()
# print(erk.get_bond_tr(0,0,0))
# liab, mac_dur = erk.get_present_value(pd.Series(data=[100000, 100000], index=[10,12]), disc_rate=0.04)
# b1, mac_dur1 = erk.get_bond_prices(10, 1, 0.04, 0.05, 1000)
# b1, mac_dur2 = erk.get_bond_prices(20, 1, 0.04, 0.05, 1000)
# print()
# liab = pd.Series(data=[100000, 100000], index=[10, 12])
# erk.get_duration_matched_pf(liab, [10,20], [1,1], 0.04, [0.05, 0.05], [1000, 1000], 130000, fr_change_sim=True)

bond_ret_10, cb_10 = erk.get_rates_gbm(rf=0.03,
                  n_years=10,
                  steps_per_yr=12,
                  n_scenarios=10,
                  volatility=0.15,
                  a=0.5,
                  b=0.03,
                  tenor=20,
                  cr=0.05,
                  fv=100,
                  ann_ret=True)
bond_ret_20, cb_20 = erk.get_rates_gbm(rf=0.03,
                  n_years=10,
                  steps_per_yr=12,
                  n_scenarios=10,
                  volatility=0.15,
                  a=0.5,
                  b=0.03,
                  tenor=20,
                  cr=0.05,
                  fv=100,
                  ann_ret=True)
bond_pf = 0.6*bond_ret_10 + 0.4*bond_ret_20
st_ret = erk.gbm_stock(s0=100,
                       n_scenarios=10,
                       steps_per_yr=12,
                       n_years=10,
                       er=0.07,
                       vol=0.15,
                       floor=0.8,
                       multiplier=3,
                       rf=0.03,
                       cppi=True,
                       ann_ret=True)
pf = 0.6*st_ret + 0.4*bond_pf
print()
