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
erk.plot_gbm()
# print(erk.get_bond_tr(0,0,0))
# liab, mac_dur = erk.get_present_value(pd.Series(data=[100000, 100000], index=[10,12]), disc_rate=0.04)
# b1, mac_dur1 = erk.get_bond_prices(10, 1, 0.04, 0.05, 1000)
# b1, mac_dur2 = erk.get_bond_prices(20, 1, 0.04, 0.05, 1000)
# print()
# liab = pd.Series(data=[100000, 100000], index=[10, 12])
# results = erk.get_duration_matched_pf(liab, [10,20], [1,1], 0.04, [0.05, 0.05], [1000, 1000], 130000, fr_change_sim=False)
# print()
# rates_gbm, zcb_gbm, zcb_rets = erk.get_rates_gbm(rf=0.03,
#                                        n_years=10,
#                                        steps_per_yr=12,
#                                        n_scenarios=100,
#                                        volatility=0.02,
#                                        a=0.5,
#                                        b=0.03)
#
# bond_ret_10, cb_10 = erk.get_btr(rates_gbm_df=rates_gbm,
#                                  n_years=10,
#                                  steps_per_yr=12,
#                                  tenor=10,
#                                  cr=0.05,
#                                  fv=100,
#                                  n_scenarios=100)
# bond_ret_30, cb_30 = erk.get_btr(rates_gbm_df=rates_gbm,
#                                  n_years=10,
#                                  steps_per_yr=12,
#                                  tenor=30,
#                                  cr=0.05,
#                                  fv=100,
#                                  n_scenarios=100)
# st_ret = erk.gbm_stock(s0=100,
#                        n_scenarios=100,
#                        steps_per_yr=12,
#                        n_years=10,
#                        er=0.07,
#                        vol=0.15,
#                        floor=0.8,
#                        multiplier=3,
#                        rf=0.03,
#                        cppi=True,
#                        ret_series=True)
# #cash return - assets with lowest possible duration
# cash_rets = (1+0.02)**(1/12) - 1
# cash_rets_df = pd.DataFrame().reindex_like(st_ret)
# cash_rets_df.loc[:] = cash_rets
#
# lhp_bonds = erk.bt_mix(bond_ret_10, bond_ret_30, erk.fixed_mix_allocator, wt_r1=0.6)
# psp_7030 = erk.bt_mix(st_ret, lhp_bonds, erk.fixed_mix_allocator, wt_r1=0.7)
# psp_floor = erk.bt_mix(st_ret, lhp_bonds, erk.floor_allocator, floor=0.75, zcb_prices=zcb_gbm, m=3)
# psp_maxdd = erk.bt_mix(st_ret, cash_rets_df, erk.floor_allocator, floor=0.75, zcb_prices=zcb_gbm, m=3, max_dd_mode=True)
# psp_7030z = erk.bt_mix(st_ret, zcb_rets, erk.fixed_mix_allocator, wt_r1=0.7)
# g_8020 = erk.bt_mix(st_ret, lhp_bonds, erk.glide_path_allocator, wt_start=0.8, wt_end=0.2)
# psp_eq = st_ret
#
# strategies = [lhp_bonds, psp_7030, psp_7030z, psp_floor, psp_maxdd, psp_eq, g_8020]
# all_str_stats = pd.DataFrame()
# for strategy in strategies:
#     stats = erk.risk_info(strategy, risk_plot=['Drawdown'], rf=0.05, alpha=0.05).mean(axis=1)
#     all_str_stats = all_str_stats.append(stats.transpose(), ignore_index=True)
#
# t_lhp_bonds = erk.get_terminal_wealth(lhp_bonds)
# t_psp_7030 = erk.get_terminal_wealth(psp_7030)
# t_psp_floor = erk.get_terminal_wealth(psp_floor)
# t_psp_maxdd = erk.get_terminal_wealth(psp_maxdd)
# t_psp_7030z = erk.get_terminal_wealth(psp_7030z)
# t_g_8020 = erk.get_terminal_wealth(g_8020)
# t_psp_eq = erk.get_terminal_wealth(psp_eq)
#
# erk.distplot_terminal_paths(floor_factor=0.75, psp_7030=t_psp_7030, psp_eq=t_psp_eq, g_8020=t_g_8020, psp_7030z=t_psp_7030z, psp_floor=t_psp_floor, psp_maxdd=t_psp_maxdd)
