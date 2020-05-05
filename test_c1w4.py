import edhec_risk_kit as erk
import pandas as pd

# rates_15 = erk.reshape_disc_rate(15, 2, 1, 0.05)
# b1, cb1, d1 = erk.get_btr(rates_gbm_df=rates_15,
#                                  n_years=15,
#                                  steps_per_yr=2,
#                                  tenor=15,
#                                  cr=0.05,
#                                  fv=1000,
#                                  n_scenarios=1)
#
# rates_5 = erk.reshape_disc_rate(5, 4, 1, 0.05)
# b2, cb2, d2 = erk.get_btr(rates_gbm_df=rates_5,
#                                  n_years=5,
#                                  steps_per_yr=4,
#                                  tenor=5,
#                                  cr=0.06,
#                                  fv=1000,
#                                  n_scenarios=1)
#
# rates_10 = erk.reshape_disc_rate(10, 1, 1, 0.05)
# b3, cb3, d3 = erk.get_btr(rates_gbm_df=rates_10,
#                                  n_years=10,
#                                  steps_per_yr=1,
#                                  tenor=10,
#                                  cr=0.00,
#                                  fv=1000,
#                                  n_scenarios=1)
liabilities = pd.Series(data=[100000, 200000, 300000], index=[3, 5, 10])
pv_liabilities, mac_dur_liabilities = erk.get_present_value(liabilities, 0.05)
results_b12 = erk.get_duration_matched_pf(liabilities, [15,5], [2,4], 0.05, [0.05,0.06], [1000,1000], 1000, False)
results_b23 = erk.get_duration_matched_pf(liabilities, [5,10], [4,1], 0.05, [0.06,0.00], [1000,1000], 1000, False)
results_b31 = erk.get_duration_matched_pf(liabilities, [10,15], [1,2], 0.05, [0.00,0.05], [1000,1000], 1000, False)
print()