import edhec_risk_kit as erk

rates_15 = erk.reshape_disc_rate(15, 2, 1, 0.05)
b1, cb1 = erk.get_btr(rates_gbm_df=rates_15,
                                 n_years=15,
                                 steps_per_yr=2,
                                 tenor=15,
                                 cr=0.05,
                                 fv=1000,
                                 n_scenarios=1)

rates_5 = erk.reshape_disc_rate(5, 4, 1, 0.05)
b2, cb2 = erk.get_btr(rates_gbm_df=rates_5,
                                 n_years=5,
                                 steps_per_yr=4,
                                 tenor=5,
                                 cr=0.06,
                                 fv=1000,
                                 n_scenarios=1)

rates_10 = erk.reshape_disc_rate(10, 1, 1, 0.05)
b3, cb3 = erk.get_btr(rates_gbm_df=rates_10,
                                 n_years=10,
                                 steps_per_yr=1,
                                 tenor=10,
                                 cr=0.00,
                                 fv=1000,
                                 n_scenarios=1)

print()