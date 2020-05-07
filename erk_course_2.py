import edhec_risk_kit as erk
import numpy as np
import pandas as pd

brka_d = pd.read_csv('data/brka_d_ret.csv', parse_dates=True, index_col=0)
brka_m = brka_d.resample('1M').apply(erk.cumulate)
fff = erk.get_df(filename='data/F-F_Research_Data_Factors_m.csv', start_period=None, end_period=None,
                 format='%Y%m', reqd_strategies=None)
print()