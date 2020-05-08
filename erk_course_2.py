import edhec_risk_kit as erk
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin

brka_d = pd.read_csv('data/brka_d_ret.csv', parse_dates=True, index_col=0)
brka_m = brka_d.resample('1M').apply(erk.cumulate)
brka_m.index -= MonthBegin(1)
fff = erk.get_df(filename='data/F-F_Research_Data_Factors_m.csv', start_period=None, end_period=None,
                 format='%Y%m', reqd_strategies=None)
rr_capm = erk.regress(brka_m, fff[['Mkt-RF', 'RF']], '1990', rfcol='RF')
rr_ffm = erk.regress(brka_m, fff, '1990', rfcol='RF')
rr_ffm_nint = erk.regress(brka_m, fff, '1990', rfcol='RF', intercept=False)
print(rr_ffm.summary())
print(rr_ffm_nint.summary())