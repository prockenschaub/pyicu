import numpy as np
import pandas as pd
from ..container.time import TimeArray, days, minutes

def mimic_age(x: pd.Series, decimals=2) -> pd.Series:
    return np.round(-x.tm.change_interval(days(1)).astype(int) / 365.25, decimals)

def mimic_abx_presc(x, val_var, **kwargs):
    idx = x.tbl.index_var
    x = x.reset_index(level=1)
    x[idx] = x[idx] + TimeArray(np.array([720]), minutes(1)) # TODO: determine why 720 is chosen? to move to midday? 
    x = x.tbl.set_index_var(idx)
    x[val_var] = True
    return x
