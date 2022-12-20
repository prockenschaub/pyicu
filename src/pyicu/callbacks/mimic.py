import numpy as np
import pandas as pd

from ..interval import days, minutes
from .item import calc_dur

def mimic_age(x: pd.Series, decimals=2) -> pd.Series:
    return np.round(-x.tm.change_interval(days(1)).astype(int) / 365.25, decimals)


def mimic_abx_presc(x, val_var, **kwargs):
    idx = x.icu.index_var
    x = x.reset_index(level=1)
    x[idx] = x[idx] + minutes(720)  # TODO: determine why 720 is chosen? to move to midday?
    x = x.icu.set_index_var(idx)
    x[val_var] = True
    return x


def mimic_dur_incv(x: pd.DataFrame, val_var: str, grp_var: str, **kwargs):
    """Calculate duration of medications in carevue"""
    ind = x.icu.index_var
    return calc_dur(x, val_var, ind, ind, grp_var)

def mimic_dur_inmv(x: pd.DataFrame, val_var: str, grp_var: str, stop_var: str, **kwargs):
    """Calculate duration of medications in metavision"""
    ind = x.icu.index_var
    return calc_dur(x, val_var, ind, stop_var, grp_var)