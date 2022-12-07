import numpy as np
import pandas as pd
from ..container.time import days

def mimic_age(x: pd.Series, decimals=2) -> pd.Series:
    return np.round(-x.tm.change_interval(days(1)).astype(int) / 365.25, decimals)
