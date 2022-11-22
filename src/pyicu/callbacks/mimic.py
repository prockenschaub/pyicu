import pandas as pd


def mimic_age(x: pd.Series) -> pd.Series:
    return -x.dt.total_seconds() / 60 / 60 / 24 / 365
