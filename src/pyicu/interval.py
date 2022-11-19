import pandas as pd


def days(x):
    return pd.Timedelta(x, "d")


def hours(x):
    return pd.Timedelta(x, "h")


def mins(x):
    return pd.Timedelta(x, "m")


def secs(x):
    return pd.Timedelta(x, "s")


def change_interval(x: pd.Timedelta, new_freq: int, new_unit: str = "h"):
    return (x // new_freq).floor(new_unit) * new_freq