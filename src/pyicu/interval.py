import pandas as pd


def days(x):
    return pd.Timedelta(x, "d")


def hours(x):
    return pd.Timedelta(x, "h")


def minutes(x):
    return pd.Timedelta(x, "m")


def seconds(x):
    return pd.Timedelta(x, "s")


def change_interval(x: pd.Series, new_interval: pd.Timedelta = hours(1)):
    """Change the interval base unit
    Args:
        x: time
        new_freq: new frequency, e.g., 1 (hour) or 5 (minutes)
        new_unit: new time unit in which the frequency is counted. Can be one of
            "d" (day), "h" (hour), "m" (minutes), "s" (seconds. Defaults to "h".
    Returns:
        time rounded new frequency
    """
    return (x // new_interval) * new_interval


def interval(x: pd.Timedelta | str | None = None) -> pd.Timedelta:
    if isinstance(x, pd.Timedelta):
        return x
    elif isinstance(x, str):
        return pd.Timedelta(x)
    elif x is None: 
        return hours(1)
    else:
        raise ValueError(f"don't know how to create an interval from {x}")
        