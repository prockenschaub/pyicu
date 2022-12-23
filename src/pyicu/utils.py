from typing import Any, List, Iterable, Callable
import warnings
import numpy as np
import pandas as pd
import string
import random

from .interval import hours

def enlist(x: Any):
    # TODO: Test for scalar instead
    if x is None:
        return None
    elif not isinstance(x, list):
        return [x]
    else:
        return x

def enlist_none(x: Any):
    if x is None:
        x = []
    return enlist(x)

def coalesce(**kwargs):
    res = {}
    for k, v in kwargs.items():
        if v is not None:
            res[k] = v
    return res


def concat_tbls(objs: Iterable[pd.DataFrame], *args, **kwargs):
    # TODO: check that all of same type
    metavars = objs[0]._metadata
    metadata = {k: getattr(objs[0], k) for k in metavars}

    # Check that all tables share the same metadata
    for obj in objs:
        if obj._metadata != metavars:
            raise ValueError(
                f"expected all tables to have the following _metadata {metavars} but got {obj._metadata} instead."
            )
        for k, v in metadata.items():
            if getattr(obj, k) != v:
                raise ValueError(f"expected all tables to have the same value `{k}`={v} but got {getattr(obj, k)} instead.")

    # Do concatenation
    res = pd.concat(objs, *args, **kwargs)
    for k, v in metadata.items():
        setattr(res, k, v)
    return res


def print_list(x: List, max_char=75):
    repr = x.__repr__()
    return repr[:max_char] + ("...]" if len(repr) > max_char else "")


def intersect(x: List, y: List):
    return sorted(set(x) & set(y), key=x.index)


def union(x: List, y: List):
    return sorted(set(x) | set(y), key=(x + y).index)


def diff(x: List, y: List):
    return sorted(set(x) - set(y), key=x.index)


def new_names(
    old_names: List[str] | pd.DataFrame | None = None,
    n: int = 1,
    chars: str = string.ascii_letters + string.digits,
    length: int = 15,
) -> str | List[str]:
    if isinstance(old_names, pd.DataFrame):
        old_names = old_names.columns

    while True:
        res = ["".join(random.choice(chars) for _ in range(length)) for _ in range(n)]
        if len(res) == len(set(res)) and len(set(res) & set(old_names)) == 0:
            break

    if n == 1:
        res = res[0]
    return res


def prcnt(x: int | float, tot: int | float) -> str:
    return f"{np.round(x / tot * 100, decimals=2)}%"


def nrow(x: pd.DataFrame) -> int:
    return x.shape[0]


def ncol(x: pd.DataFrame) -> int:
    return x.shape[1]


def rm_na(x, cols: str | List[str] | None = None, mode: str = "all"):
    return x.dropna(how=mode, subset=cols, axis=0)


def rm_na_val_var(x: pd.DataFrame, col: str = "val_var") -> pd.DataFrame:
    n_row = nrow(x)
    x = rm_na(x, col)
    n_rm = n_row - nrow(x)

    if n_rm > 0:
        print(f"removed {n_rm} ({prcnt(n_rm, n_row)}) of rows due to missing values")
    return x


def expand(
    x: pd.DataFrame,
    start_var: str | None = None,
    end_var: str | None = None,
    step_size: pd.Timedelta | None = None,
    new_index: str | None = None,
    keep_vars: str | List[str] | None = None,
    aggregate: bool | str | Callable = False,
) -> pd.DataFrame:
    """Expand table to one row per `step_size` from `start_var` to `end_var`

    Note: An aspect to keep in mind when applying expand() to a win_tbl object 
        is that values simply are repeated for all time-steps that fall into a 
        given validity interval. This gives correct results when a win_tbl for 
        example contains data on infusions as rates, but might not lead to 
        correct results when infusions are represented as drug amounts 
        administered over a given time-span. In such a scenario it might be 
        desirable to evenly distribute the total amount over the corresponding 
        time steps (currently not implemented).

    Args:
        x: ts_tbl or win_tbl
        start_var: name of the column containing the start time. If None, use index variable if exists. Defaults to None.
        end_var: name of the column containing the end time. Must be specified for ts_tbl. If win_tbl and None, use duration variable. Defaults to None.
        step_size: size of the time steps to expand from start to end. If None, infer base interval of `x`. Defaults to None.
        new_index: TODO: add this feature
        keep_vars: which variables to retain in the output (id and index are always retained). Defaults to None.
        aggregate: whether and how rows with the same id and index should be aggregated. Defaults to False.

    Returns:
        ts_tbl
    """
    if x.icu.is_pandas() or x.icu.is_id_tbl():
        # TODO: also handle these
        raise NotImplementedError()

    if x.icu.is_ts_tbl() and end_var is None: 
        raise ValueError(f'`end_var` must be specified to expand a ts_tbl')

    if start_var is None:
        start_var = x.icu.index_var
    if end_var is None:
        end_var = new_names(x)
    if step_size is None: 
        warnings.warn(f'step size for expansion was not provided, using approximation')
        step_size = x.icu.interval
    
    if x.icu.is_ts_tbl():
        x = x.reset_index(level=1)
    elif x.icu.is_win_tbl():
        dur_var = x.icu.dur_var
        x = x.reset_index(level=[1, 2])
        x[end_var] = x[start_var] + x[dur_var]
        # TODO: deal with negative dur_var by setting it to zero

    steps = new_names(x)
    x[steps] = x.apply(lambda row: pd.timedelta_range(row[start_var], row[end_var], freq=step_size), axis=1)
    x = x.explode(steps)

    x.drop(columns=list(set(x.columns) - set(enlist(steps) + enlist_none(keep_vars))), inplace=True)
    x.rename(columns={steps: start_var}, inplace=True)
    x.set_index(start_var, append=True, inplace=True)

    if aggregate != False:
        x = x.groupby(level=[0, 1]).aggregate(aggregate)

    return x

def create_intervals(
    x: pd.DataFrame, 
    by_vars: str | List[str] | None = None,
    interval: pd.Timedelta | None = None, 
    overhang: pd.Timedelta = hours(1),
    max_len: pd.Timedelta = hours(6),
    dur_var: str = "duration"
) -> pd.DataFrame:
    """Creat time intervals between two subsequent measurements of the same group

    Note: this function differs slightly from the original function in ricu. 
          Some of the behaviour in ricu was obscure and not well documented, 
          so this alternative implementation was chosen. 
          TODO: open issue in ricu Github repo

    Args:
        x: ts_tbl
        by_vars: variables to group measurements by. If None, measurements are grouped by id, e.g., icustay. 
            Defaults to None.  
        interval: the base interval of table. If None, this will be approximated. Defaults to None. 
        overhang: value for the last time interval (which is undefined). Defaults to hours(1).
        max_len: maximum interval length. Defaults to hours(6).
        dur_var: name of the new duration column. Defaults to "duration".

    Returns:
        win_tbl
    """
    if not x.icu.is_ts_tbl():
        raise TypeError(f'expected `x` to be ts_tbl')

    if by_vars is None: 
        by_vars = x.icu.id_var
    by_vars = enlist(by_vars)

    id = x.icu.id_var
    ind = x.icu.index_var
    
    if interval is None:
        warnings.warn(f'base interval was not provided when creating intervals, using approximation')
        interval = x.icu.interval

    x = x.reset_index(level=1)
    x[dur_var] = x.groupby(id)[ind].shift(-1) - x[ind]
    x[dur_var] = x[dur_var].fillna(overhang) - interval
    x[dur_var] = x[dur_var].clip(lower=hours(0), upper=max_len-interval)

    return x.icu.as_win_tbl(index_var=ind, dur_var=dur_var)

def expand_intervals(x: pd.DataFrame, keep_vars: str | List[str] | None = None, grp_var: str | None = None, step_size: pd.Timedelta | None = None) -> pd.DataFrame:
    """Wrapper to first create a win_tbl using `create_intervals` and then `expand`

    Args:
        x: ts_tbl
        keep_vars: which variables to retain in the output (id and index are always retained). Defaults to None.
        grp_var: variables to group measurements by. If None, measurements are grouped by id, e.g., icustay. 
            Defaults to None.  
        step_size: size of the time steps to expand from start to end. If None, infer base interval of `x`. Defaults to None.

    Returns:
        ts_tbl
    """
    id = x.icu.id_var
    ind = x.icu.index_var
    dur = 'duration'
    grp = enlist(id) + enlist(grp_var)
    x = create_intervals(x, grp, overhang=hours(1), max_len=hours(6), dur_var=dur, interval=step_size)
    x = expand(x, ind, dur, keep_vars=enlist(id)+enlist_none(keep_vars), step_size=step_size)
    return x
