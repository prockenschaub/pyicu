from typing import Any, List, Iterable, Callable
import numpy as np
import pandas as pd
import string
import random

def enlist(x: Any):
    # TODO: Test for scalar instead
    if x is None:
        return None
    elif not isinstance(x, list):
        return [x]
    else:
        return x


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
    step_size: pd.Timedelta = None,
    new_index: str | None = None,
    keep_vars: List[str] | None = None,
    aggregate: bool | str | Callable = False,
) -> pd.DataFrame:
    if x.icu.is_pandas() or x.icu.is_id_tbl() or x.icu.is_ts_tbl():
        # TODO: also handle these
        raise NotImplementedError()

    if start_var is None:
        start_var = x.icu.index_var
    if end_var is None:
        end_var = new_names(x)

    if x.icu.is_win_tbl():
        dur_var = x.icu.dur_var
        x = x.reset_index(level=[1, 2])

        if end_var not in x.columns:
            x[end_var] = x[start_var] + x[dur_var]
            # TODO: deal with negative dur_var by setting it to zero

    steps = new_names(x)
    x[steps] = x.apply(lambda row: pd.timedelta_range(row[start_var], row[end_var], freq=step_size), axis=1)
    x = x.explode(steps)

    x.drop(columns=[start_var, end_var, dur_var], inplace=True)
    x.rename(columns={steps: start_var}, inplace=True)
    x.set_index(start_var, append=True, inplace=True)

    if aggregate != False:
        x = x.groupby(level=[0, 1]).aggregate(aggregate)

    return x

# Transformed from ricu (may contain bugs)
def fill_gaps(x, limits=None, start_var="start", end_var="end"):
    assert x.is_unique

    if pd.api.types.is_timedelta64_dtype(limits):
        assert len(limits) == 2
        limits = x.groupby(x.columns.difference([start_var, end_var])).first().reset_index()
        limits[start_var], limits[end_var] = limits.iloc[:, -2], limits.iloc[:, -1]

    if isinstance(limits, pd.DataFrame):
        id_vars = limits.columns.tolist()
    else:
        id_vars = x.columns.tolist()

    join = pd.DataFrame()
    join[start_var] = limits[start_var]
    join[end_var] = limits[end_var]
    join["step_size"] = pd.Series([pd.Timedelta(x).total_seconds() for x in x.time_step()], dtype="timedelta64[s]")
    join["new_index"] = x.index_var()
    join = join.explode("new_index").reset_index(drop=True)

    return x.merge(join.drop_duplicates(), on=id_vars)

# Transformed from ricu (may contain bugs)
def slide(x, expr, before, after=pd.Timedelta(hours=0), **kwargs):
    assert pd.api.types.is_scalar(before)
    assert pd.api.types.is_timedelta64_dtype(before)
    assert pd.api.types.is_scalar(after)
    assert pd.api.types.is_timedelta64_dtype(after)

    id_cols = x.id_vars()
    ind_col = x.index_var()
    interva = x.interval()
    time_ut = x.time_unit()

    before = pd.Timedelta(before, unit=time_ut)
    after = pd.Timedelta(after, unit=time_ut)

    join = x.copy()
    join["min_time"] = join[ind_col] - before
    join["max_time"] = join[ind_col] + after
    join = join[id_cols + ["min_time", "max_time"]]

    if before == pd.Timedelta(0):
        msg = None
    else:
        msg = "Using 'by reference' syntax in expressions passed to `slide()` " \
              "might yield undesired results if `before` > 0."

    res = x.hopper(expr, join, lwr_col="min_time", upr_col="max_time",
                   warn_msg=msg, **kwargs)

    if not isinstance(res, pd.DataFrame):
        res[ind_col] = res["min_time"] + before
        res = res.as_ts_tbl(index_var=ind_col, interval=interva, by_ref=True)

    res = res.rm_cols(["min_time", "max_time"], skip_absent=True, by_ref=True)

    return res