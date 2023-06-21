"""
ICU class meta data utilities

The two data classes `id_tbl` and `ts_tbl`, are used to represent ICU
patient data, consist of a DataFrame alongside some meta data. This
includes marking columns that have special meaning and for data
representing measurements ordered in time, the step size. The following
utility functions can be used to extract columns and column names with
special meaning, as well as query a `ts_tbl` object regarding its time
series related meta data.

@details
The following functions can be used to query an object for columns or
column names that represent a distinct aspect of the data:

Methods:
    - id_vars(x): Extracts ID variables from an object
    - id_var(x): Extracts a single ID variable from an object
    - id_col(x): Extracts the column with the ID designation from an object
    - index_var(x): Extracts the index variable from a ts_tbl object
    - index_col(x): Extracts the column with the index designation from a ts_tbl object
    - dur_var(x): Returns the name of the column encoding the data validity interval for win_tbl objects
    - dur_col(x): Returns the difftime vector corresponding to the dur_var for win_tbl objects
    - meta_vars(x): Returns the meta variables (ID and index variables) for ts_tbl objects or ID variables for id_tbl objects
    - data_vars(x): Returns the data variables (columns that are not meta variables) for an object
    - data_var(x): Returns the name of a single data variable or fails if multiple data columns exist
    - data_col(x): Returns the column vector of a single data variable or throws an error if multiple data columns exist
    - time_vars(x): Returns the time variables (columns of type difftime) for an object
    - interval(x): Returns the time series interval length as a scalar-valued difftime object
    - time_unit(x): Returns the time unit of the time series interval as a string
    - time_step(x): Returns the time series step size as a numeric value in the unit returned by time_unit()
"""
import pandas as pd
from sympy import Interval
from pyicu.utils_cli import stop_generic
from pyicu.assertions import has_cols, obeys_interval
from pyicu.utils_cli import warn_dots
from functools import singledispatch

def id_vars(x):
    return id_vars.dispatch(x)

def id_vars_id_tbl(x):
    return x.attr("id_vars")

def id_vars_default(x):
    return stop_generic(x, ".Generic")

def id_var(x):
    res = id_vars(x)
    assert isinstance(res, str)
    return res

def id_col(x):
    return x[id_var(x)]

def index_var(x):
    return index_var.dispatch(x)

def index_var_ts_tbl(x):
    return x.attr("index_var")

def index_var_default(x):
    return stop_generic(x, ".Generic")

def index_col(x):
    return x[index_var(x)]

def dur_var(x):
    return dur_var.dispatch(x)

def dur_var_win_tbl(x):
    return x.attr("dur_var")

def dur_var_default(x):
    return stop_generic(x, ".Generic")

def dur_col(x):
    return x[dur_var(x)]

def dur_unit(x):
    return dur_col(x).units # Changed from units(dur_col(x))

@singledispatch
def meta_vars(x):
    pass
    #return meta_vars.dispatch(x)

@meta_vars.register(pd.DataFrame)
def meta_vars_id_tbl(x):
    return id_vars(x)

@meta_vars.register(pd.DataFrame)
def meta_vars_ts_tbl(x):
    return id_vars(x) + index_var(x)

@meta_vars.register(pd.DataFrame)
def meta_vars_win_tbl(x):
    return id_vars(x) + index_var(x) + dur_var(x)

@meta_vars.register(pd.DataFrame)
def meta_vars_default(x):
    return stop_generic(x, ".Generic")

def data_vars(x):
    return x.columns.difference(meta_vars(x))

def data_var(x):
    res = data_vars(x)
    assert isinstance(res, str)
    return res

def data_col(x):
    return x[data_var(x)]


def interval(x):
    return interval.dispatch(x)

def interval_ts_tbl(x):
    return x.interval

def interval_default(x):
    raise NotImplementedError("interval.default is not implemented.")

def interval_difftime(x):
    dif = [y-x for x, y in zip(x[:-1], x[1:])]
    res = min(filter(lambda d: d > 0, dif), default=None)
    
    if res is None:
        raise ValueError("Unable to find a positive interval.")
    
    assert obeys_interval(x, res)
    
    return res

#def stop_generic(x, generic):
#    raise NotImplementedError("Generic method not implemented.")


   
def rename_cols(x, new, old=None, skip_absent=False, by_ref=False, **kwargs):
    if callable(new):
        new = new(old, **kwargs)
    else:
        warn_dots(**kwargs)
    
    assert pd.Series(new).is_unique and pd.Series(old).is_unique and len(new) == len(old) \
        and isinstance(skip_absent, bool) and isinstance(by_ref, bool) \
        and pd.Series(rename(list(x.columns), new, old)).is_unique
    
    if set(new) == set(old):
        return x.copy() if by_ref else x
    
    return col_renamer(x, new, old, skip_absent, by_ref)

# Transformed from ricu (may contain bugs)
def col_renamer(x, new, old=None, skip_absent=False, by_ref=False):
    if isinstance(x, pd.core.window.Window):
        return col_renamer_win_tbl(x, new, old, skip_absent, by_ref)
    elif isinstance(x, pd.core.window.TimeWindow):
        return col_renamer_ts_tbl(x, new, old, skip_absent, by_ref)
    elif isinstance(x, pd.core.groupby.DataFrameGroupBy):
        return col_renamer_id_tbl(x, new, old, skip_absent, by_ref)
    elif isinstance(x, pd.DataFrame):
        return col_renamer_data_table(x, new, old, skip_absent, by_ref)
    else:
        return col_renamer_default(x)

def col_renamer_win_tbl(x, new, old=None, skip_absent=False, by_ref=False):
    old_dur = dur_var(x)

    if old_dur in old:
        new_dur = [new[i] for i, old_name in enumerate(old) if old_name == old_dur]

        if not by_ref:
            x = x.copy()
            by_ref = True

        x.dur_var = unname(new_dur)

    return col_renamer_ts_tbl(x, new, old, skip_absent, by_ref)

def col_renamer_ts_tbl(x, new, old=None, skip_absent=False, by_ref=False):
    old_ind = index_var(x)
    intval = Interval(x)

    if old_ind in old:
        new_ind = [new[i] for i, old_name in enumerate(old) if old_name == old_ind]

        if not by_ref:
            x = x.copy()
            by_ref = True

        x.index_var = unname(new_ind)

    return col_renamer_id_tbl(x, new, old, skip_absent, by_ref)

def col_renamer_id_tbl(x, new, old=None, skip_absent=False, by_ref=False):
    if skip_absent:
        hits = [old_name in x.columns for old_name in old]
        if sum(hits) == 0:
            return x

        new = [new[i] for i, hit in enumerate(hits) if hit]
        old = [old_name for old_name, hit in zip(old, hits) if hit]

    old_id = id_vars(x)

    if any(old_id in old):
        new_id = rename(old_id, new, old)

        if not by_ref:
            x = x.copy()
            by_ref = True

        x.id_vars = unname(new_id)

    return col_renamer_data_table(x, new, old, skip_absent, by_ref)

def col_renamer_data_table(x, new, old=None, skip_absent=False, by_ref=False):
    if not skip_absent:
        assert has_cols(x, old)

    if not by_ref:
        x = x.copy()

    x.rename(columns=dict(zip(old, new)), inplace=True)

    check_valid(x)

# TODO: Implement this function
def col_renamer_default(x):
    raise NotImplementedError

# TODO: Implement this function
def unname(lst):
    return [item for sublist in lst for item in sublist]

# TODO: Implement this function
def rename(lst, new, old):
    return [new if name == old else name for name in lst]

# TODO: Implement this function
def check_valid(x):
    # Placeholder function, add the appropriate checks for validity of x
    pass