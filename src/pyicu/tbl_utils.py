import pandas as pd
from pyicu.utils_cli import stop_generic
from pyicu.assertions import obeys_interval

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

def meta_vars(x):
    return meta_vars.dispatch(x)

def meta_vars_id_tbl(x):
    return id_vars(x)

def meta_vars_ts_tbl(x):
    return id_vars(x) + index_var(x)

def meta_vars_win_tbl(x):
    return id_vars(x) + index_var(x) + dur_var(x)

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
