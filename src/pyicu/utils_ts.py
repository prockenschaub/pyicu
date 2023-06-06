import pandas as pd
from datetime import timedelta
from pyicu.assertions import is_unique, is_scalar, has_col, is_difftime, has_length
from pyicu.container.table import rm_cols, TableAccessor
from pyicu.tbl_utils import id_vars, index_var

def time_unit(x):
    assert isinstance(x, pd.DataFrame), "x must be a pandas DataFrame"
    
    freq = x.index.freq
    if freq is not None:
        return freq.freqstr
    else:
        return None

# Transformed from ricu (may contain bugs)
def slide_index(x, expr, index, before, after=timedelta(hours=0), **kwargs):

    assert is_difftime(index)
    assert has_length(index)
    assert is_unique(index)
    assert is_scalar(before)
    assert is_difftime(before)
    assert is_scalar(after)
    assert is_difftime(after)

    id_cols = id_vars(x)
    ind_col = index_var(x)
    interva = TableAccessor.interval(x)
    time_ut = time_unit(x)

    before.units = time_ut # may be incorrect
    after.units = time_ut # may be incorrect

    join = x.groupby(id_cols).apply(lambda group: group.assign(min_time=group[index] - before, max_time=group[index] + after))

    res = hopper(x, expr, join, lwr_col="min_time", upr_col="max_time", warn_msg="Using 'by reference' syntax in expressions passed to `slide_index()` most likely will not yield the desired results.", **kwargs)

    if not TableAccessor.is_ts_tbl(res):
        res[ind_col] = res["min_time"] + before
        res = TableAccessor.as_ts_tbl(res, index_var=ind_col, interval=interva, by_ref=True)

    res = rm_cols(res, ["min_time", "max_time"], skip_absent=True, by_ref=True)

    return res


# Transformed from ricu (may contain bugs)
def hop(x, expr, windows, full_window=False, lwr_col="min_time", upr_col="max_time",
        left_closed=True, right_closed=True, eval_env=None, **kwargs):

    msg = "Using 'by reference' syntax in expressions passed to `hop()` most likely will not yield the desired results."
    return hopper(x, expr, windows, full_window, lwr_col, upr_col, left_closed, right_closed, eval_env, msg, **kwargs)


# Transformed from ricu (may contain bugs)
def hopper(x, expr, windows, full_window=False, lwr_col="min_time", upr_col="max_time",
           left_closed=True, right_closed=True, eval_env=None, warn_msg=False, **kwargs):
    
    assert isinstance(x, pd.DataFrame), "x must be a pandas DataFrame"
    assert isinstance(windows, pd.DataFrame), "windows must be a pandas DataFrame"
    
    # Helper function to apply units to columns
    def apply_units(df, cols, units):
        for col in cols:
            df[col] = df[col].astype(f"timedelta64[{units}]")
        return df
    
    # Check conditions
    assert has_col(x, lwr_col) and has_col(x, upr_col), f"{lwr_col} or {upr_col} column is missing in x"
    assert is_unique(x, lwr_col) and is_unique(x, upr_col), f"{lwr_col} or {upr_col} column is not unique in x"
    assert isinstance(full_window, bool), "full_window must be a boolean value"
    
    if x.empty:
        return x
    
    win_id = windows.columns.tolist()
    tbl_id = x.columns.tolist()
    
    orig_unit = x.dtypes[tbl_id.index("time")].unit
    
    win_cols = [lwr_col, upr_col]
    
    windows = apply_units(windows, win_cols, orig_unit)
    
    tbl_ind = "time"  # Assuming the time index column in x is named "time"
    
    if full_window:
        extremes = x.groupby(tbl_id)[tbl_ind].agg(grp_min="min", grp_max="max").reset_index()
        join = [f"{tbl_id[i]}=={win_id[i]}" for i in range(len(tbl_id))] + [f"grp_min<={lwr_col}", f"grp_max>={upr_col}"]
        windows = pd.merge(windows, extremes, left_on=join, right_on=join, how="inner")
        windows.rename(columns={win_id[i]: tbl_id[i] for i in range(len(tbl_id))}, inplace=True)
    
    tmp_col = [f"tmp_{i}" for i in range(2)]
    x[tmp_col] = x[tbl_ind].values.reshape(-1, 1)
    
    def cleanup():
        x.drop(tmp_col, axis=1, inplace=True)
    
    cleanup()  # Register cleanup function to be called on exit
    
    join = [f"{tbl_id[i]}=={win_id[i]}" for i in range(len(tbl_id))]
    join += [f"{tmp_col[1]}<={upr_col}" if right_closed else f"{tmp_col[1]}<{upr_col}",
             f"{tmp_col[0]}>={lwr_col}" if left_closed else f"{tmp_col[0]}>lwr_col"]
    
    if isinstance(expr, pd.core.computation.ops.QuotedExpr):
        env = expr.env
        exp = expr.expr
    else:
        env = eval_env if eval_env is not None else pd.eval._DEFAULT_EVAL_ENV
        exp = expr
    
    if warn_msg is not None and exp[0] == ":=":
        # Raise a warning message here
        pass
    
    res = pd.merge(x, windows, left_on=join, right_on=join, how="inner")
    res = res[exp]
    
    res.rename(columns={win_cols[i]: tmp_col[i] for i in range(len(win_cols))}, inplace=True)
    
    return res