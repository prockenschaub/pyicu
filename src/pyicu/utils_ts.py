'''
Time series utility functions
==============================

ICU data as handled by `ricu` is mostly comprised of time series data and as
such, several utility functions are available for working with time series
data in addition to a class dedicated to representing time series data (see
`ts_tbl()`). Some terminology to begin with: a time series is considered
to have gaps if, per (combination of) ID variable value(s), some time steps
are missing. Expanding and collapsing mean to change between
representations where time steps are explicit or encoded as interval with
start and end times. For sliding window-type operations, `slide()` means to
iterate over time-windows, `slide_index()` means to iterate over certain
time-windows, selected relative to the index, and `hop()` means to iterate
over time-windows selected in absolute terms.

Attributes:
    None

Methods:
    has_gaps(): Check if a `ts_tbl` object has gaps. #### not implemented
    is_regular(): Check if a `ts_tbl` object is regular. #### not implemented
    fill_gaps(): Fill missing time steps in a `ts_tbl` object with `NA` values.
    remove_gaps(): Remove time steps consisting of `NA` values in a `ts_tbl` object. #### not implemented
    expand(): Expand a `ts_tbl` object where time steps are encoded as intervals.
    collapse(): Collapse a `ts_tbl` object where time steps are explicit. #### not implemented
    slide(): Perform sliding-window operations on a `ts_tbl` object.
    slide_index(): Perform sliding-window operations on specific time-windows.
    hop(): Perform sliding-window operations on absolute time-windows.
    
'''
import pandas as pd
from datetime import timedelta
from pyicu.assertions import is_unique, is_scalar, has_col, is_difftime, has_length
from pyicu.container.table import TableAccessor
from pyicu.tbl_utils import id_vars, index_var, rm_cols
from pyicu.utils import new_names
from typing import List, Callable

def time_unit(x):
    assert isinstance(x, pd.DataFrame), "x must be a pandas DataFrame"
    
    freq = x.index.freq
    if freq is not None:
        return freq.freqstr
    else:
        return None
    
def slide(x, expr, before, after=pd.Timedelta(hours=0), **kwargs):
    '''
    Perform sliding window-type operations over a time series.

    Args:
        x: A `ts_tbl` object representing the time series.
        expr: Expression to be evaluated per window.
        before: The time duration before each time step to include in the window.
        after: The time duration after each time step to include in the window. Default is 0.
        **kwargs: Additional arguments for sliding window operations.

    Returns:
        A `ts_tbl` object with sliding window operations over the time series.

    '''
        
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

def slide_index(x, expr, index, before, after=timedelta(hours=0), **kwargs):
    '''
    Perform sliding window-type operations relative to specific time points.

    Args:
        x: A `ts_tbl` object representing the time series.
        expr: Expression to be evaluated per window.
        index: The specific time points relative to which the windows are selected.
        before: The time duration before each index to include in the window.
        after: The time duration after each index to include in the window. Default is 0.
        **kwargs: Additional arguments for sliding window operations.

    Returns:
        A `ts_tbl` object with sliding window operations relative to specific time points.

    '''

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


def expand(
    x: pd.DataFrame,
    start_var: str | None = None,
    end_var: str | None = None,
    step_size: pd.Timedelta = None,
    new_index: str | None = None,
    keep_vars: List[str] | None = None,
    aggregate: bool | str | Callable = False,
) -> pd.DataFrame:
    '''
    Expand a time series to make time steps explicit.

    Args:
        x: A `ts_tbl` object representing the time series.
        start_var: Name of the column representing the lower window bounds.
        end_var: Name of the column representing the upper window bounds.
        step_size: Controls the step size used to interpolate between `start_var` and `end_var`.
        new_index: Name of the new index column.
        keep_vars: Names of the columns to hold onto.

    Returns:
        A `ts_tbl` object with time steps made explicit.

    '''
        
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

def hop(x, expr, windows, full_window=False, lwr_col="min_time", upr_col="max_time",
        left_closed=True, right_closed=True, eval_env=None, **kwargs):
    '''
    Apply sliding-window operations / expressions on absolute time-windows defined by intervals.
    [Wrapper around hopper() function with predefined warning message]

    Args:
        x: A pandas DataFrame representing the data to apply the expression to.
        expr: Expression to be evaluated per window.
        windows: A pandas DataFrame defining the intervals for window selection.
        full_window: Boolean value indicating whether the full window should be used. Default is False.
        lwr_col: Name of the column in x representing the lower bound of each window. Default is "min_time".
        upr_col: Name of the column in x representing the upper bound of each window. Default is "max_time".
        left_closed: Boolean value indicating whether the left endpoint of each window is closed. Default is True.
        right_closed: Boolean value indicating whether the right endpoint of each window is closed. Default is True.
        eval_env: Environment for evaluating the expression. Default is None.
        **kwargs: Additional arguments for hopper.

    Returns:
        A pandas DataFrame with the result of applying the expression to the windows.

    '''
    
    msg = "Using 'by reference' syntax in expressions passed to `hop()` most likely will not yield the desired results."
    return hopper(x, expr, windows, full_window, lwr_col, upr_col, left_closed, right_closed, eval_env, msg, **kwargs)

def hopper(x, expr, windows, full_window=False, lwr_col="min_time", upr_col="max_time",
           left_closed=True, right_closed=True, eval_env=None, warn_msg=False, **kwargs):
    '''
    Apply sliding-window operations / expressions on absolute time-windows defined by intervals.

    Args:
        x: A pandas DataFrame representing the data to apply the expression to.
        expr: Expression to be evaluated per window.
        windows: A pandas DataFrame defining the intervals for window selection.
        full_window: Boolean value indicating whether the full window should be used. Default is False.
        lwr_col: Name of the column in x representing the lower bound of each window. Default is "min_time".
        upr_col: Name of the column in x representing the upper bound of each window. Default is "max_time".
        left_closed: Boolean value indicating whether the left endpoint of each window is closed. Default is True.
        right_closed: Boolean value indicating whether the right endpoint of each window is closed. Default is True.
        eval_env: Environment for evaluating the expression. Default is None.
        warn_msg: Boolean value indicating whether a warning message should be displayed. Default is False.
        **kwargs: Additional arguments for hopper.

    Returns:
        A pandas DataFrame with the result of applying the expression to the windows.

    '''
    
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

def fill_gaps(x, limits=None, start_var="start", end_var="end"):
    '''
    Fill gaps in a DataFrame based on specified limits.

    Args:
        x: The DataFrame to fill gaps in.
        limits: A table with columns for lower and upper window bounds or a
                length 2 difftime vector. If `limits` is a difftime vector,
                it is converted into a DataFrame with start and end variables
                inferred from `x` DataFrame. If `limits` is a DataFrame,
                it specifies the lower and upper window bounds for filling
                the gaps. (Default: None)
        start_var: The name of the column representing the lower window bound.
                   (Default: "start")
        end_var: The name of the column representing the upper window bound.
                 (Default: "end")

    Returns:
        A DataFrame with gaps filled based on the specified limits.
    '''
    
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