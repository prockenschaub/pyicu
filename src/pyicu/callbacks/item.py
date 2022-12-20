import pandas as pd


def calc_dur(x: pd.DataFrame, val_var: str, min_var: str, max_var: str, grp_var: str | None = None) -> pd.DataFrame:
    """Calculate duration of medications

    Calculate durations by taking either per id or per combination of id and `grp_var`
    the minimum for `min_var` and the maximum of `max_var` and returning the
    time difference among the two.

    Note: primarily used to determine vasopressor durations.

    Args:
        x: medication data
        val_var: name of the value variable in the output
        min_var: name of start time
        max_var: name of end time
        grp_var: grouping variable (for example linking infusions). Defaults to None. 

    Returns:
        table with durations
    """
    id = x.icu.id_var
    ind = x.icu.index_var

    if x.shape[0] == 0:
        x[val_var] = x[ind]
        return x

    x = x.reset_index()
    by = [id, grp_var] if grp_var is not None else [id]
    grpd = x.groupby(by)
    
    res = grpd.agg({min_var: "min"})
    res.rename(columns={min_var: ind})
    res[val_var] = grpd[max_var].max() - res[ind]
    res = res.reset_index()
    res = res.icu.as_ts_tbl(id, ind)

    return res
