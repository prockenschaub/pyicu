from functools import reduce
from typing import Dict

import pandas as pd

from pyicu.assertions import has_interval, has_col, all_fun
from ..interval import hours
from ..utils import expand
from .misc import collect_concepts
from typing import Union
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas.api.types import is_datetime64_any_dtype
from pyicu.utils import slide, fill_gaps
from pyicu.tbl_utils import meta_vars
from pyicu.container.table import rm_cols, TableAccessor
from pyicu.utils_misc import chr_ply
from pyicu.interval import change_interval
from pyicu.tbl_utils import index_var
from pyicu.assertions import is_interval

def gcs(
    x: Dict,
    valid_win: pd.Timedelta = hours(6),
    sed_impute: str = "max",
    set_na_max: bool = True,
    interval: pd.Timedelta = None,
    **kwargs,
) -> pd.DataFrame:
    """Aggregate components of the Glasgow Coma Scale (GCS) into a total score

    Aggregating components (whenever the total score tgcs is not already available) requires coinciding
    availability of an eye (egcs), verbal (vgcs) and motor (mgcs) score. In order to match values,
    a last observation carry forward imputation scheme over the time span specified by `valid_win` is performed.
    Furthermore passing "max" as sed_impute will assume maximal points for time steps where the patient is
    sedated (as indicated by sed_gcs), while passing "prev" will assign the last observed value previous to
    the current sedation window and passing False will in turn use raw values. Finally, passing True as
    `set_na_max` will assume maximal points for missing values (after matching and potentially applying sed_impute).

    Args:
        x: GCS components "egcs", "vgcs", "mgcs", "tgcs", and "ett_gcs"
        valid_win: maximal time window for which a GCS value is valid if no newer measurement is available. Defaults to hours(6).
        sed_impute: imputation scheme for values taken when patient was sedated (i.e. unconscious). Defaults to "max".
        set_na_max: whether to impute values that are still missing after locf imputation with the maximum possible value. Defaults to True.
        interval: time interval at which GCS components were measured. If None, this is derived from the data. Defaults to None.

    Returns:
        Total GCS for every hour at which at least one component was measured
    """
    if sed_impute not in ["max", "prev", "none", "verb"]:
        raise ValueError(f'`sed_impute` must be one of ["max", "prev", "none", "verb"], got {sed_impute}')

    cnc = ["egcs", "vgcs", "mgcs", "tgcs", "ett_gcs"]
    res = collect_concepts(x, cnc, interval, **kwargs)

    sed = res.pop("ett_gcs")

    def merge(x, y):
        id = x.icu.id_var
        ind = x.icu.index_var
        return x.merge(y, on=[id, ind], how="outer", sort=True)

    res = reduce(merge, list(res.values()))

    # Apply ffill only within valid_win
    # TODO: this is likely to become really slow if data size increases, see if performance can be improved
    def ffill_within(df, window):
        return df.reset_index(level=0, drop=True).rolling(window, closed="both").apply(lambda x: x.ffill()[-1])

    res = res.groupby(level=0, group_keys=True).apply(lambda df: ffill_within(df, valid_win))

    if sed_impute == "none":
        cnc = cnc[:4]
    else:
        sed = sed[sed[cnc[4]]]

        if sed.icu.is_win_tbl():
            sed = expand(sed, aggregate="any")

        res = merge(res, sed)

    if sed_impute == "max":
        res.loc[~res[cnc[4]].isna() & res[cnc[4]], cnc[3]] = 15
    elif sed_impute == "verb":
        res.loc[~res[cnc[4]].isna() & res[cnc[4]], cnc[1]] = 5
        res.loc[~res[cnc[4]].isna() & res[cnc[4]], cnc[3]] = pd.NA
    elif sed_impute == "prev":
        # TODO: implement
        raise NotImplementedError()

    if set_na_max:
        for component, max_val in zip(cnc[:3], [4, 5, 6]):
            res[component].fillna(max_val, inplace=True)

    res[cnc[3]].fillna(res[cnc[:3]].sum(axis=1), inplace=True)

    res.rename(columns={cnc[3]: "gcs"}, inplace=True)
    res.drop(columns=cnc[:3] + cnc[4:], inplace=True)

    return res

# Transformed from ricu (may contain bugs)
def collect_dots(concepts, interval=None, *args, merge_dat=False):
    assert isinstance(concepts, list)
    assert all(isinstance(concept, str) for concept in concepts)

    dots = list(args)

    if len(concepts) == 1:
        assert len(dots) == 1

        res = dots[0]

        if TableAccessor.is_ts_tbl(res):
            ival = interval if interval is not None else interval(res)
            assert has_interval(res, ival)
        else:
            assert isinstance(res, pd.DataFrame)

        return res

    if len(dots) == 1:
        dots = dots[0]

    if dots is None:
        dots = {}
    else:
        dots = dict(dots)

    if dots is not None:
        dots = {concepts[i]: dots[concepts[i]] for i in range(len(concepts))}

    assert set(dots.keys()) == set(concepts)

    res = {concept: dots[concept] for concept in concepts}

    assert all(has_col(res[concept], concept) for concept in concepts)

    if merge_dat:
        res = reduce(pd.merge, list(res.values()), all=True)
    else:
        res["ival_checked"] = check_interval(res, interval)

    return res

def check_ival(x, iv):
    return isinstance(x, pd.DataFrame) and (not is_datetime64_any_dtype(x) or has_interval(x, iv))

def check_interval(dat, ival=None):
    def has_interval(x, iv):
        # Define your logic to check if dataframe x has the desired interval iv
        # This function should return True or False based on your requirements
        pass

    def interval(dat):
        # Define your logic to determine the interval of the time series dataframe dat
        # This function should return the interval value or raise an exception if it cannot be determined
        pass

    if hasattr(dat, "ival_checked"):
        ival = getattr(dat, "ival_checked")

    elif isinstance(dat, pd.DataFrame) and TableAccessor.is_ts_tbl(dat):
        if ival is None:
            ival = interval(dat)
        else:
            assert has_interval(dat, ival)

    elif isinstance(dat, pd.DataFrame) or all_fun(dat, lambda x: not TableAccessor.is_ts_tbl(x)):
        ival = None

    else:
        if ival is None:
            for x in dat:
                if TableAccessor.is_ts_tbl(x):
                    ival = interval(x)
                    break

        assert all_fun(dat, check_ival, ival)

    return ival

def pafi(*args, match_win: Union[int, Timedelta] = Timedelta(hours=2),
        mode: str = "match_vals", fix_na_fio2: bool = True, interval = None) -> pd.DataFrame:
    """
    Calculate the PaO2/FiO2 (or Horowitz index) for a given time point.
    
    Args:
        *args: Additional arguments.
        match_win (int or Timedelta): Maximum time difference between two measurements for calculating their ratio.
        mode (str): Calculation mode. Options: 'match_vals', 'extreme_vals', 'fill_gaps'.
        fix_na_fio2 (bool): Impute missing FiO2 values with 21%.
        interval: Time interval specification for 'fill_gaps' mode.
        
    Returns:
        DataFrame: Resulting data with calculated PaO2/FiO2 values.
    
    Note:
        - 'match_vals' allows a time difference of at most 'match_win' between two measurements for calculating their ratio.
        - 'extreme_vals' uses the worst PaO2 and FiO2 values within the time window specified by 'match_win'.
        - 'fill_gaps' is a variation of 'extreme_vals' that evaluates ratios at every time point specified by 'interval',
        rather than only when a measurement for either PaO2 or FiO2 is available.
        - If 'fix_na_fio2' is True, missing FiO2 values are imputed with 21, the percentage (by volume) of oxygen in air.
    """
    valid_modes = {
        "match_vals": "match_vals",
        "extreme_vals": "extreme_vals",
        "fill_gaps": "fill_gaps"
    }
    
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Available options are: {', '.join(valid_modes)}.")
    
    mode = valid_modes[mode]
    
    assert isinstance(fix_na_fio2, bool)
    
    cnc = ["po2", "fio2"]
    res = collect_dots(cnc, interval, *args)
    res = match_fio2(res, match_win, mode, cnc[1] if fix_na_fio2 else None)
    
    res = res[(~res[cnc[0]].isna()) & (~res[cnc[1]].isna()) & (res[cnc[1]] != 0)]
    res["pafi"] = 100 * res[cnc[0]] / res[cnc[1]]
    res = rm_cols(res, cnc)
    
    return res


def match_fio2(x: pd.DataFrame, match_win: Union[int, Timedelta], mode: str, fio2 = None) -> pd.DataFrame:
    match_win = pd.Timedelta(match_win)
    
    assert match_win > check_interval(x)
    
    if mode == "match_vals":
        on12 = [f"{meta_vars(x[1])}=={meta_vars(x[2])}"]
        on21 = [f"{meta_vars(x[2])}=={meta_vars(x[1])}"]
        
        x = pd.concat([
            x[1].merge(x[2], left_on=on12, right_on=on12, suffixes=("_1", "_2"), how="left", validate="many_to_one", on=None, sort=False),
            x[2].merge(x[1], left_on=on21, right_on=on21, suffixes=("_2", "_1"), how="left", validate="many_to_one", on=None, sort=False)
        ])
        x = x.drop_duplicates()
        
    else:
        x = pd.concat(x).merge(x, left_index=True, right_index=True, how="outer")

        if mode == "fill_gaps":
            x = fill_gaps(x)
        else:
            assert mode == "extreme_vals"
        
        win_expr = {
            "o2sat": lambda x: x.min_or_na(),
            "fio2": lambda x: x.max_or_na()
        }
        
        x = slide(x, win_expr, before=match_win, full_window=False)
    
    if fio2 is not None:
        x.loc[x[fio2].isna(), fio2] = 21
    
    return x

def vent_ind(*args, match_win: Union[int, Timedelta] = 6, min_length: Union[int, Timedelta] = 30,
            interval=None) -> pd.DataFrame:
    """
    Determine time windows during which patients are mechanically ventilated.

    Args:
        *args: Additional arguments.
        match_win (int or Timedelta): Maximum time difference between start and end events for ventilation.
        min_length (int or Timedelta): Minimal time span between a ventilation start and end time.
        interval: Time interval specification.

    Returns:
        DataFrame: Time windows during which patients are mechanically ventilated.

    Note:
        - Durations are represented by the 'dur_var' column in the returned DataFrame.
        - The 'data_var' column indicates the ventilation status with True values.
        - Currently, no clear distinction between invasive and non-invasive ventilation is made.
    """
    subset_true = lambda x, col: x[x[col].is_true()]
    calc_dur = lambda x, y: x + match_win if y.isna() else y - x
    
    final_int = interval
    
    cnc = ["vent_start", "vent_end", "mech_vent"]
    res = collect_dots(cnc, None, ...)
    
    interval = check_interval(res)
    
    if final_int is None:
        final_int = interval
    
    match_win = Timedelta(hours=match_win)
    min_length = Timedelta(minutes=min_length)
    
    assert is_interval(final_int) and min_length < match_win and interval < min_length
    
    if res[2].shape[0] > 0:
        assert res[0].shape[0] == 0 and res[1].shape[0] == 0
        
        res[2][["vent_ind", "mech_vent"]] = [~res[2]["mech_vent"].isna(), None]
        
        res = change_interval(res[2], final_int, by_ref=True)
        
        return res
    
    assert res[2].shape[0] == 0
    
    res = [subset_true(res[i], cnc[i]) for i in range(len(res)-1)]
    var = "vent_dur"
    
    if res[1].shape[0] > 0:
        idx_vars = [chr_ply(res, index_var)[i] for i in range(len(res))]
        res[1][[var, idx_vars[1]]] = [res[1][idx_vars[1]], res[1][idx_vars[1]] - Timedelta(minutes=1)]
        
        jon = [f" == ".join(reversed([meta_vars(df) for df in res])) for res in reversed(res)]
        jon = " and ".join(jon)
        
        res = res[1].merge(res[0], roll=-match_win, on=jon)
        res[[var] + cnc] = [calc_dur(res[idx_vars[1]], res[var]), None, None]
        res = res[res[var] >= min_length]
    else:
        res = res[0][[var, "vent_start"]].assign(vent_start=match_win)
    
    res = change_interval(res, final_int, by_ref=True)
    res = res.groupby("max").agg("max").reset_index()
    res["vent_ind"] = True
    
    return TableAccessor.as_win_tbl(res, dur_var=var, by_ref=True)


