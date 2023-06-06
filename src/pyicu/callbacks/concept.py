from functools import reduce
from typing import Dict

import pandas as pd

from pyicu.container.table import TableAccessor # Changed from: from pyicu.container.table import is_ts_tbl
from pyicu.assertions import has_interval, has_col, all_fun
from ..interval import hours
from ..utils import expand
from .misc import collect_concepts


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

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from pandas.api.types import is_timedelta64_dtype

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

