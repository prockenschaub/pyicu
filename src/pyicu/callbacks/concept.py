from functools import reduce
from typing import Dict

import pandas as pd

from ..interval import hours, minutes
from ..utils import expand, new_names
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
            sed = expand(sed, keep_vars=cnc[4], aggregate="any")

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


def vaso60(
    x: Dict, 
    max_gap: pd.Timedelta = minutes(5), 
    interval: pd.Timedelta | None = None, 
    **kwargs
) -> pd.DataFrame:

    dat = collect_concepts(x, {'rate': '_rate$', 'dur': '_dur$'}, interval=interval)
    rate = dat['rate']
    dur = dat['dur'].icu.as_win_tbl().sort_index()

    id = dur.icu.id_var
    ind_var = dur.icu.index_var
    dur_var = dur.icu.dur_var
    dur.reset_index(level=[1, 2], inplace=True)
    dur[dur_var] += dur[ind_var]
    
    # NOTE 1: ricu uses another function merge_ranges here, which relies on a function in the 
    #         data.table package that does not exist in pandas (and which as of 23/12/2022 has
    #         a bug, see https://github.com/eth-mds/ricu/issues/26). However, this function 
    #         is only ever used in vaso60, so we don't need to implement all special cases here. 
    #         Instead, we use the simple solution suggested here: 
    #         https://stackoverflow.com/questions/66163453/pandas-groupby-and-shift-not-doing-quite-what-i-need
    # NOTE 2: Technically, the comparison should already happen on grouped data: 
    #         (grpd[ind_var].shift(0)>grpd[dur_var].shift()). With the below version, it is not
    #         guaranteed that each patients group starts at 0 (e.g., if it overlaps with the last
    #         range of the previous patient). However, the technically correct version forces pandas to 
    #         group twice. Since we are throwing away the "group" column afterwards, we do not care
    #         if groups don't start at 0, so we group only once to save some time.
    # TODO: implement max_gap
    dur["group"]=(dur[ind_var]>dur[dur_var].shift()-max_gap).groupby(level=0).cumsum()
    dur = dur.set_index("group", append=True)
    dur = dur.groupby(level=[0, 1]).agg({ind_var: "min", dur_var: "max"})
    dur.reset_index(level=1, drop=True, inplace=True)
    dur60 = dur[dur[dur_var] - dur[ind_var] >= pd.Timedelta(60, "minutes")]

    ind_rate = rate.icu.index_var
    temp = new_names(rate)
    rate = rate.reset_index(level=1)
    rate.rename(columns={ind_rate: temp}, inplace=True)
    
    res = rate.merge(dur60, on=id)
    res = res[(res[temp] >= res[ind_var]) & (res[temp] <= res[dur_var])]
    res.drop(columns=[ind_var, dur_var], inplace=True)
    res.rename(columns={temp: ind_var}, inplace=True)
    
    if max_gap != pd.Timedelta(0, "hours"):
        res.reset_index(inplace=True)
        res.drop_duplicates(inplace=True)

    res = res.icu.as_ts_tbl(id, ind_var)

    return res
#vaso60 <- function(..., max_gap = mins(5L), interval = NULL) {

#   dur <- dat[["dur"]]
#   dva <- data_vars(dur)
#   idx <- index_var(dur)
#   dur <- dur[get(dva) > 0, ]

#   dur <- dur[, c(dva) := get(idx) + get(dva)]
#   dur <- merge_ranges(dur, idx, dva, max_gap = max_gap, by_ref = TRUE)
#   dur <- dur[get(dva) - get(idx) >= hours(1L), ]

#   rate <- dat[["rate"]]
#   temp <- new_names(c(colnames(dur), colnames(rate)), 2L)
#   rate <- rate[, c(temp) := get(index_var(rate))]
#   on.exit(rm_cols(rate, temp, by_ref = TRUE))

#   join <- c(
#     paste(id_vars(rate), id_vars(dur), sep = " == "),
#     paste(temp, c(">=", "<="), c(index_var(dur), data_vars(dur)))
#   )

#   res <- rate[dur, on = join, nomatch = NULL]
#   res <- rm_cols(res, temp, by_ref = TRUE)
#   res <- rename_cols(res, sub, data_vars(res), by_ref = TRUE,
#                      pattern = "_rate$", replacement = "60")
#   res <- change_interval(res, final_int, by_ref = TRUE)

#   if (max_gap < 0L) {
#     res <- unique(res)
#   }

#   aggregate(res, "max")
# }