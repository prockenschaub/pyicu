from functools import reduce
from typing import Dict

import pandas as pd

from ..interval import hours
from ..utils import expand
from .misc import collect_concepts


def gcs(x: Dict, valid_win: pd.Timedelta = hours(6), sed_impute: str = "max", set_na_max: bool = True, interval: pd.Timedelta = None, **kwargs):
    """_summary_

    Args:
        x: _description_
        valid_win: _description_. Defaults to hours(6).
        sed_impute: _description_. Defaults to "max".
        set_na_max: _description_. Defaults to True.
        interval: _description_. Defaults to None.
    """
    if sed_impute not in ["max", "prev", "none", "verb"]:
        raise ValueError(f'`sed_impute` must be one of ["max", "prev", "none", "verb"], got {sed_impute}')

    cnc = ["egcs", "vgcs", "mgcs", "tgcs", "ett_gcs"]
    res = collect_concepts(x, cnc, interval, **kwargs)

    sed = res.pop('ett_gcs')

    def merge(x, y):
        id = x.icu.id_var
        ind = x.icu.index_var
        return x.merge(y, on=[id, ind], how='outer', sort=True)

    res = reduce(merge, list(res.values()))

    # Apply ffill only within valid_win
    # TODO: this is likely to become really slow, see if performance can be improved
    def ffill_within(df, window):
        return df.reset_index(level=0, drop=True).rolling(window, closed='both').apply(lambda x: x.ffill()[-1])
    
    res = (res.\
        groupby(level=0, group_keys=True).\
        apply(lambda df: ffill_within(df, valid_win))
    )  

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
    res.drop(columns=cnc[:3]+cnc[4:], inplace=True)

    return res

# gcs <- function(..., valid_win = hours(6), sed_impute: str = c("max", "prev", "none", "verb"),
#                 set_na_max = TRUE, interval = NULL) {

#   expr <- substitute(list(egcs = fun(egcs), vgcs = fun(vgcs),
#                           mgcs = fun(mgcs), tgcs = fun(tgcs)),
#                      list(fun = locf))

#   res <- slide(res, !!expr, before = valid_win)

#   if (identical(sed_impute, "none")) {

#     cnc <- cnc[-5L]

#   } else {

#     sed <- sed[is_true(get(cnc[5L])), ]

#     if (is_win_tbl(sed)) {
#       sed <- expand(sed, aggregate = "any")
#     }

#     res <- merge(res, sed, all.x = TRUE)
#   }

#   if (identical(sed_impute, "max")) {

#     res <- res[is_true(get(cnc[5L])), c(cnc[4]) := 15]

#   } else if (identical(sed_impute, "verb")) {

#     res <- res[is_true(get(cnc[5L])), c(cnc[c(2L, 4L)]) := list(5, NA_real_)]

#   } else if (identical(sed_impute, "prev")) {

#     idv <- id_vars(res)
#     res <- res[, c(cnc[-5L]) := lapply(.SD, replace_na, 0), .SDcols = cnc[-5L]]
#     res <- res[is_true(get(cnc[5L])), c(cnc[-5L]) := NA_real_]
#     res <- res[, c(cnc[-5L]) := lapply(.SD, replace_na, type = "locf"),
#                .SDcols = cnc[-5L], by = c(idv)]
#     res <- res[, c(cnc[-5L]) := lapply(.SD, zero_to_na), .SDcols = cnc[-5L]]
#   }

#   if (set_na_max) {
#     res <- res[, c(cnc[1L:3L]) := Map(replace_na, .SD, c(4, 5, 6)),
#                .SDcols = cnc[1L:3L]]
#   }

#   res <- res[is.na(get(cnc[4L])), c(cnc[4L]) := rowSums(.SD),
#              .SDcols = cnc[1L:3L]]

#   if (set_na_max) {
#     res <- res[, c(cnc[4L]) := list(replace_na(get(cnc[4L]), 15))]
#   }

#   res <- rename_cols(res, "gcs", cnc[4L], by_ref = TRUE)
#   res <- rm_cols(res, cnc[-4L], by_ref = TRUE)

#   res
# }