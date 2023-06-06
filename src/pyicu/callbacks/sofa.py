import numpy as np
import pandas as pd

from pyicu.utils import expand, fill_gaps, slide
from pyicu.tbl_utils import index_var, id_vars
from pyicu.utils_ts import hop, slide_index
from pyicu.callbacks.concept import collect_dots
from pyicu.container.table import rm_cols, rename_cols
from .misc import collect_concepts


def sofa_single(cnc, nme, fun):
    def score(x, interval, **kwargs):
        dat = collect_concepts(x, cnc, interval, **kwargs)
        dat[nme] = fun(dat[cnc])
        dat = dat.drop(columns=cnc)
        return dat

    return score

sofa_coag = sofa_single("plt", "sofa_coag", lambda x: 4 - pd.cut(x, [-np.inf, 20, 50, 100, 150, np.inf], labels=False, right=False))

sofa_liver = sofa_single("bili", "sofa_liver", lambda x: pd.cut(x, [-np.inf, 1.2, 2, 6, 12, np.inf], labels=False, right=False))

sofa_cns = sofa_single("gcs", "sofa_cns", lambda x: 4 - pd.cut(x, [-np.inf, 6, 10, 13, 15, np.inf], labels=False, right=False))

# Transformed from ricu (may contain bugs)
def max_or_na(x):
    if np.all(pd.isna(x)):
        return x[0]
    res = np.nanmean(x)  # Maybe need to replace with appropriate aggregation function
    if np.isnan(res):
        return x[0]
    else:
        return res

# Transformed from ricu (may contain bugs)
def sofa_score(*args, worst_val_fun=max_or_na, explicit_wins=False,
               win_length="24H", keep_components=False, interval=None):
    
    cnc = ["sofa_resp", "sofa_coag", "sofa_liver", "sofa_cardio", "sofa_cns", "sofa_renal"]
    dat = collect_dots(cnc, interval, *args, merge_dat=True)
    
    assert worst_val_fun is not None

    win_length = pd.Timedelta(win_length)
    
    if isinstance(worst_val_fun, str):
        worst_val_fun = eval(worst_val_fun)
    
    expr = lambda x: worst_val_fun(x)

    if not explicit_wins:
        res = fill_gaps(dat)
        res = slide(res, expr, before=win_length, full_window=False, cols=cnc)

    else:
        if explicit_wins is True:
            assert isinstance(win_length, pd.Timedelta)

            ind = index_var(dat)

            win = dat.groupby(id_vars(dat))[ind].max().reset_index()
            win["min_time"] = win[ind] - win_length

            res = hop(dat, expr, win, cols=cnc)
        else:
            explicit_wins = pd.Timedelta(explicit_wins)

            res = slide_index(dat, expr, explicit_wins, before=win_length, full_window=False, cols=cnc)

    res["sofa"] = res[cnc].sum(axis=1, skipna=True)

    if keep_components:
        res = rename_cols(res, [col + "_comp" for col in cnc], cnc, by_ref=True)
    else:
        res = rm_cols(res, cnc, by_ref=True)

    return res

# Transformed from ricu (may contain bugs)
def sofa_cardio(interval=None, **kwargs):
    def score_calc(map, dopa, norepi, dobu, epi):
        if dopa > 15 or epi > 0.1 or norepi > 0.1:
            return 4
        elif dopa > 5 or (epi > 0 and epi <= 0.1) or (norepi > 0 and norepi <= 0.1):
            return 3
        elif (dopa > 0 and dopa <= 5) or dobu > 0:
            return 2
        elif map < 70:
            return 1
        else:
            return 0

    cnc = ["map", "dopa60", "norepi60", "dobu60", "epi60"]

    dat = collect_dots(cnc, interval, **kwargs, merge_dat=True)
    dat["sofa_cardio"] = dat.apply(
        lambda row: score_calc(
            row["map"], row["dopa60"], row["norepi60"], row["dobu60"], row["epi60"]
        ),
        axis=1
    )
    dat = rm_cols(dat, cnc, by_ref=True)

    return dat

# Transformed from ricu (may contain bugs)
def sofa_resp(interval=None, **kwargs):
    def score_calc(x):
        if x < 100:
            return 4
        elif x < 200:
            return 3
        elif x < 300:
            return 2
        elif x < 400:
            return 1
        else:
            return 0

    vent_var = "vent_ind"
    pafi_var = "pafi"

    cnc = [pafi_var, vent_var]

    dat = collect_dots(cnc, interval, **kwargs)
    dat = pd.merge(dat[pafi_var], expand(dat[vent_var], aggregate="any"), how="outer")

    dat.loc[(dat[pafi_var] < 200) & (~dat[vent_var]), pafi_var] = 200
    dat["sofa_resp"] = dat[pafi_var].apply(score_calc)

    dat = rm_cols(dat, cnc, by_ref=True)

    return dat

# Transformed from ricu (may contain bugs)
def sofa_renal(interval=None, **kwargs):
    def score_calc(cre, uri):
        if cre >= 5 or uri < 200:
            return 4
        elif (cre >= 3.5 and cre < 5) or uri < 500:
            return 3
        elif cre >= 2 and cre < 3.5:
            return 2
        elif cre >= 1.2 and cre < 2:
            return 1
        else:
            return 0

    cnc = ["crea", "urine24"]

    dat = collect_dots(cnc, interval, **kwargs, merge_dat=True)
    dat["sofa_renal"] = dat.apply(lambda row: score_calc(row["crea"], row["urine24"]), axis=1)
    dat = rm_cols(dat, cnc, by_ref=True)

    return dat