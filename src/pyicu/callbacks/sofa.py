import numpy as np
import pandas as pd
from pyicu.tbl_utils import index_var, id_vars
from pyicu.utils_ts import hop, slide_index, expand, fill_gaps, slide
from pyicu.callbacks.concept import collect_dots
from pyicu.container.table import TableAccessor
from pyicu.tbl_utils import rename_cols
from .misc import collect_concepts


def sofa_single(cnc, nme, fun):
    '''
    Applies a given function to a concept and creates a new concept based on the result.

    Args:
        cnc (str): Concept name.
        nme (str): Name of the new concept to be created.
        fun (function): Function to apply to the input concept.

    Returns:
        function: A function that takes input data, applies the specified function to the concept, and returns the updated data.

    Notes:
        - The returned function expects the input data to be a pandas DataFrame.
        - The function creates a new column in the DataFrame with the name `nme` and fills it with the results of applying `fun` to the values of the `cnc` column.
        - The `cnc` column is then dropped from the DataFrame.
    '''
    
    def score(x, interval, **kwargs):
        dat = collect_concepts(x, cnc, interval, **kwargs)
        dat[nme] = fun(dat[cnc])
        dat = dat.drop(columns=cnc)
        return dat

    return score

sofa_coag = sofa_single("plt", "sofa_coag", lambda x: 4 - pd.cut(x, [-np.inf, 20, 50, 100, 150, np.inf], labels=False, right=False))

sofa_liver = sofa_single("bili", "sofa_liver", lambda x: pd.cut(x, [-np.inf, 1.2, 2, 6, 12, np.inf], labels=False, right=False))

sofa_cns = sofa_single("gcs", "sofa_cns", lambda x: 4 - pd.cut(x, [-np.inf, 6, 10, 13, 15, np.inf], labels=False, right=False))

def max_or_na(x):
    '''
    Calculates the maximum value from the input data or returns the first value if all values are NaN.

    Args:
        x (array-like): Input data.

    Returns:
        float: The maximum value from the input data or the first value if all values are NaN.

    Notes:
        - This function is typically used as a parameter in the `worst_val_fun` argument of the `sofa_score` function.
    '''
    
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
    '''        
    Calculate the SOFA (Sequential Organ Failure Assessment) score.

    The SOFA score is a commonly used assessment tool for tracking a patient's
    status during a stay at an ICU. It quantifies organ function by aggregating
    6 individual scores representing the respiratory, cardiovascular, hepatic,
    coagulation, renal, and neurological systems. The `sofa_score()` function
    is used as a callback function to the `sofa` concept but is exported as
    there are a few arguments that can be used to modify some aspects of the
    presented SOFA implementation.

    Args:
        *args: Concept data, either passed as a list or individual arguments.
        worst_val_fun (function): Function used to calculate worst values over windows.
        explicit_wins (bool): If False, iterate over all time steps. If True, use only
                       the last time step per patient. If a vector of times is
                       provided, iterate over these explicit time points.
        win_length (str or timedelta): Time frame to look back and apply the `worst_val_fun`.
        keep_components (bool): Logical flag indicating whether to return the
                         individual components alongside the aggregated score
                         (with a suffix `_comp` added to their names).
        interval (str or None): Time series interval (only used for checking the consistency
                  of input data). If None, the interval of the first data
                  object is used.

    Returns:
        A pandas DataFrame representing the SOFA scores.

    Notes:
        The `sofa_score()` function calculates, for each component, the worst
        value over a moving window specified by `win_length`, using the function
        passed as `worst_val_fun`. The default `max_or_na()` function returns
        `NA` instead of `-Inf/Inf` when no measurement is available over an
        entire window. When calculating the overall score by summing up
        components per time-step, an `NA` value is treated as 0.

        Building on separate concepts, measurements for each component are
        converted to a component score based on the SOFA score definition by
        Vincent et al.

    References:
        Vincent, J.-L., Moreno, R., Takala, J. et al. The SOFA (Sepsis-related
        Organ Failure Assessment) score to describe organ dysfunction/failure.
        Intensive Care Med 22, 707–710 (1996).
        https://doi.org/10.1007/BF01709751
    '''
    
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
        res = TableAccessor.rm_cols(res, cnc, by_ref=True)

    return res

def sofa_cardio(interval=None, **kwargs):
    '''
    Calculates the cardiovascular component of the Sequential Organ Failure Assessment (SOFA) score.

    Args:
        interval (str or None): Interval for data collection.
        **kwargs: Keyword arguments to pass to the `collect_dots` function.

    Returns:
        DataFrame: A DataFrame containing the calculated SOFA cardiovascular score.

    Notes:
        - The function requires the `collect_dots` function to be defined and available.
        - The `interval` parameter determines the interval for data collection.
        - Additional keyword arguments are passed to the `collect_dots` function for data collection.
    '''
    
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
    dat = TableAccessor.rm_cols(dat, cnc, by_ref=True)

    return dat

def sofa_resp(interval=None, **kwargs):
    '''
    Calculates the respiratory component of the Sequential Organ Failure Assessment (SOFA) score.

    Args:
        interval (str or None): Interval for data collection.
        **kwargs: Keyword arguments to pass to the `collect_dots` function.

    Returns:
        DataFrame: A DataFrame containing the calculated SOFA respiratory score.

    Notes:
        - The function requires the `collect_dots` function to be defined and available.
        - The `interval` parameter determines the interval for data collection.
        - Additional keyword arguments are passed to the `collect_dots` function for data collection.
    '''
    
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

    dat = TableAccessor.rm_cols(dat, cnc, by_ref=True)

    return dat

def sofa_renal(interval=None, **kwargs):
    '''
    Calculates the renal component of the Sequential Organ Failure Assessment (SOFA) score.

    Args:
        interval (str or None): Interval for data collection.
        **kwargs: Keyword arguments to pass to the `collect_dots` function.

    Returns:
        DataFrame: A DataFrame containing the calculated SOFA renal score.

    Notes:
        - The function requires the `collect_dots` function to be defined and available.
        - The `interval` parameter determines the interval for data collection.
        - Additional keyword arguments are passed to the `collect_dots` function for data collection.
    '''
    
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
    dat = TableAccessor.rm_cols(dat, cnc, by_ref=True)

    return dat