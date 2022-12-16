import numpy as np
import pandas as pd
from .misc import collect_concepts


def sofa_single(cnc, nme, fun):
    def score(x, interval, **kwargs):
        dat = collect_concepts(x, cnc, interval, **kwargs)
        dat[nme] = fun(dat[cnc])
        dat = dat.drop(columns=cnc)
        return dat

    return score


sofa_coag = sofa_single("plt", "sofa_coag", lambda x: 4 - pd.cut(x, [-np.inf, 20, 50, 100, 150, np.inf], labels=False))


def findInterval(x, **kwargs):
    return x
