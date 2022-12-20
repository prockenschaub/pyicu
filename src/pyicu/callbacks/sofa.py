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


sofa_coag = sofa_single("plt", "sofa_coag", lambda x: 4 - pd.cut(x, [-np.inf, 20, 50, 100, 150, np.inf], labels=False, right=False))

sofa_liver = sofa_single("bili", "sofa_liver", lambda x: pd.cut(x, [-np.inf, 1.2, 2, 6, 12, np.inf], labels=False, right=False))

sofa_cns = sofa_single("gcs", "sofa_cns", lambda x: 4 - pd.cut(x, [-np.inf, 6, 10, 13, 15, np.inf], labels=False, right=False))