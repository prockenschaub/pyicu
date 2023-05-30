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

#ToDo

# Example from ricu in R
'''
sofa_cns <- sofa_single(
  "gcs", "sofa_cns", function(x) 4L - findInterval(x, c(6, 10, 13, 15))
)
'''

# sofa_cardio, without sofa_single in R

'''
sofa_cardio <- function(..., interval = NULL) {

  score_calc <- function(map, dopa, norepi, dobu, epi) {
    fifelse(
      is_true(dopa > 15 | epi > 0.1 | norepi > 0.1), 4L, fifelse(
        is_true(dopa > 5 | (epi > 0 &    epi <= 0.1) |
                        (norepi > 0 & norepi <= 0.1)), 3L, fifelse(
          is_true((dopa > 0 & dopa <= 5) | dobu > 0), 2L, fifelse(
            is_true(map < 70), 1L, 0L
          )
        )
      )
    )
  }

  cnc <- c("map", "dopa60", "norepi60", "dobu60", "epi60")
  dat <- collect_dots(cnc, interval, ..., merge_dat = TRUE)

  dat <- dat[, c("sofa_cardio") := score_calc(
    get("map"), get("dopa60"), get("norepi60"), get("dobu60"), get("epi60")
  )]
  dat <- rm_cols(dat, cnc, by_ref = TRUE)

  dat
}
'''

# sofa_renal, withoputn sofa_single in R

'''
sofa_renal <- function(..., interval = NULL) {

  score_calc <- function(cre, uri) {
    fifelse(
      is_true(cre >= 5 | uri < 200), 4L, fifelse(
        is_true((cre >= 3.5 & cre < 5) | uri < 500), 3L, fifelse(
          is_true(cre >= 2 & cre < 3.5), 2L, fifelse(
            is_true(cre >= 1.2 & cre < 2), 1L, 0L
          )
        )
      )
    )
  }

  cnc <- c("crea", "urine24")
  dat <- collect_dots(cnc, interval, ..., merge_dat = TRUE)

  dat <- dat[, c("sofa_renal") := score_calc(get("crea"), get("urine24"))]
  dat <- rm_cols(dat, cnc, by_ref = TRUE)

  dat
}
'''