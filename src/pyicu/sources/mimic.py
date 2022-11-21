from functools import reduce
from pathlib import Path
from typing import List

import pandas as pd

from . import Src
from .utils import order_rename
from ..configs import SrcCfg


class MIMIC(Src):
    name = "mimic"
    # TODO: fix upper/lower case difference between cols and defaults (see for example admissions)

    def __init__(self, cfg: SrcCfg = None, data_dir: Path = None):
        super().__init__(cfg, data_dir)

    def _id_win_helper(self):
        def merge_inter(x: pd.DataFrame, y: pd.DataFrame):
            join_vars = list(set(x.columns).intersection(set(y.columns)))
            return x.merge(y, on=join_vars)

        def get_id_tbl(row):
            _, (_, id, start, end, tbl, aux) = row
            cols = [id, start, end]
            if aux is not None:
                cols += [aux]
            return self[tbl].data.to_table(columns=cols).to_pandas()

        cfg = self.cfg.ids.cfg.copy()
        cfg = cfg.sort_index(ascending=False)
        cfg["aux"] = list(cfg.id)[1:] + [None] 

        res = list(map(get_id_tbl, cfg.iterrows()))
        res = reduce(merge_inter, res)

        # Fix the DOB for patients > 89 years, which have their DOB set to 300 years before their first
        # admission: https://mimic.mit.edu/docs/iii/tables/patients/#dob
        # TODO: This currently calculates from the current admission, change to first admission per patient
        def guess_dob(row):
            if row["dob"] < pd.to_datetime("2000-01-01"):
                return row["first_admittime"] - pd.Timedelta(90 * 365.25, "days")
            return row["dob"]

        res['first_admittime'] = res.groupby('subject_id').admittime.cummin()
        res["dob"] = res.apply(guess_dob, axis=1)
        res.drop(columns='first_admittime', inplace=True)

        origin = res[cfg.start.values[-1]]
        for col in pd.concat((cfg.start, cfg.end)):
            res[col] -= origin

        return order_rename(res, cfg.id.to_list(), cfg.start.to_list(), cfg.end.to_list())

    def _map_difftime(self, tbl: pd.DataFrame, id_var: str, time_vars: str | List[str]):
        tbl = tbl.merge(self.id_origin(id_var, origin_name="origin"), on=id_var)
        for var in time_vars:
            tbl[var] = tbl[var] - tbl["origin"]
        tbl.drop(columns="origin", inplace=True)
        return tbl
