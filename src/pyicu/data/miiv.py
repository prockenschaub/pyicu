from functools import reduce
from pathlib import Path
from typing import List

import pandas as pd

from .base import Src
from .utils import order_rename
from ..configs import SrcCfg

class MIIV(Src):
    name = "miiv"

    def __init__(self, cfg: SrcCfg = None, data_dir: Path = None):
        super().__init__(cfg, data_dir)

    def _id_win_helper(self):
        def merge_inter(x: pd.DataFrame, y: pd.DataFrame):
            join_vars = list(set(x.columns).intersection(set(y.columns)))
            return x.merge(y, on=join_vars)

        def get_id_tbl(row):
            _, (_, id, start, end, tbl, aux) = row
            return self[tbl].data.to_table(columns=[id, start, end, aux]).to_pandas()

        age = "anchor_age" 
        cfg = self.cfg.ids.cfg.copy()
        cfg['aux'] = [age] + list(cfg.id)[:-1]

        res = list(map(get_id_tbl, cfg.iterrows()))
        res.reverse()
        res = reduce(merge_inter, res)

        # TODO: remove hard-coded variable name
        res["anchor_year"] = pd.to_datetime((res['anchor_year'] - res['anchor_age']).astype('str')+'-1-1')
        res.drop(age, axis=1, inplace=True)

        origin = res[cfg.start.values[-1]]
        for col in pd.concat((cfg.start, cfg.end)):
            res[col] -= origin

        return order_rename(res, cfg.id.to_list(), cfg.start.to_list(), cfg.end.to_list())

    def _map_difftime(self, tbl: pd.DataFrame, id_var: str, time_vars: str | List[str]):
        tbl = tbl.merge(self.id_origin(id_var, origin_name="origin"), on=id_var)
        for var in time_vars:
            tbl[var] = tbl[var] - tbl['origin']
        tbl.drop(columns='origin', inplace=True)
        return tbl
