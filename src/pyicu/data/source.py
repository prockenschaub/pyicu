import re
from pathlib import Path
from typing import List, Callable, Union

import pandas as pd
import pyarrow.dataset as ds

from ..configs.source import SrcCfg
from ..configs.table import TblCfg

class Src():
    def __init__(self, cfg: SrcCfg, data_dir: Path = None):
        self.cfg = cfg
        self.data_dir = data_dir

        for t in self.cfg.tbl_cfg:
            if t.imp_files_exist(self.data_dir):
                setattr(self, t.name, SrcTbl(t, data_dir))

    @property
    def name(self):
        return self.cfg.name

    @property
    def tables(self) -> List[str]:
        return [t.name for t in self.cfg.tbl_cfg]

    @property
    def available(self) -> str:
        imported = [t.name for t in self.cfg.tbl_cfg if t.imp_files_exist(self.data_dir)]
        return f"{self.cfg.name}: {len(imported)} of {len(self.tables)} tables available"

    def __getitem__(self, table: str) -> "SrcTbl":
        if not isinstance(table, str):
            raise TypeError(f'Expected str, got {table.__class__}')
        self._check_table(table)
        return getattr(self, table)

    def _check_table(self, table):
        if not table in self.tables:
            raise ValueError(f'Table {table} is not defined for source {self.name}')
        if not hasattr(self, table):
            raise ValueError(f'Table {table} has not been imported yet for source {self.name}')

    def load_sel_item(
        self, 
        table: str, 
        sub_var: str, 
        ids: Union[str, int, List], 
        callback: Callable = None, 
        **kwargs
    ) -> pd.DataFrame:
        self._check_table(table)
        if not isinstance(ids, list):
            ids = [ids]
        
        # TODO: hand id_vars not being set as defaults for many tables
        # TODO: convert units
        subset = self[table].data.to_table(
            columns=self[table].meta_vars + [sub_var],
            filter=ds.field(sub_var).isin(ids)
        )
        subset = subset.to_pandas()

        if callback is not None: 
            subset = callback(subset)

        return subset

    def load_col_item(
        self, 
        table: str, 
        val_var: str, 
        unit_val: str = None, 
        callback: Callable = None, 
        **kwargs
    ) -> pd.DataFrame:
        self._check_table(table)

        # TODO: move common elements in loading items to one function
        # TODO: handle units
        subset = self[table].data.to_table(
            columns=self[table].meta_vars + [val_var]
        )
        subset = subset.to_pandas()

        if callback is not None: 
            subset = callback(subset)

        return subset



def defaults_to_str(defaults):
    repr = ''
    for d, v in list(defaults.items()):
        if d != 'time_vars':
            if repr != '':
                repr += ', '
            repr += f"`{v} ({re.sub('_vars?', '', d)})`"
    return repr

def time_vars_to_str(defaults):
    repr = ''
    time_vars = defaults['time_vars']
    if isinstance(time_vars, str):
        time_vars = [time_vars]

    for v in time_vars:
        if repr != '':
            repr += ', '
        repr += f'`{v}`'
    return repr


class SrcTbl():
    def __init__(self, cfg: TblCfg, data_dir: Path = None) -> None:
        self.cfg = cfg
        self.data_dir = data_dir
        self.data = ds.dataset(self.data_dir/f"{self.name}.parquet")

    @property
    def name(self):
        return self.cfg.name

    @property
    def num_rows(self):
        return self.data.count_rows()

    @property
    def num_cols(self):
        return len(self.data.schema)

    @property
    def columns(self):
        return self.data.schema.names

    @property
    def meta_vars(self):
        vars = []
        if self.cfg.defaults is not None:
            for d, v in self.cfg.defaults.items():
                if d in ['id_var', 'index_var']:
                    if isinstance(v, str):
                        vars += [v]
                    else:
                        vars += v 
        return vars

    def __repr__(self):
        repr =  f"# <SrcTbl>:  [{self.num_rows} x {self.num_cols}]\n"
        if self.cfg.defaults is not None:
            repr += f"# Defaults:  {defaults_to_str(self.cfg.defaults)}\n"
            if 'time_vars' in self.cfg.defaults.keys():
                repr += f"# Time vars: {time_vars_to_str(self.cfg.defaults)}\n"
        glimpse = self.data.head(5).to_pandas()
        repr += glimpse.__repr__()
        repr += f'\n... with {self.num_rows-5} more rows'
        return repr
    