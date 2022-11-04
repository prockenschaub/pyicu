import abc
import re
from pathlib import Path
from functools import reduce
from typing import Type, List, Callable, Union

import pandas as pd
import pyarrow.dataset as ds

from ..configs.source import SrcCfg
from ..configs.table import TblCfg
from ..tables import IdTbl, TsTbl

class Src():
    def __init__(self, cfg: SrcCfg, data_dir: Path = None):
        self.cfg = cfg
        self.data_dir = data_dir

        for t in self.tbl_cfg:
            if t.is_imported(self.data_dir):
                setattr(self, t.name, SrcTbl(t, data_dir))

    @property
    def name(self):
        return self.cfg.name

    @property
    def id_cfg(self):
        return self.cfg.ids

    @property
    def tables(self) -> List[str]:
        return [t.name for t in self.cfg.tbls]

    @property
    def tbl_cfg(self):
        return self.cfg.tbls

    @property
    def available(self) -> str:
        imported = [t.name for t in self.tbl_cfg if t.imp_files_exist(self.data_dir)]
        return f"{self.cfg.name}: {len(imported)} of {len(self.tables)} tables available"

    def __getitem__(self, table: str) -> Type["SrcTbl"]:
        if not isinstance(table, str):
            raise TypeError(f'Expected str, got {table.__class__}')
        self._check_table(table)
        return getattr(self, table)

    def _check_table(self, table):
        if not table in self.tables:
            raise ValueError(f'Table {table} is not defined for source {self.name}')
        if not hasattr(self, table):
            raise ValueError(f'Table {table} has not been imported yet for source {self.name}')

    def id_origin(self, id: str, origin_name: str = None, copy: bool = True):
        id_info = self.id_cfg[id]
        tbl = id_info['table']
        id = id_info['id']
        start = id_info['start']
        # TODO: allow for cases where start is not defined (set it to 0)
        origin = self[tbl].data.to_table(columns=[id, start]).to_pandas()
        origin = self._rename_ids(origin)

        if origin_name is not None:
            origin = origin.rename(columns={start: origin_name})
        if copy:
            origin = origin.copy()
        return origin.drop_duplicates()

    def id_windows(self, copy: bool = True):
        if hasattr(self, "_id_windows"):
            res = getattr(self, "_id_windows")
        else:
            res = self._id_win_helper()
            # TODO: add checks that _id_win_helper returned a valid window
            setattr(self, '_id_windows', res)

        if copy:
            res = res.copy()

        return res

    @abc.abstractmethod
    def _id_win_helper(self):
        raise NotImplementedError()

    def id_map(self, id_var: str, win_var: str, in_time: str = None, out_time: str = None):
        if hasattr(self, "_id_map"):
            res = getattr(self, "_id_map")
        else:
            res = self._id_map_helper(id_var, win_var)
            # TODO: add checks that _id_win_helper returned a valid window
            setattr(self, '_id_map', res)

    def _id_map_helper(self, id_var: str, win_var: str):
        # TODO: add metadata to pandas table (id_vars, index_vars, etc.)
        raise NotImplementedError()

    def load_src(self, tbl: str, rows=None, cols=None) -> pd.DataFrame:
        tbl = self[tbl]
        if rows is None:
            tbl = tbl.to_table(columns=cols).to_pandas()
        else:
            tbl = tbl.take(rows, columns=cols).to_pandas()
        return self._rename_ids(tbl)

    @abc.abstractmethod
    def _map_difftime(self, tbl, id_vars, time_vars):
        raise NotImplementedError()

    def _resolve_id_hint(self, tbl: Type["SrcTbl"], hint: str):
        if hint in self.id_cfg.name:
            return self.id_cfg.name[(self.id_cfg.name == hint).idxmax()]
        hits = self.id_cfg.id.isin(tbl.columns)
        if sum(hits) == 0:
            raise ValueError(f"No overlap between configured id var options and available columns for table {tbl.name}.")
        opts = self.id_cfg.loc[hits, :]
        return opts.loc[opts.index.max(), 'name']  # TODO: make this work with IdCfg.index_vars()

    def _rename_ids(self, tbl: pd.DataFrame):
        mapper = {r['id']: r['name'] for _, r in self.id_cfg.iterrows()}
        return tbl.rename(columns=mapper)

    def load_difftime(self, tbl: str, rows=None, cols=None, id_hint=None, time_vars=None):
        tbl = self[tbl]
        
        # Parse id and time variables
        id_var = self._resolve_id_hint(self[tbl], id_hint)
        if time_vars is None:
            time_vars = tbl.defaults.get('time_vars')
        time_vars = list(set(time_vars) & set(tbl.columns))
        
        # Load the table from disk
        tbl = self.load_src(tbl, rows, cols)
        
        # Calculate difftimes for time variables using the id origin
        if len(time_vars) > 0:
            tbl = self._map_difftime(tbl, id_var, time_vars)
        return IdTbl(tbl, id_vars=id_var)

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
            repr += f"`{v}` ({re.sub('_vars?', '', d)})"
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
    # TODO: define ID options
    def __init__(self, cfg: TblCfg, data_dir: Path = None) -> None:
        self.name = cfg.name
        self.defaults = cfg.defaults.copy()
        
        if cfg.partitioning:
            self.data = ds.dataset(data_dir/f"{self.name}")
        else:
            self.data = ds.dataset(data_dir/f"{self.name}.parquet")

    @property
    def num_rows(self):
        return self.data.count_rows()

    @property
    def num_cols(self):
        return len(self.data.schema)

    @property
    def columns(self):
        return self.data.schema.names

    def to_pandas(self):
        return self.data.to_table().to_pandas()

    def to_id_tbl(self):
        return IdTbl(self.to_pandas(), id_vars=self.defaults.get('id_vars'))

    def to_ts_tbl(self):
        return TsTbl(self.to_pandas(), id_vars=self.defaults.get('id_vars'), index_var=self.defaults.get('index_var'))

    def __repr__(self):
        repr =  f"# <SrcTbl>:  [{self.num_rows} x {self.num_cols}]\n"
        if self.defaults is not None:
            repr += f"# Defaults:  {defaults_to_str(self.defaults)}\n"
            if 'time_vars' in self.defaults.keys():
                repr += f"# Time vars: {time_vars_to_str(self.defaults)}\n"
        glimpse = self.head(5).to_pandas()
        repr += glimpse.__repr__()
        repr += f'\n... with {self.num_rows-5} more rows'
        return repr

    def __getattr__(self, attr):
        """Forward any unknown attributes to the underlying pyarrow.dataset.FileSystemDataset
        """ 
        return getattr(self.data, attr) 




class MIIV(Src):
    def __init__(self, cfg: SrcCfg, data_dir: Path = None):
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

    def _map_difftime(self, tbl: pd.DataFrame, id_var: str, time_vars: Union[str, List[str]]):
        tbl = tbl.merge(self.id_origin(id_var, origin_name="origin"), on=id_var)
        for var in time_vars:
            tbl[var] = tbl[var] - tbl['origin']
        tbl.drop(columns='origin', inplace=True)
        return tbl


def order_rename(df: pd.DataFrame, id_var: List[str], st_var: List[str], ed_var: List[str]):
    def add_suffix(x: List[str], s: str):
        return [f"{i}_{s}" for i in x]
    old_names = id_var + st_var + ed_var
    new_names = id_var + add_suffix(id_var, "start") + add_suffix(id_var, "end")
    df = df[old_names] # Reorder
    df = df.rename({o: n for o, n in zip(old_names, new_names)}, axis='columns')
    return df
