import abc
from pathlib import Path
from typing import Type, List, Callable

import pandas as pd
import pyarrow.dataset as ds

from ..configs import SrcCfg, TblCfg
from ..container import IdTbl, TsTbl
from .utils import defaults_to_str, time_vars_to_str

class Src():
    def __init__(self, cfg: SrcCfg, data_dir: Path = None):
        self.cfg = cfg
        self.data_dir = data_dir

        for t in self.tbl_cfg:
            if t.is_imported(self.data_dir):
                setattr(self, t.name, SrcTbl(self, t, data_dir))

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
        elif isinstance(rows, ds.Expression):
            tbl = tbl.to_table(filter=rows, columns=cols).to_pandas()
        else:
            # TODO: should we check for other types here or just forward to take
            tbl = tbl.take(rows, columns=cols).to_pandas()
        return self._rename_ids(tbl)

    @abc.abstractmethod
    def _map_difftime(self, tbl, id_vars, time_vars):
        raise NotImplementedError()

    def _resolve_id_hint(self, tbl: Type["SrcTbl"], hint: str):
        if hint in self.id_cfg.name:
            res = self.id_cfg.loc[(self.id_cfg.name == hint).idxmax(), :]
        else: 
            hits = self.id_cfg.id.isin(tbl.columns)
            if sum(hits) == 0:
                raise ValueError(f"No overlap between configured id var options and available columns for table {tbl.name}.")
            opts = self.id_cfg.loc[hits, :]
            res = opts.loc[opts.index.max(), :]  # TODO: make this work with IdCfg.index_vars()
        return (res['name'], res['id'])

    def _rename_ids(self, tbl: pd.DataFrame):
        mapper = {r['id']: r['name'] for _, r in self.id_cfg.iterrows()}
        return tbl.rename(columns=mapper)

    def _add_columns(
        self, 
        tbl: str, 
        cols: str | List[str] | None, 
        new: str | List[str] | None
    ) -> List[str]:
        if new is None:
            return cols
        elif isinstance(new, str):
            new = [new]
        if cols is None: 
            return self[tbl].columns
        elif isinstance(cols, str):
            cols = [cols]
        # TODO: find a solution that preserves order
        return list(set(cols) | set(new))

    def load_difftime(self, tbl: str, rows=None, cols=None, id_hint=None, time_vars=None):
        # Parse id and time variables
        if id_hint is None:
            id_hint = self[tbl].src.id_cfg.id_var
        id_var = self._resolve_id_hint(self[tbl], id_hint)
        cols = self._add_columns(tbl, cols, id_var[1]) # TODO: fix how ids are resolved and when we switch to general id names
        if time_vars is None:
            time_vars = self[tbl].defaults.get('time_vars')
        time_vars = list(set(time_vars) & set(cols))
        
        # Load the table from disk
        tbl = self.load_src(tbl, rows, cols)
        
        # Calculate difftimes for time variables using the id origin
        if len(time_vars) > 0:
            tbl = self._map_difftime(tbl, id_var[0], time_vars)
        return IdTbl(tbl, id_var=id_var[0])

    def load_id_tbl(self, tbl: str, rows=None, cols=None, id_var=None, interval=None, time_vars=None, **kwargs):
        # TODO: Upgrade or downgrade ID if it differs from what's returned by load_difftime
        # TODO: Implement ability to change intervals
        if id_var is None:
            id_var = self[tbl].defaults.get('id_var') or self.id_cfg.id.values[-1]
        return self.load_difftime(tbl, rows, cols, id_var, time_vars)

    def load_ts_tbl(self, tbl: str, rows=None, cols=None, id_var=None, index_var=None, interval=None, time_vars=None, **kwargs):
        if id_var is None:
            id_var = self[tbl].defaults.get('id_var') or self.id_cfg.id.values[-1]
        if index_var is None:
            index_var = self[tbl].defaults.get('index_var')
        
        cols = self._add_columns(tbl, cols, [id_var, index_var])
        res = self.load_difftime(tbl, rows, cols, id_var, time_vars)
        res = TsTbl(res, id_var=res.id_var, index_var=index_var, guess_index_var=True)
        # TODO: Upgrade or downgrade ID if it differs from what's returned by load_difftime
        # TODO: Implement ability to change intervals
        return res

    def _choose_target(self, target) -> Callable:
        match target:
            case "id_tbl":
                return self.load_id_tbl
            case "ts_tbl":
                return self.load_ts_tbl
            # TODO: add support for win_tbl
            case _:
                raise ValueError(f"Cannot load object with target class {target}")

    def load_sel(
        self, 
        tbl: str, 
        sub_var: str, 
        ids: str | int | List | None, 
        cols: List[str] | None = None,
        **kwargs
    ) -> pd.DataFrame:
        self._check_table(tbl)
        if not isinstance(ids, list):
            ids = [ids]
        return self._do_load_sel(tbl, sub_var, ids, cols, **kwargs)

    def _do_load_sel(self, tbl, sub_var, ids, cols=None, **kwargs):
        # TODO: convert units
        fun = self._choose_target(kwargs.get("target"))
        return fun(tbl, rows=ds.field(sub_var).isin(ids), cols=cols, **kwargs)

    def load_col(
        self, 
        tbl: str, 
        val_var: str, 
        unit_val: str = None, 
        **kwargs
    ) -> pd.DataFrame:
        self._check_table(tbl)
        # TODO: handle units
        return self._do_load_col(tbl, val_var, **kwargs)

    def _do_load_col(self, tbl: str, val_var: str, unit_val: str = None, **kwargs):
        cols = kwargs.pop("cols", None)
        if cols is None:
            cols = [val_var]
        else:
            cols = cols + [val_var]
        fun = self._choose_target(kwargs.get("target"))
        return fun(tbl, cols=cols, **kwargs)


class SrcTbl():
    # TODO: define ID options
    def __init__(self, src: Src, cfg: TblCfg, data_dir: Path = None) -> None:
        self.src = src
        self.name = cfg.name
        self.defaults = cfg.defaults.copy()
        
        if cfg.partitioning:
            self.data = ds.dataset(data_dir/f"{self.name}")
        else:
            self.data = ds.dataset(data_dir/f"{self.name}.parquet")

    @property
    def num_rows(self) -> int:
        return self.data.count_rows()

    @property
    def num_cols(self) -> int:
        return len(self.data.schema)

    @property
    def columns(self) -> List[str]:
        return self.data.schema.names

    @property
    def id_var(self) -> str | None:
        return self.defaults.get('id_var')

    @property
    def index_var(self) -> str | None:
        return self.defaults.get('index_var')

    @property
    def time_vars(self) -> List[str] | None:
        return self.defaults.get('time_vars')

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


