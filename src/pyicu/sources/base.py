import abc
import warnings
from pathlib import Path
from typing import Type, List, Callable

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc

from ..interval import mins
from ..utils import new_names, enlist
from ..configs import SrcCfg, TblCfg
from ..configs.load import load_src_cfg
from ..container import pyICUTbl, IdTbl, TsTbl
from .utils import defaults_to_str, time_vars_to_str, pyarrow_types_to_pandas


class Src:
    def __init__(self, cfg: SrcCfg = None, data_dir: Path = None):
        if cfg is None and hasattr(self, "name"):
            cfg = load_src_cfg(self.name)

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
        imported = [t.name for t in self.tbl_cfg if t.is_imported(self.data_dir)]
        return f"{self.name}: {len(imported)} of {len(self.tables)} tables available"

    def __getitem__(self, table: str) -> Type["SrcTbl"]:
        if not isinstance(table, str):
            raise TypeError(f"expected str, got {table.__class__}")
        self._check_table(table)
        return getattr(self, table)

    def _check_table(self, table):
        if not table in self.tables:
            raise ValueError(f"table {table} is not defined for source {self.name}")
        if not hasattr(self, table):
            raise ValueError(f"table {table} has not been imported yet for source {self.name}")

    def id_origin(self, id: str, origin_name: str = None, copy: bool = True):
        # TODO: refactor this code and make accessing id configs more natural
        id_info = self.id_cfg.cfg[self.id_cfg.cfg['id'] == id].squeeze()
        tbl = id_info["table"]
        start = id_info["start"]
        # TODO: allow for cases where start is not defined (set it to 0)
        origin = self[tbl].data.to_table(columns=[id, start]).to_pandas(types_mapper=pyarrow_types_to_pandas)

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
            setattr(self, "_id_windows", res)

        if copy:
            res = res.copy()

        return res

    @abc.abstractmethod
    def _id_win_helper(self):
        raise NotImplementedError()

    def id_map(self, id_var: str, win_var: str, in_time: str = None, out_time: str = None):
        """Return a mapping between two ID systems (e.g., hospital and ICU admissions) including start and end dates

        Args:
            id_var: ID variable to which all returned times are relative to
            win_var: ID variable for which in/out times are returned
            in_time: column name for the ID start time. If None, this column is omitted. Defaults to None.
            out_time: column name for the ID end time. If None, this column is omitted. Defaults to None.

        Note: ID vars are defined as names of the IDs in the respective source dataset, e.g., 'icustay_id'
            and 'hadm_id' in MIMIC III.

        Memoization: Since this function is called frequently during data loading and might involve 
            somewhat expensive operations, it relies on an internal helper function `id_map_helper()`)` 
            which performs the heavy lifting and is cached. 

        Example:
            To get a mapping between hospital admissions and ICU stays in MIMIC III with ICU admission 
            and discharge times relative to time of hospital admission, one can run 

            mimic.id_map('hadm_id', 'icustay_id', in_time='start', out_time='end')

        Returns:
            table with mappings between the two IDs and relative start and end times
        """
        key = f"_id_map_{id_var}_{win_var}"
        
        if hasattr(self, key):
            res = getattr(self, key)
        else:
            res = self._id_map_helper(id_var, win_var)
            # TODO: add checks that _id_win_helper returned a valid window
            setattr(self, key, res)

        cols = [id_var, win_var]

        if in_time is not None: 
            inn = win_var + "_start"
            cols.append(in_time)
            res = res.rename(columns={inn: in_time})
        if out_time is not None: 
            out = win_var + "_end"
            cols.append(out_time)
            res = res.rename(columns={out: out_time})

        return res[cols]

    def _id_map_helper(self, id_var: str, win_var: str):
        """Internal calculation of a mapping between two ID systems (e.g., hospital and ICU admissions)

        Args:
            id_var: ID variable to which all returned times are relative to
            win_var: ID variable for which in/out times are returned

        Returns:
            table with mappings from `id_var` to `win_war` and start end end times of `win_var` relative
            to the start time of `id_var`
        """
        map = self.id_windows()
        map_id = map.id_var

        io_vars = [win_var + '_start', win_var + '_end']

        if not id_var == map_id:
            ori = new_names(map)
            map[ori] = map[id_var+'_start']
            map = map.drop(columns=map.columns.difference([id_var, win_var] + io_vars + [ori]))
            map[io_vars] = map[io_vars].apply(lambda x: x - map[ori])
        
        kep = map.columns.difference([id_var, win_var] + io_vars)
        map = map.drop(columns=kep)
        map = map.drop_duplicates()

        return IdTbl(map, id_var=id_var)

    def load_src(self, tbl: str, rows=None, cols=None) -> pd.DataFrame:
        tbl = self[tbl]
        if rows is None:
            tbl = tbl.to_table(columns=cols).to_pandas(types_mapper=pyarrow_types_to_pandas)
        elif isinstance(rows, ds.Expression):
            tbl = tbl.to_table(filter=rows, columns=cols).to_pandas(types_mapper=pyarrow_types_to_pandas)
        else:
            # TODO: should we check for other types here or just forward to take
            tbl = tbl.take(rows, columns=cols).to_pandas(types_mapper=pyarrow_types_to_pandas)
        return tbl

    @abc.abstractmethod
    def _map_difftime(self, tbl, id_vars, time_vars):
        raise NotImplementedError()

    def _resolve_id_hint(self, tbl: Type["SrcTbl"], hint: str):
        if hint in self.id_cfg.name:
            res = self.id_cfg.loc[(self.id_cfg.name == hint).idxmax(), :]
        else:
            hits = self.id_cfg.id.isin(tbl.columns)
            if sum(hits) == 0:
                raise ValueError(f"no overlap between configured id var options and available columns for table {tbl.name}.")
            opts = self.id_cfg.loc[hits, :]
            res = opts.loc[opts.index.max(), :]  # TODO: make this work with IdCfg.index_vars()
        return (res["name"], res["id"])

    def _rename_ids(self, tbl: pyICUTbl):
        mapper = {r["id"]: r["name"] for _, r in self.id_cfg.iterrows()}
        tbl = tbl.rename(columns=mapper)
        tbl.id_var = mapper[tbl.id_var]
        return tbl

    def _add_columns(self, tbl: str, cols: str | List[str] | None, new: str | List[str] | None) -> List[str]:
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
        _, id_var = self._resolve_id_hint(self[tbl], id_hint)
        cols = self._add_columns(tbl, cols, id_var)  # TODO: fix how ids are resolved and when we switch to general id names
        if time_vars is None:
            time_vars = enlist(self[tbl].defaults.get("time_vars"))
        time_vars = list(set(time_vars) & set(cols))

        # Load the table from disk
        tbl = self.load_src(tbl, rows, cols)

        # Calculate difftimes for time variables using the id origin
        if len(time_vars) > 0:
            tbl = self._map_difftime(tbl, id_var, time_vars)
        return IdTbl(tbl, id_var=id_var)

    def load_id_tbl(self, tbl: str, rows=None, cols=None, id_var=None, interval=None, time_vars=None, **kwargs):
        # TODO: Implement ability to change intervals
        if id_var is None:
            id_var = self[tbl].defaults.get("id_var") or self.id_cfg.id.values[-1]
        res = self.load_difftime(tbl, rows, cols, id_var, time_vars)
        res = self.change_id(res, id_var, cols=time_vars, keep_old_id=False)
        return res

    def load_ts_tbl(
        self, tbl: str, rows=None, cols=None, id_var=None, index_var=None, interval=None, time_vars=None, **kwargs
    ):
        if id_var is None:
            id_var = self[tbl].defaults.get("id_var") or self.id_cfg.id.values[-1]
        if index_var is None:
            index_var = self[tbl].defaults.get("index_var")

        cols = self._add_columns(tbl, cols, [index_var])
        res = self.load_difftime(tbl, rows, cols, id_var, time_vars)
        res = TsTbl(res, id_var=res.id_var, index_var=index_var, guess_index_var=True)
        res = self.change_id(res, id_var, cols=time_vars, keep_old_id=False)
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
                raise ValueError(f"cannot load object with target class {target}")

    def load_sel(
        self, tbl: str, sub_var: str, ids: str | int | List | None, cols: List[str] | None = None, **kwargs
    ) -> pd.DataFrame:
        self._check_table(tbl)
        if not isinstance(ids, list):
            ids = [ids]
        return self._do_load_sel(tbl, sub_var, ids, cols, **kwargs)

    def _do_load_sel(self, tbl, sub_var, ids, cols=None, **kwargs):
        # TODO: convert units
        fun = self._choose_target(kwargs.get("target"))
        return fun(tbl, rows=ds.field(sub_var).isin(ids), cols=cols, **kwargs)

    def load_rgx(
        self, tbl: str, sub_var: str, regex: str | None, cols: List[str] | None = None, **kwargs
    ) -> pd.DataFrame:
        self._check_table(tbl)
        return self._do_load_rgx(tbl, sub_var, regex, cols, **kwargs)

    def _do_load_rgx(self, tbl, sub_var, regex, cols=None, **kwargs):
        # TODO: convert units
        fun = self._choose_target(kwargs.get("target"))
        return fun(tbl, rows=pc.match_substring_regex(ds.field(sub_var), regex), cols=cols, **kwargs)


    def load_col(self, tbl: str, val_var: str, unit_val: str = None, **kwargs) -> pd.DataFrame:
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

    def change_id(self, tbl: pyICUTbl, target_id, keep_old_id :bool = True, id_type: bool = False, **kwargs) -> pyICUTbl:
        # TODO: enable id_type
        orig_id = tbl.id_var
        if target_id == orig_id:
            return tbl
       
        ori = self.id_cfg.cfg[self.id_cfg.cfg['id'] == orig_id].squeeze()
        fin = self.id_cfg.cfg[self.id_cfg.cfg['id'] == target_id].squeeze()

        if ori.name < fin.name: # this is the position index, not the column `name`
            res = self.upgrade_id(tbl, target_id, **kwargs)
        elif ori.name > fin.name:
            raise NotImplementedError()
        else:
            raise ValueError("cannot handle conversion of IDs with identical positions")

        if not keep_old_id:
            res = res.drop(columns=orig_id)
        return res

    def _change_id_helper(
        self, 
        tbl: pyICUTbl, 
        target_id: str, 
        cols: str | List[str] | None = None, 
        dir: str = 'down', 
        **kwargs    
    ):
        idx = tbl.id_var

        cols = enlist(cols)
        if cols is not None:
            sft = new_names(tbl)
        else: 
            sft = None

        if dir == "down":
            map = self.id_map(target_id, idx, sft, None)
        else:
            map = self.id_map(idx, target_id, sft, None)

        res = tbl.merge(map, on=idx, **kwargs)       

        if cols is not None:
            for c in cols:
                res[c] = res[c] - res[sft]

        res.set_id_var(target_id)
        res.drop(columns=sft, inplace=True)
        return res

    def upgrade_id(self, tbl, target_id, cols = None, **kwargs):
        if cols is None:
            cols = tbl.time_vars
        if isinstance(tbl, IdTbl):
            return self._upgrade_id_id_tbl(tbl, target_id, cols, **kwargs)
        elif isinstance(tbl, TsTbl):
            return self._upgrade_id_ts_tbl(tbl, target_id, cols, **kwargs)
        else:
            raise TypeError("currently only ids of IdTbl and TsTbl objects can be upgraded")

    def _upgrade_id_id_tbl(self, tbl, target_id, cols, **kwargs):
        return self._change_id_helper(tbl, target_id, cols, "up", **kwargs)

    def _upgrade_id_ts_tbl(self, tbl, target_id, cols, **kwargs):
        if tbl.index_var not in cols:
            raise ValueError(f'index var `{tbl.index_var}` must be part of the cols parameter')
        
        if tbl.interval != mins(1):
            warnings.warn("Changing the ID of non-minute resolution data will change the interval to 1 minute")

        sft = new_names(tbl)
        id = tbl.id_var 
        ind = tbl.index_var

        map = self.id_map(id, target_id, sft, ind)
        
        # TODO: pandas currently does not have a direct equivalent to R data.table's rolling join
        #       It can be approximated with pandas.merge_asof but needs additional sorting and 
        #       does not propagate rolls outside of ends (see data.table's `rollends` parameter).
        #       this code may be slow and may need revisiting/refactoring.
        tbl = tbl.sort_values(ind)
        map = map.sort_values(ind)
        fwd = pd.merge_asof(tbl, map, on=ind, by=id, direction='forward')
        not_matched = fwd[fwd[target_id].isna()][tbl.columns]
        bwd = pd.merge_asof(not_matched, map, on=ind, by=id, direction='backward')
        res = pd.concat((fwd[~fwd[target_id].isna()], bwd), axis=0)
        res = res.sort_values([target_id, ind])

        for c in cols:
            res[c] = res[c] - res[sft]

        res.drop(columns=sft, inplace=True)
        res = TsTbl(res, id_var=target_id, index_var=ind, interval=mins(1))
        return res

class SrcTbl:
    # TODO: define ID options
    def __init__(self, src: Src, cfg: TblCfg, data_dir: Path = None) -> None:
        self.src = src
        self.name = cfg.name
        self.defaults = cfg.defaults.copy()

        if cfg.partitioning:
            self.data = ds.dataset(data_dir / f"{self.name}")
        else:
            self.data = ds.dataset(data_dir / f"{self.name}.parquet")

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
        # TODO: infer from overall config defaults
        return self.defaults.get("id_var")

    @property
    def index_var(self) -> str | None:
        return self.defaults.get("index_var")

    @property
    def time_vars(self) -> List[str] | None:
        return self.defaults.get("time_vars")

    def to_pandas(self):
        return self.data.to_table().to_pandas(types_mapper=pyarrow_types_to_pandas)

    def to_id_tbl(self):
        return IdTbl(self.to_pandas(), id_var=self.defaults.get("id_vars"))

    def to_ts_tbl(self):
        return TsTbl(self.to_pandas(), id_var=self.defaults.get("id_vars"), index_var=self.defaults.get("index_var"))

    def __repr__(self):
        repr = f"# <SrcTbl>:  [{self.num_rows} x {self.num_cols}]\n"
        if self.defaults is not None:
            repr += f"# Defaults:  {defaults_to_str(self.defaults)}\n"
            if "time_vars" in self.defaults.keys():
                repr += f"# Time vars: {time_vars_to_str(self.defaults)}\n"
        glimpse = self.head(5).to_pandas()
        repr += glimpse.__repr__()
        repr += f"\n... with {self.num_rows-5} more rows"
        return repr

    def __getattr__(self, attr):
        """Forward any unknown attributes to the underlying pyarrow.dataset.FileSystemDataset"""
        return getattr(self.data, attr)
