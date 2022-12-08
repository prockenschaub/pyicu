import abc
import warnings
from pathlib import Path
from typing import Type, List, Callable
from pandas._typing import ArrayLike

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc

from ..utils import enlist, intersect, union, new_names, rm_na
from ..configs import SrcCfg, TblCfg, IdCfg
from ..configs.load import load_src_cfg
from ..container.time import TimeArray, TimeDtype, minutes, hours, milliseconds
from .utils import defaults_to_str, time_vars_to_str, pyarrow_types_to_pandas


class Src:
    """Base class for a data source object handling the data access

    Each specific data source like MIMIC III should subclass `Src` and override 
    the data source-specific helper functions `_id_win_helper` and `_map_difftime`. 

    Args: 
        cfg: data source configuration
        data_dir: path to the folder with the imported data files
    """
    def __init__(self, cfg: SrcCfg = None, data_dir: Path = None):
        if cfg is None and hasattr(self, "name"):
            cfg = load_src_cfg(self.name)

        self.cfg = cfg
        self.data_dir = data_dir

        # Attach all imported tables of this source
        for t in self.tbl_cfg:
            if t.is_imported(self.data_dir):
                setattr(self, t.name, SrcTbl(self, t, data_dir))

    @property
    def name(self) -> str:
        """Name of the data source"""
        return self.cfg.name

    @property
    def id_cfg(self) -> IdCfg:
        """Id types configurations"""
        return self.cfg.ids

    @property
    def tables(self) -> List[str]:
        """List of tables that are defined for this source"""
        return [t.name for t in self.cfg.tbls]

    @property
    def tables_imported(self) -> List[str]:
        """List of tables for which data has been imported"""
        return [t.name for t in self.tbl_cfg if t.is_imported(self.data_dir)]

    def print_available(self) -> str:
        return f"{self.name}: {len(self.tables_imported)} of {len(self.tables)} tables available"

    @property
    def tbl_cfg(self) -> List[TblCfg]:
        """Table configurations for all defined tables"""
        return self.cfg.tbls

    def id_origin(self, id: str, origin_name: str = None) -> pd.DataFrame:
        """Obtain start times for a given Id column

        Args:
            id: name of the source Id column
            origin_name: column name for the returned start column. Defaults to None, in which case the source column name is used.
            
        Returns:
            a table mapping each Id entry to its start time
        """
        # TODO: refactor this code and make accessing id configs more natural
        id_info = self.id_cfg.cfg[self.id_cfg.cfg["id"] == id].squeeze()
        tbl = id_info["table"]
        start = id_info["start"]
        # TODO: allow for cases where start is not defined (set it to 0)
        origin = self[tbl].data.to_table(columns=[id, start]).to_pandas(types_mapper=pyarrow_types_to_pandas)

        if origin_name is not None:
            origin = origin.rename(columns={start: origin_name})
        return origin.tbl.set_id_var(id)

    def id_windows(self, copy: bool = True):
        """Obtain start and end times for all available Id columns

        Args:
            copy: logical flag indicating whether to return a copy of the memoized table for safety. Defaults to True.

        Note: ID vars are defined as names of the IDs in the respective source dataset, e.g., 'icustay_id'
            and 'hadm_id' in MIMIC III.

        Memoization: Since this function is called frequently during data loading and might involve
            somewhat expensive operations, it relies on an internal helper function `id_map_helper()`)`
            which performs the heavy lifting and is cached.

        Returns:
            a table with all available Id columns and their respective start and end dates
        """
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
        """Helper function specifying how start and end times for all Id types are calculated

        Note: when adding a new data sources to pyicu, a class specific implementation of this
            function is required. 
        """
        raise NotImplementedError()

    def id_map(self, id_var: str, win_var: str, in_time: str = None, out_time: str = None, copy: bool = True) -> pd.DataFrame:
        """Return a mapping between two ID systems (e.g., hospital and ICU admissions) including start and end dates

        Args:
            id_var: ID variable to which all returned times are relative to
            win_var: ID variable for which in/out times are returned
            in_time: column name for the ID start time. If None, this column is omitted. Defaults to None.
            out_time: column name for the ID end time. If None, this column is omitted. Defaults to None.
            copy: logical flag indicating whether to return a copy of the memoized table for safety. Defaults to True.

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
            setattr(self, key, res)

        if copy:
            res = res.copy()

        cols = [win_var]

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
        map_id = map.tbl.id_var

        io_vars = [win_var + "_start", win_var + "_end"]

        if not id_var == map_id:
            ori = new_names(map)
            map[ori] = map[id_var + "_start"]
            map = map.drop(columns=map.columns.difference([id_var, win_var] + io_vars + [ori]))
            map[io_vars] = map[io_vars].apply(lambda x: x - map[ori])

        kep = map.columns.difference([id_var, win_var] + io_vars)
        map = map.drop(columns=kep)
        map = map.drop_duplicates()

        return map.tbl.as_id_tbl(id_var)

    def load_src(
        self, 
        tbl: str, 
        rows: ds.Expression | ArrayLike | None = None, 
        cols: List[str] | None = None
    ) -> pd.DataFrame:
        """Load a (sub-)table from the underlying raw data

        Args:
            tbl: name of the source table that should be loaded
            rows: a definition of which rows to load from the table. If rows is None, the entire table is loaded.
                If rows is an array of numeric indices, rows with the corresponding row number are loaded.
                If rows is a pyarrow expression, rows that fulfill that expression are returned. Defaults to None.
            cols: names of the columns that should be loaded. If cols is None, all columns will be loaded. Defaults to None.

        Returns:
            table loaded into memory
        """
        tbl = self[tbl]
        if rows is None:
            tbl = tbl.to_table(columns=cols).to_pandas(types_mapper=pyarrow_types_to_pandas)
        elif isinstance(rows, ds.Expression):
            tbl = tbl.to_table(filter=rows, columns=cols).to_pandas(types_mapper=pyarrow_types_to_pandas)
        else:
            # TODO: should we check for other types here or just forward to take
            tbl = tbl.take(rows, columns=cols).to_pandas(types_mapper=pyarrow_types_to_pandas)
        return tbl

    def load_difftime(
        self, 
        tbl: str, 
        rows: ds.Expression | ArrayLike | None = None, 
        cols: List[str] | None = None, 
        id_hint: str | None = None, 
        time_vars: List[str] | None = None,
        interval: TimeDtype = minutes(1),
    ) -> pd.DataFrame:
        """Load a (sub-)table and calculate times relative to the Id origin

        Uses `self.load_src()` to load the actual data.

        Args:
            tbl: name of the source table that should be loaded
            rows: a definition of which rows to load from the table. If rows is None, the entire table is loaded.
                If rows is an array of numeric indices, rows with the corresponding row number are loaded.
                If rows is a pyarrow expression, rows that fulfill that expression are returned. Defaults to None.
            cols: names of the columns that should be loaded. If cols is None, all columns will be loaded. Defaults to None.
            id_hint: name of the Id column to use as the origin. For example this may be 'icustay_id' in MIMIC III. If None,
                the Id system with the highest cardinality among the available ones is chosen. Defaults to None.
            time_vars: names of the time variables that should be converted into relative times. If None, time variables
                are inferred from the default table configuration. Defaults to None.

        Note: if `id_hint` does not exist in the table, the Id system with the highest cardinality among the available ones
            is returned instead. This can later be changed to the required Id systems by calling `tbl.change_id()`.

        Returns:
            table loaded into memory and converted to relative times
        """
        # Parse id and time variables
        if id_hint is None:
            id_hint = self[tbl].src.id_cfg.id_var
        _, id_var = self._resolve_id_hint(self[tbl], id_hint)
        if time_vars is None:
            time_vars = enlist(self[tbl].defaults.get("time_vars"))

        cols = self._add_columns(tbl, cols, id_var)
        time_vars = intersect(cols, time_vars)

        # Load the table from disk
        res = self.load_src(tbl, rows, cols)
        res.tbl.set_id_var(id_var, inplace=True)

        # Calculate difftimes for time variables using the id origin
        if len(time_vars) > 0:
            res = self._map_difftime(res, id_var, time_vars)
        for var in time_vars:
            res[var] = TimeArray(res[var], milliseconds(1))
        res.tbl.change_interval(interval, inplace=True)
        return res

    def load_id_tbl(
        self, 
        tbl: str, 
        rows: ds.Expression | ArrayLike | None = None, 
        cols: List[str] | None = None, 
        id_var: str | None = None, 
        time_vars: List[str] = None, 
        interval: TimeDtype = hours(1),
        **kwargs
    ) -> pd.DataFrame:
        """Load data as an IdTbl object, i.e., a table with an Id column but without a designated time index

        Note: Relies on `self.load_difftime()` to load the actual data and convert times. Note further that `self.load_difftime()` 
            already returns an IdTbl object but the required Id type may not be available in the source table. In that case, 
            `self.load_difftime()` returns the Id system with the highest available cardinality. This function 
            additionally calls `tbl.change_id` to map Id systems and ensure that the desired Id type is returned.

        Args:
            tbl: name of the source table that should be loaded
            rows: a definition of which rows to load from the table. If rows is None, the entire table is loaded.
                If rows is an array of numeric indices, rows with the corresponding row number are loaded.
                If rows is a pyarrow expression, rows that fulfill that expression are returned. Defaults to None.
            cols: names of the columns that should be loaded. If cols is None, all columns will be loaded. Defaults to None.
            id_var: name of the Id column. If None, the name is inferred from the default table configuration (if it exists)
                or is taken as Id system with the highest cardinality among the available ones. Defaults to None.
            time_vars: names of the time variables that should be converted into relative times. If None, time variables
                are inferred from the default table configuration. Defaults to None.
            interval: time resolution to which `time_vars` are rounded to. Defaults to hours(1).

        Returns:
            table loaded into memory, converted into relative times, and assigned an Id column
        """
        if id_var is None:
            id_var = self[tbl].defaults.get("id_var") or self.id_cfg.id.values[-1]
        if time_vars is None:
            time_vars = enlist(self[tbl].defaults.get("time_vars"))
        
        cols = self._add_columns(tbl, cols)
        time_vars = intersect(cols, time_vars)

        res = self.load_difftime(tbl, rows, cols, id_var, time_vars)
        res = res.tbl.change_id(self, id_var, cols=time_vars, keep_old_id=False)
        res = res[res.index.notnull()] # TODO: change this to a function
        if interval is not None:
            res = res.tbl.change_interval(interval, time_vars)
        return res

    def load_ts_tbl(
        self, 
        tbl: str, 
        rows=None, 
        cols: List[str] | None = None, 
        id_var: str | None = None, 
        index_var: str | None = None, 
        time_vars: List[str] | None = None, 
        interval: TimeDtype = hours(1),
        **kwargs
    ):
        """Load data as a TsTbl object, i.e., with an Id column and a designated time index

        Note: Relies on `self.load_id_tbl()` to load the data, handle times, and set Ids

        Args:
            tbl: name of the source table that should be loaded
            rows: a definition of which rows to load from the table. If rows is None, the entire table is loaded.
                If rows is an array of numeric indices, rows with the corresponding row number are loaded.
                If rows is a pyarrow expression, rows that fulfill that expression are returned. Defaults to None.
            cols: names of the columns that should be loaded. If cols is None, all columns will be loaded. Defaults to None.
            id_var: name of the Id column. If None, the name is inferred from the default table configuration (if it exists)
                or is taken as Id system with the highest cardinality among the available ones. Defaults to None.
            time_vars: names of the time variables that should be converted into relative times. If None, time variables
                are inferred from the default table configuration. Defaults to None.
            interval: time resolution to which `time_vars` are rounded to. Defaults to hours(1).

        Returns:
            table loaded into memory, converted into relative times, and assigned an Id column
        """
        if index_var is None:
            index_var = self[tbl].defaults.get("index_var")

        cols = self._add_columns(tbl, cols, enlist(index_var))
        res = self.load_id_tbl(tbl, rows, cols, id_var, time_vars, interval)
        res = rm_na(res, cols=[index_var])
        res = res.tbl.as_ts_tbl(index_var=index_var)
        return res

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

    def load_rgx(self, tbl: str, sub_var: str, regex: str | None, cols: List[str] | None = None, **kwargs) -> pd.DataFrame:
        self._check_table(tbl)
        return self._do_load_rgx(tbl, sub_var, regex, cols, **kwargs)

    def _do_load_rgx(self, tbl, sub_var, regex, cols=None, **kwargs):
        # TODO: convert units
        fun = self._choose_target(kwargs.get("target"))
        return fun(tbl, rows=pc.match_substring_regex(ds.field(sub_var), regex), cols=cols, **kwargs)

    def load_col(self, tbl: str, val_var: str, cols: List[str] | None = None, **kwargs) -> pd.DataFrame:
        self._check_table(tbl)
        # TODO: handle units
        return self._do_load_col(tbl, val_var, cols, **kwargs)

    def _do_load_col(self, tbl: str, val_var: str, cols: List[str] | None = None, **kwargs):
        if cols is None:
            cols = [val_var]
        else:
            cols = cols + [val_var]
        fun = self._choose_target(kwargs.get("target"))
        return fun(tbl, cols=cols, **kwargs)

    @abc.abstractmethod
    def _map_difftime(self, tbl: pd.DataFrame, id_var: str, time_vars: List[str]) -> pd.DataFrame:
        """Calculate the time difference in milliseconds to an Id origin time

        Args:
            tbl: _description_
            id_var: _description_
            time_vars: _description_

        Raises:
            table with time in milliseconds since origin
        """
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

    def _choose_target(self, target) -> Callable:
        match target:
            case "id_tbl":
                return self.load_id_tbl
            case "ts_tbl":
                return self.load_ts_tbl
            # TODO: add support for win_tbl
            case _:
                raise ValueError(f"cannot load object with target class {target}")

    def _add_columns(
        self, 
        tbl: str, 
        cols: str | List[str] | None = None, 
        new: str | List[str] | None = None
    ) -> List[str]:
        if new is None:
            new = []
        elif isinstance(new, str):
            new = [new]
        if cols is None:
            cols = self[tbl].columns
        elif isinstance(cols, str):
            cols = [cols]
        return union(cols, new)

    def _check_table(self, table):
        if not table in self.tables:
            raise ValueError(f"table {table} is not defined for source {self.name}")
        if not hasattr(self, table):
            raise ValueError(f"table {table} has not been imported yet for source {self.name}")

    def __getitem__(self, table: str) -> Type["SrcTbl"]:
        if not isinstance(table, str):
            raise TypeError(f"expected str, got {table.__class__}")
        self._check_table(table)
        return getattr(self, table)


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
        res = self.to_pandas()
        return res.tbl.as_id_var(self.defaults.get("id_vars"))

    def to_ts_tbl(self):
        res = self.to_id_tbl()
        return res.tbl.as_ts_tbl(index_var=self.defaults.get("index_var"))

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
