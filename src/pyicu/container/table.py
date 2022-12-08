import warnings
import pandas as pd
from typing import List, Dict
from pandas.api.types import is_numeric_dtype, is_timedelta64_dtype, is_bool_dtype, is_string_dtype, is_categorical_dtype


from ..utils import enlist, new_names, print_list
from ..sources import Src
from .time import TimeDtype, minutes
from .unit import UnitDtype


@pd.api.extensions.register_dataframe_accessor("tbl")
class TableAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def _validate(self):
        # TODO: change to useful error messages that account for the fact that
        #       some methods are available always (e.g., is_id_tbl or change_interval)
        #       and some are only available for id_tbl or ts_tbl
        obj = self._obj

        # verify there is a named index
        if None in obj.index.names:
            raise AttributeError("table must have named index to use .tbl")

        if isinstance(obj.index, pd.MultiIndex):
            levels = obj.index.levels
            if len(levels) == 2:
                if not isinstance(levels[-1].dtype, TimeDtype):
                    raise AttributeError("if there are two index levels, the second must be a time index (ts_tbl)")
            elif len(levels) == 3:
                if not (isinstance(levels[-2].dtype, TimeDtype) and isinstance(levels[-1].dtype, TimeDtype)):
                    raise AttributeError("if there are two index levels, the second must be a time index (win_tbl)")
            else:
                raise AttributeError("only MultiIndices with 2 or 3 levels are supported")
        elif isinstance(obj.index.dtype, TimeDtype):
            raise AttributeError("must have at least one non-time index")

    def is_id_tbl(self) -> bool:
        try:
            self._validate()
            return not isinstance(self._obj.index, pd.MultiIndex)
        except AttributeError:
            return False

    def as_id_tbl(self, id_var: str | None = None):
        new_obj = self._obj
        if self.is_id_tbl():
            if id_var is None or self.id_var == id_var: 
                return new_obj
        elif self.is_ts_tbl():
            new_obj = new_obj.reset_index(level=1)
        else:
            raise NotImplementedError() # Add logic
        return new_obj.tbl.set_id_var(id_var)

    def is_ts_tbl(self) -> bool:
        try:
            self._validate()
            return isinstance(self._obj.index, pd.MultiIndex)
        except AttributeError:
            return False

    def as_ts_tbl(self, id_var: str | None = None, index_var: str | None = None):
        new_obj = self._obj
        if self.is_id_tbl() and not (id_var is None or self.id_var == id_var):
            new_obj = new_obj.set_id_var(id_var)
        elif self.is_ts_tbl():
            if self.id_var is not None and self.id_var != id_var:
                new_obj = new_obj.tbl.set_id_var(id_var)
            if self.index_var is None or self.index_var == index_var:
                return new_obj
            else: 
                raise NotImplementedError()
        else: 
            new_obj = self.as_id_tbl(id_var)
            
        if index_var is None: 
            raise NotImplementedError()
        return new_obj.tbl.set_index_var(index_var)

    @property
    def id_var(self) -> str:
        return self._obj.index.names[0]

    def set_id_var(self, id_var: str, inplace: bool = False) -> pd.DataFrame:
        if not id_var in self._obj.columns:
            if id_var in self._obj.index.names:
                warnings.warn(f"{id_var} is already part of the metadata, left unchanged")
            else: 
                raise ValueError(f"tried to set Id to unknown column {id_var}")
        new_obj = self._obj
        if None not in self._obj.index.names:
            new_obj = self._obj.reset_index(inplace=inplace)
        return new_obj.set_index(id_var, drop=True, inplace=inplace)

    @property
    def index_var(self) -> str:
        if self.is_id_tbl():
            raise AttributeError("id_tbl does not have an index_var attribute")
        return self._obj.index.names[1]

    def set_index_var(self, index_var: str, drop: bool = False, inplace: bool = False) -> pd.DataFrame:
        self._validate()
        if not index_var in self._obj.columns:
            if index_var in self._obj.index.names:
                warnings.warn(f"{index_var} is already part of the metadata, left unchanged")
            else: 
                raise ValueError(f"tried to set Index to unknown column {index_var}")
        if not isinstance(self._obj[index_var].dtype, TimeDtype):
            raise TypeError(f"index var must be TimeDtype, got {self._obj[index_var].dtype}") 
        if isinstance(self._obj.index, pd.MultiIndex):
            new_obj = self._obj.reset_index(level=1, drop=drop, inplace=inplace)
        else:
            new_obj = self._obj
        return new_obj.set_index(index_var, drop=True, append=True, inplace=inplace)
    
    @property
    def time_vars(self):
        # TODO: add the time index
        return [c for c in self._obj.columns if isinstance(self._obj[c].dtype, TimeDtype)]

    def rename_all(self, mapper: Dict, inplace: bool = False) -> pd.DataFrame:
        new_obj = self._obj
        new_obj = self._obj.rename(columns=mapper, inplace=inplace, errors="ignore")
        new_obj = self._obj.rename_axis(index=mapper, inplace=inplace)
        return new_obj

    def change_interval(self, interval: TimeDtype, inplace: bool = False) -> pd.DataFrame:
        index_var = None
        new_obj = self._obj
        
        if self.is_ts_tbl():
            index_var = self.index_var
            new_obj = new_obj.reset_index(index_var, inplace=inplace)
            
        for col in new_obj.tbl.time_vars:
            new_obj[col] = new_obj[col].astype(interval, copy=not inplace)
        
        if index_var is not None:
            new_obj = new_obj.tbl.set_index_var(index_var, inplace=inplace)
        
        return new_obj
    
    def change_id(
        self, 
        src: Src, 
        target_id, 
        keep_old_id: bool = True, 
        id_type: bool = False, 
        **kwargs
    ) -> pd.DataFrame:
        self._validate()
        # TODO: enable id_type
        orig_id = self.id_var
        if target_id == orig_id:
            return self._obj

        ori = src.id_cfg.cfg[src.id_cfg.cfg["id"] == orig_id].squeeze()
        fin = src.id_cfg.cfg[src.id_cfg.cfg["id"] == target_id].squeeze()

        if ori.name < fin.name:  # this is the position index, not the column `name`
            res = self.upgrade_id(src, target_id, **kwargs)
        elif ori.name > fin.name:
            res = self.downgrade_id(src, target_id, **kwargs)
        else:
            raise ValueError(
                f"cannot handle conversion of Id's with identical positions in the Id config: {orig_id} -> {target_id}"
            )

        if not keep_old_id:
            res = res.drop(columns=orig_id)
        return res

    def _change_id_helper(
        self, 
        src: Src,  
        target_id: str, 
        cols: str | List[str] | None = None, 
        dir: str = "down", 
        **kwargs
    ) -> pd.DataFrame:
        idx = self.id_var

        cols = enlist(cols)
        if cols is not None:
            sft = new_names(self._obj)
        else:
            sft = None

        if dir == "down":
            map = src.id_map(target_id, idx, sft, None)
        else:
            map = src.id_map(idx, target_id, sft, None)

        res = self._obj.merge(map, on=idx, **kwargs)

        if cols is not None:
            for c in cols:
                res[c] = res[c] - res[sft]
            res.drop(columns=sft, inplace=True)

        res = res.tbl.set_id_var(target_id)
        return res

    def upgrade_id(self, src: Src, target_id: str, cols: str | List[str] | None = None, **kwargs) -> pd.DataFrame:
        if cols is None:
            cols = self.time_vars

        if self.is_id_tbl():
            return self._upgrade_id_id_tbl(src, target_id, cols, **kwargs)
        elif self.is_ts_tbl():
            return self._upgrade_id_ts_tbl(src, target_id, cols, **kwargs)

    def _upgrade_id_id_tbl(self, src: Src, target_id: str, cols: str | List[str] | None = None, **kwargs):
        return self._change_id_helper(src, target_id, cols, "up", **kwargs)

    def _upgrade_id_ts_tbl(self, src: Src, target_id, cols, **kwargs):
        if self.index_var not in cols:
            raise ValueError(f"index var `{self.index_var}` must be part of the cols parameter")

        # if self.interval != minutes(1):
        #     warnings.warn("Changing the ID of non-minute resolution data will change the interval to 1 minute")

        sft = new_names(self)
        id = self.id_var
        ind = self.index_var

        map = src.id_map(id, target_id, sft, ind)

        # TODO: pandas currently does not have a direct equivalent to R data.table's rolling join
        #       It can be approximated with pandas.merge_asof but needs additional sorting and
        #       does not propagate rolls outside of ends (see data.table's `rollends` parameter).
        #       this code may be slow and may need revisiting/refactoring.
        tbl = self._obj.sort_values(ind)
        map = map.sort_values(ind)
        fwd = pd.merge_asof(tbl, map, on=ind, by=id, direction="forward")
        not_matched = fwd[fwd[target_id].isna()][tbl.columns]
        bwd = pd.merge_asof(not_matched, map, on=ind, by=id, direction="backward")
        res = pd.concat((fwd[~fwd[target_id].isna()], bwd), axis=0)
        res = res.sort_values([target_id, ind])

        for c in cols:
            res[c] = res[c] - res[sft]

        res.drop(columns=sft, inplace=True)
        #res = TsTbl(res, id_var=target_id, index_var=ind, interval=mins(1))
        return res

    def downgrade_id(self, src: Src, target_id: str, cols: str | List[str] | None = None, **kwargs):
        if cols is None:
            cols = self.time_vars

        if self.is_id_tbl():
            return self._downgrade_id_id_tbl(src, target_id, cols, **kwargs)
        elif self.is_ts_tbl():
            return self._downgrade_id_ts_tbl(src, target_id, cols, **kwargs)

    def _downgrade_id_id_tbl(self, tbl, target_id, cols, **kwargs):
        return self._change_id_helper(tbl, target_id, cols, "down", **kwargs)

    def _downgrade_id_ts_tbl(self, src: Src, target_id, cols, **kwargs):
        if self.index_var not in cols:
            raise ValueError(f"index var `{self.index_var}` must be part of the cols parameter")

        if self.interval != minutes(1):
            warnings.warn("Changing the ID of non-minute resolution data will change the interval to 1 minute")

        # TODO: reset index var here so it can be changed
        res = self._change_id_helper(src, target_id, cols, "down", **kwargs)
        res.tbl.set_index_var(self.index_var)  # reset index var
        res.tbl.change_interval(minutes(1), cols=cols)
        return res
    
    # def merge(
    #     self,
    #     right: pd.DataFrame | pd.Series,
    #     how: str = "inner",
    #     on: Union[IndexLabel, None] = None,
    #     left_on: Union[IndexLabel, None] = None,
    #     right_on: Union[IndexLabel, None] = None,
    #     *args,
    #     **kwargs,
    # ) -> pd.DataFrame:
    #     if on is None and left_on is None and right_on is None:
    #         warnings.warn(f"automatically merged on column {self.id_var}.")
    #         return super().merge(right, how, on=self.id_var, *args, **kwargs)
    #     else:
    #         return super().merge(right, how, on, left_on, right_on, *args, **kwargs)

    def aggregate(
        self, 
        func=None, 
        by=None, 
        vars=None, 
        *args,
        **kwargs
    ) -> pd.DataFrame:
        by, vars = enlist(by), enlist(vars)
        if by is None:
            by = self._obj.index.names
        if vars is None: 
            vars = self._obj.columns
        if func is None:
            if all([is_bool_dtype(c) for _, c in self._obj[vars].items()]):
                func = "any"
            elif all([is_numeric_dtype(c) or 
                        is_timedelta64_dtype(c) or 
                        isinstance(c.dtype, (TimeDtype, UnitDtype)) 
                    for _, c in self._obj[vars].items()]):
                func = "median"
            elif all([is_string_dtype(c) or is_categorical_dtype(c) for _, c in self._obj[vars].items()]):
                func = "first"
            else:
                raise ValueError(f"when automatically determining an aggregation function, {print_list(vars)} are required to be of the same type")

        grpd = self._obj.groupby(by)
        return grpd[vars].agg(func)


