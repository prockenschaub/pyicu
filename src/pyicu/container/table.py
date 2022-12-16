import warnings
import pandas as pd
from typing import List, Dict
from pandas.api.types import is_numeric_dtype, is_timedelta64_dtype, is_datetime64_any_dtype, is_bool_dtype, is_string_dtype, is_categorical_dtype


from ..utils import enlist, new_names, print_list
from .time import TimeDtype, minutes
from .unit import UnitDtype


@pd.api.extensions.register_dataframe_accessor("tbl")
class TableAccessor:
    """Decorator for pandas DataFrames that adds additional functionality for dealing with ICU tables
    """
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def _validate(self):
        """Check that pandas object fits the requirements of an ICU table

        ICU tables must have a named index. Currently, one, two, or three index levels are supported:
            - a table with one index corresponds to ricu's id_tbl and the index will be interpreted as an observation identifier
            - a table with two indices corresponds to ricu's ts_tbl. The second level will be interpreted as a time index. 
            - a table with three indices corresponds to ricu's win_tbl. The third level will be interpreted as a duration index.  
        Index levels two and three must be of TimeDtype to ensure they have a corresponding interval frequency recorded. More 
        than three levels are not currently supported. 
        """
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
                    raise AttributeError("if there are three index levels, the second and third must be a time index (win_tbl)")
            else:
                raise AttributeError("only MultiIndices with two or three levels are supported")
        elif isinstance(obj.index.dtype, TimeDtype):
            raise AttributeError("must have at least one non-time index")

    def is_id_tbl(self) -> bool:
        """Check if the underlying object is an Id table"""
        try:
            self._validate()
            return not isinstance(self._obj.index, pd.MultiIndex)
        except AttributeError:
            return False

    def as_id_tbl(self, id_var: str | None = None) -> pd.DataFrame:
        """Modify a DataFrame to conform to Id table structure
        
        Id tables have a single, named, non-time index such as "icustay_id". 

        Args: 
            id_var: name of the column that should be set as id variable. If None and no id variable has
                been set yet, the first non-time column is chosen as id variable. Defaults to None. 

        Return: 
            pandas object as Id table
        """
        new_obj = self._obj
        if new_obj.tbl.is_ts_tbl():
            new_obj = new_obj.reset_index(level=1)
        if new_obj.tbl.is_id_tbl():
            if id_var is None or new_obj.tbl.id_var == id_var: 
                return new_obj
        if id_var is None:
            for c in new_obj.columns:
                col_type = new_obj[c].dtype
                if not isinstance(col_type, TimeDtype) and not is_timedelta64_dtype(col_type) and not is_datetime64_any_dtype(col_type):
                    id_var = c
                    break
            if id_var is None: 
                raise TypeError(f'tried to set id variable automatically but no suitable non-time column could be found')
        return new_obj.tbl.set_id_var(id_var)

    def is_ts_tbl(self) -> bool:
        """Check if the underlying object is a Ts table"""
        try:
            self._validate()
            return isinstance(self._obj.index, pd.MultiIndex) and len(self._obj.index.levels) == 2
        except AttributeError:
            return False

    def as_ts_tbl(self, id_var: str | None = None, index_var: str | None = None) -> pd.DataFrame:
        """Modify a DataFrame to conform to Ts table structure
        
        TS tables have a named two-level index where the second level is a time index. 
        For example, "icustay_id" and "charttime". 

        Args: 
            id_var: name of the column that should be set as id variable. If None and no id variable has
                been set yet, the first non-time column is chosen as id variable. Defaults to None. 
            index_var: name of the column that should be set as time index. If None and no index variable
                has been setyet, the first time column is chosen as index variable. Defaults to None. 

        Return: 
            pandas object as Ts table
        """
        new_obj = self._obj
        if new_obj.tbl.is_id_tbl() and not (id_var is None or new_obj.tbl.id_var == id_var):
            new_obj = new_obj.tbl.set_id_var(id_var)
        elif new_obj.tbl.is_ts_tbl():
            if id_var is not None and new_obj.tbl.id_var != id_var:
                new_obj = new_obj.tbl.set_id_var(id_var)
            if index_var is None or new_obj.tbl.index_var == index_var:
                return new_obj
        else: 
            new_obj = new_obj.tbl.as_id_tbl(id_var)
            
        if index_var is None: 
            for c in new_obj.columns:
                col_type = new_obj[c].dtype
                if isinstance(col_type, TimeDtype):
                    index_var = c
                    break
            if index_var is None: 
                raise TypeError(f'tried to set index variable automatically but no suitable time column could be found')
        return new_obj.tbl.set_index_var(index_var)

    @property
    def id_var(self) -> str:
        return self._obj.index.names[0]

    def set_id_var(self, id_var: str, drop: bool = False, inplace: bool = False) -> pd.DataFrame:
        """Set an id for the table

        Args:
            id_var: name of the column that should be used as time index
            inplace: _description_. Defaults to False.

        Raises:
            ValueError: _description_

        Returns:
            _description_
        """
        if not id_var in self._obj.columns:
            if id_var in self._obj.index.names:
                warnings.warn(f"{id_var} is already part of the metadata, left unchanged")
                return self._obj
            else: 
                raise ValueError(f"tried to set Id to unknown column {id_var}")
        new_obj = self._obj
        old_names = []
        if None not in self._obj.index.names:
            old_names = list(new_obj.index.names)
            new_obj = self._obj.reset_index(drop=drop, inplace=inplace)
        return new_obj.set_index([id_var]+old_names[1:], drop=True, inplace=inplace)

    @property
    def index_var(self) -> str:
        """The name of the time index"""
        if self.is_id_tbl():
            raise AttributeError("id_tbl does not have an index_var attribute")
        return self._obj.index.names[1]

    @property
    def interval(self) -> str:
        """the interval of the time index"""
        if self.is_id_tbl():
            raise AttributeError("id_tbl does not have an interval attribute")
        return self._obj.index.dtypes[1]

    def set_index_var(self, index_var: str, drop: bool = False, inplace: bool = False) -> pd.DataFrame:
        """Set a time index for the table

        Args:
            index_var: name of the column that should be used as time index
            drop: whether any existing time index should dropped (True) or turned into a column (False). Defaults to False.
            inplace: whether to modify the DataFrame rather than creating a new one. Defaults to False.

        Returns:
            table with time index
        """
        self._validate()
        if not index_var in self._obj.columns:
            if index_var in self._obj.index.names:
                warnings.warn(f"{index_var} is already part of the metadata, left unchanged")
                return self._obj
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
        """List of all time variables among table columns"""
        # TODO: add the time index
        return [c for c in self._obj.columns if isinstance(self._obj[c].dtype, TimeDtype)]

    def rename_all(self, mapper: Dict, inplace: bool = False) -> pd.DataFrame:
        """Rename both columns and index names 

        Args:
            mapper: a mapping dictionary as accepted by pandas.DataFrame.rename
            inplace: whether to modify the DataFrame rather than creating a new one. Defaults to False.

        Returns:
            renamed table
        """
        new_obj = self._obj
        new_obj = new_obj.rename(columns=mapper, inplace=inplace, errors="ignore")
        new_obj = new_obj.rename_axis(index=mapper, inplace=inplace)
        return new_obj

    def change_interval(self, interval: TimeDtype, cols: str | List[str] | None = None, inplace: bool = False) -> pd.DataFrame:
        """Change the time interval of time columns

        Args:
            interval: new time interval
            cols: time variables for which to change the interval. If None, change interval for 
                all time variables, including those in the pandas index. Defaults to None.
            inplace: whether to modify the DataFrame rather than creating a new one. Defaults to False.

        Returns:
            table with changed time interval
        """
        self._validate()
        
        index_var = None
        new_obj = self._obj
        cols = enlist(cols)

        if self.is_ts_tbl():
            index_var = self.index_var
            new_obj = new_obj.reset_index(index_var, inplace=inplace)

        if cols is None:
            cols = new_obj.tbl.time_vars
            
        for col in cols:
            new_obj[col] = new_obj[col].astype(interval, copy=not inplace)
        
        if index_var is not None:
            new_obj = new_obj.tbl.set_index_var(index_var, inplace=inplace)
        
        return new_obj
    
    def change_id(
        self, 
        src: "Src", 
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
        src: "Src",  
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
            map = map.reset_index()
        else:
            map = src.id_map(idx, target_id, sft, None)

        res = self._obj.merge(map, on=idx, **kwargs)

        if cols is not None:
            for c in cols:
                if dir == "down":
                    res[c] = res[c] + res[sft]
                else:
                    res[c] = res[c] - res[sft]
            res.drop(columns=sft, inplace=True)

        res = res.tbl.set_id_var(target_id)
        return res

    def upgrade_id(self, src: "Src", target_id: str, cols: str | List[str] | None = None, **kwargs) -> pd.DataFrame:
        if cols is None:
            cols = self.time_vars

        if self.is_id_tbl():
            return self._upgrade_id_id_tbl(src, target_id, cols, **kwargs)
        elif self.is_ts_tbl():
            return self._upgrade_id_ts_tbl(src, target_id, cols, **kwargs)

    def _upgrade_id_id_tbl(self, src: "Src", target_id: str, cols: str | List[str] | None = None, **kwargs):
        return self._change_id_helper(src, target_id, cols, "up", **kwargs)

    def _upgrade_id_ts_tbl(self, src: "Src", target_id, cols, id_type=False, **kwargs):
        if self.index_var not in cols:
            raise ValueError(f"index var `{self.index_var}` must be part of the cols parameter")

        x = self._obj

        sft = new_names(x)
        id = x.tbl.id_var
        ind = x.tbl.index_var
        interval = x.tbl.interval

        if id_type:
            id_nms = src.id_cfg.map_type_to_id()
            map = src.id_map(id_nms[id], id_nms[target_id], sft, ind)
            map = map.tbl.rename_all(src.id_cfg.map_id_to_type())
        else:
            map = src.id_map(id, target_id, sft, ind)
        map = map.tbl.set_index_var(x.tbl.index_var)
        map = map.tbl.change_interval(interval)

        # TODO: pandas currently does not have a direct equivalent to R data.table's rolling join
        #       determine match groups ourself (maybe move into function if needed more often).
        a = pd.DataFrame({'which': 0}, index=x.index)
        b = pd.DataFrame({'which': 1}, index=map.index)

        c = pd.concat((a, b))
        c = c.sort_index(ascending=False)
        c['group'] = c.groupby(level=0).which.cumsum().clip(lower=1)
        c = c[c.which == 0]

        x['group'] = c.loc[x.index, 'group']
        
        b = b.sort_index(ascending=False)
        b['group'] = b.groupby(level=0).which.cumsum() # TODO: this is currently wrong
        map['group'] = b.loc[map.index, 'group']

        x = x.reset_index()
        x = x.merge(map, on=[id, 'group'])
        for c in cols:
            x[c] = x[c] - x[sft]
        x = x.tbl.as_ts_tbl(target_id, ind)
        x = x.drop(columns=[sft, 'group'])

        return x

    def downgrade_id(self, src: "Src", target_id: str, cols: str | List[str] | None = None, **kwargs):
        if cols is None:
            cols = self.time_vars

        if self.is_id_tbl():
            return self._downgrade_id_id_tbl(src, target_id, cols, **kwargs)
        elif self.is_ts_tbl():
            return self._downgrade_id_ts_tbl(src, target_id, cols, **kwargs)

    def _downgrade_id_id_tbl(self, tbl, target_id, cols, **kwargs):
        return self._change_id_helper(tbl, target_id, cols, "down", **kwargs)

    def _downgrade_id_ts_tbl(self, src: "Src", target_id, cols, **kwargs):
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
                kwargs['numeric_only'] = False
            elif all([is_string_dtype(c) or is_categorical_dtype(c) for _, c in self._obj[vars].items()]):
                func = "first"
            else:
                raise ValueError(f"when automatically determining an aggregation function, {print_list(vars)} are required to be of the same type")

        grpd = self._obj.groupby(by)
        return grpd[vars].agg(func, *args, **kwargs)


