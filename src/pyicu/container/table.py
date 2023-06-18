"""
Tabular ICU data classes

In order to simplify handling of tabular ICU data, `ricu` provides the following S3 classes:
`id_tbl`, `ts_tbl`, and `win_tbl`. These classes consist of a `data.table` object
alongside some meta data, and S3 dispatch is used to enable more natural behavior for
some data manipulation tasks. For example, when merging two tables, a default for the `by` argument
can be chosen more sensibly if columns representing patient ID and timestamp information
can be identified.
"""

import warnings
import pandas as pd
from typing import List, Dict
from pandas.api.types import (
    is_numeric_dtype,
    is_timedelta64_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype,
    is_string_dtype,
    is_categorical_dtype,
)

from ..interval import minutes, change_interval
from ..utils import enlist, print_list, new_names
from pyicu.utils_cli import stop_ricu, stop_generic
from pyicu.tbl_utils import id_vars, index_var, dur_var, dur_col, meta_vars, index_col, interval
from pyicu.assertions import assert_that, is_unique, is_disjoint, is_difftime, has_cols, obeys_interval, same_unit, is_flag
from .unit import UnitDtype


@pd.api.extensions.register_dataframe_accessor("icu")
class TableAccessor:
    """Decorator for pandas DataFrames that adds additional functionality for dealing with ICU tables"""

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
                if not is_timedelta64_dtype(levels[-1].dtype):
                    raise AttributeError("if there are two index levels, the second must be a time index (ts_tbl)")
            elif len(levels) == 3:
                if not (is_timedelta64_dtype(levels[-2].dtype) and is_timedelta64_dtype(levels[-1].dtype)):
                    raise AttributeError(
                        "if there are three index levels, the second and third must be a time index (win_tbl)"
                    )
            else:
                raise AttributeError("only MultiIndices with two or three levels are supported")
        elif is_timedelta64_dtype(obj.index.dtype):
            raise AttributeError("must have at least one non-time index")

    def is_pandas(self) -> bool:
        """Check if the underlying object is neiter id_tbl, ts_tbl, nor win_tbl"""
        try:
            self._validate()
            return False
        except AttributeError:
            return True

    def is_id_tbl(self) -> bool:
        """Check if the underlying object is an id_tbl"""
        return (not self.is_pandas()) and (not isinstance(self._obj.index, pd.MultiIndex))

    def is_ts_tbl(self) -> bool:
        """Check if the underlying object is a ts_tbl"""
        return (not self.is_pandas()) and isinstance(self._obj.index, pd.MultiIndex) and len(self._obj.index.levels) == 2

    def is_win_tbl(self) -> bool:
        """Check if the underlying object is a win_tbl"""
        return (not self.is_pandas()) and isinstance(self._obj.index, pd.MultiIndex) and len(self._obj.index.levels) == 3

    def as_id_tbl(self, id_var: str | None = None) -> pd.DataFrame:
        """Modify a DataFrame to conform to id_tbl structure

        id_tbl have a single, named, non-time index such as "icustay_id".

        Args:
            id_var: name of the column that should be set as id variable. If None and no id variable has
                been set yet, the first non-time column is chosen as id variable. Defaults to None.

        Return:
            pandas object as id_tbl
        """
        new_obj = self._obj
        if new_obj.icu.is_win_tbl():
            new_obj = new_obj.reset_index(level=2)
        if new_obj.icu.is_ts_tbl():
            new_obj = new_obj.reset_index(level=1)
        if new_obj.icu.is_id_tbl():
            if id_var is None or new_obj.icu.id_var == id_var:
                # table is already an id_tbl with required structure
                return new_obj

        if id_var is None:
            # try to determine id_var automatically
            for c in new_obj.columns:
                col_type = new_obj[c].dtype
                if not (is_timedelta64_dtype(col_type) or is_datetime64_any_dtype(col_type)):
                    id_var = c
                    break

        if id_var is None:
            raise TypeError(f"tried to set id variable automatically but no suitable non-time column could be found")
        return new_obj.icu.set_id_var(id_var)

    def as_ts_tbl(self, id_var: str | None = None, index_var: str | None = None) -> pd.DataFrame:
        """Modify a DataFrame to conform to ts_tbl structure

        ts_tbl have a named two-level index where the second level is a time index.
        For example, "icustay_id" and "charttime".

        Args:
            id_var: name of the column that should be set as id variable. If None and no id variable has
                been set yet, the first non-time column is chosen as id variable. Defaults to None.
            index_var: name of the column that should be set as time index. If None and no index variable
                has been set yet, the first time column is chosen as index variable. Defaults to None.

        Return:
            pandas object as ts_tbl
        """
        new_obj = self._obj
        if new_obj.icu.is_win_tbl():
            new_obj = new_obj.reset_index(level=2)
        if new_obj.icu.is_ts_tbl():
            if (id_var is None or new_obj.icu.index_var == index_var) and (
                index_var is None or new_obj.icu.index_var == index_var
            ):
                # table is already a ts_tbl with required structure
                return new_obj
            elif index_var is None:
                index_var = new_obj.icu.index_var

        if new_obj.icu.is_pandas():
            new_obj = new_obj.icu.as_id_tbl(id_var)
        elif id_var is not None:
            new_obj = new_obj.icu.set_id_var(id_var)

        new_obj.icu._validate()

        if index_var is None:
            # try to determine index_var automatically
            for c in new_obj.columns:
                col_type = new_obj[c].dtype
                if is_timedelta64_dtype(col_type):
                    index_var = c
                    break
        if index_var is None:
            raise TypeError(f"tried to set index variable automatically but no suitable time column could be found")

        return new_obj.icu.set_index_var(index_var)

    def as_win_tbl(self, id_var: str | None = None, index_var: str | None = None, dur_var: str | None = None) -> pd.DataFrame:
        """Modify a DataFrame to conform to win_tbl structure

        win_tbl have a named three-level index where the second level is a time index and the third denotes a time duration.
        For example, "icustay_id", "starttime", and "duration", where duration is equal to "starttime" - "endtime".

        Args:
            id_var: name of the column that should be set as id variable. If None and no id variable has
                been set yet, the first non-time column is chosen as id variable. Defaults to None.
            index_var: name of the column that should be set as time index. If None and no index variable
                has been set yet, the first time column is chosen as index variable. Defaults to None.
            dur_var: name of the column that should be set as duration index. If None and no duration variable
                has been set yet, the second time column is chosen as duration variable. Defaults to None.

        Return:
            pandas object as win_tbl
        """
        new_obj = self._obj
        if new_obj.icu.is_win_tbl():
            if dur_var is None or new_obj.icu.dur_var == dur_var:
                # table is already a win_tbl with required structure
                return new_obj

        if new_obj.icu.is_pandas():
            new_obj = new_obj.icu.as_id_tbl(id_var)
        elif id_var is not None:
            new_obj = new_obj.icu.set_id_var(id_var)

        if new_obj.icu.is_id_tbl():
            new_obj = new_obj.icu.as_ts_tbl(index_var=index_var)
        elif index_var is not None:
            new_obj = new_obj.icu.set_index_var(index_var)

        new_obj.icu._validate()

        if dur_var is None:
            # try to determine dur_var automatically
            for c in new_obj.columns:
                col_type = new_obj[c].dtype
                if is_timedelta64_dtype(col_type) and c != index_var:
                    dur_var = c
                    break
        if dur_var is None:
            raise TypeError(f"tried to set duration variable automatically but no suitable time column could be found")

        return new_obj.icu.set_dur_var(dur_var)

    @property
    def id_var(self) -> str:
        """The name of the unit id (for id_tbls, ts_tbls, or win_tbls)"""
        self._validate()
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
                return self._obj
            else:
                raise ValueError(f"tried to set id to unknown column {id_var}")
        new_obj = self._obj
        old_names = []
        if None not in self._obj.index.names:
            old_names = list(new_obj.index.names)
            new_obj = self._obj.reset_index(drop=drop, inplace=inplace)
        return new_obj.set_index([id_var] + old_names[1:], drop=True, inplace=inplace)

    @property
    def index_var(self) -> str:
        """The name of the time index (for ts_tbls or win_tbls)"""
        self._validate()
        if self.is_id_tbl():
            raise AttributeError("id_tbl does not have an index_var attribute")
        return self._obj.index.names[1]

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
                return self._obj
            else:
                raise ValueError(f"tried to set index to unknown column {index_var}")
        if not is_timedelta64_dtype(self._obj[index_var].dtype):
            raise TypeError(f"index var must be timedelta, got {self._obj[index_var].dtype}")
        if isinstance(self._obj.index, pd.MultiIndex):
            new_obj = self._obj.reset_index(level=1, drop=drop, inplace=inplace)
        else:
            new_obj = self._obj
        return new_obj.set_index(index_var, drop=True, append=True, inplace=inplace)

    @property
    def dur_var(self) -> str:
        """The name of the duration index (for win_tbls)"""
        self._validate()
        if not self.is_win_tbl():
            raise AttributeError("only win_tbl have dur_var attribute")
        return self._obj.index.names[2]

    def set_dur_var(self, dur_var: str, drop: bool = False, inplace: bool = False) -> pd.DataFrame:
        """Set a duration index for the table

        Args:
            dur_var: name of the column that should be used as duration index
            drop: whether any existing duration index should dropped (True) or turned into a column (False). Defaults to False.
            inplace: whether to modify the DataFrame rather than creating a new one. Defaults to False.

        Returns:
            table with time index
        """
        self._validate()
        if self.is_id_tbl():
            raise AttributeError(f"can only set duration index on ts_tbls or win_tbls")
        if not dur_var in self._obj.columns:
            if dur_var in self._obj.index.names:
                return self._obj
            else:
                raise ValueError(f"tried to set duration to unknown column {dur_var}")
        if not is_timedelta64_dtype(self._obj[dur_var].dtype):
            raise TypeError(f"duration var must be timedelta, got {self._obj[dur_var].dtype}")
        if isinstance(self._obj.index, pd.MultiIndex) and len(self._obj.index.levels) == 3:
            new_obj = self._obj.reset_index(level=2, drop=drop, inplace=inplace)
        else:
            new_obj = self._obj
        return new_obj.set_index(dur_var, drop=True, append=True, inplace=inplace)

    @property
    def time_vars(self):
        """List of all time variables among table columns"""
        # TODO: add the time index
        return [c for c in self._obj.columns if is_timedelta64_dtype(self._obj[c].dtype)]

    @property
    def interval(self) -> str:
        """the interval of the time index"""
        if self.is_id_tbl():
            raise AttributeError("id_tbl does not have an interval attribute")
        units = self._obj.index.get_level_values(1).components.max() != 0
        smallest = units[::-1].idxmax()
        values = self._obj.index.get_level_values(1).components[smallest]
        values = values[values != 0]

        return pd.Timedelta(values.min() if len(values) > 0 else 1, smallest)

    def change_interval(
        self, interval: pd.Timedelta, cols: str | List[str] | None = None, inplace: bool = False
    ) -> pd.DataFrame:
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
            cols = new_obj.icu.time_vars

        for col in cols:
            if inplace:
                # TODO: currently raises warnings, wait for it to be resolved on pandas side
                #       https://github.com/pandas-dev/pandas/issues/48673
                raise NotImplementedError()
            else:
                new_obj[col] = change_interval(new_obj[col], interval)

        if index_var is not None:
            new_obj = new_obj.icu.set_index_var(index_var, inplace=inplace)

        return new_obj

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

    def change_id(self, src: "Src", target_id, keep_old_id: bool = True, id_type: bool = False, **kwargs) -> pd.DataFrame:
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
        self, src: "Src", target_id: str, cols: str | List[str] | None = None, dir: str = "down", **kwargs
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

        res = res.icu.set_id_var(target_id)
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
        id = x.icu.id_var
        ind = x.icu.index_var

        if id_type:
            id_nms = src.id_cfg.map_type_to_id()
            map = src.id_map(id_nms[id], id_nms[target_id], sft, ind)
            map = map.icu.rename_all(src.id_cfg.map_id_to_type())
        else:
            map = src.id_map(id, target_id, sft, ind)
        map = map.icu.set_index_var(x.icu.index_var)

        # TODO: pandas currently does not have a direct equivalent to R data.table's rolling join
        #       determine match groups ourself (maybe move into function if needed more often).
        #       Maybe pandas.merge_asof could help but couldn't get it to fully work yet.
        a = pd.DataFrame({"which": 0}, index=x.index)
        b = pd.DataFrame({"which": 1}, index=map.index)

        c = pd.concat((a, b))
        c = c.sort_index(ascending=False)
        c["group"] = c.groupby(level=0).which.cumsum().clip(lower=1)
        c = c[c.which == 0]

        x["group"] = c.loc[x.index, "group"]

        b = b.sort_index(ascending=False)
        b["group"] = b.groupby(level=0).which.cumsum()
        map["group"] = b.loc[map.index, "group"]

        x = x.reset_index()
        x = x.merge(map, on=[id, "group"])
        for c in cols:
            x[c] = x[c] - x[sft]
        x = x.icu.as_ts_tbl(target_id, ind)
        x = x.drop(columns=[sft, "group"])

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
        res.icu.set_index_var(self.index_var)  # reset index var
        res.icu.change_interval(minutes(1), cols=cols)
        return res

    def aggregate(self, func=None, by=None, vars=None, *args, **kwargs) -> pd.DataFrame:
        by, vars = enlist(by), enlist(vars)
        if by is None:
            by = self._obj.index.names
        if vars is None:
            vars = self._obj.columns
        if func is None:
            if all([is_bool_dtype(c) for _, c in self._obj[vars].items()]):
                func = "any"
            elif all(
                [
                    is_numeric_dtype(c) or is_timedelta64_dtype(c) or isinstance(c.dtype, UnitDtype)
                    for _, c in self._obj[vars].items()
                ]
            ):
                func = "median"
                kwargs["numeric_only"] = False
            elif all([is_string_dtype(c) or is_categorical_dtype(c) for _, c in self._obj[vars].items()]):
                func = "first"
            else:
                raise ValueError(
                    f"when automatically determining an aggregation function, {print_list(vars)} are required to be of the same type"
                )

        grpd = self._obj.groupby(by)
        return grpd[vars].agg(func, *args, **kwargs)
    
    ## New functions
    def reclass_tbl(self, x, template, stop_on_fail=True):
        template = template

        return self.reclass_tbl_impl(x, template, stop_on_fail)

    def reclass_tbl_impl(self, x, template, stop_on_fail=True):
        template_class = template.__class__.__name__

        if template_class == "NULL":
            return x
        elif template_class == "id_tbl":
            x = self.reclass_tbl_impl(self, template, template, stop_on_fail)
            x = self.set_attributes(x, id_vars=id_vars(template), _class=template_class)
            self.check_valid(x, stop_on_fail)
            return x
        elif template_class == "ts_tbl":
            x = self.reclass_tbl_impl(template, template, stop_on_fail)
            x = self.set_attributes(x, index_var=index_var(template), interval=interval(template), _class=template_class)
            if self.validate_tbl(x):
                return x
            return self.reclass_tbl_impl(template, template, stop_on_fail)
        elif template_class == "win_tbl":
            x = self.reclass_tbl_impl(template, template, stop_on_fail)
            x = self.set_attributes(x, dur_var=dur_var(template), _class=template_class)
            if self.validate_tbl(x):
                return x
            return self.reclass_tbl_impl(template, template, stop_on_fail)
        elif template_class == "data.table":
            return self.check_valid(self.unclass_tbl(x), stop_on_fail)
        elif template_class == "data.frame":
            return self.check_valid(pd.DataFrame(self.unclass_tbl(x)), stop_on_fail)
        else:
            return stop_generic(template, "reclass_tbl")

    def try_reclass(self, x, template):
        if isinstance(x, pd.DataFrame):
            return self.reclass_tbl(x, template, stop_on_fail=False)
        else:
            return x

    def as_ptype(self, x):
        return self.as_ptype_impl(x)

    def as_ptype_impl(self, x):
        x_class = x.__class__.__name__

        if x_class == "id_tbl":
            return self.reclass_tbl(pd.DataFrame(map(x, lambda col: col[0])[meta_vars(x)]), x)
        elif x_class == "data.table":
            return pd.DataFrame()
        elif x_class == "data.frame":
            return pd.DataFrame()
        else:
            return stop_generic(x, "as_ptype")
        
    def validate_that(*assertions):
        for assertion in assertions:
            if not assertion:
                return False
        return True

    def validate_tbl(self, x):
        res = self.validate_that(
            isinstance(x, pd.DataFrame), is_unique(list(x.columns))
        )
        
        if not res:
            return res
        
        self.validate_tbl.dispatch(type(x))(x)

    def validate_tbl_id_tbl(self, x):
        idv = id_vars(x)
        res = self.validate_that(
            has_cols(x, idv), is_unique(idv)
        )
        
        if res:
            self.validate_tbl.dispatch(type(x))(x)
        else:
            return res

    def validate_tbl_ts_tbl(self, x):
        index = index_col(x)
        inval = interval(x)
        invar = index_var(x)
        
        res = self.validate_that(
            isinstance(invar, str), has_cols(x, invar), is_disjoint(id_vars(x), invar),
            obeys_interval(index, inval), same_unit(index, inval)
        )
        
        if res:
            self.validate_tbl.dispatch(type(x))(x)
        else:
            return res

    def validate_tbl_win_tbl(self, x):
        dvar = dur_var(x)
        
        res = self.validate_that(
            isinstance(dvar, str), has_cols(x, dvar), is_disjoint(id_vars(x), dvar),
            is_disjoint(dvar, index_var(x)), is_difftime(dur_col(x))
        )
        
        if res: # Changed from is_true()
            self.validate_tbl.dispatch(type(x))(x)
        else:
            return res

    def validate_tbl_data_frame(x):
        return True

    def validate_tbl_default(x):
        stop_generic(x, ".Generic")

    def check_valid(self, x, stop_on_fail=True):
        res = self.validate_tbl(x)
        
        if res:
            return x
        elif stop_on_fail:
            stop_ricu(res, class_=["valid_check_fail", getattr(res, "assert_class")])
        else:
            return self.unclass_tbl(x)
        
    def unclass_tbl(self, x):
        return self.unclass_tbl.dispatch(x)

    def unclass_tbl_data_frame(x):
        return x

    def strip_class(x):
        if hasattr(x, '__dict__'):
            x.__dict__.clear()
        return x

    def unclass_tbl_win_tbl(self, x):
        return self.unclass_tbl(self.et_attributes(x, dur_var=None, class_=self.strip_class(x)))

    def unclass_tbl_ts_tbl(self, x):
        return self.unclass_tbl(
            self.set_attributes(x, index_var=None, interval=None, class_=self.strip_class(x))
        )

    def unclass_tbl_id_tbl(self, x):
        return self.set_attributes(x, id_vars=None, class_=self.strip_class(x))

    def unclass_tbl_default(x):
        stop_generic(x, ".Generic")

    unclass_tbl.dispatch = {
        "data.frame": unclass_tbl_data_frame,
        "win_tbl": unclass_tbl_win_tbl,
        "ts_tbl": unclass_tbl_ts_tbl,
        "id_tbl": unclass_tbl_id_tbl,
        "default": unclass_tbl_default,
    }

    def set_attributes(x, **kwargs):
        dot = kwargs
        nms = list(dot.keys())

        assert_that(isinstance(dot, dict), len(dot) > 0, nms is not None, len(nms) == len(set(nms)))

        for key, value in dot.items():
            setattr(x, key, value)

        return x
    
    def rm_cols(self, x, cols, skip_absent=False, by_ref=False):
        assert is_flag(skip_absent) and is_flag(by_ref)

        if skip_absent:
            cols = list(set(cols) & set(x.columns))
        else:
            cols = list(set(cols))

        if len(cols) == 0:
            return x

        assert has_cols(x, cols)

        if not by_ref:
            x = x.copy()

        if self.is_id_tbl(x) and any(col in meta_vars(x) for col in cols):
            ptyp = self.as_ptype(x)
        else:
            ptyp = None

        x = x.drop(columns=cols) # Remove columns specified as cols

        self.try_reclass(x, ptyp)

        return x