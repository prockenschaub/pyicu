from __future__ import annotations
import warnings
import pandas as pd
from typing import List, Union, Type
from pandas._typing import Axes, Dtype, IndexLabel

from .utils import enlist, print_list, new_names
from .interval import change_interval, print_interval, mins


class MeasuredSeries(pd.Series):
    _metadata = ["unit"]

    def __init__(self, *args, unit=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if unit is not None:
            self.unit = unit

    @property
    def _constructor(self):
        return MeasuredSeries

    def __repr__(self) -> str:
        unit = ""
        if hasattr(self, "unit"):
            unit += f"\nunit: {self.unit}"
        return super().__repr__() + unit


def parse_columns(x: Union[str, int, List], columns):
    if isinstance(x, str):
        if not x in columns:
            raise ValueError(f"could not find column {x}.")
        return x
    elif isinstance(x, int):
        return columns[x]
    elif isinstance(x, list):
        return [columns[i] if isinstance(i, int) else i for i in x]
    else:
        raise TypeError(f"expected str, int, or list, got {x.__class__}")


class IdTbl(pd.DataFrame):
    _metadata = ["id_var"]

    def __init__(
        self,
        data=None,
        index: Axes = None,
        columns: Axes = None,
        dtype: Dtype = None,
        copy: bool = None,
        id_var: Union[str, int] = None,
    ):
        super().__init__(
            data,
            index,
            columns,
            dtype,
            copy,
        )
        if id_var is None and not hasattr(self, "id_var"):
            id_var = 0
        if id_var is not None:
            id_var = parse_columns(id_var, self.columns)
            self.set_id_var(id_var)

    @property
    def _constructor(self):
        return IdTbl

    @property
    def _constructor_sliced(self):
        return MeasuredSeries

    @property
    def meta_vars(self) -> List[str]:
        return [self.id_var]

    @property
    def data_vars(self) -> List[str]:
        return [c for c in self.columns if c not in self.meta_vars]

    @property
    def data_var(self) -> str:
        data_vars = self.data_vars
        if len(data_vars) > 1:
            raise ValueError(f"expected a single data variable for tbl but found multiple {print_list(data_vars)}")
        return data_vars[0]

    @property
    def time_vars(self) -> List[str] | None:
        try:
            times = self.select_dtypes(include=["datetime64", "timedelta64"])
        except IndexError as e:
            return None
        return times.columns.to_list()

    def set_id_var(self, id_var: str):
        if not id_var in self.columns:
            raise ValueError(f"tried to change Id var to unknown column {id_var}")
        self.id_var = id_var
        if len(self.columns) > 0 and self.columns[0] != id_var:
            move_column(self, self.id_var, 0)
    
    def change_interval(self, new_interval: pd.Timedelta, cols: str | List[str] | None = None) -> "IdTbl":
        if cols is not None:
            for c in cols:
                self[c] = change_interval(self[c], new_interval)
        return self

    def change_id(
        self, 
        src: "Src", 
        target_id, 
        keep_old_id: bool = True, 
        id_type: bool = False, 
        **kwargs
    ) -> "IdTbl" | "TsTbl":
        # TODO: enable id_type
        orig_id = self.id_var
        if target_id == orig_id:
            return self

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
    ) -> "IdTbl" | "TsTbl":
        idx = self.id_var

        cols = enlist(cols)
        if cols is not None:
            sft = new_names(self)
        else:
            sft = None

        if dir == "down":
            map = src.id_map(target_id, idx, sft, None)
        else:
            map = src.id_map(idx, target_id, sft, None)

        res = self.merge(map, on=idx, **kwargs)

        if cols is not None:
            for c in cols:
                res[c] = res[c] - res[sft]
            res.drop(columns=sft, inplace=True)

        res.set_id_var(target_id)
        return res

    def upgrade_id(self, src: "Src", target_id: str, cols: str | List[str] | None = None, **kwargs) -> "IdTbl":
        if cols is None:
            cols = self.time_vars
        return self._upgrade_id_helper(src, target_id, cols, **kwargs)

    def _upgrade_id_helper(self, src: "Src", target_id: str, cols: str | List[str] | None = None, **kwargs) -> "IdTbl":
        return self._change_id_helper(src, target_id, cols, "up", **kwargs)

    def downgrade_id(self, src: "Src", target_id: str, cols: str | List[str] | None = None, **kwargs):
        if cols is None:
            cols = self.time_vars
        return self._downgrade_id_helper(src, target_id, cols, **kwargs)

    def _downgrade_id_helper(self, tbl, target_id, cols, **kwargs):
        return self._change_id_helper(tbl, target_id, cols, "down", **kwargs)

    def to_pandas(self) -> pd.DataFrame:
        """Return the underlying pandas.DataFrame.

        Returns:
            pandas.DataFrame
        """
        return pd.DataFrame(self)

    def merge(
        self,
        right: Union[pd.DataFrame, pd.Series],
        how: str = "inner",
        on: Union[IndexLabel, None] = None,
        left_on: Union[IndexLabel, None] = None,
        right_on: Union[IndexLabel, None] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        if on is None and left_on is None and right_on is None:
            warnings.warn(f"automatically merged on column {self.id_var}.")
            return super().merge(right, how, on=self.id_var, *args, **kwargs)
        else:
            return super().merge(right, how, on, left_on, right_on, *args, **kwargs)

    def __repr__(self):
        repr = f"# <IdTbl>: {self.shape[0]} x {self.shape[1]}\n"
        repr += f"# Id var:     {self.id_var}\n"
        repr += self._repr_data()
        return repr

    def _repr_data(self):
        repr = ""
        units = {s.name: s.unit for _, s in self.items() if hasattr(s, "unit")}
        if len(units) > 0:
            repr += "# Units:   "
            for n, u in units.items():
                repr += f"`{n}` [{u}]"
            repr += "\n"

        repr += super().__repr__()
        return repr


class TsTbl(IdTbl):

    _metadata = ["id_var", "index_var", "interval"]

    def __init__(
        self,
        data=None,
        index: Axes = None,
        columns: Axes = None,
        dtype: Dtype = None,
        copy: bool = None,
        id_var: Union[str, int] = None,
        index_var: Union[str, int] = None,
        guess_index_var: bool = False,
        interval: pd.Timedelta = None,
    ):
        super().__init__(data, index, columns, dtype, copy, id_var)
        if index_var is None and not hasattr(self, "index_var"):
            if guess_index_var:
                # NOTE: need extra flag to distinguish between a new object init
                #       where we actually want to infer the index and between pandas
                #       internal subsetting functions that are called before
                #       __finalize__
                time_vars = self.select_dtypes(include="timedelta").columns
                if len(time_vars) != 1:
                    raise ValueError(
                        "to automatically determine the index column,", "exactly one `timedelta` column is required."
                    )
                index_var = time_vars[0]
        if index_var is not None:
            if isinstance(index_var, (str, int)):
                index_var = parse_columns(index_var, self.columns)
            else:
                raise TypeError(f"expected `index_var` to be str, int, or None, ", f"got {index_var.__class__}")
            self.set_index_var(index_var)

        if interval is None:
            interval = pd.Timedelta(1, "h")
        self.interval = interval

    @property
    def _constructor(self):
        return TsTbl

    @property
    def _constructor_sliced(self):
        return MeasuredSeries

    @property
    def meta_vars(self) -> List[str]:
        return [self.id_var, self.index_var]

    def to_pandas(self) -> pd.DataFrame:
        """Return the underlying pandas.DataFrame.

        Returns:
            pandas.DataFrame
        """
        return pd.DataFrame(self)

    def merge(
        self,
        right: Union[pd.DataFrame, pd.Series],
        how: str = "inner",
        on: Union[IndexLabel, None] = None,
        left_on: Union[IndexLabel, None] = None,
        right_on: Union[IndexLabel, None] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        if on is None and left_on is None and right_on is None:
            warnings.warn(f"automatically merged on columns {[self.id_var, self.index_var]}.")
            return super().merge(right, how, on=[self.id_var, self.index_var], *args, **kwargs)
        else:
            return super().merge(right, how, on, left_on, right_on, *args, **kwargs)

    def set_index_var(self, index_var: str):
        if not index_var in self.columns:
            raise ValueError(f"tried to change Index var to unknown column {index_var}")
        self.index_var = index_var
        if len(self.columns) > 1 and self.columns[1] != index_var:
            move_column(self, self.index_var, 1)

    def change_interval(self, new_interval: pd.Timedelta, cols: str | List[str] | None = None) -> Type["TsTbl"]:
        if cols is None:
            cols = enlist(self.index_var)
        super().change_interval(new_interval, cols)
        if self.index_var in cols:
            self.interval = new_interval
        return self

    def _upgrade_id_helper(self, src: "Src", target_id, cols, **kwargs):
        if self.index_var not in cols:
            raise ValueError(f"index var `{self.index_var}` must be part of the cols parameter")

        if self.interval != mins(1):
            warnings.warn("Changing the ID of non-minute resolution data will change the interval to 1 minute")

        sft = new_names(self)
        id = self.id_var
        ind = self.index_var

        map = src.id_map(id, target_id, sft, ind)

        # TODO: pandas currently does not have a direct equivalent to R data.table's rolling join
        #       It can be approximated with pandas.merge_asof but needs additional sorting and
        #       does not propagate rolls outside of ends (see data.table's `rollends` parameter).
        #       this code may be slow and may need revisiting/refactoring.
        tbl = self.sort_values(ind)
        map = map.sort_values(ind)
        fwd = pd.merge_asof(tbl, map, on=ind, by=id, direction="forward")
        not_matched = fwd[fwd[target_id].isna()][tbl.columns]
        bwd = pd.merge_asof(not_matched, map, on=ind, by=id, direction="backward")
        res = pd.concat((fwd[~fwd[target_id].isna()], bwd), axis=0)
        res = res.sort_values([target_id, ind])

        for c in cols:
            res[c] = res[c] - res[sft]

        res.drop(columns=sft, inplace=True)
        res = TsTbl(res, id_var=target_id, index_var=ind, interval=mins(1))
        return res

    def _downgrade_id_helper(self, src: "Src", target_id, cols, **kwargs):
        if self.index_var not in cols:
            raise ValueError(f"index var `{self.index_var}` must be part of the cols parameter")

        if self.interval != mins(1):
            warnings.warn("Changing the ID of non-minute resolution data will change the interval to 1 minute")

        res = self._change_id_helper(src, target_id, cols, "down", **kwargs)
        res.set_index_var(self.index_var)  # reset index var
        res.change_interval(mins(1), cols=cols)
        return res

    def __repr__(self):
        repr = f"# <TsTbl>:    {self.shape[0]} x {self.shape[1]}\n"
        repr += f"# Id var:     {self.id_var}\n"
        repr += f"# Index var:  {self.index_var if hasattr(self, 'index_var') else 'N/A'} ({print_interval(self.interval)})\n"
        repr += super()._repr_data()
        return repr


def move_column(df: pd.DataFrame, col_name: str, pos: int = 0) -> None:
    col = df.pop(col_name)
    df.insert(pos, col_name, col)
