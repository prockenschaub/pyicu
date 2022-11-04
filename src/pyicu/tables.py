import warnings
import pandas as pd
from typing import Any, List, Union
from pandas._typing import Axes, Dtype, IndexLabel

from .utils import enlist


class pyICUSeries(pd.Series):
    _metadata = ["unit"]

    def __init__(self, *args, unit=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if unit is not None: 
            self.unit = unit

    @property
    def _constructor(self):
        return pyICUSeries

    def __repr__(self) -> str:
        unit = ""
        if hasattr(self, "unit"):
            unit += f"\nunit: {self.unit}"
        return super().__repr__() + unit


def parse_columns(x: Union[str, int, List], columns):
    if isinstance(x, int):
        return columns[x]
    elif isinstance(x, list):
        return [columns[i] if isinstance(i, int) else i for i in x]
    else: 
        raise TypeError(f"Expected int or list, got {x.__class__}")


class pyICUTbl(pd.DataFrame):
    _metadata = ["id_vars"]

    def __init__(
        self,
        data=None,
        index: Axes = None,
        columns: Axes = None,
        dtype: Dtype = None,
        copy: bool = None,
        id_vars: Union[str, int, List] = 0,
    ):
        super().__init__(
            data,
            index,
            columns,
            dtype,
            copy,
        )
        id_vars = enlist(id_vars)
        id_vars = parse_columns(id_vars, self.columns)
        self.id_vars = id_vars

    def to_pandas(self) -> pd.DataFrame:
        """Return the underlying pandas.DataFrame.

        Returns:
            pandas.DataFrame
        """
        return pd.DataFrame(self)

    def __repr__(self):
        repr = ""
        units = {s.name: s.unit for _, s in self.items() if hasattr(s, 'unit')}
        if len(units) > 0:
            repr += "# Units:   "
            for n, u in units.items():
                repr += f"`{n}` [{u}]"
            repr += "\n"

        repr += super().__repr__()
        return repr


class IDTbl(pyICUTbl):

    @property
    def _constructor(self):
        return IDTbl

    @property
    def meta_vars(self):
        return list(set(self.columns).difference(set(self.id_vars)))

    def merge(
        self,
        right: Union[pd.DataFrame, pd.Series],
        how: str = "inner",
        on: Union[IndexLabel, None] = None,
        left_on: Union[IndexLabel, None] = None,
        right_on: Union[IndexLabel, None] = None,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        if on is None and left_on is None and right_on is None:
            warnings.warn(f"Automaically merged on column(s) {self.id_vars}.")
            return super().merge(right, how, on=self.id_vars, *args, **kwargs)
        else:
            return super().merge(right, how, on, left_on, right_on, *args, **kwargs)

    def __repr__(self):
        repr =  f"# <IDTbl>: {self.shape[0]} x {self.shape[1]}\n"
        repr += f"# ID var:  {self.id_vars}\n"
        repr += super().__repr__()
        return repr


class TSTbl(pyICUTbl):

    _metadata = ["id_vars", "index_var"]

    def __init__(
        self,
        data=None,
        index: Axes = None,
        columns: Axes = None,
        dtype: Dtype = None,
        copy: bool = None,
        id_vars: Union[str, int, List] = 0,
        index_var: Union[str, int] = None
    ):
        super().__init__(
            data,
            index,
            columns,
            dtype,
            copy,
            id_vars
        )
        if isinstance(index_var, (str, int)):
            self.index_var = parse_columns(index_var, self.columns)
        elif index_var is None:
            raise NotImplementedError()
        else: 
            raise TypeError(f"Expected `index_var` to be str, int, or None, got {index_var.__class__}")
        
    @property
    def _constructor(self):
        return IDTbl

    @property
    def _constructor_sliced(self):
        return pyICUSeries

    @property
    def meta_vars(self):
        return list(set(self.columns).difference(set(self.id_vars)))

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
        **kwargs
    ) -> pd.DataFrame:
        if on is None and left_on is None and right_on is None:
            warnings.warn(f"Automaically merged on column(s) {self.id_vars}.")
            return super().merge(right, how, on=self.id_vars, *args, **kwargs)
        else:
            return super().merge(right, how, on, left_on, right_on, *args, **kwargs)

    def __repr__(self):
        repr =  f"# <TSTbl>:    {self.shape[0]} x {self.shape[1]}\n"
        repr += f"# ID var:     {self.id_vars}\n"
        repr += f"# Index var:  {self.index_var}\n"
        repr += super().__repr__()
        return repr