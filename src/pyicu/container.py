import warnings
import pandas as pd
from typing import List, Union
from pandas._typing import Axes, Dtype, IndexLabel

from .utils import print_list

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


class pyICUTbl(pd.DataFrame):
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
            self.id_var = id_var
        move_column(self, self.id_var, 0)

    @property
    def data_vars(self) -> List[str]:
        return [c for c in self.columns if c not in self.meta_vars]

    @property
    def data_var(self) -> str:
        data_vars = self.data_vars
        if len(data_vars) > 1:
            raise ValueError(f"expected a single data variable for tbl but found multiple {print_list(data_vars)}")
        return data_vars[0]

    def to_pandas(self) -> pd.DataFrame:
        """Return the underlying pandas.DataFrame.

        Returns:
            pandas.DataFrame
        """
        return pd.DataFrame(self)

    def __repr__(self):
        repr = ""
        units = {s.name: s.unit for _, s in self.items() if hasattr(s, "unit")}
        if len(units) > 0:
            repr += "# Units:   "
            for n, u in units.items():
                repr += f"`{n}` [{u}]"
            repr += "\n"

        repr += super().__repr__()
        return repr


class IdTbl(pyICUTbl):
    @property
    def _constructor(self):
        return IdTbl

    @property
    def meta_vars(self) -> List[str]:
        return [self.id_var]

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
        repr = f"# <IDTbl>: {self.shape[0]} x {self.shape[1]}\n"
        repr += f"# ID var:  {self.id_var}\n"
        repr += super().__repr__()
        return repr


class TsTbl(pyICUTbl):

    _metadata = ["id_var", "index_var"]

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
                self.index_var = parse_columns(index_var, self.columns)
            else:
                raise TypeError(f"expected `index_var` to be str, int, or None, ", f"got {index_var.__class__}")
            move_column(self, self.index_var, 1)

    @property
    def _constructor(self):
        return TsTbl

    @property
    def _constructor_sliced(self):
        return pyICUSeries

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

    def __repr__(self):
        repr = f"# <TSTbl>:    {self.shape[0]} x {self.shape[1]}\n"
        repr += f"# ID var:     {self.id_var}\n"
        repr += f"# Index var:  {self.index_var if hasattr(self, 'index_var') else 'None'}\n"
        repr += super().__repr__()
        return repr


def move_column(df: pd.DataFrame, col_name: str, pos: int = 0) -> None:
    col = df.pop(col_name)
    df.insert(pos, col_name, col)
