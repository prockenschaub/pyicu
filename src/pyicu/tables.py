import warnings
import pandas as pd
from typing import List, Union
from pandas._typing import Axes, Dtype, IndexLabel


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
            raise ValueError(f"Could not find column {x}.")
        return x
    elif isinstance(x, int):
        return columns[x]
    elif isinstance(x, list):
        return [columns[i] if isinstance(i, int) else i for i in x]
    else: 
        raise TypeError(f"Expected int or list, got {x.__class__}")


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
        if id_var is None:
            id_var = 0
        id_var = parse_columns(id_var, self.columns)
        self.id_var = id_var

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


class IdTbl(pyICUTbl):

    @property
    def _constructor(self):
        return IdTbl

    @property
    def meta_vars(self):
        return list(set(self.columns).difference(set(self.id_var)))

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
            warnings.warn(f"Automatically merged on column {self.id_var}.")
            return super().merge(right, how, on=self.id_var, *args, **kwargs)
        else:
            return super().merge(right, how, on, left_on, right_on, *args, **kwargs)

    def __repr__(self):
        repr =  f"# <IDTbl>: {self.shape[0]} x {self.shape[1]}\n"
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
        guess_index_var: bool = False
    ):
        super().__init__(
            data,
            index,
            columns,
            dtype,
            copy,
            id_var
        )
        if isinstance(index_var, (str, int)):
            self.index_var = parse_columns(index_var, self.columns)
        elif index_var is None and guess_index_var:
            # NOTE: need extra flag to distinguish between a new object init 
            #       where we actually want to infer the index and between pandas 
            #       internal subsetting functions that are called before 
            #       __finalize__
            time_vars = self.select_dtypes(include='timedelta').columns
            if len(time_vars) != 1:
                raise ValueError(
                    "In order to automatically determine the index column,",
                    "exactly one `timedelta` column is required."
                )

        else: 
            raise TypeError(
                f"Expected `index_var` to be str, int, or None, ",
                f"got {index_var.__class__}"
            )
        
    @property
    def _constructor(self):
        return TsTbl

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
            warnings.warn(f"Automatically merged on columns {[self.id_var, self.index_var]}.")
            return super().merge(right, how, on=[self.id_var, self.index_var], *args, **kwargs)
        else:
            return super().merge(right, how, on, left_on, right_on, *args, **kwargs)

    def __repr__(self):
        repr =  f"# <TSTbl>:    {self.shape[0]} x {self.shape[1]}\n"
        repr += f"# ID var:     {self.id_var}\n"
        repr += f"# Index var:  {self.index_var}\n"
        repr += super().__repr__()
        return repr