from copy import deepcopy
import numpy as np
import pandas as pd
from typing import List, Union
from pandas._typing import Axes, Dtype



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


class IDTbl(pd.DataFrame):

    _metadata = ["id_vars"]

    def __init__(
        self,
        data=None,
        index: Axes = None,
        columns: Axes = None,
        dtype: Dtype = None,
        copy: bool = None,
        id_vars: Union[str, List[str]] = 0,
    ):
        super().__init__(
            data,
            index,
            columns,
            dtype,
            copy,
        )
        if isinstance(id_vars, str):
            id_vars = [id_vars]
        elif isinstance(id_vars, int):
            id_vars = [self.columns[id_vars]] # TODO: allow for a list of integer
        if isinstance(id_vars, list):
            self.id_vars = id_vars
        else: 
            raise TypeError(f"Expected `id_vars` to be (a list) of str, got {id_vars.__class__}")

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

    def __repr__(self):
        repr =  f"# <IDTbl>: {self.shape[0]} x {self.shape[1]}\n"
        repr += f"# ID var:  {self.id_vars}\n"
                 
        units = {s.name: s.unit for _, s in self.items() if hasattr(s, 'unit')}
        if len(units) > 0:
            repr += "# Units:   "
            for n, u in units.items():
                repr += f"`{n}` [{u}]"
            repr += "\n"

        repr += super().__repr__()
        return repr