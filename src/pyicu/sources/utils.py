from typing import List, Dict
import re
import pandas as pd
import pyarrow as pa

from ..utils import enlist
from ..container import IdTbl


def defaults_to_str(defaults: Dict) -> str:
    """Render SrcTbl non-time var defaults as a readable string representation

    See also: `SrcTbl`

    Args:
        defaults: table defaults stored as a dictionary

    Returns:
        string representation
    """
    repr = ""
    for d, v in list(defaults.items()):
        if d != "time_vars":
            if repr != "":
                repr += ", "
            repr += f"`{v}` ({re.sub('_vars?', '', d)})"
    return repr


def time_vars_to_str(defaults: Dict) -> str:
    """Render SrcTbl time variables as a readable string representation

    See also: `SrcTbl`

    Args:
        defaults: table defaults stored as a dictionary

    Returns:
        string representation
    """
    repr = ""
    time_vars = enlist(defaults["time_vars"])

    for v in time_vars:
        if repr != "":
            repr += ", "
        repr += f"`{v}`"
    return repr


def order_rename(df: pd.DataFrame, id_var: List[str], st_var: List[str], ed_var: List[str]) -> pd.DataFrame:
    """Helper function for creating Id windows that orders and renames columns

    Args:
        df: Id windows of a Src
        id_var: names of the Id variables
        st_var: names of the start time variables
        ed_var: names of the end time variables

    Returns:
        input `DataFrame` with renamed `id_vars` first, then `st_vars`, then `ed_vars`
    """

    def add_suffix(x: List[str], s: str):
        return [f"{i}_{s}" for i in x]

    old_names = id_var + st_var + ed_var
    new_names = id_var + add_suffix(id_var, "start") + add_suffix(id_var, "end")
    df = df[old_names]  # Reorder
    df = df.rename({o: n for o, n in zip(old_names, new_names)}, axis="columns")
    return IdTbl(df)


def pyarrow_types_to_pandas(x: pa.DataType):
    mapping = {
        pa.int16(): pd.Int16Dtype(),
        pa.int32(): pd.Int32Dtype(),
        pa.int64(): pd.Int64Dtype(),
    }
    return mapping.get(x)
