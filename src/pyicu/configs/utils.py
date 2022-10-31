import os
import sys
import platform
import pyarrow as pa
from typing import List, Dict, Union
from pathlib import Path

def check_attributes_in_dict(dict: Dict, att_names: Union[str, List[str]], cfg_name: str, cfg_type: str):
    """_summary_

    Args:
        dict (_type_): _description_
        att_name (_type_): _description_
        cfg_name (_type_): _description_
        cfg_type (_type_): _description_

    Raises:
        ValueError: _description_
    """
    if isinstance(att_names, str):
        att_names = [att_names]
    for att in att_names:
        if not att in dict.keys():
            raise ValueError(f'No `{att}` attribute provided for {cfg_type} config {cfg_name}')


def get_data_dir(subdir: str = None, create: bool = True) -> Path:
    res = os.getenv("PYICU_DATA_PATH")

    if res is None:
        system = platform.system()
        if system == "Darwin":
            res = os.getenv("XDG_DATA_HOME", default="~/Library/Application Support")
        elif system == "Windows":
            res = os.getenv("LOCALAPPDATA", default=os.getenv("APPDATA"))
        else:
            res = os.getenv("XDG_DATA_HOME", default="~/.local/share")

        res += "ricu"
  
    if subdir is not None:
        if not isinstance(subdir, str):
            raise TypeError(f'expected `subdir` to be str, got {subdir.__class__}')
        res += subdir
  
    res = Path(res)

    if create:
        res.mkdir(parents=True, exist_ok=True)

    return res


def parse_col_types(type: Union[str,  pa.DataType]) -> pa.DataType:
    if isinstance(type, pa.DataType):
        return type
    
    # readr types for compatibility with ricu
    if type == "col_integer":
        res = pa.int32()
    elif type == "col_double":
        res = pa.float32()
    elif type == "col_logical":
        res = pa.bool_()
    elif type == "col_character":
        res = pa.string()
    elif type == "col_datetime":
        res = pa.timestamp(unit='s')
    # pyarrow types
    else:
        try:
            res = getattr(sys.modules["pyarrow"], type)
        except:
            raise ValueError(f"got unrecognised column spec {type}")

    return res
