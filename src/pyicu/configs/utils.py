import os
import sys
import platform
import json
import pyarrow as pa
from typing import List, Dict, Callable
from pathlib import Path
from operator import add as concat
from functools import reduce
from typing import List, Dict, Callable
from importlib_resources import files

def check_attributes_in_dict(
    dict: Dict, 
    att_names: str | List[str], 
    cfg_name: str, 
    cfg_type: str
):
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


def parse_col_types(type: str | pa.DataType) -> pa.DataType:
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
        # TODO: 'us' needed to read in MIMIC IV procedureevents. 
        #       can we safely convert this to 's' and save space later?
        res = pa.timestamp('us')
    # pyarrow types
    else:
        try:
            res = getattr(sys.modules["pyarrow"], type)
        except:
            raise ValueError(f"got unrecognised column spec {type}")

    return res



def default_config_path() -> List[Path]:
    """Path to directories with configs that are defined by default by pyicu.

    Returns:
        list of directory paths
    """
    return files("config")._paths


def user_config_path() -> List[Path]:
    """Path to directories with configs that are defined by the user.

    Users can change which directories are search by setting the 
    "PYICU_CONFIG_PATH" environment variables.

    Example: 
        import os
        os.setenv("PYICU_CONFIG_PATH", "/Users/johndoe/pyicu_configs/")

    Returns:
        list of directory paths
    """
    res = os.getenv("PYICU_CONFIG_PATH", "")
    return [Path(p) for p in res.split()]


def config_paths() -> List[Path]:
    """Get both default and user defined config directories.

    Returns:
        list of directory paths
    """
    default = default_config_path()
    user = user_config_path()
    return  (user if user is not None else []) + default


def get_config(
    name: str = "concept-dict", 
    cfg_dirs: Path | List[Path] | None = None,
    combine_fun: Callable = concat
) -> List[Dict]:
    """Read one or more JSON config files from a list of directories. 

    All config files must have be named {name}.json. 

    Args:
        name: name of config files. Defaults to "concept-dict".
        cfg_dirs: paths to config directories. Defaults to None.
        combine_fun: function used to reconcile concepts from multiple files, e.g., in the case of 
            a concept being defined by more than one file. Defaults to simply concatinating the lists, 
            keeping any duplicates.

    Returns:
        list of configurations parsed from JSON
    """
    def read_if_exists(x, **kwargs):
        if x.exists():
            f = open(x)
            return json.load(f)

    if cfg_dirs is None:
        cfg_dirs = config_paths()
    if isinstance(cfg_dirs, Path):
        cfg_dirs = [cfg_dirs]
    res = [read_if_exists(d/f"{name}.json") for d in cfg_dirs]

    if combine_fun is None:
        return res
    else:
        return reduce(combine_fun, res)

