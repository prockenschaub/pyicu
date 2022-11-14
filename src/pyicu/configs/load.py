import os
import json
from typing import List, Dict, Callable
from pathlib import Path
from operator import add as concat
from functools import reduce
from importlib_resources import files

from ..utils import enlist
from . import SrcCfg


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
    return (user if user is not None else []) + default


def get_config(
    name: str = "concept-dict", cfg_dirs: Path | List[Path] | None = None, combine_fun: Callable = concat
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
    res = [read_if_exists(d / f"{name}.json") for d in cfg_dirs]

    if combine_fun is None:
        return res
    else:
        return reduce(combine_fun, res)


def combine_srcs(x: Dict, y: Dict):
    # NOTE: this differs from how ricu combines sources, as ricu favours x.
    #       this way of favouring y should allow later definitions to overwrite
    #       earlier ones, which may be more sensible?
    # TODO: double check in which order configs are loaded (default first or
    #       user specified first?)
    return x.copy().update(y)


def read_src_cfg(
    src: str | List[str] | None, name: str = "data-sources", cfg_dirs: Path | List[Path] = None
) -> Dict | List[Dict]:
    src = enlist(src)
    if cfg_dirs is None:
        cfg_dirs = config_paths()
    elif isinstance(cfg_dirs, Path):
        cfg_dirs = [cfg_dirs]

    res = get_config(name, cfg_dirs, combine_srcs)

    if src is not None:
        res = [x for x in res if x["name"] in src]
        if len(res) == 1:
            res = res[0]

    return res


def load_src_cfg(
    src: str | List[str] | None, name: str = "data-sources", cfg_dirs: Path | List[Path] = None
) -> SrcCfg | List[SrcCfg]:
    res = read_src_cfg(src, name, cfg_dirs)
    if isinstance(res, list):
        res = [SrcCfg.from_dict(x) for x in res]
    else:
        res = SrcCfg.from_dict(res)
    return res
