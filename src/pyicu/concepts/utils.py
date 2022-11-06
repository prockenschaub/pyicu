import os
import json
from operator import add as concat
from pathlib import Path
from functools import reduce
from typing import List, Dict, Callable
from importlib_resources import files
from pyicu.utils import intersect
from .item import ITEM_MAP, SelItem

def default_config_path() -> List[Path]:
    return files("config")._paths

def user_config_path() -> List[Path]:
    res = os.getenv("RICU_CONFIG_PATH", "")
    return [Path(p) for p in res.split()]

def config_paths() -> List[Path]:
    default = default_config_path()
    user = user_config_path()
    return  (user if user is not None else []) + default

def get_config(
    name: str | None = "concept-dict", 
    cfg_dirs: Path | List[Path] | None = None,
    combine_fun: Callable = concat,
    **kwargs
) -> List:
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

def combine_sources(x: Dict, y: Dict, nme: str) -> Dict:
    x, y = x.copy(), y.copy()
    
    if x.get("class") == "rec_cncpt":
        raise ValueError(f"Cannot merge recursive concept `{nme}`")

    if not isinstance(y.get("sources"), dict) or len(y) != 1 or "sources" not in y.keys():
        raise ValueError(f"Cannot merge concept `{nme}` due to malformed `sources` entry")

    x['sources'] = x.get("sources") | y.get("sources")
    return x

def combine_concepts(x: Dict, y: Dict) -> Dict:
    dups = intersect(list(x.keys()), list(y.keys()))
    if len(dups) > 0:
        for dup in dups:
            x[dup] = combine_sources(x[dup], y[dup], dup)
            y.pop(dup)
    return x | y

def read_dictionary(name: str, cfg_dirs: Path | List[Path] = None) -> Dict:
    if cfg_dirs is None: 
        cfg_dirs = []
    elif isinstance(cfg_dirs, Path):
        cfg_dirs = [cfg_dirs]
    return get_config(name, config_paths() + cfg_dirs, combine_concepts)


def parse_dictionary(dictionary, src, concepts=None):
    dictionary = {
        k:v for k, v in dictionary.items() 
        if concepts is not None and k in concepts
    }

def parse_item(src, x):
    res = []
    for item_spec in x:
        ItemClass = ITEM_MAP.get(item_spec.pop("class", None))
        if ItemClass is None:
            ItemClass = SelItem
        res += [ItemClass(src, **item_spec)]
    return res