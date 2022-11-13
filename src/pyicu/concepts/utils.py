import os
import json
from operator import add as concat
from pathlib import Path
from functools import reduce
from typing import List, Dict, Callable
from importlib_resources import files
from pyicu.utils import intersect

from . import Concept, CONCEPT_CLASSES, Item, ITEM_CLASSES

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

def parse_items(src: str, x: Dict) -> List[Item]:
    x = x.copy()
    res = []
    for y in x:
        ItemClass = ITEM_CLASSES.get(y.pop("class", None))
        if ItemClass is None:
            ItemClass = ITEM_CLASSES['sel_itm']
        res += [ItemClass(src, **y)]
    return res

def parse_concept(name: str, x: Dict) -> Concept:
    x = x.copy()

    # Parse items that define the concept
    items = []
    srcs = x.pop("sources", None)
    if srcs is not None:
        for src, y in srcs.items():
            items += parse_items(src, y)
    
    # Build the concept class
    class_nm = x.pop("class", None)
    if isinstance(class_nm, list): # To deal with unt_cncpt
        class_nm = class_nm[0]
    ConceptClass = CONCEPT_CLASSES.get(class_nm)
    if ConceptClass is None:
        ConceptClass = CONCEPT_CLASSES['num_cncpt']
    elif ConceptClass == CONCEPT_CLASSES['rec_cncpt']:
        items = x.pop("concepts", None) 

    return ConceptClass(name, items, **x)