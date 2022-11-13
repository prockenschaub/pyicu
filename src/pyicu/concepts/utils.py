import os
import json
from operator import add as concat
from pathlib import Path
from functools import reduce
from typing import List, Dict, Callable
from importlib_resources import files
from pyicu.utils import intersect

from . import Concept, concept_class, Item, item_class

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


def combine_sources(x: Dict, y: Dict, nme: str) -> Dict:
    """Merge the items of a new concept definition into an existing concept definition.

    Items are defined in the 'sources' element of a concept definition. 

    Args:
        x: an existing concept definition
        y: another concept definition that should be merged
        nme: concept name

    Raises:
        ValueError: if a recursive concept is passed, as they cannot be sensibly merged
        ValueError: if the 'sources' element of `y` is not a dict or no 'sources' element is present
        ValueError: if `y` contains elements other than 'sources' TODO: revisit the need for this requirement

    Returns:
        a concept with all elements of `x` and the merged 'sources' of `x` and `y`
    """
    x, y = x.copy(), y.copy()
    
    if x.get("class") == "rec_cncpt":
        raise ValueError(f"Cannot merge recursive concept `{nme}`")

    if not isinstance(y.get("sources"), dict) or "sources" not in y.keys():
        raise ValueError(f"Cannot merge concept `{nme}` due to malformed `sources` entry")

    if len(y) != 1:
        raise ValueError(f"Cannot merge concept `{nme}` due to non-`sources` entry in both definition")

    x['sources'] = x.get("sources") | y.get("sources")
    return x


def combine_concepts(x: Dict, y: Dict) -> Dict:
    """Merge two concepts.

    Args:
        x: an existing concept definition
        y: another concept definition that should be merged

    Note: y must only define new or alternative items. All other information, such as target or min/max 
        may not be updated. TODO: revisit the need for this requirement

    Returns:
        a merged concept
    """
    dups = intersect(list(x.keys()), list(y.keys()))
    if len(dups) > 0:
        for dup in dups:
            x[dup] = combine_sources(x[dup], y[dup], dup)
            y.pop(dup)
    return x | y


def read_dictionary(name: str, cfg_dirs: Path | List[Path] = None) -> Dict:
    """Read clinical concept definitions from from a list of directories. 

    Args:
        name: name of config files. Defaults to "concept-dict".
        cfg_dirs: paths to addconfig directories. Defaults to None.

    Returns:
        dictionary of clinical concepts read from JSON
    """
    if cfg_dirs is None: 
        cfg_dirs = config_paths()
    elif isinstance(cfg_dirs, Path):
        cfg_dirs = [cfg_dirs]
    return get_config(name, cfg_dirs, combine_concepts)


def parse_items(src: str, x: List[Dict]) -> List[Item]:
    """Parse an `Item` definition read from JSON

    Args:
        src: name of a data source, e.g., 'miiv' for MIMIC IV.
        x: a list of item definitions

    Returns:
        a parsed `Item` class
    """
    x = x.copy()
    res = []
    for y in x:
        ItemClass = item_class(y.pop("class", None))
        res += [ItemClass(src, **y)]
    return res

def parse_concept(name: str, x: Dict) -> Concept:
    """Parse a `Concept` definition read from JSON

    Args:
        name: name of the concept
        x: a concept definition

    Returns:
        a parsed `Concept` class
    """
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
    ConceptClass = concept_class(class_nm)
    if ConceptClass == concept_class('rec_cncpt'):
        items = x.pop("concepts", None) 

    return ConceptClass(name, items, **x)
