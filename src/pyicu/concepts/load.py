from pathlib import Path
from typing import List, Dict
from pyicu.utils import intersect

from . import Concept, concept_class, Item, item_class
from ..configs.load import config_paths, get_config


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
        raise ValueError(f"cannot merge recursive concept `{nme}`")

    if not isinstance(y.get("sources"), dict) or "sources" not in y.keys():
        raise ValueError(f"cannot merge concept `{nme}` due to malformed `sources` entry")

    if len(y) != 1:
        raise ValueError(f"cannot merge concept `{nme}` due to non-`sources` entry in both definition")

    x["sources"] = x.get("sources") | y.get("sources")
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


def read_dictionary(name: str = "concept-dict", cfg_dirs: Path | List[Path] = None) -> Dict:
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


# TODO: Look at whether these should be integrated into Item/Concept classes
#       as `from_dict` methods
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
    if isinstance(class_nm, list):  # To deal with unt_cncpt
        class_nm = class_nm[0]
    ConceptClass = concept_class(class_nm)
    if ConceptClass == concept_class("rec_cncpt"):
        items = x.pop("concepts", None)

    return ConceptClass(name, items, **x)
