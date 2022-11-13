from typing import List, Dict, Type
from pathlib import Path

from ..data import Src
from ..container import pyICUTbl
from ..utils import enlist, print_list
from .concept import Concept
from .load import read_dictionary, parse_concept

class ConceptDict():
    """Collection of medical concepts

    See also: pyutils.concepts.Concept

    Args:
        concepts: list of included concepts. Defaults to None.
    """
    def __init__(self, concepts: List[Concept]) -> None:
        self.concepts = concepts

    def load_concepts(self, concepts: str | List[str], src: Src) -> Dict[str, pyICUTbl]:
        """Load data for a concept from a data source

        Args:
            concepts: the names of one or more concepts to load; concept must be defined in this dictionary.
            src: a data source, e.g., MIMIC IV

        Returns:
            dict with names and data of the loaded concepts
        """
        concepts = enlist(concepts)
        not_avail = list(set(concepts)-set(self.concepts))
        if len(not_avail) > 0:
            raise ValueError(f"Attempted to load concepts that haven't been defined: {not_avail}")
        # TODO: add progress bar
        res = [self[c].load(src) for c in concepts]
        return res
        
    def merge(self, other: Type['ConceptDict'], overwrite: bool = False) -> Type['ConceptDict']:
        """Merge another concept dictionaries

        In the case of duplicate concept definitions and `overwrite==True`, the definitions in `other`
        will be retained

        Args:
            other: another concept dictionary.
            overwrite: should duplicate concepts be overwritten? Defaults to False.

        Raises:
            ValueError: if there are duplicate concept definitions and `overwrite==False`
        """
        overlap = list(set(self.concepts) & set(other.concepts))
        if not overwrite and len(overlap) > 0:
            raise ValueError(
                f"Duplicate concepts found when merging dictionaries: {print_list(overlap)}. "
                f"Use `overwrite==True` if duplicate concepts should be overwritten."
            )
        return ConceptDict(self.concepts.copy().update(other.concepts))

    def from_dict(x: Dict) -> Type['ConceptDict']:
        """Parse medical concepts from a dict, e.g., as read from JSON

        Args: 
            x: nested dictionary containing the concept definitions

        Returns:
            parsed medical concept dictionary
        """
        concepts = {k: parse_concept(k, v) for k, v in x.items()}
        return ConceptDict(concepts)
        
    def from_dirs(name: str = 'concept-dict', cfg_dirs: Path | List[Path] = None) -> Type['ConceptDict']:
        """Parse medical concepts from one or more JSON config files

        Returns:
            parsed medical concept dictionary
        """
        dictionary = read_dictionary(name, cfg_dirs)
        return ConceptDict.from_dict(dictionary)
        
    def __getitem__(self, concept_name: str) -> Concept:
        """Return a single medical concept

        Args:
            concept_name: name of the concept to return

        Returns:
            medical concept
        """
        return self.concepts[concept_name]
    