from typing import List, Dict, Type
from pathlib import Path

from ..data.source import Src
from ..tables import pyICUTbl
from .concept import Concept
from .utils import read_dictionary, parse_concept

class ConceptDict():
    def __init__(self, concepts: List[Concept] = None) -> None:
        self.concepts = concepts

    def load_concepts(self, concepts: str | List[str], src: Src) -> Dict[str, pyICUTbl]:
        # TODO: check for existence with an intelligible error when a concept is missing
        # TODO: add progress bar
        res = [self[c].load(src) for c in concepts]
        return res
        
    def from_dict(x: Dict) -> Type['ConceptDict']:
        concepts = {k: parse_concept(k, v) for k, v in x.items()}
        return ConceptDict(concepts)
        
    def from_dirs(name: str = 'concept-dict', cfg_dirs: Path | List[Path] = None) -> Type['ConceptDict']:
        dictionary = read_dictionary(name, cfg_dirs)
        return ConceptDict.from_dict(dictionary)
        
    def __getitem__(self, idx: str) -> Concept:
        return self.concepts[idx]