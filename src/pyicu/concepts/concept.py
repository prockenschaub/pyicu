from typing import List

import pandas as pd

from ..utils import concat_tbls
from .item import Item
from ..data.source import Src

class Concept():
    def __init__(self, name, items, description=None, category=None, aggregate=None, interval=None, target="ts_tbl") -> None:
        self.name = name
        self.items = items
        self.description = description
        self.category = category
        self.aggregate = aggregate
        
        if target is None:
            target = ""
        self.target = target
    
    def src_items(self, src: Src) -> List[Item]:
        return [i for i in self.items if i.src == src.name]

    def is_defined(self, src: Src):
        return len(self.src_items(src)) > 0

    def load(self, src: Src, **kwargs):
        if not self.is_defined(src):
            return None

        items = self.src_items(src)
        res = [i.load(src, self.target, None) for i in items]

        # TODO: check that the return has the same column names etc.
        return concat_tbls(res, axis=0)


class NumConcept(Concept):
    def __init__(self, name, items, unit=None, min=None, max=None, **kwargs) -> None:
        super().__init__(name, items, **kwargs)
        self.unit = unit
        self.min = min
        self.max = max

class UntConcept(Concept):
    def __init__(self, name, items, unit=None, min=None, max=None, **kwargs) -> None:
        super().__init__(name, items, unit, min, max, **kwargs)

class FctConcept(Concept):
    def __init__(self, name, items, levels=None, **kwargs) -> None:
        super().__init__(name, items, **kwargs)
        self.levels = levels

class LglConcept(Concept):
    def __init__(self, name, items, **kwargs) -> None:
        super().__init__(name, items, **kwargs)

class RecConcept(Concept):
    def __init__(self, name, items, callback=None, **kwargs) -> None:
        super().__init__(name, items, **kwargs)
        self.callback = callback


