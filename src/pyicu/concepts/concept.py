

class Concept():
   def __init__(self, name, items, description=None, category=None, aggregate=None, interval=None, target=None) -> None:
      self.name = name
      self.items = items
      self.description = description
      self.category = category
      self.aggregate = aggregate
      self.target = target


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


