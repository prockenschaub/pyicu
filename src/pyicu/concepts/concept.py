from typing import List
import pandas as pd

from .item import Item
from ..sources import Src
from ..utils import concat_tbls


class Concept:
    """Base class for a clinical concept

    Concept objects are used in pyicu as a way to specify how a clinical concept, such as heart rate can be
    loaded from a data source.

    Sub-classes have been defined, each representing a different data-scenario and holding
    further class-specific information. The following sub-classes to `Concept` are available:

        `NumConcept`: The most widely used concept type is indented for concepts representing numerical
            measurements. Additional information that can be specified includes a string-valued unit
            specification, alongside a plausible range which can be used during data loading.

        `UntConcept`: A recent (experimental) addition which inherits from `NumConcept` but instead of
            manual unit conversion, leverages the udunits library.

        `FctConcept`: In case of categorical concepts, such as sex, a set of factor levels can be specified,
            against which the loaded data is checked.

        `LglConcept`: A special case of `FctConcept`, this allows only for logical values (True and False).

        `RecConcept`: More involved concepts, such as a SOFA score can pull in other concepts. Recursive
            concepts can build on other recursive concepts up to arbitrary recursion depth. Owing to the
            more complicated nature of such concepts, a callback function can be specified which is used in
            data loading for concept-specific post-processing steps.


    Args:
        name: short concept name
        items: list of `Item` objects that define how data is loaded from a specific data source
        description: more elaborate description of the clinical concept. Defaults to None.
        category: a category designation, e.g., "vitals". Defaults to None.
        aggregate: a string-valued specification for an aggregation function. Defaults to None.
        interval: the time step size obeyed by the concept, e.g., 1 hour. Defaults to None, which requires
            a specification at loading.
        target: a target class specification. Defaults to "ts_tbl".
    """

    def __init__(
        self,
        name: str,
        items: List[Item],
        description: str = None,
        category: str = None,
        aggregate: str = None,
        interval: pd.Timestamp = None,
        target: str = "ts_tbl",
    ) -> None:
        self.name = name
        self.items = items
        self.description = description
        self.category = category
        self.aggregate = aggregate

        if target is None:
            target = ""
        self.target = target

    def src_items(self, src: Src) -> List[Item]:
        """Return all defined `Item`s for a given data source.

        Args:
            src: data source, e.g., MIMIC IV.

        Returns:
            list of `Item`s
        """
        return [i for i in self.items if i.src == src.name]

    def is_defined(self, src: Src) -> bool:
        """Are there any `Item`s defined for a given data source?

        Args:
            src: data source, e.g., MIMIC IV.

        Returns:
            True or False
        """
        return len(self.src_items(src)) > 0

    def load(self, src: Src, **kwargs):
        """Load concept data from a given data source

        Args:
            src: data source, e.g., MIMIC IV.

        Returns:
            table of the class `self.target`
        """
        if not self.is_defined(src):
            return None

        items = self.src_items(src)
        res = [i.load(src, self.target, None) for i in items]

        # TODO: check that the return has the same column names etc.
        return concat_tbls(res, axis=0)


class NumConcept(Concept):
    """A numerical clinical concept like heart rate

    See also: `Concept`

    Args:
        name: short concept name
        items: list of `Item` objects that define how data is loaded from a specific data source
        unit: string-valued definition of the measurement unit
        min: lowest plausible value
        max: highest plausible value
    """

    def __init__(
        self, name: str, items: List[Item], unit: str | None = None, min: float | None = None, max=None, **kwargs
    ) -> None:
        super().__init__(name, items, **kwargs)
        self.unit = unit
        self.min = min
        self.max = max

    def load(self, src: Src, **kwargs):
        """Load numeric concept data from a given data source

        Args:
            src: data source, e.g., MIMIC IV.

        Returns:
            table of the class `self.target`
        """
        res = super().load(src, **kwargs)

        # # Remove missing values
        # n_total <- nrow(x)
        # x <- rm_na_val_var(x, col)

        # # Remove out of range 
        # n_nonmis <- nrow(x)
        # keep  <- check_bound(x[[col]], min, `>=`) & check_bound(x[[col]], max, `<=`)
        # x <- x[keep, ]

        # n_rm <- n_nonmis - nrow(x)

        # if (n_rm > 0L) {
        #     msg_progress("removed {n_rm} ({prcnt(n_rm, n_total)}) of rows due to out
        #                 of range entries")
        # }

        return res


class UntConcept(NumConcept):
    """A numerical clinical concept with automatic unit conversion via the udunits library

    See also: `NumConcept`

    Args:
        name: short concept name
        items: list of `Item` objects that define how data is loaded from a specific data source
        min: lowest plausible value
        max: highest plausible value
    """

    def __init__(self, name: str, items: List[Item], min: float | None = None, max=None, **kwargs) -> None:
        super().__init__(name, items, min=min, max=max, **kwargs)


class FctConcept(Concept):
    """A categorical clinical concept like biological sex

     See also: `Concept`

    Args:
        name: short concept name
        items: list of `Item` objects that define how data is loaded from a specific data source
        levels: list of possible values
    """

    def __init__(self, name: str, items: List[Item], levels: List[str] = None, **kwargs) -> None:
        super().__init__(name, items, **kwargs)
        self.levels = levels

    def load(self, src: Src, **kwargs):
        """Load numeric concept data from a given data source

        Args:
            src: data source, e.g., MIMIC IV.

        Returns:
            table of the class `self.target`
        """
        res = super().load(src, **kwargs)
        res['val_var'] = pd.Categorical(res['val_var'], categories=self.levels)
        return res

class LglConcept(FctConcept):
    """A binary clinical concept with True or False

    See also: `FctConcept`

    Args:
        name: short concept name
        items: list of `Item` objects that define how data is loaded from a specific data source
    """

    def __init__(self, name: str, items: List[Item], **kwargs) -> None:
        super().__init__(name, items, levels=[True, False], **kwargs)


class RecConcept(Concept):
    """A clinical concept derived from one or more other concepts, such as SOFA score.

    See also: `Concept`

    Args:
        name: short concept name
        items: list of `Item` objects that define how data is loaded from a specific data source
        callback: name of a function to be called on the returned data used for data cleanup operations
    """

    def __init__(self, name, items, callback: str = None, **kwargs) -> None:
        super().__init__(name, items, **kwargs)
        self.callback = callback


def concept_class(x: str) -> Concept:
    """Map string to concept class

    Args:
        x: string specification of the concept as defined by ricu

    Returns:
        concept class
    """
    match x:
        case "num_cncpt":
            return NumConcept
        case "unt_cncpt":
            return UntConcept
        case "fct_cncpt":
            return FctConcept
        case "lgl_cncpt":
            return LglConcept
        case "rec_cncpt":
            return RecConcept
        case _:
            return NumConcept
