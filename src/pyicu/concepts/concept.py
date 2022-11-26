from typing import List, Callable
from operator import le, ge
import pandas as pd
import numpy as np

from .item import Item
from ..sources import Src
from ..utils import concat_tbls, enlist, diff
from ..interval import hours
from ..container import IdTbl, TsTbl, MeasuredSeries


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
        interval: pd.Timestamp = hours(1),
        target: str = "ts_tbl",
    ) -> None:
        self.name = name
        self.items = items
        self.description = description
        self.category = category
        self.aggregate = aggregate
        self.interval = interval

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

    def load(self, src: Src, interval: pd.Timedelta = hours(1), **kwargs):
        """Load concept data from a given data source

        Args:
            src: data source, e.g., MIMIC IV.
            interval: time resolution to which `time_vars` are rounded to. Defaults to hours(1).

        Returns:
            table of the class `self.target`
        """
        if not self.is_defined(src):
            return None

        items = self.src_items(src)
        res = [i.load(src, self.target, interval, **kwargs) for i in items]

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
        # TODO: force val_var to be numeric

        res = filter_bounds(res, "val_var", self.min, self.max)
        res = report_set_unit(res, "unit_var", "val_var", self.unit)

        res.drop(columns=diff(res.data_vars, ["val_var"]), errors="ignore", inplace=True)
        res.rename(columns={"val_var": self.name}, inplace=True)

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
        res["val_var"] = pd.Categorical(res["val_var"], categories=self.levels)
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


def prcnt(x: int | float, tot: int | float) -> str:
    return f"{np.round(x / tot * 100, decimals=2)}%"

def nrow(x: pd.DataFrame) -> int:
    return x.shape[0]

def ncol(x: pd.DataFrame) -> int:
    return x.shape[1]

def rm_na(x, cols : str | List[str] | None = None, mode: str = "all"):
    if cols is None:
        cols = enlist(x.data_vars())
    return x.dropna(how=mode, subset=cols, axis=0)

def rm_na_val_var(x: IdTbl | TsTbl, col: str = "val_var") -> IdTbl | TsTbl:
    n_row = nrow(x)
    x = rm_na(x, col)
    n_rm = n_row - nrow(x)

    if n_rm > 0:
        print(f"removed {n_rm} ({prcnt(n_rm, n_row)}) of rows due to missing values")
    return x

def filter_bounds(x: IdTbl | TsTbl, col: str, min: float, max:float):
    def check_bound(vc: MeasuredSeries, val: int | None, op: Callable):
        nna = ~vc.isna()
        if val is None:
            return nna
        return nna & op(vc, val)
    
    n_total = nrow(x)
    x = rm_na_val_var(x, col)

    n_nonmis = nrow(x)
    keep = check_bound(x[col], min, ge) & check_bound(x[col], max, le)
    x = x[keep]

    n_rm = n_nonmis - nrow(x)
    if n_rm > 0:
        print(f"removed {n_rm} ({prcnt(n_rm, n_total)}) of rows due to out of range values")

    return x

def report_set_unit(x: IdTbl | TsTbl, unit_var: str, val_var: str, unit: str | List[str]):
    # TODO: should unit be allowed to be None?
    unit = enlist(unit)
    
    if unit_var in x.columns:
        nm, ct = np.unique(x[unit_var], return_counts=True)
        pct = [prcnt(i, ct.sum()) for i in ct]

        if len(unit) > 1:
            ok = [i.lower() in [u.lower() for u in unit] for i in nm]
            if not all(ok):
                print(f"not all units are in [{','.join(unit)}]: ") # TODO: add counts and prcnt 
        elif len(nm) > 1:
            print("multiple units detected: ") # TODO: add counts and prcnt 
    
    x[val_var].unit = unit[0]
    return x


