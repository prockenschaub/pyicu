from __future__ import annotations
from typing import Any, Sequence
import re
import operator
import numpy as np
import pandas as pd

from .array import BaseDtype, BaseArray

@pd.api.extensions.register_extension_dtype
class UnitDtype(BaseDtype):
    """
    An ExtensionDtype for unit-aware measurement data.
    """
    # Required for all parameterized dtypes
    _metadata = ('unit',)
    _match = re.compile(r'(U|u)nit\[(?P<unit>.+)\]')

    def __init__(self, unit=None):
        self._unit = unit

    def __str__(self) -> str:
        return f'unit[{self.unit}]'

    # TestDtypeTests
    def __hash__(self) -> int:
        return hash(str(self))

    # TestDtypeTests
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.name == other
        else:
            return isinstance(other, type(self)) and self.unit == other.unit

    # Required for pickle compat (see GH26067)
    def __setstate__(self, state) -> None:
        self._unit = state['unit']

    # Required for all ExtensionDtype subclasses
    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this dtype.
        """
        return UnitArray

    # Recommended for parameterized dtypes
    @classmethod
    def construct_from_string(cls, string: str) -> UnitDtype:
        """
        Construct an UnitDtype from a string.

        Example
        -------
        >>> UnitDtype.construct_from_string('unit[mg]')
        unit['mg']
        """
        if not isinstance(string, str):
            msg = f"'construct_from_string' expects a string, got {type(string)}"
            raise TypeError(msg)

        msg = f"Cannot construct a '{cls.__name__}' from '{string}'"
        match = cls._match.match(string)

        if match:
            d = match.groupdict()
            try:
                return cls(unit=d['unit'])
            except (KeyError, TypeError, ValueError) as err:
                raise TypeError(msg) from err
        else:
            raise TypeError(msg)

    @property
    def unit(self) -> str:
        """
        The measurement unit.
        """
        return self._unit


class UnitArray(BaseArray):
    """
    An ExtensionArray for unit-aware measurement data.
    """
    # TODO: add pyarrow support https://pandas.pydata.org/docs/development/extending.html#compatibility-with-apache-arrow
    _dtype = UnitDtype()

    # Include `copy` param for TestInterfaceTests
    def __init__(self, data, unit: str = None, copy: bool=False):
        if isinstance(data, np.ndarray) and data.dtype == 'bool':
            return data
        # TODO: check that data is numeric
        self._data = np.array(data, copy=copy)
        if unit is not None: 
            self._dtype._unit = unit

    # TestUnaryOpsTests
    def __invert__(self) -> UnitArray:
        """
        Element-wise inverse of this array.
        """
        return type(self)(super().__invert__(), unit=self.dtype.unit)

    def _ensure_same_units(self, other) -> UnitArray:
        """
        Helper method to ensure `self` and `other` have the same units.
        """
        if isinstance(other, type(self)) and self.dtype.unit != other.dtype.unit:
            return other.asunit(self.dtype.unit)
        else:
            return other

    def _apply_operator(self, op, other, recast=False) -> np.ndarray | UnitArray:
        """
        Helper method to apply an operator `op` between `self` and `other`.

        Some ops require the result to be recast into UnitArray:
        * Comparison ops: recast=False
        * Arithmetic ops: recast=True
        """
        f = operator.attrgetter(op)
        data, other = np.array(self), np.array(self._ensure_same_units(other))
        result = f(data)(other)
        return result if not recast else type(self)(result, unit=self.dtype.unit)

    # Required for all ExtensionArray subclasses
    @classmethod
    def _from_sequence(cls, data, dtype=None, copy: bool=False):
        """
        Construct a new UnitArray from a sequence of scalars.
        """
        if dtype is None:
            dtype = UnitDtype()

        if not isinstance(dtype, UnitDtype):
            msg = f"'{cls.__name__}' only supports 'UnitDtype' dtype"
            raise ValueError(msg)
        else:
            return cls(data, unit=dtype.unit, copy=copy)

    # Required for all ExtensionArray subclasses
    @classmethod
    def _from_factorized(cls, uniques: np.ndarray, original: UnitArray):
        """
        Reconstruct an UnitArray after factorization.
        """
        return cls(uniques, unit=original.dtype.unit)

    # Required for all ExtensionArray subclasses
    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[UnitArray]) -> UnitArray:
        """
        Concatenate multiple UnitArrays.
        """
        # ensure same units
        counts = pd.value_counts([array.dtype.unit for array in to_concat])
        unit = counts.index[0]

        if counts.size > 1:
            to_concat = [a.asunit(unit) for a in to_concat]

        return cls(np.concatenate(to_concat), unit=unit)

    @property
    def unit(self):
        return self.dtype.unit

    # Required for all ExtensionArray subclasses
    def copy(self):
        """
        Return a copy of the array.
        """
        return type(self)(super().copy(), unit=self.unit)

    def asunit(self, unit: str) -> UnitArray:
        """
        Cast to another unit.
        """
        # TODO: implement for UntConcept
        raise NotImplementedError()