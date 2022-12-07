# The code for the extension dtype is heavily based on 
# https://stackoverflow.com/questions/68893521/simple-example-of-pandas-extensionarray

from __future__ import annotations
from typing import Any, Sequence
import re
import operator
import numpy as np
import pandas as pd


@pd.api.extensions.register_extension_dtype
class UnitDtype(pd.core.dtypes.dtypes.PandasExtensionDtype):
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

    # Required for all ExtensionDtype subclasses
    @property
    def type(self):
        """
        The scalar type for the array (e.g., int).
        """
        return np.generic

    # Required for all ExtensionDtype subclasses
    @property
    def name(self) -> str:
        """
        A string representation of the dtype.
        """
        return str(self)

    @property
    def unit(self) -> str:
        """
        The measurement unit.
        """
        return self._unit


class UnitArray(pd.api.extensions.ExtensionArray):
    """
    An ExtensionArray for unit-aware measurement data.
    """

    _dtype = UnitDtype()

    # Include `copy` param for TestInterfaceTests
    def __init__(self, data, unit: str = None, copy: bool=False):
        if isinstance(data, np.ndarray) and data.dtype == 'bool':
            return data
        self._data = np.array(data, copy=copy)
        if unit is not None: 
            self._dtype._unit = unit

    # Required for all ExtensionArray subclasses
    def __getitem__(self, index: int) -> UnitArray | Any:
        """
        Select a subset of self.
        """
        if isinstance(index, int):
            return self._data[index]
        else:
            # Check index for TestGetitemTests
            index = pd.core.indexers.check_array_indexer(self, index)
            return type(self)(self._data[index])

    # TestSetitemTests
    def __setitem__(self, index: int, value: np.generic) -> None:
        """
        Set one or more values in-place.
        """
        # Check index for TestSetitemTests
        index = pd.core.indexers.check_array_indexer(self, index)

        # Upcast to value's type (if needed) for TestMethodsTests
        if self._data.dtype < type(value):
            self._data = self._data.astype(type(value))

        # TODO: Validate value for TestSetitemTests
        # value = self._validate_setitem_value(value)

        self._data[index] = value

    # Required for all ExtensionArray subclasses
    def __len__(self) -> int:
        """
        Length of this array.
        """
        return len(self._data)

    # TestUnaryOpsTests
    def __invert__(self) -> UnitArray:
        """
        Element-wise inverse of this array.
        """
        data = ~self._data
        return type(self)(data, unit=self.dtype.unit)

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

    def _apply_operator_if_not_series(self, op, other, recast=False) -> np.ndarray | UnitArray:
        """
        Wraps _apply_operator only if `other` is not Series/DataFrame.
        
        Some ops should return NotImplemented if `other` is a Series/DataFrame:
        https://github.com/pandas-dev/pandas/blob/e7e7b40722e421ef7e519c645d851452c70a7b7c/pandas/tests/extension/base/ops.py#L115
        """
        if isinstance(other, (pd.Series, pd.DataFrame)):
            return NotImplemented
        else:
            return self._apply_operator(op, other, recast=recast)

    # Required for all ExtensionArray subclasses
    @pd.core.ops.unpack_zerodim_and_defer('__eq__')
    def __eq__(self, other):
        return self._apply_operator('__eq__', other, recast=False)

    # TestComparisonOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__ne__')
    def __ne__(self, other):
        return self._apply_operator('__ne__', other, recast=False)

    # TestComparisonOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__lt__')
    def __lt__(self, other):
        return self._apply_operator('__lt__', other, recast=False)

    # TestComparisonOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__gt__')
    def __gt__(self, other):
        return self._apply_operator('__gt__', other, recast=False)

    # TestComparisonOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__le__')
    def __le__(self, other):
        return self._apply_operator('__le__', other, recast=False)

    # TestComparisonOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__ge__')
    def __ge__(self, other):
        return self._apply_operator('__ge__', other, recast=False)
    
    # TestArithmeticOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__add__')
    def __add__(self, other) -> UnitArray:
        return self._apply_operator_if_not_series('__add__', other, recast=True)

    # TestArithmeticOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__sub__')
    def __sub__(self, other) -> UnitArray:
        return self._apply_operator_if_not_series('__sub__', other, recast=True)

    # TestArithmeticOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__mul__')
    def __mul__(self, other) -> UnitArray:
        return self._apply_operator_if_not_series('__mul__', other, recast=True)

    # TestArithmeticOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__truediv__')
    def __truediv__(self, other) -> UnitArray:
        return self._apply_operator_if_not_series('__truediv__', other, recast=True)

    # TestUnaryOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__pos__')
    def __pos__(self, other) -> UnitArray:
        return self._apply_operator_if_not_series('__pos__', other, recast=True)

    # TestUnaryOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__neg__')
    def __neg__(self, other) -> UnitArray:
        return self._apply_operator_if_not_series('__neg__', other, recast=True)

    # TestUnaryOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__abs__')
    def __abs__(self, other) -> UnitArray:
        return self._apply_operator_if_not_series('__abs__', other, recast=True)

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

    # TestParsingTests
    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype=None, copy: bool=False) -> UnitArray:
        """
        Construct a new UnitArray from a sequence of strings.
        """
        scalars = pd.to_numeric(strings, errors='raise')
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)

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

    # Required for all ExtensionArray subclasses
    @property
    def dtype(self):
        """
        An instance of UnitDtype.
        """
        return self._dtype

    # Required for all ExtensionArray subclasses
    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.
        """
        return self._data.nbytes

    @property
    def unit(self):
        return self.dtype.unit

    # Test*ReduceTests
    def all(self) -> bool:
        return all(self)

    def any(self) -> bool:  # Test*ReduceTests
        return any(self)

    def sum(self) -> np.generic:  # Test*ReduceTests
        return self._data.sum()

    def mean(self) -> np.generic:  # Test*ReduceTests
        return self._data.mean()

    def max(self) -> np.generic:  # Test*ReduceTests
        return self._data.max()

    def min(self) -> np.generic:  # Test*ReduceTests
        return self._data.min()

    def prod(self) -> np.generic:  # Test*ReduceTests
        return self._data.prod()

    def std(self) -> np.generic:  # Test*ReduceTests
        return pd.Series(self._data).std()

    def var(self) -> np.generic:  # Test*ReduceTests
        return pd.Series(self._data).var()

    def median(self) -> np.generic:  # Test*ReduceTests
        return np.median(self._data)

    def skew(self) -> np.generic:  # Test*ReduceTests
        return pd.Series(self._data).skew()

    def kurt(self) -> np.generic:  # Test*ReduceTests
        return pd.Series(self._data).kurt()

    # Test*ReduceTests
    def _reduce(self, name: str, *, skipna: bool=True, **kwargs):
        """
        Return a scalar result of performing the reduction operation.
        """
        f = operator.attrgetter(name)
        return f(self)()

    # Required for all ExtensionArray subclasses
    def isna(self):
        """
        A 1-D array indicating if each value is missing.
        """
        return pd.isnull(self._data)

    # Required for all ExtensionArray subclasses
    def copy(self):
        """
        Return a copy of the array.
        """
        copied = self._data.copy()
        return type(self)(copied, unit=self.unit)

    # Required for all ExtensionArray subclasses
    def take(self, indices, allow_fill=False, fill_value=None):
        """
        Take elements from an array.
        """
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = pd.core.algorithms.take(self._data, indices, allow_fill=allow_fill,
                                         fill_value=fill_value)
        return self._from_sequence(result, dtype=self.dtype)

    # TestMethodsTests
    def value_counts(self, dropna: bool=True):
        """
        Return a Series containing descending counts of unique values (excludes NA values by default).
        """
        return pd.core.algorithms.value_counts(self._data, dropna=dropna)

    def asunit(self, unit: str) -> UnitArray:
        """
        Cast to another unit.
        """
        # TODO: implement for UntConcept
        raise NotImplementedError()