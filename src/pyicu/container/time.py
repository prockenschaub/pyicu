from __future__ import annotations
import re
import operator
import numpy as np
import pandas as pd
from typing import Any, Sequence

from pandas.core.base import PandasObject, NoNewAttributesMixin
from pandas.core.accessor import PandasDelegate, delegate_names
from pandas.api.types import is_timedelta64_dtype


def days(x):
    return TimeDtype(x, 'day')

def hours(x):
    return TimeDtype(x, 'hour')

def minutes(x):
    return TimeDtype(x, 'minute')

def seconds(x):
    return TimeDtype(x, 'second')

def milliseconds(x):
    return TimeDtype(x, 'millisecond')


@pd.api.extensions.register_extension_dtype
class TimeDtype(pd.core.dtypes.dtypes.PandasExtensionDtype):
    """
    An ExtensionDtype for unit-aware measurement data.
    """
    # Required for all parameterized dtypes
    _metadata = ('freq', 'unit',)
    _match = re.compile(r"(T|t)ime\[(?P<freq>\d+) (?P<unit>[a-z/]*)\]")

    def __init__(self, freq=1, unit='hour'):
        if unit not in ['day', 'hour', 'minute', 'second', 'millisecond']:
            msg = f"'{type(self).__name__}' only supports 'day', 'hour', 'minute', 'second', and 'millisecond'"
            raise ValueError(msg)
        if not isinstance(freq, (int, float)):
            msg = f"time frequency must be int or float, got {freq.__class__}"
            raise TypeError(msg)
        self._freq = freq
        self._unit = unit

    def __str__(self) -> str:
        return f'time[{self.freq} {self.unit}]'

    # TestDtypeTests
    def __hash__(self) -> int:
        return hash(str(self))

    # TestDtypeTests
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.name == other
        else:
            return isinstance(other, type(self)) and self.freq == other.freq and self.unit == other.unit

    # Required for pickle compat (see GH26067)
    def __setstate__(self, state) -> None:
        self._freq = state['freq']
        self._unit = state['unit']

    # Required for all ExtensionDtype subclasses
    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this dtype.
        """
        return TimeArray

    # Recommended for parameterized dtypes
    @classmethod
    def construct_from_string(cls, string: str) -> TimeDtype:
        """
        Construct an TimeDtype from a string.
        Example
        -------
        >>> TimeDtype.construct_from_string('time[1 hour]')
        time[1 hour]
        """
        if not isinstance(string, str):
            msg = f"'construct_from_string' expects a string, got {type(string)}"
            raise TypeError(msg)

        msg = f"Cannot construct a '{cls.__name__}' from '{string}'"
        match = cls._match.match(string)

        if match:
            d = match.groupdict()
            try:
                return cls(freq=d['freq'], unit=d['unit'])
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
    def freq(self) -> int:
        """
        The measurement unit.
        """
        return self._freq

    @property
    def unit(self) -> str:
        """
        The measurement unit.
        """
        return self._unit


class TimeArray(pd.api.extensions.ExtensionArray):
    """
    An ExtensionArray for unit-aware measurement data.
    """

    _dtype = TimeDtype()

    # Include `copy` param for TestInterfaceTests
    def __init__(self, data, interval: TimeDtype = milliseconds(1), copy: bool=False):
        if isinstance(data, pd.Series):
            if is_timedelta64_dtype(data):
                data = TimeArray(data.astype(np.int64) // 10**6, freq=1, unit='millisec')
                
            else:
                data = data.values
        if isinstance(data, np.ndarray):
            if data.dtype == 'bool':
                return data
            elif np.issubdtype(data.dtype, np.timedelta64): 
                data = TimeArray(data.astype(np.int64) // 10**6, freq=1, unit='millisec')
            elif not np.issubdtype(data.dtype, (int, float)):
                raise TypeError(f'expected int, float, or timedelta, got {data.dtype}')
        if isinstance(data, TimeArray):
            data = data.change_interval(interval)
        self._data = np.array(data, copy=copy)
        self._dtype = interval

    # Required for all ExtensionArray subclasses
    def __getitem__(self, index: int) -> TimeArray | Any:
        """
        Select a subset of self.
        """
        if isinstance(index, int):
            return self._data[index]
        else:
            # Check index for TestGetitemTests
            index = pd.core.indexers.check_array_indexer(self, index)
            return type(self)(self._data[index], self.dtype)

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
    def __invert__(self) -> TimeArray:
        """
        Element-wise inverse of this array.
        """
        data = ~self._data
        return type(self)(data, freq=self.dtype.freq, unit=self.dtype.unit)

    def _apply_operator(self, op, other, recast=False) -> np.ndarray | TimeArray:
        """
        Helper method to apply an operator `op` between `self` and `other`.
        Some ops require the result to be recast into TimeArray:
        * Comparison ops: recast=False
        * Arithmetic ops: recast=True
        """
        f = operator.attrgetter(op)
        data, other = np.array(self), np.array(other)
        result = f(data)(other)
        return result if not recast else type(self)(result, freq=self.dtype.freq, unit=self.dtype.unit)

    def _apply_operator_if_not_series(self, op, other, recast=False) -> np.ndarray | TimeArray:
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
    def __add__(self, other) -> TimeArray:
        return self._apply_operator_if_not_series('__add__', other, recast=True)

    # TestArithmeticOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__sub__')
    def __sub__(self, other) -> TimeArray:
        return self._apply_operator_if_not_series('__sub__', other, recast=True)

    # TestArithmeticOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__mul__')
    def __mul__(self, other) -> TimeArray:
        return self._apply_operator_if_not_series('__mul__', other, recast=True)

    # TestArithmeticOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__truediv__')
    def __truediv__(self, other) -> TimeArray:
        return self._apply_operator_if_not_series('__truediv__', other, recast=True)

    # TestUnaryOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__pos__')
    def __pos__(self, other) -> TimeArray:
        return self._apply_operator_if_not_series('__pos__', other, recast=True)

    # TestUnaryOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__neg__')
    def __neg__(self, other) -> TimeArray:
        return self._apply_operator_if_not_series('__neg__', other, recast=True)

    # TestUnaryOpsTests
    @pd.core.ops.unpack_zerodim_and_defer('__abs__')
    def __abs__(self, other) -> TimeArray:
        return self._apply_operator_if_not_series('__abs__', other, recast=True)

    # Required for all ExtensionArray subclasses
    @classmethod
    def _from_sequence(cls, data, dtype=None, copy: bool=False):
        """
        Construct a new TimeArray from a sequence of scalars.
        """
        if dtype is None:
            dtype = TimeDtype()

        if not isinstance(dtype, TimeDtype):
            msg = f"'{cls.__name__}' only supports 'TimeDtype' dtype"
            raise ValueError(msg)
        else:
            return cls(data, dtype, copy=copy)

    # TestParsingTests
    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype=None, copy: bool=False) -> TimeArray:
        """
        Construct a new TimeArray from a sequence of strings.
        """
        scalars = pd.to_numeric(strings, errors='raise')
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)

    # Required for all ExtensionArray subclasses
    @classmethod
    def _from_factorized(cls, uniques: np.ndarray, original: TimeArray):
        """
        Reconstruct an TimeArray after factorization.
        """
        return cls(uniques, freq=original.dtype.freq, unit=original.dtype.unit)

    # Required for all ExtensionArray subclasses
    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[TimeArray]) -> TimeArray:
        """
        Concatenate multiple TimeArrays.
        """
        # TODO: check for same time interval
        # ensure same freqs
        counts = pd.value_counts([array.dtype.freq for array in to_concat])
        freq = counts.index[0]

        # ensure same units
        counts = pd.value_counts([array.dtype.unit for array in to_concat])
        unit = counts.index[0]

        return cls(np.concatenate(to_concat), TimeDtype(int(freq), unit))

    # Required for all ExtensionArray subclasses
    @property
    def dtype(self):
        """
        An instance of TimeDtype.
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

    @property
    def freq(self):
        return self.dtype.freq

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
        return type(self)(copied, self.dtype)

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

    def asunit(self, unit: str) -> TimeArray:
        """
        Cast to another unit.
        """
        # TODO: implement for UntConcept
        raise NotImplementedError()

    def change_interval(self, x: TimeDtype):
        td = pd.to_timedelta(self._data, unit=self.unit)
        new_data = (td // pd.Timedelta(x.freq, x.unit)).astype(int)
        return TimeArray(new_data, x)

    def obeys_interval(self) -> bool:
        return np.all(np.mod(self._data, self.dtype.freq) == 0)



@delegate_names(
    delegate=TimeArray, accessors=["freq", "unit"], typ="property"
)
@delegate_names(
    delegate=TimeArray,
    accessors=[
        "change_interval",
        "obeys_interval",
    ],
    typ="method",
)
@pd.api.extensions.register_series_accessor("tm")
class TimeAccessor(PandasDelegate, PandasObject, NoNewAttributesMixin):
    def __init__(self, data):
        self._validate(data)
        self._parent = data.values
        self._index = data.index
        self._name = data.name
        self._freeze()

    @staticmethod
    def _validate(data):
        if not isinstance(data.dtype, TimeDtype):
            raise AttributeError("can only use .time accessor with 'time' dtype")

    def _delegate_property_get(self, name):
        return getattr(self._parent, name)

    def _delegate_property_set(self, name, new_values):
        return setattr(self._parent, name, new_values)

    def obeys_interval(self) -> int:
        return self._parent.obeys_interval()

    def _delegate_method(self, name, *args, **kwargs):
        from pandas import Series

        method = getattr(self._parent, name)
        res = method(*args, **kwargs)
        if res is not None:
            return Series(res, index=self._index, name=self._name)
