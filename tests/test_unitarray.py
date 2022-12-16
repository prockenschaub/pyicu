import operator

import numpy as np
from pandas import Series
import pytest

from pandas.tests.extension.base.casting import BaseCastingTests  # noqa
from pandas.tests.extension.base.constructors import BaseConstructorsTests  # noqa
from pandas.tests.extension.base.dtype import BaseDtypeTests  # noqa
from pandas.tests.extension.base.getitem import BaseGetitemTests  # noqa
from pandas.tests.extension.base.groupby import BaseGroupbyTests  # noqa
from pandas.tests.extension.base.interface import BaseInterfaceTests  # noqa
from pandas.tests.extension.base.io import BaseParsingTests  # noqa
from pandas.tests.extension.base.methods import BaseMethodsTests  # noqa
from pandas.tests.extension.base.missing import BaseMissingTests  # noqa
from pandas.tests.extension.base.ops import (  # noqa
    BaseArithmeticOpsTests,
    BaseComparisonOpsTests,
    BaseOpsUtil,
    BaseUnaryOpsTests,
)
from pandas.tests.extension.base.printing import BasePrintingTests  # noqa
from pandas.tests.extension.base.reduce import (  # noqa
    BaseBooleanReduceTests,
    BaseNoReduceTests,
    BaseNumericReduceTests,
)
from pandas.tests.extension.base.reshaping import BaseReshapingTests  # noqa
from pandas.tests.extension.base.setitem import BaseSetitemTests  # noqa

from pyicu.container.unit import UnitDtype, UnitArray


@pytest.fixture
def dtype():
    """
    A fixture providing the ExtensionDtype to validate.
    """
    return UnitDtype(unit='heisl')


@pytest.fixture
def data():
    """
    Length-100 array for this type.
    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """
    return UnitArray(np.arange(100), unit='heisl')


@pytest.fixture
def data_for_twos():
    """
    Length-100 array in which all the elements are two.
    """
    return UnitArray(np.array([2] * 100), unit='heisl')


@pytest.fixture
def data_missing():
    """
    Length-2 array with [NA, Valid].
    """
    return UnitArray(np.array([np.nan, 2]), unit='heisl')


@pytest.fixture(params=['data', 'data_missing'])
def all_data(request, data, data_missing):
    """
    Parameterized fixture giving 'data' and 'data_missing'.
    """
    if request.param == 'data':
        return data
    elif request.param == 'data_missing':
        return data_missing


@pytest.fixture
def data_repeated(data):
    """
    Generate many datasets.

    Parameters
    ----------
    data : fixture implementing `data`

    Returns
    -------
    Callable[[int], Generator]:
        A callable that takes a `count` argument and
        returns a generator yielding `count` datasets.
    """
    def gen(count):
        for _ in range(count):
            yield data

    return gen


@pytest.fixture
def data_for_sorting():
    """
    Length-3 array with a known sort order.
    This should be three items [B, C, A] with A < B < C.
    """
    return UnitArray(np.array([2, 3, 1]), unit='heisl')


@pytest.fixture
def data_missing_for_sorting():
    """
    Length-3 array with a known sort order.
    This should be three items [B, NA, A] with A < B and NA missing.
    """
    return UnitArray(np.array([2, np.nan, 1]), unit='heisl')


@pytest.fixture
def na_cmp():
    """
    Binary operator for comparing NA values.
    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.
    By default, uses ``operator.is_``.
    """
    return lambda a, b: np.array_equal(a, b, equal_nan=True)


@pytest.fixture
def na_value():
    """
    The scalar missing value for this type. Default 'None'.
    """
    return np.nan


@pytest.fixture
def data_for_grouping():
    """
    Data for factorization, grouping, and unique tests.
    Expected to be like [B, B, NA, NA, A, A, B, C] where A < B < C and NA is missing.
    """
    return UnitArray(np.array([2, 2, np.nan, np.nan, 1, 1, 2, 3]), unit='heisl')


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """
    Whether to box the data in a Series.
    """
    return request.param


@pytest.fixture(
    params=[
        lambda x: 1,
        lambda x: [1] * len(x),
        lambda x: Series([1] * len(x)),
    ],
    ids=['scalar', 'list', 'series'],
)
def groupby_apply_op(request):
    """
    Functions to test groupby.apply().
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_frame(request):
    """
    Boolean fixture to support Series and Series.to_frame() comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_series(request):
    """
    Boolean fixture to support arr and Series(arr) comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy(request):
    """
    Boolean fixture to support comparison testing of ExtensionDtype array    and numpy array.
    """
    return request.param


@pytest.fixture(params=['ffill', 'bfill'])
def fillna_method(request):
    """
    Parameterized fixture giving method parameters 'ffill' and 'bfill' for
    Series.fillna(method=<method>) testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_array(request):
    """
    Boolean fixture to support ExtensionDtype _from_sequence method testing.
    """
    return request.param


@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    """
    Simple fixture for testing keys in sorting methods.
    Tests None (no key) and the identity key.
    """
    return request.param


# TODO: Finish implementing all operators
_all_arithmetic_operators = [
    '__add__',
    #  '__radd__',
    '__sub__',
    #  '__rsub__',
    '__mul__',
    #  '__rmul__',
    #  '__floordiv__',
    #  '__rfloordiv__',
    '__truediv__',
    #  '__rtruediv__',
    #  '__pow__',
    #  '__rpow__',
    #  '__mod__',
    #  '__rmod__',
]
@pytest.fixture(params=_all_arithmetic_operators)
def all_arithmetic_operators(request):
    """
    Fixture for dunder names for common arithmetic operations.
    """
    return request.param


_all_numeric_reductions = [
    'sum',
    'max',
    'min',
    'mean',
    'prod',
    'std',
    'var',
    'median',
    'kurt',
    'skew',
]
@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request):
    """
    Fixture for numeric reduction names.
    """
    return request.param


_all_boolean_reductions = ['all', 'any']
@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request):
    """
    Fixture for boolean reduction names.
    """
    return request.param


_all_reductions = _all_numeric_reductions + _all_boolean_reductions
@pytest.fixture(params=_all_reductions)
def all_reductions(request):
    """
    Fixture for all (boolean + numeric) reduction names.
    """
    return request.param


_all_compare_operators = [
    '__eq__',
    '__ne__',
    '__le__',
    '__lt__',
    '__ge__',
    '__gt__',
]
@pytest.fixture(params=_all_compare_operators)
def comparison_op(request):
    """
    Fixture for dunder names for common compare operations:

    * >=
    * >
    * ==
    * !=
    * <
    * <=
    """
    return request.param


class TestCastingTests(BaseCastingTests):
    pass


class TestConstructorsTests(BaseConstructorsTests):
    pass


class TestDtypeTests(BaseDtypeTests):
    pass


class TestGetitemTests(BaseGetitemTests):
    pass


class TestGroupbyTests(BaseGroupbyTests):
    pass


class TestInterfaceTests(BaseInterfaceTests):
    pass


class TestParsingTests(BaseParsingTests):
    pass


class TestMethodsTests(BaseMethodsTests):
    def test_insert_invalid(self, data, invalid_scalar):
        # No invalid scalar for measurements
        pass


class TestMissingTests(BaseMissingTests):
    pass


class TestArithmeticOpsTests(BaseArithmeticOpsTests):
    series_scalar_exc = None
    frame_scalar_exc = None
    series_array_exc = None
    divmod_exc = TypeError  # TODO: Implement divmod


class TestComparisonOpsTests(BaseComparisonOpsTests):
    # See pint-pandas test suite
    def _compare_other(self, s, data, op_name, other):
        op = self.get_op_from_name(op_name)
        result = op(s, other)
        expected = op(s.to_numpy(), other)
        assert (result == expected).all()


class TestOpsUtil(BaseOpsUtil):
    pass


class TestUnaryOpsTests(BaseUnaryOpsTests):
    pass


class TestPrintingTests(BasePrintingTests):
    pass


class TestBooleanReduceTests(BaseBooleanReduceTests):
    pass


class TestNumericReduceTests(BaseNumericReduceTests):
    pass


# AFAICT NoReduce and Boolean+NumericReduce are mutually exclusive
# class TestNoReduceTests(BaseNoReduceTests):
    # pass


class TestReshapingTests(BaseReshapingTests):
    pass


class TestSetitemTests(BaseSetitemTests):
    pass