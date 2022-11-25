import pytest
import numpy as np
from pyicu.container import MeasuredSeries, IdTbl, TsTbl, parse_columns

# MeasuredSeries ---------------------------------------------
@pytest.fixture
def example_series():
    return MeasuredSeries([1.0, 2.0, 3.0])


def test_series_unit(example_series):
    x = MeasuredSeries(example_series, unit="mg/dL")
    assert x.unit == "mg/dL"


def test_series_construct(example_series):
    x = example_series[:1]
    assert x.values == np.array([1.0])


# IdTbl -------------------------------------------------------


# Utils -------------------------------------------------------
def test_parse_columns():
    assert parse_columns("col1", ["col1", "col2"]) == "col1"
    assert parse_columns(0, ["col1", "col2"]) == "col1"
    assert parse_columns(["col1"], ["col1", "col2"]) == ["col1"]


def test_parse_columns_errors():
    with pytest.raises(ValueError) as e_info:
        parse_columns("col1", ["col2", "col3"])
    assert e_info.match("could not find column")
    with pytest.raises(TypeError) as e_info:
        parse_columns(("col1",), ["col2", "col3"])
    assert e_info.match("expected str, int, or list")
