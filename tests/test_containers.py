import pytest
import numpy as np
from pyicu.container import IdTbl, TsTbl
from pyicu.container.table import parse_columns

# MeasuredSeries ---------------------------------------------


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
