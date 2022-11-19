import pytest

from pyicu.interval import *


def test_change_interval():
    assert change_interval(hours(6), hours(1)) == hours(6)
    assert change_interval(hours(6), hours(5)) == hours(5)
    assert change_interval(hours(4), hours(5)) == hours(0)
    assert change_interval(hours(6), days(1)) == days(0)
    assert change_interval(hours(36), days(1)) == days(1)
    assert change_interval(hours(6), mins(1)) == mins(360)
