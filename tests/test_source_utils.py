import pytest

from pyicu.sources.utils import *

@pytest.fixture
def chartevents(mimic_demo_cfg):
    return [t for t in mimic_demo_cfg.tbls if t.name == "chartevents"][0]

def test_defaults_to_str(chartevents):
    res = defaults_to_str(chartevents.defaults)
    assert res == "`charttime` (index), `valuenum` (val), `valueuom` (unit)"

def test_time_vars_to_str(chartevents):
    res = time_vars_to_str(chartevents.defaults)
    assert res == "`charttime`, `storetime`"