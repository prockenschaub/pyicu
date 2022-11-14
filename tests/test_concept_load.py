import pytest
from pathlib import Path

from pyicu.container import IdTbl, TsTbl
from pyicu.concepts import NumConcept

if not Path("tests/data/mimiciii-demo/1.4").exists():
    pytest.skip("no local test data", allow_module_level=True)

def test_load_num_cnctp_id_tbl(default_dict, mimic_demo):
    assert isinstance(default_dict['weight'], NumConcept)
    res = default_dict.load_concepts("weight", mimic_demo)
    assert isinstance(res, list)
    assert len(res) == 1
    res = res[0]
    assert isinstance(res, IdTbl)
    assert res.id_var == "icustay"

def test_load_num_cnctp_ts_tbl(default_dict, mimic_demo):
    assert isinstance(default_dict['hr'], NumConcept)
    res = default_dict.load_concepts("hr", mimic_demo)
    assert isinstance(res, list)
    assert len(res) == 1
    res = res[0]
    assert isinstance(res, TsTbl)
    assert res.id_var == "icustay"
    assert res.index_var == "charttime"

