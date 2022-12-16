import pytest
from pathlib import Path

from pyicu.concepts import NumConcept

if not Path("tests/data/mimiciii-demo/1.4").exists():
    pytest.skip("no local test data", allow_module_level=True)


def test_load_num_cnctp_id_tbl(default_dict, mimic_demo):
    assert isinstance(default_dict["weight"], NumConcept)
    res = default_dict.load_concepts("weight", mimic_demo)
    assert res.tbl.is_id_tbl()
    assert res.tbl.id_var == "icustay"


def test_load_num_cnctp_ts_tbl(default_dict, mimic_demo):
    assert isinstance(default_dict["hr"], NumConcept)
    res = default_dict.load_concepts("hr", mimic_demo)
    assert res.tbl.is_ts_tbl()
    assert res.tbl.id_var == "icustay"
    assert res.tbl.index_var == "time"
