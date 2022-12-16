import pytest
from pathlib import Path
import numpy as np
import pandas as pd

from pyicu.sources import MIMIC

# TODO: fix names between mimic and mimic-demo
# def test_default_load(mimic_demo_cfg):
#     src = MIMIC()


def test_src_properties(mimic_demo):
    name = mimic_demo.name

    assert name == "mimic"
    assert np.all(np.equal(mimic_demo.id_cfg.name.values, np.array(["patient", "hadm", "icustay"])))
    assert mimic_demo.tables[:3] == ["admissions", "callout", "caregivers"]
    assert mimic_demo.tbl_cfg[0].name == "admissions"


def test_src_tbl_rows_and_columns(mimic_demo):
    assert mimic_demo.chartevents.num_rows == 758355
    assert mimic_demo.chartevents.num_cols == 15
    assert mimic_demo.procedures_icd.columns == ["row_id", "subject_id", "hadm_id", "seq_num", "icd9_code"]


def test_src_tbl_id_var(mimic_demo):
    assert mimic_demo.chartevents.id_var is None
    assert mimic_demo.caregivers.id_var == "cgid"


def test_src_tbl_index_var(mimic_demo):
    assert mimic_demo.chartevents.index_var == "charttime"


def test_src_tbl_time_vars(mimic_demo):
    assert mimic_demo.chartevents.time_vars == ["charttime", "storetime"]


def test_src_tbl_to_pandas(mimic_demo):
    res = mimic_demo.chartevents.to_pandas()
    assert isinstance(res, pd.DataFrame)


def test_src_tbl_to_id_tbl(mimic_demo):
    res = mimic_demo.chartevents.to_id_tbl()
    assert res.tbl.is_id_tbl()


def test_src_tbl_to_ts_tbl(mimic_demo):
    res = mimic_demo.chartevents.to_ts_tbl()
    assert res.tbl.is_ts_tbl()


if not Path("tests/data/mimiciii-demo/1.4").exists():
    pytest.skip("no local test data", allow_module_level=True)


def test_src_tbls_available(mimic_demo):
    res = mimic_demo.print_available()
    assert res == "mimic: 25 of 25 tables available"


def test_some_tables_not_imported(mimic_demo_cfg):
    src = MIMIC(mimic_demo_cfg, data_dir=Path("tests/data/mimic-iii-demo-raw/1.4"))
