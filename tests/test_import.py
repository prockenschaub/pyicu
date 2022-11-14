import pytest
from pathlib import Path

from pyicu.configs import SrcCfg

raw_path = Path("tests/data/mimiciii-demo-raw/1.4")

if not raw_path.exists():
    pytest.skip("no local test data", allow_module_level=True)

def test_import(mimic_demo_cfg):
    if (raw_path/"admissions.parquet").exists():
        (raw_path/"admissions.parquet").unlink()
    mimic_demo_cfg.do_import(raw_path, tables="admissions")
    assert (raw_path/"admissions.parquet").exists()
    (raw_path/"admissions.parquet").unlink()
    