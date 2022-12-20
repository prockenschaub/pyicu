import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def result_path():
    return Path("~/pyicu-test-data")


@pytest.fixture(params=["sex", "los_hosp", "los_icu"])
def id_concept(request):
    return request.param


@pytest.fixture(params=["abx", "hr", "temp", "death"])
def ts_concept(request):
    return request.param


@pytest.fixture(params=["hour", "minute"])
def interval(request):
    return request.param


def test_result_id_tbl(mimic_demo, default_dict, result_path, id_concept):
    exp = pd.read_csv(result_path / f"{id_concept}.csv")
    res = default_dict.load_concepts(id_concept, mimic_demo)
    res = res.reset_index()
    res = res.astype(exp.dtypes)
    pd.testing.assert_frame_equal(res, exp)


def test_result_ts_tbl(mimic_demo, default_dict, result_path, ts_concept, interval):
    exp = pd.read_csv(result_path / f"{ts_concept}_{interval}.csv")
    res = default_dict.load_concepts(ts_concept, mimic_demo, interval=pd.Timedelta(1, interval))
    res = res.reset_index()
    res['time'] = res['time'] // pd.Timedelta(1, interval)
    res = res.astype(exp.dtypes)
    pd.testing.assert_frame_equal(res, exp)
