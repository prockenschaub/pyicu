import pytest
import numpy as np
import pandas as pd
from pyicu.interval import hours, minutes


@pytest.fixture
def id_tbl(example_df):
    return example_df.set_index(["id_var"])


@pytest.fixture
def ts_tbl(example_df):
    return example_df.set_index(["id_var", "index_var"])


@pytest.fixture
def win_tbl(example_df):
    return example_df.set_index(["id_var", "index_var", "dur_var"])


def test_validate(id_tbl, ts_tbl, win_tbl):
    id_tbl.icu._validate()
    ts_tbl.icu._validate()
    win_tbl.icu._validate()


def test_validate_errors(example_df):
    with pytest.raises(AttributeError) as e_info:
        example_df.icu._validate()
    assert e_info.match("table must have named index.*")
    with pytest.raises(AttributeError) as e_info:
        example_df.set_index(["index_var", "id_var"]).icu._validate()
    assert e_info.match(".*the second must be a time index.*")
    with pytest.raises(AttributeError) as e_info:
        example_df.set_index(["index_var", "id_var", "dur_var"]).icu._validate()
    assert e_info.match(".*the second and third must be a time index.*")
    with pytest.raises(AttributeError) as e_info:
        example_df.set_index(["id_var", "index_var", "dur_var", "val_var"]).icu._validate()
    assert e_info.match("only .* two or three levels are supported")
    with pytest.raises(AttributeError) as e_info:
        example_df.set_index(["index_var"]).icu._validate()
    assert e_info.match("must have at least one non-time index")

    example_df.set_index(["id_var", "index_var"]).icu._validate()
    example_df.set_index(["id_var", "index_var", "dur_var"]).icu._validate()


def test_is_id_tbl(example_df, id_tbl, ts_tbl, win_tbl):
    assert example_df.icu.is_id_tbl() == False
    assert id_tbl.icu.is_id_tbl() == True
    assert ts_tbl.icu.is_id_tbl() == False
    assert win_tbl.icu.is_id_tbl() == False


def test_set_id(example_df):
    with pytest.raises(ValueError) as e_info:
        example_df.icu.set_id_var("unknown_var")
    assert e_info.match("tried to set id to unknown column")


def test_pandas_as_id_tbl(example_df):
    id_tbl1 = example_df.icu.as_id_tbl(id_var="id_var")
    assert id_tbl1.icu.id_var == "id_var"
    id_tbl2 = example_df.icu.as_id_tbl()
    assert id_tbl2.icu.id_var == "id_var"
    id_tbl3 = example_df[["index_var", "id_var"]].icu.as_id_tbl(id_var="id_var")
    assert id_tbl3.icu.id_var == "id_var"
    with pytest.raises(TypeError) as e_info:
        example_df[["index_var"]].icu.as_id_tbl()
    assert e_info.match(".*no suitable non-time column could be found.*")


def test_id_tbl_as_id_tbl(id_tbl):
    id_tbl1 = id_tbl.icu.as_id_tbl(id_var="id_var")
    assert id_tbl1.icu.id_var == "id_var"
    id_tbl2 = id_tbl.icu.as_id_tbl(id_var="val_var")
    assert id_tbl2.icu.id_var == "val_var"


def test_ts_tbl_as_id_tbl(ts_tbl):
    id_tbl1 = ts_tbl.icu.as_id_tbl(id_var="val_var")
    assert id_tbl1.icu.is_id_tbl()
    assert id_tbl1.icu.id_var == "val_var"
    assert all([i in id_tbl1.columns for i in ts_tbl.index.names])
    id_tbl2 = ts_tbl.icu.as_id_tbl()
    assert id_tbl2.icu.is_id_tbl()
    assert id_tbl2.icu.id_var == "id_var"

def test_win_tbl_as_id_tbl(win_tbl):
    id_tbl1 = win_tbl.icu.as_id_tbl(id_var="val_var")
    assert id_tbl1.icu.is_id_tbl()
    assert id_tbl1.icu.id_var == "val_var"
    assert all([i in id_tbl1.columns for i in win_tbl.index.names])
    id_tbl2 = win_tbl.icu.as_id_tbl()
    assert id_tbl2.icu.is_id_tbl()
    assert id_tbl2.icu.id_var == "id_var"


def test_is_ts_tbl(example_df, id_tbl, ts_tbl, win_tbl):
    assert example_df.icu.is_ts_tbl() == False
    assert id_tbl.icu.is_ts_tbl() == False
    assert ts_tbl.icu.is_ts_tbl() == True
    assert win_tbl.icu.is_ts_tbl() == False


def test_index_var(id_tbl, ts_tbl):
    with pytest.raises(AttributeError) as e_info:
        id_tbl.icu.index_var
    assert e_info.match("id_tbl does not have an index_var attribute")
    assert ts_tbl.icu.index_var == "index_var"


def test_set_index(id_tbl):
    with pytest.raises(ValueError) as e_info:
        id_tbl.icu.set_index_var("unknown_var")
    assert e_info.match("tried to set index to unknown column.*")
    with pytest.raises(TypeError) as e_info:
        id_tbl.icu.set_index_var("val_var")
    assert e_info.match("index var must be timedelta.*")


def test_pandas_as_ts_tbl(example_df):
    ts_tbl1 = example_df.icu.as_ts_tbl(id_var="id_var", index_var="index_var")
    assert ts_tbl1.icu.id_var == "id_var" and ts_tbl1.icu.index_var == "index_var"
    ts_tbl2 = example_df.icu.as_ts_tbl(id_var="id_var")
    assert ts_tbl2.icu.id_var == "id_var" and ts_tbl2.icu.index_var == "index_var"
    ts_tbl3 = example_df[["id_var", "val_var", "index_var"]].icu.as_ts_tbl(id_var="id_var")
    assert ts_tbl3.icu.id_var == "id_var" and ts_tbl3.icu.index_var == "index_var"
    with pytest.raises(TypeError) as e_info:
        example_df[["id_var", "val_var"]].icu.as_ts_tbl()
    assert e_info.match(".*no suitable time column could be found.*")


def test_id_tbl_as_ts_tbl(id_tbl):
    ts_tbl1 = id_tbl.icu.as_ts_tbl(id_var="val_var", index_var="index_var")
    assert ts_tbl1.icu.id_var == "val_var" and ts_tbl1.icu.index_var == "index_var"


def test_ts_tbl_as_ts_tbl(ts_tbl):
    ts_tbl1 = ts_tbl.icu.as_ts_tbl()
    assert ts_tbl1.icu.id_var == "id_var" and ts_tbl1.icu.index_var == "index_var"
    ts_tbl2 = ts_tbl.icu.as_ts_tbl(id_var="val_var")
    assert ts_tbl2.icu.id_var == "val_var" and ts_tbl2.icu.index_var == "index_var"
    ts_tbl3 = ts_tbl.icu.as_ts_tbl(id_var="val_var", index_var="dur_var")
    assert ts_tbl3.icu.id_var == "val_var" and ts_tbl3.icu.index_var == "dur_var"


def test_win_tbl_as_ts_tbl(win_tbl):
    ts_tbl1 = win_tbl.icu.as_ts_tbl()
    assert ts_tbl1.icu.is_ts_tbl()
    assert ts_tbl1.icu.id_var == "id_var" and ts_tbl1.icu.index_var == "index_var"


def test_set_dur(ts_tbl):
    with pytest.raises(ValueError) as e_info:
        ts_tbl.icu.set_dur_var("unknown_var")
    assert e_info.match("tried to set duration to unknown column.*")
    with pytest.raises(TypeError) as e_info:
        ts_tbl.icu.set_dur_var("val_var")
    assert e_info.match("duration var must be timedelta.*")


def test_is_win_tbl(example_df, id_tbl, ts_tbl, win_tbl):
    assert example_df.icu.is_win_tbl() == False
    assert id_tbl.icu.is_win_tbl() == False
    assert ts_tbl.icu.is_win_tbl() == False
    assert win_tbl.icu.is_win_tbl() == True


def test_pandas_as_win_tbl(example_df):
    def assert_vars(x, id, index, dur):
        assert x.icu.id_var == id
        assert x.icu.index_var == index
        assert x.icu.dur_var == dur
    
    win_tbl1 = example_df.icu.as_win_tbl(id_var="id_var", index_var="index_var", dur_var="dur_var")
    assert_vars(win_tbl1, "id_var", "index_var", "dur_var")
    win_tbl2 = example_df.icu.as_win_tbl(id_var="id_var")
    assert_vars(win_tbl2, "id_var", "index_var", "dur_var")
    win_tbl3 = example_df[["id_var", "val_var", "dur_var", "index_var"]].icu.as_win_tbl(id_var="id_var")
    assert_vars(win_tbl3, "id_var", "dur_var", "index_var")
    with pytest.raises(TypeError) as e_info:
        example_df[["id_var", "index_var", "val_var"]].icu.as_win_tbl()
    assert e_info.match(".*no suitable time column could be found.*")


def test_id_tbl_as_win_tbl(id_tbl):
    win_tbl1 = id_tbl.icu.as_win_tbl(dur_var="dur_var")
    assert win_tbl1.icu.is_win_tbl()
    assert win_tbl1.icu.dur_var == "dur_var"
    win_tbl2 = id_tbl.icu.as_win_tbl(index_var="dur_var")
    assert win_tbl2.icu.dur_var == "index_var"


def test_ts_tbl_as_win_tbl(ts_tbl):
    win_tbl1 = ts_tbl.icu.as_win_tbl(dur_var="dur_var")
    assert win_tbl1.icu.is_win_tbl()
    assert win_tbl1.icu.dur_var == "dur_var"
    win_tbl2 = ts_tbl.icu.as_win_tbl()
    assert win_tbl2.icu.is_win_tbl()
    assert win_tbl2.icu.dur_var == "dur_var"


def test_win_tbl_as_win_tbl(win_tbl):
    win_tbl1 = win_tbl.icu.as_win_tbl()
    assert win_tbl1.icu.is_win_tbl()
    assert win_tbl1.icu.dur_var == "dur_var"
    win_tbl['new_dur_var'] = win_tbl.index.get_level_values(2)
    win_tbl2 = win_tbl.icu.as_win_tbl(dur_var="new_dur_var")
    assert win_tbl2.icu.is_win_tbl()
    assert win_tbl2.icu.dur_var == "new_dur_var"
    assert "dur_var" in win_tbl2.columns
    

def test_rename_all(ts_tbl):
    renamed = ts_tbl.icu.rename_all({"id_var": "id", "val_var": "val"})
    assert renamed.icu.id_var == "id"
    assert "val" in renamed.columns


def test_change_interval_id_tbl(id_tbl):
    changed = id_tbl.icu.change_interval(hours(2))
    assert np.all(changed.index_var == pd.to_timedelta([0.0, 0.0, 0.0, 2.0, 0.0], "hours"))


def test_change_interval_ts_tbl(ts_tbl):
    changed = ts_tbl.icu.change_interval(hours(2))
    assert np.all(changed.index.get_level_values(1) == pd.to_timedelta([0.0, 0.0, 0.0, 2.0, 0.0], "hours"))
    changed = ts_tbl.icu.change_interval(hours(2), cols="index_var")
    assert np.all(changed.index.get_level_values(1) == pd.to_timedelta([0.0, 0.0, 0.0, 2.0, 0.0], "hours"))
