import pytest
import numpy as np
from pyicu.container.time import hours, minutes


@pytest.fixture
def id_tbl(example_df):
    return example_df.set_index(["id_var"])


@pytest.fixture
def ts_tbl(example_df):
    return example_df.set_index(["id_var", "index_var"])


@pytest.fixture
def win_tbl(example_df):
    return example_df.set_index(["id_var", "index_var", "win_var"])


def test_validate(id_tbl, ts_tbl, win_tbl):
    id_tbl.tbl._validate()
    ts_tbl.tbl._validate()
    win_tbl.tbl._validate()


def test_validate_errors(example_df):
    with pytest.raises(AttributeError) as e_info:
        example_df.tbl._validate()
    assert e_info.match("table must have named index.*")
    with pytest.raises(AttributeError) as e_info:
        example_df.set_index(["index_var", "id_var"]).tbl._validate()
    assert e_info.match(".*the second must be a time index.*")
    with pytest.raises(AttributeError) as e_info:
        example_df.set_index(["index_var", "id_var", "win_var"]).tbl._validate()
    assert e_info.match(".*the second and third must be a time index.*")
    with pytest.raises(AttributeError) as e_info:
        example_df.set_index(["id_var", "index_var", "win_var", "val_var"]).tbl._validate()
    assert e_info.match("only .* two or three levels are supported")
    with pytest.raises(AttributeError) as e_info:
        example_df.set_index(["index_var"]).tbl._validate()
    assert e_info.match("must have at least one non-time index")

    example_df.set_index(["id_var", "index_var"]).tbl._validate()
    example_df.set_index(["id_var", "index_var", "win_var"]).tbl._validate()


def test_is_id_tbl(example_df, id_tbl, ts_tbl, win_tbl):
    assert example_df.tbl.is_id_tbl() == False
    assert id_tbl.tbl.is_id_tbl() == True
    assert ts_tbl.tbl.is_id_tbl() == False
    assert win_tbl.tbl.is_id_tbl() == False


def test_set_id(example_df):
    with pytest.warns(Warning) as w_info:
        example_df.tbl.set_id_var("id_var").tbl.set_id_var("id_var")
    with pytest.raises(ValueError) as e_info:
        example_df.tbl.set_id_var("unknown_var")
    assert e_info.match("tried to set Id to unknown column")


def test_pandas_as_id_tbl(example_df):
    id_tbl1 = example_df.tbl.as_id_tbl(id_var="id_var")
    assert id_tbl1.tbl.id_var == "id_var"
    id_tbl2 = example_df.tbl.as_id_tbl()
    assert id_tbl2.tbl.id_var == "id_var"
    id_tbl3 = example_df[["index_var", "id_var"]].tbl.as_id_tbl(id_var="id_var")
    assert id_tbl3.tbl.id_var == "id_var"
    with pytest.raises(TypeError) as e_info:
        example_df[["index_var"]].tbl.as_id_tbl()
    assert e_info.match(".*no suitable non-time column could be found.*")


def test_id_tbl_as_id_tbl(id_tbl):
    id_tbl1 = id_tbl.tbl.as_id_tbl(id_var="id_var")
    assert id_tbl1.tbl.id_var == "id_var"
    id_tbl2 = id_tbl.tbl.as_id_tbl(id_var="val_var")
    assert id_tbl2.tbl.id_var == "val_var"


def test_ts_tbl_as_id_tbl(ts_tbl):
    id_tbl1 = ts_tbl.tbl.as_id_tbl(id_var="val_var")
    assert id_tbl1.tbl.is_id_tbl()
    assert id_tbl1.tbl.id_var == "val_var"
    assert all([i in id_tbl1.columns for i in ts_tbl.index.names])
    id_tbl2 = ts_tbl.tbl.as_id_tbl()
    assert id_tbl2.tbl.is_id_tbl()
    assert id_tbl2.tbl.id_var == "id_var"


def test_is_ts_tbl(example_df, id_tbl, ts_tbl, win_tbl):
    assert example_df.tbl.is_ts_tbl() == False
    assert id_tbl.tbl.is_ts_tbl() == False
    assert ts_tbl.tbl.is_ts_tbl() == True
    assert win_tbl.tbl.is_ts_tbl() == False


def test_index_var(id_tbl, ts_tbl):
    with pytest.raises(AttributeError) as e_info:
        id_tbl.tbl.index_var
    assert e_info.match("id_tbl does not have an index_var attribute")
    assert ts_tbl.tbl.index_var == "index_var"


def test_interval(id_tbl, ts_tbl):
    with pytest.raises(AttributeError) as e_info:
        id_tbl.tbl.interval
    assert e_info.match("id_tbl does not have an interval attribute")
    assert ts_tbl.tbl.interval == hours(1)


def test_set_index(id_tbl):
    with pytest.warns(Warning) as w_info:
        id_tbl.tbl.set_index_var("index_var").tbl.set_index_var("index_var")
    with pytest.raises(ValueError) as e_info:
        id_tbl.tbl.set_index_var("unknown_var")
    assert e_info.match("tried to set Index to unknown column.*")
    with pytest.raises(TypeError) as e_info:
        id_tbl.tbl.set_index_var("val_var")
    assert e_info.match("index var must be TimeDtype.*")


def test_pandas_as_ts_tbl(example_df):
    ts_tbl1 = example_df.tbl.as_ts_tbl(id_var="id_var", index_var="index_var")
    assert ts_tbl1.tbl.id_var == "id_var" and ts_tbl1.tbl.index_var == "index_var"
    ts_tbl2 = example_df.tbl.as_ts_tbl(id_var="id_var")
    assert ts_tbl2.tbl.id_var == "id_var" and ts_tbl2.tbl.index_var == "index_var"
    ts_tbl3 = example_df[["id_var", "val_var", "index_var"]].tbl.as_ts_tbl(id_var="id_var")
    assert ts_tbl3.tbl.id_var == "id_var" and ts_tbl3.tbl.index_var == "index_var"
    with pytest.raises(TypeError) as e_info:
        example_df[["id_var", "val_var"]].tbl.as_ts_tbl()
    assert e_info.match(".*no suitable time column could be found.*")


def test_id_tbl_as_ts_tbl(id_tbl):
    ts_tbl1 = id_tbl.tbl.as_ts_tbl(id_var="val_var", index_var="index_var")
    assert ts_tbl1.tbl.id_var == "val_var" and ts_tbl1.tbl.index_var == "index_var"


def test_ts_tbl_as_ts_tbl(ts_tbl):
    ts_tbl1 = ts_tbl.tbl.as_ts_tbl()
    assert ts_tbl1.tbl.id_var == "id_var" and ts_tbl1.tbl.index_var == "index_var"
    ts_tbl2 = ts_tbl.tbl.as_ts_tbl(id_var="val_var")
    assert ts_tbl2.tbl.id_var == "val_var" and ts_tbl2.tbl.index_var == "index_var"
    ts_tbl3 = ts_tbl.tbl.as_ts_tbl(id_var="val_var", index_var="win_var")
    assert ts_tbl3.tbl.id_var == "val_var" and ts_tbl3.tbl.index_var == "win_var"


def test_rename_all(ts_tbl):
    renamed = ts_tbl.tbl.rename_all({"id_var": "id", "val_var": "val"})
    assert renamed.tbl.id_var == "id"
    assert "val" in renamed.columns


def test_change_interval_id_tbl(id_tbl):
    changed = id_tbl.tbl.change_interval(minutes(5))
    assert changed.index_var.dtype == minutes(5)
    assert changed.win_var.dtype == minutes(5)
    assert np.all(changed.index_var.values == np.array([0.0, 12.0, 12.0, 24.0, 0.0]))


def test_change_interval_ts_tbl(ts_tbl):
    changed = ts_tbl.tbl.change_interval(minutes(5))
    assert changed.index.levels[1].dtype == minutes(5)
    assert changed.win_var.dtype == minutes(5)
    assert np.all(changed.index.get_level_values(1) == np.array([0.0, 12.0, 12.0, 24.0, 0.0]))
    changed = ts_tbl.tbl.change_interval(minutes(5), cols="index_var")
    assert changed.index.levels[1].dtype == minutes(5)
    assert np.all(changed.index.get_level_values(1) == np.array([0.0, 12.0, 12.0, 24.0, 0.0]))
