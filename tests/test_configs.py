import pytest


@pytest.fixture
def id_cfg(mimic_demo_cfg):
    return mimic_demo_cfg.ids


def test_id_cfg_id_var(id_cfg):
    assert id_cfg.id_var == "icustay"


def test_id_cfg_getter_type_error(id_cfg):
    with pytest.raises(TypeError) as e_info:
        id_cfg[0]
    assert e_info.match("expected an Id type .* as string")


def test_id_cfg_getter_value_error(id_cfg):
    with pytest.raises(ValueError) as e_info:
        id_cfg["non-existing-id"]
    assert e_info.match("Id type .* not defined")
