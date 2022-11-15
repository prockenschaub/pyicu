import pytest

import pandas as pd


def test_mimic_id_windows(mimic_demo):
    res = mimic_demo.id_windows()
    res = mimic_demo.id_windows(copy=False) # call again to check cache
    # TODO: add more assertions here to check that the correct id windows are returned
    assert isinstance(res, pd.DataFrame)

# TODO: implement
# def test_mimic_id_map(mimic_demo):
#     res = mimic_demo.id_map('icustay', 'hadm')