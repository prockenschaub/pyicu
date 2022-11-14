import pytest

import pandas as pd

def test_mimic_id_windows(mimic_demo):
    res = mimic_demo._id_win_helper()
    # TODO: add more assertions here to check that the correct id windows are returned
    assert isinstance(res, pd.DataFrame)


