"""
    Dummy conftest.py for pyicu.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import pytest
import pandas as pd
from pathlib import Path
from pyicu.concepts import ConceptDict
from pyicu.configs.load import load_src_cfg
from pyicu.sources import MIMIC
from pyicu.interval import hours

# test_data_dir = Path("tests/data/mimiciii-demo/1.4") # TODO: something is corrupted with abx, check
test_data_dir = Path("examples/data/physionet.org/files/mimiciii-demo/1.4")


@pytest.fixture
def example_df():
    return pd.DataFrame(
        {
            "id_var": [1, 1, 2, 2, 3],
            "index_var": pd.to_timedelta([0.0, 1.0, 1.0, 2.0, 0.0], "hours"),
            "dur_var": pd.to_timedelta([1.0, 0.0, 1.0, 0.0, 1.0], "hours"),
            "val_var": [3.0, 2.0, 12.0, 42.0, 0.0],
        }
    )


@pytest.fixture
def default_dict():
    return ConceptDict.from_defaults()


@pytest.fixture
def mimic_demo_cfg():
    return load_src_cfg("mimic_demo")


@pytest.fixture
def mimic_demo(mimic_demo_cfg):
    return MIMIC(mimic_demo_cfg, test_data_dir)
