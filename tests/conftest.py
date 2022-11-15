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

test_data_dir = Path("tests/data/mimiciii-demo/1.4")

@pytest.fixture
def example_df():
    return pd.DataFrame({
        "id_var": [1, 1, 2, 2, 3],
        "index_var": [0, 1, 1, 2, 0],
        "val_var": [3., 2., 12., 42., 0.]
    })


@pytest.fixture
def default_dict():
    return ConceptDict.from_defaults()


@pytest.fixture
def mimic_demo_cfg():
    return load_src_cfg("mimic_demo")


@pytest.fixture
def mimic_demo(mimic_demo_cfg):
    return MIMIC(mimic_demo_cfg, test_data_dir)
