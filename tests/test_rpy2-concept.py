import pytest
import os
import subprocess
from pathlib import Path
from contextlib import contextmanager
import rpy2
import sys
import pyicu
from pyicu.configs.load import load_src_cfg
from pyicu.sources import MIMIC
from pyicu.concepts import ConceptDict
from rpy2.robjects.packages import importr
from concepts_to_test import concepts

def prepare_rpy2():
    # Import R's "base" and "utils" package
    base = importr('base')
    utils = importr('utils')
    utils.chooseCRANmirror(ind=1)
    
    # Install "ricu" package from GitHub, using "remotes" library
    utils.install_packages('remotes')
    remotes = importr('remotes')
    remotes.install_github("eth-mds/ricu")
    ricu = importr('ricu')

    # Install mimic_demo dataset for ricu
    utils.install_packages("mimic.demo", repos = "https://eth-mds.github.io/physionet-demo")
    
    return ricu

def download_mimic_demo_pyicu():
    @contextmanager
    def directory(path):
        oldpwd = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(oldpwd)

    url = "https://physionet.org/files/mimiciii-demo/1.4/"

    download_dir = Path("examples/data")
    download_dir.mkdir(parents=True, exist_ok=True)

    with directory(download_dir):
        subprocess.run(["wget", "-r", "-N", "-c", "-np", url])
    
    return download_dir

@pytest.fixture(params=concepts)
def concept_param(request):
    return request.param

@pytest.fixture
def load_ricu_concept(concept_param):
    ricu = prepare_rpy2()
    concept_ricu = ricu.load_concepts(concept_param, "mimic_demo", verbose = rpy2.robjects.vectors.BoolVector([False]))
    return concept_ricu

@pytest.fixture
def load_pyicu_concept(concept_param):
    # Prepare pyicu and load mimic_demo dataset
    download_mimic_demo_pyicu()
    if pyicu.__file__.endswith("pyicu/src/pyicu/__init__.py"):
        sys.path.append(pyicu.__file__[:-22])
        
    # Create mimic_demo object
    data_dir = Path("examples/data/physionet.org/files/mimiciii-demo/1.4")
    mimic_cfg = load_src_cfg("mimic_demo")
    mimic_cfg.do_import(data_dir)
    mimic_demo = MIMIC(mimic_cfg, data_dir)

    # Load concepts from pyicu
    concepts = ConceptDict.from_defaults()
    concept_pyicu = concepts.load_concepts(concept_param, mimic_demo)
    return concept_pyicu

@pytest.mark.parametrize("concept_param", concepts)
def test_compare_ricu_pyicu(load_ricu_concept, load_pyicu_concept):    
    # Load concepts from ricu and pyicu
    concept_ricu = load_ricu_concept
    concept_pyicu = load_pyicu_concept
    
    # Check if the length of the two concepts are the same
    assert len(concept_ricu[0]) == len(concept_pyicu)
    
    print(concept_pyicu[0])
    
    #print(concept_ricu[0][0])
    #print(concept_pyicu[0][0])
    
    # Check if icu stay ids of first icu stay are the same
    # assert concept_ricu[0][0] == concept_pyicu[0][0]

    # Print results from ricu and pyicu
    print(concept_ricu)
    print(concept_pyicu)
