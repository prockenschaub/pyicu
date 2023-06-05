### Test rpy2 to compare code/fucntions in "pyicu" and "ricu"
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

# import R's "base" and "utils" package
from rpy2.robjects.packages import importr
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

# Load "hr" concept from ricu
hr_ricu = ricu.load_concepts("hr", "mimic_demo", verbose = rpy2.robjects.vectors.BoolVector([False]))

# Download mimic demo dataset for pyicu
@contextmanager
def directory(path):
    # Code copied from https://stackoverflow.com/questions/299446/how-do-i-change-directory-back-to-my-original-working-directory-with-python
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

# Load "hr" concept from pyicu
if pyicu.__file__.endswith("pyicu/src/pyicu/__init__.py"):
    sys.path.append(pyicu.__file__[:-22])
    
data_dir = Path("examples/data/physionet.org/files/mimiciii-demo/1.4")
mimic_cfg = load_src_cfg("mimic_demo")
mimic_cfg.do_import(data_dir)
print(mimic_cfg)
mimic_demo = MIMIC(mimic_cfg, data_dir)
print(mimic_demo.print_available())

concepts = ConceptDict.from_defaults()
print("Concept: ", concepts['hr'])
print("Items:   ", concepts['hr'].src_items(mimic_demo))
hr_pyicu = concepts.load_concepts("hr", mimic_demo)

# Print results from ricu and pyicu
print(hr_ricu)
print(hr_pyicu)


# ToDo:
# Continue implementing helper functions until sofa score works!
# Rewrite --> form of unit tests --> unit tests for checking the concepts e.g. sofa score!
# Finally, refactor the helper functions!
# Create pull request!
# Tools: Github Copilot
