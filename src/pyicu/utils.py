from typing import Any, List, Iterable
import numpy as np
import pandas as pd
import string
import random

def enlist(x: Any):
    # TODO: Test for scalar instead
    if x is None:
        return None
    elif not isinstance(x, list):
        return [x]
    else:
        return x

def coalesce(**kwargs):
    res = {}
    for k, v in kwargs.items():
        if v is not None:
            res[k] = v
    return res

def concat_tbls(objs: Iterable[pd.DataFrame], *args, **kwargs):
    # TODO: check that all of same type
    metavars = objs[0]._metadata
    metadata = {k: getattr(objs[0], k) for k in metavars}

    # Check that all tables share the same metadata
    for obj in objs:
        if obj._metadata != metavars:
            raise ValueError(
                f"expected all tables to have the following _metadata {metavars} but got {obj._metadata} instead."
            )
        for k, v in metadata.items():
            if getattr(obj, k) != v:
                raise ValueError(f"expected all tables to have the same value `{k}`={v} but got {getattr(obj, k)} instead.")

    # Do concatenation
    res = pd.concat(objs, *args, **kwargs)
    for k, v in metadata.items():
        setattr(res, k, v)
    return res


def print_list(x: List, max_char=75):
    repr = x.__repr__()
    return repr[:max_char] + ("...]" if len(repr) > max_char else "")


def intersect(x: List, y: List):
    return sorted(set(x) & set(y), key=x.index)


def union(x: List, y: List):
    return sorted(set(x) | set(y), key=(x + y).index)


def diff(x: List, y: List):
    return sorted(set(x) - set(y), key=x.index)

def prcnt(x: int | float, tot: int | float) -> str:
    return f"{np.round(x / tot * 100, decimals=2)}%"


def nrow(x: pd.DataFrame) -> int:
    return x.shape[0]

def ncol(x: pd.DataFrame) -> int:
    return x.shape[1]

def rm_na(x, cols: str | List[str] | None = None, mode: str = "all"):
    return x.dropna(how=mode, subset=cols, axis=0)

def rm_na_val_var(x: pd.DataFrame, col: str = "val_var") -> pd.DataFrame:
    n_row = nrow(x)
    x = rm_na(x, col)
    n_rm = n_row - nrow(x)

    if n_rm > 0:
        print(f"removed {n_rm} ({prcnt(n_rm, n_row)}) of rows due to missing values")
    return x

def new_names(
    old_names: List[str] | pd.DataFrame | None = None,
    n: int = 1,
    chars: str = string.ascii_letters + string.digits,
    length: int = 15,
) -> str | List[str]:
    if isinstance(old_names, pd.DataFrame):
        old_names = old_names.columns

    while True:
        res = ["".join(random.choice(chars) for _ in range(length)) for _ in range(n)]
        if len(res) == len(set(res)) and len(set(res) & set(old_names)) == 0:
            break

    if n == 1:
        res = res[0]
    return res