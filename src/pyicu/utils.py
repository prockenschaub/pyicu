from typing import Any, List, Iterable
import numpy as np
import pandas as pd
from .container import pyICUTbl

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


def concat_tbls(objs: Iterable[pyICUTbl], *args, **kwargs):
    # TODO: check that all of same type
    metavars = objs[0]._metadata
    metadata = {k: getattr(objs[0], k) for k in metavars}
    
    # Check that all tables share the same metadata
    for obj in objs:
        if obj._metadata != metavars:
            raise ValueError(
                f"Expected all tables to have the following _metadata {metavars}, "
                f"but got {obj._metadata} instead."
            )
        for k, v in metadata.items():
            if getattr(obj, k) != v:
                raise ValueError(
                    f"Expected all tables to have the same value `{k}`={v}, "
                    f"but got {getattr(obj, k)} instead."
                )
    
    # Do concatenation
    res = pd.concat(objs, *args, **kwargs)
    for k, v in metadata.items():
        setattr(res, k, v)
    return res
    


def print_list(x: List, max_char=75):
    repr = x.__repr__()
    return repr[:max_char] + ('...]' if len(repr) > max_char else '')


def intersect(x: List, y: List):
    return sorted(set(x) & set(y), key = x.index)

def union(x: List, y: List):
    return sorted(set(x) | set(y), key = x.index)