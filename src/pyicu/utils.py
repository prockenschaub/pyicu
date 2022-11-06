from typing import Any, List

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

def print_list(x: List, max_char=75):
    repr = x.__repr__()
    return repr[:max_char] + ('...]' if len(repr) > max_char else '')


def intersect(x: List, y: List):
    return sorted(set(x) & set(y), key = x.index)

def union(x: List, y: List):
    return sorted(set(x) | set(y), key = x.index)