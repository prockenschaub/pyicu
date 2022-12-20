import operator
from typing import Any, Callable, Dict, List
from numpy.typing import ArrayLike

import numpy as np
import pandas as pd

from ..utils import enlist, print_list
from ..sources import Src
from ..interval import minutes


def identity_callback(x: Any, *args, **kwargs) -> Any:
    return x


def transform_fun(fun: Callable, *args, **kwargs) -> Callable:
    transf_args = list(args)
    transf_kwargs = dict(kwargs)

    def transformer(x: pd.DataFrame, val_var=None, *args, **kwargs):
        # TODO: this currently changes values by reference. is that okay or do we need to copy?
        if val_var is None:
            val_var = x.columns[0]  # TODO: pick a reasonable default, at least
        x[val_var] = fun(x[val_var], *transf_args, **transf_kwargs)
        return x

    return transformer


def set_val(val: int | float | str | bool) -> Callable:
    def setter(x):
        return x.apply(lambda _: val)

    return setter


def fahr_to_cels(x: ArrayLike) -> ArrayLike:
    return (x - 32) * 5 / 9


def convert_unit(fun, new, rgx=None, ignore_case=True, *args, **kwargs):
    def chr_to_fun(a):
        if callable(a):
            return a
        a = str(a)
        return lambda _: a

    fun = enlist(fun)

    rgx = enlist(rgx)
    if rgx is None:
        rgx = [None for _ in range(len(fun))]

    new = enlist(new)
    new = [chr_to_fun(i) for i in new]

    # TODO: implement broadcasting
    length = len(new)

    def converter(x, val_var, unit_var, *args, **kwargs):
        for i in range(length):
            if rgx[i] is None:
                x[val_var] = fun[i](x[val_var])
                x[unit_var] = new[i](x[unit_var])
            else:
                rows = x[unit_var].str.contains(rgx[i], case=not ignore_case)
                x.loc[rows, val_var] = fun[i](x.loc[rows, val_var])
                x.loc[rows, unit_var] = new[i](x.loc[rows, unit_var])
        return x

    return converter


def binary_op(op: str | Callable, y: ArrayLike) -> Callable:
    if isinstance(op, str):
        op = getattr(operator, op)

    def calculator(x: ArrayLike):
        return op(x, y)

    return calculator


def comp_na(op: str | Callable, y: ArrayLike) -> Callable:
    op = binary_op(op, y)

    def comparator(x: ArrayLike):
        return ~x.isna() & op(x)

    return comparator


def combine_callbacks(*args) -> Callable:
    funcs = list(args)

    def combinator(x, *args, **kwargs):
        for f in funcs:
            x = f(x, *args, **kwargs)
        return x

    return combinator


def apply_map(map: Dict, var: str = "val_var"):
    def mapper(x, *args, **kwargs):
        val_var = kwargs.get("val_var")
        map_var = kwargs.get(var)
        x[val_var] = x[map_var].replace(map)
        return x

    return mapper


def los_callback(src: Src, itm: "Item", id_type: str, interval: pd.Timedelta) -> pd.DataFrame:
    win = itm.win_type
    cfg = src.id_cfg

    res = src.id_map(cfg[id_type].id, cfg[win].id, in_time="start", out_time="end")

    if win == id_type:
        res["val_var"] = res["end"]
    else:
        res["val_var"] = res["end"] - res["start"]

        if (
            cfg.loc[cfg.name == win].index > cfg.loc[cfg.name == id_type].index
        ):  # TODO: refactor after changing how id_cfg works
            res = res.drop_duplicates()

    res["val_var"] = res["val_var"] // minutes(1) / 60 / 24
    res = res.drop(columns=[cfg[win].id, "start", "end"], errors="ignore")

    return res


def collect_concepts(
    x,
    concepts: str | List[str],
    interval,
    merge_dat=False,
    **kwargs,
):
    # TODO: improve error messages here
    concepts = enlist(concepts)
    if isinstance(x, pd.DataFrame):
        if len(concepts) > 1:
            raise ValueError(f"expected {len(concepts)} concepts {print_list(concepts)} but received a single table instead")
        x = {concepts[0]: x}
    elif not isinstance(x, dict):
        raise TypeError(f"expected `x` to be a DataFrame or dict of DataFrames, got {x.__class__}")

    x = {n: v for n, v in x.items() if n in concepts}

    if len(x) != len(concepts):
        raise ValueError(f"expected concepts {concepts} but got {list(x.keys())}")
    if any([not isinstance(i, pd.DataFrame) for i in x.values()]):
        raise TypeError(f"expected `x` to be a DataFrame or dict of DataFrames, got dict with other types")
    if any([k not in v.columns for k, v in x.items()]):
        raise ValueError(f"concept DataFrames must contain a column with the concept name")

    # TODO: check interval
    # TODO: enable merge_dat

    if len(x) == 1:
        return list(x.values())[0]
    return x

def ts_to_win_tbl(win_dur: pd.Timedelta) -> Callable:
    def converter(x: pd.DataFrame, *args, **kwargs):
        x['dur_var'] = win_dur
        return x.tbl.as_win_tbl(dur_var="dur_var")
        
    return converter