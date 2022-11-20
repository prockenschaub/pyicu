from typing import Any, Callable, Dict
from numpy.typing import ArrayLike
from pyicu.container import pyICUTbl, pyICUSeries
from ..utils import enlist
import operator

def identity_callback(x: Any, *args, **kwargs) -> Any:
    return x

def transform_fun(fun: Callable, *args, **kwargs) -> Callable:
    transf_args = list(args)
    transf_kwargs = dict(kwargs)

    def transformer(x: pyICUTbl, val_var=None, *args, **kwargs):
        # TODO: this currently changes values by reference. is that okay or do we need to copy?
        if val_var is None:
            val_var = x.data_var
        x[val_var] = fun(x[val_var], *transf_args, **transf_kwargs)
        return x
    
    return transformer


def set_val(val: int | float | str | bool) -> Callable:
    def setter(x: pyICUSeries):
        x.loc[:] = val
        return x
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


def binary_op(op: str | Callable, y: ArrayLike):
    if isinstance(op, str):
        op = getattr(operator, op)
    def calculator(x: ArrayLike):
        return op(x, y)
    return calculator

def apply_map(map: Dict, var: str='val_var'):
    def mapper(x, *args, **kwargs):
        val_var = kwargs.get('val_var')
        map_var = kwargs.get(var)
        x[val_var] = x[map_var].replace(map)
        return x
    return mapper
