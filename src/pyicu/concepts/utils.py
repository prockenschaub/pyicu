from typing import Dict, Callable
import pyicu.callbacks as callbacks


def str_to_fun(x: str | Callable | None, globals: Dict = None) -> Callable:
    if x is None:
        return callbacks.identity_callback
    elif callable(x):
        return x
    if globals is None:
        globals = callbacks.__dict__
    try:
        x = eval(x, globals)
    except Exception as e:
        raise ValueError(f"cannot convert string {x} to function, failed because: {e}")
    if not callable(x):
        raise ValueError(f"expected function as callback but got {x.__class__}")
    return x
