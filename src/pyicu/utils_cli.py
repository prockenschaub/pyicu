import click
import textwrap
import inspect
import warnings

def fmt_msg(msg, envir=None, indent=0, exdent=0):
    if envir is None:
        envir = globals().copy()

    msg = [click.style(m, fg='yellow') for m in msg]  # Example: applying yellow color to each message
    msg = [textwrap.fill(m, initial_indent=' ' * indent, subsequent_indent=' ' * exdent, break_long_words=True, break_on_hyphens=False) for m in msg]

    return '\n'.join(msg)

def format_assert(message, class_, envir=None, **kwargs):
    res = fmt_msg(message, envir=envir, **kwargs)
    setattr(res, "assert_class", class_)
    return res

import numpy as np
from itertools import groupby

def suggest(x, opts, n=1, fixed=False, **kwargs):
    dis = adist(x, opts, fixed=fixed, **kwargs)
    res = np.apply_along_axis(top_n, axis=1, arr=dis, n=n)
    
    if res.shape[0] == 1 and n == 1:
        res = res.T
    
    res = [list(group) for _, group in groupby(opts[np.argsort(res)], key=lambda i: res[i])]
    res = dict(zip(x, res))
    
    return res

def adist(x, y, fixed=False):
    len_x, len_y = len(x), len(y)
    dis = np.zeros((len_x, len_y))
    
    for i in range(len_x):
        for j in range(len_y):
            if fixed:
                dis[i, j] = int(x[i] != y[j])
            else:
                dis[i, j] = levenshtein_distance(x[i], y[j])
    
    return dis

def top_n(arr, n=1):
    return np.argpartition(arr, n)[:n]

def levenshtein_distance(s, t):
    len_s, len_t = len(s), len(t)
    dp = np.zeros((len_s + 1, len_t + 1))
    
    for i in range(len_s + 1):
        dp[i, 0] = i
    
    for j in range(len_t + 1):
        dp[0, j] = j
    
    for i in range(1, len_s + 1):
        for j in range(1, len_t + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    
    return dp[len_s, len_t]

class RicuError(Exception):
    def __init__(self, message, _class=None, **kwargs):
        self.message = message
        self._class = _class
        self.kwargs = kwargs

def stop_ricu(message, _class=None, envir=None, indent=0, exdent=0, **kwargs):
    error_message = message if not envir else f"{message} (envir={envir})"
    error_class = _class if _class else ["ricu_err"]
    
    raise RicuError(error_message, _class=error_class, **kwargs)

def stop_generic(x, fun):
    assert isinstance(fun, str)

    frame = inspect.currentframe().f_back
    class_names = ", ".join(frame.f_globals.get("__annotations__", {}).get("es", []))
    class_names = f" {class_names}" if class_names else ""
    
    function_name = frame.f_code.co_name
    error_message = f"No applicable method for generic function `{fun}()` and class{class_names} {x.__class__.__name__}."
    
    stop_ricu(error_message, _class="generic_no_fun")

def warn_arg(args):
    assert isinstance(args, list) and all(isinstance(arg, str) for arg in args), "args must be a list of strings"
    assert len(args) > 0, "args cannot be empty"
    warning_message = f"Ignoring argument(s) passed as {args}"
    warnings.warn(warning_message, category=UserWarning)

def warn_dots(*args, ok_args=None):
    if len(args) > 0:
        frame = inspect.currentframe().f_back
        arg_names = inspect.getargnames(frame)
        if ok_args is not None:
            arg_names = [arg_name for arg_name in arg_names if arg_name not in ok_args]
        warn_arg(arg_names)

    return None




