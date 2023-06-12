from pyicu.utils_cli import fmt_msg, format_assert, suggest
#from pyicu.container.table import TableAccessor
from pyicu.interval import seconds
#from pyicu.utils_misc import col_ply
import math
import datetime
import pint
import numpy as np
import pandas as pd

def get_message(res, assertion, assertion_env):
    return "Assertion failed: " + assertion

def see_if(*args, env=None, msg=None):
    asserts = [expr for expr in args]
    for assertion in asserts:
        try:
            res = eval(assertion, env)
        except AssertionError as e:
            res = False
            msg = e.args[0]
        
        #check_result(res)
        
        if not res:
            if msg is None:
                msg = get_message(res, assertion, env)
            return {"msg": msg, "result": False}
    
    return {"result": True}

'''
def see_if(*args, env=None, msg=None):
    asserts = [(assertion, env) for assertion in args]
    for assertion, assertion_env in asserts:
        try:
            eval(assertion, assertion_env)
        except AssertionError as e:
            res = False
            msg = e.args[0]
        else:
            res = True

        if not res:
            if msg is None:
                msg = get_message(res, assertion, assertion_env)
            return False, msg

    return True
'''

def assert_that(*args, env=None, msg=None, _class=None):
    res = see_if(*args, env=env)
    if res is True:
        return True

    if msg is None:
        msg = res.get("msg")
    else:
        msg = fmt_msg(msg, envir=env)

    cls = [msg.get("assert_class"), _class, "ricu_err", "assertError"]

    raise AssertionError(msg, _class=cls)


def fail_type(arg_name, _class):
    assert_that(isinstance(arg_name, str), isinstance(_class, str))

    def inner(call, env):
        msg = f"{str(call[arg_name])} is not a `{_class}` object"
        format_assert(msg, f"is_{_class}_assert")

    return inner


def is_type(_type):
    def res(x):
        return isinstance(x, _type)

    res.on_failure = fail_type("x", _type)

    return res

def is_flag(x):
    return isinstance(x, bool) and len(x) == 1

def is_scalar(x):
    atomic_types = (int, float, str, bool, complex)
    return isinstance(x, atomic_types) and len(x) == 1


def is_number(x):
    numeric_types = (int, float, complex)
    return is_scalar(x) and isinstance(x, numeric_types)


def is_intish(x):
    numeric_types = (int, float, complex)
    return isinstance(x, int) or (isinstance(x, numeric_types) and all(x == math.trunc(x)) and not np.isnan(x))


def no_na(x):
    return not any(np.isnan(x))


def has_length(x, length=math.nan):
    if math.isnan(length):
        return len(x) > 0
    else:
        return len(x) == length


def has_rows(x):
    return len(x) > 0


def are_in(x, opts, na_rm=False):
    assert_that(has_length(x), has_length(opts), isinstance(x, str), isinstance(opts, str))
    return all(elem in opts for elem in x) if not na_rm else all(elem in opts for elem in x if not pd.isna(elem))

def is_count(obj):
    return isinstance(obj, int) and obj >= 0

def bullet(*args, level=1):
    assert_that(is_count(level))
    assert_that(level <= 3)
    symbols = ["\u2022", "\u25cf", "-"]
    bullet_symbol = symbols[level-1]
    bullet_text = ' '.join(args)
    return bullet_symbol + " " + bullet_text


def chr_ply(x, fun, length=1, use_names=False, *args, **kwargs):
    result = np.vectorize(fun)(x, *args, **kwargs)
    if use_names:
        result = {name: value for name, value in zip(x, result)}
    return result.astype(str).tolist()

def in_failure(call, env):
    x = eval(call["x"], env)
    opts = eval(call["opts"], env)
    sug = suggest([elem for elem in x if elem not in opts], opts)

    if len(sug) == 1:
        format_assert(
            f"{dir(sug)} was not found among the provided options. Did you possibly mean {sug[0]} instead?",
            "are_in_assert",
        )
    else:
        format_assert(
            [
                "None of the following were found among the provided options. Did you possibly mean:",
                bullet(chr_ply(sug, lambda x: "'" + x + "'"), f" instead of '{dir(sug)}'"),
            ],
            "are_in_assert",
            exdent=[0] + [len(sug)] * 2,
        )


are_in.on_failure = in_failure


def is_in(x, opts, na_rm=False):
    assert_that(isinstance(x, str)) and are_in(x, opts, na_rm)


is_in.on_failure = in_failure


def has_col(x, col):
    return has_cols(x, col, 1)

def is_unique(values):
    unique_values = np.unique(values)
    return len(values) == len(unique_values)

def setdiff(a, b):
    set_a = set(a)
    set_b = set(b)
    diff = set_a.difference(set_b)
    return list(diff)

def has_cols(x, cols, length=math.nan):
    if math.isnan(length):
        len_check = assert_that(has_length(cols))
    else:
        len_check = assert_that(is_count(length), all_equal(len(cols), length))
    assert_that(isinstance(cols, str), is_unique(cols)) and len_check and len(setdiff(cols, x.columns)) == 0

'''
has_cols.on_failure = lambda call, env: format_assert(
    f"{str(call['x'])} does not contain {qty(len(cols))} column{'s' if len(cols) > 1 else ''} {quote_bt(cols)}",
    "has_cols_assert",
)
'''


def has_interval(x, interval):
    pass
    #assert_that(TableAccessor.is_ts_tbl(x), is_interval(interval)) and same_time(interval(x), interval)

'''
has_interval.on_failure = lambda call, env: format_assert(
    f"{str(call['x'])} is not on the time scale of {format(ival)}", "has_interval_assert"
)
'''

def is_difftime(obj):
    return isinstance(obj, datetime.timedelta)

def is_interval(x, length=math.nan):
    assert_that(is_difftime(x), has_length(x, length)) and all(x >= 0 | np.isnan(x))


is_interval.on_failure = lambda call, env: format_assert(
    f"Not all of {str(call['x'])} represent positive time intervals", "is_interval_assert"
)

def has_time_cols(x, cols, length=math.nan):
    assert_that(has_cols(x, cols, length)) # Bug in col_ply multiprocessing, so commented that out for now
   # assert_that(has_cols(x, cols, length)) and all(col_ply(x, cols, is_difftime))

'''
has_time_cols.on_failure = lambda call, env: format_assert(
    f"{qty(len(cols))} Column{'s' if len(cols) > 1 else ''} {quote_bt(cols)} of {str(call['x'])} "
    f"{qty(len(cols))} {'does' if len(cols) == 1 else 'do'} not represent time intervals",
    "has_time_cols_assert",
)
'''


def obeys_interval(x, interval, na_rm=True, tolerance=seconds(1e-3)):
    assert_that(
        is_difftime(x),
        is_scalar(interval),
        is_interval(interval),
        is_scalar(tolerance),
        is_interval(tolerance),
    ) and (np.isnan(interval) or all(x % interval < tolerance | np.isnan(x)))

'''
obeys_interval.on_failure = lambda call, env: format_assert(
    f"{str(call['x'])} is not compatible with an interval of {format(ival)}", "obeys_interval_assert"
)
'''

def same_unit(x, y):
    ureg = pint.UnitRegistry()
    unit_x = ureg.get_dimensionality(x.units)
    unit_y = ureg.get_dimensionality(y.units)
    return unit_x == unit_y


same_unit.on_failure = lambda call, env: format_assert(
    f"{str(call['x'])} and {str(call['y'])} are not measured in the same unit", "same_unit_assert"
)


def same_time(x, y, tolerance=seconds(1e-3)):
    assert_that(same_unit(x, y)) and all(abs(x - y) < tolerance)


same_time.on_failure = lambda call, env: format_assert(
    f"{str(call['x'])} and {str(call['y'])} are not on the same time scale", "same_time_assert"
)

def all_fun(x, fun, na_rm=False, **kwargs):
    assert callable(fun), "fun must be a function"
    return all([fun(item, **kwargs) for item in x if (item is not None) or na_rm])


all_fun.on_failure = lambda call, env: format_assert(
    f"some of {str(call['x'])} do not satisfy function `{str(call['fun'])}`", "all_fun_assert"
)

def all_map(fun, *args, **kwargs):
    assert callable(fun), "fun must be a function"
    result = map(fun, *args, **kwargs)
    return all(map(bool, result))


all_map.on_failure = lambda call, env: format_assert(
    f"some invocations of `{str(call['fun'])}` do not evaluate to `TRUE`", "all_map_assert"
)


def all_null(x):
    return all_fun(x, lambda item: item is None)


all_null.on_failure = lambda call, env: format_assert(
    f"some of {str(call['x'])} are not `NULL`", "all_null_assert"
)


def same_length(x, y):
    return len(x) == len(y)


same_length.on_failure = lambda call, env: format_assert(
    f"{str(call['x'])} does not have the same length as {str(call['y'])}", "same_length_assert"
)


def is_disjoint(x, y):
    return len(set(x).intersection(y)) == 0


is_disjoint.on_failure = lambda call, env: format_assert(
    f"{str(call['x'])} and {str(call['y'])} have a nonempty intersection", "is_disjoint_assert"
)


def not_null(x):
    return x is not None


not_null.on_failure = lambda call, env: format_assert(f"{str(call['x'])} is NULL", "not_null_assert")


def null_or(x, what, **kwargs):
    return (x is None) or what(x, **kwargs)


null_or.on_failure = lambda call, env: format_assert(
    f"{str(call['x'])} is neither NULL, nor {str(call['what'])}", "null_or_assert"
)


def evals_to_fun(x):
    if callable(x):
        return True
    elif isinstance(x, str):
        try:
            eval_result = eval(x)
            if callable(eval_result):
                return True
        except Exception:
            pass
    return False


evals_to_fun.on_failure = lambda call, env: format_assert(
    f"{str(call['x'])} is neither a function nor evaluates to a function", "evals_fun_assert"
)


def all_equal(x, y, **kwargs):
    return np.array_equal(x, y, **kwargs)


all_equal.on_failure = lambda call, env: format_assert(
    f"{str(call['x'])} and {str(call['y'])} are not equal", "all_equal_assert"
)
