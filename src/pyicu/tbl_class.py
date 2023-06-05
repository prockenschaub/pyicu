from pyicu.utils_cli import stop_ricu, stop_generic
from pyicu.tbl_utils import id_vars, index_var, dur_var, dur_col, meta_vars, index_col
from pyicu.assertions import assert_that, is_unique, is_disjoint, is_difftime, has_cols, obeys_interval, same_unit
from pyicu.container.table import TableAccessor
import pandas as pd

def reclass_tbl(x, template, stop_on_fail=True):
    template = template

    return reclass_tbl_impl(x, template, stop_on_fail)

def reclass_tbl_impl(x, template, stop_on_fail=True):
    template_class = template.__class__.__name__

    if template_class == "NULL":
        return x
    elif template_class == "id_tbl":
        x = reclass_tbl_impl(template, template, stop_on_fail)
        x = set_attributes(x, id_vars=id_vars(template), _class=template_class)
        check_valid(x, stop_on_fail)
        return x
    elif template_class == "ts_tbl":
        x = reclass_tbl_impl(template, template, stop_on_fail)
        x = set_attributes(x, index_var=index_var(template), interval=TableAccessor.interval(template), _class=template_class)
        if validate_tbl(x):
            return x
        return reclass_tbl_impl(template, template, stop_on_fail)
    elif template_class == "win_tbl":
        x = reclass_tbl_impl(template, template, stop_on_fail)
        x = set_attributes(x, dur_var=dur_var(template), _class=template_class)
        if validate_tbl(x):
            return x
        return reclass_tbl_impl(template, template, stop_on_fail)
    elif template_class == "data.table":
        return check_valid(unclass_tbl(x), stop_on_fail)
    elif template_class == "data.frame":
        return check_valid(pd.DataFrame(unclass_tbl(x)), stop_on_fail)
    else:
        return stop_generic(template, "reclass_tbl")

def try_reclass(x, template):
    if isinstance(x, pd.DataFrame):
        return reclass_tbl(x, template, stop_on_fail=False)
    else:
        return x

def as_ptype(x):
    return as_ptype_impl(x)

def as_ptype_impl(x):
    x_class = x.__class__.__name__

    if x_class == "id_tbl":
        return reclass_tbl(pd.DataFrame(map(x, lambda col: col[0])[meta_vars(x)]), x)
    elif x_class == "data.table":
        return pd.DataFrame()
    elif x_class == "data.frame":
        return pd.DataFrame()
    else:
        return stop_generic(x, "as_ptype")
    
def validate_that(*assertions):
    for assertion in assertions:
        if not assertion:
            return False
    return True

def validate_tbl(x):
    res = validate_that(
        isinstance(x, pd.DataFrame), is_unique(list(x.columns))
    )
    
    if not res:
        return res
    
    validate_tbl.dispatch(type(x))(x)

def validate_tbl_id_tbl(x):
    idv = id_vars(x)
    res = validate_that(
        has_cols(x, idv), is_unique(idv)
    )
    
    if res:
        validate_tbl.dispatch(type(x))(x)
    else:
        return res

def validate_tbl_ts_tbl(x):
    index = index_col(x)
    inval = TableAccessor.interval(x)
    invar = index_var(x)
    
    res = validate_that(
        isinstance(invar, str), has_cols(x, invar), is_disjoint(id_vars(x), invar),
        obeys_interval(index, inval), same_unit(index, inval)
    )
    
    if res:
        validate_tbl.dispatch(type(x))(x)
    else:
        return res

def validate_tbl_win_tbl(x):
    dvar = dur_var(x)
    
    res = validate_that(
        isinstance(dvar, str), has_cols(x, dvar), is_disjoint(id_vars(x), dvar),
        is_disjoint(dvar, index_var(x)), is_difftime(dur_col(x))
    )
    
    if res: # Changed from is_true()
        validate_tbl.dispatch(type(x))(x)
    else:
        return res

def validate_tbl_data_frame(x):
    return True

def validate_tbl_default(x):
    stop_generic(x, ".Generic")

def check_valid(x, stop_on_fail=True):
    res = validate_tbl(x)
    
    if res:
        return x
    elif stop_on_fail:
        stop_ricu(res, class_=["valid_check_fail", getattr(res, "assert_class")])
    else:
        return unclass_tbl(x)
    
def unclass_tbl(x):
    return unclass_tbl.dispatch(x)

def unclass_tbl_data_frame(x):
    return x

def strip_class(x):
    if hasattr(x, '__dict__'):
        x.__dict__.clear()
    return x

def unclass_tbl_win_tbl(x):
    return unclass_tbl(set_attributes(x, dur_var=None, class_=strip_class(x)))

def unclass_tbl_ts_tbl(x):
    return unclass_tbl(
        set_attributes(x, index_var=None, interval=None, class_=strip_class(x))
    )

def unclass_tbl_id_tbl(x):
    return set_attributes(x, id_vars=None, class_=strip_class(x))

def unclass_tbl_default(x):
    stop_generic(x, ".Generic")

unclass_tbl.dispatch = {
    "data.frame": unclass_tbl_data_frame,
    "win_tbl": unclass_tbl_win_tbl,
    "ts_tbl": unclass_tbl_ts_tbl,
    "id_tbl": unclass_tbl_id_tbl,
    "default": unclass_tbl_default,
}

def set_attributes(x, **kwargs):
    dot = kwargs
    nms = list(dot.keys())

    assert_that(isinstance(dot, dict), len(dot) > 0, nms is not None, len(nms) == len(set(nms)))

    for key, value in dot.items():
        setattr(x, key, value)

    return x


