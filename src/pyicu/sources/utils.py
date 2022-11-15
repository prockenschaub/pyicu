from typing import List
import re
import pandas as pd


def defaults_to_str(defaults):
    repr = ""
    for d, v in list(defaults.items()):
        if d != "time_vars":
            if repr != "":
                repr += ", "
            repr += f"`{v}` ({re.sub('_vars?', '', d)})"
    return repr


def time_vars_to_str(defaults):
    repr = ""
    time_vars = defaults["time_vars"]
    if isinstance(time_vars, str):
        time_vars = [time_vars]

    for v in time_vars:
        if repr != "":
            repr += ", "
        repr += f"`{v}`"
    return repr


def order_rename(df: pd.DataFrame, id_var: List[str], st_var: List[str], ed_var: List[str]):
    def add_suffix(x: List[str], s: str):
        return [f"{i}_{s}" for i in x]

    old_names = id_var + st_var + ed_var
    new_names = id_var + add_suffix(id_var, "start") + add_suffix(id_var, "end")
    df = df[old_names]  # Reorder
    df = df.rename({o: n for o, n in zip(old_names, new_names)}, axis="columns")
    return df
