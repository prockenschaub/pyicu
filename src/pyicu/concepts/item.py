from ..utils import coalesce, enlist, print_list
from ..data.source import Src
from ..tables import pyICUTbl


class Item():
    def __init__(
        self, 
        src, 
        table, 
        id_var=None,
        index_var=None,
        dur_var=None, 
        callback=None,
        **kwargs
    ) -> None:
        self.src = src
        self.tbl = table
        self.callback = callback
        self.data_vars = kwargs
        self.meta_vars = coalesce(id_var=id_var, index_var=index_var, dur_var=dur_var)

    def load(self, src: Src, interval=None) -> pyICUTbl:
        pass


class SelItem(Item):
    def __init__(self, src, table, sub_var, ids, callback=None, **kwargs) -> None:
        super().__init__(src, table, sub_var=sub_var, callback=callback, **kwargs)
        self.ids = ids

    def __repr__(self) -> str:
        return f"<SelItem:{self.src}> {self.tbl}.{self.data_vars['sub_var']} in {print_list(enlist(self.ids))}"


class RgxItem(Item):
    def __init__(self, src, table, sub_var, regex, callback=None, **kwargs) -> None:
        super().__init__(src, table, sub_var=sub_var, callback=callback, **kwargs)
        self.regex = regex

    def __repr__(self) -> str:
        return f"<SgxItem:{self.src}> {self.tbl}.{self.data_vars['sub_var']} like {self.regex}"


class ColItem(Item):
    def __init__(self, src, table, val_var, unit_val=None, callback=None, **kwargs) -> None:
        super().__init__(src, table, val_var=val_var, unit_val=unit_val, callback=callback, **kwargs)
        
    def __repr__(self) -> str:
        return f"<ColItem: {self.src}> {self.tbl}.{self.data_vars['val_var']}"


class FunItem(Item):
    def __init__(self, src, table=None, win_type=None, callback=None, **kwargs) -> None:
        super().__init__(src, table, callback=callback, **kwargs)
        self.win_type = win_type
        
    def __repr__(self) -> str:
        return f"<FunItem: {self.src}> {self.callback.__name__}({self.tbl or '?'})"


ITEM_MAP = {
    "sel_itm": SelItem,
    "rgx_itm": RgxItem,
    "col_itm": ColItem,
    "fun_itm": FunItem
}