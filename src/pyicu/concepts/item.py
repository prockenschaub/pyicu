from typing import Dict

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

    def _try_add_vars(self, var_dict: Dict[str, str], type: str = "data_vars") -> None:
        vars = getattr(self, type)
        for k, v in var_dict.items():
            if k not in vars and v is not None:
                vars[k] = v

    def load(self, src: Src, interval=None) -> pyICUTbl:
        raise NotImplementedError()


class SelItem(Item):
    def __init__(self, src, table, sub_var, ids, callback=None, **kwargs) -> None:
        super().__init__(src, table, sub_var=sub_var, callback=callback, **kwargs)
        self.ids = ids

    def load(self, src: Src, target=None, interval=None) -> pyICUTbl:
        # TODO: somehow dynamically add unit_var for num_cncpts
        self._try_add_vars({'val_var': src[self.tbl].defaults.get('val_var')})
        return src.load_sel(self.tbl, self.data_vars['sub_var'], self.ids, cols=list(self.data_vars.values()), target=target, interval=interval)

    def __repr__(self) -> str:
        return f"<SelItem:{self.src}> {self.tbl}.{self.data_vars['sub_var']} in {print_list(enlist(self.ids))}"


class RgxItem(Item):
    def __init__(self, src, table, sub_var, regex, callback=None, **kwargs) -> None:
        super().__init__(src, table, sub_var=sub_var, callback=callback, **kwargs)
        self.regex = regex

    def load(self, src: Src, target=None, interval=None) -> pyICUTbl:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"<RgxItem:{self.src}> {self.tbl}.{self.data_vars['sub_var']} like {self.regex}"


class ColItem(Item):
    def __init__(self, src, table, val_var, unit_val=None, callback=None, **kwargs) -> None:
        super().__init__(src, table, val_var=val_var, unit_val=unit_val, callback=callback, **kwargs)
        
    def load(self, src: Src, target=None, interval=None) -> pyICUTbl:
        # TODO: somehow dynamically add unit_var for num_cncpts
        self._try_add_vars({'val_var': src[self.tbl].defaults.get('val_var')})
        return src.load_col(self.tbl, self.data_vars['val_var'], self.data_vars.get('unit_var'), target=target, interval=interval)
    
    def __repr__(self) -> str:
        return f"<ColItem: {self.src}> {self.tbl}.{self.data_vars['val_var']}"


class FunItem(Item):
    def __init__(self, src, table=None, win_type=None, callback=None, **kwargs) -> None:
        super().__init__(src, table, callback=callback, **kwargs)
        self.win_type = win_type
        
    def load(self, src: Src, target=None, interval=None) -> pyICUTbl:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"<FunItem: {self.src}> {self.callback.__name__}({self.tbl or '?'})"
