from abc import abstractmethod
from typing import List, Dict

import pandas as pd

from ..interval import hours
from ..utils import coalesce, enlist, print_list
from ..sources import Src
from .utils import str_to_fun


class Item:
    """Item objects are used in pyicu as a way to specify how individual data items corresponding to
    clinical concepts (see also concept()), such as heart rate can be loaded from a data source.

    Sub-classes have been defined, each representing a different data-scenario and holding
    further class-specific information. The following sub-classes to `Item` are available:

        `SelItem`: The most widely used item class is intended for the situation where rows of interest can
            be identified by looking for occurrences of a set of IDs (ids) in a column (sub_var). An
            example for this is heart rate hr on mimic, where the IDs 211 and 220045 are looked up in the
            itemid column of chartevents.

        `RgxItem`: As alternative to the value-matching approach of `SelItem` objects, this class
            identifies rows using regular expressions. Used for example for insulin in eicu, where the
            regular expression `^insulin (250.+)?\\(((ml|units)/hr)?\\)$` is matched against the drugname
            column of infusiondrug.

        `ColItem`: This item class can be used if no row-subsetting is required. An example for this is
            heart rate (hr) on eicu, where the table vitalperiodic contains an entire column dedicated to
            heart rate measurements.

        `FunItem`: Intended for the scenario where data of interest is not directly available from a table,
            this itm class offers most flexibility. A function can be specified as callback and this
            function will be called with arguments x (the object itself), patient_ids, id_type and
            interval (see load_concepts()) and is expected to return an object as specified by the target
            entry. TODO: update this doc when `FunItem` is fully implemented.

    Args:
        src: name of the data source for which this item is defined
        table: name of the source table from which to retrieve data
        id_var: name of the observation ID, e.g., stay_id of table chartevents in source MIMIC IV. Defaults
            to None, in which case this is determined at runtime based on defaults set for the data source
            and source table.
        index_var: name of the time index, e.g., chartttime of table chartevents in source MIMIC IV.
            Defaults to None, in which case this is determined at runtime (if applicable) based on defaults
            set for the data source and source table.
        dur_var: name of the duration time, e.g., duration of table drugitems in source AUMC. Defaults to
            None, in which case this is determined at runtime (if applicable) based on defaults set for the
            data source and source table.
        callback: name of a function to be called on the returned data used for data cleanup operations.
        interval: a default data loading interval
    """

    def __init__(
        self,
        src: str,
        table: str,
        id_var: str | None = None,
        index_var: str | None = None,
        dur_var: str | None = None,
        callback: str | None = None,
        interval: pd.Timedelta | None = None,
        **kwargs,
    ) -> None:
        self.src = src
        self.tbl = table
        self.callback = callback
        self.target = kwargs.pop("target", None)
        self.data_vars = kwargs
        self.meta_vars = coalesce(id_var=id_var, index_var=index_var, dur_var=dur_var)

    def _choose_id(self, src: Src, id_type: str) -> None:
        opts = src.id_cfg.cfg
        if id_type is None:
            id_var = opts.loc[opts.index.max(), "id"].iloc[0]
        else:
            id_var = opts.loc[opts.name == id_type, "id"].iloc[0]
        self._try_add_vars({"id_var": id_var}, type="meta_vars")

    def _try_add_vars(self, var_dict: Dict[str, str], type: str = "data_vars") -> None:
        """Add one or more variables to `data_vars` or `meta_vars` if they haven't been set yet.

        Args:
            var_dict: names and values to be set
            type: variable set. Defaults to "data_vars".
        """
        vars = getattr(self, type)
        for k, v in var_dict.items():
            if k not in vars and v is not None:
                vars[k] = v

    @abstractmethod
    def load(self, src: Src, id_type: str = "icustay", target: str = None, interval: pd.Timedelta = hours(1)) -> pd.DataFrame:
        """Load item data from a data source at a given time interval

        Args:
            src: data source, e.g., MIMIC IV.
            id_type: patient id type to return, e.g., "icustay". Defaults to "icustay".
            target: a target class specification, e.g., "ts_tbl". Defaults to "ts_tbl".
            interval: the time interval used to discretize time stamps with. Defaults to 1 hour.

        Raises:
            NotImplementedError: this method needs to be overridden by the individual `Item` classes.

        Returns:
            loaded data
        """
        raise NotImplementedError()

    def do_callback(self, src: Src, res: pd.DataFrame) -> pd.DataFrame:
        fun = str_to_fun(self.callback)
        res = fun(res, **self.meta_vars, **self.data_vars, env=src)  # TODO: add kwargs
        return res

    def standardise_cols(self, src: Src, res: pd.DataFrame) -> pd.DataFrame:
        res = res.rename(columns={v: k for k, v in self.data_vars.items()})
        map_dict = src.id_cfg.map_id_to_type()
        res = res.tbl.rename_all(map_dict)
        if res.tbl.is_ts_tbl():
            res = res.tbl.rename_all({res.tbl.index_var: "time"})
        return res


class SelItem(Item):
    """Select rows of interest by looking for occurrences of a set of IDs

    See also: `Item`

    Args:
        src: name of the data source for which this item is defined
        table: name of the source table from which to retrieve data
        sub_var: column name used for subsetting
        ids: list of ids used to subset table rows
        callback: name of a function to be called on the returned data used for data cleanup operations.
    """

    def __init__(
        self, src: str, table: str, sub_var: str, ids: List[int | str], callback: str | None = None, **kwargs
    ) -> None:
        super().__init__(src, table, sub_var=sub_var, callback=callback, **kwargs)
        self.ids = ids

    def load(
        self, src: Src, id_type: str = "icustay", target: str = None, interval: pd.Timedelta = hours(1), **kwargs
    ) -> pd.DataFrame:
        """Load item data from a data source at a given time interval

        See also: `Item.load()`
        """
        self._choose_id(src, id_type)
        self._try_add_vars({k: v for k, v in src[self.tbl].defaults.items() if k in ["index_var"]}, type="meta_vars")
        self._try_add_vars({k: v for k, v in src[self.tbl].defaults.items() if k in ["val_var", "unit_var"]}, type="data_vars")
        if self.target is not None: 
            target = self.target
        res = src.load_sel(
            self.tbl,
            self.data_vars["sub_var"],
            self.ids,
            cols=list(self.data_vars.values()),
            target=target,
            interval=interval,
            **self.meta_vars,
        )
        res = self.do_callback(src, res)
        res = self.standardise_cols(src, res)
        res.drop(columns="sub_var", inplace=True)
        return res

    def __repr__(self) -> str:
        return f"<SelItem:{self.src}> {self.tbl}.{self.data_vars['sub_var']} in {print_list(enlist(self.ids))}"


class RgxItem(Item):
    """Select rows of interest by matching a regular expression to IDs

    See also: `Item`

    Args:
        src: name of the data source for which this item is defined
        table: name of the source table from which to retrieve data
        sub_var: column name used for subsetting
        regex: regular expression determining which rows to select
        callback: name of a function to be called on the returned data used for data cleanup operations.
    """

    def __init__(self, src: str, table: str, sub_var: str, regex: str, callback: str | None = None, **kwargs) -> None:
        super().__init__(src, table, sub_var=sub_var, callback=callback, **kwargs)
        self.regex = regex

    def load(
        self, src: Src, id_type: str = "icustay", target: str = None, interval: pd.Timedelta = hours(1), **kwargs
    ) -> pd.DataFrame:
        """Load item data from a data source at a given time interval

        See also: `Item.load()`
        """
        self._choose_id(src, id_type)
        self._try_add_vars({k: v for k, v in src[self.tbl].defaults.items() if k in ["index_var"]}, type="meta_vars")
        self._try_add_vars({k: v for k, v in src[self.tbl].defaults.items() if k in ["val_var", "unit_var"]}, type="data_vars")
        if self.target is not None: 
            target = self.target
        res = src.load_rgx(
            self.tbl,
            self.data_vars["sub_var"],
            self.regex,
            cols=list(self.data_vars.values()),
            target=target,
            interval=interval,
            **self.meta_vars,
        )
        res = self.do_callback(src, res)
        res = self.standardise_cols(src, res)
        res.drop(columns="sub_var", inplace=True)
        return res

    def __repr__(self) -> str:
        return f"<RgxItem:{self.src}> {self.tbl}.{self.data_vars['sub_var']} like {self.regex}"


class ColItem(Item):
    """Select an entire column

    See also: `Item`

    Args:
        src: name of the data source for which this item is defined
        table: name of the source table from which to retrieve data
        val_var: column name containing the measurement values
        unit_val: string valued unit to be used in case no default unit_var is available for the given
            table. Defaults to None.
        callback: name of a function to be called on the returned data used for data cleanup operations.
    """

    def __init__(
        self, src: str, table: str, val_var: str, unit_val: str | None = None, callback: str | None = None, **kwargs
    ) -> None:
        super().__init__(src, table, val_var=val_var, callback=callback, **kwargs)
        self.unit_val = unit_val

    def load(
        self, src: Src, id_type: str = "icustay", target: str = None, interval: pd.Timedelta = hours(1), **kwargs
    ) -> pd.DataFrame:
        """Load item data from a data source at a given time interval

        See also: `Item.load()`
        """
        self._choose_id(src, id_type)
        self._try_add_vars({k: v for k, v in src[self.tbl].defaults.items() if k in ["index_var"]}, type="meta_vars")
        self._try_add_vars({k: v for k, v in src[self.tbl].defaults.items() if k in ["val_var", "unit_var"]}, type="data_vars")
        if self.target is not None: 
            target = self.target
        res = src.load_col(
            self.tbl,
            self.data_vars["val_var"],
            cols=list(self.data_vars.values()),
            target=target,
            interval=interval,
            **self.meta_vars,
        )
        res = self.do_callback(src, res)
        res = self.standardise_cols(src, res)
        if self.unit_val is not None:
            res["unit_var"] = self.unit_val
        return res

    def __repr__(self) -> str:
        return f"<ColItem: {self.src}> {self.tbl}.{self.data_vars['val_var']}"


class FunItem(Item):
    """TBD"""

    def __init__(
        self, src: str, table: str = None, win_type: str | None = None, callback: str | None = None, **kwargs
    ) -> None:
        super().__init__(src, table, callback=callback, **kwargs)
        self.win_type = win_type

    def load(self, src: Src, id_type: str = "icustay", interval=None, **kwargs) -> pd.DataFrame:
        """Load item data from a data source at a given time interval

        See also: `Item.load()`
        """
        fun = str_to_fun(self.callback)
        res = fun(src=src, itm=self, id_type=id_type, interval=interval)
        res = self.standardise_cols(src, res)
        return res

    def __repr__(self) -> str:
        return f"<FunItem: {self.src}> {self.callback.__name__}({self.tbl or '?'})"


def item_class(x: str) -> Item:
    """Map string to item class

    Args:
        x: string specification of the item as defined by ricu

    Returns:
        item class
    """
    match x:
        case "sel_itm":
            return SelItem
        case "rgx_itm":
            return RgxItem
        case "col_itm":
            return ColItem
        case "fun_itm":
            return FunItem
        case _:
            return SelItem
