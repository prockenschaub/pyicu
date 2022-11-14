import warnings
from typing import Type, Dict, Union, List
from pathlib import Path

from .id import IdCfg
from .table import TblCfg
from .utils import check_attributes_in_dict
from ..utils import enlist


class SrcCfg:
    def __init__(self, name, ids: IdCfg, tbls: TblCfg, **kwargs) -> None:
        self.name = name
        self.ids = ids
        self.tbls = tbls

    def from_dict(x: Dict) -> Type["SrcCfg"]:
        """Create a source configuration from dict (e.g., read through JSON)

        Raises:
            ValueError:  if "name" is not among the keys of x
            ValueError:  if "tables" is not among the keys of x

        Returns:
            SrcCfg: configuration created from dictionary
        """
        check_attributes_in_dict(x, "name", "unnamed", "source")
        name = x["name"]

        check_attributes_in_dict(x, ["id_cfg", "tables"], name, "source")
        ids = IdCfg.from_dict(x["id_cfg"])
        tbls = [TblCfg.from_tuple(t) for t in x["tables"].items()]

        return SrcCfg(name, ids, tbls)

    def do_import(
        self,
        data_dir: Path,
        out_dir: Path = None,
        tables: Union[str, List[str]] = None,
        force: bool = False,
        verbose: bool = True,
        cleanup: bool = False,
        **kwargs,
    ):
        # TODO: implement force
        if out_dir is None:
            out_dir = data_dir
        if tables is None:
            tables = [t.name for t in self.tbls]
        elif isinstance(tables, str):
            tables = enlist(tables)

        done = [t.name for t in self.tbls if t.is_imported(out_dir)]
        skip = set(done).intersection(tables)
        todo = set(tables).difference(done)

        if verbose and not force and len(skip) > 0:
            print(f"The following tables have already been imported and will be skipped: {skip}")

        if len(todo) == 0:
            warnings.warn(f"All required tables have already been imported for source `{self.name}`.")
            return None

        for tbl in self.tbls:
            if tbl.name in todo:
                tbl.do_import(data_dir, out_dir)

        if verbose:
            print(f"Successfully imported {len(todo)} tables.")

    def __repr__(self) -> str:
        return f"<SrcCfg>: {self.name} (tables: {len(self.tbls)})>"
