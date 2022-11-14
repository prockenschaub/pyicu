import warnings
from itertools import chain
from typing import Type, List, Tuple, Dict
from pathlib import Path

import pyarrow.csv as pv
import pyarrow.parquet as pq
import pyarrow.compute as pc

import numpy as np

from .utils import parse_col_types
from ..utils import enlist


DEFAULTS = ["id_var", "index_var", "val_var", "unit_var", "time_vars"]


class TblCfg:
    def __init__(
        self,
        name: str,
        files: str = None,
        cols: List[Type["ColumnSpec"]] = None,
        defaults: dict = None,
        num_rows: int = None,
        partitioning: Dict = None,
        **kwargs,
    ):
        self.name = name
        self.files = files
        self.cols = cols
        self.num_rows = num_rows

        if partitioning is not None:
            if not hasattr(partitioning, "col") and hasattr(partitioning, "breaks"):
                raise ValueError(f"partition definition for {name} must contain 'col' and 'breaks'.")
            partitioning["breaks"] = enlist(partitioning["breaks"])
        self.partitioning = partitioning

        self._set_defaults(defaults)

    def from_dict(x: Dict, name: str) -> Type["TblCfg"]:
        """Create ColumnSpec from dictionary (e.g., read through JSON)

        Args:
            x (Dict): cfg defining the table.
            name (str): name of the table

        Example:
            {
                'files': 'admissiondrug.csv.gz',
                'cols': {
                    'patientunitstayid': {
                        'name': 'patientunitstayid',
                        'spec': 'col_integer'
                    }
                }
            }

        Returns:
            TblCfg: configuration created from dictionary
        """
        x = x.copy()

        if "cols" in x.keys():
            cols = x.pop("cols")
            cols = [ColumnSpec.from_tuple(i) for i in cols.items()]
        else:
            cols = None

        return TblCfg(name, cols=cols, **x)

    def from_tuple(x: Tuple[str, Dict]) -> Type["TblCfg"]:
        """Create ColumnSpec from tuple

        Args:
            x (Tuple[str, Dict]): name/cfg pair defining the table.

        Example:
            (
                'admissiondrug',
                {
                    'files': 'admissiondrug.csv.gz',
                    'cols': {
                        'patientunitstayid': {
                            'name': 'patientunitstayid',
                            'spec': 'col_integer'
                        }
                    }
                }
            )

        Returns:
            TableCfg: configuration created from tuple
        """
        name, cfg = x
        return TblCfg.from_dict(cfg, name)

    def _set_defaults(self, defaults: dict):
        defs = set(defaults.keys())
        cols = set(chain(*[v if isinstance(v, list) else [v] for v in defaults.values()]))

        if len(defs.difference(DEFAULTS)) > 0:
            raise ValueError(
                f"tried to set unknown defaults {defs.difference(DEFAULTS)} for table {self.name}."
                f"Must be one of {DEFAULTS}."
            )
        if len(cols.difference([c.name for c in self.cols])) > 0:
            raise ValueError(f"tried to set unknown columns {cols.difference(self.cols)} as defaults for table {self.name}.")

        self.defaults = defaults

    def _check_raw_files_exist(self, data_dir: Path) -> bool:
        # TODO: check if this works with > 1 file
        files = self.files
        if isinstance(self.files, str):
            files = [files]
        for f in files:
            if not (data_dir / self.files).exists():
                raise FileNotFoundError(f"Source file {data_dir/f} of table {self.name} not found.")

    def is_imported(self, data_dir: Path) -> bool:
        # TODO: currently does not check for the correct number of rows
        if self.partitioning is None:
            return (data_dir / f"{self.name}.parquet").exists()
        else:
            folder = data_dir / f"{self.name}"
            parts = [f.stem for f in folder.glob("*.parquet")]
            # TODO: check for the filenames of the parts
            return folder.exists() and len(parts) == (len(self.partitioning["breaks"]) + 1)

    def do_import(self, data_dir: Path, out_dir: Path = None, progress: bool = None, cleanup: bool = False, **kwargs):
        # TODO: implement cleanup
        # TODO: implement progress bar
        # TODO: account for setups like HiRID with ZIP folders
        self._check_raw_files_exist(data_dir)

        if out_dir is None:
            out_dir = data_dir

        col_types = {c.col: c.spec for c in self.cols}  # TODO: rename to PYICU column names
        tbl = pv.read_csv(data_dir / self.files, convert_options=pv.ConvertOptions(column_types=col_types))

        if self.num_rows is not None and tbl.num_rows != self.num_rows:
            raise warnings.warn(
                f"expected {self.num_rows} rows but got {tbl.num_rows} rows for table`{self.name}`",
            )

        if self.partitioning:
            self._write_partitions(tbl, out_dir)
        else:
            self._write_single_file(tbl, out_dir, self.name)

    def _write_single_file(self, tbl, out_dir, file_name):
        pq.write_table(tbl, out_dir / f"{file_name}.parquet")

    def _write_partitions(self, tbl, out_dir):
        part_dir = out_dir / f"{self.name}"
        part_dir.mkdir(parents=True, exist_ok=True)

        col = tbl[self.partitioning["col"]]
        breaks = [-np.inf] + self.partitioning["breaks"] + [np.inf]

        for part in range(len(breaks) - 1):
            lower = pc.greater_equal(col, breaks[part])
            upper = pc.less(col, breaks[part + 1])
            sub_tbl = tbl.filter(pc.and_(lower, upper))
            self._write_single_file(sub_tbl, part_dir, f"{part}")

    def __repr__(self) -> str:
        info = ""
        if self.num_rows is not None:
            info += f"rows: {self.num_rows}"
        if self.cols is not None:
            if info != "":
                info += ", "
            info += f"cols: {len(self.cols)}"

        return f"<TableCfg: {self.name} ({info})>"


class ColumnSpec:
    """Specification of a single column of a TableCfg

    Args:
        name (str): column name as used by pyicu
        col (str): column name as used in the raw data
        spec (str):  name of readr::cols() column specification, for compatibility
            with R. Mapped to python equivalents.
    """

    def __init__(self, name: str, col: str, spec: str, **kwargs):
        self.name = name
        self.col = col
        self.spec = parse_col_types(spec)

    def from_dict(x: Dict, col: str) -> Type["ColumnSpec"]:
        """Create ColumnSpec from tuple

        Args:
            x (Tuple[str, dict]): col/spec pair specifying the column

        Example:
            {
                'name': 'patientunitstayid',
                'spec': 'col_integer'
            }

        Raises:
            ValueError: if "name" is not among the keys of x
            ValueError: if "spec" is not among the keys of x

        Returns:
            ColumnSpec: specification created from tuple
        """
        x = x.copy()

        if not "name" in x.keys():
            raise ValueError(f"no `name` attribute provided for column config {col}")
        name = x.pop("name")

        if not "spec" in x.keys():
            raise ValueError(f"no `spec` attribute provided for column config {col}")
        spec = x.pop("spec")

        return ColumnSpec(name, col, spec, **x)

    def from_tuple(x: Tuple[str, Dict]) -> Type["ColumnSpec"]:
        """Create ColumnSpec from tuple

        Args:
            x (Tuple[str, dict]): col/spec pair specifying the column

        Example:
            (
                'patientunitstayid',
                {
                    'name': 'patientunitstayid',
                    'spec': 'col_integer'
                }
            )

        Raises:
            ValueError: if "name" is not among the keys of x
            ValueError: if "spec" is not among the keys of x

        Returns:
            ColumnSpec: specification created from tuple
        """
        col, spec = x
        return ColumnSpec.from_dict(spec, col)

    def __repr__(self) -> str:
        return f"<ColumnSpec: {self.name} ({self.spec})>"
