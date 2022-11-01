import warnings
from typing import Type, List, Tuple, Dict
from pathlib import Path

import pyarrow.csv as pv
import pyarrow.parquet as pq

from .utils import parse_col_types

class TblCfg():
    def __init__(
        self, 
        name: str, 
        files: str = None, 
        cols: List[Type["ColumnSpec"]] = None, 
        num_rows: int = None,
        partitioning: Dict = None,
        **kwargs
    ):
        self.name = name
        self.files = files
        self.cols = cols
        self.num_rows = num_rows
        self.partitioning = partitioning


    def from_dict(x: Dict, name: str) -> Type['TblCfg']:
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

        if 'cols' in x.keys():
            cols = x.pop('cols')
            cols = [ColumnSpec.from_tuple(i) for i in cols.items()]
        else:
            cols = None

        return TblCfg(name, cols=cols, **x)


    def from_tuple(x: Tuple[str, Dict]) -> Type['TblCfg']:
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


    def raw_files_exist(self, data_dir: Path) -> bool:
        # TODO: check if this works with > 1 file
        return (data_dir/self.files).exists()
    

    def imp_files_exist(self, data_dir: Path) -> bool:
        return (data_dir/f'{self.name}.parquet').exists()


    def do_import(self, data_dir: Path, out_dir: Path = None, progress: bool = None, cleanup: bool = False, **kwargs):
        # TODO: implement cleanup
        # TODO: implement progress bar
        if not self.raw_files_exist(data_dir):
            raise FileNotFoundError(f'Source file {data_dir/self.files} not found during import of table {self.name}.')
        if out_dir is None:
            out_dir = data_dir

        col_types = {c.col: c.spec for c in self.cols}  # TODO: rename to PYICU column names
        tbl = pv.read_csv(
            data_dir/self.files,
            convert_options=pv.ConvertOptions(column_types=col_types)
        )

        if self.num_rows is not None and tbl.num_rows != self.num_rows:
            raise warnings.warn(
                f"expected {self.num_rows} rows but got {tbl.num_rows} rows for table`{self.name}`",   
            )

        if self.partitioning:
            # NOTE: currently no partitioning is implemented, as standard parquet
            #       files appear to give adequate performance. This may need to 
            #       be implemented in the future
            raise NotImplementedError()
        else: 
            pq.write_table(tbl, out_dir/f"{self.name}.parquet")


    def __repr__(self) -> str:
        info = ""
        if self.num_rows is not None:
            info += f"rows: {self.num_rows}" 
        if self.cols is not None:
            if info != "":
                info += ", "
            info += f"cols: {len(self.cols)}"
        
        return f"<TableCfg: {self.name} ({info})>"


class ColumnSpec():
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

    def from_dict(x: Dict, col: str) -> Type['ColumnSpec']:
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

        if not 'name' in x.keys():
            raise ValueError(f'No `name` attribute provided for column config {col}')
        name = x.pop('name')

        if not 'spec' in x.keys():
            raise ValueError(f'No `spec` attribute provided for column config {col}')
        spec = x.pop('spec')

        return ColumnSpec(name, col, spec, **x)


    def from_tuple(x: Tuple[str, Dict]) -> Type['ColumnSpec']:
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
