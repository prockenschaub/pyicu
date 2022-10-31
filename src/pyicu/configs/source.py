import warnings
from typing import Type, Dict, Union, List
from pathlib import Path

from .id import IdCfg
from .table import TableCfg
from .utils import check_attributes_in_dict


class SourceCfg():
    def __init__(self, name, id_cfg, tbl_cfg, **kwargs) -> None:
        self.name = name
        self.id_cfg = id_cfg
        self.tbl_cfg = tbl_cfg

    def from_dict(x: Dict) -> Type['SourceCfg']:
        """Create a source configuration from dict (e.g., read through JSON)

        Raises:
            ValueError:  if "name" is not among the keys of x
            ValueError:  if "tables" is not among the keys of x

        Returns:
            SourceCfg: configuration created from dictionary
        """
        check_attributes_in_dict(x, 'name', 'unnamed', 'source')
        name = x['name']

        check_attributes_in_dict(x, ['id_cfg', 'tables'], name, 'source')
        id_cfg = IdCfg.from_dict(x['id_cfg'])
        tbl_cfg = [TableCfg.from_tuple(t) for t in x['tables'].items()]

        return SourceCfg(name, id_cfg, tbl_cfg)
    
    def do_import(
        self, 
        data_dir: Path, 
        out_dir: Path = None, 
        tables: Union[str, List[str]] = None, 
        force: bool = False, 
        verbose: bool = True, 
        cleanup: bool = False, 
        **kwargs
    ):
        # TODO: implement force
        # TODO: implement verbose
        if out_dir is None:
            out_dir = data_dir
        if tables is None: 
            tables = [t.name for t in self.tbl_cfg]

        done = [t.name for t in self.tbl_cfg if t.imp_files_exist(out_dir)]
        skip = set(done).intersection(tables)
        todo = set(tables).difference(done)
        
        if verbose and not force and len(skip) > 0:
            print(f"The following tables have already been imported and will be skipped: {skip}")

        if len(todo) == 0:
            warnings.warn(f"All required tables have already been imported for source `{self.name}`.")
            return None

        for tbl in self.tbl_cfg:
            if tbl.name in todo:
                tbl.do_import(data_dir, out_dir)

        if verbose:
            print(f"Successfully imported {len(todo)} tables.")
        

    def __repr__(self) -> str:
        return f"<SourceCfg: {self.name} (tables: {len(self.tbl_cfg)})>"