from typing import Type, Dict
import pandas as pd
from .utils import check_attributes_in_dict

class IdCfg():
    def __init__(self, cfg: pd.DataFrame = None):
        self.cfg = cfg   # TODO: change to a better name
        self.cfg.sort_values('position', inplace=True)

    def from_dict(x: Dict) -> Type['IdCfg']:
        """_summary_
        """
        attr = ['name', 'id', 'position', 'start', 'end', 'table']

        ids = []
        for name, cfg in x.items():
            cfg = cfg.copy()
            cfg['name'] = name
            check_attributes_in_dict(cfg, attr, name, 'id')
            ids += [cfg]

        cfgs = pd.DataFrame(ids, columns=attr)
        cfgs.set_index('position', inplace=True)
        return IdCfg(cfgs)
            
    def __getitem__(self, id: str) -> pd.Series:
        if not isinstance(id, str):
            raise TypeError(f'Expected an ID type (e.g., icustay) as string, got {id.__class__}.')
        if not any(self.cfg.name == id):
            raise ValueError(f'ID type {id} not defined.')
        return self.cfg[self.cfg.name == id].squeeze()

    @property
    def names(self) -> pd.Series:
        return self.cfg.name

    @property
    def ids(self) -> pd.Series:
        return self.cfg.id

    @property
    def starts(self) -> pd.Series:
        return self.cfg.start

    @property
    def ends(self) -> pd.Series:
        return self.cfg.end

    @property
    def tables(self) -> pd.Series:
        return self.cfg.table

    def __repr__(self) -> str:
        repr = ""
        for id in self.cfg.id:
            if repr != "":
                repr += " < "
            repr += id
        
        return f"<IdCfg>: [{repr}]\n{self.cfg.__repr__()}"