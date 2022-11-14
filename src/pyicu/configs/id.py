from typing import Type, Dict
import pandas as pd
from .utils import check_attributes_in_dict

class IdCfg():
    def __init__(self, cfg: pd.DataFrame = None):
        self.cfg = cfg   # TODO: change to a better name
        self.cfg.sort_values('position', inplace=True)

    @property
    def id_var(self):
        return self.loc[self.index.max(), 'name']

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
            raise TypeError(f'expected an Id type (e.g., icustay) as string, got {id.__class__}.')
        if not any(self.cfg.name == id):
            raise ValueError(f'Id type {id} not defined.')
        return self.cfg[self.cfg.name == id].squeeze()

    def __getattr__(self, attr):
        """Forward any unknown attributes to the underlying pd.DataFrame
        """ 
        return getattr(self.cfg, attr) 

    def __repr__(self) -> str:
        repr = ""
        for id in self.cfg.id:
            if repr != "":
                repr += " < "
            repr += id
        
        return f"<IdCfg>: [{repr}]\n{self.cfg.__repr__()}"