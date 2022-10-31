from typing import Type, Tuple, Dict
import pandas as pd
from .utils import check_attributes_in_dict

class IdCfg():
    def __init__(self, cfg: pd.DataFrame = None):
        self.cfg = cfg
        self.cfg.sort_values('position', inplace=True)

    def from_dict(x: Dict) -> Type['IdCfg']:
        """_summary_
        """
        attr = ['id', 'position', 'start', 'end', 'table']

        ids = []
        for name, cfg in x.items():
            cfg = cfg.copy()
            cfg['id'] = name
            check_attributes_in_dict(cfg, attr, name, 'id')
            ids += [cfg]

        cfgs = pd.DataFrame(ids, columns=attr)
        return IdCfg(cfgs)
            
    def __repr__(self) -> str:
        repr = ""
        for id in self.cfg.id:
            if repr != "":
                repr += " < "
            repr += id
        
        return f"<IdCfg: [{repr}]>"