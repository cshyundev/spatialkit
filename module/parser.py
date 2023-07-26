import yaml
from typing import Dict, Any

def read_yaml(path:str) -> Dict[str, Any]:
    with open(path,"r") as f:
        dict = yaml.load(f)
    return dict