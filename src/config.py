from pathlib import Path
import yaml

def load_config(path):
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
