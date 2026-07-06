from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_yaml(filename: str):

    path = PROJECT_ROOT / "configs" / filename

    with open(path, "r") as f:
        return yaml.safe_load(f)