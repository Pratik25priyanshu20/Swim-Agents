import json
import yaml
from pathlib import Path

def load_config(path: str) -> dict:
    """
    Load configuration from YAML or JSON file.
    Args:
        path (str): Path to the config file (absolute or relative)
    Returns:
        dict: Parsed configuration
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    ext = config_path.suffix.lower()

    with open(config_path, "r", encoding="utf-8") as f:
        if ext in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif ext == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")