import pandas as pd
from pathlib import Path

def load_excel(file_path: Path, sheet_name: str = None) -> pd.DataFrame:
    return pd.read_excel(file_path, sheet_name=sheet_name)