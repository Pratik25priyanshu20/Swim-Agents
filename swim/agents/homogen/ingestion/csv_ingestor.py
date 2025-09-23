# swim/agents/homogen/ingestion/csv_ingestor.py

import pandas as pd

def load_csv(path, encoding='utf-8', separator=','):
    return pd.read_csv(path, encoding=encoding, sep=separator)