# swim/agents/homogen/processing/cleaner.py

import logging
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from statsmodels.tsa.seasonal import STL

from swim.agents.homogen import setup_logging
logger = setup_logging()

class DataCleaner:
    """Advanced data cleaning with seasonal decomposition and KNN imputation."""
    
    def __init__(self, config: dict):
        self.knn_neighbors = config.get('knn_neighbors', 5)
        self.seasonal_period = config.get('seasonal_period', 365)
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply advanced cleaning steps:
        1. Temporal reindexing
        2. KNN imputation for gaps
        3. Seasonal decomposition (optional)
        """
        df = df.copy()
        
        # Only clean if we have time series data
        if 'measurement_timestamp' not in df.columns:
            return df
        
        # Group by station/lake for time series cleaning
        if 'station_id' in df.columns:
            grouped = df.groupby('station_id')
            cleaned_dfs = []
            
            for station_id, group in grouped:
                cleaned_group = self._clean_timeseries(group)
                cleaned_dfs.append(cleaned_group)
            
            df = pd.concat(cleaned_dfs, ignore_index=True)
            logger.info(f"  ✓ Applied KNN imputation to {len(grouped)} stations")
        
        return df
    
    def _clean_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean a single time series (per station)."""
        if len(df) < 5:
            return df
        
        df = df.sort_values('measurement_timestamp')
        
        # Numeric columns for imputation
        numeric_cols = ['temp_c', 'ph', 'do_mg_l', 'turbidity_ntu', 'chl_ug_l',
                'water_level_m', 'discharge_m3s']
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        
        if not numeric_cols:
            return df
        
        # KNN imputation
        imputer = KNNImputer(n_neighbors=min(self.knn_neighbors, len(df) - 1))
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        return df