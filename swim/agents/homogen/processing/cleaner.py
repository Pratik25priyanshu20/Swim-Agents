# swim/agents/homogen/processing/cleaner.py

import logging
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from swim.agents.homogen import setup_logging

logger = setup_logging()

class DataCleaner:
    """Advanced data cleaning with safe KNN imputation."""

    def __init__(self, config: dict):
        self.knn_neighbors = config.get('knn_neighbors', 5)
        self.seasonal_period = config.get('seasonal_period', 365)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply KNN imputation safely per station group."""
        df = df.copy()

        if "measurement_timestamp" not in df.columns or "station_id" not in df.columns:
            logger.warning("Cleaner: Missing required columns — skipping cleaning.")
            return df

        cleaned = []
        for station_id, group in df.groupby("station_id"):
            try:
                cleaned_group = self._clean_group(group)
                cleaned.append(cleaned_group)
            except Exception as e:
                logger.warning(f"❌ KNN failed for group with {len(group)} rows: {e}")

        if cleaned:
            return pd.concat(cleaned, ignore_index=True)
        else:
            logger.warning("Cleaner: No station groups were cleaned.")
            return df  # fallback

    def _clean_group(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("measurement_timestamp").reset_index(drop=True)

        numeric_cols = [
            "temp_c", "ph", "do_mg_l", "turbidity_ntu",
            "chl_ug_l", "water_level_m", "discharge_m3s"
        ]
        present_cols = [col for col in numeric_cols if col in df.columns]

        if len(df) < 3 or len(present_cols) == 0:
            raise ValueError("Not enough data or columns for KNN imputation.")

        knn_input = df[present_cols].apply(pd.to_numeric, errors="coerce")

        # Check for completely missing columns (all NaNs)
        all_nan_cols = knn_input.columns[knn_input.isna().all()]
        if len(all_nan_cols) == len(knn_input.columns):
            raise ValueError("All imputation columns are fully missing.")

        imputer = KNNImputer(n_neighbors=min(self.knn_neighbors, len(df)-1))
        imputed = imputer.fit_transform(knn_input)

        if imputed.shape[1] != len(present_cols):
            raise ValueError(f"Shape mismatch after imputation: {imputed.shape} vs {len(present_cols)} columns")

        df[present_cols] = pd.DataFrame(imputed, columns=present_cols)
        return df