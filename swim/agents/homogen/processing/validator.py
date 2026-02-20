# swim/agents/homogen/processing/validator.py


import pandas as pd
import numpy as np
from datetime import datetime, timezone

from swim.agents.homogen import setup_logging
logger = setup_logging()


class DataValidator:
    """
    Enhanced validator that handles both metadata and measurement data.
    """

    def __init__(self):
        self.validation_summary = {}

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main validation entry point."""
        df = df.copy()

        # Detect data type
        data_type = df.get('data_type', pd.Series(['unknown'])).iloc[0] if len(df) > 0 else 'unknown'
        is_metadata = (data_type == 'metadata') or self._is_metadata_only(df)

        if is_metadata:
            logger.info("[Validator] Detected metadata source - skipping measurement validation")
            df = self._validate_metadata(df)
        else:
            logger.info("[Validator] Validating measurement data")
            df = self._validate_measurements(df)

        # Store summary
        self.validation_summary = {
            "data_type": "metadata" if is_metadata else "measurement",
            "remaining_rows": len(df),
            "columns_present": list(df.columns),
        }

        return df

    def _is_metadata_only(self, df: pd.DataFrame) -> bool:
        """Check if dataframe contains only metadata (no measurements)."""
        measurement_cols = [
            "temp_c", "ph", "do_mg_l", "turbidity_ntu",
            "chl_ug_l", "water_level_m", "discharge_m3s"
        ]
        
        # Check if any measurement columns exist with data
        has_measurements = any(
            col in df.columns and df[col].notna().any() 
            for col in measurement_cols
        )
        
        return not has_measurements

    def _validate_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate metadata sources (station info, locations)."""
        # 1. Validate coordinates if present
        if "latitude" in df.columns and "longitude" in df.columns:
            invalid_coords = df["latitude"].isna() | df["longitude"].isna()
            invalid_count = invalid_coords.sum()
            
            if invalid_count > 0:
                logger.warning(f"[Validator] {invalid_count} records with invalid coordinates")
        
        # 2. Check for required metadata fields
        required = ["station_id", "latitude", "longitude"]
        missing = [col for col in required if col not in df.columns or df[col].isna().all()]
        
        if missing:
            logger.warning(f"[Validator] Missing critical metadata: {missing}")
        
        logger.info(f"[Validator] Metadata validation complete: {len(df)} stations")
        return df

    def _validate_measurements(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate measurement data sources."""
        numeric_cols = [
            "temp_c", "ph", "do_mg_l", "turbidity_ntu",
            "chl_ug_l", "water_level_m", "discharge_m3s"
        ]

        # 1. Timestamp validation
        df = self._validate_timestamps(df)

        # 2. Enforce numeric types
        df = self._coerce_numeric(df, numeric_cols)

        # 3. Drop empty measurements
        df = self._drop_empty_measurements(df, numeric_cols)

        # 4. Validate coordinate data if present
        if "latitude" in df.columns and "longitude" in df.columns:
            invalid_coords = df["latitude"].isna() | df["longitude"].isna()
            if invalid_coords.any():
                logger.info(f"[Validator] {invalid_coords.sum()} records without coordinates (keeping anyway)")

        logger.info(f"[Validator] Measurement validation complete: {len(df)} records")
        return df

    def _validate_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean timestamps."""
        if "measurement_timestamp" not in df.columns:
            logger.warning("[Validator] No timestamp column found")
            return df

        # Convert to UTC
        df["measurement_timestamp"] = pd.to_datetime(
            df["measurement_timestamp"], errors="coerce", utc=True
        )

        now_utc = datetime.now(timezone.utc)

        # Remove future timestamps
        future_mask = df["measurement_timestamp"] > now_utc
        future_count = future_mask.sum()

        if future_count > 0:
            logger.warning(f"[Validator] Removed {future_count} future timestamps")
            df.loc[future_mask, "measurement_timestamp"] = pd.NaT

        # Check validity rate
        valid_count = df["measurement_timestamp"].notna().sum()
        total_count = len(df)
        
        if valid_count == 0:
            logger.error("[Validator] No valid timestamps found!")
        elif valid_count < total_count * 0.5:
            logger.warning(f"[Validator] Low timestamp validity: {valid_count}/{total_count}")
        else:
            logger.info(f"[Validator] {valid_count}/{total_count} valid timestamps")

        return df

    def _coerce_numeric(self, df: pd.DataFrame, numeric_cols) -> pd.DataFrame:
        """Force numeric types for water quality parameters."""
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric, coercing errors to NaN
                original_count = df[col].notna().sum()
                df[col] = pd.to_numeric(df[col], errors="coerce")
                new_count = df[col].notna().sum()
                
                if original_count > new_count:
                    logger.warning(f"[Validator] Coerced {original_count - new_count} non-numeric values in '{col}'")
        
        return df

    def _drop_empty_measurements(self, df: pd.DataFrame, numeric_cols) -> pd.DataFrame:
        """Remove rows with no valid measurements."""
        valid_cols = [c for c in numeric_cols if c in df.columns]

        if not valid_cols:
            logger.warning("[Validator] No measurement columns found - keeping all rows")
            return df

        # Keep rows that have at least one valid measurement
        has_data_mask = df[valid_cols].notna().any(axis=1)
        removed = (~has_data_mask).sum()

        if removed > 0:
            logger.info(f"[Validator] Removed {removed} rows with no measurements")

        return df[has_data_mask]

    def get_validation_summary(self, df: pd.DataFrame = None) -> dict:
        """Return validation summary for metadata output."""
        if df is not None:
            self.validation_summary["rows_after_validation"] = len(df)
        return self.validation_summary