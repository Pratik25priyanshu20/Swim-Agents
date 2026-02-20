import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from typing import Dict, Any

from swim.agents.homogen import setup_logging
from swim.agents.homogen.utils.column_mapper import map_columns_auto

logger = setup_logging()

# Canonical mappings
CANON_MAP = {
    "temperature": ("temp_c", float),
    "WT": ("temp_c", float),
    "ph": ("ph", float),
    "PH": ("ph", float),
    "dissolved_oxygen": ("do_mg_l", float),
    "O2": ("do_mg_l", float),
    "turbidity": ("turbidity_ntu", float),
    "TRB": ("turbidity_ntu", float),
    "water_level_cm": ("water_level_m", lambda v: float(v) / 100.0),
    "W": ("water_level_m", lambda v: float(v) / 100.0),
    "discharge_m3s": ("discharge_m3s", float),
    "Q": ("discharge_m3s", float),
    "chlorophyll_a": ("chl_ug_l", float),
    "chl_a": ("chl_ug_l", float),
}

RANGE_GUARDS = {
    "temp_c": (-2.0, 40.0),
    "ph": (6.0, 9.5),
    "do_mg_l": (0.0, 20.0),
    "turbidity_ntu": (0.0, 1000.0),
    "water_level_m": (-50.0, 200.0),
    "discharge_m3s": (0.0, 100000.0),
    "chl_ug_l": (0.0, 1000.0),
}

DEFAULT_SOURCE_TRUST = {
    "PEGELONLINE": 0.95,
    "LfU Bayern": 0.97,
    "BrightSky": 0.92,
    "EEA Bathing Water": 0.90,
    "WAMO": 0.93,
    "GUAMU": 0.93,
    "Sentinel-2": 0.88,
    "Landsat": 0.88,
    "gemstat": 0.92,
    "bwd": 0.90,
}


class DataHarmonizer:
    """Enhanced data harmonizer with robust canonicalization and quality scoring."""
    
    def __init__(self, config, parameter_mappings: Dict, unit_conversions: Dict):
        self.config = config
        self.parameter_mappings = parameter_mappings
        self.unit_conversions = unit_conversions
    
    def harmonize(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Main harmonization pipeline."""
        df = df.copy()

        if df.empty:
            logger.warning(f"[Harmonizer] Empty dataframe for {source_name}")
            return df

        # Step 0: Fuzzy column mapping
        column_map = map_columns_auto(df.columns.tolist())
        df.rename(columns=column_map, inplace=True)
        logger.info(f"🧠 Mapped columns: {column_map}")

        # Step 1: Canonicalize parameters
        df = self._canonicalize_parameters(df, source_name)
        
        # Step 2: Apply range guards
        df = self._apply_range_guards(df)
        
        # Step 3: Detect outliers
        df = self._detect_outliers(df)
        
        # Step 4: Compute quality scores
        df = self._compute_quality_scores(df, source_name)
        
        # Step 5: Standardize schema (FIXED)
        df = self._standardize_schema(df, source_name)

        logger.info(f"✅ Harmonized {len(df)} records from {source_name}")
        return df

    def _canonicalize_parameters(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Convert source-specific columns to canonical format."""
        canonical_cols = {}
        transformations = []
        
        for col in df.columns:
            if col in CANON_MAP:
                canon_name, converter = CANON_MAP[col]
                try:
                    values = df[col].copy()
                    if values.dtype == 'object':
                        values = values.str.replace(',', '.', regex=False)
                    
                    canonical_cols[canon_name] = values.apply(
                        lambda v: converter(v) if pd.notna(v) else np.nan
                    )
                    
                    if canon_name != col:
                        transformations.append(f"{col}→{canon_name}")
                except Exception as e:
                    logger.warning(f"Failed to canonicalize {col}: {e}")
                    canonical_cols[col] = df[col]
            else:
                canonical_cols[col] = df[col]

        result = pd.DataFrame(canonical_cols)

        if transformations:
            logger.info(f"Applied {len(transformations)} canonical mappings: {', '.join(transformations[:5])}")
        
        return result

    def _apply_range_guards(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove values outside acceptable ranges."""
        for param, (min_val, max_val) in RANGE_GUARDS.items():
            if param in df.columns:
                original_count = df[param].notna().sum()
                mask = (df[param] < min_val) | (df[param] > max_val)
                removed = mask.sum()
                
                if removed > 0:
                    df.loc[mask, param] = np.nan
                    logger.warning(f"Removed {removed}/{original_count} out-of-range values for {param}")
        
        return df

    def _mad_zscores(self, values):
        """Calculate Modified Z-scores using Median Absolute Deviation."""
        x = values.dropna().values
        if len(x) < 5:
            return pd.Series([0.0] * len(values), index=values.index)
        
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad == 0:
            mad = 1e-9
        
        z = 0.6745 * (values - med) / mad
        return z.fillna(0.0)

    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and flag statistical outliers."""
        outlier_flags = []
        
        for param in RANGE_GUARDS:
            if param in df.columns and df[param].notna().sum() >= 5:
                z_scores = self._mad_zscores(df[param])
                outliers = np.abs(z_scores) > 5.0
                count = outliers.sum()
                
                if count > 0:
                    df.loc[outliers, param] = np.nan
                    outlier_flags.extend([f"{param}_outlier"] * count)
                    logger.info(f"Flagged {count} outliers in {param}")
        
        df['outlier_flags'] = ','.join(set(outlier_flags)) if outlier_flags else ''
        return df

    def _compute_quality_scores(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Compute quality score based on trust, completeness, and recency."""
        trust = self._get_source_trust(source_name)
        canonical_params = list(RANGE_GUARDS.keys())
        
        # Completeness score
        present_params = [c for c in canonical_params if c in df.columns]
        if present_params:
            completeness = df[present_params].notna().mean(axis=1)
        else:
            completeness = 0.0

        # Recency score
        if 'measurement_timestamp' in df.columns:
            timestamps = pd.to_datetime(df['measurement_timestamp'], errors='coerce', utc=True)
            now = datetime.now(timezone.utc)
            age_days = (now - timestamps).dt.total_seconds() / 86400.0
            recency = np.maximum(0.0, 1.0 - np.maximum(0.0, age_days - 7.0) / 21.0)
        else:
            recency = 0.8

        # Outlier penalty
        outlier_penalty = df['outlier_flags'].str.len().fillna(0) * 0.05
        outlier_penalty = np.minimum(outlier_penalty, 0.15)

        # Final quality score
        quality_score = (
            0.60 * trust +
            0.25 * completeness +
            0.15 * recency -
            outlier_penalty
        )

        df['quality_score'] = quality_score.clip(0.0, 1.0).round(3)
        return df

    def _get_source_trust(self, source_name: str) -> float:
        """Get trust score for data source."""
        for key, trust in DEFAULT_SOURCE_TRUST.items():
            if key.lower() in source_name.lower():
                return trust
        return 0.85

    def _standardize_schema(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """FIXED: Create standardized schema without losing data."""
        df['harmonized_at'] = datetime.now(timezone.utc).isoformat()
        df['harmonization_version'] = '2.0'
        df['source_name'] = source_name
        
        # Essential columns that MUST be present
        required_cols = [
            'measurement_timestamp', 'quality_score', 'outlier_flags',
            'source_name', 'harmonized_at', 'harmonization_version'
        ]
        
        # Optional metadata columns
        optional_metadata = [
            'station_name', 'station_id', 'lake', 'municipality',
            'latitude', 'longitude', 'geometry',
            'country_code', 'water_body_type', 'water_body_name', 'data_type'
        ]
        
        # Water quality parameters
        measurement_cols = list(RANGE_GUARDS.keys())
        
        # Build final column list - only include columns that actually have data
        final_cols = []
        
        # Add required columns
        for col in required_cols:
            if col not in df.columns:
                df[col] = '' if col in ['outlier_flags', 'source_name'] else None
            final_cols.append(col)
        
        # Add optional metadata only if they exist with data
        for col in optional_metadata:
            if col in df.columns:
                final_cols.append(col)
        
        # Add measurement columns only if they exist
        for col in measurement_cols:
            if col in df.columns:
                final_cols.append(col)
        
        # Add any remaining columns not yet included
        for col in df.columns:
            if col not in final_cols:
                final_cols.append(col)
        
        return df[final_cols]

    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """FIXED: Aggregate measurements to daily resolution with proper grouping."""
        if 'measurement_timestamp' not in df.columns:
            logger.warning("No measurement_timestamp column for aggregation")
            return df
        
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['measurement_timestamp'] = pd.to_datetime(
            df['measurement_timestamp'], utc=True, errors='coerce'
        )
        
        # Remove rows without valid timestamps
        df = df.dropna(subset=['measurement_timestamp'])
        
        if df.empty:
            logger.warning("No valid timestamps for aggregation")
            return df
        
        # Extract date for grouping
        df['date'] = df['measurement_timestamp'].dt.date
        
        # Determine grouping columns based on what exists
        group_cols = ['date']
        for col in ['lake', 'station_id', 'latitude', 'longitude']:
            if col in df.columns and df[col].notna().any():
                group_cols.append(col)
        
        # Get numeric columns that actually exist and have data
        numeric_cols = [
            col for col in RANGE_GUARDS.keys() 
            if col in df.columns and df[col].notna().any()
        ]
        
        if not numeric_cols:
            logger.warning("No numeric columns to aggregate")
            return df
        
        # Weighted aggregation function
        def weighted_mean(group, col):
            weights = group['quality_score'].fillna(0.5)
            values = group[col]
            valid = values.notna() & (weights > 0)
            
            if not valid.any():
                return np.nan
            
            return (values[valid] * weights[valid]).sum() / weights[valid].sum()
        
        # Build aggregation dictionary
        agg_dict = {
            col: lambda x, col=col: weighted_mean(x, col) 
            for col in numeric_cols
        }
        agg_dict['quality_score'] = 'mean'
        agg_dict['measurement_timestamp'] = 'first'
        
        # Perform aggregation
        try:
            aggregated = df.groupby(group_cols, dropna=False).apply(
                lambda g: pd.Series({
                    **{col: weighted_mean(g, col) for col in numeric_cols},
                    'quality_score': g['quality_score'].mean(),
                    'measurement_timestamp': g['measurement_timestamp'].iloc[0]
                })
            ).reset_index()
            
            logger.info(f"Aggregated {len(df)} records → {len(aggregated)} daily records")
            return aggregated
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}", exc_info=True)
            return df