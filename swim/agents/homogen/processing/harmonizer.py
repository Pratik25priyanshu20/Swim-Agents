# swim/agents/homogen/processing/harmonizer.py

import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from typing import Dict, Any

from swim.agents.homogen import setup_logging
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
}

class DataHarmonizer:
    """Enhanced data harmonizer with canonicalization and quality scoring."""
    
    def __init__(self, config, parameter_mappings: Dict, unit_conversions: Dict):
        self.config = config
        self.parameter_mappings = parameter_mappings
        self.unit_conversions = unit_conversions
    
    def harmonize(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        Main harmonization pipeline:
        1. Canonicalize parameters
        2. Apply range guards
        3. Detect outliers
        4. Compute quality scores
        5. Standardize schema
        """
        df = df.copy()
        
        # Step 1: Canonicalize parameters
        df = self._canonicalize_parameters(df, source_name)
        
        # Step 2: Apply range guards
        df = self._apply_range_guards(df)
        
        # Step 3: Detect and flag outliers
        df = self._detect_outliers(df)
        
        # Step 4: Compute quality scores
        df = self._compute_quality_scores(df, source_name)
        
        # Step 5: Standardize schema
        df = self._standardize_schema(df)
        
        logger.info(f"✅ Harmonized {len(df)} records from {source_name}")
        return df
    
    def _canonicalize_parameters(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Convert heterogeneous parameter names to canonical form."""
        canonical_cols = {}
        transformations = []
        
        for col in df.columns:
            if col in CANON_MAP:
                canon_name, converter = CANON_MAP[col]
                try:
                    # Convert values
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
        """Remove values outside physically plausible ranges."""
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
        """Compute Modified Z-scores using Median Absolute Deviation."""
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
        """Flag outliers using MAD-based robust Z-scores."""
        outlier_flags = []
        
        for param in RANGE_GUARDS.keys():
            if param in df.columns and df[param].notna().sum() >= 5:
                z_scores = self._mad_zscores(df[param])
                outliers = np.abs(z_scores) > 5.0
                
                if outliers.any():
                    outlier_count = outliers.sum()
                    df.loc[outliers, param] = np.nan
                    outlier_flags.extend([f"{param}_outlier"] * outlier_count)
                    logger.info(f"Flagged {outlier_count} outliers in {param}")
        
        df['outlier_flags'] = ','.join(set(outlier_flags)) if outlier_flags else ''
        return df
    
    def _compute_quality_scores(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        Compute quality score based on:
        - Source trust (60%)
        - Completeness (25%)
        - Recency (15%)
        """
        # Source trust
        trust = self._get_source_trust(source_name)
        
        # Completeness (% of canonical params present)
        canonical_params = list(RANGE_GUARDS.keys())
        completeness = df[[c for c in canonical_params if c in df.columns]].notna().mean(axis=1)
        
        # Recency (1.0 for <= 7 days, linear decay to 0 over next 21 days)
        if 'measurement_timestamp' in df.columns:
            now = datetime.now(timezone.utc)
            timestamps = pd.to_datetime(df['measurement_timestamp'], errors='coerce')
            age_days = (now - timestamps).dt.total_seconds() / 86400.0
            recency = np.maximum(0.0, 1.0 - np.maximum(0.0, age_days - 7.0) / 21.0)
        else:
            recency = 0.8  # Default if no timestamp
        
        # Penalty for outliers
        outlier_penalty = df['outlier_flags'].str.len().fillna(0) * 0.05
        outlier_penalty = np.minimum(outlier_penalty, 0.15)  # Cap at 15%
        
        # Combined score
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
        return 0.85  # Default trust
    
    def _standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all expected columns exist."""
        df['harmonized_at'] = datetime.now(timezone.utc).isoformat()
        df['harmonization_version'] = '2.0'
        
        # Standard metadata columns
        expected_cols = [
            'station_name', 'station_id', 'lake', 'municipality',
            'latitude', 'longitude', 'geometry',
            'temp_c', 'ph', 'do_mg_l', 'turbidity_ntu', 'chl_ug_l',
            'water_level_m', 'discharge_m3s',
            'measurement_timestamp', 'quality_score', 'outlier_flags',
            'source_name', 'data_type',
            'country_code', 'water_body_type', 'water_body_name',
            'harmonized_at', 'harmonization_version'
        ]
        
        for col in expected_cols:
            if col not in df.columns:
                df[col] = np.nan if df.empty else None
        
        return df[expected_cols]
    
    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate measurements to daily resolution per station/lake.
        Uses quality-weighted averaging.
        """
        if 'measurement_timestamp' not in df.columns:
            return df
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['measurement_timestamp']).dt.date
        
        # Group by location and date
        group_cols = ['lake', 'station_id', 'date']
        group_cols = [c for c in group_cols if c in df.columns]
        
        if not group_cols:
            return df
        
        # Numeric columns to aggregate
        numeric_cols = ['temp_c', 'ph', 'do_mg_l', 'turbidity_ntu', 'chl_ug_l', 
                    'water_level_m', 'discharge_m3s']
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        
        # Quality-weighted mean
        def weighted_mean(group, col):
            weights = group['quality_score']
            values = group[col]
            return (values * weights).sum() / weights.sum()
        
        agg_dict = {}
        for col in numeric_cols:
            agg_dict[col] = lambda x, col=col: weighted_mean(x, col)
        
        agg_dict['quality_score'] = 'mean'
        agg_dict['measurement_timestamp'] = 'first'
        
        aggregated = df.groupby(group_cols).agg(agg_dict).reset_index()
        
        logger.info(f"Aggregated {len(df)} records → {len(aggregated)} daily records")
        return aggregated