# swim/agents/homogen/processing/validator.py

import logging
import pandas as pd
import numpy as np
from typing import Dict, List

from swim.agents.homogen import setup_logging
logger = setup_logging()

class DataValidator:
    """Validates harmonized data against quality standards."""
    
    REQUIRED_FIELDS = [
        'station_id', 'measurement_timestamp', 'latitude', 'longitude'
    ]
    
    COORDINATE_BOUNDS = {
        'latitude': (-90, 90),
        'longitude': (-180, 180)
    }
    
    def __init__(self):
        self.validation_results = {}
    
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive validation pipeline:
        1. Required fields check
        2. Coordinate bounds validation
        3. Temporal consistency check
        4. Data completeness assessment
        """
        df = df.copy()
        flags = []
        
        # 1. Required fields
        missing_fields = [f for f in self.REQUIRED_FIELDS if f not in df.columns]
        if missing_fields:
            logger.warning(f"Missing required fields: {missing_fields}")
            flags.append('missing_required_fields')
        
        # 2. Coordinate validation
        df = self._validate_coordinates(df, flags)
        
        # 3. Temporal validation
        df = self._validate_timestamps(df, flags)
        
        # 4. Completeness check
        df = self._assess_completeness(df)
        
        # Add validation summary
        df['validation_flags'] = ','.join(set(flags)) if flags else ''
        df['validated_at'] = pd.Timestamp.now(tz='UTC').isoformat()
        
        logger.info(f"✅ Validation complete: {len(df)} records processed")
        if flags:
            logger.warning(f"⚠️  Validation flags: {', '.join(set(flags))}")
        
        return df
    
    def _validate_coordinates(self, df: pd.DataFrame, flags: List[str]) -> pd.DataFrame:
        """Ensure coordinates are within valid ranges."""
        if 'latitude' in df.columns and 'longitude' in df.columns:
            lat_min, lat_max = self.COORDINATE_BOUNDS['latitude']
            lon_min, lon_max = self.COORDINATE_BOUNDS['longitude']
            
            invalid_coords = (
                (df['latitude'] < lat_min) | (df['latitude'] > lat_max) |
                (df['longitude'] < lon_min) | (df['longitude'] > lon_max)
            )
            
            if invalid_coords.any():
                count = invalid_coords.sum()
                logger.warning(f"Found {count} records with invalid coordinates")
                df.loc[invalid_coords, ['latitude', 'longitude']] = np.nan
                flags.append('invalid_coordinates')
        
        return df
    
    def _validate_timestamps(self, df: pd.DataFrame, flags: List[str]) -> pd.DataFrame:
        """Check temporal consistency."""
        if 'measurement_timestamp' in df.columns:
            # Ensure timestamps are datetime
            df['measurement_timestamp'] = pd.to_datetime(
                df['measurement_timestamp'], 
                errors='coerce'
            )
            
            # Flag future dates
            now = pd.Timestamp.now(tz='UTC')
            future_dates = df['measurement_timestamp'] > now
            
            if future_dates.any():
                count = future_dates.sum()
                logger.warning(f"Found {count} records with future timestamps")
                df.loc[future_dates, 'measurement_timestamp'] = pd.NaT
                flags.append('future_timestamps')
            
            # Flag very old dates (pre-1900)
            old_dates = df['measurement_timestamp'] < pd.Timestamp('1900-01-01', tz='UTC')
            if old_dates.any():
                count = old_dates.sum()
                logger.warning(f"Found {count} records with pre-1900 timestamps")
                flags.append('suspicious_old_dates')
        
        return df
    
    def _assess_completeness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate data completeness score."""
        important_fields = [
            'temp_c', 'ph', 'do_mg_l', 'turbidity_ntu', 'chl_ug_l',
            'water_level_m', 'discharge_m3s'
        ]
        
        present_fields = [f for f in important_fields if f in df.columns]
        
        if present_fields:
            completeness = df[present_fields].notna().mean(axis=1)
            df['data_completeness'] = completeness.round(3)
        else:
            df['data_completeness'] = 0.0
        
        return df
    
    def get_validation_summary(self, df: pd.DataFrame) -> Dict:
        """Generate validation summary statistics."""
        summary = {
            'total_records': len(df),
            'records_with_flags': (df['validation_flags'] != '').sum() if 'validation_flags' in df.columns else 0,
            'avg_completeness': df['data_completeness'].mean() if 'data_completeness' in df.columns else 0.0,
            'avg_quality_score': df['quality_score'].mean() if 'quality_score' in df.columns else 0.0,
        }
        
        if 'validation_flags' in df.columns:
            flag_counts = df['validation_flags'].str.split(',').explode().value_counts()
            summary['flag_breakdown'] = flag_counts.to_dict()
        
        return summary