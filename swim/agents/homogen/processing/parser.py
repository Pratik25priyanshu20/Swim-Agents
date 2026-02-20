# swim/agents/homogen/processing/parser.py


import pandas as pd
import numpy as np
from datetime import datetime
from swim.agents.homogen import setup_logging

logger = setup_logging()


class DataParser:
    """Enhanced parser that handles both metadata and measurement data."""

    def __init__(self, parameter_mappings=None, metadata_df=None):
        self.parameter_mappings = parameter_mappings or {}
        self.metadata_df = metadata_df if metadata_df is not None else pd.DataFrame()

    def parse(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Main parsing function with intelligent source detection."""
        df = df.copy()

        if df.empty:
            logger.warning(f"[Parser] Input DataFrame for {source_name} is empty.")
            return df

        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]

        # Detect if this is metadata or measurement data
        is_metadata = self._is_metadata_source(df, source_name)
        
        if is_metadata:
            logger.info(f"[Parser] Detected METADATA source: {source_name}")
            df = self._parse_metadata(df, source_name)
        else:
            logger.info(f"[Parser] Detected MEASUREMENT source: {source_name}")
            df = self._parse_measurements(df, source_name)

        logger.info(f"[Parser] Parsed {len(df)} records for {source_name}")
        return df

    def _is_metadata_source(self, df: pd.DataFrame, source_name: str) -> bool:
        """Detect if this is a metadata-only source."""
        # Check source name first
        if 'metadata' in source_name.lower():
            return True
        
        # Check if it has typical metadata columns but no measurements
        metadata_cols = ['station_id', 'station_name', 'lake', 'latitude', 'longitude', 
                        'municipality', 'country_code', 'water_body_type']
        
        # Common measurement indicators
        measurement_indicators = [
            'measurement_value', 'measurement_parameter',  # Long format
            'temp_c', 'temperature', 'ph', 'ph_wert',  # Wide format water quality
            'do_mg_l', 'dissolved_oxygen', 'turbidity', 'turbidity_ntu', 
            'chl_ug_l', 'chlorophyll_a', 'water_level_m', 'discharge_m3s'
        ]
        
        has_metadata = any(col in df.columns for col in metadata_cols)
        has_measurement_cols = any(col.lower() in [m.lower() for m in measurement_indicators] 
                                for col in df.columns)
        
        # If it has measurement_value or measurement_parameter, it's measurement data in long format
        if 'measurement_value' in df.columns or 'measurement_parameter' in df.columns:
            return False
        
        # If it has typical water quality columns, it's measurement data
        if has_measurement_cols:
            return False
        
        # If it has metadata but no measurements, it's metadata-only
        return has_metadata and not has_measurement_cols

    def _parse_metadata(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Parse metadata-only sources (station info, locations, etc.)."""
        # Normalize coordinates
        df = self._ensure_coordinates(df)
        
        # Ensure station identifier
        if 'station_id' not in df.columns:
            if 'id' in df.columns:
                df.rename(columns={'id': 'station_id'}, inplace=True)
            elif 'STATION_ID' in df.columns:
                df.rename(columns={'STATION_ID': 'station_id'}, inplace=True)
        
        # Add source tracking
        df['source_name'] = source_name
        df['data_type'] = 'metadata'
        
        # No timestamp needed for metadata
        logger.info(f"[Parser] Metadata parsing complete: {len(df)} stations")
        return df

    def _parse_measurements(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Parse measurement data sources."""
        # 1. Normalize coordinates
        df = self._ensure_coordinates(df)

        # 2. Normalize timestamps
        df = self._sanitize_timestamps(df)

        # 3. Apply parameter mappings
        for raw, mapped in self.parameter_mappings.items():
            if raw in df.columns:
                df.rename(columns={raw: mapped}, inplace=True)

        # 4. Metadata enrichment
        df = self._enrich_with_metadata(df)
        
        # 5. Add source tracking
        df['source_name'] = source_name
        df['data_type'] = 'measurement'

        return df

    def _ensure_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize coordinate column names and validate ranges."""
        coord_map = {
            "lon": "longitude", "lng": "longitude", "long": "longitude",
            "lat": "latitude", "Latitude": "latitude", "Longitude": "longitude",
            "LATITUDE": "latitude", "LONGITUDE": "longitude"
        }

        rename_dict = {}
        for col in df.columns:
            col_l = col.lower().strip()
            if col_l in coord_map and coord_map[col_l] not in df.columns:
                rename_dict[col] = coord_map[col_l]

        if rename_dict:
            df = df.rename(columns=rename_dict)
            logger.info(f"[Parser] Renamed coordinates: {rename_dict}")

        # Validate and clean coordinates
        if "latitude" in df.columns:
            df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
            invalid_lat = (df["latitude"] > 90) | (df["latitude"] < -90)
            if invalid_lat.any():
                logger.warning(f"[Parser] Removed {invalid_lat.sum()} invalid latitude values")
                df.loc[invalid_lat, "latitude"] = np.nan

        if "longitude" in df.columns:
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
            invalid_lon = (df["longitude"] > 180) | (df["longitude"] < -180)
            if invalid_lon.any():
                logger.warning(f"[Parser] Removed {invalid_lon.sum()} invalid longitude values")
                df.loc[invalid_lon, "longitude"] = np.nan

        return df

    def _sanitize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize timestamp columns."""
        ts_cols = ["measurement_timestamp", "timestamp", "date_time", "datetime", "date", "Date"]

        found_col = None
        for col in ts_cols:
            if col in df.columns:
                found_col = col
                break

        if found_col:
            try:
                # Try parsing timestamps
                df[found_col] = pd.to_datetime(df[found_col], errors="coerce", utc=True)
                
                # Count valid timestamps
                valid_count = df[found_col].notna().sum()
                total_count = len(df)
                
                if valid_count == 0:
                    logger.warning(f"[Parser] No valid timestamps found in '{found_col}' column")
                elif valid_count < total_count * 0.5:
                    logger.warning(f"[Parser] Only {valid_count}/{total_count} valid timestamps in '{found_col}'")
                else:
                    logger.info(f"[Parser] Parsed {valid_count}/{total_count} timestamps from '{found_col}'")
                
                # Rename to standard column name
                if found_col != "measurement_timestamp":
                    df.rename(columns={found_col: "measurement_timestamp"}, inplace=True)
                    
            except Exception as e:
                logger.error(f"[Parser] Failed to parse timestamps from '{found_col}': {e}")
                df["measurement_timestamp"] = pd.NaT
        else:
            logger.warning("[Parser] No timestamp column found.")
            df["measurement_timestamp"] = pd.NaT

        return df

    def _enrich_with_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Integrate metadata from external source."""
        if self.metadata_df.empty or "station_id" not in df.columns:
            return df
        
        meta_cols = ["station_id", "lake", "municipality", "country_code", "water_body_type"]
        joinable = [col for col in meta_cols if col in self.metadata_df.columns]
        
        if joinable:
            # Only merge if station_id exists and has values
            if df["station_id"].notna().any():
                df = df.merge(
                    self.metadata_df[joinable].drop_duplicates(), 
                    on="station_id", 
                    how="left",
                    suffixes=('', '_meta')
                )
                logger.info(f"[Parser] Enriched with metadata: {joinable}")
        
        return df