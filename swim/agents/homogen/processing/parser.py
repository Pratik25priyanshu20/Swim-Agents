import pandas as pd
from shapely.geometry import Point

class DataParser:
    def __init__(self, field_mappings: dict, metadata_df: pd.DataFrame):
        self.field_mappings = field_mappings
        self.metadata_df = metadata_df

    def parse(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        df = df.rename(columns={k: v for k, v in self.field_mappings.items() if k in df.columns})
        if source_name == 'samples': return self._parse_groundwater(df)
        if source_name in ['gemstat_metadata', 'bwd_metadata', 'bwd_synthetic']: return self._parse_metadata(df)
        return df

    def _parse_groundwater(self, df):
        if 'station_id' in df.columns and not self.metadata_df.empty:
            df = df.merge(self.metadata_df[['station_id', 'latitude', 'longitude']], on='station_id', how='left')
        df['measurement_value'] = pd.to_numeric(df['measurement_value'].astype(str).str.replace(',', '.'), errors='coerce')
        df['measurement_timestamp'] = pd.to_datetime(df['measurement_timestamp'], errors='coerce')
        df['source_name'] = 'groundwater_quality'
        df['data_type'] = 'in_situ'
        df['geometry'] = df.apply(lambda r: Point(r['longitude'], r['latitude']).wkt if pd.notna(r['longitude']) and pd.notna(r['latitude']) else None, axis=1)
        return df

    def _parse_metadata(self, df):
        df['latitude'] = pd.to_numeric(df.get('latitude'), errors='coerce')
        df['longitude'] = pd.to_numeric(df.get('longitude'), errors='coerce')
        df['geometry'] = df.apply(lambda r: Point(r['longitude'], r['latitude']).wkt if pd.notna(r['longitude']) and pd.notna(r['latitude']) else None, axis=1)
        df['measurement_timestamp'] = pd.NaT
        df['source_name'] = 'bathing_water_metadata'
        df['data_type'] = 'tertiary'
        return df