# swim/agents/homogen/processing/harmonizer.py

from datetime import datetime
import pandas as pd

class DataHarmonizer:
    def __init__(self, parameter_mappings, unit_conversions, config):
        self.parameter_mappings = parameter_mappings
        self.unit_conversions = unit_conversions
        self.config = config

    def harmonize(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        df = df.copy()
        if 'measurement_unit' in df.columns:
            df['measurement_unit_standardized'] = df['measurement_unit'].map(self.unit_conversions).fillna(df['measurement_unit'])
        df['harmonized_at'] = datetime.now()
        df['harmonization_version'] = '1.0'
        df['collection_date'] = df.get('measurement_timestamp', pd.NaT)

        expected_cols = [
            'station_name','station_id','municipality','latitude','longitude','geometry',
            'measurement_parameter','measurement_value','measurement_unit','measurement_unit_standardized',
            'quality_flag','source_name','data_type','collection_date',
            'country_code','eu_national_code','water_body_type','water_body_name',
            'season_quality_status','monitoring_calendar_status','management_status',
            'bathing_water_profile_url','classification_date','assessment_year','reporting_year',
            'harmonized_at','harmonization_version'
        ]
        for col in expected_cols:
            if col not in df:
                df[col] = pd.NaT if 'date' in col else None
        return df[expected_cols]