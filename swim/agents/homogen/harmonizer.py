# swim/agents/homogen/processing/harmonizer.py

from datetime import datetime
import pandas as pd


class DataHarmonizer:
    def __init__(self, harmonization_config=None, parameter_mappings=None, unit_conversions=None):
        self.config = harmonization_config or {}
        self.parameter_mappings = parameter_mappings or {}
        self.unit_conversions = unit_conversions or {}

    def harmonize(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        df = df.copy()

        # Apply unit conversion if column exists
        if 'measurement_unit' in df.columns:
            df['measurement_unit_standardized'] = (
                df['measurement_unit']
                .map(self.unit_conversions)
                .fillna(df['measurement_unit'])
            )

        # Add metadata columns
        df['harmonized_at'] = datetime.now()
        df['harmonization_version'] = '1.0'
        df['collection_date'] = df.get('measurement_timestamp', pd.NaT)

        # Fill any expected columns that are missing
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
            if col not in df.columns:
                df[col] = pd.NaT if 'date' in col else None

        return df[expected_cols]