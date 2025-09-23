# swim/agents/homogen/core_pipeline.py
"""
HOMOGEN Agent Core Pipeline - Updated to enrich `samples` with metadata for spatial information.
"""
"""
import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
from shapely.geometry import Point

from swim.shared.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    name: str
    path: str
    format: str
    encoding: str = 'utf-8'
    separator: str = ','
    sheet_name: Optional[str] = None

@dataclass
class HarmonizationConfig:
    target_crs: str = "EPSG:4326"
    temporal_resolution: str = "daily"
    quality_threshold: float = 0.8
    output_format: str = "parquet"

class HOMOGENPipeline:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_sources_config = self._load_data_sources_config()
        self.harmonization_rules = self._load_harmonization_rules()
        self.global_settings = self._load_global_settings()

        # Load GEMSTAT metadata to enrich samples
        gemstat_path = self.project_root / "data/harmonized/gemstat_metadata.parquet"
        self.metadata_df = pd.read_parquet(gemstat_path) if gemstat_path.exists() else pd.DataFrame()

        self.ingestor = DataIngestor(self.project_root)
        self.parser = DataParser(self.harmonization_rules.get("parameter_mappings", {}), self.metadata_df)
        self.harmonizer = DataHarmonizer(
            harmonization_config=HarmonizationConfig(),
            parameter_mappings=self.harmonization_rules.get("parameter_mappings", {}),
            unit_conversions=self.harmonization_rules.get("unit_conversions", {})
        )
        self.validator = DataValidator()
        self.data_sources = {name: DataSource(**details) for name, details in self.data_sources_config.get('data_sources', {}).items()}
        self.harmonized_data = {}

    def _load_data_sources_config(self) -> Dict:
        path = self.project_root / "swim/agents/homogen/config/data_sources.yaml"
        return load_config(str(path))

    def _load_harmonization_rules(self) -> Dict:
        path = self.project_root / "swim/agents/homogen/config/harmonization_rules.yaml"
        return load_config(str(path))

    def _load_global_settings(self) -> Dict:
        path = self.project_root / "config/settings/development.yaml"
        return load_config(str(path))

    def run_pipeline(self, source_names: Optional[List[str]] = None):
        if source_names is None:
            source_names = list(self.data_sources.keys())
        for source_name in source_names:
            try:
                logger.info(f"\n--- Processing Source: {source_name} ---")
                source = self.data_sources[source_name]
                raw = self.ingestor.ingest(source)
                parsed = self.parser.parse(raw, source_name)
                harmonized = self.harmonizer.harmonize(parsed, source_name)
                validated = self.validator.validate(harmonized)
                self.harmonized_data[source_name] = validated
                self._save_harmonized_data(validated, source_name)
            except Exception as e:
                logger.error(f"Pipeline error in {source_name}: {e}", exc_info=True)
        return self.harmonized_data

    def _save_harmonized_data(self, df: pd.DataFrame, source_name: str):
        out_dir = self.project_root / "data/harmonized"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / f"{source_name}.parquet", engine="pyarrow", index=False)
        metadata = {
            "source_name": source_name,
            "record_count": len(df),
            "columns": list(df.columns),
            "data_types": df.dtypes.astype(str).to_dict(),
            "processing_timestamp": datetime.now().isoformat(),
            "quality_stats": {
                "avg_quality_score": float(df['quality_score'].mean()) if 'quality_score' in df.columns else None,
                "records_with_flags": int((df['validation_flags'] != '').sum()) if 'validation_flags' in df.columns else 0
            },
        }
        with open(out_dir / f"{source_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

class DataIngestor:
    def __init__(self, project_root):
        self.project_root = project_root

    def ingest(self, source: DataSource) -> pd.DataFrame:
        path = self.project_root / source.path
        if not path.exists(): raise FileNotFoundError(f"Not found: {path}")
        if source.format.lower() == 'csv':
            return pd.read_csv(path, encoding=source.encoding, sep=source.separator)
        elif source.format.lower() in ['xlsx', 'xls']:
            return pd.read_excel(path, sheet_name=source.sheet_name)
        else:
            raise ValueError(f"Unsupported format: {source.format}")

class DataParser:
    def __init__(self, field_mappings: Dict, metadata_df: pd.DataFrame):
        self.field_mappings = field_mappings
        self.metadata_df = metadata_df

    def parse(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        df = df.rename(columns={k: v for k, v in self.field_mappings.items() if k in df.columns})
        if source_name == 'samples': return self._parse_groundwater(df)
        if source_name == 'gemstat_metadata': return self._parse_metadata(df)
        if source_name == 'bwd_metadata': return self._parse_metadata(df)
        return df

    def _parse_groundwater(self, df):
        if 'station_id' in df.columns and not self.metadata_df.empty:
            df = df.merge(self.metadata_df[['station_id', 'latitude', 'longitude']], on='station_id', how='left')
        df['measurement_value'] = pd.to_numeric(df['measurement_value'].astype(str).str.replace(',', '.'), errors='coerce')
        df['measurement_timestamp'] = pd.to_datetime(df['measurement_timestamp'], errors='coerce')
        df['source_name'] = 'groundwater_quality'; df['data_type'] = 'in_situ'
        df['geometry'] = df.apply(lambda r: Point(r['longitude'], r['latitude']).wkt if pd.notna(r['longitude']) and pd.notna(r['latitude']) else None, axis=1)
        return df

    def _parse_metadata(self, df):
        df['latitude'] = pd.to_numeric(df.get('latitude', pd.Series()), errors='coerce')
        df['longitude'] = pd.to_numeric(df.get('longitude', pd.Series()), errors='coerce')
        df['geometry'] = df.apply(lambda r: Point(r['longitude'], r['latitude']).wkt if pd.notna(r['longitude']) and pd.notna(r['latitude']) else None, axis=1)
        df['measurement_timestamp'] = pd.NaT
        df['source_name'] = 'bathing_water_metadata'; df['data_type'] = 'tertiary'
        return df

class DataHarmonizer:
    def __init__(self, harmonization_config, parameter_mappings, unit_conversions):
        self.config = harmonization_config
        self.parameter_mappings = parameter_mappings
        self.unit_conversions = unit_conversions

    def harmonize(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        df = df.copy()
        if 'measurement_unit' in df:
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

class DataValidator:
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        df['quality_score'] = 1.0
        df['validation_flags'] = ''
        df = self._validate_required(df)
        return df

    def _validate_required(self, df):
        for col in ['station_id', 'latitude', 'longitude', 'geometry']:
            if col in df.columns:
                missing = df[col].isna()
                df.loc[missing, 'quality_score'] *= 0.6
                df.loc[missing, 'validation_flags'] += f'MISSING_{col.upper()};'
        return df

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[3]
    pipeline = HOMOGENPipeline(project_root)
    pipeline.run_pipeline()
    logger.info("\n✅ HOMOGEN pipeline finished. Check 'data/harmonized/' for output.")
    
    
    
        
        """
        
        
        
        # Note: The above code is a complete implementation of the HOMOGEN agent core pipeline.
        # It includes data ingestion, parsing, harmonization, validation, and saving the results.
        # Make sure to adjust paths and configurations as needed for your specific environment. 
        
        
        

 # swim/agents/homogen/core_pipeline.py

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

import pandas as pd

from swim.shared.utils.config_loader import load_config
from swim.agents.homogen.processing.parser import DataParser
from swim.agents.homogen.processing.harmonizer import DataHarmonizer
from swim.agents.homogen.processing.validator import DataValidator
from swim.agents.homogen.ingestion.csv_ingestor import load_csv
from swim.agents.homogen.ingestion.excel_ingestor import load_excel
from swim.agents.homogen.tools import compute_geo_bounds  # <--  Geo bounding box function

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

@dataclass
class DataSource:
    name: str
    path: str
    format: str
    encoding: str = 'utf-8'
    separator: str = ','
    sheet_name: str = None

@dataclass
class HarmonizationConfig:
    target_crs: str = "EPSG:4326"
    temporal_resolution: str = "daily"
    quality_threshold: float = 0.8
    output_format: str = "parquet"

class HOMOGENPipeline:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_sources_config = load_config(project_root / "swim/agents/homogen/config/data_sources.yaml")
        self.harmonization_rules = load_config(project_root / "swim/agents/homogen/config/harmonization_rules.yaml")
        self.global_settings = load_config(project_root / "config/settings/development.yaml")

        self.data_sources = {name: DataSource(**info) for name, info in self.data_sources_config["data_sources"].items()}
        gemstat_meta_path = project_root / "data/harmonized/gemstat_metadata.parquet"
        self.metadata_df = pd.read_parquet(gemstat_meta_path) if gemstat_meta_path.exists() else pd.DataFrame()

        self.parser = DataParser(self.harmonization_rules.get("parameter_mappings", {}), self.metadata_df)
        self.harmonizer = DataHarmonizer(
            config=HarmonizationConfig(),
            parameter_mappings=self.harmonization_rules.get("parameter_mappings", {}),
            unit_conversions=self.harmonization_rules.get("unit_conversions", {})
        )
        self.validator = DataValidator()
        self.harmonized_data = {}

    def run_pipeline(self, source_names: Optional[List[str]] = None):
        if source_names is None:
            source_names = list(self.data_sources.keys())

        for source_name in source_names:
            logger.info(f"\n--- Processing Source: {source_name} ---")
            try:
                source = self.data_sources[source_name]
                raw_df = self._load_data(source)
                parsed_df = self.parser.parse(raw_df, source_name)
                harmonized_df = self.harmonizer.harmonize(parsed_df, source_name)
                validated_df = self.validator.validate(harmonized_df)
                self.harmonized_data[source_name] = validated_df
                self._save_output(validated_df, source_name)
            except Exception as e:
                logger.error(f"❌ Pipeline error in {source_name}: {e}", exc_info=True)

        logger.info("\n✅ HOMOGEN pipeline finished. Check 'data/harmonized/' for output.")
        return self.harmonized_data

    def _load_data(self, source: DataSource) -> pd.DataFrame:
        path = self.project_root / source.path
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")

        if source.format.lower() == 'csv':
            return load_csv(path, encoding=source.encoding, separator=source.separator)
        elif source.format.lower() in ['xlsx', 'xls']:
            return load_excel(path, sheet_name=source.sheet_name)
        else:
            raise ValueError(f"Unsupported format: {source.format}")

    def _save_output(self, df: pd.DataFrame, source_name: str):
        out_dir = self.project_root / "data/harmonized"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save harmonized data
        df.to_parquet(out_dir / f"{source_name}.parquet", engine="pyarrow", index=False)

        # ⛰️ Compute geo bounding box
        geo_bbox = compute_geo_bounds(df)

        # Save metadata summary
        metadata = {
            "source_name": source_name,
            "record_count": len(df),
            "columns": list(df.columns),
            "data_types": df.dtypes.astype(str).to_dict(),
            "processing_timestamp": datetime.now().isoformat(),
            "geo_bbox": geo_bbox,  # ✅ spatial metadata
            "quality_stats": {
                "avg_quality_score": float(df['quality_score'].mean()) if 'quality_score' in df.columns else None,
                "records_with_flags": int((df['validation_flags'] != '').sum()) if 'validation_flags' in df.columns else 0
            },
        }
        with open(out_dir / f"{source_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

# ------------------ Entry Point ------------------ #
if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[3]
    pipeline = HOMOGENPipeline(project_root)
    pipeline.run_pipeline()