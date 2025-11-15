# swim/agents/homogen/core_pipeline.py

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass

import pandas as pd
import numpy as np

from swim.shared.utils.config_loader import load_config
from swim.agents.homogen.processing.parser import DataParser
from swim.agents.homogen.processing.harmonizer import DataHarmonizer
from swim.agents.homogen.processing.cleaner import DataCleaner
from swim.agents.homogen.processing.validator import DataValidator
from swim.agents.homogen.ingestion.csv_ingestor import load_csv
from swim.agents.homogen.ingestion.excel_ingestor import load_excel
from swim.agents.homogen.tools import compute_geo_bounds

from swim.agents.homogen import setup_logging
logger = setup_logging()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)

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
    quality_threshold: float = 0.7
    output_format: str = "parquet"
    enable_cleaning: bool = True
    enable_aggregation: bool = True

class HOMOGENPipeline:
    """
    Enhanced HOMOGEN Pipeline with complete harmonization workflow:
    1. Data Ingestion (CSV/Excel)
    2. Parsing & Schema Mapping
    3. Canonicalization & Range Guards
    4. Advanced Cleaning (Outliers, Seasonal, KNN Imputation)
    5. Quality Scoring
    6. Validation
    7. Daily Aggregation
    8. Output Generation
    """
    
    def __init__(self, project_root: Path, config: Optional[HarmonizationConfig] = None):
        self.project_root = project_root
        self.config = config or HarmonizationConfig()
        
        # Load configurations
        self.data_sources_config = load_config(
            project_root / "swim/agents/homogen/config/data_sources.yaml"
        )
        self.harmonization_rules = load_config(
            project_root / "swim/agents/homogen/config/harmonization_rules.yaml"
        )
        self.global_settings = load_config(
            project_root / "config/settings/development.yaml"
        )

        # Initialize data sources
        self.data_sources = {
            name: DataSource(**info) 
            for name, info in self.data_sources_config["data_sources"].items()
        }
        
        # Load metadata if available
        gemstat_meta_path = project_root / "data/harmonized/gemstat_metadata.parquet"
        self.metadata_df = (
            pd.read_parquet(gemstat_meta_path) 
            if gemstat_meta_path.exists() 
            else pd.DataFrame()
        )

        # Initialize processing components
        self.parser = DataParser(
            self.harmonization_rules.get("parameter_mappings", {}),
            self.metadata_df
        )
        
        self.harmonizer = DataHarmonizer(
            config=self.config,
            parameter_mappings=self.harmonization_rules.get("parameter_mappings", {}),
            unit_conversions=self.harmonization_rules.get("unit_conversions", {})
        )
        
        self.cleaner = DataCleaner(config={
            'knn_neighbors': 5,
            'seasonal_period': 365
        })
        
        self.validator = DataValidator()
        
        self.harmonized_data = {}
        self.processing_stats = {}

def run_pipeline(self, source_names: Optional[List[str]] = None, 
                 skip_cleaning: bool = False, skip_aggregation: bool = False) -> Dict[str, pd.DataFrame]:
    """Execute harmonization pipeline with robust error handling."""
    
    if source_names is None:
        source_names = list(self.data_sources.keys())

    logger.info(f"\n{'='*80}")
    logger.info(f"🌊 HOMOGEN Pipeline v2.0 - Processing {len(source_names)} sources")
    logger.info(f"{'='*80}\n")

    successful_sources = []
    failed_sources = []

    for source_name in source_names:
        logger.info(f"\n{'─'*80}")
        logger.info(f"📦 Processing: {source_name}")
        logger.info(f"{'─'*80}")
        
        try:
            # Validate source exists
            if source_name not in self.data_sources:
                raise ValueError(f"Unknown source: {source_name}")
            
            source = self.data_sources[source_name]
            
            # Step 1: Load with error handling
            try:
                raw_df = self._load_data(source)
                if raw_df.empty:
                    raise ValueError(f"Empty dataset from {source_name}")
                logger.info(f"  ✓ [LOAD] {len(raw_df)} raw records")
            except FileNotFoundError as e:
                logger.error(f"  ❌ File not found: {source.path}")
                failed_sources.append((source_name, "file_not_found"))
                continue
            except Exception as e:
                logger.error(f"  ❌ Load error: {e}")
                failed_sources.append((source_name, "load_error"))
                continue

            # Step 2: Parse
            try:
                parsed_df = self.parser.parse(raw_df, source_name)
                logger.info(f"  ✓ [PARSE] {len(parsed_df)} records")
            except Exception as e:
                logger.error(f"  ❌ Parse error: {e}")
                failed_sources.append((source_name, "parse_error"))
                continue

            # Step 3: Harmonize
            try:
                harmonized_df = self.harmonizer.harmonize(parsed_df, source_name)
                logger.info(f"  ✓ [HARMONIZE] {len(harmonized_df)} records")
            except Exception as e:
                logger.error(f"  ❌ Harmonization error: {e}")
                failed_sources.append((source_name, "harmonization_error"))
                continue

            # Step 4: Clean (optional, with fallback)
            if not skip_cleaning and self.config.enable_cleaning:
                try:
                    harmonized_df = self.cleaner.clean(harmonized_df)
                    logger.info(f"  ✓ [CLEAN] {len(harmonized_df)} records")
                except Exception as e:
                    logger.warning(f"  ⚠️ Cleaning failed, continuing: {e}")
                    # Continue without cleaning

            # Step 5: Validate
            try:
                validated_df = self.validator.validate(harmonized_df)
                logger.info(f"  ✓ [VALIDATE] {len(validated_df)} records")
            except Exception as e:
                logger.error(f"  ❌ Validation error: {e}")
                failed_sources.append((source_name, "validation_error"))
                continue

            # Step 6: Aggregate (optional)
            if not skip_aggregation and self.config.enable_aggregation:
                if 'measurement_timestamp' in validated_df.columns:
                    try:
                        validated_df = self.harmonizer.aggregate_daily(validated_df)
                        logger.info(f"  ✓ [AGGREGATE] {len(validated_df)} daily records")
                    except Exception as e:
                        logger.warning(f"  ⚠️ Aggregation failed, using raw data: {e}")

            # Store results
            self.harmonized_data[source_name] = validated_df
            successful_sources.append(source_name)
            
            # Save output
            try:
                self._save_output(validated_df, source_name)
            except Exception as e:
                logger.error(f"  ❌ Save error: {e}")
                # Continue - data is still in memory

        except Exception as e:
            logger.error(f"  ❌ Unexpected error: {e}", exc_info=True)
            failed_sources.append((source_name, "unexpected_error"))

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ HOMOGEN Pipeline Complete")
    logger.info(f"{'='*80}")
    logger.info(f"  • Successful: {len(successful_sources)}/{len(source_names)}")
    if failed_sources:
        logger.warning(f"  • Failed sources:")
        for name, reason in failed_sources:
            logger.warning(f"    - {name}: {reason}")
    logger.info(f"  • Total records: {sum(len(df) for df in self.harmonized_data.values())}")
    logger.info(f"{'='*80}\n")
    
    return self.harmonized_data


    def _load_data(self, source: DataSource) -> pd.DataFrame:
        """Load data from file based on format."""
        path = self.project_root / source.path
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        if source.format.lower() == 'csv':
            return load_csv(path, encoding=source.encoding, separator=source.separator)
        elif source.format.lower() in ['xlsx', 'xls']:
            return load_excel(path, sheet_name=source.sheet_name)
        else:
            raise ValueError(f"Unsupported file format: {source.format}")

    def _save_output(self, df: pd.DataFrame, source_name: str):
        """Save harmonized data and metadata."""
        out_dir = self.project_root / "data/harmonized"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save harmonized data
        if self.config.output_format == 'parquet':
            output_path = out_dir / f"{source_name}.parquet"
            df.to_parquet(output_path, engine="pyarrow", index=False)
        else:
            output_path = out_dir / f"{source_name}.csv"
            df.to_csv(output_path, index=False)
        
        logger.info(f"  💾 Saved: {output_path.name}")

        # Compute metadata
        geo_bbox = compute_geo_bounds(df)
        quality_stats = self._compute_quality_stats(df)
        validation_summary = self.validator.get_validation_summary(df)

        metadata = {
            "source_name": source_name,
            "record_count": len(df),
            "columns": list(df.columns),
            "processing_timestamp": datetime.now().isoformat(),
            "harmonization_version": "2.0",
            "geo_bbox": geo_bbox,
            "quality_stats": quality_stats,
            "validation_summary": validation_summary,
            "config": {
                "target_crs": self.config.target_crs,
                "temporal_resolution": self.config.temporal_resolution,
                "cleaning_enabled": self.config.enable_cleaning,
                "aggregation_enabled": self.config.enable_aggregation
            }
        }

        # Save metadata
        metadata_path = out_dir / f"{source_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  📄 Saved: {metadata_path.name}")

    def _compute_quality_stats(self, df: pd.DataFrame) -> dict:
        """Compute comprehensive quality statistics."""
        stats = {}
        
        if 'quality_score' in df.columns:
            stats['avg_quality_score'] = float(df['quality_score'].mean())
            stats['min_quality_score'] = float(df['quality_score'].min())
            stats['max_quality_score'] = float(df['quality_score'].max())
            stats['median_quality_score'] = float(df['quality_score'].median())
            stats['quality_std'] = float(df['quality_score'].std())
        
        if 'data_completeness' in df.columns:
            stats['avg_completeness'] = float(df['data_completeness'].mean())
        
        # Parameter availability
        canonical_params = ['temp_c', 'ph', 'do_mg_l', 'turbidity_ntu', 'chl_ug_l',
                           'water_level_m', 'discharge_m3s']
        present_params = [p for p in canonical_params if p in df.columns]
        stats['parameters_available'] = present_params
        stats['parameter_coverage'] = len(present_params) / len(canonical_params)
        
        return stats

    def get_summary(self) -> dict:
        """Get pipeline execution summary."""
        if not self.harmonized_data:
            return {"status": "no_data", "message": "Pipeline has not been run yet"}
        
        summary = {
            "status": "completed",
            "pipeline_version": "2.0",
            "sources_processed": len(self.harmonized_data),
            "total_records": sum(len(df) for df in self.harmonized_data.values()),
            "processing_stats": self.processing_stats,
            "sources": {}
        }
        
        for name, df in self.harmonized_data.items():
            summary["sources"][name] = {
                "records": len(df),
                "avg_quality": float(df['quality_score'].mean()) if 'quality_score' in df.columns else None,
                "avg_completeness": float(df['data_completeness'].mean()) if 'data_completeness' in df.columns else None,
                "columns": len(df.columns)
            }
        
        return summary

    def export_cube(self, output_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Export harmonized data as a unified data cube (long format).
        
        Structure: lake × time × parameter
        """
        if not self.harmonized_data:
            raise ValueError("No harmonized data available. Run pipeline first.")
        
        # Combine all sources
        all_data = pd.concat(
            [df.assign(data_source=name) for name, df in self.harmonized_data.items()],
            ignore_index=True
        )
        
        # Convert to long format
        id_cols = ['lake', 'station_id', 'measurement_timestamp', 'latitude', 
                  'longitude', 'data_source', 'quality_score']
        id_cols = [c for c in id_cols if c in all_data.columns]
        
        value_cols = ['temp_c', 'ph', 'do_mg_l', 'turbidity_ntu', 'chl_ug_l',
                     'water_level_m', 'discharge_m3s']
        value_cols = [c for c in value_cols if c in all_data.columns]
        
        cube = all_data.melt(
            id_vars=id_cols,
            value_vars=value_cols,
            var_name='parameter',
            value_name='value'
        )
        
        # Remove null values
        cube = cube.dropna(subset=['value'])
        
        if output_path:
            cube.to_parquet(output_path, engine='pyarrow', index=False)
            logger.info(f"📦 Exported data cube: {output_path}")
        
        return cube


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[3]
    
    config = HarmonizationConfig(
        enable_cleaning=True,
        enable_aggregation=True,
        quality_threshold=0.7
    )
    
    pipeline = HOMOGENPipeline(project_root, config)
    results = pipeline.run_pipeline()
    
    # Print summary
    summary = pipeline.get_summary()
    print("\n📊 Pipeline Summary:")
    print(json.dumps(summary, indent=2))
    
    # Export data cube
    cube_path = project_root / "data/harmonized/unified_cube.parquet"
    cube = pipeline.export_cube(cube_path)
    print(f"\n📦 Exported unified cube: {len(cube)} records")