# swim/agents/homogen/core_pipeline.py
# COMPLETE UPDATED VERSION - Replace your entire file with this

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
from swim.agents.homogen.utils.geo_utils import compute_geo_bounds

from swim.agents.homogen import setup_logging
logger = setup_logging()
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')


# -----------------------------------------------
# Data Classes
# -----------------------------------------------
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


# -----------------------------------------------
# Pipeline Class
# -----------------------------------------------
class HOMOGENPipeline:
    """
    HOMOGEN Harmonization Pipeline:
    ingestion → parsing → harmonization → cleaning → validation → aggregation → export
    """

    def __init__(self, project_root: Path, config: Optional[HarmonizationConfig] = None):
        self.project_root = project_root
        self.config = config or HarmonizationConfig()

        # Load configs
        self.data_sources_config = load_config(project_root / "swim/agents/homogen/config/data_sources.yaml")
        self.harmonization_rules = load_config(project_root / "swim/agents/homogen/config/harmonization_rules.yaml")
        self.global_settings = load_config(project_root / "config/settings/development.yaml")

        # Source definitions
        self.data_sources = {
            name: DataSource(**info)
            for name, info in self.data_sources_config["data_sources"].items()
        }

        # Metadata (optional)
        meta_path = project_root / "data/harmonized/gemstat_metadata.parquet"
        self.metadata_df = pd.read_parquet(meta_path) if meta_path.exists() else pd.DataFrame()

        # Modules
        self.parser = DataParser(self.harmonization_rules.get("parameter_mappings", {}), self.metadata_df)
        self.harmonizer = DataHarmonizer(
            config=self.config,
            parameter_mappings=self.harmonization_rules.get("parameter_mappings", {}),
            unit_conversions=self.harmonization_rules.get("unit_conversions", {})
        )
        self.cleaner = DataCleaner(config={'knn_neighbors': 5, 'seasonal_period': 365})
        self.validator = DataValidator()

        self.harmonized_data = {}
        self.processing_stats = {}

    # -----------------------------------------------------
    def _should_skip_cleaning(self, df: pd.DataFrame) -> bool:
        """Determine if cleaning should be skipped for this dataframe."""
        # Skip if it's metadata
        if len(df) > 0:
            data_type = df.get('data_type', pd.Series(['unknown'])).iloc[0]
            if data_type == 'metadata':
                return True
        
        # Skip if no numeric water quality columns
        measurement_cols = [
            "temp_c", "ph", "do_mg_l", "turbidity_ntu",
            "chl_ug_l", "water_level_m", "discharge_m3s"
        ]
        has_measurements = any(
            col in df.columns and df[col].notna().any() 
            for col in measurement_cols
        )
        
        return not has_measurements

    def _should_skip_aggregation(self, df: pd.DataFrame) -> bool:
        """Determine if aggregation should be skipped."""
        # Skip if no valid timestamps
        if 'measurement_timestamp' not in df.columns:
            return True
        
        valid_timestamps = df['measurement_timestamp'].notna().sum()
        if valid_timestamps == 0:
            logger.info("Skipping aggregation: no valid timestamps")
            return True
        
        # Skip if it's metadata
        if len(df) > 0:
            data_type = df.get('data_type', pd.Series(['unknown'])).iloc[0]
            if data_type == 'metadata':
                return True
        
        return False

    # -----------------------------------------------------
    def run_pipeline(self, source_names: Optional[List[str]] = None,
                     skip_cleaning: bool = False, skip_aggregation: bool = False) -> Dict[str, pd.DataFrame]:

        if source_names is None:
            source_names = list(self.data_sources.keys())

        logger.info(f"\n{'='*80}\n🌊 HOMOGEN Pipeline v2.0 - Processing {len(source_names)} sources\n{'='*80}")

        successful_sources, failed_sources = [], []

        for source_name in source_names:
            logger.info(f"\n{'─'*80}\n📦 Processing: {source_name}\n{'─'*80}")
            try:
                source = self.data_sources[source_name]

                # Step 1 — Load
                try:
                    raw_df = self._load_data(source)
                    if raw_df.empty:
                        raise ValueError(f"Empty dataset from {source_name}")
                    logger.info(f"  ✓ [LOAD] {len(raw_df)} raw records")
                except Exception as e:
                    logger.error(f"  ❌ Load error: {e}")
                    failed_sources.append((source_name, "load_error"))
                    continue

                # Step 2 — Parse
                try:
                    parsed_df = self.parser.parse(raw_df, source_name)
                    logger.info(f"  ✓ [PARSE] {len(parsed_df)} records")
                except Exception as e:
                    logger.error(f"  ❌ Parse error: {e}")
                    failed_sources.append((source_name, "parse_error"))
                    continue

                # Step 3 — Harmonize
                try:
                    harmonized_df = self.harmonizer.harmonize(parsed_df, source_name)
                    logger.info(f"  ✓ [HARMONIZE] {len(harmonized_df)} records")
                except Exception as e:
                    logger.error(f"  ❌ Harmonization error: {e}")
                    failed_sources.append((source_name, "harmonization_error"))
                    continue

                # Step 4 — Cleaning (with smart skip logic)
                should_skip_cleaning = self._should_skip_cleaning(harmonized_df)
                if not skip_cleaning and self.config.enable_cleaning and not should_skip_cleaning:
                    try:
                        harmonized_df = self.cleaner.clean(harmonized_df)
                        logger.info(f"  ✓ [CLEAN] {len(harmonized_df)} records")
                    except Exception as e:
                        logger.warning(f"  ⚠️ Cleaning failed, continuing: {e}")
                else:
                    if should_skip_cleaning:
                        logger.info(f"  ⊘ [CLEAN] Skipped (metadata or no measurements)")
                    else:
                        logger.info(f"  ⊘ [CLEAN] Skipped by configuration")

                # Step 5 — Validation
                try:
                    validated_df = self.validator.validate(harmonized_df)
                    logger.info(f"  ✓ [VALIDATE] {len(validated_df)} records")
                except Exception as e:
                    logger.error(f"  ❌ Validation error: {e}")
                    failed_sources.append((source_name, "validation_error"))
                    continue

                # Step 6 — Aggregation (with smart skip logic)
                should_skip_agg = self._should_skip_aggregation(validated_df)
                if not skip_aggregation and self.config.enable_aggregation and not should_skip_agg:
                    try:
                        validated_df = self.harmonizer.aggregate_daily(validated_df)
                        logger.info(f"  ✓ [AGGREGATE] {len(validated_df)} daily records")
                    except Exception as e:
                        logger.warning(f"  ⚠️ Aggregation failed: {e}")
                else:
                    if should_skip_agg:
                        logger.info(f"  ⊘ [AGGREGATE] Skipped (no timestamps or metadata)")
                    else:
                        logger.info(f"  ⊘ [AGGREGATE] Skipped by configuration")

                # Save
                self.harmonized_data[source_name] = validated_df
                successful_sources.append(source_name)

                # Step 7 — Save Output
                try:
                    self._save_output(validated_df, source_name)
                except Exception as e:
                    logger.error(f"  ❌ Save error: {e}")

            except Exception as e:
                logger.error(f"  ❌ Unexpected error: {e}", exc_info=True)
                failed_sources.append((source_name, "unexpected_error"))

        # Summary
        logger.info(f"\n{'='*80}\n✅ HOMOGEN Pipeline Complete\n{'='*80}")
        logger.info(f"  • Successful: {len(successful_sources)}/{len(source_names)}")
        if failed_sources:
            logger.warning("  • Failed sources:")
            for name, reason in failed_sources:
                logger.warning(f"    - {name}: {reason}")
        
        total_records = sum(len(df) for df in self.harmonized_data.values())
        logger.info(f"  • Total records: {total_records:,}\n{'='*80}")

        return self.harmonized_data

    # -----------------------------------------------------
    def _load_data(self, source: DataSource) -> pd.DataFrame:
        path = self.project_root / source.path
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        fmt = source.format.lower()
        if fmt == "csv":
            return load_csv(path, encoding=source.encoding, separator=source.separator)
        elif fmt in ["xlsx", "xls"]:
            return load_excel(path, sheet_name=source.sheet_name)
        elif fmt == "parquet":
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {source.format}")

    def _save_output(self, df: pd.DataFrame, source_name: str):
        out_dir = self.project_root / "data/harmonized"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save harmonized file
        data_path = out_dir / f"{source_name}.{self.config.output_format}"
        if self.config.output_format == "parquet":
            df_to_save = self._prepare_for_parquet(df)
            try:
                df_to_save.to_parquet(data_path, index=False)
            except Exception as e:
                logger.warning(f"  ⚠️ Parquet save failed ({e}); falling back to CSV")
                data_path = out_dir / f"{source_name}.csv"
                df_to_save.to_csv(data_path, index=False)
        else:
            df.to_csv(data_path, index=False)
        logger.info(f"  💾 Saved: {data_path.name}")

        # Save metadata (with serialization-safe values)
        metadata = {
            "source_name": source_name,
            "record_count": int(len(df)),
            "columns": list(df.columns),
            "processing_timestamp": datetime.now().isoformat(),
            "geo_bbox": compute_geo_bounds(df),
            "quality_stats": self._compute_quality_stats(df),
            "validation_summary": self.validator.get_validation_summary(df),
            "harmonization_version": "2.0"
        }
        # Fix: make everything JSON-serializable
        metadata = json.loads(json.dumps(metadata, default=str))

        meta_path = out_dir / f"{source_name}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  📄 Saved: {meta_path.name}")

    def _prepare_for_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure dataframe can be safely serialized to parquet."""
        if df.empty:
            return df

        df_out = df.copy()
        for col in df_out.columns:
            if pd.api.types.is_object_dtype(df_out[col]) or pd.api.types.is_string_dtype(df_out[col]):
                df_out[col] = df_out[col].astype("string")
        return df_out

    def _compute_quality_stats(self, df: pd.DataFrame) -> dict:
        stats = {}
        if "quality_score" in df.columns:
            # Handle potential NaN values safely
            quality_values = df["quality_score"].dropna()
            if len(quality_values) > 0:
                stats = {
                    "avg_quality_score": float(quality_values.mean()),
                    "min_quality_score": float(quality_values.min()),
                    "max_quality_score": float(quality_values.max()),
                    "median_quality_score": float(quality_values.median()),
                    "quality_std": float(quality_values.std()) if len(quality_values) > 1 else 0.0,
                }
            else:
                stats = {
                    "avg_quality_score": None,
                    "min_quality_score": None,
                    "max_quality_score": None,
                    "median_quality_score": None,
                    "quality_std": None,
                }
        return stats

    def get_summary(self) -> dict:
        if not self.harmonized_data:
            return {"status": "no_data", "message": "Pipeline has not been run yet"}
        
        summary = {
            "status": "completed",
            "pipeline_version": "2.0",
            "sources_processed": len(self.harmonized_data),
            "total_records": sum(len(df) for df in self.harmonized_data.values()),
            "sources": {},
        }
        
        for name, df in self.harmonized_data.items():
            avg_quality = None
            if "quality_score" in df.columns:
                quality_values = df["quality_score"].dropna()
                if len(quality_values) > 0:
                    avg_quality = float(quality_values.mean())
            
            summary["sources"][name] = {
                "records": len(df),
                "avg_quality": avg_quality,
                "columns": len(df.columns)
            }
        
        return summary

    def export_cube(self, output_path: Optional[Path] = None) -> pd.DataFrame:
        if not self.harmonized_data:
            raise ValueError("No harmonized data available. Run pipeline first.")

        all_data = pd.concat(
            [df.assign(data_source=name) for name, df in self.harmonized_data.items()],
            ignore_index=True
        )

        id_cols = ["lake", "station_id", "measurement_timestamp", "latitude", "longitude", "data_source", "quality_score"]
        id_cols = [c for c in id_cols if c in all_data.columns]

        value_cols = ["temp_c", "ph", "do_mg_l", "turbidity_ntu", "chl_ug_l", "water_level_m", "discharge_m3s"]
        value_cols = [c for c in value_cols if c in all_data.columns]

        if not value_cols:
            logger.warning("No value columns found for cube export")
            return all_data

        cube = all_data.melt(id_vars=id_cols, value_vars=value_cols, var_name="parameter", value_name="value")
        cube = cube.dropna(subset=["value"])

        if output_path:
            cube.to_parquet(output_path, engine="pyarrow", index=False)
            logger.info(f"📦 Exported data cube: {output_path}")

        return cube


# CLI execution
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    pipeline = HOMOGENPipeline(project_root)
    results = pipeline.run_pipeline()
    print("\n📊 Pipeline Summary:")
    print(json.dumps(pipeline.get_summary(), indent=2))

    cube_path = project_root / "data/harmonized/unified_cube.parquet"
    cube = pipeline.export_cube(cube_path)
    print(f"\n📦 Exported unified cube: {len(cube)} records")
