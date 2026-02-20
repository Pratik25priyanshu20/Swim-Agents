#swim/agents/homogen/tools.py

from langchain.tools import tool
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import traceback
from datetime import datetime, timezone

from swim.agents.homogen.utils.geo_utils import compute_geo_bounds 
from swim.agents.homogen.core_pipeline import HOMOGENPipeline
from swim.agents.homogen import setup_logging

logger = setup_logging()

# ========================
# UTILITY FUNCTIONS
# ========================
def _get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).resolve().parents[3]


def _get_harmonized_dir() -> Path:
    """Get harmonized data directory."""
    return _get_project_root() / "data/harmonized"


def _safe_load_parquet(filepath: Path) -> tuple[pd.DataFrame, str]:
    """Safely load parquet file with error handling."""
    try:
        if not filepath.exists():
            return None, f"❌ File not found: {filepath.name}"
        
        df = pd.read_parquet(filepath)
        if df.empty:
            return None, f"⚠️ File is empty: {filepath.name}"
        
        return df, None
    except Exception as e:
        return None, f"❌ Error loading {filepath.name}: {str(e)}"


# ========================
# PIPELINE TOOLS
# ========================
@tool
def run_homogen_pipeline(source_names: str = "") -> str:
    """
    Run the HOMOGEN harmonization pipeline on specified data sources.
    
    Args:
        source_names: Comma-separated list of source names, or empty for all sources
    
    Returns:
        Pipeline execution summary with success/failure details
    """
    try:
        project_root = _get_project_root()
        pipeline = HOMOGENPipeline(project_root)

        sources = [s.strip() for s in source_names.split(",")] if source_names else None
        
        logger.info(f"Starting pipeline for sources: {sources or 'all'}")
        results = pipeline.run_pipeline(source_names=sources)

        summary = pipeline.get_summary()
        
        # Format response
        response = "✅ HOMOGEN Pipeline Completed\n\n"
        response += f"📊 **Summary:**\n"
        response += f"  • Sources processed: {summary['sources_processed']}\n"
        response += f"  • Total records: {summary['total_records']:,}\n"
        response += f"  • Pipeline version: {summary['pipeline_version']}\n\n"
        
        if summary['total_records'] == 0:
            response += "⚠️ **Warning:** No records were successfully harmonized\n"
            response += "💡 **Suggestion:** Check source data quality and availability\n"
        else:
            response += "**Source Details:**\n"
            for name, info in summary['sources'].items():
                avg_q = info.get('avg_quality')
                quality_str = f"{avg_q:.2f}" if avg_q and not np.isnan(avg_q) else "N/A"
                response += f"  • {name}: {info['records']:,} records (Quality: {quality_str})\n"
        
        return response

    except Exception as e:
        logger.error("Pipeline execution failed", exc_info=True)
        return f"❌ Pipeline failed: {str(e)}\n💡 Check logs for details:\n{traceback.format_exc()[:500]}"


# ========================
# DATA EXPLORATION TOOLS
# ========================
@tool
def list_available_lakes(input: str = "") -> str:
    """
    List all unique lake names in the harmonized dataset.
    
    Returns:
        Formatted list of available lakes
    """
    try:
        filepath = _get_harmonized_dir() / "lake_samples.parquet"
        df, error = _safe_load_parquet(filepath)
        
        if error:
            return f"{error}\n💡 Run the pipeline first: run_homogen_pipeline()"
        
        if "lake" not in df.columns:
            return "❌ 'lake' column not found in data"
        
        lakes = df["lake"].dropna().str.strip().str.title().unique()
        if len(lakes) == 0:
            return "⚠️ No lakes found in dataset"
        
        response = f"🗺️ **Available Lakes ({len(lakes)}):**\n"
        for i, lake in enumerate(sorted(lakes), 1):
            response += f"  {i}. {lake}\n"
        
        return response
        
    except Exception as e:
        logger.error(f"Error listing lakes: {e}", exc_info=True)
        return f"❌ Failed to list lakes: {str(e)}"


@tool
def get_lake_quality(lake_name: str, parameter: str = None, year: int = None) -> str:
    """
    Get water quality statistics for a specific lake.
    
    Args:
        lake_name: Name of the lake (case-insensitive)
        parameter: Optional specific parameter (e.g., 'temp_c', 'ph', 'turbidity_ntu')
        year: Optional year filter
    
    Returns:
        Formatted water quality statistics and interpretation
    """
    try:
        filepath = _get_harmonized_dir() / "lake_samples.parquet"
        df, error = _safe_load_parquet(filepath)
        
        if error:
            return error
        
        if 'lake' not in df.columns:
            return "❌ 'lake' column not found"
        
        # Filter by lake
        match = df['lake'].dropna().str.lower().str.strip() == lake_name.lower().strip()
        filtered = df[match]
        
        if filtered.empty:
            available = df['lake'].dropna().unique()[:5]
            return (f"❌ No data for lake '{lake_name}'\n"
                   f"💡 Try one of these: {', '.join(available)}")
        
        # Apply year filter
        if year and 'measurement_timestamp' in filtered.columns:
            filtered['year'] = pd.to_datetime(filtered['measurement_timestamp']).dt.year
            filtered = filtered[filtered['year'] == year]
            if filtered.empty:
                return f"❌ No data for {lake_name} in {year}"
        
        # Define available parameters
        params = {
            'temp_c': 'Temperature (°C)',
            'ph': 'pH',
            'do_mg_l': 'Dissolved Oxygen (mg/L)',
            'turbidity_ntu': 'Turbidity (NTU)',
            'chl_ug_l': 'Chlorophyll-a (μg/L)',
            'water_level_m': 'Water Level (m)'
        }
        
        # Single parameter query
        if parameter:
            param_col = parameter.lower().strip()
            if param_col not in params:
                return f"❌ Parameter '{parameter}' not found\n💡 Available: {', '.join(params.keys())}"
            
            if param_col not in filtered.columns or filtered[param_col].isna().all():
                return f"⚠️ No {params[param_col]} data available for {lake_name}"
            
            stats = filtered[param_col].describe()
            
            # Interpret results
            response = f"📊 **{params[param_col]} at {lake_name.title()}**"
            if year:
                response += f" ({year})"
            response += f"\n\n"
            response += f"  • Count: {int(stats['count'])} measurements\n"
            response += f"  • Mean: {stats['mean']:.2f}\n"
            response += f"  • Range: {stats['min']:.2f} - {stats['max']:.2f}\n"
            response += f"  • Std Dev: {stats['std']:.2f}\n\n"
            
            # Add interpretation
            if param_col == 'temp_c':
                if stats['mean'] > 25:
                    response += "🔴 **Alert:** High temperature may stress aquatic life\n"
                elif stats['mean'] < 10:
                    response += "🔵 **Note:** Cold water temperatures\n"
            elif param_col == 'ph':
                if stats['mean'] < 6.5 or stats['mean'] > 8.5:
                    response += "⚠️ **Alert:** pH outside optimal range (6.5-8.5)\n"
            elif param_col == 'do_mg_l':
                if stats['mean'] < 5:
                    response += "🔴 **Alert:** Low dissolved oxygen - risk to aquatic life\n"
            elif param_col == 'chl_ug_l':
                if stats['mean'] > 30:
                    response += "🟢 **Alert:** High chlorophyll - potential algal bloom\n"
            
            return response
        
        # Multi-parameter summary
        response = f"🌊 **Water Quality Summary: {lake_name.title()}**"
        if year:
            response += f" ({year})"
        response += f"\n\n"
        response += f"📍 Total measurements: {len(filtered):,}\n\n"
        
        for param_col, param_name in params.items():
            if param_col in filtered.columns and filtered[param_col].notna().any():
                stats = filtered[param_col].describe()
                response += f"**{param_name}:**\n"
                response += f"  Mean: {stats['mean']:.2f}, Range: [{stats['min']:.2f}, {stats['max']:.2f}]\n\n"
        
        # Overall quality assessment
        if 'quality_score' in filtered.columns:
            avg_quality = filtered['quality_score'].mean()
            response += f"**Data Quality Score:** {avg_quality:.2f}/1.00\n"
            if avg_quality > 0.8:
                response += "✅ High quality data\n"
            elif avg_quality > 0.6:
                response += "⚠️ Moderate quality data\n"
            else:
                response += "❌ Low quality data - use with caution\n"
        
        return response
        
    except Exception as e:
        logger.error(f"Error in get_lake_quality: {e}", exc_info=True)
        return f"❌ Failed to retrieve lake quality: {str(e)}"


@tool
def get_habs_summary(lake_name: str) -> str:
    """
    Get Harmful Algal Bloom (HAB) indicators for a lake.
    
    Args:
        lake_name: Name of the lake
    
    Returns:
        HAB risk assessment and chlorophyll-a analysis
    """
    try:
        filepath = _get_harmonized_dir() / "lake_samples.parquet"
        df, error = _safe_load_parquet(filepath)
        
        if error:
            return error
        
        filtered = df[df["lake"].dropna().str.lower().str.strip() == lake_name.lower().strip()]
        
        if filtered.empty:
            return f"❌ No data for lake '{lake_name}'"
        
        response = f"🦠 **HAB Risk Assessment: {lake_name.title()}**\n\n"
        
        # Chlorophyll-a analysis (primary HAB indicator)
        if "chl_ug_l" in filtered.columns and filtered["chl_ug_l"].notna().any():
            chl_stats = filtered["chl_ug_l"].describe()
            chl_mean = chl_stats['mean']
            chl_max = chl_stats['max']
            
            response += f"**Chlorophyll-a Levels:**\n"
            response += f"  • Mean: {chl_mean:.2f} μg/L\n"
            response += f"  • Max: {chl_max:.2f} μg/L\n"
            response += f"  • Measurements: {int(chl_stats['count'])}\n\n"
            
            # WHO Guidelines interpretation
            if chl_max > 100:
                risk = "🔴 **CRITICAL**"
                advice = "High risk of toxic blooms. Avoid water contact."
            elif chl_max > 50:
                risk = "🟠 **HIGH**"
                advice = "Moderate bloom risk. Monitor closely."
            elif chl_max > 30:
                risk = "🟡 **MODERATE**"
                advice = "Elevated levels. Watch for visible blooms."
            elif chl_max > 10:
                risk = "🟢 **LOW**"
                advice = "Normal levels. No immediate concern."
            else:
                risk = "✅ **MINIMAL**"
                advice = "Chlorophyll levels are safe."
            
            response += f"**HAB Risk Level:** {risk}\n"
            response += f"💡 **Recommendation:** {advice}\n\n"
        else:
            response += "⚠️ No chlorophyll-a data available\n\n"
        
        # Additional indicators
        indicators = {
            "temp_c": ("Temperature", "°C"),
            "ph": ("pH", ""),
            "turbidity_ntu": ("Turbidity", "NTU")
        }
        
        response += "**Supporting Indicators:**\n"
        for col, (name, unit) in indicators.items():
            if col in filtered.columns and filtered[col].notna().any():
                mean_val = filtered[col].mean()
                response += f"  • {name}: {mean_val:.2f}{unit}\n"
        
        return response
        
    except Exception as e:
        logger.error(f"Error in get_habs_summary: {e}", exc_info=True)
        return f"❌ Failed to get HAB summary: {str(e)}"


# ========================
# DATA QUALITY TOOLS
# ========================
@tool
def summarize_harmonized_data(input: str = "") -> str:
    """
    Summarize all harmonized datasets with metadata.
    
    Returns:
        Summary of record counts, quality scores, and data sources
    """
    try:
        output_dir = _get_harmonized_dir()
        
        if not output_dir.exists():
            return "❌ Harmonized data directory not found\n💡 Run pipeline first"
        
        summaries = []
        total_records = 0
        
        for file in output_dir.glob("*_metadata.json"):
            try:
                with open(file) as f:
                    meta = json.load(f)
                    
                    records = meta.get('record_count', 0)
                    total_records += records
                    
                    avg_quality = meta.get("quality_stats", {}).get("avg_quality_score")
                    quality_str = f"{avg_quality:.2f}" if avg_quality else "N/A"
                    
                    summary = f"  • **{meta['source_name']}**: {records:,} records (Quality: {quality_str})"
                    summaries.append(summary)
            except Exception as e:
                logger.warning(f"Failed to read {file.name}: {e}")
        
        if not summaries:
            return "⚠️ No metadata files found\n💡 Run the pipeline to generate data"
        
        response = f"📦 **Harmonized Data Summary**\n\n"
        response += f"Total Records: {total_records:,}\n"
        response += f"Data Sources: {len(summaries)}\n\n"
        response += "**Sources:**\n"
        response += "\n".join(summaries)
        
        return response
        
    except Exception as e:
        logger.error(f"Error summarizing data: {e}", exc_info=True)
        return f"❌ Failed to summarize data: {str(e)}"


@tool
def detect_outliers(filepath: str, parameter: str) -> str:
    """
    Detect statistical outliers in a water quality parameter.
    
    Args:
        filepath: Path to CSV file
        parameter: Column name to analyze
    
    Returns:
        Outlier detection summary with statistics
    """
    try:
        df = pd.read_csv(filepath)
        
        if parameter not in df.columns:
            available = [c for c in df.columns if c in ['temp_c', 'ph', 'do_mg_l', 'turbidity_ntu', 'chl_ug_l']]
            return f"❌ Parameter '{parameter}' not found\n💡 Try: {', '.join(available)}"
        
        values = df[parameter].dropna()
        
        if len(values) < 5:
            return f"⚠️ Insufficient data ({len(values)} values) for outlier detection"
        
        # MAD-based Z-scores
        median = values.median()
        mad = np.median(np.abs(values - median))
        
        if mad == 0:
            return f"⚠️ Cannot detect outliers (MAD = 0)"
        
        z_scores = 0.6745 * (values - median) / mad
        outliers = np.abs(z_scores) > 5.0
        n_outliers = outliers.sum()
        
        response = f"🔍 **Outlier Detection: {parameter}**\n\n"
        response += f"  • Total values: {len(values)}\n"
        response += f"  • Median: {median:.2f}\n"
        response += f"  • MAD: {mad:.2f}\n"
        response += f"  • Outliers detected: {n_outliers} ({n_outliers/len(values)*100:.1f}%)\n\n"
        
        if n_outliers == 0:
            response += "✅ No significant outliers detected"
        else:
            outlier_vals = values[outliers].values[:10]
            response += f"**Outlier values (sample):**\n"
            response += f"  {[round(v, 2) for v in outlier_vals]}\n\n"
            response += "💡 **Suggestion:** Review these values for measurement errors"
        
        return response
        
    except Exception as e:
        logger.error(f"Outlier detection failed: {e}", exc_info=True)
        return f"❌ Outlier detection failed: {str(e)}"


# ========================
# FILE OPERATION TOOLS
# ========================
@tool
def load_csv(filepath: str) -> str:
    """Load and inspect a CSV file."""
    try:
        df = pd.read_csv(filepath)
        return f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns from {Path(filepath).name}"
    except Exception as e:
        return f"❌ Error loading CSV: {str(e)}"


@tool
def show_columns(filepath: str) -> str:
    """Display column names and types from a CSV file."""
    try:
        df = pd.read_csv(filepath, nrows=5)
        response = f"📋 **Columns in {Path(filepath).name}:**\n\n"
        for col in df.columns:
            dtype = df[col].dtype
            response += f"  • {col} ({dtype})\n"
        return response
    except Exception as e:
        return f"❌ Error reading columns: {str(e)}"


@tool
def validate_sample(filepath: str) -> str:
    """Check data quality and missing values in a CSV."""
    try:
        df = pd.read_csv(filepath)
        missing = df.isna().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        response = f"🔍 **Data Quality Report: {Path(filepath).name}**\n\n"
        response += f"Total rows: {len(df):,}\n"
        response += f"Total columns: {len(df.columns)}\n\n"
        
        if missing.sum() == 0:
            response += "✅ No missing values detected!"
        else:
            response += "**Missing Values:**\n"
            for col in missing.index:
                if missing[col] > 0:
                    response += f"  • {col}: {missing[col]} ({missing_pct[col]}%)\n"
        
        return response
    except Exception as e:
        return f"❌ Validation error: {str(e)}"


@tool
def compute_bbox(filepath: str) -> str:
    """Compute geographic bounding box for a CSV dataset."""
    try:
        df = pd.read_csv(filepath)
        
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return "❌ Dataset missing latitude/longitude columns"
        
        bbox = compute_geo_bounds(df)
        
        response = "📍 **Geographic Bounding Box:**\n\n"
        response += f"  • Latitude: [{bbox['min_latitude']:.4f}, {bbox['max_latitude']:.4f}]\n"
        response += f"  • Longitude: [{bbox['min_longitude']:.4f}, {bbox['max_longitude']:.4f}]\n"
        response += f"  • BBOX: {bbox['bbox']}\n"
        
        return response
    except Exception as e:
        return f"❌ Error computing bbox: {str(e)}"