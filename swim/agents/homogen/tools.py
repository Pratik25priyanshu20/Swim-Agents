# swim/agents/homogen/tools.py

from langchain.tools import tool
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime, timezone

from swim.agents.homogen import setup_logging
logger = setup_logging()

# ========================
# CANONICAL MAPPING & GUARDS
# ========================
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
    "water_level_m_NHN": ("water_level_m_nhn", float),
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
    "water_level_m_nhn": (-300.0, 1000.0),
    "discharge_m3s": (0.0, 100000.0),
    "chl_ug_l": (0.0, 1000.0),
}

# ========================
# UTILITY FUNCTIONS
# ========================
def compute_geo_bounds(df: pd.DataFrame) -> dict:
    """Compute bounding box (min/max lat/lon) for a dataset."""
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return {}
    return {
        "min_latitude": float(df["latitude"].min()),
        "max_latitude": float(df["latitude"].max()),
        "min_longitude": float(df["longitude"].min()),
        "max_longitude": float(df["longitude"].max()),
        "bbox": f"[{df['latitude'].min()}, {df['longitude'].min()}, {df['latitude'].max()}, {df['longitude'].max()}]"
    }

def mad_zscores(values):
    """Compute Modified Z-scores using Median Absolute Deviation."""
    x = [float(v) for v in values if isinstance(v, (int, float))]
    if len(x) < 5:
        return [0.0 for _ in values]
    med = np.median(x)
    mad = np.median(np.abs(np.array(x) - med)) or 1e-9
    zscores = []
    for v in values:
        if isinstance(v, (int, float)):
            zscores.append(0.6745 * (v - med) / mad)
        else:
            zscores.append(np.nan)
    return zscores

def apply_range_guards(df: pd.DataFrame) -> pd.DataFrame:
    """Apply range guards to canonical parameters."""
    for param, (min_val, max_val) in RANGE_GUARDS.items():
        if param in df.columns:
            mask = (df[param] < min_val) | (df[param] > max_val)
            if mask.any():
                logger.warning(f"Removing {mask.sum()} out-of-range values for {param}")
                df.loc[mask, param] = np.nan
    return df

# ========================
# LANGGRAPH TOOLS
# ========================

@tool
def run_homogen_pipeline(source_names: str = "") -> str:
    """
    Execute the full HOMOGEN harmonization pipeline.
    
    Args:
        source_names: Comma-separated list of source names (e.g., "samples,gemstat_metadata"). 
                     Leave empty to process all sources.
    
    Returns:
        Status message with processing summary.
    """
    try:
        from swim.agents.homogen.core_pipeline import HOMOGENPipeline
        from pathlib import Path
        
        project_root = Path(__file__).resolve().parents[3]
        pipeline = HOMOGENPipeline(project_root)
        
        sources = [s.strip() for s in source_names.split(",")] if source_names else None
        results = pipeline.run_pipeline(source_names=sources)
        
        summary = f"✅ Pipeline completed. Processed {len(results)} sources:\n"
        for name, df in results.items():
            summary += f"  • {name}: {len(df)} records\n"
        
        return summary
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return f"❌ Pipeline failed: {str(e)}"


@tool
def load_csv(filepath: str) -> str:
    """Load a CSV file and return the number of rows loaded."""
    try:
        df = pd.read_csv(filepath)
        return f"✅ Loaded {len(df)} rows from {filepath}"
    except Exception as e:
        return f"❌ Error loading CSV: {str(e)}"


@tool
def show_columns(filepath: str) -> str:
    """Display the column names of the given CSV file."""
    try:
        df = pd.read_csv(filepath)
        return f"📋 Columns: {', '.join(df.columns.tolist())}"
    except Exception as e:
        return f"❌ Error reading columns: {str(e)}"


@tool
def validate_sample(filepath: str) -> str:
    """Check and report missing values per column in a CSV."""
    try:
        df = pd.read_csv(filepath)
        missing = df.isna().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        result = "🔍 Data Quality Report:\n"
        for col in missing.index:
            if missing[col] > 0:
                result += f"  • {col}: {missing[col]} missing ({missing_pct[col]}%)\n"
        
        if missing.sum() == 0:
            result = "✅ No missing values detected!"
        
        return result
    except Exception as e:
        return f"❌ Validation error: {str(e)}"


@tool
def compute_bbox(filepath: str) -> str:
    """Compute the bounding box (min/max lat/lon) for a CSV dataset."""
    try:
        df = pd.read_csv(filepath)
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return "❌ Dataset missing latitude/longitude columns."
        
        bbox = compute_geo_bounds(df)
        return f"📍 Bounding Box:\n  Lat: [{bbox['min_latitude']:.4f}, {bbox['max_latitude']:.4f}]\n  Lon: [{bbox['min_longitude']:.4f}, {bbox['max_longitude']:.4f}]"
    except Exception as e:
        return f"❌ Error computing bbox: {str(e)}"


@tool
def summarize_harmonized_data(input: str = "") -> str:
    """Summarize record count and quality score from harmonized metadata JSON files."""
    try:
        output_dir = Path(__file__).resolve().parents[3] / "data/harmonized"
        if not output_dir.exists():
            return "❌ Harmonized data directory not found. Run pipeline first."
        
        summaries = []
        for file in output_dir.glob("*_metadata.json"):
            with open(file) as f:
                meta = json.load(f)
                avg_quality = meta.get("quality_stats", {}).get("avg_quality_score")
                summary = f"📦 {meta['source_name']}: {meta['record_count']} records"
                if avg_quality is not None:
                    summary += f", Quality: {avg_quality:.2f}"
                summaries.append(summary)
        
        return "\n".join(summaries) if summaries else "ℹ️ No metadata files found."
    except Exception as e:
        return f"❌ Error reading metadata: {str(e)}"


@tool
def list_available_lakes(input: str = "") -> str:
    """List all unique lake names available in the harmonized data."""
    try:
        path = Path(__file__).resolve().parents[3] / "data/harmonized/lake_samples.parquet"
        if not path.exists():
            return "❌ lake_samples.parquet not found. Run pipeline first."
        
        df = pd.read_parquet(path)
        if "lake" not in df.columns:
            return "❌ 'lake' column missing in data."
        
        lakes = df["lake"].dropna().str.strip().str.title().unique().tolist()
        if not lakes:
            return "ℹ️ No lakes found in the dataset."
        
        return "🗺️ Available Lakes:\n" + "\n".join(f"  • {lake}" for lake in sorted(lakes))
    except Exception as e:
        return f"❌ Error listing lakes: {str(e)}"


@tool
def get_lake_quality(lake_name: str, parameter: str = None, year: int = None) -> str:
    """
    Returns water quality statistics for a given lake.
    
    Args:
        lake_name: Name of the lake (case-insensitive)
        parameter: Optional specific parameter (e.g., 'turbidity', 'ph')
        year: Optional year filter
    
    Returns:
        Formatted statistics or error message
    """
    try:
        path = Path(__file__).resolve().parents[3] / "data/harmonized/lake_samples.parquet"
        if not path.exists():
            return "❌ lake_samples.parquet not found."
        
        df = pd.read_parquet(path)
        if 'lake' not in df.columns:
            return "❌ 'lake' column not found in data."
        
        # Filter by lake name
        match = df['lake'].dropna().str.lower().str.strip() == lake_name.lower().strip()
        filtered = df[match]
        
        if year:
            filtered = filtered[filtered.get('year', 0) == year]
        
        if filtered.empty:
            return f"❌ No data found for lake '{lake_name}'" + (f" in year {year}." if year else ".")
        
        # Standard numeric columns
        numeric_cols = ['temperature', 'temp_c', 'turbidity', 'turbidity_ntu', 'ph', 
                       'do_mg_l', 'chl_ug_l', 'quality_score']
        
        if parameter:
            param_lower = parameter.strip().lower()
            matching_col = next((col for col in filtered.columns if col.lower() == param_lower), None)
            
            if not matching_col:
                return f"❌ Parameter '{parameter}' not found. Available: {', '.join(numeric_cols)}"
            
            stats = filtered[matching_col].describe()
            return (f"📊 {matching_col.replace('_', ' ').title()} at {lake_name.title()}" + 
                   (f" (Year {year})" if year else "") + ":\n" +
                   f"  Count: {int(stats['count'])}\n" +
                   f"  Mean: {stats['mean']:.2f}\n" +
                   f"  Min: {stats['min']:.2f}\n" +
                   f"  Max: {stats['max']:.2f}\n" +
                   f"  Std: {stats['std']:.2f}")
        else:
            result = f"📍 Water Quality Summary for {lake_name.title()}"
            if year:
                result += f" (Year: {year})"
            result += ":\n"
            
            for col in numeric_cols:
                if col in filtered.columns and filtered[col].notna().any():
                    stats = filtered[col].describe()
                    result += (f"  • {col.replace('_', ' ').title()}: "
                             f"Mean={stats['mean']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}\n")
            
            return result.strip()
    except Exception as e:
        logger.error(f"Error in get_lake_quality: {e}", exc_info=True)
        return f"❌ Error retrieving lake quality: {str(e)}"


@tool
def get_habs_summary(lake_name: str) -> str:
    """
    Summarizes Harmful Algal Bloom (HABs) indicators for a given lake.
    
    Args:
        lake_name: Name of the lake (case-insensitive)
    
    Returns:
        HABs statistics including bloom probability, cyanobacteria density, toxin levels
    """
    try:
        path = Path(__file__).resolve().parents[3] / "data/harmonized/lake_samples.parquet"
        if not path.exists():
            return "❌ lake_samples.parquet not found."
        
        df = pd.read_parquet(path)
        filtered = df[df["lake"].dropna().str.lower().str.strip() == lake_name.lower().strip()]
        
        if filtered.empty:
            return f"❌ No HABs data found for lake '{lake_name}'."
        
        indicators = ["bloom_probability", "cyanobacteria_density", "toxin_levels", "chl_ug_l"]
        result = f"🦠 HABs Summary for {lake_name.title()}:\n"
        
        for col in indicators:
            if col in filtered.columns and filtered[col].notna().any():
                stats = filtered[col].describe()
                result += (f"  • {col.replace('_', ' ').title()}: "
                          f"Mean={stats['mean']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}\n")
        
        if "bloom_status" in filtered.columns:
            bloom_counts = filtered["bloom_status"].value_counts().to_dict()
            result += f"  • Bloom Status: {bloom_counts}\n"
        
        return result.strip()
    except Exception as e:
        return f"❌ Error retrieving HABs data: {str(e)}"


@tool
def detect_outliers(filepath: str, parameter: str) -> str:
    """
    Detect outliers in a specific parameter using MAD-based robust Z-scores.
    
    Args:
        filepath: Path to CSV file
        parameter: Column name to analyze
    
    Returns:
        Summary of detected outliers
    """
    try:
        df = pd.read_csv(filepath)
        
        if parameter not in df.columns:
            return f"❌ Parameter '{parameter}' not found in dataset."
        
        values = df[parameter].dropna()
        if len(values) < 5:
            return f"⚠️ Insufficient data ({len(values)} values) for outlier detection."
        
        z_scores = mad_zscores(values.tolist())
        outlier_mask = np.abs(z_scores) > 5.0
        n_outliers = np.sum(outlier_mask)
        
        if n_outliers == 0:
            return f"✅ No outliers detected in '{parameter}'"
        
        outlier_values = values[outlier_mask].tolist()
        return (f"⚠️ Detected {n_outliers} outliers in '{parameter}':\n"
               f"  Values: {[round(v, 2) for v in outlier_values[:10]]}" + 
               (" ..." if n_outliers > 10 else ""))
    except Exception as e:
        return f"❌ Outlier detection failed: {str(e)}"