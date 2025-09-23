

from langchain.tools import tool
import pandas as pd
from pathlib import Path
import json

# -------------------------------
# 1. Load CSV
# -------------------------------
@tool
def load_csv(filepath: str) -> str:
    """Load a CSV file and return the number of rows loaded."""
    df = pd.read_csv(filepath)
    return f"Loaded {len(df)} rows from {filepath}"


# -------------------------------
# 2. Show Columns
# -------------------------------
@tool
def show_columns(filepath: str) -> str:
    """Display the column names of the given CSV file."""
    df = pd.read_csv(filepath)
    return f"Columns: {df.columns.tolist()}"


# -------------------------------
# 3. Validate Sample
# -------------------------------
@tool
def validate_sample(filepath: str) -> str:
    """Check and report missing values per column in a CSV."""
    df = pd.read_csv(filepath)
    missing = df.isna().sum().to_dict()
    return f"Missing values per column: {missing}"


# -------------------------------
# 4. Compute Geo Bounds (Utility)
# -------------------------------
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


# -------------------------------
# 5. Compute BBOX
# -------------------------------
@tool
def compute_bbox(filepath: str) -> str:
    """Compute the bounding box (min/max lat/lon) for a CSV dataset."""
    df = pd.read_csv(filepath)
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return "Dataset missing latitude/longitude columns."
    bbox = compute_geo_bounds(df)
    return f"Bounding Box: {bbox}"


# -------------------------------
# 6. Summarize Harmonized Output
# -------------------------------
@tool
def summarize_harmonized_data(input: str = "") -> str:
    """Summarize record count and quality score from harmonized metadata JSON files."""
    output_dir = Path(__file__).resolve().parents[3] / "data/harmonized"
    summaries = []

    for file in output_dir.glob("*.json"):
        with open(file) as f:
            meta = json.load(f)
            avg_quality = meta["quality_stats"].get("avg_quality_score")
            summary = f"{meta['source_name']}: {meta['record_count']} records"
            if avg_quality is not None:
                summary += f", Avg quality: {avg_quality:.2f}"
            summaries.append(summary)

    return "\n".join(summaries) if summaries else "No metadata files found."


# -------------------------------
# 7. Get Water Quality for a Lake
# -------------------------------
@tool
def get_lake_quality(lake_name: str, parameter: str = None, year: int = None) -> str:
    """
    Returns water quality statistics (e.g., turbidity, pH) for a given lake.
    Optionally filters by parameter and/or year.
    """
    import pandas as pd
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "data/harmonized/lake_samples.parquet"
    if not path.exists():
        return "❌ lake_samples.parquet not found."

    df = pd.read_parquet(path)
    if 'lake' not in df.columns:
        return "❌ 'lake' column not found in data."

    # Normalize lake names
    match = df['lake'].dropna().str.lower().str.strip() == lake_name.lower().strip()
    filtered = df[match]

    if year:
        filtered = filtered[filtered['year'] == year]

    if filtered.empty:
        return f"No data found for lake '{lake_name}'" + (f" in year {year}." if year else ".")

    # Use standard columns only
    numeric_cols = ['temperature', 'turbidity', 'ph', 'quality_score']

    if parameter:
        parameter = parameter.strip().lower()
        if parameter not in filtered.columns.str.lower():
            return f"'{parameter}' not found in data."
        column = next(col for col in filtered.columns if col.lower() == parameter)
        stats = filtered[column].describe()
        return (f"📊 {column.title()} at {lake_name.title()}" + (f" (Year {year})" if year else "") + ":\n"
                f"Count: {int(stats['count'])}, Mean: {stats['mean']:.2f}, "
                f"Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
    else:
        result = f"📍 Water Quality Summary for {lake_name.title()}"
        if year:
            result += f" (Year: {year})"
        result += ":\n"
        for col in numeric_cols:
            if col in filtered.columns:
                stats = filtered[col].describe()
                result += (f"- {col.title()}: Count={int(stats['count'])}, "
                           f"Mean={stats['mean']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}\n")
        return result.strip()
    
    
    
    

@tool
def list_available_lakes(input: str = "") -> str:
    """
    Lists all unique lake names available in the lake_samples.parquet file.
    """
    from pathlib import Path
    import pandas as pd

    path = Path(__file__).resolve().parents[3] / "data/harmonized/lake_samples.parquet"
    if not path.exists():
        return "lake_samples.parquet not found."

    df = pd.read_parquet(path)
    if "lake" not in df.columns:
        return "'lake' column missing in data."

    lakes = df["lake"].dropna().str.strip().str.title().unique().tolist()
    if not lakes:
        return "No lakes found in the dataset."

    return "🗺️ Available Lakes:\n" + "\n".join(f"- {lake}" for lake in sorted(lakes))


# -------------------------------
# 8. Get HABs Summary for a Lake
# -------------------------------
@tool
def get_habs_summary(lake_name: str) -> str:
    """
    Summarizes Harmful Algal Bloom (HABs) indicators for a given lake name.
    Returns bloom probability, cyanobacteria density, toxin levels, and bloom status.
    """
    import pandas as pd
    from pathlib import Path

    file_path = Path(__file__).resolve().parents[3] / "data/harmonized/lake_samples.parquet"
    if not file_path.exists():
        return "lake_samples.parquet not found."

    df = pd.read_parquet(file_path)

    # Match lake name case-insensitively
    filtered = df[df["lake"].dropna().str.lower().str.strip() == lake_name.lower().strip()]

    if filtered.empty:
        return f"No HABs data found for lake '{lake_name}'."

    indicators = ["bloom_probability", "cyanobacteria_density", "toxin_levels"]
    result = f"HABs Summary for {lake_name}:\n"
    for col in indicators:
        if col in filtered.columns:
            stats = filtered[col].describe()
            result += (
                f"- {col.replace('_', ' ').title()}: "
                f"Count={int(stats['count'])}, "
                f"Mean={stats['mean']:.2f}, "
                f"Min={stats['min']:.2f}, "
                f"Max={stats['max']:.2f}\n"
            )

    if "bloom_status" in filtered.columns:
        bloom_counts = filtered["bloom_status"].value_counts().to_dict()
        result += f"- Bloom Status distribution: {bloom_counts}\n"

    return result.strip()