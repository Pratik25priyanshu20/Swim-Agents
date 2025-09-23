import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from swim.agents.calibro.config.lake_config import LAKES

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path("outputs")


# -----------------------------
# Basic Tools
# -----------------------------

def list_lakes(_: str = "") -> str:
    """Return list of lake names from config file."""
    lake_names = [lake["name"] for lake in LAKES]
    return "\n".join(f"• {name}" for name in lake_names)


def summarize_all_lakes(_: str = "") -> str:
    """Summarize timeseries CSV outputs for each lake in the /outputs folder."""
    if not OUTPUT_DIR.exists():
        return "⚠️ No outputs directory found. Please run the pipeline first."

    csv_files = list(OUTPUT_DIR.glob("*_timeseries.csv"))
    if not csv_files:
        return "⚠️ No CSV files found. Run the pipeline to generate lake data."

    summaries = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        lake_name = csv_file.stem.replace("_timeseries", "").replace("_", " ")
        dates = df["date"].dropna()
        if dates.empty:
            summaries.append(f"{lake_name}: No valid data.")
            continue

        date_range = f"{dates.min()} to {dates.max()}"
        num_records = len(df)
        summaries.append(f"✅ {lake_name}: {num_records} records ({date_range})")

    return "\n".join(summaries)


# -----------------------------
# CSV File Listing & Summarizing
# -----------------------------

def list_uploaded_csvs(_: str = "") -> str:
    """List all uploaded CSV files in the calibro/data/ directory."""
    csvs = list(DATA_DIR.glob("*.csv"))
    if not csvs:
        return "⚠️ No CSV files found in data/ folder."
    return "\n".join(f"• {csv.name}" for csv in csvs)


def summarize_uploaded_csv(file_name: str) -> str:
    """Generate summary statistics and metadata for a selected CSV file."""
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        return f"❌ File '{file_name}' not found in data/"

    try:
        df = pd.read_csv(file_path)
        summary = df.describe(include='all').transpose()
        columns = ", ".join(df.columns[:5])
        return f"""✅ File: {file_name}
Columns: {columns}...
Rows: {len(df)}

{summary.head(5)}"""
    except Exception as e:
        return f"⚠️ Error reading {file_name}: {e}"


# -----------------------------
# Advanced Tools
# -----------------------------

def filter_by_lake(file_name: str, lake_name: str) -> str:
    """Filter a CSV file by lake name (case-insensitive match on 'lake_name' column)."""
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        return f"❌ File '{file_name}' not found."

    try:
        df = pd.read_csv(file_path)
        filtered = df[df["lake_name"].str.lower() == lake_name.lower()]
        if filtered.empty:
            return f"⚠️ No records found for lake '{lake_name}' in '{file_name}'"
        return filtered.head(5).to_string(index=False)
    except Exception as e:
        return f"⚠️ Error filtering by lake: {e}"


def filter_by_date_range(file_name: str, start_date: str, end_date: str) -> str:
    """Filter a CSV file based on timestamp range. Columns must include 'timestamp'."""
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        return f"❌ File '{file_name}' not found."

    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        filtered = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
        if filtered.empty:
            return f"⚠️ No data between {start_date} and {end_date} in '{file_name}'"
        return filtered.head(5).to_string(index=False)
    except Exception as e:
        return f"⚠️ Error filtering by date: {e}"


def plot_index_timeseries(file_name: str, index_name: str) -> str:
    """Plot time-series of a selected index (e.g., chlorophyll_index) for all lakes in the file."""
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        return f"❌ File '{file_name}' not found."

    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        if index_name not in df.columns:
            return f"❌ Index '{index_name}' not found in '{file_name}'."

        plt.figure(figsize=(10, 4))
        for lake in df["lake_name"].unique():
            lake_df = df[df["lake_name"] == lake]
            plt.plot(lake_df["timestamp"], lake_df[index_name], label=lake)

        plt.title(f"{index_name} Over Time")
        plt.xlabel("Date")
        plt.ylabel(index_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = OUTPUT_DIR / "temp_plot.png"
        plt.savefig(save_path)
        return f"📈 Plot saved to {save_path}"
    except Exception as e:
        return f"⚠️ Error generating plot: {e}"


def summarize_quality_metrics(file_name: str) -> str:
    """Summarize calibration quality metrics from a CSV (RMSE, R², score)."""
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        return f"❌ File '{file_name}' not found."

    try:
        df = pd.read_csv(file_path)
        required = {"validation_rmse", "validation_r2", "model_performance_score"}
        if not required.issubset(df.columns):
            return "⚠️ Required quality metrics not found in file."

        summary = df[list(required)].describe().round(3)
        return f"📊 Calibration Metrics Summary:\n{summary}"
    except Exception as e:
        return f"⚠️ Error summarizing metrics: {e}"