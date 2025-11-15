# calibro/core_pipeline.py


import ee
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime


# Initialize Earth Engine with Retry Logic

def initialize_earth_engine(max_retries=3):
    """Initialize Earth Engine with retry logic"""
    for attempt in range(max_retries):
        try:
            ee.Initialize()
            print("✅ Earth Engine initialized.")
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                print("🔑 Earth Engine not initialized. Authenticating...")
                try:
                    ee.Authenticate()
                    ee.Initialize()
                    print("✅ Earth Engine authenticated and initialized.")
                    return True
                except Exception as auth_error:
                    print(f"❌ Failed to initialize Earth Engine: {auth_error}")
                    return False
            time.sleep(2 ** attempt)
    return False

# Initialize on module load
initialize_earth_engine()


from swim.agents.calibro.config.lake_config import LAKES, DATE_START, DATE_END
from swim.agents.calibro.utils.satellite_fetch import (
    lake_to_aoi,
    mask_s2_clouds_shadows,
    add_rrs_from_sr,
    add_indices,
    build_water_mask,
    chl_from_ndci,
    dogliotti_turbidity,
    percentiles_25_50_75,
    spatial_outlier_mask
)

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# Safe GEE Operation Wrapper

def safe_gee_operation(operation, operation_name="GEE Operation", max_retries=3):
    """
    Safely execute GEE operations with retry logic.
    
    Args:
        operation: Callable that performs the GEE operation
        operation_name: Name for logging
        max_retries: Maximum number of retry attempts
    
    Returns:
        Result of operation or None on failure
    """
    for attempt in range(max_retries):
        try:
            result = operation()
            return result
        except ee.EEException as e:
            if "Too many concurrent" in str(e) or "Quota exceeded" in str(e):
                wait_time = 2 ** attempt
                print(f"⚠️ {operation_name}: Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    print(f"❌ {operation_name} failed after {max_retries} attempts")
                    raise
            else:
                print(f"❌ {operation_name} error: {e}")
                raise
        except Exception as e:
            print(f"❌ {operation_name} unexpected error: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    return None


# Enhanced Time Series Extraction with Confidence Metrics


def timeseries_table(col: ee.ImageCollection, geom: ee.Geometry, lake_name: str) -> pd.DataFrame:
    """
    Extract time series with confidence metrics and quality flags.
    
    Args:
        col: ImageCollection with processed images
        geom: Lake geometry
        lake_name: Name of the lake
    
    Returns:
        DataFrame with values and confidence metrics
    """
    def per_img(im):
        # Select all relevant bands including QA flags
        bands_to_extract = ["NDCI", "Turbidity_FNU", "TSS_rel"]
        
        # Add QA bands if they exist
        qa_bands = ["Chl_TEMP_QA", "Chl_SPATIAL_QA", "Turb_TEMP_QA", "Turb_SPATIAL_QA"]
        for qa_band in qa_bands:
            try:
                im.select(qa_band)
                bands_to_extract.append(qa_band)
            except:
                pass
        
        stats = im.select(bands_to_extract).reduceRegion(
            reducer=ee.Reducer.median(), 
            geometry=geom, 
            scale=10, 
            maxPixels=1e10
        )
        
        # Add metadata
        feat_props = stats.combine({
            "date": im.date().format("YYYY-MM-dd"),
            "lake": lake_name,
            "timestamp": im.date().millis()
        }, overwrite=True)
        
        return ee.Feature(None, feat_props)
    
    def extract_features():
        fc = ee.FeatureCollection(col.map(per_img))
        return fc.getInfo().get("features", [])
    
    # Safe extraction with retry
    features = safe_gee_operation(
        extract_features,
        f"Time series extraction for {lake_name}"
    )
    
    if not features:
        print(f"⚠️ No features extracted for {lake_name}")
        return pd.DataFrame()
    
    recs = [f["properties"] for f in features]
    df = pd.DataFrame.from_records(recs)

    if df.empty:
        return df

    # Parse dates
    df["date"] = pd.to_datetime(df["date"])
    
    # Convert numeric columns
    numeric_cols = ["NDCI_median", "Turbidity_FNU_median", "TSS_rel_median"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Rename for clarity
    rename_dict = {
        "NDCI_median": "NDCI",
        "Turbidity_FNU_median": "Turbidity_FNU",
        "TSS_rel_median": "TSS_rel"
    }
    df = df.rename(columns=rename_dict)
    
    # Sort by date
    df = df.sort_values("date")
    
    # Add data quality score
    df = add_quality_score(df)

    return df


# Data Quality Scoring


def add_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add composite quality score based on QA flags and data completeness.
    
    Args:
        df: DataFrame with observations
    
    Returns:
        DataFrame with quality_score column
    """
    if df.empty:
        return df
    
    quality_scores = []
    
    for idx, row in df.iterrows():
        score = 100  # Start with perfect score
        
        # Deduct for missing values
        if pd.isna(row.get('NDCI')):
            score -= 30
        if pd.isna(row.get('Turbidity_FNU')):
            score -= 20
        
        # Check QA flags (if present)
        qa_flags = ['Chl_TEMP_QA', 'Chl_SPATIAL_QA', 'Turb_TEMP_QA', 'Turb_SPATIAL_QA']
        for flag in qa_flags:
            if flag in df.columns:
                if row.get(flag) == 0 or pd.isna(row.get(flag)):
                    score -= 10
        
        # Ensure score is between 0-100
        score = max(0, min(100, score))
        quality_scores.append(score)
    
    df['quality_score'] = quality_scores
    return df

# ------------------------
# Validation Functions
# ------------------------
def validate_calibration_output(df: pd.DataFrame, lake_name: str) -> Tuple[bool, Dict]:
    """
    Validate calibrated data quality.
    
    Args:
        df: DataFrame with calibrated data
        lake_name: Name of the lake
    
    Returns:
        Tuple of (is_valid, validation_report)
    """
    checks = {
        'completeness': False,
        'reasonable_ranges': False,
        'temporal_coverage': False,
        'low_anomalies': False
    }
    
    report = {
        'lake_name': lake_name,
        'total_observations': len(df),
        'issues': []
    }
    
    if df.empty:
        report['issues'].append("No data available")
        return False, report
    
    # Check 1: Completeness
    checks['completeness'] = len(df) > 0
    
    # Check 2: Reasonable ranges
    range_ok = True
    if 'Chl_mg_m3' in df.columns:
        chl_range_ok = df['Chl_mg_m3'].between(0, 200).all()
        if not chl_range_ok:
            report['issues'].append("Chlorophyll-a values outside reasonable range [0-200]")
            range_ok = False
    
    if 'Turbidity_FNU' in df.columns:
        turb_range_ok = df['Turbidity_FNU'].between(0, 300).all()
        if not turb_range_ok:
            report['issues'].append("Turbidity values outside reasonable range [0-300]")
            range_ok = False
    
    checks['reasonable_ranges'] = range_ok
    
    # Check 3: Temporal coverage
    if 'date' in df.columns:
        date_range = (df['date'].max() - df['date'].min()).days
        checks['temporal_coverage'] = date_range > 7  # At least 1 week
        report['temporal_coverage_days'] = date_range
    
    # Check 4: Low anomalies (basic check)
    if 'Chl_mg_m3' in df.columns:
        mean_chl = df['Chl_mg_m3'].mean()
        std_chl = df['Chl_mg_m3'].std()
        if std_chl < mean_chl * 2:  # Coefficient of variation check
            checks['low_anomalies'] = True
        else:
            report['issues'].append("High variability detected in chlorophyll-a")
    
    # Overall validation
    is_valid = all(checks.values())
    report['checks'] = checks
    report['is_valid'] = is_valid
    
    return is_valid, report

# ------------------------
# Enhanced Export with Metadata
# ------------------------
def export_tabs(df: pd.DataFrame, lake_name: str, metadata: Optional[Dict] = None):
    """
    Export data with comprehensive metadata.
    
    Args:
        df: DataFrame to export
        lake_name: Name of the lake
        metadata: Additional metadata to include
    """
    safe = lake_name.replace(" ", "_").replace("/", "-")
    csv_fp = OUT_DIR / f"{safe}_timeseries.csv"
    json_fp = OUT_DIR / f"{safe}_timeseries.json"
    meta_fp = OUT_DIR / f"{safe}_metadata.json"
    
    # Export main data
    df.to_csv(csv_fp, index=False)
    df.to_json(json_fp, orient='records', indent=2, date_format='iso')
    
    # Create and export metadata
    export_metadata = {
        'lake_name': lake_name,
        'export_date': datetime.now().isoformat(),
        'total_observations': len(df),
        'date_range': {
            'start': df['date'].min().isoformat() if not df.empty else None,
            'end': df['date'].max().isoformat() if not df.empty else None
        },
        'columns': list(df.columns),
        'calibration_version': 'CALIBRO_v2.0',
        'processing_pipeline': 'Sentinel-2 SR Harmonized with Cloud Masking'
    }
    
    if metadata:
        export_metadata.update(metadata)
    
    # Add summary statistics
    if not df.empty:
        stats = {}
        for col in ['NDCI', 'Turbidity_FNU', 'TSS_rel']:
            if col in df.columns:
                stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        export_metadata['summary_statistics'] = stats
    
    # Save metadata
    import json
    with open(meta_fp, 'w') as f:
        json.dump(export_metadata, f, indent=2)
    
    print(f"✅ Saved: {csv_fp}")
    print(f"✅ Saved: {json_fp}")
    print(f"✅ Saved: {meta_fp}")

# ------------------------
# Enhanced Plotting
# ------------------------
def plot_timeseries(df: pd.DataFrame, lake_name: str):
    """
    Create enhanced time series plots with quality indicators.
    
    Args:
        df: DataFrame with time series data
        lake_name: Name of the lake
    
    Returns:
        List of figure objects
    """
    if df is None or df.empty:
        print(f"⚠️ No data to plot for {lake_name}.")
        return []

    date = pd.to_datetime(df['date'])
    fig_list = []
    
    # Quality-aware plotting
    if 'quality_score' in df.columns:
        colors = df['quality_score'].apply(lambda x: 'green' if x > 70 else ('orange' if x > 40 else 'red'))
    else:
        colors = 'blue'

    # NDCI / Chlorophyll-a plot
    if 'NDCI' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        scatter = ax.scatter(date, df['NDCI'], c=colors, marker='o', s=30, alpha=0.6)
        ax.plot(date, df['NDCI'], lw=1, alpha=0.3, color='gray')
        ax.set_title(f"{lake_name} – NDCI (Chlorophyll-a Proxy)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("NDCI (unitless)", fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_list.append(fig)

    # Turbidity plot
    if 'Turbidity_FNU' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        scatter = ax.scatter(date, df['Turbidity_FNU'], c=colors, marker='o', s=30, alpha=0.6)
        ax.plot(date, df['Turbidity_FNU'], lw=1, alpha=0.3, color='gray')
        ax.set_title(f"{lake_name} – Turbidity (FNU)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Turbidity (FNU)", fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_list.append(fig)

    # TSS proxy plot
    if 'TSS_rel' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        scatter = ax.scatter(date, df['TSS_rel'], c=colors, marker='o', s=30, alpha=0.6)
        ax.plot(date, df['TSS_rel'], lw=1, alpha=0.3, color='gray')
        ax.set_title(f"{lake_name} – TSS Proxy (Rrs)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_list.append(fig)

    plt.show()
    return fig_list

# ------------------------
# Optimized Collection Filtering
# ------------------------
def optimize_collection_filtering(s2_col: ee.ImageCollection) -> ee.ImageCollection:
    """
    Pre-filter collection for performance and quality.
    
    Args:
        s2_col: Raw Sentinel-2 collection
    
    Returns:
        Optimized collection
    """
    return (s2_col
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .limit(500)  # Prevent timeout on very large collections
            .select(['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'QA60']))

# ------------------------
# Enhanced Lake Processing
# ------------------------
def process_lake(lake: Dict) -> Dict:
    """
    Process a single lake with enhanced error handling and metrics.
    
    Args:
        lake: Lake configuration dictionary
    
    Returns:
        Dictionary with processing results
    """
    lake_name = lake["name"]
    print(f"\n{'='*60}")
    print(f"🌊 Processing: {lake_name}")
    print(f"{'='*60}")
    
    try:
        AOI = lake_to_aoi(lake)
        
        # Fetch and filter Sentinel-2 data
        s2_raw = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(AOI)
                  .filterDate(DATE_START, DATE_END))
        
        s2 = optimize_collection_filtering(s2_raw)
        
        s2 = (s2.map(mask_s2_clouds_shadows)
              .map(add_rrs_from_sr)
              .map(add_indices)
              .map(lambda im: im.addBands(build_water_mask(im)))
              .map(lambda im: im.updateMask(im.select("WATER_MASK"))))

        def add_products(im):
            return dogliotti_turbidity(chl_from_ndci(im))

        s2p = s2.map(add_products)

        # Calculate quality metrics
        chl_q1, chl_med, chl_q3, chl_iqr = percentiles_25_50_75(s2p, "Chl_mg_m3")
        turb_q1, turb_med, turb_q3, turb_iqr = percentiles_25_50_75(s2p, "Turbidity_FNU")

        # Temporal QA
        chl_low = chl_q1.subtract(chl_iqr.multiply(1.5))
        chl_high = chl_q3.add(chl_iqr.multiply(1.5))
        chl_temp_ok = chl_med.gte(chl_low).And(chl_med.lte(chl_high)).rename("Chl_TEMP_QA")

        turb_low = turb_q1.subtract(turb_iqr.multiply(1.5))
        turb_high = turb_q3.add(turb_iqr.multiply(1.5))
        turb_temp_ok = turb_med.gte(turb_low).And(turb_med.lte(turb_high)).rename("Turb_TEMP_QA")

        # Spatial QA
        wm = s2.first().select("WATER_MASK")
        chl_spatial_ok = spatial_outlier_mask(chl_med.updateMask(wm), "Chl_mg_m3_median")
        turb_spatial_ok = spatial_outlier_mask(turb_med.updateMask(wm), "Turbidity_FNU_median")

        # Clean data
        chl_clean = chl_med.updateMask(wm).updateMask(chl_temp_ok).updateMask(chl_spatial_ok)
        turb_clean = turb_med.updateMask(wm).updateMask(turb_temp_ok).updateMask(turb_spatial_ok)

        # Build final composite
        final = (ee.Image()
                .addBands(chl_clean.rename("Chl_mg_m3_median"))
                .addBands(chl_iqr.rename("Chl_mg_m3_IQR"))
                .addBands(chl_temp_ok.rename("Chl_TEMP_QA"))
                .addBands(chl_spatial_ok.rename("Chl_SPATIAL_QA"))
                .addBands(turb_clean.rename("Turbidity_FNU_median"))
                .addBands(turb_iqr.rename("Turbidity_FNU_IQR"))
                .addBands(turb_temp_ok.rename("Turb_TEMP_QA"))
                .addBands(turb_spatial_ok.rename("Turb_SPATIAL_QA"))
                .addBands(wm.rename("WATER_MASK"))
                ).clip(AOI)

        return {
            "status": "success",
            "lake_name": lake_name,
            "aoi": AOI,
            "per_image": s2p,
            "final": final
        }
    
    except Exception as e:
        print(f"❌ Error processing {lake_name}: {e}")
        return {
            "status": "error",
            "lake_name": lake_name,
            "error": str(e)
        }

# ------------------------
# Main Pipeline Runner
# ------------------------
def run_calibro_pipeline(select_lake="ALL", validate=True):
    """
    Run the complete CALIBRO calibration pipeline.
    
    Args:
        select_lake: "ALL" or specific lake name
        validate: Whether to run validation checks
    """
    print("\n" + "="*70)
    print("🚀 CALIBRO Pipeline v2.0 - Enhanced Satellite Calibration")
    print("="*70)
    
    results = {}
    master_df = []
    validation_reports = []

    if select_lake == "ALL":
        to_process = LAKES
    else:
        lake = next((lk for lk in LAKES if lk["name"] == select_lake), None)
        if not lake:
            raise ValueError(f"❌ Lake '{select_lake}' not found in configuration.")
        to_process = [lake]

    for lake in to_process:
        name = lake["name"]
        
        # Process lake
        out = process_lake(lake)
        results[name] = out
        
        if out["status"] != "success":
            print(f"⚠️ Skipping {name} due to processing error")
            continue

        aoi = out["aoi"]
        col = out["per_image"]
        
        # Check data availability
        count = safe_gee_operation(
            lambda: col.size().getInfo(),
            f"Count images for {name}"
        )
        
        if count is None or count == 0:
            print(f"⚠️ No images for {name} — check date range or masks")
            continue
        
        print(f"🛰️ Images after filtering: {count}")

        # Extract time series
        df = timeseries_table(col, aoi, name)
        
        if df.empty:
            print(f"⚠️ No time series data extracted for {name}")
            continue
        
        print(f"📊 Extracted {len(df)} observations")
        print(df.head(2))
        
        # Validate if requested
        if validate:
            is_valid, val_report = validate_calibration_output(df, name)
            validation_reports.append(val_report)
            
            if is_valid:
                print(f"✅ Validation passed for {name}")
            else:
                print(f"⚠️ Validation issues for {name}:")
                for issue in val_report.get('issues', []):
                    print(f"   • {issue}")
        
        # Export with metadata
        metadata = {
            'image_count': count,
            'validation': validation_reports[-1] if validation_reports else None
        }
        export_tabs(df, name, metadata)
        
        # Add to master dataset
        master_df.append(df)
        
        # Plot
        plot_timeseries(df, name)

    # Export combined dataset
    if master_df:
        all_df = pd.concat(master_df, ignore_index=True)
        all_csv = OUT_DIR / "ALL_LAKES_timeseries.csv"
        all_json = OUT_DIR / "ALL_LAKES_timeseries.json"
        
        all_df.to_csv(all_csv, index=False)
        all_df.to_json(all_json, orient='records', indent=2, date_format='iso')
        
        print(f"\n✅ Combined dataset: {all_csv}")
        print(f"✅ Combined dataset: {all_json}")
    
    # Print summary
    print("\n" + "="*70)
    print("🎯 CALIBRO Pipeline Complete")
    print("="*70)
    print(f"Processed: {len([r for r in results.values() if r['status'] == 'success'])}/{len(to_process)} lakes")
    print(f"Total observations: {sum(len(df) for df in master_df)}")
    print(f"Output directory: {OUT_DIR.absolute()}")
    
    if validation_reports:
        valid_count = sum(1 for r in validation_reports if r.get('is_valid'))
        print(f"Validation: {valid_count}/{len(validation_reports)} lakes passed")
    
    print("="*70 + "\n")
    
    return results, master_df, validation_reports