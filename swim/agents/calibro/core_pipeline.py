# calibro/core_pipeline.py
# calibro/core_pipeline.py

import ee
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from pathlib import Path

# ------------------------
# ✅ Initialize Earth Engine
# ------------------------
try:
    ee.Initialize()
    print("✅ Earth Engine initialized.")
except Exception:
    print("🔑 Earth Engine not initialized. Authenticating...")
    ee.Authenticate()
    ee.Initialize()
    print("✅ Earth Engine authenticated and initialized.")

# ------------------------
# Imports
# ------------------------
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


def timeseries_table(col: ee.ImageCollection, geom: ee.Geometry, lake_name: str) -> pd.DataFrame:
    def per_img(im):
        stats = im.select(["NDCI", "Turbidity_FNU", "TSS_rel"]).reduceRegion(
            reducer=ee.Reducer.median(), geometry=geom, scale=10, maxPixels=1e10
        )
        feat_props = stats.combine({"date": im.date().format("YYYY-MM-dd"),
                                    "lake": lake_name}, overwrite=True)
        return ee.Feature(None, feat_props)

    fc = ee.FeatureCollection(col.map(per_img))
    features = fc.getInfo().get("features", [])
    recs = [f["properties"] for f in features] if features else []
    df = pd.DataFrame.from_records(recs)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    for c in ["NDCI_median", "Turbidity_FNU_median", "TSS_rel_median"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.rename(columns={
        "NDCI_median": "NDCI",
        "Turbidity_FNU_median": "Turbidity_FNU",
        "TSS_rel_median": "TSS_rel"
    }).sort_values("date")

    return df


def export_tabs(df, lake_name):
    safe = lake_name.replace(" ", "_").replace("/", "-")
    csv_fp = OUT_DIR / f"{safe}_timeseries.csv"
    json_fp = OUT_DIR / f"{safe}_timeseries.json"
    df.to_csv(csv_fp, index=False)
    df.to_json(json_fp, orient='records', indent=2, date_format='iso')
    print(f"Saved: {csv_fp}\nSaved: {json_fp}")


def plot_timeseries(df, lake_name):
    if df is None or df.empty:
        print(f"No data to plot for {lake_name}.")
        return []

    date = pd.to_datetime(df['date'])
    fig_list = []

    if 'NDCI' in df:
        plt.figure(figsize=(10, 3.2))
        plt.plot(date, df['NDCI'], marker='o', lw=1)
        plt.title(f"{lake_name} – NDCI (Chl-a proxy)")
        plt.xlabel("Date")
        plt.ylabel("NDCI (unitless)")
        plt.grid(True, alpha=0.3)
        fig_list.append(plt.gcf())

    if 'Turbidity_FNU' in df:
        plt.figure(figsize=(10, 3.2))
        plt.plot(date, df['Turbidity_FNU'], marker='o', lw=1)
        plt.title(f"{lake_name} – Turbidity (FNU)")
        plt.xlabel("Date")
        plt.ylabel("Turbidity (FNU)")
        plt.grid(True, alpha=0.3)
        fig_list.append(plt.gcf())

    if 'TSS_rel' in df:
        plt.figure(figsize=(10, 3.2))
        plt.plot(date, df['TSS_rel'], marker='o', lw=1)
        plt.title(f"{lake_name} – TSS proxy (Rrs)")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        fig_list.append(plt.gcf())

    plt.show()
    return fig_list


def process_lake(lake: Dict) -> Dict:
    AOI = lake_to_aoi(lake)

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(AOI)
          .filterDate(DATE_START, DATE_END)
          .map(mask_s2_clouds_shadows)
          .map(add_rrs_from_sr)
          .map(add_indices)
          .map(lambda im: im.addBands(build_water_mask(im)))
          .map(lambda im: im.updateMask(im.select("WATER_MASK"))))

    def add_products(im):
        return dogliotti_turbidity(chl_from_ndci(im))

    s2p = s2.map(add_products)

    chl_q1, chl_med, chl_q3, chl_iqr = percentiles_25_50_75(s2p, "Chl_mg_m3")
    turb_q1, turb_med, turb_q3, turb_iqr = percentiles_25_50_75(s2p, "Turbidity_FNU")

    chl_low = chl_q1.subtract(chl_iqr.multiply(1.5))
    chl_high = chl_q3.add(chl_iqr.multiply(1.5))
    chl_temp_ok = chl_med.gte(chl_low).And(chl_med.lte(chl_high)).rename("Chl_TEMP_QA")

    turb_low = turb_q1.subtract(turb_iqr.multiply(1.5))
    turb_high = turb_q3.add(turb_iqr.multiply(1.5))
    turb_temp_ok = turb_med.gte(turb_low).And(turb_med.lte(turb_high)).rename("Turb_TEMP_QA")

    wm = s2.first().select("WATER_MASK")
    chl_spatial_ok = spatial_outlier_mask(chl_med.updateMask(wm), "Chl_mg_m3_median")
    turb_spatial_ok = spatial_outlier_mask(turb_med.updateMask(wm), "Turbidity_FNU_median")

    chl_clean = chl_med.updateMask(wm).updateMask(chl_temp_ok).updateMask(chl_spatial_ok)
    turb_clean = turb_med.updateMask(wm).updateMask(turb_temp_ok).updateMask(turb_spatial_ok)

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
        "aoi": AOI,
        "per_image": s2p,
        "final": final
    }


def run_calibro_pipeline(select_lake="ALL"):
    results = {}
    master_df = []

    if select_lake == "ALL":
        to_process = LAKES
    else:
        lake = next((lk for lk in LAKES if lk["name"] == select_lake), None)
        if not lake:
            raise ValueError(f"Lake '{select_lake}' not found.")
        to_process = [lake]

    for lake in to_process:
        name = lake["name"]
        print(f"\n🌊 Processing lake: {name}")
        out = process_lake(lake)
        results[name] = out

        aoi = out["aoi"]
        col = out["per_image"]
        count = col.size().getInfo()
        print(f"🛰️ Images after filtering: {count}")
        if count == 0:
            print("⚠️ No images — try adjusting date range or masks.")
            continue

        df = timeseries_table(col, aoi, name)
        if not df.empty:
            print(df.head(2))
            export_tabs(df, name)
            master_df.append(df)
            plot_timeseries(df, name)

    if master_df:
        all_df = pd.concat(master_df, ignore_index=True)
        all_df.to_csv("outputs/ALL_LAKES_timeseries.csv", index=False)
        all_df.to_json("outputs/ALL_LAKES_timeseries.json", orient='records', indent=2)
        print("\n✅ Exported all lakes data to /outputs")

    print("\n🎯 Calibro pipeline run complete.")