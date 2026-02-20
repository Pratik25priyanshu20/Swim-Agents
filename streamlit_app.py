"""
SWIM Platform — Surface Water Intelligence & Monitoring Dashboard
Advanced interactive dashboard with real-time agent monitoring,
interactive maps, and calibrated risk visualization.
"""

import os
import math
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from swim.agents.predikt.config import (
    GERMAN_LAKES,
    FEATURE_WEIGHTS,
    RISK_THRESHOLDS,
    FORECAST_HORIZONS,
    PARAMETER_RANGES,
)
from swim.agents.visios.visios_agent import VisiosAgent

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VISIOS_DIR = DATA_DIR / "visios_images"
MODELS_DIR = BASE_DIR / "models"

RISK_COLORS = {
    "low": "#22c55e",
    "moderate": "#eab308",
    "high": "#f97316",
    "critical": "#ef4444",
    "unknown": "#6b7280",
}

AGENT_INFO = {
    "HOMOGEN": {"icon": "📦", "port": 10001, "desc": "Data Harmonization", "color": "#6366f1"},
    "CALIBRO": {"icon": "🛰️", "port": 10002, "desc": "Satellite Calibration", "color": "#06b6d4"},
    "VISIOS":  {"icon": "📸", "port": 10003, "desc": "Visual Analysis", "color": "#8b5cf6"},
    "PREDIKT": {"icon": "🧠", "port": 10004, "desc": "ML Prediction", "color": "#f59e0b"},
    "ORCHESTRATOR": {"icon": "🎯", "port": 10000, "desc": "Pipeline Control", "color": "#ec4899"},
}

TROPHIC_COLORS = {
    "oligotrophic": "#22d3ee",
    "mesotrophic": "#facc15",
    "eutrophic": "#f97316",
}

# ---------------------------------------------------------------------------
# PAGE CONFIG & CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SWIM Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
  --bg-primary: #0a0e17;
  --bg-secondary: #111827;
  --bg-card: rgba(17, 24, 39, 0.7);
  --bg-card-hover: rgba(31, 41, 55, 0.8);
  --border: rgba(75, 85, 99, 0.3);
  --border-glow: rgba(59, 130, 246, 0.15);
  --text-primary: #f9fafb;
  --text-secondary: #9ca3af;
  --text-muted: #6b7280;
  --accent-blue: #3b82f6;
  --accent-cyan: #06b6d4;
  --accent-purple: #8b5cf6;
  --accent-green: #22c55e;
  --accent-orange: #f97316;
  --accent-red: #ef4444;
  --gradient-hero: linear-gradient(135deg, #0f172a 0%, #1e1b4b 40%, #0f172a 100%);
}

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: var(--text-primary);
}

.stApp {
  background: var(--bg-primary);
}

/* Main container */
.block-container {
  padding-top: 1rem !important;
  padding-bottom: 2rem !important;
  max-width: 1400px;
}

/* Hide default streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar */
section[data-testid="stSidebar"] > div {
  background: var(--bg-secondary);
  border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stSlider label {
  color: var(--text-secondary) !important;
  font-size: 0.85rem;
}

/* Hero header */
.hero-banner {
  background: var(--gradient-hero);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 28px 36px;
  margin-bottom: 24px;
  position: relative;
  overflow: hidden;
}

.hero-banner::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle at 30% 50%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
              radial-gradient(circle at 70% 50%, rgba(139, 92, 246, 0.06) 0%, transparent 50%);
  pointer-events: none;
}

.hero-banner h1 {
  margin: 0;
  font-size: 32px;
  font-weight: 800;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, #f9fafb 0%, #93c5fd 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  position: relative;
}

.hero-banner .subtitle {
  margin: 6px 0 0 0;
  color: var(--text-secondary);
  font-size: 14px;
  font-weight: 400;
  position: relative;
}

.hero-banner .badge {
  display: inline-block;
  background: rgba(34, 197, 94, 0.15);
  border: 1px solid rgba(34, 197, 94, 0.3);
  color: #22c55e;
  padding: 2px 10px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  margin-left: 12px;
  position: relative;
}

/* Glass card */
.glass-card {
  background: var(--bg-card);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
  transition: all 0.2s ease;
}

.glass-card:hover {
  background: var(--bg-card-hover);
  border-color: var(--border-glow);
  box-shadow: 0 0 20px rgba(59, 130, 246, 0.05);
}

/* Agent cards */
.agent-card {
  background: var(--bg-card);
  backdrop-filter: blur(20px);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 20px;
  text-align: center;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.agent-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  border-radius: 12px 12px 0 0;
}

.agent-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

.agent-icon {
  font-size: 28px;
  margin-bottom: 6px;
}

.agent-name {
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 1px;
  margin-bottom: 2px;
}

.agent-desc {
  font-size: 11px;
  color: var(--text-muted);
}

.agent-port {
  font-size: 10px;
  color: var(--text-muted);
  margin-top: 4px;
  font-family: 'SF Mono', 'Fira Code', monospace;
}

.agent-status {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent-green);
  margin-right: 4px;
  box-shadow: 0 0 6px rgba(34, 197, 94, 0.5);
  animation: pulse-green 2s infinite;
}

@keyframes pulse-green {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* Metric card */
.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 20px;
}

.metric-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}

.metric-value {
  font-size: 28px;
  font-weight: 800;
  line-height: 1.2;
}

.metric-sub {
  font-size: 11px;
  color: var(--text-secondary);
  margin-top: 2px;
}

/* Section headers */
.section-header {
  font-size: 18px;
  font-weight: 700;
  color: var(--text-primary);
  margin: 24px 0 12px 0;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}

/* Risk indicator */
.risk-badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.risk-low { background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
.risk-moderate { background: rgba(234, 179, 8, 0.15); color: #eab308; border: 1px solid rgba(234, 179, 8, 0.3); }
.risk-high { background: rgba(249, 115, 22, 0.15); color: #f97316; border: 1px solid rgba(249, 115, 22, 0.3); }
.risk-critical { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: var(--bg-secondary);
  padding: 4px;
  border-radius: 12px;
  border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
  border-radius: 8px;
  padding: 8px 16px;
  color: var(--text-secondary);
  font-weight: 500;
  font-size: 13px;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
  background: rgba(59, 130, 246, 0.1) !important;
  color: var(--accent-blue) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
  background-color: transparent !important;
}

.stTabs [data-baseweb="tab-border"] {
  display: none;
}

/* Dataframe styling */
.stDataFrame {
  border-radius: 12px;
  overflow: hidden;
}

/* Button styling */
.stButton > button {
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  padding: 8px 24px;
  transition: all 0.2s ease;
}

.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

/* Streamlit metric override */
div[data-testid="stMetricValue"] {
  color: var(--text-primary) !important;
  font-weight: 700;
}

div[data-testid="stMetricLabel"] {
  color: var(--text-secondary) !important;
}

/* Expander */
.streamlit-expanderHeader {
  background: var(--bg-card) !important;
  border-radius: 8px !important;
}

/* Separator */
.divider {
  height: 1px;
  background: var(--border);
  margin: 20px 0;
}

/* Lake info card */
.lake-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 12px;
}

.lake-name {
  font-size: 16px;
  font-weight: 700;
  margin-bottom: 4px;
}

.lake-region {
  font-size: 12px;
  color: var(--text-muted);
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--text-muted);
}

/* Map legend */
.map-legend {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 16px;
  font-size: 12px;
}

.legend-item {
  display: flex;
  align-items: center;
  margin: 4px 0;
}

.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 8px;
  display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PLOTLY THEME
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#9ca3af", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="rgba(75,85,99,0.2)", zerolinecolor="rgba(75,85,99,0.2)"),
    yaxis=dict(gridcolor="rgba(75,85,99,0.2)", zerolinecolor="rgba(75,85,99,0.2)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hoverlabel=dict(bgcolor="#1f2937", font_size=12, font_family="Inter"),
)

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


@st.cache_data(show_spinner=False, ttl=300)
def _load_satellite_points(max_rows: int = 2000) -> pd.DataFrame:
    data_dir = DATA_DIR / "raw" / "satellite"
    if not data_dir.exists():
        return pd.DataFrame()
    cols = [
        "acquisition_date", "latitude", "longitude",
        "chlorophyll_index", "turbidity_index", "cloud_coverage",
        "lake_name", "satellite_name", "sensor_type",
    ]
    frames = []
    for path in data_dir.glob("*.csv"):
        try:
            df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in cols)
            df["source_file"] = path.name
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["latitude", "longitude"])
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=42)
    return data


@st.cache_data(show_spinner=False, ttl=300)
def _load_visios_points() -> pd.DataFrame:
    agent = VisiosAgent()
    rows = []
    for name in agent.list_images():
        path = agent.image_dir / name
        gps = agent.extract_gps_data(path)
        if gps:
            rows.append({
                "image": name,
                "latitude": gps["latitude"],
                "longitude": gps["longitude"],
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=300)
def _load_visios_summary() -> dict:
    agent = VisiosAgent()
    return agent.summarize_batch()


def _predikt_lakes_df() -> pd.DataFrame:
    rows = []
    for lake_name, meta in GERMAN_LAKES.items():
        rows.append({
            "lake": lake_name,
            "latitude": meta["lat"],
            "longitude": meta["lon"],
            "area_km2": meta.get("area_km2", 0),
            "depth_max_m": meta.get("depth_max_m", 0),
            "trophic_status": meta.get("trophic_status", "unknown"),
            "region": meta.get("region", ""),
            "elevation_m": meta.get("elevation_m", 0),
        })
    return pd.DataFrame(rows)


def _predikt_forecast(horizon: int) -> Optional[dict]:
    try:
        from swim.agents.predikt.predikt_agent import PrediktAgent
        agent = PrediktAgent()
        return agent.predict_german_lakes(horizon_days=horizon)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return None


def _load_pipeline_runs(limit: int = 200) -> List[dict]:
    try:
        from swim.shared.database.connection import db as swim_db
        return swim_db.get_recent_runs(limit=limit)
    except Exception:
        return []


def _risk_color(level: str) -> str:
    return RISK_COLORS.get(level, RISK_COLORS["unknown"])


def _risk_badge_html(level: str) -> str:
    css_class = f"risk-{level}" if level in RISK_COLORS else "risk-unknown"
    return f'<span class="risk-badge {css_class}">{level}</span>'


# ---------------------------------------------------------------------------
# PLOTLY CHART BUILDERS
# ---------------------------------------------------------------------------
def _build_gauge(value: float, title: str, thresholds: bool = True) -> go.Figure:
    """Build a radial gauge chart for risk score."""
    if thresholds:
        steps = [
            dict(range=[0, 0.30], color="rgba(34, 197, 94, 0.15)"),
            dict(range=[0.30, 0.60], color="rgba(234, 179, 8, 0.15)"),
            dict(range=[0.60, 0.80], color="rgba(249, 115, 22, 0.15)"),
            dict(range=[0.80, 1.0], color="rgba(239, 68, 68, 0.15)"),
        ]
        bar_color = (
            "#22c55e" if value < 0.30
            else "#eab308" if value < 0.60
            else "#f97316" if value < 0.80
            else "#ef4444"
        )
    else:
        steps = []
        bar_color = "#3b82f6"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(font=dict(size=36, color="#f9fafb"), valueformat=".2f"),
        title=dict(text=title, font=dict(size=13, color="#9ca3af")),
        gauge=dict(
            axis=dict(range=[0, 1], tickcolor="#4b5563", dtick=0.2,
                      tickfont=dict(size=10, color="#6b7280")),
            bar=dict(color=bar_color, thickness=0.3),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=steps,
            threshold=dict(
                line=dict(color="#f9fafb", width=2),
                thickness=0.75,
                value=value,
            ),
        ),
    ))
    fig.update_layout(
        height=220,
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "xaxis" and k != "yaxis"},
    )
    return fig


def _build_radar(weights: Dict[str, float], title: str = "Feature Importance") -> go.Figure:
    """Build a radar/spider chart for feature weights."""
    categories = list(weights.keys())
    values = list(weights.values())
    values.append(values[0])  # close the polygon
    categories.append(categories[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.15)',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=6, color='#3b82f6'),
        name=title,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, max(weights.values()) * 1.3],
                gridcolor="rgba(75,85,99,0.2)", tickfont=dict(size=9, color="#6b7280"),
            ),
            angularaxis=dict(
                gridcolor="rgba(75,85,99,0.2)",
                tickfont=dict(size=10, color="#9ca3af"),
            ),
        ),
        height=350,
        showlegend=False,
        **{k: v for k, v in PLOTLY_LAYOUT.items()
           if k not in ("xaxis", "yaxis")},
    )
    return fig


def _build_donut(labels: list, values: list, colors: list, title: str = "") -> go.Figure:
    """Build a donut chart."""
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="#0a0e17", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, color="#f9fafb"),
        hoverinfo="label+value+percent",
    ))
    fig.update_layout(
        height=300,
        showlegend=False,
        annotations=[dict(text=title, x=0.5, y=0.5, font_size=13,
                          font_color="#9ca3af", showarrow=False)],
        **{k: v for k, v in PLOTLY_LAYOUT.items()
           if k not in ("xaxis", "yaxis")},
    )
    return fig


def _build_lake_comparison(lakes_df: pd.DataFrame) -> go.Figure:
    """Build a grouped bar chart comparing lake properties."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Surface Area (km²)", "Max Depth (m)"),
        horizontal_spacing=0.15,
    )
    colors = [TROPHIC_COLORS.get(t, "#6b7280") for t in lakes_df["trophic_status"]]

    fig.add_trace(go.Bar(
        x=lakes_df["lake"], y=lakes_df["area_km2"],
        marker_color=colors, name="Area",
        text=lakes_df["area_km2"], textposition="outside",
        textfont=dict(size=10, color="#9ca3af"),
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=lakes_df["lake"], y=lakes_df["depth_max_m"],
        marker_color=colors, name="Depth",
        text=lakes_df["depth_max_m"], textposition="outside",
        textfont=dict(size=10, color="#9ca3af"),
    ), row=1, col=2)

    fig.update_layout(
        height=320, showlegend=False,
        **{k: v for k, v in PLOTLY_LAYOUT.items()
           if k not in ("xaxis", "yaxis")},
    )
    fig.update_xaxes(tickfont=dict(size=10, color="#9ca3af"), tickangle=-30)
    fig.update_yaxes(gridcolor="rgba(75,85,99,0.2)")
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color="#9ca3af")
    return fig


def _build_risk_timeline(runs_df: pd.DataFrame) -> go.Figure:
    """Build interactive risk score timeline."""
    fig = go.Figure()

    for location in runs_df["location_name"].unique():
        loc_df = runs_df[runs_df["location_name"] == location].sort_values("created_at")
        colors = [_risk_color(l) for l in loc_df["risk_level"]]
        fig.add_trace(go.Scatter(
            x=loc_df["created_at"],
            y=loc_df["risk_score"],
            mode="lines+markers",
            name=location,
            line=dict(width=2),
            marker=dict(size=6, color=colors, line=dict(width=1, color="#0a0e17")),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Risk Score: %{y:.3f}<br>"
                "Date: %{x}<extra></extra>"
            ),
            text=[location] * len(loc_df),
        ))

    # Threshold lines
    for level, thresh in [("Moderate", 0.30), ("High", 0.60), ("Critical", 0.80)]:
        fig.add_hline(
            y=thresh, line_dash="dot",
            line_color=RISK_COLORS.get(level.lower(), "#6b7280"),
            opacity=0.4,
            annotation_text=level,
            annotation_font=dict(size=10, color="#6b7280"),
        )

    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Risk Score",
        yaxis=dict(range=[0, 1.05], gridcolor="rgba(75,85,99,0.2)"),
        xaxis=dict(gridcolor="rgba(75,85,99,0.2)"),
        hovermode="x unified",
        **{k: v for k, v in PLOTLY_LAYOUT.items()
           if k not in ("xaxis", "yaxis")},
    )
    return fig


def _build_forecast_bars(predictions: dict) -> go.Figure:
    """Build horizontal bar chart for lake predictions."""
    lakes = list(predictions.keys())
    probs = [predictions[l].get("bloom_probability", 0) for l in lakes]
    uncertainties = [predictions[l].get("uncertainty", 0) for l in lakes]
    risk_levels = [predictions[l].get("risk_level", "unknown") for l in lakes]
    colors = [_risk_color(r) for r in risk_levels]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=lakes, x=probs,
        orientation="h",
        marker=dict(color=colors, line=dict(color="#0a0e17", width=1)),
        error_x=dict(type="data", array=uncertainties, color="#6b7280", thickness=1.5),
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
        textfont=dict(size=11, color="#f9fafb"),
        hovertemplate="<b>%{y}</b><br>Bloom probability: %{x:.1%}<br>Risk: " +
                      "<extra></extra>",
    ))

    fig.add_vline(x=0.30, line_dash="dot", line_color="#eab308", opacity=0.4)
    fig.add_vline(x=0.60, line_dash="dot", line_color="#f97316", opacity=0.4)
    fig.add_vline(x=0.80, line_dash="dot", line_color="#ef4444", opacity=0.4)

    fig.update_layout(
        height=max(200, len(lakes) * 60 + 80),
        xaxis=dict(range=[0, 1.15], title="Bloom Probability",
                   gridcolor="rgba(75,85,99,0.2)"),
        yaxis=dict(gridcolor="rgba(75,85,99,0.2)"),
        showlegend=False,
        **{k: v for k, v in PLOTLY_LAYOUT.items()
           if k not in ("xaxis", "yaxis")},
    )
    return fig


# ---------------------------------------------------------------------------
# MAP BUILDERS
# ---------------------------------------------------------------------------
def _build_advanced_map(
    lake_df: pd.DataFrame,
    sat_df: pd.DataFrame,
    visios_df: pd.DataFrame,
    predictions: Optional[dict] = None,
    show_lakes: bool = True,
    show_sat: bool = True,
    show_visios: bool = True,
    show_heatmap: bool = False,
    map_style: str = "dark",
    zoom: float = 5.6,
) -> pdk.Deck:
    """Build a multi-layer interactive PyDeck map."""
    layers = []

    # Satellite heatmap layer
    if show_heatmap and show_sat and not sat_df.empty:
        heat_df = sat_df.copy()
        heat_df["chlorophyll_index"] = pd.to_numeric(
            heat_df.get("chlorophyll_index", 0), errors="coerce"
        ).fillna(0)
        heat_df["weight"] = (heat_df["chlorophyll_index"] - heat_df["chlorophyll_index"].min()) / \
                            (heat_df["chlorophyll_index"].max() - heat_df["chlorophyll_index"].min() + 1e-6)
        layers.append(pdk.Layer(
            "HeatmapLayer",
            data=heat_df,
            get_position="[longitude, latitude]",
            get_weight="weight",
            radiusPixels=60,
            intensity=1.5,
            threshold=0.1,
            color_range=[
                [0, 104, 55],
                [26, 152, 80],
                [102, 189, 99],
                [254, 217, 118],
                [253, 141, 60],
                [215, 48, 31],
            ],
            opacity=0.6,
        ))

    # Satellite scatter layer
    if show_sat and not sat_df.empty and not show_heatmap:
        sat_vis = sat_df.copy()
        chlor = pd.to_numeric(sat_vis.get("chlorophyll_index", 0), errors="coerce").fillna(0)
        cmin, cmax = chlor.min(), chlor.max()
        scaled = (chlor - cmin) / (cmax - cmin + 1e-6)
        sat_vis["r"] = (30 + 200 * scaled).astype(int).clip(0, 255)
        sat_vis["g"] = (180 - 100 * scaled).astype(int).clip(0, 255)
        sat_vis["b"] = (80 - 40 * scaled).astype(int).clip(0, 255)
        sat_vis["a"] = 180

        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=sat_vis,
            get_position="[longitude, latitude]",
            get_radius=4000,
            get_fill_color="[r, g, b, a]",
            pickable=True,
            auto_highlight=True,
            highlight_color=[255, 255, 255, 60],
        ))

    # VISIOS image locations
    if show_visios and not visios_df.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=visios_df,
            get_position="[longitude, latitude]",
            get_radius=5000,
            get_fill_color="[139, 92, 246, 200]",
            get_line_color="[139, 92, 246, 255]",
            line_width_min_pixels=2,
            stroked=True,
            pickable=True,
            auto_highlight=True,
        ))

    # Lake markers with risk-colored 3D columns
    if show_lakes and not lake_df.empty:
        lake_vis = lake_df.copy()
        if predictions:
            lake_vis["risk_score"] = lake_vis["lake"].map(
                lambda l: predictions.get(l, {}).get("bloom_probability", 0.15)
            )
            lake_vis["risk_level"] = lake_vis["lake"].map(
                lambda l: predictions.get(l, {}).get("risk_level", "low")
            )
        else:
            lake_vis["risk_score"] = 0.15
            lake_vis["risk_level"] = "low"

        lake_vis["elevation"] = lake_vis["risk_score"] * 50000
        lake_vis["r"] = lake_vis["risk_level"].map(
            lambda l: {"low": 34, "moderate": 234, "high": 249, "critical": 239}.get(l, 100)
        )
        lake_vis["g"] = lake_vis["risk_level"].map(
            lambda l: {"low": 197, "moderate": 179, "high": 115, "critical": 68}.get(l, 100)
        )
        lake_vis["b"] = lake_vis["risk_level"].map(
            lambda l: {"low": 94, "moderate": 8, "high": 22, "critical": 68}.get(l, 100)
        )

        # 3D column for risk
        layers.append(pdk.Layer(
            "ColumnLayer",
            data=lake_vis,
            get_position="[longitude, latitude]",
            get_elevation="elevation",
            elevation_scale=1,
            radius=6000,
            get_fill_color="[r, g, b, 200]",
            pickable=True,
            auto_highlight=True,
            highlight_color=[255, 255, 255, 80],
        ))

        # Text labels
        layers.append(pdk.Layer(
            "TextLayer",
            data=lake_vis,
            get_position="[longitude, latitude]",
            get_text="lake",
            get_size=13,
            get_color="[249, 250, 251, 220]",
            get_angle=0,
            get_text_anchor="'middle'",
            get_alignment_baseline="'top'",
            billboard=True,
            font_family="'Inter', sans-serif",
            font_weight=600,
        ))

    # View state
    center_lat = 49.5 if lake_df.empty else lake_df["latitude"].mean()
    center_lon = 10.5 if lake_df.empty else lake_df["longitude"].mean()

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=35 if any(isinstance(l, pdk.Layer) and l.type == "ColumnLayer" for l in layers) else 0,
        bearing=0,
    )

    map_styles = {
        "dark": "mapbox://styles/mapbox/dark-v11",
        "satellite": "mapbox://styles/mapbox/satellite-streets-v12",
        "light": "mapbox://styles/mapbox/light-v11",
    }

    tooltip = {
        "html": """
            <div style="padding: 8px; font-family: Inter, sans-serif;">
            <b style="font-size: 14px;">{lake}</b><br/>
            <span style="color: #9ca3af;">Image: {image}</span><br/>
            <span style="color: #9ca3af;">Satellite: {satellite_name}</span><br/>
            <span style="color: #9ca3af;">Chlorophyll: {chlorophyll_index}</span><br/>
            <span style="color: #9ca3af;">Turbidity: {turbidity_index}</span><br/>
            <span style="color: #9ca3af;">Trophic: {trophic_status}</span>
            </div>
        """,
        "style": {
            "backgroundColor": "#111827",
            "color": "#f9fafb",
            "border": "1px solid rgba(75,85,99,0.3)",
            "borderRadius": "8px",
        },
    }

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=map_styles.get(map_style, map_styles["dark"]),
        tooltip=tooltip,
    )


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0;">
        <span style="font-size: 36px;">🌊</span>
        <h3 style="margin: 4px 0 0 0; font-size: 18px; font-weight: 800;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            SWIM Platform
        </h3>
        <p style="font-size: 11px; color: #6b7280; margin: 0;">v2.0 | A2A Protocol</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("**Map Controls**")
    show_lakes = st.checkbox("Lake markers", value=True, key="map_lakes")
    show_sat = st.checkbox("Satellite points", value=True, key="map_sat")
    show_visios = st.checkbox("VISIOS photos", value=True, key="map_visios")
    show_heatmap = st.checkbox("Chlorophyll heatmap", value=False, key="map_heat")
    map_style = st.selectbox("Map style", ["dark", "satellite", "light"], index=0)
    map_zoom = st.slider("Zoom level", 4.0, 10.0, 5.8, 0.1)
    sat_sample = st.slider("Satellite sample size", 200, 5000, 1500, 100)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("**Image Upload**")
    uploads = st.file_uploader(
        "Add lake photos",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="upload",
    )
    if uploads:
        VISIOS_DIR.mkdir(parents=True, exist_ok=True)
        saved = 0
        for f in uploads:
            with open(VISIOS_DIR / f.name, "wb") as out:
                out.write(f.getbuffer())
            saved += 1
        st.success(f"Saved {saved} image(s)")
        st.cache_data.clear()


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>SWIM Dashboard <span class="badge">LIVE</span></h1>
    <p class="subtitle">Surface Water Intelligence & Monitoring — Multi-Agent HABs Early Warning System</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "🎯 Command Center",
    "🗺️ Live Map",
    "🧠 Predictions",
    "🛰️ Satellite Intel",
    "📸 Visual Analysis",
    "📚 Knowledge Base",
    "⚡ Pipeline Control",
    "📊 Risk Analytics",
])


# =========================================================================
# TAB 0: COMMAND CENTER
# =========================================================================
with tabs[0]:
    # Agent status cards
    cols = st.columns(5)
    for i, (name, info) in enumerate(AGENT_INFO.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="agent-card" style="border-top: 3px solid {info['color']};">
                <div class="agent-icon">{info['icon']}</div>
                <div class="agent-name">{name}</div>
                <div class="agent-desc">{info['desc']}</div>
                <div class="agent-port">
                    <span class="agent-status"></span>
                    :{info['port']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # Key metrics row
    sat_df = _load_satellite_points(max_rows=sat_sample)
    visios_summary = _load_visios_summary()
    visios_count = visios_summary.get("statistics", {}).get("total_images", 0) if "error" not in visios_summary else 0
    model_count = len(list(MODELS_DIR.glob("*.pkl"))) if MODELS_DIR.exists() else 0
    lake_count = len(GERMAN_LAKES)

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Monitored Lakes</div>
            <div class="metric-value" style="color: #3b82f6;">{lake_count}</div>
            <div class="metric-sub">Across Germany</div>
        </div>
        """, unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Satellite Records</div>
            <div class="metric-value" style="color: #06b6d4;">{len(sat_df):,}</div>
            <div class="metric-sub">Sentinel-2 readings</div>
        </div>
        """, unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Images Analyzed</div>
            <div class="metric-value" style="color: #8b5cf6;">{visios_count}</div>
            <div class="metric-sub">Visual detections</div>
        </div>
        """, unsafe_allow_html=True)
    with mc4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ML Models</div>
            <div class="metric-value" style="color: #f59e0b;">{model_count}</div>
            <div class="metric-sub">Trained artifacts</div>
        </div>
        """, unsafe_allow_html=True)
    with mc5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">A2A Agents</div>
            <div class="metric-value" style="color: #ec4899;">5</div>
            <div class="metric-sub">Autonomous services</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Lake overview + Feature radar side by side
    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="section-header">Lake Database</div>', unsafe_allow_html=True)
        lakes_df = _predikt_lakes_df()
        fig = _build_lake_comparison(lakes_df)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">Feature Importance (PREDIKT)</div>', unsafe_allow_html=True)
        fig = _build_radar(FEATURE_WEIGHTS)
        st.plotly_chart(fig, use_container_width=True)

    # Trophic distribution
    st.markdown('<div class="section-header">Lake Trophic Classification</div>', unsafe_allow_html=True)
    left2, mid2, right2 = st.columns([1, 1, 2])
    with left2:
        trophic_counts = lakes_df["trophic_status"].value_counts()
        fig = _build_donut(
            labels=trophic_counts.index.tolist(),
            values=trophic_counts.values.tolist(),
            colors=[TROPHIC_COLORS.get(t, "#6b7280") for t in trophic_counts.index],
            title="Trophic",
        )
        st.plotly_chart(fig, use_container_width=True)
    with mid2:
        for _, row in lakes_df.iterrows():
            tc = TROPHIC_COLORS.get(row["trophic_status"], "#6b7280")
            st.markdown(f"""
            <div class="lake-card" style="border-left: 3px solid {tc};">
                <div class="lake-name">{row['lake']}</div>
                <div class="lake-region">{row['region']} | {row['trophic_status']} | {row['area_km2']} km²</div>
            </div>
            """, unsafe_allow_html=True)
    with right2:
        # Quick mini-map
        mini_deck = _build_advanced_map(
            lakes_df, pd.DataFrame(), pd.DataFrame(),
            show_lakes=True, show_sat=False, show_visios=False,
            map_style="dark", zoom=5.2,
        )
        st.pydeck_chart(mini_deck, height=350)


# =========================================================================
# TAB 1: LIVE MAP
# =========================================================================
with tabs[1]:
    st.markdown('<div class="section-header">Interactive Monitoring Map</div>', unsafe_allow_html=True)

    # Legend
    st.markdown("""
    <div class="map-legend">
        <div class="legend-item"><span class="legend-dot" style="background: #3b82f6;"></span> PREDIKT Lakes (3D columns = risk level)</div>
        <div class="legend-item"><span class="legend-dot" style="background: #22c55e;"></span> CALIBRO Satellite (color = chlorophyll)</div>
        <div class="legend-item"><span class="legend-dot" style="background: #8b5cf6;"></span> VISIOS Photo Locations</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    lake_df = _predikt_lakes_df()
    sat_map_df = _load_satellite_points(max_rows=sat_sample)
    visios_map_df = _load_visios_points()

    # Load predictions if available
    predictions = None
    if "predikt_forecast" in st.session_state and st.session_state["predikt_forecast"]:
        predictions = st.session_state["predikt_forecast"].get("predictions", {})

    deck = _build_advanced_map(
        lake_df, sat_map_df, visios_map_df,
        predictions=predictions,
        show_lakes=show_lakes,
        show_sat=show_sat,
        show_visios=show_visios,
        show_heatmap=show_heatmap,
        map_style=map_style,
        zoom=map_zoom,
    )
    st.pydeck_chart(deck, height=600)

    # Data summary below map
    st.markdown("")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Satellite Points", f"{len(sat_map_df):,}")
    mc2.metric("Photo Locations", len(visios_map_df))
    mc3.metric("Monitored Lakes", len(lake_df))
    mc4.metric("Map Layers", sum([show_lakes, show_sat, show_visios, show_heatmap]))


# =========================================================================
# TAB 2: PREDICTIONS
# =========================================================================
with tabs[2]:
    st.markdown('<div class="section-header">PREDIKT Bloom Forecasting Engine</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 3])
    with left:
        horizon = st.selectbox(
            "Forecast Horizon",
            [3, 7, 14],
            index=1,
            format_func=lambda x: f"{x}-day ({FORECAST_HORIZONS[x]['name']})",
            key="pred_horizon",
        )
        st.markdown(f"""
        <div class="glass-card" style="margin-top: 8px;">
            <div style="font-size: 11px; color: #6b7280; text-transform: uppercase;">Uncertainty</div>
            <div style="font-size: 20px; font-weight: 700; color: #f59e0b;">
                ±{FORECAST_HORIZONS[horizon]['uncertainty_base']:.0%}
            </div>
            <div style="font-size: 11px; color: #6b7280; margin-top: 4px;">
                {FORECAST_HORIZONS[horizon]['description']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Run Forecast", key="run_pred", use_container_width=True):
            with st.spinner("Running ML ensemble..."):
                st.session_state["predikt_forecast"] = _predikt_forecast(horizon)

    with right:
        forecast = st.session_state.get("predikt_forecast")
        if forecast:
            predictions = forecast.get("predictions", {})
            summary = forecast.get("summary", {})

            # Gauge row
            g1, g2, g3 = st.columns(3)
            with g1:
                avg_prob = summary.get("average_bloom_probability", 0)
                fig = _build_gauge(avg_prob, "Avg Bloom Probability")
                st.plotly_chart(fig, use_container_width=True)
            with g2:
                high_risk = len(summary.get("high_risk_lakes", []))
                fig = _build_gauge(high_risk / max(len(predictions), 1), "High Risk Ratio")
                st.plotly_chart(fig, use_container_width=True)
            with g3:
                max_prob = max((p.get("bloom_probability", 0) for p in predictions.values()), default=0)
                fig = _build_gauge(max_prob, "Peak Risk Score")
                st.plotly_chart(fig, use_container_width=True)

            # Per-lake bars
            fig = _build_forecast_bars(predictions)
            st.plotly_chart(fig, use_container_width=True)

            # Detailed table
            with st.expander("Detailed Predictions Table"):
                pred_rows = []
                for lake, data in predictions.items():
                    pred_rows.append({
                        "Lake": lake,
                        "Bloom Prob.": f"{data.get('bloom_probability', 0):.1%}",
                        "Risk Level": data.get("risk_level", "unknown"),
                        "Confidence": f"{data.get('confidence', 0):.1%}",
                        "Uncertainty": f"±{data.get('uncertainty', 0):.1%}",
                    })
                st.dataframe(pd.DataFrame(pred_rows), use_container_width=True)
        else:
            st.info("Click **Run Forecast** to generate predictions for all German lakes.")


# =========================================================================
# TAB 3: SATELLITE INTEL
# =========================================================================
with tabs[3]:
    st.markdown('<div class="section-header">CALIBRO Satellite Intelligence</div>', unsafe_allow_html=True)

    sat_data = _load_satellite_points(max_rows=sat_sample)
    if sat_data.empty:
        st.info("No satellite data found in `data/raw/satellite/`.")
    else:
        sat_data["acquisition_date"] = pd.to_datetime(sat_data["acquisition_date"], errors="coerce")
        sat_data["chlorophyll_index"] = pd.to_numeric(sat_data.get("chlorophyll_index", 0), errors="coerce")
        sat_data["turbidity_index"] = pd.to_numeric(sat_data.get("turbidity_index", 0), errors="coerce")
        sat_data["cloud_coverage"] = pd.to_numeric(sat_data.get("cloud_coverage", 0), errors="coerce")

        # Top metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", f"{len(sat_data):,}")
        m2.metric("Avg Cloud Coverage", f"{sat_data['cloud_coverage'].mean():.1f}%")
        m3.metric("Unique Lakes", sat_data["lake_name"].nunique() if "lake_name" in sat_data.columns else 0)
        m4.metric("Date Range", f"{sat_data['acquisition_date'].min():%Y-%m} → {sat_data['acquisition_date'].max():%Y-%m}"
                   if sat_data["acquisition_date"].notna().any() else "N/A")

        st.markdown("")

        left, right = st.columns(2)
        with left:
            # Chlorophyll distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=sat_data["chlorophyll_index"].dropna(),
                nbinsx=40,
                marker_color="#06b6d4",
                opacity=0.8,
                name="Chlorophyll Index",
            ))
            fig.update_layout(
                title=dict(text="Chlorophyll Index Distribution", font=dict(size=14, color="#f9fafb")),
                xaxis_title="Chlorophyll Index",
                yaxis_title="Count",
                height=300,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            # Turbidity distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=sat_data["turbidity_index"].dropna(),
                nbinsx=40,
                marker_color="#f59e0b",
                opacity=0.8,
                name="Turbidity Index",
            ))
            fig.update_layout(
                title=dict(text="Turbidity Index Distribution", font=dict(size=14, color="#f9fafb")),
                xaxis_title="Turbidity Index",
                yaxis_title="Count",
                height=300,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Chlorophyll vs Turbidity scatter
        if sat_data["chlorophyll_index"].notna().sum() > 10:
            fig = px.scatter(
                sat_data.dropna(subset=["chlorophyll_index", "turbidity_index"]),
                x="chlorophyll_index",
                y="turbidity_index",
                color="cloud_coverage",
                color_continuous_scale="Viridis",
                opacity=0.6,
                title="Chlorophyll vs Turbidity (color = cloud coverage)",
            )
            fig.update_layout(height=400, **PLOTLY_LAYOUT)
            fig.update_coloraxes(colorbar=dict(
                title="Cloud %",
                tickfont=dict(size=10, color="#9ca3af"),
                titlefont=dict(size=11, color="#9ca3af"),
            ))
            st.plotly_chart(fig, use_container_width=True)

        # Data table
        with st.expander("Raw Data Table"):
            st.dataframe(
                sat_data.sort_values("acquisition_date", ascending=False).head(100),
                use_container_width=True,
            )


# =========================================================================
# TAB 4: VISUAL ANALYSIS
# =========================================================================
with tabs[4]:
    st.markdown('<div class="section-header">VISIOS Visual Bloom Detection</div>', unsafe_allow_html=True)

    summary = _load_visios_summary()

    if "error" in summary:
        st.info("No images found. Upload lake photos via the sidebar to begin analysis.")
    else:
        stats = summary.get("statistics", {})
        dist = summary.get("summary", {})
        high_risk = summary.get("high_risk_locations", [])

        # Key metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Images Analyzed", stats.get("total_images", 0))
        m2.metric("Avg Bloom Score", f"{stats.get('average_bloom_score', 0):.1%}")
        m3.metric("High Risk Detections", len(high_risk))
        m4.metric("GPS-Tagged Images", stats.get("gps_images", 0))

        st.markdown("")

        left, right = st.columns(2)
        with left:
            # Classification donut
            if dist:
                class_labels = list(dist.keys())
                class_values = list(dist.values())
                class_colors = ["#22c55e", "#eab308", "#f97316", "#ef4444"]
                if len(class_colors) < len(class_labels):
                    class_colors += ["#6b7280"] * (len(class_labels) - len(class_colors))
                fig = _build_donut(class_labels, class_values, class_colors[:len(class_labels)], "Classes")
                st.plotly_chart(fig, use_container_width=True)

        with right:
            # Classification bar
            if dist:
                fig = go.Figure()
                bar_colors = ["#22c55e", "#eab308", "#f97316", "#ef4444"]
                if len(bar_colors) < len(dist):
                    bar_colors += ["#6b7280"] * (len(dist) - len(bar_colors))
                fig.add_trace(go.Bar(
                    x=list(dist.keys()),
                    y=list(dist.values()),
                    marker_color=bar_colors[:len(dist)],
                    text=list(dist.values()),
                    textposition="outside",
                    textfont=dict(color="#f9fafb", size=12),
                ))
                fig.update_layout(
                    title=dict(text="Classification Distribution", font=dict(size=14, color="#f9fafb")),
                    height=300,
                    xaxis=dict(tickangle=-20),
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)

        # High risk alerts
        if high_risk:
            st.markdown('<div class="section-header">High Risk Detections</div>', unsafe_allow_html=True)
            for item in high_risk:
                name = item if isinstance(item, str) else item.get("image", "unknown")
                score = item.get("score", "N/A") if isinstance(item, dict) else ""
                st.markdown(f"""
                <div class="glass-card" style="border-left: 3px solid #ef4444; margin-bottom: 8px; padding: 10px 16px;">
                    <span style="font-weight: 600;">{name}</span>
                    <span style="color: #ef4444; float: right;">{score}</span>
                </div>
                """, unsafe_allow_html=True)

    # Image list
    with st.expander("All Available Images"):
        agent = VisiosAgent()
        images = agent.list_images()
        if images:
            img_cols = st.columns(min(4, len(images)))
            for i, img_name in enumerate(images[:20]):
                with img_cols[i % 4]:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align:center; padding: 10px;">
                        <div style="font-size: 24px;">📷</div>
                        <div style="font-size: 11px; color: #9ca3af; word-break: break-all;">{img_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.write("No images found.")


# =========================================================================
# TAB 5: KNOWLEDGE BASE
# =========================================================================
with tabs[5]:
    st.markdown('<div class="section-header">RAG Knowledge Base</div>', unsafe_allow_html=True)

    # --- Section 1: Upload ---
    st.markdown("**Upload Documents**")
    kb_uploads = st.file_uploader(
        "Upload PDF, TXT, or CSV files to the knowledge base",
        type=["pdf", "txt", "csv", "md"],
        accept_multiple_files=True,
        key="kb_upload",
    )
    kb_category = st.selectbox(
        "Category",
        ["uploaded", "policy", "climate", "report", "lake_info"],
        key="kb_category",
    )
    if kb_uploads and st.button("Ingest Documents", key="kb_ingest_btn"):
        from swim.rag.ingest import ingest_uploaded_bytes
        total_chunks = 0
        for f in kb_uploads:
            with st.spinner(f"Ingesting {f.name}..."):
                try:
                    n = ingest_uploaded_bytes(f.name, f.read(), category=kb_category)
                    total_chunks += n
                    st.success(f"{f.name}: {n} chunks added")
                except Exception as exc:
                    st.error(f"{f.name}: {exc}")
        if total_chunks:
            st.success(f"Total: {total_chunks} chunks ingested")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # --- Section 2: Stats ---
    kb_left, kb_right = st.columns(2)
    with kb_left:
        st.markdown("**Knowledge Base Stats**")
        if st.button("Refresh Stats", key="kb_refresh"):
            st.cache_data.clear()
        try:
            from swim.rag.ingest import get_kb_stats
            stats = get_kb_stats()
            st.metric("Total Documents", stats["total_documents"])
            st.markdown(f"Index built: **{'Yes' if stats['index_built'] else 'No'}**")
        except Exception as exc:
            stats = {"total_documents": 0, "categories": {}, "index_built": False}
            st.warning(f"Could not load stats: {exc}")

    with kb_right:
        cats = stats.get("categories", {})
        if cats:
            cat_labels = list(cats.keys())
            cat_values = list(cats.values())
            cat_colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]
            if len(cat_colors) < len(cat_labels):
                cat_colors += ["#6b7280"] * (len(cat_labels) - len(cat_colors))
            fig = _build_donut(cat_labels, cat_values, cat_colors[:len(cat_labels)], "Categories")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No documents in the knowledge base yet.")

    if stats.get("total_documents", 0) > 0:
        if st.button("Rebuild Index", key="kb_rebuild"):
            from swim.rag.document_processor import knowledge_base as _kb
            with st.spinner("Building index..."):
                _kb.build_index()
                _kb.save()
            st.success("Index rebuilt and saved")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # --- Section 3: Query ---
    st.markdown("**Query Knowledge Base**")
    kb_question = st.text_input("Enter your question", key="kb_question", placeholder="e.g. What are HABs?")
    kb_top_k = st.slider("Number of results", 1, 10, 3, key="kb_top_k")

    if kb_question and st.button("Search", key="kb_search_btn"):
        from swim.rag.ingest import query_kb
        with st.spinner("Searching..."):
            results = query_kb(kb_question, top_k=kb_top_k)
        if results:
            for r in results:
                score_color = "#22c55e" if r["score"] > 0.5 else "#eab308" if r["score"] > 0.2 else "#6b7280"
                st.markdown(f"""
                <div class="glass-card" style="margin-bottom: 10px; border-left: 3px solid {score_color};">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                        <span style="font-size: 11px; color: #9ca3af;">{r['source']}</span>
                        <span style="font-size: 11px; color: {score_color};">Score: {r['score']:.4f}</span>
                    </div>
                    <div style="font-size: 12px; color: #d1d5db;">{r['text'][:500]}</div>
                    <div style="margin-top: 4px;">
                        <span class="risk-badge risk-low" style="font-size: 10px;">{r['category']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No results found.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # --- Section 4: Browse ---
    st.markdown("**Browse Documents**")
    try:
        from swim.rag.ingest import get_kb_stats as _get_stats
        browse_stats = _get_stats()
        browse_cats = browse_stats.get("categories", {})
        if browse_cats:
            from swim.rag.document_processor import knowledge_base as _browse_kb
            grouped = {}
            for doc in _browse_kb.documents:
                grouped.setdefault(doc.category, []).append(doc)
            for cat, docs in sorted(grouped.items()):
                with st.expander(f"{cat} ({len(docs)} documents)"):
                    for doc in docs[:50]:
                        st.markdown(f"**{doc.source}** — {doc.text[:200]}...")
        else:
            st.info("Knowledge base is empty. Upload documents above to get started.")
    except Exception:
        st.info("Knowledge base is empty. Upload documents above to get started.")


# =========================================================================
# TAB 6: PIPELINE CONTROL
# =========================================================================
with tabs[6]:
    st.markdown('<div class="section-header">Orchestrator Pipeline Control</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 2])

    with left:
        use_a2a = st.checkbox("Use A2A Protocol", value=False, key="orch_a2a")
        lake_options = ["custom"] + list(GERMAN_LAKES.keys())
        selected_lake = st.selectbox("Target Lake", lake_options, key="orch_lake")

        if selected_lake == "custom":
            lat = st.number_input("Latitude", value=47.5, format="%.4f", key="orch_lat")
            lon = st.number_input("Longitude", value=9.2, format="%.4f", key="orch_lon")
        else:
            lat, lon = None, None

        horizon = st.selectbox(
            "Forecast Horizon",
            [3, 7, 14],
            index=1,
            format_func=lambda x: f"{x} days",
            key="orch_horizon",
        )

        agent = VisiosAgent()
        images = agent.list_images()
        image_choice = st.selectbox("VISIOS Image", ["none"] + images, key="orch_img")
        skip_homogen = st.checkbox("Skip HOMOGEN", value=False, key="orch_skip")

        run_btn = st.button("Execute Pipeline", key="orch_run", use_container_width=True)

    with right:
        # Pipeline visualization
        st.markdown("""
        <div class="glass-card">
            <div style="text-align: center; padding: 12px 0;">
                <div style="display: flex; justify-content: center; align-items: center; gap: 8px; flex-wrap: wrap;">
                    <div style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3);
                                border-radius: 8px; padding: 8px 14px; font-size: 12px; font-weight: 600;">
                        📦 HOMOGEN
                    </div>
                    <div style="color: #6b7280; font-size: 16px;">→</div>
                    <div style="display: flex; flex-direction: column; gap: 4px;">
                        <div style="background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.3);
                                    border-radius: 8px; padding: 6px 14px; font-size: 12px; font-weight: 600;">
                            🛰️ CALIBRO
                        </div>
                        <div style="background: rgba(139, 92, 246, 0.15); border: 1px solid rgba(139, 92, 246, 0.3);
                                    border-radius: 8px; padding: 6px 14px; font-size: 12px; font-weight: 600;">
                            📸 VISIOS
                        </div>
                    </div>
                    <div style="color: #6b7280; font-size: 16px;">→</div>
                    <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3);
                                border-radius: 8px; padding: 8px 14px; font-size: 12px; font-weight: 600;">
                        🧠 PREDIKT
                    </div>
                    <div style="color: #6b7280; font-size: 16px;">→</div>
                    <div style="background: rgba(236, 72, 153, 0.15); border: 1px solid rgba(236, 72, 153, 0.3);
                                border-radius: 8px; padding: 8px 14px; font-size: 12px; font-weight: 600;">
                        🎯 RISK FUSION
                    </div>
                </div>
                <div style="margin-top: 8px; font-size: 11px; color: #6b7280;">
                    Sequential → Parallel → Sequential → Fusion
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        if run_btn:
            if selected_lake != "custom":
                lake_meta = GERMAN_LAKES[selected_lake]
                location = {
                    "name": selected_lake,
                    "latitude": lake_meta["lat"],
                    "longitude": lake_meta["lon"],
                }
            else:
                location = {"name": "custom_location", "latitude": lat, "longitude": lon}

            if use_a2a:
                with st.spinner("Running pipeline via A2A protocol..."):
                    import asyncio
                    from swim.agents.orchestrator.a2a_orchestrator import SWIMOrchestrator

                    async def _run():
                        orch = SWIMOrchestrator()
                        try:
                            return await orch.run_pipeline(
                                location=location,
                                horizon_days=horizon,
                                image_name=None if image_choice == "none" else image_choice,
                            )
                        finally:
                            await orch.close()

                    result = asyncio.run(_run())
                    st.session_state["pipeline_result"] = result
            else:
                with st.spinner("Running full orchestration..."):
                    from swim.agents.main_agent.controller import MainAgentController

                    controller = MainAgentController()
                    result = controller.run_full_pipeline(
                        location=location,
                        horizon_days=horizon,
                        image_name=None if image_choice == "none" else image_choice,
                        run_homogen=not skip_homogen,
                    )
                    st.session_state["pipeline_result"] = result

        result = st.session_state.get("pipeline_result")
        if result:
            protocol = result.get("metadata", {}).get("protocol", "direct")
            risk = result.get("risk_assessment", {})
            score = risk.get("score", risk.get("calibrated_score", 0))
            level = risk.get("level", "unknown")

            st.markdown(f"""
            <div class="glass-card" style="border-left: 3px solid {_risk_color(level)};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 11px; color: #6b7280; text-transform: uppercase;">Pipeline Result</div>
                        <div style="font-size: 24px; font-weight: 800; color: {_risk_color(level)};">
                            {score:.3f}
                        </div>
                    </div>
                    <div>{_risk_badge_html(level)}</div>
                </div>
                <div style="font-size: 11px; color: #6b7280; margin-top: 4px;">Protocol: {protocol}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Full JSON Response"):
                st.json(result)


# =========================================================================
# TAB 7: RISK ANALYTICS
# =========================================================================
with tabs[7]:
    st.markdown('<div class="section-header">Historical Risk Analytics</div>', unsafe_allow_html=True)

    runs = _load_pipeline_runs(limit=500)

    if not runs:
        st.info("No pipeline runs recorded yet. Execute the pipeline to start collecting analytics data.")
    else:
        runs_df = pd.DataFrame(runs)
        runs_df["created_at"] = pd.to_datetime(runs_df["created_at"], errors="coerce")
        runs_df = runs_df.sort_values("created_at")

        # Top-level stats
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Runs", len(runs_df))
        m2.metric("Unique Locations", runs_df["location_name"].nunique())
        avg_risk = runs_df["risk_score"].mean()
        m3.metric("Avg Risk Score", f"{avg_risk:.3f}")
        critical = len(runs_df[runs_df["risk_level"] == "critical"])
        m4.metric("Critical Alerts", critical)
        avg_time = runs_df["total_time_seconds"].mean() if "total_time_seconds" in runs_df.columns else 0
        m5.metric("Avg Pipeline Time", f"{avg_time:.1f}s")

        st.markdown("")

        # Risk timeline
        st.markdown('<div class="section-header">Risk Score Timeline</div>', unsafe_allow_html=True)
        fig = _build_risk_timeline(runs_df)
        st.plotly_chart(fig, use_container_width=True)

        # Distribution + Per-location side by side
        left, right = st.columns(2)

        with left:
            st.markdown('<div class="section-header">Risk Level Distribution</div>', unsafe_allow_html=True)
            level_counts = runs_df["risk_level"].value_counts()
            fig = _build_donut(
                labels=level_counts.index.tolist(),
                values=level_counts.values.tolist(),
                colors=[_risk_color(l) for l in level_counts.index],
                title="Risk",
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown('<div class="section-header">Per-Location Summary</div>', unsafe_allow_html=True)
            if "location_name" in runs_df.columns:
                loc_summary = runs_df.groupby("location_name").agg(
                    Runs=("risk_score", "count"),
                    Avg_Risk=("risk_score", "mean"),
                    Max_Risk=("risk_score", "max"),
                    Avg_Time=("total_time_seconds", "mean"),
                ).round(3)
                st.dataframe(loc_summary, use_container_width=True)

        # Recent runs table
        with st.expander("Recent Pipeline Runs"):
            display_cols = [c for c in [
                "created_at", "location_name", "risk_level", "risk_score",
                "total_time_seconds", "errors", "trace_id",
            ] if c in runs_df.columns]
            st.dataframe(
                runs_df[display_cols].sort_values("created_at", ascending=False).head(50),
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("")
st.markdown(f"""
<div style="text-align: center; padding: 16px 0; border-top: 1px solid rgba(75,85,99,0.3);">
    <span style="color: #6b7280; font-size: 11px;">
        SWIM Platform v2.0 | Multi-Agent HABs Early Warning System |
        Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
        Powered by Google A2A Protocol
    </span>
</div>
""", unsafe_allow_html=True)
