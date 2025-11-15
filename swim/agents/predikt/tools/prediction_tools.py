# swim/agents/predikt/tools/prediction_tools.py

from langchain.tools import tool
from typing import Optional
from datetime import datetime
from swim.agents.predikt.predikt_agent import PrediktAgent

# Initialize agent ONCE
agent = PrediktAgent()


# ============================================================
# 1. PREDICT SINGLE LAKE
# ============================================================
@tool
def predict_single_location(location_name: str, horizon_days: str = "7") -> str:
    """
    Predict HABs risk for a specific German lake (3, 7, 14‑day forecasts).
    """
    try:
        horizon = int(horizon_days)
    except ValueError:
        return "❌ horizon_days must be 3, 7, or 14"

    if horizon not in [3, 7, 14]:
        return "❌ horizon_days must be 3, 7, or 14"

    if location_name not in agent.GERMAN_LAKES:
        available = ", ".join(agent.GERMAN_LAKES.keys())
        return f"❌ Lake '{location_name}' not found. Available: {available}"

    lake_info = agent.GERMAN_LAKES[location_name]
    location = {"name": location_name, "latitude": lake_info["lat"], "longitude": lake_info["lon"]}

    result = agent.predict_bloom_probability(location, horizon_days=horizon)

    output = f"""🔮 **HABs Prediction – {location_name}**

**Forecast Period:** {result['prediction_horizon_days']} days  
**Valid Until:** {result['valid_until'][:10]}

**Risk Assessment:**  
  • Risk Level: {result['risk_level'].upper()}  
  • Bloom Probability: {result['bloom_probability']:.1%}  
  • Confidence: {result['confidence']:.1%}  
  • Uncertainty: ±{result['uncertainty']:.1%}  
  • Interval: [{result['prediction_interval']['lower']:.1%} – {result['prediction_interval']['upper']:.1%}]

**Top Factors:**  
"""
    if result["contributing_factors"]:
        for factor in result["contributing_factors"][:3]:
            output += f"  • {factor['factor']}: {factor['value']} {factor['unit']} ({factor['status']})\n"
    else:
        output += "  • No significant risk factors detected\n"

    cond = result["current_conditions"]

    output += f"""
**Current Conditions:**  
  • Chlorophyll‑a: {cond['chlorophyll_a']:.1f} µg/L  
  • Water Temp: {cond['water_temperature']:.1f}°C  
  • Turbidity: {cond['turbidity']:.1f} NTU  

**Model:** {result['model_used'].replace('_',' ').title()}  
**Accuracy:** {result['base_accuracy']:.1%}  
*Predicted at: {result['predicted_at'][:19]}*
"""
    return output


# ============================================================
# 2. FORECAST ALL GERMAN LAKES
# ============================================================
@tool
def forecast_all_german_lakes(horizon_days: str = "7") -> str:
    """Forecast HABs risk for all monitored German lakes."""
    try:
        horizon = int(horizon_days)
    except ValueError:
        return "❌ horizon_days must be a number"

    forecast = agent.predict_german_lakes(horizon_days=horizon)
    summary = forecast["summary"]

    output = f"""🗺️ **German Lakes HABs Forecast**

**Summary:**  
  • Total Lakes: {summary['total_lakes']}  
  • Horizon: {summary['prediction_horizon_days']} days  
  • Avg Risk: {summary['average_bloom_probability']:.1%}  
  • Range: {summary['min_bloom_probability']:.1%} – {summary['max_bloom_probability']:.1%}

**Risk Distribution:**  
"""
    for risk in ["low", "moderate", "high", "critical"]:
        count = summary["risk_distribution"].get(risk, 0)
        bar = "█" * count
        output += f"  {risk.capitalize():10s} | {bar} {count}\n"

    if summary["high_risk_lakes"]:
        output += f"\n⚠️ **High‑Risk Lakes:** {', '.join(summary['high_risk_lakes'])}\n"
    else:
        output += "\n✅ No high‑risk lakes detected.\n"

    output += "\n**Individual Forecasts:**\n"
    risk_emoji = {"low": "🟢", "moderate": "🟡", "high": "🟠", "critical": "🔴"}

    for name, pred in sorted(forecast["predictions"].items(), key=lambda x: x[1]["bloom_probability"], reverse=True):
        output += f"  {risk_emoji[pred['risk_level']]} {name}: {pred['bloom_probability']:.1%} ({pred['risk_level']})\n"

    output += f"\n*Generated: {summary['predicted_at'][:19]}*"
    return output


# ============================================================
# 3. COMPARE 3/7/14‑DAY HORIZONS
# ============================================================
@tool
def compare_prediction_horizons(location_name: str) -> str:
    """Compare 3‑day, 7‑day, and 14‑day forecasts for a lake."""
    if location_name not in agent.GERMAN_LAKES:
        return f"❌ '{location_name}' not found."

    lake = agent.GERMAN_LAKES[location_name]
    location = {"name": location_name, "latitude": lake["lat"], "longitude": lake["lon"]}

    preds = {h: agent.predict_bloom_probability(location, h) for h in [3, 7, 14]}

    output = f"📊 **Prediction Horizon Comparison – {location_name}**\n\n"

    for h in [3, 7, 14]:
        p = preds[h]
        output += f"""**{h}‑Day Forecast:**  
  • Probability: {p['bloom_probability']:.1%}  
  • Risk: {p['risk_level'].capitalize()}  
  • Confidence: {p['confidence']:.1%}  
  • Uncertainty: ±{p['uncertainty']:.1%}  
  • Range: [{p['prediction_interval']['lower']:.1%} – {p['prediction_interval']['upper']:.1%}]  
  • Accuracy: {p['base_accuracy']:.1%}\n\n"""

    output += """**Insights:**  
  • 3‑day → Highest accuracy  
  • 7‑day → Balanced forecast  
  • 14‑day → Strategic, higher uncertainty  
"""
    return output


# ============================================================
# 4. RISK FACTORS EXPLANATION
# ============================================================
@tool
def get_risk_factors_explanation(location_name: str, horizon_days: str = "7") -> str:
    """Explain environmental risk factors contributing to bloom risk."""
    try:
        horizon = int(horizon_days)
    except ValueError:
        return "❌ horizon_days must be numeric"

    if location_name not in agent.GERMAN_LAKES:
        return f"❌ Lake '{location_name}' not found."

    loc = agent.GERMAN_LAKES[location_name]
    location = {"name": location_name, "latitude": loc["lat"], "longitude": loc["lon"]}

    result = agent.predict_bloom_probability(location, horizon_days=horizon)

    output = f"""🔬 **Risk Factor Analysis – {location_name}**

**Overall:** {result['risk_level'].upper()} | {result['bloom_probability']:.1%}

**Contributing Factors:**  
"""

    if not result["contributing_factors"]:
        output += "  • No significant contributing factors.\n"
    else:
        for factor in result["contributing_factors"]:
            bar = "█" * int(factor["importance"] * 20)
            output += f"  • {factor['factor']}: {factor['value']} {factor['unit']} ({factor['status']}) {bar}\n"

    cond = result["current_conditions"]

    output += f"""
**Current Conditions:**  
  • Chlorophyll‑a: {cond['chlorophyll_a']:.1f} µg/L  
  • Water Temp: {cond['water_temperature']:.1f}°C  
  • Turbidity: {cond['turbidity']:.1f} NTU  
  • pH: {cond['ph']:.1f}  
  • DO: {cond['dissolved_oxygen']:.1f} mg/L  
  • Nitrogen: {cond['total_nitrogen']:.2f} mg/L  
  • Phosphorus: {cond['total_phosphorus']:.3f} mg/L  
  • Wind: {cond['wind_speed']:.1f} m/s  
  • Solar: {cond['solar_radiation']:.0f} W/m²  
"""

    return output


# ============================================================
# 5. HISTORY
# ============================================================
@tool
def get_prediction_history(limit: str = "10") -> str:
    """Return recent prediction history."""
    try:
        limit = int(limit)
    except ValueError:
        limit = 10

    history = agent.prediction_history[-limit:]

    if not history:
        return "📜 No prediction history available."

    output = f"📜 **Recent Predictions (Last {len(history)}):**\n\n"

    for entry in reversed(history):
        output += f"""• {entry['location']} ({entry['horizon_days']}‑day)
  Probability: {entry['probability']:.1%} | Risk: {entry['risk_level'].capitalize()}
  Time: {entry['timestamp'][:19]}

"""
    return output


# ============================================================
# 6. AGENT STATUS
# ============================================================
@tool
def get_agent_status() -> str:
    """System health, performance, accuracy, recent trends."""
    info = agent.get_agent_info()

    total_preds = len(agent.prediction_history)

    if agent.prediction_history:
        recent = agent.prediction_history[-10:]
        avg_risk = sum(p["probability"] for p in recent) / len(recent)
        trend = "increasing" if recent[-1]["probability"] > recent[0]["probability"] else "stable"
    else:
        avg_risk = 0
        trend = "no data"

    output = f"""⚙️ **PREDIKT Agent Status**

**System:** {info['status']}  
**Version:** {info['version']}  
**Function:** {info['function']}

**Performance:**  
  • Total Predictions: {total_preds}  
  • Recent Avg Risk: {avg_risk:.1%}  
  • Trend: {trend.capitalize()}  
  • Monitored Lakes: {len(info['german_lakes'])}

**Model:**  
  • Architecture: {info['model_config']['architecture']}  
  • Base Accuracy: {info['model_config']['base_accuracy']:.1%}  
  • Spatial Resolution: {info['model_config']['spatial_resolution_km']} km  
  • Temporal Resolution: {info['model_config']['temporal_resolution_hours']} hr  

**Risk Thresholds:**  
  Low: 0–30% | Moderate: 30–60% | High: 60–80% | Critical: 80–100%
"""
    return output


# ============================================================
# 7. WEEKLY REPORT
# ============================================================
@tool
def generate_weekly_report(include_history: str = "true") -> str:
    """Produce a weekly 7‑day HABs forecast report."""
    forecast = agent.predict_german_lakes(7)
    summary = forecast["summary"]

    output = f"""📋 **Weekly HABs Forecast Report**
==================================================

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Period:** 7‑day forecast  

**Summary:**  
  • Lakes: {summary['total_lakes']}  
  • Avg Risk: {summary['average_bloom_probability']:.1%}  
  • Range: {summary['min_bloom_probability']:.1%} – {summary['max_bloom_probability']:.1%}  

**Risk Distribution:**  
"""
    dist = summary["risk_distribution"]
    for r in ["low", "moderate", "high", "critical"]:
        count = dist.get(r, 0)
        pct = (count / summary["total_lakes"]) * 100
        bar = "█" * int(pct / 5)
        output += f"  {r.capitalize():10s} | {bar} {pct:.0f}%\n"

    if summary["high_risk_lakes"]:
        output += f"\n⚠️ **High‑Risk Lakes:**\n"
        for lake in summary["high_risk_lakes"]:
            pred = forecast["predictions"][lake]
            output += f"  • {lake}: {pred['bloom_probability']:.1%} ({pred['risk_level']})\n"
    else:
        output += "\nNo high‑risk alerts.\n"

    output += "\n**Detailed Forecasts:**\n"
    emoji = {"low": "🟢", "moderate": "🟡", "high": "🟠", "critical": "🔴"}

    for name, pred in sorted(forecast["predictions"].items(), key=lambda x: x[1]["bloom_probability"], reverse=True):
        output += f"  {emoji[pred['risk_level']]} {name}: {pred['bloom_probability']:.1%} ({pred['risk_level']})\n"

    if include_history.lower() == "true" and agent.prediction_history:
        output += "\n**Recent History:**\n"
        for entry in agent.prediction_history[-7:][::-1]:
            output += f"  • {entry['location']}: {entry['probability']:.1%} ({entry['timestamp'][:10]})\n"

    output += f"""
==================================================
Model: {agent.model_config['architecture']}  
Accuracy: {agent.model_config['base_accuracy']:.1%}  

Generated by PREDIKT v{agent.version}
"""
    return output


# ============================================================
# 8. CUSTOM LOCATION
# ============================================================
@tool
def predict_custom_location(latitude: str, longitude: str, location_name: str = "Custom", horizon_days: str = "7") -> str:
    """Predict HAB risk at any lat/lon."""
    try:
        lat = float(latitude)
        lon = float(longitude)
        horizon = int(horizon_days)
    except ValueError:
        return "❌ Invalid input. Use numerical values."

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return "❌ Invalid geographic coordinates."

    if horizon not in [3, 7, 14]:
        return "❌ horizon_days must be 3, 7, or 14"

    result = agent.predict_bloom_probability(
        {"name": location_name, "latitude": lat, "longitude": lon},
        horizon_days=horizon,
    )

    output = f"""🌍 **Custom Location HABs Prediction**

**Location:** {location_name}  
**Coordinates:** {lat:.4f}°N, {lon:.4f}°E  
**Horizon:** {horizon} days  

**Risk:** {result['risk_level'].upper()}  
**Probability:** {result['bloom_probability']:.1%}  
**Confidence:** {result['confidence']:.1%}  
**Uncertainty:** ±{result['uncertainty']:.1%}  

Valid Until: {result['valid_until'][:10]}
"""
    return output


# ============================================================
# 9. AGENT INFO
# ============================================================
@tool
def get_predikt_info() -> str:
    """Return agent configuration & model metadata."""
    info = agent.get_agent_info()
    cfg = agent.get_training_details()

    output = f"""🔮 **PREDIKT Agent Overview**

**Function:** {info['function']}  
**Version:** {info['version']}  
**Status:** {info['status']}  

**Capabilities:**  
  • Monitored Lakes: {len(info['german_lakes'])}  
  • Horizons: {', '.join(map(str, info['prediction_horizons']))} days  
  • Risk Levels: {', '.join(info['risk_levels'])}  
  • Total Predictions: {info['total_predictions']}  

**Model Configuration:**  
  • Architecture: {cfg['architecture']}  
  • Training Data: {cfg['training_data']}  
  • Records: {cfg['training_records']:,}  
  • Accuracy: {cfg['base_accuracy']:.1%}  
  • Validation: {cfg['validation']}  
  • Duration: {cfg['training_duration']}  
"""
    return output