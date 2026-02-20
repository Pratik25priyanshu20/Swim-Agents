# swim/agents/calibro/enhanced_tools/tool_wrappers.py
from langchain_core.tools import tool
import pandas as pd
import io

from .trend_analysis import AdvancedTrendAnalyzer
from .lake_finder import EnhancedWaterBodyProcessor
from .bloom_risk import AdvancedRiskAssessment

trend_analyzer = AdvancedTrendAnalyzer()
lake_finder = EnhancedWaterBodyProcessor()
bloom_model = AdvancedRiskAssessment()

@tool
def analyze_trends_tool(df_str: str, lake_name: str) -> str:
    """Analyze trends for a lake's water quality time series."""
    df = pd.read_csv(io.StringIO(df_str))
    result = trend_analyzer.analyze_water_quality_trends(df, lake_name)
    return str(result)

@tool
def find_water_bodies_tool(query: str) -> str:
    """Find water bodies near a location."""
    results = lake_finder.discover_water_bodies_by_location(query)
    return str(results)

@tool
def assess_bloom_risk_tool(df_str: str, lake_name: str) -> str:
    """Assess bloom risk for a specific lake based on time series."""
    df = pd.read_csv(io.StringIO(df_str))
    result = bloom_model.assess_bloom_risk(df, lake_name)
    return str(result)

@tool
def analyze_specific_water_body(water_body_name: str, csv_file: str = "") -> str:
    """Analyze the water quality of a specific lake from a CSV."""
    df = pd.read_csv(f"data/processed/{csv_file}")
    trends = trend_analyzer.analyze_water_quality_trends(df, water_body_name)
    risks = bloom_model.assess_bloom_risk(df, water_body_name)
    return f"📊 Water Quality Trends:\n{trends}\n\n🚨 Bloom Risk:\n{risks}"

@tool
def compare_multiple_water_bodies(water_body_names: list, parameter: str = "chlorophyll") -> str:
    """Compare water quality parameter trends across multiple lakes."""
    summaries = []
    for name in water_body_names:
        try:
            filename = f"data/processed/{name.lower().replace(' ', '_')}_data.csv"
            df = pd.read_csv(filename)
            trends = trend_analyzer.analyze_water_quality_trends(df, name)
            summaries.append(f"{name} → {trends[parameter]}")
        except Exception as e:
            summaries.append(f"{name} → Error: {str(e)}")
    return "\n\n".join(summaries)