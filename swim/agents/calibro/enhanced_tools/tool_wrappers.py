# swim/agents/calibro/tools/tool_wrappers.py
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
    df = pd.read_csv(io.StringIO(df_str))
    result = trend_analyzer.analyze_water_quality_trends(df, lake_name)
    return str(result)

@tool
def find_water_bodies_tool(query: str) -> str:
    results = lake_finder.discover_water_bodies_by_location(query)
    return str(results)

@tool
def assess_bloom_risk_tool(df_str: str, lake_name: str) -> str:
    df = pd.read_csv(io.StringIO(df_str))
    result = bloom_model.assess_bloom_risk(df, lake_name)
    return str(result)