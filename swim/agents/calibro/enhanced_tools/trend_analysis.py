#swim/agents/calibro/enhanced_tools/trend_analysis.py

import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime
from scipy import stats

class AdvancedTrendAnalyzer:
    """Analyze trends in water quality metrics (chlorophyll-a and turbidity)."""
    
    def __init__(self):
        self.required_columns = ['date', 'Chl_a_mg_m3', 'Turbidity_FNU']
    
    def analyze_water_quality_trends(self, df: pd.DataFrame, lake_name: str) -> Dict:
        """Analyze temporal trends in water quality metrics."""
        if df.empty or any(col not in df.columns for col in self.required_columns):
            return {"error": "Dataframe missing required columns or empty."}
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Numerical index for time
        df['time_index'] = (df['date'] - df['date'].min()).dt.days

        trend_results = {
            "lake_name": lake_name,
            "trend_analysis_date": datetime.now().strftime('%Y-%m-%d'),
            "chlorophyll_a": self._analyze_trend(df, 'Chl_a_mg_m3'),
            "turbidity": self._analyze_trend(df, 'Turbidity_FNU')
        }

        return trend_results

    def _analyze_trend(self, df: pd.DataFrame, column: str) -> Dict:
        """Perform trend analysis on a single parameter."""
        series = df[['time_index', column]].dropna()
        if len(series) < 5:
            return {"status": "insufficient data"}

        x = series['time_index'].values
        y = series[column].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        trend_type = self._classify_trend(slope, p_value)

        return {
            "slope": round(slope, 4),
            "intercept": round(intercept, 2),
            "r_squared": round(r_value**2, 3),
            "p_value": round(p_value, 3),
            "trend": trend_type,
            "sample_size": len(series)
        }

    def _classify_trend(self, slope: float, p_value: float) -> str:
        """Classify trend direction and significance."""
        if p_value > 0.1:
            return "No significant trend"
        if slope > 0:
            return "Increasing"
        elif slope < 0:
            return "Decreasing"
        return "Stable"
    
    
analyzer = AdvancedTrendAnalyzer()

    
def analyze_temporal_trends(**kwargs):
    return analyzer.analyze_water_quality_trends(**kwargs)
    
