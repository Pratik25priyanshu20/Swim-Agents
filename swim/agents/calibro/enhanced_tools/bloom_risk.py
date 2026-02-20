# swim/agents/calibro/enhanced_tools/bloom_risk.py
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
from scipy import stats

class AdvancedRiskAssessment:
    """Advanced harmful algal bloom risk assessment"""
    
    def __init__(self):
        self.bloom_risk_threshold = 20.0  # mg/m³ chlorophyll-a
        self.high_risk_threshold = 30.0
        
    def assess_bloom_risk(self, df: pd.DataFrame, lake_name: str) -> Dict:
        """Comprehensive harmful algal bloom risk assessment"""
        if df.empty:
            return {"error": "No data available for risk assessment"}
        
        print(f"🚨 Performing bloom risk assessment for {lake_name}...")
        
        risk_factors = {}
        
        # Current chlorophyll-a levels
        if 'Chl_a_mg_m3' in df.columns:
            current_chl = df['Chl_a_mg_m3'].tail(5).mean()
            risk_factors['chlorophyll_a'] = {
                'current_level': round(current_chl, 2),
                'threshold_ratio': current_chl / self.bloom_risk_threshold,
                'risk_level': self._assess_parameter_risk(current_chl, 'chlorophyll_a')
            }
        
        # Trend analysis for risk
        trend_risk = self._assess_trend_risk(df)
        risk_factors['trend_risk'] = trend_risk
        
        # Seasonal risk
        seasonal_risk = self._assess_seasonal_risk(df)
        risk_factors['seasonal_risk'] = seasonal_risk
        
        # Variability risk
        variability_risk = self._assess_variability_risk(df)
        risk_factors['variability_risk'] = variability_risk
        
        # Overall risk calculation
        overall_risk = self._calculate_overall_risk(risk_factors)
        
        # Generate recommendations
        recommendations = self._generate_risk_recommendations(overall_risk, risk_factors)
        
        return {
            'lake_name': lake_name,
            'assessment_date': datetime.now().strftime('%Y-%m-%d'),
            'overall_risk_level': overall_risk['level'],
            'risk_score': overall_risk['score'],
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'monitoring_frequency': self._recommend_monitoring_frequency(overall_risk['level'])
        }

    def _assess_parameter_risk(self, value: float, param_type: str) -> str:
        """Assess risk level for a parameter"""
        if param_type == 'chlorophyll_a':
            if value < 5:
                return "Low"
            elif value < 15:
                return "Moderate"
            elif value < 30:
                return "High"
            else:
                return "Very High"
        return "Unknown"

    def _assess_trend_risk(self, df: pd.DataFrame) -> Dict:
        """Assess risk based on recent trends"""
        if 'Chl_a_mg_m3' not in df.columns or len(df) < 5:
            return {"risk_level": "Unknown", "reason": "Insufficient data"}
        
        recent_data = df.tail(10)['Chl_a_mg_m3'].dropna()
        if len(recent_data) < 3:
            return {"risk_level": "Unknown", "reason": "Insufficient recent data"}
        
        # Simple trend detection
        x = np.arange(len(recent_data))
        slope, _, r_value, p_value, _ = stats.linregress(x, recent_data)
        
        if p_value < 0.1 and slope > 0.5:
            risk_level = "High"
            reason = f"Significant increasing trend detected (slope: {slope:.3f})"
        elif p_value < 0.1 and slope < -0.5:
            risk_level = "Low"
            reason = f"Decreasing trend detected (slope: {slope:.3f})"
        elif abs(slope) > 0.2:
            risk_level = "Moderate"
            reason = f"Moderate trend detected (slope: {slope:.3f})"
        else:
            risk_level = "Low"
            reason = "No significant trend detected"
        
        return {
            "risk_level": risk_level,
            "slope": round(slope, 3),
            "r_squared": round(r_value**2, 3),
            "p_value": round(p_value, 3),
            "reason": reason
        }

    def _assess_seasonal_risk(self, df: pd.DataFrame) -> Dict:
        """Assess risk based on seasonal patterns"""
        if 'date' not in df.columns:
            return {"risk_level": "Unknown", "reason": "No date information"}
        
        current_month = datetime.now().month
        
        # High-risk months (typically summer)
        high_risk_months = [6, 7, 8, 9]  # June through September
        moderate_risk_months = [4, 5, 10]  # Spring and early fall
        
        if current_month in high_risk_months:
            risk_level = "High"
            reason = "Currently in high-risk season (summer/early fall)"
        elif current_month in moderate_risk_months:
            risk_level = "Moderate"
            reason = "Currently in moderate-risk season"
        else:
            risk_level = "Low"
            reason = "Currently in low-risk season (winter/early spring)"
        
        return {
            "risk_level": risk_level,
            "current_month": current_month,
            "reason": reason
        }

    def _assess_variability_risk(self, df: pd.DataFrame) -> Dict:
        """Assess risk based on parameter variability"""
        if 'Chl_a_mg_m3' not in df.columns:
            return {"risk_level": "Unknown", "reason": "No chlorophyll data"}
        
        recent_data = df.tail(10)['Chl_a_mg_m3'].dropna()
        if len(recent_data) < 3:
            return {"risk_level": "Unknown", "reason": "Insufficient data for variability"}
            
        cv = recent_data.std() / recent_data.mean() * 100
        
        if cv > 50:
            risk_level = "High"
            reason = f"High variability detected (CV: {cv:.1f}%)"
        elif cv > 30:
            risk_level = "Moderate"
            reason = f"Moderate variability detected (CV: {cv:.1f}%)"
        else:
            risk_level = "Low"
            reason = f"Low variability (CV: {cv:.1f}%)"
        
        return {
            "risk_level": risk_level,
            "coefficient_variation": round(cv, 1),
            "reason": reason
        }

    def _calculate_overall_risk(self, risk_factors: Dict) -> Dict:
        """Calculate overall risk score and level"""
        risk_weights = {
            'chlorophyll_a': 0.4,
            'trend_risk': 0.3,
            'seasonal_risk': 0.2,
            'variability_risk': 0.1
        }
        
        risk_scores = {
            'Low': 1,
            'Moderate': 2,
            'High': 3,
            'Very High': 4,
            'Unknown': 2
        }
        
        total_score = 0
        total_weight = 0
        
        for factor, weight in risk_weights.items():
            if factor in risk_factors:
                if factor == 'chlorophyll_a':
                    score = risk_scores.get(risk_factors[factor]['risk_level'], 2)
                else:
                    score = risk_scores.get(risk_factors[factor]['risk_level'], 2)
                total_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_score = total_score / total_weight
        else:
            weighted_score = 2.0
        
        if weighted_score <= 1.5:
            level = "Low"
        elif weighted_score <= 2.5:
            level = "Moderate"
        elif weighted_score <= 3.5:
            level = "High"
        else:
            level = "Very High"
        
        return {
            'score': round(weighted_score, 2),
            'level': level
        }

    def _generate_risk_recommendations(self, overall_risk: Dict, risk_factors: Dict) -> List[str]:
        """Generate risk-based recommendations"""
        recommendations = []
        risk_level = overall_risk['level']
        
        if risk_level == "Low":
            recommendations.extend([
                "Continue routine monitoring",
                "Maintain current management practices"
            ])
        elif risk_level == "Moderate":
            recommendations.extend([
                "Increase monitoring frequency to weekly",
                "Prepare contingency plans for bloom management",
                "Monitor weather conditions for bloom-favorable patterns"
            ])
        elif risk_level == "High":
            recommendations.extend([
                "Implement daily monitoring",
                "Issue public health advisory",
                "Activate early warning systems",
                "Prepare water treatment interventions"
            ])
        else:  # Very High
            recommendations.extend([
                "Immediate action required - activate all monitoring systems",
                "Issue public health warnings - restrict water use",
                "Implement emergency response protocols",
                "Consider immediate intervention measures"
            ])
        
        # Specific recommendations based on risk factors
        if 'trend_risk' in risk_factors and risk_factors['trend_risk']['risk_level'] == "High":
            recommendations.append("Investigate sources of increasing nutrient loading")
        
        if 'seasonal_risk' in risk_factors and risk_factors['seasonal_risk']['risk_level'] == "High":
            recommendations.append("Intensify monitoring during peak bloom season")
        
        return recommendations

    def _recommend_monitoring_frequency(self, risk_level: str) -> str:
        """Recommend monitoring frequency based on risk level"""
        frequencies = {
            "Low": "Monthly",
            "Moderate": "Weekly",
            "High": "Daily",
            "Very High": "Continuous (real-time if possible)"
        }
        return frequencies.get(risk_level, "Weekly")
    
    
risk_assessor = AdvancedRiskAssessment()

# Exported callable
def assess_bloom_risk_comprehensive(**kwargs):
    return risk_assessor.assess_bloom_risk(**kwargs)
    
