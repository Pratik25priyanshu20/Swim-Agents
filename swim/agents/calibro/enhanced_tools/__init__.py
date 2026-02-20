#swim/agents/calibro/enhanced_tools/__init__.py
from .trend_analysis import analyze_temporal_trends
from .bloom_risk import assess_bloom_risk_comprehensive
from .tool_wrappers import (
    analyze_specific_water_body,
    compare_multiple_water_bodies
)
from .lake_finder import EnhancedWaterBodyProcessor

# Instantiate the processor
lake_processor = EnhancedWaterBodyProcessor()

# Expose callable functions
discover_water_bodies_near_location = lake_processor.discover_water_bodies_by_location

# Class wrapper for other orchestrator tools
class CalibroEnhancedTools:
    def __init__(self):
        pass

    def discover_water_bodies_near_location(self, **kwargs):
        return discover_water_bodies_near_location(**kwargs)

    def analyze_temporal_trends(self, **kwargs):
        return analyze_temporal_trends(**kwargs)

    def assess_bloom_risk_comprehensive(self, **kwargs):
        return assess_bloom_risk_comprehensive(**kwargs)

    def analyze_specific_water_body(self, **kwargs):
        return analyze_specific_water_body(**kwargs)

    def compare_multiple_water_bodies(self, **kwargs):
        return compare_multiple_water_bodies(**kwargs)