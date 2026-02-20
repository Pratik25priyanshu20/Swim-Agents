# tests/test_predikt_agent.py

"""Unit tests for the PREDIKT agent core logic."""

import pytest
from swim.agents.predikt.predikt_agent import PrediktAgent
from swim.agents.predikt.config import GERMAN_LAKES, RISK_THRESHOLDS, FEATURE_WEIGHTS


class TestPrediktAgent:
    def setup_method(self):
        self.agent = PrediktAgent()

    def test_init_loads_all_lakes(self):
        assert len(self.agent.GERMAN_LAKES) == 5
        assert "Bodensee" in self.agent.GERMAN_LAKES
        assert "Müritz" in self.agent.GERMAN_LAKES

    def test_lakes_match_config(self):
        for lake in GERMAN_LAKES:
            assert lake in self.agent.GERMAN_LAKES

    def test_classify_risk_low(self):
        assert self.agent.classify_risk(0.1) == "low"
        assert self.agent.classify_risk(0.29) == "low"

    def test_classify_risk_moderate(self):
        assert self.agent.classify_risk(0.3) == "moderate"
        assert self.agent.classify_risk(0.59) == "moderate"

    def test_classify_risk_high(self):
        assert self.agent.classify_risk(0.6) == "high"
        assert self.agent.classify_risk(0.79) == "high"

    def test_classify_risk_critical(self):
        assert self.agent.classify_risk(0.8) == "critical"
        assert self.agent.classify_risk(0.95) == "critical"

    def test_predict_bloom_probability_returns_required_keys(self, sample_location):
        result = self.agent.predict_bloom_probability(sample_location, horizon_days=7)
        required_keys = [
            "location", "prediction_horizon_days", "bloom_probability",
            "confidence", "uncertainty", "risk_level", "model_used",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_bloom_probability_range(self, sample_location):
        result = self.agent.predict_bloom_probability(sample_location)
        assert 0.0 <= result["bloom_probability"] <= 1.0

    def test_confidence_range(self, sample_location):
        result = self.agent.predict_bloom_probability(sample_location)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_german_lakes_all(self):
        result = self.agent.predict_german_lakes(horizon_days=3)
        assert "summary" in result
        assert "predictions" in result
        assert result["summary"]["total_lakes"] == 5

    def test_risk_distribution_sums_correctly(self):
        result = self.agent.predict_german_lakes()
        dist = result["summary"]["risk_distribution"]
        total = sum(dist.values())
        assert total == 5

    def test_prediction_history_tracks(self, sample_location):
        before = len(self.agent.prediction_history)
        self.agent.predict_bloom_probability(sample_location)
        assert len(self.agent.prediction_history) == before + 1

    def test_get_agent_info(self):
        info = self.agent.get_agent_info()
        assert info["status"] == "operational"
        assert "Bodensee" in info["german_lakes"]

    def test_contributing_factors_returns_list(self, sample_location):
        result = self.agent.predict_bloom_probability(sample_location)
        factors = result["contributing_factors"]
        assert isinstance(factors, list)
        assert len(factors) > 0

    def test_generate_conditions_has_all_fields(self):
        cond = self.agent.generate_conditions()
        for field in ["chlorophyll_a", "water_temperature", "turbidity"]:
            assert field in cond

    def test_different_horizons_produce_different_uncertainty(self, sample_location):
        r3 = self.agent.predict_bloom_probability(sample_location, 3)
        r14 = self.agent.predict_bloom_probability(sample_location, 14)
        assert r3["uncertainty"] < r14["uncertainty"]


class TestPrediktConfig:
    def test_german_lakes_have_required_fields(self):
        for name, meta in GERMAN_LAKES.items():
            assert "lat" in meta
            assert "lon" in meta
            assert "trophic_status" in meta

    def test_risk_thresholds_cover_full_range(self):
        levels = list(RISK_THRESHOLDS.values())
        assert levels[0]["min"] == 0.0
        assert levels[-1]["max"] == 1.0

    def test_feature_weights_sum_to_one(self):
        total = sum(FEATURE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01
