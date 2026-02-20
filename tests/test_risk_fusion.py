# tests/test_risk_fusion.py

"""Unit tests for the calibrated risk fusion and probability extraction."""

import pytest


class TestCalibratedRiskFusion:
    """Test compute_calibrated_risk_fusion from the risk_fusion module."""

    def test_no_scores_returns_unknown(self):
        from swim.agents.orchestrator.risk_fusion import compute_calibrated_risk_fusion
        results = {"homogen": {"status": "error"}, "calibro": {"status": "error"}}
        risk = compute_calibrated_risk_fusion(results)
        assert risk["level"] == "unknown"
        assert risk["score"] == 0.0
        assert risk["fusion_method"] == "none"

    def test_single_agent_success(self):
        from swim.agents.orchestrator.risk_fusion import compute_calibrated_risk_fusion
        results = {
            "predikt": {"status": "success", "summary": "bloom probability: 0.45"},
            "calibro": {"status": "error"},
        }
        risk = compute_calibrated_risk_fusion(results)
        assert risk["level"] in ("low", "moderate", "high")
        assert risk["sources_used"] == 1
        assert risk["fusion_method"] == "calibrated_weighted_average"
        assert "confidence_interval" in risk
        assert risk["confidence_interval"]["lower"] <= risk["score"]
        assert risk["confidence_interval"]["upper"] >= risk["score"]

    def test_all_agents_low(self):
        from swim.agents.orchestrator.risk_fusion import compute_calibrated_risk_fusion
        results = {
            "predikt": {"status": "success", "summary": "probability: 0.10"},
            "calibro": {"status": "success", "summary": "risk score: 0.15"},
            "visios": {"status": "success", "summary": "bloom probability: 0.12"},
            "homogen": {"status": "success", "summary": '{"score": 0.20}'},
        }
        risk = compute_calibrated_risk_fusion(results)
        assert risk["level"] == "low"
        assert risk["sources_used"] == 4

    def test_all_agents_critical(self):
        from swim.agents.orchestrator.risk_fusion import compute_calibrated_risk_fusion
        results = {
            "predikt": {"status": "success", "summary": "probability: 0.90"},
            "calibro": {"status": "success", "summary": "risk score: 0.85"},
            "visios": {"status": "success", "summary": "probability: 0.88"},
            "homogen": {"status": "success", "summary": "risk score: 0.80"},
        }
        risk = compute_calibrated_risk_fusion(results)
        assert risk["level"] in ("high", "critical")

    def test_skipped_agents_ignored(self):
        from swim.agents.orchestrator.risk_fusion import compute_calibrated_risk_fusion
        results = {
            "predikt": {"status": "success", "summary": "probability: 0.50"},
            "calibro": {"status": "success", "summary": "risk score: 0.40"},
            "visios": {"status": "skipped"},
            "homogen": {"status": "success", "summary": "risk score: 0.30"},
        }
        risk = compute_calibrated_risk_fusion(results)
        assert risk["sources_used"] == 3

    def test_uncertainty_quantification(self):
        from swim.agents.orchestrator.risk_fusion import compute_calibrated_risk_fusion
        results = {
            "predikt": {"status": "success", "summary": "probability: 0.50"},
            "calibro": {"status": "success", "summary": "risk score: 0.50"},
        }
        risk = compute_calibrated_risk_fusion(results)
        assert "uncertainty" in risk
        assert "agent_disagreement" in risk["uncertainty"]
        assert "extraction_uncertainty" in risk["uncertainty"]
        assert "coverage_factor" in risk["uncertainty"]

    def test_agent_scores_detail(self):
        from swim.agents.orchestrator.risk_fusion import compute_calibrated_risk_fusion
        results = {
            "predikt": {"status": "success", "summary": "bloom probability: 0.60"},
        }
        risk = compute_calibrated_risk_fusion(results)
        assert risk["agent_scores"] is not None
        assert len(risk["agent_scores"]) == 1
        score = risk["agent_scores"][0]
        assert score["agent"] == "predikt"
        assert "raw_probability" in score
        assert "calibrated_probability" in score
        assert "extraction_source" in score

    def test_json_extraction_preferred(self):
        """JSON extraction should be used when available and have high confidence."""
        from swim.agents.orchestrator.risk_fusion import compute_calibrated_risk_fusion
        results = {
            "predikt": {"status": "success", "summary": '{"bloom_probability": 0.72, "risk_level": "high"}'},
        }
        risk = compute_calibrated_risk_fusion(results)
        assert risk["sources_used"] == 1
        assert risk["agent_scores"][0]["extraction_source"] == "json"
        assert risk["agent_scores"][0]["extraction_confidence"] == 0.95


class TestExtractProbability:
    """Test the structured probability extraction."""

    def setup_method(self):
        from swim.agents.orchestrator.risk_fusion import extract_probability
        self.extract = extract_probability

    def test_decimal(self):
        result = self.extract("bloom probability: 0.72")
        assert result is not None
        assert abs(result["raw_value"] - 0.72) < 0.01

    def test_percentage(self):
        result = self.extract("risk score: 45%")
        assert result is not None
        assert abs(result["raw_value"] - 0.45) < 0.01

    def test_json_object(self):
        result = self.extract('Result: {"bloom_probability": 0.65}')
        assert result is not None
        assert abs(result["raw_value"] - 0.65) < 0.01
        assert result["source"] == "json"

    def test_bare_decimal(self):
        result = self.extract("The predicted value is 0.33 for this lake")
        assert result is not None
        assert abs(result["raw_value"] - 0.33) < 0.01
        assert result["source"] == "bare_decimal"

    def test_no_match(self):
        assert self.extract("No numbers here") is None

    def test_confidence_ranking(self):
        """JSON extraction should have higher confidence than bare decimal."""
        json_result = self.extract('{"probability": 0.5}')
        bare_result = self.extract("the value is 0.5 approximately")
        assert json_result["confidence"] > bare_result["confidence"]

    def test_percentage_near_keyword(self):
        result = self.extract("There is a 65% bloom risk at this location")
        assert result is not None
        assert abs(result["raw_value"] - 0.65) < 0.01
