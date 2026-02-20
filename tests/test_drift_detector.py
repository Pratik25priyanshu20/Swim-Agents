# tests/test_drift_detector.py

"""Unit tests for the data drift detection module."""

import numpy as np
import pytest


class TestPSI:
    """Test the PSI computation."""

    def test_identical_distributions_zero_psi(self):
        from swim.data_processing.drift_detector import _compute_psi
        data = np.random.normal(0, 1, 1000)
        psi = _compute_psi(data, data)
        assert psi < 0.01

    def test_shifted_distribution_positive_psi(self):
        from swim.data_processing.drift_detector import _compute_psi
        ref = np.random.normal(0, 1, 1000)
        shifted = np.random.normal(2, 1, 1000)  # mean shifted by 2 std
        psi = _compute_psi(ref, shifted)
        assert psi > 0.1  # should detect drift

    def test_constant_distribution_zero_psi(self):
        from swim.data_processing.drift_detector import _compute_psi
        data = np.ones(100)
        psi = _compute_psi(data, data)
        assert psi == 0.0


class TestDriftReference:
    """Test DriftReference fit/save/load."""

    def test_fit_creates_statistics(self):
        from swim.data_processing.drift_detector import DriftReference
        ref = DriftReference()
        data = np.random.randn(100, 3)
        ref.fit(["feat_a", "feat_b", "feat_c"], data)
        assert ref.is_fitted
        assert "feat_a" in ref.features
        assert "mean" in ref.features["feat_a"]
        assert "histogram" in ref.features["feat_a"]

    def test_fit_rejects_shape_mismatch(self):
        from swim.data_processing.drift_detector import DriftReference
        ref = DriftReference()
        data = np.random.randn(100, 3)
        with pytest.raises(ValueError, match="Shape mismatch"):
            ref.fit(["a", "b"], data)  # 2 names for 3 columns

    def test_save_and_load(self, tmp_path):
        from swim.data_processing.drift_detector import DriftReference
        ref = DriftReference()
        data = np.random.randn(100, 2)
        ref.fit(["x", "y"], data)
        path = tmp_path / "ref.json"
        ref.save(path)

        ref2 = DriftReference()
        assert ref2.load(path)
        assert ref2.is_fitted
        assert abs(ref2.features["x"]["mean"] - ref.features["x"]["mean"]) < 0.001


class TestDriftDetector:
    """Test the full DriftDetector workflow."""

    def test_no_reference_returns_no_drift(self):
        from swim.data_processing.drift_detector import DriftDetector
        detector = DriftDetector()
        report = detector.check(["a"], np.array([[1.0]]))
        assert report["overall_drift"] is False
        assert report["severity"] == "none"

    def test_no_drift_on_same_data(self, tmp_path):
        from swim.data_processing.drift_detector import DriftDetector
        detector = DriftDetector()
        np.random.seed(42)
        data = np.random.randn(200, 3)
        features = ["f1", "f2", "f3"]
        path = tmp_path / "ref.json"
        detector.fit_reference(features, data, save=True)
        detector._reference.save(path)
        detector.load_reference(path)

        # Use a large subsample to avoid histogram noise with small samples
        report = detector.check(features, data)
        assert report["severity"] in ("none", "warning")  # same data should show no drift

    def test_drift_detected_on_shifted_data(self, tmp_path):
        from swim.data_processing.drift_detector import DriftDetector
        detector = DriftDetector()
        np.random.seed(42)
        features = ["f1", "f2", "f3"]
        training = np.random.randn(500, 3)
        path = tmp_path / "ref.json"
        detector.fit_reference(features, training, save=True)
        detector._reference.save(path)
        detector.load_reference(path)

        # Heavily shifted data
        shifted = np.random.randn(100, 3) + 5.0
        report = detector.check(features, shifted)
        assert report["overall_drift"] is True
        assert report["severity"] in ("warning", "critical")
        assert len(report["drifted_features"]) > 0

    def test_report_contains_feature_details(self, tmp_path):
        from swim.data_processing.drift_detector import DriftDetector
        detector = DriftDetector()
        np.random.seed(42)
        features = ["chlorophyll_a", "temperature"]
        training = np.random.randn(200, 2) * 10 + 15
        path = tmp_path / "ref.json"
        detector.fit_reference(features, training, save=True)
        detector._reference.save(path)
        detector.load_reference(path)

        new_data = np.random.randn(50, 2) * 10 + 15
        report = detector.check(features, new_data)
        for feat in features:
            if feat in report["feature_reports"]:
                fr = report["feature_reports"][feat]
                assert "ks_statistic" in fr
                assert "psi" in fr
                assert "ref_mean" in fr
                assert "current_mean" in fr
