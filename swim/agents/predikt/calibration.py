# swim/agents/predikt/calibration.py

"""Probability calibration for PREDIKT predictions using isotonic regression."""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from swim.shared.paths import MODEL_DIR

logger = logging.getLogger(__name__)

CALIBRATOR_PATH = MODEL_DIR / "predikt_calibrator.pkl"


class ConfidenceCalibrator:
    """
    Calibrates raw model probabilities to well-calibrated confidence scores.

    Uses isotonic regression fitted on historical predictions vs. outcomes.
    Falls back to a simple Platt-style sigmoid if insufficient data.
    """

    def __init__(self):
        self._calibrator = None
        self._loaded = False

    def load(self, path: Path = CALIBRATOR_PATH):
        """Load a previously fitted calibrator."""
        if path.exists():
            try:
                with open(path, "rb") as f:
                    self._calibrator = pickle.load(f)
                self._loaded = True
                logger.info("Loaded confidence calibrator from %s", path)
            except Exception as exc:
                logger.warning("Failed to load calibrator: %s", exc)

    def fit(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        method: str = "isotonic",
    ):
        """Fit the calibrator on historical predictions vs actual outcomes."""
        if len(predicted_probs) < 20:
            logger.warning("Insufficient data for calibration (%d samples)", len(predicted_probs))
            return

        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self._calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._calibrator.fit(predicted_probs, actual_outcomes)
        elif method == "platt":
            from sklearn.linear_model import LogisticRegression
            self._calibrator = LogisticRegression()
            self._calibrator.fit(predicted_probs.reshape(-1, 1), actual_outcomes)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        self._loaded = True
        logger.info("Calibrator fitted on %d samples (method=%s)", len(predicted_probs), method)

    def calibrate(self, raw_probability: float) -> float:
        """Calibrate a single raw probability to a well-calibrated confidence score."""
        if not self._loaded or self._calibrator is None:
            return self._heuristic_calibration(raw_probability)

        try:
            arr = np.array([raw_probability])
            if hasattr(self._calibrator, "predict"):
                return float(np.clip(self._calibrator.predict(arr), 0, 1)[0])
            elif hasattr(self._calibrator, "predict_proba"):
                return float(self._calibrator.predict_proba(arr.reshape(-1, 1))[:, 1][0])
        except Exception:
            return self._heuristic_calibration(raw_probability)

    def save(self, path: Path = CALIBRATOR_PATH):
        """Save the fitted calibrator."""
        if self._calibrator is None:
            logger.warning("No calibrator to save")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._calibrator, f)
        logger.info("Calibrator saved to %s", path)

    @staticmethod
    def _heuristic_calibration(prob: float) -> float:
        """Heuristic calibration: pushes extreme values toward center slightly."""
        # Shrink towards 0.5 by 10% (mild regularization)
        return 0.9 * prob + 0.1 * 0.5

    def compute_reliability_diagram(
        self, predicted: np.ndarray, actual: np.ndarray, n_bins: int = 10
    ) -> Dict[str, List[float]]:
        """Compute reliability diagram data (fraction of positives vs mean predicted)."""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        fraction_positive = []
        bin_counts = []

        for i in range(n_bins):
            mask = (predicted >= bins[i]) & (predicted < bins[i + 1])
            if mask.sum() > 0:
                bin_centers.append(float((bins[i] + bins[i + 1]) / 2))
                fraction_positive.append(float(actual[mask].mean()))
                bin_counts.append(int(mask.sum()))

        return {
            "bin_centers": bin_centers,
            "fraction_positive": fraction_positive,
            "bin_counts": bin_counts,
        }


# Global singleton
confidence_calibrator = ConfidenceCalibrator()
