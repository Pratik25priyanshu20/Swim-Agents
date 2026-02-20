# swim/data_processing/drift_detector.py

"""Data drift detection using Kolmogorov-Smirnov test and Population Stability Index (PSI).

Compares incoming feature distributions against a saved reference (training)
distribution to flag when model inputs have shifted significantly.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from swim.shared.paths import MODEL_DIR

logger = logging.getLogger(__name__)

REFERENCE_PATH = MODEL_DIR / "drift_reference.json"

# Thresholds
KS_ALPHA = 0.05            # p-value below this ⇒ drift detected
PSI_WARNING = 0.1           # PSI above this ⇒ moderate drift
PSI_CRITICAL = 0.2          # PSI above this ⇒ severe drift


# ---------------------------------------------------------------------------
# PSI calculation
# ---------------------------------------------------------------------------

def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index between two 1-D distributions.

    PSI = Σ (p_i - q_i) * ln(p_i / q_i)
    where p_i = proportion of current in bin i, q_i = proportion of reference in bin i.
    """
    # Build bins from the reference distribution
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    if min_val == max_val:
        return 0.0
    bins = np.linspace(min_val, max_val, n_bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)

    # Proportions (add small epsilon to avoid division by zero)
    eps = 1e-8
    ref_prop = (ref_counts + eps) / (ref_counts.sum() + eps * n_bins)
    cur_prop = (cur_counts + eps) / (cur_counts.sum() + eps * n_bins)

    psi = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))
    return psi


# ---------------------------------------------------------------------------
# Reference distribution management
# ---------------------------------------------------------------------------

class DriftReference:
    """Stores per-feature summary statistics from the training distribution."""

    def __init__(self):
        self.features: Dict[str, Dict[str, Any]] = {}
        self.created_at: Optional[str] = None

    def fit(self, feature_names: List[str], data: np.ndarray):
        """Compute and store reference statistics from training data.

        Args:
            feature_names: list of feature names (length = data.shape[1])
            data: 2-D array (n_samples, n_features)
        """
        if data.ndim != 2 or data.shape[1] != len(feature_names):
            raise ValueError(
                f"Shape mismatch: data has {data.shape[1]} cols but {len(feature_names)} feature names"
            )

        self.features = {}
        for i, name in enumerate(feature_names):
            col = data[:, i].astype(float)
            col = col[np.isfinite(col)]
            if len(col) == 0:
                continue
            self.features[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "p25": float(np.percentile(col, 25)),
                "p50": float(np.percentile(col, 50)),
                "p75": float(np.percentile(col, 75)),
                "n_samples": int(len(col)),
                # Store histogram for PSI
                "histogram": np.histogram(col, bins=10)[0].tolist(),
                "bin_edges": np.histogram(col, bins=10)[1].tolist(),
            }
        self.created_at = datetime.now().isoformat()
        logger.info("Drift reference fitted on %d samples, %d features", data.shape[0], len(self.features))

    def save(self, path: Path = REFERENCE_PATH):
        """Save reference to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"features": self.features, "created_at": self.created_at}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Drift reference saved to %s", path)

    def load(self, path: Path = REFERENCE_PATH) -> bool:
        """Load reference from JSON. Returns True if loaded successfully."""
        if not path.exists():
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.features = payload.get("features", {})
            self.created_at = payload.get("created_at")
            logger.info("Drift reference loaded (%d features, from %s)", len(self.features), self.created_at)
            return True
        except Exception as exc:
            logger.warning("Failed to load drift reference: %s", exc)
            return False

    @property
    def is_fitted(self) -> bool:
        return bool(self.features)


# ---------------------------------------------------------------------------
# Drift Detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """Detects data drift by comparing incoming data against a saved reference.

    Usage:
        detector = DriftDetector()
        detector.load_reference()  # loads from disk
        report = detector.check(feature_names, new_data_array)
    """

    def __init__(self):
        self._reference = DriftReference()

    def load_reference(self, path: Path = REFERENCE_PATH) -> bool:
        return self._reference.load(path)

    def fit_reference(self, feature_names: List[str], data: np.ndarray, save: bool = True):
        """Fit a new reference from training data."""
        self._reference.fit(feature_names, data)
        if save:
            self._reference.save()

    @property
    def has_reference(self) -> bool:
        return self._reference.is_fitted

    def check(
        self,
        feature_names: List[str],
        data: np.ndarray,
    ) -> Dict[str, Any]:
        """Run drift detection on new data.

        Returns a report dict with:
          - overall_drift: bool (True if any feature has significant drift)
          - severity: "none" | "warning" | "critical"
          - feature_reports: per-feature KS and PSI results
          - summary: human-readable summary
        """
        if not self._reference.is_fitted:
            return {
                "overall_drift": False,
                "severity": "none",
                "feature_reports": {},
                "summary": "No reference distribution available — drift check skipped.",
                "timestamp": datetime.now().isoformat(),
            }

        if data.ndim == 1:
            data = data.reshape(1, -1)

        feature_reports: Dict[str, Dict[str, Any]] = {}
        drifted_features: List[str] = []
        max_psi = 0.0

        for i, name in enumerate(feature_names):
            if name not in self._reference.features:
                continue
            if i >= data.shape[1]:
                continue

            ref_stats = self._reference.features[name]
            current_col = data[:, i].astype(float)
            current_col = current_col[np.isfinite(current_col)]

            if len(current_col) == 0:
                continue

            report: Dict[str, Any] = {"feature": name}

            # --- KS test ---
            # Reconstruct reference samples from histogram for KS test
            ref_samples = self._reconstruct_samples(ref_stats)
            if ref_samples is not None and len(current_col) >= 2:
                ks_stat, ks_pvalue = stats.ks_2samp(ref_samples, current_col)
                report["ks_statistic"] = round(float(ks_stat), 4)
                report["ks_pvalue"] = round(float(ks_pvalue), 6)
                report["ks_drift"] = ks_pvalue < KS_ALPHA
            else:
                report["ks_statistic"] = None
                report["ks_pvalue"] = None
                report["ks_drift"] = False

            # --- PSI ---
            if ref_samples is not None and len(current_col) >= 5:
                psi = _compute_psi(ref_samples, current_col)
                report["psi"] = round(psi, 4)
                report["psi_severity"] = (
                    "critical" if psi >= PSI_CRITICAL
                    else "warning" if psi >= PSI_WARNING
                    else "none"
                )
                max_psi = max(max_psi, psi)
            else:
                report["psi"] = None
                report["psi_severity"] = "none"

            # --- Mean shift ---
            report["ref_mean"] = round(ref_stats["mean"], 4)
            report["current_mean"] = round(float(np.mean(current_col)), 4)
            ref_std = ref_stats["std"] or 1.0
            report["mean_shift_sigmas"] = round(
                abs(report["current_mean"] - report["ref_mean"]) / ref_std, 2
            )

            if report.get("ks_drift") or report.get("psi_severity") in ("warning", "critical"):
                drifted_features.append(name)

            feature_reports[name] = report

        # Overall verdict
        if max_psi >= PSI_CRITICAL or len(drifted_features) >= 3:
            severity = "critical"
        elif max_psi >= PSI_WARNING or len(drifted_features) >= 1:
            severity = "warning"
        else:
            severity = "none"

        overall_drift = severity != "none"

        if overall_drift:
            summary = (
                f"Drift detected in {len(drifted_features)} feature(s): "
                f"{', '.join(drifted_features)}. "
                f"Max PSI={max_psi:.3f}, severity={severity}. "
                f"Consider retraining the model."
            )
            logger.warning("Data drift detected: %s", summary)
        else:
            summary = f"No significant drift detected across {len(feature_reports)} features."

        return {
            "overall_drift": overall_drift,
            "severity": severity,
            "drifted_features": drifted_features,
            "feature_reports": feature_reports,
            "summary": summary,
            "max_psi": round(max_psi, 4),
            "timestamp": datetime.now().isoformat(),
        }

    @staticmethod
    def _reconstruct_samples(ref_stats: Dict[str, Any], n: int = 500) -> Optional[np.ndarray]:
        """Reconstruct approximate samples from stored histogram.

        Generates samples uniformly within each bin, proportional to bin counts.
        """
        hist = ref_stats.get("histogram")
        edges = ref_stats.get("bin_edges")
        if hist is None or edges is None:
            return None

        hist = np.array(hist, dtype=float)
        edges = np.array(edges, dtype=float)
        total = hist.sum()
        if total == 0:
            return None

        samples = []
        for count, lo, hi in zip(hist, edges[:-1], edges[1:]):
            bin_n = max(1, int(round(count / total * n)))
            samples.append(np.random.uniform(lo, hi, bin_n))
        return np.concatenate(samples)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

drift_detector = DriftDetector()
