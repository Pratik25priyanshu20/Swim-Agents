# swim/agents/predikt/retraining.py

"""Model retraining pipeline for PREDIKT — ingests new data, retrains ensemble, evaluates, and hot-swaps."""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from swim.shared.paths import MODEL_DIR, PROCESSED_DIR, HARMONIZED_DIR
from swim.data_processing.drift_detector import DriftReference

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "chlorophyll_a", "water_temperature", "turbidity", "dissolved_oxygen",
    "ph", "total_nitrogen", "total_phosphorus", "solar_radiation",
    "wind_speed", "precipitation",
]


def load_training_data(path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load and prepare training data from the harmonized/processed data."""
    data_path = path or PROCESSED_DIR / "ml_ready_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = pd.read_csv(data_path)

    # Ensure required features exist
    available = [f for f in FEATURE_NAMES if f in df.columns]
    if len(available) < 5:
        raise ValueError(f"Insufficient features: found {available}")

    X = df[available].apply(pd.to_numeric, errors="coerce").fillna(0).values

    # Generate binary labels: bloom if chlorophyll_a > 20 and water_temp > 18
    if "chlorophyll_a" in df.columns and "water_temperature" in df.columns:
        chl = pd.to_numeric(df["chlorophyll_a"], errors="coerce").fillna(0)
        temp = pd.to_numeric(df["water_temperature"], errors="coerce").fillna(0)
        y = ((chl > 20) & (temp > 18)).astype(int).values
    else:
        # Fallback: random labels for structure testing
        y = (np.random.rand(len(X)) > 0.7).astype(int)

    return X, y


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    n_cv_folds: int = 5,
) -> Dict[str, Any]:
    """Train a Random Forest + Gradient Boosting ensemble and evaluate."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)

    # Cross-validation
    rf_scores = cross_val_score(rf, X_scaled, y, cv=n_cv_folds, scoring="accuracy")
    gb_scores = cross_val_score(gb, X_scaled, y, cv=n_cv_folds, scoring="accuracy")

    # Final fit on full data
    rf.fit(X_scaled, y)
    gb.fit(X_scaled, y)

    # Ensemble predictions
    rf_proba = rf.predict_proba(X_scaled)[:, 1]
    gb_proba = gb.predict_proba(X_scaled)[:, 1]
    ensemble_proba = (rf_proba + gb_proba) / 2
    ensemble_pred = (ensemble_proba > 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, ensemble_pred)),
        "f1_score": float(f1_score(y, ensemble_pred, zero_division=0)),
        "rf_cv_accuracy": float(rf_scores.mean()),
        "gb_cv_accuracy": float(gb_scores.mean()),
        "training_samples": len(X),
        "positive_rate": float(y.mean()),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y, ensemble_proba))
    except ValueError:
        metrics["roc_auc"] = 0.0

    bundle = {
        "rf_model": rf,
        "gb_model": gb,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES[:X.shape[1]],
        "trained_at": datetime.now().isoformat(),
    }

    return {"bundle": bundle, "metrics": metrics}


def save_model(bundle: Dict, metrics: Dict, model_dir: Path = MODEL_DIR):
    """Save the trained model bundle and metadata."""
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "predikt_ensemble.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    meta_path = model_dir / "model_metadata.json"
    metadata = {
        "model_type": "ensemble",
        "feature_names": bundle["feature_names"],
        "trained_at": bundle["trained_at"],
        "metrics": metrics,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model saved to %s (accuracy: %.3f)", model_path, metrics["accuracy"])


def retrain(data_path: Optional[Path] = None, min_accuracy: float = 0.6) -> Dict[str, Any]:
    """Full retraining pipeline: load data → train → evaluate → save if improved."""
    logger.info("Starting retraining pipeline...")

    X, y = load_training_data(data_path)
    logger.info("Loaded %d samples (%d features)", X.shape[0], X.shape[1])

    result = train_ensemble(X, y)
    metrics = result["metrics"]
    logger.info("Training complete: accuracy=%.3f, f1=%.3f", metrics["accuracy"], metrics["f1_score"])

    if metrics["accuracy"] < min_accuracy:
        logger.warning(
            "New model accuracy (%.3f) below threshold (%.3f) — not saving",
            metrics["accuracy"],
            min_accuracy,
        )
        return {"status": "rejected", "metrics": metrics, "reason": "below_threshold"}

    # Check if new model is better than existing
    existing_meta_path = MODEL_DIR / "model_metadata.json"
    if existing_meta_path.exists():
        with open(existing_meta_path, "r") as f:
            existing = json.load(f)
        existing_acc = existing.get("metrics", {}).get("accuracy", 0)
        if metrics["accuracy"] <= existing_acc:
            logger.info(
                "New model (%.3f) not better than existing (%.3f) — keeping old model",
                metrics["accuracy"],
                existing_acc,
            )
            return {"status": "no_improvement", "metrics": metrics, "existing_accuracy": existing_acc}

    save_model(result["bundle"], metrics)

    # Save drift reference from training data for future drift detection
    try:
        ref = DriftReference()
        ref.fit(FEATURE_NAMES[:X.shape[1]], X)
        ref.save()
        logger.info("Drift reference updated from retraining data")
    except Exception as exc:
        logger.warning("Failed to save drift reference: %s", exc)

    return {"status": "saved", "metrics": metrics}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(retrain())
