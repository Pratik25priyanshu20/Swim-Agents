# swim/agents/orchestrator/risk_fusion.py

"""Calibrated probabilistic risk fusion for the SWIM pipeline.

Replaces the heuristic regex-based probability extraction with:
  1. Structured multi-strategy probability parsing
  2. Per-agent confidence calibration via isotonic regression
  3. Bayesian-weighted fusion with proper uncertainty quantification
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from swim.agents.predikt.calibration import confidence_calibrator
from swim.shared.config import get_risk_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured probability extraction
# ---------------------------------------------------------------------------

def extract_probability(text: str) -> Optional[Dict[str, Any]]:
    """Extract probability from agent response text using multiple strategies.

    Returns a dict with:
      - raw_value: the extracted float in [0,1]
      - source: which extraction strategy matched
      - confidence: how confident we are in the extraction (0-1)

    Returns None if no probability could be extracted.
    """
    if not text:
        return None

    # Strategy 1: Try JSON parsing — agents may return structured JSON
    json_result = _try_json_extraction(text)
    if json_result is not None:
        return json_result

    # Strategy 2: Labelled numeric patterns (highest confidence regex)
    labelled_result = _try_labelled_patterns(text)
    if labelled_result is not None:
        return labelled_result

    # Strategy 3: Percentage near keywords
    pct_result = _try_percentage_near_keyword(text)
    if pct_result is not None:
        return pct_result

    # Strategy 4: Bare decimal in [0,1] (lowest confidence)
    bare_result = _try_bare_decimal(text)
    if bare_result is not None:
        return bare_result

    return None


def _try_json_extraction(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse a JSON block from the text containing probability fields."""
    # Find JSON objects in the text
    for match in re.finditer(r'\{[^{}]+\}', text):
        try:
            obj = json.loads(match.group())
            for key in ("bloom_probability", "probability", "risk_score", "score"):
                if key in obj:
                    val = float(obj[key])
                    if val > 1.0:
                        val /= 100.0
                    if 0.0 <= val <= 1.0:
                        return {
                            "raw_value": val,
                            "source": "json",
                            "confidence": 0.95,
                            "context": key,
                        }
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return None


def _try_labelled_patterns(text: str) -> Optional[Dict[str, Any]]:
    """Match 'probability: 0.65' or 'risk score: 72%' style patterns."""
    patterns = [
        (r"(?:bloom[_ ])?probability[:\s]*(\d+\.?\d*)%", "probability_pct"),
        (r"(?:bloom[_ ])?probability[:\s]*(0\.\d+)", "probability_decimal"),
        (r"risk[_ ]?score[:\s]*(\d+\.?\d*)%", "risk_score_pct"),
        (r"risk[_ ]?score[:\s]*(0\.\d+)", "risk_score_decimal"),
        (r"confidence[:\s]*(\d+\.?\d*)%", "confidence_pct"),
    ]
    for pattern, source in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            if val > 1.0:
                val /= 100.0
            if 0.0 <= val <= 1.0:
                return {
                    "raw_value": val,
                    "source": f"labelled_{source}",
                    "confidence": 0.85,
                    "context": match.group(0)[:60],
                }
    return None


def _try_percentage_near_keyword(text: str) -> Optional[Dict[str, Any]]:
    """Match 'X% bloom/risk/probability' or 'bloom/risk of X%'."""
    patterns = [
        r"(\d+\.?\d*)%\s*(?:bloom|risk|probability|chance)",
        r"(?:bloom|risk|probability|chance)\s*(?:of|is|at)?\s*(\d+\.?\d*)%",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            if val > 1.0:
                val /= 100.0
            if 0.0 <= val <= 1.0:
                return {
                    "raw_value": val,
                    "source": "pct_near_keyword",
                    "confidence": 0.7,
                    "context": match.group(0)[:60],
                }
    return None


def _try_bare_decimal(text: str) -> Optional[Dict[str, Any]]:
    """Match a standalone 0.XX decimal as last resort."""
    match = re.search(r"\b(0\.\d{1,3})\b", text)
    if match:
        val = float(match.group(1))
        if 0.0 <= val <= 1.0:
            return {
                "raw_value": val,
                "source": "bare_decimal",
                "confidence": 0.4,
                "context": match.group(0),
            }
    return None


# ---------------------------------------------------------------------------
# Calibrated risk fusion
# ---------------------------------------------------------------------------

def compute_calibrated_risk_fusion(
    results: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute calibrated risk fusion from agent results.

    Differences from the old heuristic approach:
      - Uses structured probability extraction with confidence scores
      - Calibrates raw probabilities via isotonic regression
      - Weights by both agent importance AND extraction confidence
      - Produces proper confidence intervals with uncertainty quantification
    """
    weights = weights or get_risk_weights()

    # Load calibrator (no-op if already loaded or no calibrator exists)
    confidence_calibrator.load()

    agent_scores: List[Dict[str, Any]] = []
    evidence: List[str] = []

    for agent_name, weight in weights.items():
        agent_result = results.get(agent_name, {})
        if agent_result.get("status") != "success":
            continue

        summary = agent_result.get("summary", "")
        extraction = extract_probability(summary)

        if extraction is None:
            continue

        raw_prob = extraction["raw_value"]
        extraction_conf = extraction["confidence"]

        # Calibrate the raw probability
        calibrated_prob = confidence_calibrator.calibrate(raw_prob)

        # Effective weight = agent weight * extraction confidence
        effective_weight = weight * extraction_conf

        agent_scores.append({
            "agent": agent_name,
            "raw_probability": raw_prob,
            "calibrated_probability": calibrated_prob,
            "extraction_source": extraction["source"],
            "extraction_confidence": extraction_conf,
            "agent_weight": weight,
            "effective_weight": effective_weight,
        })

        evidence.append(
            f"{agent_name.upper()}: {calibrated_prob:.1%} "
            f"(raw={raw_prob:.1%}, via {extraction['source']})"
        )

    if not agent_scores:
        return {
            "level": "unknown",
            "score": 0.0,
            "calibrated_score": 0.0,
            "confidence": "0%",
            "confidence_interval": {"lower": 0.0, "upper": 0.0},
            "evidence": ["No valid data sources"],
            "recommendation": "Increase monitoring",
            "fusion_method": "none",
        }

    # Weighted average using effective weights
    total_weight = sum(s["effective_weight"] for s in agent_scores)
    fused_score = sum(
        s["calibrated_probability"] * s["effective_weight"]
        for s in agent_scores
    ) / total_weight

    # Raw (uncalibrated) score for comparison
    raw_fused = sum(
        s["raw_probability"] * s["effective_weight"]
        for s in agent_scores
    ) / total_weight

    # --- Uncertainty quantification ---
    # 1. Epistemic uncertainty: disagreement between agents
    calibrated_probs = np.array([s["calibrated_probability"] for s in agent_scores])
    agent_disagreement = float(np.std(calibrated_probs)) if len(calibrated_probs) > 1 else 0.15

    # 2. Extraction uncertainty: how confident we are in the extracted values
    avg_extraction_conf = np.mean([s["extraction_confidence"] for s in agent_scores])
    extraction_uncertainty = 1.0 - avg_extraction_conf

    # 3. Source coverage: fewer sources = more uncertainty
    n_sources = len(agent_scores)
    coverage_factor = min(1.0, 0.4 + n_sources * 0.15)  # maxes at 1.0 with 4 sources

    # Combined uncertainty (conservative estimate)
    total_uncertainty = min(0.35, agent_disagreement * 0.5 + extraction_uncertainty * 0.3 + (1 - coverage_factor) * 0.2)

    # Confidence interval
    ci_lower = max(0.0, fused_score - total_uncertainty)
    ci_upper = min(1.0, fused_score + total_uncertainty)

    # Overall confidence percentage
    confidence_pct = coverage_factor * avg_extraction_conf

    # Risk level classification
    if fused_score > 0.75:
        level = "critical"
    elif fused_score > 0.5:
        level = "high"
    elif fused_score > 0.3:
        level = "moderate"
    else:
        level = "low"

    recommendation = {
        "low": "No action needed",
        "moderate": "Continue standard monitoring",
        "high": "Increase monitoring frequency",
        "critical": "Issue immediate public advisory",
    }[level]

    return {
        "level": level,
        "score": round(fused_score, 4),
        "raw_score": round(raw_fused, 4),
        "calibrated_score": round(fused_score, 4),
        "confidence": f"{confidence_pct:.0%}",
        "confidence_interval": {
            "lower": round(ci_lower, 4),
            "upper": round(ci_upper, 4),
        },
        "uncertainty": {
            "total": round(total_uncertainty, 4),
            "agent_disagreement": round(agent_disagreement, 4),
            "extraction_uncertainty": round(extraction_uncertainty, 4),
            "coverage_factor": round(coverage_factor, 4),
        },
        "evidence": evidence,
        "recommendation": recommendation,
        "sources_used": n_sources,
        "agent_scores": agent_scores,
        "fusion_method": "calibrated_weighted_average",
        "timestamp": datetime.now().isoformat(),
    }
