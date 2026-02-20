# swim/shared/config.py

"""Centralized configuration loaded from config.yaml + environment variables."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from swim.shared.paths import PROJECT_ROOT
from swim.shared.utils.config_loader import load_config

CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> Dict[str, Any]:
    """Load and return the global SWIM configuration."""
    if CONFIG_PATH.exists():
        cfg = load_config(str(CONFIG_PATH))
    else:
        cfg = {}

    # Override with environment variables where applicable
    if os.getenv("SWIM_LLM_MODEL"):
        cfg.setdefault("llm", {})["model"] = os.getenv("SWIM_LLM_MODEL")
    if os.getenv("SWIM_LOG_LEVEL"):
        cfg.setdefault("logging", {})["level"] = os.getenv("SWIM_LOG_LEVEL")

    return cfg


def get_llm_config() -> Dict[str, Any]:
    return get_config().get("llm", {})


def get_agent_config(agent_name: str) -> Dict[str, Any]:
    return get_config().get("agents", {}).get(agent_name, {})


def get_risk_weights() -> Dict[str, float]:
    return get_config().get("risk_fusion", {}).get("weights", {
        "predikt": 0.4, "calibro": 0.3, "visios": 0.2, "homogen": 0.1,
    })


def get_alerting_config() -> Dict[str, Any]:
    return get_config().get("alerting", {})
