# swim/shared/health.py

"""Health check endpoint mixin for A2A agent servers."""

import os
from pathlib import Path
from typing import Any, Dict, List

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from swim.shared.paths import DATA_DIR, HARMONIZED_DIR, MODEL_DIR


def _check_env_vars(required: List[str]) -> Dict[str, bool]:
    return {var: bool(os.getenv(var)) for var in required}


def _check_paths(paths: Dict[str, Path]) -> Dict[str, bool]:
    return {name: path.exists() for name, path in paths.items()}


def build_health_check(
    agent_name: str,
    required_env: List[str] = None,
    required_paths: Dict[str, Path] = None,
):
    """Return a Starlette Route for /health."""
    required_env = required_env or ["GEMINI_API_KEY"]
    required_paths = required_paths or {}

    async def health(request: Request) -> JSONResponse:
        env_status = _check_env_vars(required_env)
        path_status = _check_paths(required_paths)

        all_ok = all(env_status.values()) and all(path_status.values())

        body = {
            "agent": agent_name,
            "status": "healthy" if all_ok else "degraded",
            "checks": {
                "env_vars": env_status,
                "paths": path_status,
            },
        }
        status_code = 200 if all_ok else 503
        return JSONResponse(body, status_code=status_code)

    return Route("/health", health)


# Pre-built health routes for each agent

HOMOGEN_HEALTH = build_health_check(
    "HOMOGEN",
    required_env=["GEMINI_API_KEY"],
    required_paths={"data_dir": DATA_DIR, "harmonized_dir": HARMONIZED_DIR},
)

CALIBRO_HEALTH = build_health_check(
    "CALIBRO",
    required_env=["GEMINI_API_KEY"],
    required_paths={"data_dir": DATA_DIR},
)

VISIOS_HEALTH = build_health_check(
    "VISIOS",
    required_env=["GEMINI_API_KEY"],
    required_paths={"visios_images": DATA_DIR / "visios_images"},
)

PREDIKT_HEALTH = build_health_check(
    "PREDIKT",
    required_env=["GEMINI_API_KEY"],
    required_paths={"model_dir": MODEL_DIR, "processed_dir": DATA_DIR / "processed"},
)

ORCHESTRATOR_HEALTH = build_health_check(
    "Orchestrator",
    required_env=["GEMINI_API_KEY"],
    required_paths={},
)
