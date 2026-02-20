# swim/shared/a2a_config.py

"""A2A protocol configuration for the SWIM platform."""

AGENT_PORTS = {
    "orchestrator": 10000,
    "homogen": 10001,
    "calibro": 10002,
    "visios": 10003,
    "predikt": 10004,
}

HOST = "localhost"


def get_agent_url(agent_name: str) -> str:
    """Return the base URL for an A2A agent server."""
    port = AGENT_PORTS[agent_name]
    return f"http://{HOST}:{port}"
