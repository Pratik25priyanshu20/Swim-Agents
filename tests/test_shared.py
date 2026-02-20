# tests/test_shared.py

"""Tests for shared infrastructure: paths, a2a_config, health checks."""

from pathlib import Path

import pytest

from swim.shared.paths import PROJECT_ROOT, DATA_DIR, HARMONIZED_DIR, MODEL_DIR
from swim.shared.a2a_config import AGENT_PORTS, get_agent_url


class TestPaths:
    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_project_root_contains_swim(self):
        assert (PROJECT_ROOT / "swim").is_dir()

    def test_data_dir_relative_to_root(self):
        assert DATA_DIR == PROJECT_ROOT / "data"

    def test_harmonized_dir(self):
        assert HARMONIZED_DIR == DATA_DIR / "harmonized"


class TestA2AConfig:
    def test_all_agents_have_ports(self):
        expected = {"orchestrator", "homogen", "calibro", "visios", "predikt"}
        assert set(AGENT_PORTS.keys()) == expected

    def test_ports_are_unique(self):
        ports = list(AGENT_PORTS.values())
        assert len(ports) == len(set(ports))

    def test_port_range(self):
        for port in AGENT_PORTS.values():
            assert 10000 <= port <= 10010

    def test_get_agent_url(self):
        url = get_agent_url("homogen")
        assert url == "http://localhost:10001"

    def test_get_agent_url_orchestrator(self):
        url = get_agent_url("orchestrator")
        assert url == "http://localhost:10000"


_a2a_available = True
try:
    import a2a
except ImportError:
    _a2a_available = False


@pytest.mark.skipif(not _a2a_available, reason="a2a-sdk not installed")
class TestAgentCards:
    def test_homogen_card(self):
        from swim.agents.homogen.agent_card import agent_card
        assert agent_card.name == "HOMOGEN"
        assert len(agent_card.skills) >= 3

    def test_calibro_card(self):
        from swim.agents.calibro.agent_card import agent_card
        assert agent_card.name == "CALIBRO"
        assert len(agent_card.skills) >= 4

    def test_visios_card(self):
        from swim.agents.visios.agent_card import agent_card
        assert agent_card.name == "VISIOS"

    def test_predikt_card(self):
        from swim.agents.predikt.agent_card import agent_card
        assert agent_card.name == "PREDIKT"

    def test_orchestrator_card(self):
        from swim.agents.orchestrator.agent_card import agent_card
        assert "Orchestrator" in agent_card.name
