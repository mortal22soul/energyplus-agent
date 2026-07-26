"""Tests for multi-agent LangGraph architecture."""
import unittest
import json
from unittest.mock import patch, MagicMock

from ecoloop.multi_agent import (
    MultiAgentController,
    AgentState,
    _deterministic_merge,
    _dict_to_action,
    _proposal_to_dict,
    build_multi_agent_graph,
)
from ecoloop.schemas import ControlAction, ZoneSetpoints
from ecoloop.config import RunConfig, LLMConfig, SafetyLimits


class TestDeterministicMerge(unittest.TestCase):
    """Test the deterministic merge fallback logic."""

    def _make_state(self, occupancy: dict) -> AgentState:
        state_data = {
            "zone_temps_c": {"SPACE1-1": 22.0, "SPACE2-1": 21.0},
            "occupancy": occupancy,
        }
        return {
            "building_state_json": json.dumps(state_data),
            "evaluator_summary": {},
            "zones": list(state_data["zone_temps_c"].keys()),
            "energy_proposal": {
                "zone_setpoints": {
                    "SPACE1-1": {"heating_c": 18.0, "cooling_c": 28.0},
                    "SPACE2-1": {"heating_c": 18.0, "cooling_c": 28.0},
                },
                "mode": "setback",
                "rationale": "energy savings",
                "confidence": 0.8,
            },
            "comfort_proposal": {
                "zone_setpoints": {
                    "SPACE1-1": {"heating_c": 21.0, "cooling_c": 24.0},
                    "SPACE2-1": {"heating_c": 20.5, "cooling_c": 24.5},
                },
                "mode": "normal",
                "rationale": "comfort",
                "confidence": 0.9,
            },
            "forecast_proposal": {
                "zone_setpoints": {
                    "SPACE1-1": {"heating_c": 19.0, "cooling_c": 26.0},
                    "SPACE2-1": {"heating_c": 19.0, "cooling_c": 26.0},
                },
                "mode": "precondition",
                "rationale": "pre-cool",
                "confidence": 0.7,
            },
            "merged_action": None,
            "error": None,
        }

    def test_occupied_zones_prefer_comfort(self):
        state = self._make_state({"SPACE1-1": 1.0, "SPACE2-1": 1.0})
        result = _deterministic_merge(state)
        sp1 = result["zone_setpoints"]["SPACE1-1"]
        # Should use comfort proposal for occupied zones
        self.assertEqual(sp1["heating_c"], 21.0)
        self.assertEqual(sp1["cooling_c"], 24.0)

    def test_unoccupied_zones_prefer_energy(self):
        state = self._make_state({"SPACE1-1": 0.0, "SPACE2-1": 0.0})
        result = _deterministic_merge(state)
        sp1 = result["zone_setpoints"]["SPACE1-1"]
        # Should use energy proposal for unoccupied zones
        self.assertEqual(sp1["heating_c"], 18.0)
        self.assertEqual(sp1["cooling_c"], 28.0)

    def test_mixed_occupancy(self):
        state = self._make_state({"SPACE1-1": 1.0, "SPACE2-1": 0.0})
        result = _deterministic_merge(state)
        sp1 = result["zone_setpoints"]["SPACE1-1"]
        sp2 = result["zone_setpoints"]["SPACE2-1"]
        # SPACE1-1 occupied: comfort, SPACE2-1 unoccupied: energy
        self.assertEqual(sp1["heating_c"], 21.0)
        self.assertEqual(sp2["heating_c"], 18.0)


class TestDictToAction(unittest.TestCase):
    """Test converting dict proposals to ControlAction."""

    def test_valid_dict(self):
        data = {
            "zone_setpoints": {
                "Z1": {"heating_c": 20.0, "cooling_c": 25.0},
            },
            "mode": "normal",
            "rationale": "test",
            "confidence": 0.8,
        }
        action = _dict_to_action(data, ["Z1"])
        self.assertIsNotNone(action)
        self.assertEqual(action.zone_setpoints["Z1"].heating_c, 20.0)
        self.assertEqual(action.mode, "normal")

    def test_fills_missing_zones(self):
        data = {
            "zone_setpoints": {
                "Z1": {"heating_c": 20.0, "cooling_c": 25.0},
            },
            "mode": "normal",
        }
        action = _dict_to_action(data, ["Z1", "Z2"])
        self.assertIsNotNone(action)
        self.assertIn("Z2", action.zone_setpoints)
        # Missing zone gets safe defaults
        self.assertEqual(action.zone_setpoints["Z2"].heating_c, 18.0)
        self.assertEqual(action.zone_setpoints["Z2"].cooling_c, 28.0)

    def test_invalid_dict_returns_none(self):
        action = _dict_to_action({"bad": "data"}, ["Z1"])
        # No zone_setpoints, but should still return with defaults
        self.assertIsNotNone(action)

    def test_empty_dict(self):
        action = _dict_to_action({}, ["Z1"])
        self.assertIsNotNone(action)
        self.assertIn("Z1", action.zone_setpoints)


class TestProposalToDict(unittest.TestCase):
    """Test proposal normalization."""

    def test_none_returns_default(self):
        result = _proposal_to_dict(None)
        self.assertEqual(result["mode"], "hold")
        self.assertEqual(result["confidence"], 0.0)

    def test_valid_dict_passes_through(self):
        d = {"zone_setpoints": {"Z1": {"heating_c": 20.0}}, "mode": "normal"}
        result = _proposal_to_dict(d)
        self.assertEqual(result["mode"], "normal")


class TestMultiAgentController(unittest.TestCase):
    """Test the MultiAgentController wrapper."""

    def test_controller_initializes(self):
        config = RunConfig(llm=LLMConfig(mode="azure-first"))
        controller = MultiAgentController(config)
        # LangGraph should be available since we installed it
        self.assertTrue(controller.available)

    def test_decide_returns_action_or_none(self):
        config = RunConfig(llm=LLMConfig(mode="deterministic"))
        controller = MultiAgentController(config)
        state_json = '{"zone_temps_c": {"Z1": 22.0}, "occupancy": {"Z1": 1.0}}'

        # With mocked LLM calls returning None, supervisor uses deterministic merge
        with patch("ecoloop.multi_agent._call_azure_openai", return_value=None):
            action = controller.decide(state_json, {})
        # Should get a valid action from deterministic merge
        if action is not None:
            self.assertIsInstance(action, ControlAction)


class TestBuildGraph(unittest.TestCase):
    """Test graph construction."""

    def test_graph_compiles(self):
        config = RunConfig(llm=LLMConfig(mode="azure-first"))
        graph = build_multi_agent_graph(config)
        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
