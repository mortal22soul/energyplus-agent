"""Tests for agent schema validation and LLM provider routing."""
import unittest
from unittest.mock import patch, MagicMock
from ecoloop.agent import LLMClient, LLMResponse
from ecoloop.schemas import ControlAction, ZoneSetpoints
from ecoloop.config import RunConfig, SafetyLimits, LLMConfig


class TestAgentSchema(unittest.TestCase):
    def setUp(self):
        config = RunConfig(
            llm=LLMConfig(mode="deterministic"),
            safety=SafetyLimits(),
        )
        self.client = LLMClient(config)

    def test_valid_json_parsed(self):
        raw = '{"action_id": "test", "zone_setpoints": {"Z1": {"heating_c": 20.0, "cooling_c": 24.0}}, "mode": "normal", "rationale": "ok", "confidence": 0.8}'
        action = self.client._parse_action(raw)
        self.assertIsNotNone(action)
        self.assertEqual(action.zone_setpoints["Z1"].heating_c, 20.0)

    def test_json_in_markdown_block(self):
        raw = '```json\n{"action_id": "x", "zone_setpoints": {"Z1": {"heating_c": 19.0, "cooling_c": 25.0}}, "mode": "hold", "rationale": "", "confidence": 0.5}\n```'
        action = self.client._parse_action(raw)
        self.assertIsNotNone(action)

    def test_invalid_json_returns_none(self):
        action = self.client._parse_action("this is not json at all")
        self.assertIsNone(action)

    def test_empty_string_returns_none(self):
        action = self.client._parse_action("")
        self.assertIsNone(action)

    def test_deterministic_fallback_when_unavailable(self):
        state_json = '{"zone_temps_c": {"Z1": 22.0}, "occupancy": {"Z1": 1.0}, "hvac_power_kw": 2.0}'
        response = self.client.decide(state_json, {})
        self.assertEqual(response.source, "deterministic")
        self.assertIsNotNone(response.action)


class TestAzureProvider(unittest.TestCase):
    """Test Azure OpenAI provider routing and error handling."""

    def _make_client(self, mode="azure"):
        config = RunConfig(
            llm=LLMConfig(mode=mode),
            safety=SafetyLimits(),
        )
        return LLMClient(config)

    def test_azure_mode_routes_to_azure(self):
        client = self._make_client(mode="azure")
        state_json = '{"zone_temps_c": {"SPACE1-1": 22.0}, "occupancy": {"SPACE1-1": 1.0}}'
        evaluator_summary = {"comfort": {"status": "occupied"}}

        with patch.object(client, '_call_azure') as mock_azure:
            mock_azure.return_value = (
                ControlAction(
                    action_id="azure-test",
                    zone_setpoints={"SPACE1-1": ZoneSetpoints(20.0, 26.0)},
                    mode="normal",
                    rationale="test",
                    confidence=0.9,
                ),
                150.0,
            )
            response = client.decide(state_json, evaluator_summary)

        self.assertEqual(response.source, "primary")
        self.assertEqual(response.model_used, "gpt-5.4-mini")
        self.assertIsNotNone(response.action)

    def test_azure_falls_back_when_credential_missing(self):
        config = RunConfig(
            llm=LLMConfig(mode="azure", azure_api_key="", azure_endpoint=""),
            safety=SafetyLimits(),
        )
        client = LLMClient(config)
        state_json = '{"zone_temps_c": {"Z1": 22.0}, "occupancy": {"Z1": 1.0}}'
        response = client.decide(state_json, {})
        self.assertEqual(response.source, "deterministic")
        self.assertEqual(response.model_used, "none")

    def test_azure_falls_back_on_api_error(self):
        client = self._make_client(mode="azure")
        state_json = '{"zone_temps_c": {"Z1": 22.0}, "occupancy": {"Z1": 1.0}}'

        with patch.object(client, '_call_azure', side_effect=RuntimeError("API error")):
            response = client.decide(state_json, {})

        self.assertEqual(response.source, "deterministic")
        self.assertIsNotNone(response.action)
        self.assertEqual(response.action.mode, "hold")

    def test_azure_first_falls_back_to_ollama(self):
        """azure-first mode: Azure fails → Ollama used as fallback."""
        client = self._make_client(mode="azure-first")
        state_json = '{"zone_temps_c": {"SPACE1-1": 22.0}, "occupancy": {"SPACE1-1": 1.0}}'

        ollama_action = ControlAction(
            action_id="ollama-fallback",
            zone_setpoints={"SPACE1-1": ZoneSetpoints(19.0, 25.0)},
            mode="normal",
            rationale="ollama fallback",
            confidence=0.7,
        )

        with patch.object(client, '_call_azure', side_effect=RuntimeError("Azure down")):
            with patch.object(client, '_call_ollama', return_value=(ollama_action, 300.0)) as mock_ollama:
                response = client.decide(state_json, {})

        self.assertEqual(response.source, "fallback")
        self.assertEqual(response.model_used, "llama3.1:8b-instruct")
        self.assertIsNotNone(response.action)
        self.assertEqual(response.action.mode, "normal")

    def test_azure_first_falls_back_to_deterministic_when_all_fail(self):
        """azure-first mode: both Azure and Ollama fail → deterministic."""
        client = self._make_client(mode="azure-first")
        state_json = '{"zone_temps_c": {"Z1": 22.0}, "occupancy": {"Z1": 1.0}}'

        with patch.object(client, '_call_azure', side_effect=RuntimeError("Azure down")):
            with patch.object(client, '_call_ollama', side_effect=RuntimeError("Ollama down")):
                response = client.decide(state_json, {})

        self.assertEqual(response.source, "deterministic")
        self.assertEqual(response.model_used, "none")
        self.assertIsNotNone(response.action)

    def test_hybrid_mode_behaves_like_azure_first(self):
        """hybrid mode is an alias for azure-first."""
        client = self._make_client(mode="hybrid")

        ollama_action = ControlAction(
            action_id="hybrid-ollama",
            zone_setpoints={"Z1": ZoneSetpoints(20.0, 26.0)},
            mode="normal",
            rationale="hybrid fallback",
            confidence=0.7,
        )

        with patch.object(client, '_call_azure', side_effect=RuntimeError("Azure down")):
            with patch.object(client, '_call_ollama', return_value=(ollama_action, 250.0)):
                response = client.decide('{"zone_temps_c": {"Z1": 22.0}, "occupancy": {"Z1": 1.0}}', {})

        self.assertEqual(response.source, "fallback")

    def test_ollama_mode_routes_correctly(self):
        client = self._make_client(mode="ollama")
        state_json = '{"zone_temps_c": {"Z1": 22.0}, "occupancy": {"Z1": 1.0}}'

        with patch.object(client, '_call_ollama') as mock_ollama:
            mock_ollama.return_value = (
                ControlAction(
                    action_id="ollama-test",
                    zone_setpoints={"Z1": ZoneSetpoints(19.0, 25.0)},
                    mode="normal",
                    rationale="test",
                    confidence=0.8,
                ),
                200.0,
            )
            response = client.decide(state_json, {})

        self.assertEqual(response.source, "primary")
        self.assertEqual(response.model_used, "llama3.1:8b-instruct")

    def test_deterministic_mode_skips_llm(self):
        client = self._make_client(mode="deterministic")
        state_json = '{"zone_temps_c": {"Z1": 22.0}, "occupancy": {"Z1": 1.0}}'
        response = client.decide(state_json, {})
        self.assertEqual(response.source, "deterministic")
        self.assertIsNotNone(response.action)


if __name__ == "__main__":
    unittest.main()
