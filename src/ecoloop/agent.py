"""Local LLM client for Eco-Loop: Azure OpenAI (primary) → Ollama (fallback) → deterministic."""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .schemas import ControlAction, ZoneSetpoints
from .config import RunConfig, LLMConfig


@dataclass(frozen=True)
class LLMResponse:
    action: ControlAction | None
    model_used: str
    latency_ms: float
    source: str  # "primary", "fallback", "deterministic"


class LLMClient:
    """Azure-first LLM client: Azure OpenAI → Ollama → deterministic fallback chain."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.llm_config = config.llm

    def decide(self, state_json: str, evaluator_summary: dict) -> LLMResponse:
        """Request a ControlAction from the LLM. Chain: Azure → Ollama → deterministic."""
        llm_mode = self.llm_config.mode

        if llm_mode == "deterministic":
            return self._deterministic_action(state_json, evaluator_summary)

        # ── Try Azure OpenAI first ──────────────────────────────────────
        if llm_mode in ("azure", "azure-first", "hybrid"):
            try:
                action, latency = self._call_azure(state_json, evaluator_summary, timeout=self.llm_config.timeout_seconds_gpu)
                if action is not None:
                    return LLMResponse(action=action, model_used=self.llm_config.azure_deployment, latency_ms=latency, source="primary")
            except Exception as exc:
                print(f"[LLM] Azure OpenAI primary failed: {exc}")

            # Fall back to Ollama if Azure fails (azure-first and hybrid modes)
            if llm_mode in ("azure-first", "hybrid"):
                try:
                    action, latency = self._call_ollama(
                        model=self.llm_config.primary_model,
                        state_json=state_json,
                        evaluator_summary=evaluator_summary,
                        timeout=self.llm_config.timeout_seconds_gpu,
                    )
                    if action is not None:
                        return LLMResponse(action=action, model_used=self.llm_config.primary_model, latency_ms=latency, source="fallback")
                except Exception as exc:
                    print(f"[LLM] Ollama fallback also failed: {exc}")

        # ── Ollama primary / fallback (standalone ollama mode) ──────────
        if llm_mode in ("ollama",):
            try:
                action, latency = self._call_ollama(
                    model=self.llm_config.primary_model,
                    state_json=state_json,
                    evaluator_summary=evaluator_summary,
                    timeout=self.llm_config.timeout_seconds_gpu,
                )
                if action is not None:
                    return LLMResponse(action=action, model_used=self.llm_config.primary_model, latency_ms=latency, source="primary")
            except Exception as exc:
                print(f"[LLM] Ollama primary failed: {exc}")
            try:
                action, latency = self._call_ollama(
                    model=self.llm_config.fallback_model,
                    state_json=state_json,
                    evaluator_summary=evaluator_summary,
                    timeout=self.llm_config.timeout_seconds_gpu,
                )
                if action is not None:
                    return LLMResponse(action=action, model_used=self.llm_config.fallback_model, latency_ms=latency, source="fallback")
            except Exception as exc:
                print(f"[LLM] Ollama fallback also failed: {exc}")

        return self._deterministic_action(state_json, evaluator_summary)

    # ── Azure OpenAI ──────────────────────────────────────────────────────

    def _call_azure(self, state_json: str, evaluator_summary: dict, timeout: int) -> tuple[ControlAction | None, float]:
        """Call Azure OpenAI Chat Completions API with structured output."""
        try:
            import requests  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError("requests package required. Install: pip install requests")

        api_key = self.llm_config.azure_api_key
        endpoint = self.llm_config.azure_endpoint.rstrip("/")
        deployment = self.llm_config.azure_deployment
        api_version = self.llm_config.azure_api_version

        if not api_key or not endpoint:
            raise RuntimeError("Azure OpenAI credentials not configured. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in .env")

        # Strip trailing /openai/v1 or /openai path if present (some Azure configs include it)
        import re
        endpoint = re.sub(r"/openai(/v\d+)?$", "", endpoint)

        url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }

        prompt = self._build_prompt(state_json, evaluator_summary)

        payload = {
            "messages": [
                {"role": "system", "content": "You are an autonomous building energy optimizer. Respond ONLY with valid JSON matching the schema provided."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_completion_tokens": 512,
        }

        t0 = time.monotonic()
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        latency = (time.monotonic() - t0) * 1000.0

        choices = body.get("choices", [])
        if not choices:
            return None, latency

        raw = choices[0].get("message", {}).get("content", "").strip()
        action = self._parse_action(raw)
        return action, latency

    # ── Ollama (existing) ─────────────────────────────────────────────────

    def _call_ollama(self, model: str, state_json: str, evaluator_summary: dict, timeout: int) -> tuple[ControlAction | None, float]:
        """Call Ollama API with structured output. Returns (action, latency_ms)."""
        try:
            import requests  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError("requests package required for Ollama integration. Install: pip install requests")

        t0 = time.monotonic()
        url = f"{self.llm_config.ollama_base_url}/api/generate"
        prompt = self._build_prompt(state_json, evaluator_summary)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 256},
        }
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        latency = (time.monotonic() - t0) * 1000.0
        raw = body.get("response", "").strip()
        action = self._parse_action(raw)
        return action, latency

    # ── Shared utilities ──────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(state_json: str, evaluator_summary: dict) -> str:
        """Build the LLM prompt with compact state + evaluator signals."""
        return f"""You are an autonomous building energy optimizer. Your objective order is: safety and occupied comfort first, then energy reduction.

Respond ONLY with valid JSON matching this schema:
{{"action_id": "uuid", "zone_setpoints": {{"ZONE_NAME": {{"heating_c": 18.0, "cooling_c": 24.0}}}}, "mode": "hold|normal|precondition|setback", "rationale": "brief reason", "confidence": 0.8}}

CONSTRAINTS:
- Heating setpoint: 18-24 C, Cooling setpoint: 22-28 C
- Heating/cooling deadband: at least 2 C
- Max setpoint change per interval: 0.5 C
- Occupied zones: prioritize comfort (PMV in [-0.5, 0.5])
- Unoccupied zones: setback or preconditioning only

CURRENT STATE:
{state_json}

EVALUATOR SIGNALS:
{json.dumps(evaluator_summary, indent=2)}

JSON action:"""

    @staticmethod
    def _parse_action(raw: str) -> ControlAction | None:
        """Parse LLM response into a ControlAction. Returns None on failure."""
        if not raw:
            return None
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    return None
            else:
                return None

        try:
            zone_setpoints = {}
            for zone, sp in data.get("zone_setpoints", {}).items():
                zone_setpoints[zone] = ZoneSetpoints(
                    heating_c=float(sp["heating_c"]),
                    cooling_c=float(sp["cooling_c"]),
                )
            return ControlAction(
                action_id=data.get("action_id", ""),
                zone_setpoints=zone_setpoints,
                mode=data.get("mode", "hold"),
                rationale=data.get("rationale", ""),
                confidence=float(data.get("confidence", 0.5)),
            )
        except (KeyError, ValueError, TypeError):
            return None

    @staticmethod
    def _deterministic_action(state_json: str, evaluator_summary: dict, reason: str = "deterministic_default") -> LLMResponse:
        """Create a conservative hold action as fallback."""
        try:
            state_data = json.loads(state_json)
            zone_setpoints = {}
            for zone in state_data.get("zone_temps_c", {}):
                zone_setpoints[zone] = ZoneSetpoints(heating_c=18.0, cooling_c=28.0)
            action = ControlAction(
                action_id="fallback-" + uuid.uuid4().hex[:8],
                zone_setpoints=zone_setpoints,
                mode="hold",
                rationale=reason,
                confidence=0.0,
            )
        except Exception:
            action = None
        return LLMResponse(action=action, model_used="none", latency_ms=0.0, source="deterministic")
