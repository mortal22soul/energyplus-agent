"""Multi-agent LangGraph architecture for Eco-Loop.

Implements a hierarchical multi-agent system:
  - EnergyAgent: minimize energy consumption
  - ComfortAgent: maintain thermal comfort (PMV in [-0.5, 0.5])
  - ForecastAgent: pre-conditioning based on weather predictions
  - SupervisorAgent: merge proposals with priority Safety > Comfort > Energy
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, TypedDict

from .schemas import ControlAction, ZoneSetpoints
from .config import RunConfig


# ── Agent State ────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """Shared state for the multi-agent graph."""
    building_state_json: str
    evaluator_summary: dict
    zones: list[str]
    energy_proposal: dict | None
    comfort_proposal: dict | None
    forecast_proposal: dict | None
    merged_action: ControlAction | None
    error: str | None


# ── Prompts ────────────────────────────────────────────────────────────

ENERGY_AGENT_PROMPT = """You are an Energy Optimization Agent for a 5-zone office building.
Your SOLE objective: minimize HVAC energy consumption while respecting hard constraints.

CONSTRAINTS (non-negotiable):
- Heating setpoint: 18-24 C, Cooling setpoint: 22-28 C
- Minimum deadband (cooling - heating): 2 C
- Max setpoint change per 15-min interval: 0.5 C
- Occupied zones: PMV must stay in [-0.5, 0.5]

STRATEGIES:
- Widen the deadband (lower heating, raise cooling) to reduce HVAC runtime
- Use setback aggressively for unoccupied zones (18/28 C)
- Avoid unnecessary heating/cooling when outdoor temps are mild
- Prioritize zones with highest power draw

CURRENT STATE:
{state_json}

EVALUATOR SIGNALS:
{evaluator_json}

Respond ONLY with valid JSON:
{{"zone_setpoints": {{"ZONE_NAME": {{"heating_c": 18.0, "cooling_c": 28.0}}}}, "mode": "setback", "rationale": "brief reason", "confidence": 0.8}}"""


COMFORT_AGENT_PROMPT = """You are a Comfort Monitoring Agent for a 5-zone office building.
Your SOLE objective: maintain thermal comfort for ALL occupied zones.

COMFORT TARGET: PMV in [-0.5, 0.5] (ASHRAE-55 Category II)

CONSTRAINTS (non-negotiable):
- Heating setpoint: 18-24 C, Cooling setpoint: 22-28 C
- Minimum deadband (cooling - heating): 2 C
- Max setpoint change per 15-min interval: 0.5 C

STRATEGIES:
- If PMV < -0.5 (too cold): raise heating setpoint
- If PMV > 0.5 (too warm): lower cooling setpoint
- Occupied zones are TOP PRIORITY
- For unoccupied zones: use moderate setback only

CURRENT STATE:
{state_json}

EVALUATOR SIGNALS:
{evaluator_json}

Respond ONLY with valid JSON:
{{"zone_setpoints": {{"ZONE_NAME": {{"heating_c": 21.0, "cooling_c": 24.0}}}}, "mode": "normal", "rationale": "brief reason", "confidence": 0.8}}"""


FORECAST_AGENT_PROMPT = """You are a Forecasting and Pre-conditioning Agent for a 5-zone office building.
Your SOLE objective: optimize HVAC based on weather predictions and occupancy schedule.

OCCUPANCY SCHEDULE: Mon-Fri 8am-6pm

CONSTRAINTS (non-negotiable):
- Heating setpoint: 18-24 C, Cooling setpoint: 22-28 C
- Minimum deadband (cooling - heating): 2 C
- Max setpoint change per 15-min interval: 0.5 C

STRATEGIES:
- If hot weather forecast + currently unoccupied: pre-cool before occupancy
- If cold weather forecast + currently unoccupied: pre-heat before occupancy
- If mild forecast: use wide deadband to save energy
- Near end of occupancy: begin gradual setback

CURRENT STATE:
{state_json}

EVALUATOR SIGNALS:
{evaluator_json}

Respond ONLY with valid JSON:
{{"zone_setpoints": {{"ZONE_NAME": {{"heating_c": 20.0, "cooling_c": 25.0}}}}, "mode": "precondition", "rationale": "brief reason", "confidence": 0.8}}"""


SUPERVISOR_PROMPT = """You are the Supervisor Agent for a 5-zone office building HVAC system.
You must merge proposals from three specialized agents into ONE final action.

PRIORITY ORDER (non-negotiable):
1. SAFETY: All setpoints must be within limits (18-24 heating, 22-28 cooling, 2C deadband)
2. COMFORT: Occupied zones must have PMV in [-0.5, 0.5]
3. ENERGY: Minimize energy where comfort allows

MERGE RULES:
- For occupied zones: use ComfortAgent's setpoints unless they violate safety
- For unoccupied zones: use EnergyAgent's setpoints (wider deadband saves energy)
- Apply ForecastAgent's pre-conditioning only for unoccupied zones approaching occupancy
- If agents disagree on an occupied zone, prefer the tighter comfort band

ENERGY PROPOSAL:
{energy_json}

COMFORT PROPOSAL:
{comfort_json}

FORECAST PROPOSAL:
{forecast_json}

CURRENT STATE:
{state_json}

Respond ONLY with valid JSON:
{{"zone_setpoints": {{"ZONE_NAME": {{"heating_c": 20.0, "cooling_c": 25.0}}}}, "mode": "normal", "rationale": "brief merged reason", "confidence": 0.8}}"""


# ── LLM Call Helper ────────────────────────────────────────────────────

def _call_azure_openai(prompt: str, config: RunConfig) -> dict | None:
    """Call Azure OpenAI and return parsed JSON dict, or None on failure."""
    import requests

    llm = config.llm
    api_key = llm.azure_api_key
    endpoint = llm.azure_endpoint.rstrip("/")
    deployment = llm.azure_deployment
    api_version = llm.azure_api_version

    if not api_key or not endpoint:
        return None

    # Strip trailing /openai/v1 or /openai path
    endpoint = re.sub(r"/openai(/v\d+)?$", "", endpoint)

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    payload = {
        "messages": [
            {"role": "system", "content": "You are an autonomous building HVAC optimization agent. Respond ONLY with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_completion_tokens": 512,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=llm.timeout_seconds_gpu)
        resp.raise_for_status()
        body = resp.json()
        choices = body.get("choices", [])
        if not choices:
            return None
        raw = choices[0].get("message", {}).get("content", "").strip()
        if not raw:
            return None
        # Clean markdown fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(cleaned[start:end])
            return None
    except requests.exceptions.HTTPError as exc:
        print(f"[MultiAgent] Azure call failed: {exc}")
        if exc.response is not None:
            print(f"  Response body: {exc.response.text}")
        return None
    except Exception as exc:
        print(f"[MultiAgent] Azure call failed: {exc}")
        return None


def _proposal_to_dict(proposal: dict | None) -> dict:
    """Normalize a proposal to a safe dict for logging/prompting."""
    if proposal is None:
        return {"zone_setpoints": {}, "mode": "hold", "rationale": "no proposal", "confidence": 0.0}
    return proposal


def _dict_to_action(data: dict, zones: list[str]) -> ControlAction | None:
    """Parse a dict into a ControlAction, returning None on failure."""
    try:
        zone_setpoints = {}
        for zone, sp in data.get("zone_setpoints", {}).items():
            zone_setpoints[zone] = ZoneSetpoints(
                heating_c=float(sp["heating_c"]),
                cooling_c=float(sp["cooling_c"]),
            )
        # Fill missing zones with safe defaults
        for zone in zones:
            if zone not in zone_setpoints:
                zone_setpoints[zone] = ZoneSetpoints(heating_c=18.0, cooling_c=28.0)

        return ControlAction(
            action_id=f"multi-{uuid.uuid4().hex[:8]}",
            zone_setpoints=zone_setpoints,
            mode=data.get("mode", "normal"),
            rationale=data.get("rationale", "multi-agent merged"),
            confidence=float(data.get("confidence", 0.7)),
        )
    except (KeyError, ValueError, TypeError):
        return None


# ── Graph Node Functions ──────────────────────────────────────────────

def agents_node(state: AgentState, config: RunConfig) -> dict:
    """Run all three sub-agents concurrently."""
    import concurrent.futures

    def run_energy():
        prompt = ENERGY_AGENT_PROMPT.format(
            state_json=state["building_state_json"],
            evaluator_json=json.dumps(state["evaluator_summary"], indent=2),
        )
        return _proposal_to_dict(_call_azure_openai(prompt, config))

    def run_comfort():
        prompt = COMFORT_AGENT_PROMPT.format(
            state_json=state["building_state_json"],
            evaluator_json=json.dumps(state["evaluator_summary"], indent=2),
        )
        return _proposal_to_dict(_call_azure_openai(prompt, config))

    def run_forecast():
        prompt = FORECAST_AGENT_PROMPT.format(
            state_json=state["building_state_json"],
            evaluator_json=json.dumps(state["evaluator_summary"], indent=2),
        )
        return _proposal_to_dict(_call_azure_openai(prompt, config))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f_energy = executor.submit(run_energy)
        f_comfort = executor.submit(run_comfort)
        f_forecast = executor.submit(run_forecast)
        
        return {
            "energy_proposal": f_energy.result(),
            "comfort_proposal": f_comfort.result(),
            "forecast_proposal": f_forecast.result()
        }


def supervisor_node(state: AgentState, config: RunConfig) -> dict:
    """Supervisor agent: merges sub-agent proposals."""
    prompt = SUPERVISOR_PROMPT.format(
        energy_json=json.dumps(state.get("energy_proposal", {}), indent=2),
        comfort_json=json.dumps(state.get("comfort_proposal", {}), indent=2),
        forecast_json=json.dumps(state.get("forecast_proposal", {}), indent=2),
        state_json=state["building_state_json"],
    )
    merged = _call_azure_openai(prompt, config)
    if merged is None:
        # Deterministic merge: prefer comfort for occupied, energy for unoccupied
        merged = _deterministic_merge(state)

    zones = state.get("zones", [])
    action = _dict_to_action(merged, zones)
    return {"merged_action": action}


def _deterministic_merge(state: AgentState) -> dict:
    """Deterministic fallback merge when supervisor LLM fails."""
    comfort = state.get("comfort_proposal", {})
    energy = state.get("energy_proposal", {})
    forecast = state.get("forecast_proposal", {})

    # Parse occupancy from state
    try:
        state_data = json.loads(state["building_state_json"])
        occupancy = state_data.get("occupancy", {})
    except (json.JSONDecodeError, KeyError):
        occupancy = {}

    merged_setpoints = {}
    all_zones = set()
    for src in [comfort, energy, forecast]:
        all_zones.update(src.get("zone_setpoints", {}).keys())

    for zone in all_zones:
        occ = occupancy.get(zone, 0)
        if occ > 0:
            # Occupied: prefer comfort
            sp = comfort.get("zone_setpoints", {}).get(zone)
            if not sp:
                sp = energy.get("zone_setpoints", {}).get(zone)
            if not sp:
                sp = {"heating_c": 20.0, "cooling_c": 25.0}
        else:
            # Unoccupied: prefer energy (wider band)
            sp = energy.get("zone_setpoints", {}).get(zone)
            if not sp:
                sp = forecast.get("zone_setpoints", {}).get(zone)
            if not sp:
                sp = {"heating_c": 18.0, "cooling_c": 28.0}
        merged_setpoints[zone] = sp

    return {
        "zone_setpoints": merged_setpoints,
        "mode": "normal",
        "rationale": "deterministic merge: comfort for occupied, energy for unoccupied",
        "confidence": 0.6,
    }


# ── LangGraph Construction ────────────────────────────────────────────

def build_multi_agent_graph(config: RunConfig):
    """Build the LangGraph StateGraph for multi-agent orchestration.

    Returns None if langgraph is not available.
    """
    try:
        from langgraph.graph import StateGraph, START, END
    except ImportError:
        return None

    graph = StateGraph(AgentState)

    # Add nodes with config binding
    graph.add_node("agents", lambda state: agents_node(state, config))
    graph.add_node("supervisor", lambda state: supervisor_node(state, config))

    # Sequential execution but agents run concurrently internally
    graph.add_edge(START, "agents")
    graph.add_edge("agents", "supervisor")
    graph.add_edge("supervisor", END)

    return graph.compile()


# ── Public API ─────────────────────────────────────────────────────────

class MultiAgentController:
    """Multi-agent LLM controller using LangGraph.

    Falls back to single-agent LLMClient if LangGraph is unavailable
    or if the graph fails.
    """

    def __init__(self, config: RunConfig):
        self.config = config
        self._graph = build_multi_agent_graph(config)

    @property
    def available(self) -> bool:
        return self._graph is not None

    def decide(self, state_json: str, evaluator_summary: dict) -> ControlAction | None:
        """Run the multi-agent graph and return a merged ControlAction."""
        if not self.available:
            return None

        try:
            state_data = json.loads(state_json)
            zones = list(state_data.get("zone_temps_c", {}).keys())
        except (json.JSONDecodeError, KeyError):
            zones = []

        initial_state: AgentState = {
            "building_state_json": state_json,
            "evaluator_summary": evaluator_summary,
            "zones": zones,
            "energy_proposal": None,
            "comfort_proposal": None,
            "forecast_proposal": None,
            "merged_action": None,
            "error": None,
        }

        try:
            t0 = time.monotonic()
            result = self._graph.invoke(initial_state)
            latency = (time.monotonic() - t0) * 1000.0
            action = result.get("merged_action")
            if action is not None:
                print(f"  [MultiAgent] Completed in {latency:.0f}ms")
            return action
        except Exception as exc:
            print(f"  [MultiAgent] Graph execution failed: {exc}")
            return None
