"""Model Context Protocol (MCP) server for Eco-Loop.

Exposes building-observation, control-proposal, simulation, and audit tools
so any MCP-compatible LLM (Claude, GPT,  via Ollama, etc.) can reason
over EnergyPlus telemetry and propose HVAC actions without touching the
underlying simulation or safety layer directly.

Run as stdio server:
    uv run ecoloop mcp

Run as HTTP server (for remote clients):
    uv run ecoloop mcp --transport streamable-http --port 8000
"""
from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .config import RunConfig
from .synthetic import generate_synthetic_run
from .safety import SafetyEngine, SafetyDecision
from .schemas import ControlAction, ZoneSetpoints
from .baseline import BaselineController

# ── Globals ────────────────────────────────────────────────────────────

_config = RunConfig()
_mcp = FastMCP("Eco-Loop Building Agent")
_safety = SafetyEngine(_config.safety)
_baseline = BaselineController()

# Cached latest state (populated by observer tools)
_latest_state: dict = {}
_latest_action: dict = {}


# ── Helper ─────────────────────────────────────────────────────────────

def _state_reader():
    """Lazily import StateReader to avoid circular deps."""
    from .state import StateReader
    return StateReader(_config)


def _find_latest_output_dir(audit_only: bool = False) -> Path | None:
    """Find the most recent output directory with usable simulation data.

    Args:
        audit_only: If True, only return dirs with audit.jsonl.
    """
    out = _config.paths.data_output
    if not out.exists():
        return None

    all_dirs = [d for d in out.iterdir() if d.is_dir()]

    with_audit = [d for d in all_dirs if (d / "audit.jsonl").exists()]
    with_zone = [d for d in all_dirs if (d / "epluszsz.csv").exists()]

    if audit_only:
        candidates = with_audit
    else:
        # Prefer zone data (most complete), then audit
        candidates = with_zone or with_audit

    if not candidates:
        return None

    return sorted(candidates, key=lambda d: d.stat().st_mtime, reverse=True)[0]


def _read_zone_temps(output_dir: Path | None = None) -> dict:
    """Read current zone temperatures from latest simulation output."""
    sr = _state_reader()
    d = output_dir or _find_latest_output_dir()
    if d is None:
        return {}
    state = sr.read_latest(d)
    if state is None:
        return {}
    return state.zone_temps_c


# ── Observation Tools ──────────────────────────────────────────────────

@_mcp.tool()
def read_zone_telemetry() -> str:
    """Read current zone temperatures, humidity, occupancy, and outdoor conditions.

    Returns a JSON snapshot of the latest EnergyPlus simulation state.
    Generates synthetic data from EPW if no simulation output exists yet.
    """
    global _latest_state

    d = _find_latest_output_dir()
    if d is None:
        # Generate synthetic data
        sim_dir = _config.paths.data_output / "sim-mcp"
        sim_dir.mkdir(parents=True, exist_ok=True)
        epw = _config.paths.default_epw()
        generate_synthetic_run(sim_dir, epw)
        d = sim_dir

    sr = _state_reader()
    state = sr.read_latest(d)
    if state is None:
        # No CSV data — try to reconstruct from the latest audit event
        audit_path = d / "audit.jsonl"
        if audit_path.exists():
            events = []
            with open(audit_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            if events:
                latest = events[-1]
                sb = latest.get("state_before", {})
                return json.dumps({
                    "timestamp": latest.get("timestamp", ""),
                    "output_dir": str(d),
                    "zone_temps_c": sb.get("zone_temps_c", {}),
                    "zone_relative_humidity_pct": sb.get("zone_relative_humidity_pct", {}),
                    "zone_pmv": sb.get("zone_pmv", {}),
                    "zone_ppd_pct": sb.get("zone_ppd_pct", {}),
                    "outdoor_temp_c": sb.get("outdoor_temp_c", 0.0),
                    "hvac_power_kw": sb.get("hvac_power_kw"),
                    "occupancy": sb.get("occupancy", {}),
                    "forecast_outdoor_temp_c": sb.get("forecast_outdoor_temp_c", [])[:12],
                    "comfort_summary": {
                        "occupied_zones": len(sb.get("occupancy", {})),
                        "violations": 0,
                        "compliance_pct": 100.0,
                    },
                }, indent=2)
        return json.dumps({"error": "No simulation data available", "output_dir": str(d)}, indent=2)

    result = {
        "timestamp": state.timestamp.isoformat(),
        "output_dir": str(d),
        "zone_temps_c": state.zone_temps_c,
        "zone_relative_humidity_pct": state.zone_relative_humidity_pct,
        "zone_pmv": state.zone_pmv,
        "zone_ppd_pct": state.zone_ppd_pct,
        "outdoor_temp_c": state.outdoor_temp_c,
        "hvac_power_kw": state.hvac_power_kw,
        "occupancy": state.occupancy,
        "forecast_outdoor_temp_c": state.forecast_outdoor_temp_c[:12],  # next 3h
    }

    # Summarize comfort
    violations = 0
    total = 0
    for zone, temp in state.zone_temps_c.items():
        occ = state.occupancy.get(zone, 0)
        if occ > 0:
            total += 1
            pmv = state.zone_pmv.get(zone, 0)
            if pmv < -0.5 or pmv > 0.5:
                violations += 1

    result["comfort_summary"] = {
        "occupied_zones": total,
        "violations": violations,
        "compliance_pct": round((total - violations) / total * 100, 1) if total else 100.0,
    }

    # Store current setpoints so safety checks use actual current values
    result["zone_heating_setpoints_c"] = state.heating_setpoints_c
    result["zone_cooling_setpoints_c"] = state.cooling_setpoints_c

    _latest_state = result
    return json.dumps(result, indent=2)


@_mcp.tool()
def read_zone_history(steps: int = 4) -> str:
    """Read recent control step history from the audit log.

    Args:
        steps: Number of recent steps to return (default 4, max 20).
    """
    steps = min(max(steps, 1), 20)

    d = _find_latest_output_dir()
    if d is None:
        return json.dumps({"error": "No simulation data available"}, indent=2)

    audit_path = d / "audit.jsonl"
    if not audit_path.exists():
        return json.dumps({"error": f"No audit log at {audit_path}", "steps_returned": 0}, indent=2)

    events = []
    with open(audit_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    recent = events[-steps:] if events else []
    return json.dumps({"steps_returned": len(recent), "events": recent}, indent=2)


@_mcp.tool()
def read_safety_limits() -> str:
    """Return the active safety engine limits and rules.

    Returns the hard constraints that every ControlAction must satisfy.
    These limits cannot be overridden by the LLM.
    """
    limits = {
        "heating_setpoint_c": {
            "min": _config.safety.heating_min_c,
            "max": _config.safety.heating_max_c,
        },
        "cooling_setpoint_c": {
            "min": _config.safety.cooling_min_c,
            "max": _config.safety.cooling_max_c,
        },
        "min_deadband_c": _config.safety.minimum_deadband_c,
        "max_setpoint_delta_per_step_c": _config.safety.max_delta_per_interval_c,
        "max_setpoint_delta_per_hour_c": _config.safety.max_delta_per_hour_c,
        "pmv_target_band": list(_config.safety.pmv_comfort_band),
        "pmv_target_description": "ASHRAE-55 Category II",
        "unoccupied_setback": {
            "heating_c": _config.safety.unoccupied_setback_heating_c,
            "cooling_c": _config.safety.unoccupied_setback_cooling_c,
        },
        "description": (
            "These limits are enforced by the SafetyEngine and cannot be bypassed. "
            "Proposed setpoints outside these ranges will be clamped."
        ),
    }
    return json.dumps(limits, indent=2)


@_mcp.tool()
def check_action_safety(action_json: str) -> str:
    """Check whether a proposed ControlAction is safe without applying it.

    Args:
        action_json: JSON string matching the ControlAction schema.

    Returns validation result: accepted, clamped, or rejected with reasons.
    """
    try:
        action_data = json.loads(action_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid JSON: {exc}"}, indent=2)

    # Rebuild ControlAction from JSON
    try:
        zone_setpoints = {}
        for zone, sp in action_data.get("zone_setpoints", {}).items():
            zone_setpoints[zone] = ZoneSetpoints(
                heating_c=float(sp["heating_c"]),
                cooling_c=float(sp["cooling_c"]),
            )
        action = ControlAction(
            action_id=action_data.get("action_id", "mcp-check"),
            zone_setpoints=zone_setpoints,
            mode=action_data.get("mode", "normal"),
            rationale=action_data.get("rationale", ""),
            confidence=float(action_data.get("confidence", 0.5)),
        )
    except (KeyError, ValueError) as exc:
        return json.dumps({"error": f"Invalid action schema: {exc}"}, indent=2)

    # Build a minimal BuildingState for validation — use CURRENT setpoints, not proposed ones
    from .schemas import BuildingState
    current_heating = _latest_state.get("zone_heating_setpoints_c", {})
    current_cooling = _latest_state.get("zone_cooling_setpoints_c", {})
    state = BuildingState(
        timestamp=_latest_state.get("timestamp", ""),
        step_index=0,
        zone_temps_c=_latest_state.get("zone_temps_c", {}),
        zone_relative_humidity_pct=_latest_state.get("zone_relative_humidity_pct", {}),
        occupancy=_latest_state.get("occupancy", {}),
        heating_setpoints_c=current_heating or {z: 18.0 for z in _latest_state.get("zone_temps_c", {})},
        cooling_setpoints_c=current_cooling or {z: 28.0 for z in _latest_state.get("zone_temps_c", {})},
        zone_pmv=_latest_state.get("zone_pmv", {}),
        zone_ppd_pct=_latest_state.get("zone_ppd_pct", {}),
        outdoor_temp_c=_latest_state.get("outdoor_temp_c", 20.0),
        hvac_power_kw=_latest_state.get("hvac_power_kw"),
    )

    decision = _safety.validate(state, action)
    result = {
        "status": decision.status,
        "reasons": decision.reasons,
        "proposed_action": action.to_dict(),
        "applied_action": decision.action.to_dict() if decision.action else None,
    }
    return json.dumps(result, indent=2)


# ── Control Tools ──────────────────────────────────────────────────────

@_mcp.tool()
def propose_setpoints(
    zone_setpoints: str,
    mode: str = "normal",
    rationale: str = "",
) -> str:
    """Propose zone setpoint changes. The SafetyEngine validates before application.

    Args:
        zone_setpoints: JSON object mapping zone names to {heating_c, cooling_c},
            e.g. '{"SPACE1-1": {"heating_c": 20.0, "cooling_c": 26.0}}'
        mode: One of "hold", "normal", "precondition", "setback".
        rationale: Short operational reason for this change.

    Returns the safety-validated action and whether it was accepted or clamped.
    """
    global _latest_action

    try:
        sp_data = json.loads(zone_setpoints)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid zone_setpoints JSON: {exc}"}, indent=2)

    # Build ControlAction
    zone_sp = {}
    for zone, sp in sp_data.items():
        zone_sp[zone] = ZoneSetpoints(
            heating_c=float(sp["heating_c"]),
            cooling_c=float(sp["cooling_c"]),
        )

    action = ControlAction(
        action_id=f"mcp-{mode}",
        zone_setpoints=zone_sp,
        mode=mode,
        rationale=rationale,
        confidence=0.8,
    )

    # Validate against safety — use CURRENT setpoints, not proposed
    from .schemas import BuildingState
    current_heating = _latest_state.get("zone_heating_setpoints_c", {})
    current_cooling = _latest_state.get("zone_cooling_setpoints_c", {})
    state = BuildingState(
        timestamp=_latest_state.get("timestamp", ""),
        step_index=0,
        zone_temps_c=_latest_state.get("zone_temps_c", {}),
        zone_relative_humidity_pct=_latest_state.get("zone_relative_humidity_pct", {}),
        occupancy=_latest_state.get("occupancy", {}),
        heating_setpoints_c=current_heating or {z: 18.0 for z in _latest_state.get("zone_temps_c", {})},
        cooling_setpoints_c=current_cooling or {z: 28.0 for z in _latest_state.get("zone_temps_c", {})},
        zone_pmv=_latest_state.get("zone_pmv", {}),
        zone_ppd_pct=_latest_state.get("zone_ppd_pct", {}),
        outdoor_temp_c=_latest_state.get("outdoor_temp_c", 20.0),
        hvac_power_kw=_latest_state.get("hvac_power_kw"),
    )

    decision = _safety.validate(state, action)

    result = {
        "proposed": action.to_dict(),
        "safety_status": decision.status,
        "reasons": decision.reasons,
        "applied_action": decision.action.to_dict() if decision.action else None,
    }

    _latest_action = result
    return json.dumps(result, indent=2)


# ── Simulation Tools ──────────────────────────────────────────────────

@_mcp.tool()
def run_simulation_comparison() -> str:
    """Run a full baseline vs AI simulation comparison.

    Generates synthetic data from EPW if needed, runs both controllers,
    and returns energy/comfort/peak comparison metrics.
    """
    from .controller import ClosedLoopController
    from .metrics import compare, ComparisonMetrics
    import json as json_mod

    # Run baseline
    print("[MCP] Running baseline simulation...")
    baseline_ctrl = ClosedLoopController(_config)
    baseline_events = baseline_ctrl.run(mode="baseline")
    b_kwh = sum(e.state_before.get("hvac_power_kw", 0) or 0 for e in baseline_events) * 0.25
    b_peak = max((e.state_before.get("hvac_power_kw", 0) or 0) for e in baseline_events)

    # Run AI
    print("[MCP] Running AI simulation...")
    ai_ctrl = ClosedLoopController(_config)
    ai_events = ai_ctrl.run(mode="ai")
    a_kwh = sum(e.state_before.get("hvac_power_kw", 0) or 0 for e in ai_events) * 0.25
    a_peak = max((e.state_before.get("hvac_power_kw", 0) or 0) for e in ai_events)

    # Compare
    cm = compare(b_kwh, a_kwh, b_peak, a_peak)
    result = {
        "baseline": {
            "energy_kwh": round(b_kwh, 3),
            "peak_kw": round(b_peak, 3),
        },
        "ai": {
            "energy_kwh": round(a_kwh, 3),
            "peak_kw": round(a_peak, 3),
        },
        "savings": {
            "energy_pct": round(cm.energy_savings_pct, 2),
            "peak_reduction_pct": round(cm.peak_reduction_pct, 2),
        },
    }
    return json.dumps(result, indent=2)


# ── Audit Tools ────────────────────────────────────────────────────────

@_mcp.tool()
def read_audit_log(steps: int = 10) -> str:
    """Read the most recent audit log entries from the latest run.

    Args:
        steps: Number of recent events to return (default 10, max 50).
    """
    steps = min(max(steps, 1), 50)

    # Look for any directory with an audit.jsonl (not just zone data dirs)
    d = _find_latest_output_dir(audit_only=True)
    if d is None:
        return json.dumps({"error": "No audit data available"}, indent=2)

    audit_path = d / "audit.jsonl"
    if not audit_path.exists():
        return json.dumps({"error": f"No audit log at {audit_path}", "run_dir": str(d)}, indent=2)

    events = []
    with open(audit_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    recent = events[-steps:] if events else []
    summary = {
        "run_dir": str(d),
        "total_events": len(events),
        "returned": len(recent),
        "events": recent,
    }
    return json.dumps(summary, indent=2)


@_mcp.tool()
def list_output_directories() -> str:
    """List all simulation output directories under data/output/."""
    out = _config.paths.data_output
    if not out.exists():
        return json.dumps({"directories": []}, indent=2)

    dirs = []
    for d in sorted(out.iterdir()):
        if d.is_dir():
            has_zsz = (d / "epluszsz.csv").exists()
            has_audit = (d / "audit.jsonl").exists()
            has_metrics = (d / "metrics.json").exists()
            dirs.append({
                "name": d.name,
                "path": str(d),
                "has_zone_data": has_zsz,
                "has_audit": has_audit,
                "has_metrics": has_metrics,
            })

    return json.dumps({"directories": dirs, "count": len(dirs)}, indent=2)


# ── Entry Point ────────────────────────────────────────────────────────

def run_server(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the MCP server with the specified transport."""
    import sys
    print(f"[Eco-Loop MCP] Starting server (transport={transport})...", file=sys.stderr)
    _mcp.run(transport=transport, host=host, port=port)
