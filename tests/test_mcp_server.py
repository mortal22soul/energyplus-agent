"""Tests for the MCP server: tool registration, underlying function logic."""
from __future__ import annotations

import json
import sys

import pytest


# ── MCP SDK availability ───────────────────────────────────────────────

try:
    from mcp.server.fastmcp import FastMCP

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

pytestmark = pytest.mark.skipif(not HAS_MCP, reason="mcp SDK not installed")


# ── Module import (once per session) ───────────────────────────────────

_MOD_NAME = "ecoloop.mcp_server"

# Clear any stale cached module
for name in list(sys.modules):
    if name == _MOD_NAME or name.startswith(_MOD_NAME + "."):
        del sys.modules[name]

_mod = __import__(_MOD_NAME, fromlist=["_mcp"])
_mcp = _mod._mcp

# Tool manager exposes tools via _tools dict
_tools = _mcp._tool_manager._tools

# Function references
read_safety_limits = _mod.read_safety_limits
check_action_safety = _mod.check_action_safety
propose_setpoints = _mod.propose_setpoints
read_zone_telemetry = _mod.read_zone_telemetry
read_audit_log = _mod.read_audit_log
list_output_directories = _mod.list_output_directories


# ── Tool registration tests ────────────────────────────────────────────

class TestMCPToolRegistration:
    """Verify the MCP server registers the expected tools."""

    def test_server_creates_fastmcp(self):
        assert _mcp is not None
        assert _mcp.name == "Eco-Loop Building Agent"

    def test_tool_count(self):
        assert len(_tools) >= 7

    def test_read_zone_telemetry_is_tool(self):
        assert "read_zone_telemetry" in _tools

    def test_read_zone_history_is_tool(self):
        assert "read_zone_history" in _tools

    def test_read_safety_limits_is_tool(self):
        assert "read_safety_limits" in _tools

    def test_check_action_safety_is_tool(self):
        assert "check_action_safety" in _tools

    def test_propose_setpoints_is_tool(self):
        assert "propose_setpoints" in _tools

    def test_run_simulation_comparison_is_tool(self):
        assert "run_simulation_comparison" in _tools

    def test_read_audit_log_is_tool(self):
        assert "read_audit_log" in _tools

    def test_list_output_directories_is_tool(self):
        assert "list_output_directories" in _tools


# ── read_safety_limits tests ───────────────────────────────────────────

class TestReadSafetyLimits:
    def test_returns_json_with_limits(self):
        result = json.loads(read_safety_limits())

        assert result["heating_setpoint_c"]["min"] == 18.0
        assert result["heating_setpoint_c"]["max"] == 24.0
        assert result["cooling_setpoint_c"]["min"] == 22.0
        assert result["cooling_setpoint_c"]["max"] == 28.0
        assert result["min_deadband_c"] == 2.0
        assert result["pmv_target_band"] == [-0.5, 0.5]
        assert "ASHRAE-55" in result["pmv_target_description"]
        assert "description" in result


# ── check_action_safety tests ──────────────────────────────────────────

class TestCheckActionSafety:
    """Ensure _latest_state is populated before safety checks."""
    def _ensure_state(self):
        read_zone_telemetry()

    def test_invalid_json_returns_error(self):
        result = json.loads(check_action_safety("not-valid-json"))
        assert "error" in result

    def test_valid_action_accepted(self):
        self._ensure_state()
        # Use setpoints within rate limit of DEFAULT setpoints (18.0/28.0)
        # Rate limit: ±0.5°C per 15-min step → heating 18.0-18.5, cooling 27.5-28.0
        action = {
            "action_id": "mcp-test-1",
            "zone_setpoints": {
                "SPACE1-1": {"heating_c": 18.5, "cooling_c": 27.5},
            },
            "mode": "normal",
            "rationale": "modest adjustment within rate limit",
            "confidence": 0.8,
        }
        result = json.loads(check_action_safety(json.dumps(action)))
        assert result["status"] == "accepted"
        # No clamping reasons (informational "held" reasons for untouched zones are fine)
        clamp_reasons = [r for r in result["reasons"] if "clamped" in r.lower() or "deadband" in r.lower() or "outside" in r.lower()]
        assert len(clamp_reasons) == 0

    def test_action_below_min_heating_is_clamped(self):
        """Proposed heating 10°C is clamped by rate limit (±0.5/step)."""
        self._ensure_state()

        action = {
            "action_id": "mcp-clamp-low",
            "zone_setpoints": {
                "SPACE1-1": {"heating_c": 10.0, "cooling_c": 26.0},
            },
            "mode": "normal",
            "rationale": "below min",
            "confidence": 0.9,
        }
        result = json.loads(check_action_safety(json.dumps(action)))
        assert result["status"] == "clamped"
        assert len(result["reasons"]) > 0
        # Proposed heating 10°C → clamped (never below min 18°C in steady state)
        # The exact value depends on current setpoint ± rate limit, so just verify
        # it changed from the proposed value (proving clamping occurred)
        applied = result["applied_action"]["zone_setpoints"]["SPACE1-1"]
        assert applied["heating_c"] != 10.0  # clamping occurred
        assert applied["heating_c"] >= 18.0  # never below safety minimum

    def test_action_above_max_cooling_is_clamped(self):
        """Proposed cooling 40°C is clamped by rate limit."""
        self._ensure_state()

        action = {
            "action_id": "mcp-clamp-high",
            "zone_setpoints": {
                "SPACE1-1": {"heating_c": 20.0, "cooling_c": 40.0},
            },
            "mode": "normal",
            "rationale": "above max",
            "confidence": 0.9,
        }
        result = json.loads(check_action_safety(json.dumps(action)))
        assert result["status"] == "clamped"
        assert len(result["reasons"]) > 0
        # Proposed cooling 40°C → clamped (never above max 28°C in steady state)
        applied = result["applied_action"]["zone_setpoints"]["SPACE1-1"]
        assert applied["cooling_c"] != 40.0  # clamping occurred
        assert applied["cooling_c"] <= 28.0  # never above safety maximum


# ── propose_setpoints tests ────────────────────────────────────────────

class TestProposeSetpoints:
    def test_invalid_json_returns_error(self):
        result = json.loads(propose_setpoints("bad-json", "normal", "test"))
        assert "error" in result

    def test_valid_proposal_accepted(self):
        # Populate _latest_state first
        read_zone_telemetry()

        # Use setpoints within rate limit of actual current values from sim-baseline
        # (~21.4/23.6): ±0.5°C per step → heating 20.9-22.0, cooling 23.1-24.1
        result_str = propose_setpoints(
            json.dumps({"SPACE1-1": {"heating_c": 21.4, "cooling_c": 24.0}}),
            "normal",
            "MCP test proposal",
        )
        result = json.loads(result_str)
        assert result["safety_status"] == "accepted"
        assert result["proposed"]["zone_setpoints"]["SPACE1-1"]["cooling_c"] == 24.0

    def test_unsafe_proposal_clamped(self):
        read_zone_telemetry()

        result_str = propose_setpoints(
            json.dumps({"SPACE1-1": {"heating_c": 10.0, "cooling_c": 40.0}}),
            "normal",
            "intentionally unsafe",
        )
        result = json.loads(result_str)
        assert result["safety_status"] == "clamped"


# ── read_zone_telemetry tests ──────────────────────────────────────────

class TestReadZoneTelemetry:
    def test_returns_json_snapshot(self):
        result = json.loads(read_zone_telemetry())

        assert "zone_temps_c" in result
        assert "zone_relative_humidity_pct" in result
        assert "outdoor_temp_c" in result
        assert "occupancy" in result
        assert "comfort_summary" in result

    def test_zone_temps_are_floats(self):
        result = json.loads(read_zone_telemetry())

        for zone, temp in result["zone_temps_c"].items():
            assert isinstance(temp, float), f"{zone} should be float, got {type(temp)}"

    def test_comfort_summary_keys(self):
        result = json.loads(read_zone_telemetry())
        cs = result["comfort_summary"]
        assert "occupied_zones" in cs
        assert "violations" in cs
        assert "compliance_pct" in cs


# ── read_audit_log tests ───────────────────────────────────────────────

class TestReadAuditLog:
    def test_returns_json_with_events(self):
        result = json.loads(read_audit_log(5))
        assert "events" in result
        assert "total_events" in result
        assert "returned" in result
        assert isinstance(result["events"], list)
        assert result["returned"] <= 5


# ── list_output_directories tests ──────────────────────────────────────

class TestListOutputDirectories:
    def test_returns_json_with_dirs(self):
        result = json.loads(list_output_directories())
        assert "directories" in result
        assert "count" in result
        assert isinstance(result["directories"], list)
