"""Closed-loop controller: offline evaluation over parsed EnergyPlus output."""
from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .schemas import BuildingState, ControlAction, StateSummary
from .state import StateReader, build_evaluator_summary
from .safety import SafetyEngine, SafetyDecision
from .agent import LLMClient, LLMResponse
from .multi_agent import MultiAgentController
from .baseline import BaselineController
from .config import RunConfig
from .audit import append_event
from .synthetic import generate_synthetic_run


@dataclass
class ControlEvent:
    step_index: int
    timestamp: datetime
    state_before: dict
    evaluator_summary: dict
    llm_response: LLMResponse
    safety_decision: SafetyDecision
    state_after: dict | None = None


class ClosedLoopController:
    """Runs the observe → decide → validate → actuate loop over pre-simulated output.

    For the demo, runs one full EnergyPlus simulation, then iterates over the
    output timesteps making decisions. This is testable, offline, and fast.
    The live callback path is proven separately (runtime spike).

    If the existing simulation output only contains sizing-day data (constant values),
    synthetic time-series data is generated from the EPW weather file to provide
    realistic time-varying conditions for the LLM to reason about.
    """

    def __init__(self, config: RunConfig, idf_path: Path | None = None, epw_path: Path | None = None):
        self.config = config
        self.idf_path = idf_path or config.paths.default_idf()
        self.epw_path = epw_path or config.paths.default_epw()
        self.state_reader = StateReader(config)
        self.safety = SafetyEngine(config.safety)
        self.llm = LLMClient(config)
        self.multi_agent = MultiAgentController(config)
        self.baseline = BaselineController()
        self.history: list[StateSummary] = []
        self.events: list[ControlEvent] = []

    def run(self, mode: str = "ai", working_idf: Path | None = None) -> list[ControlEvent]:
        """Run the control loop in offline mode over parsed simulation output.

        Args:
            mode: "ai" for LLM-driven, "baseline" for fixed schedule.
            working_idf: optional IDF path to run (for baseline comparison with
                         a different setpoint schedule).

        Returns:
            List of ControlEvents, one per control step.
        """
        steps = self.config.total_steps
        timestep_min = self.config.control.timestep_minutes

        # Set up run directory
        run_id = f"{mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_dir = self.config.paths.data_output / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir

        # Run simulation (or reuse existing)
        sim_dir = self._run_or_reuse_simulation(working_idf, run_dir, mode)

        # Read all timesteps
        all_rows = self._read_all_timesteps(sim_dir)
        if not all_rows:
            raise RuntimeError(f"No data rows found in {sim_dir}")

        # Control at every timestep_min boundary
        control_indices = list(range(0, min(len(all_rows), steps), 1))

        self.events = []
        self.history = []

        from .mpc import SimpleRCThermalModel, ZONE_THERMAL_PARAMS
        thermal_models = {
            z: SimpleRCThermalModel(ZONE_THERMAL_PARAMS.get(z))
            for z in ["SPACE1-1", "SPACE2-1", "SPACE3-1", "SPACE4-1", "SPACE5-1"]
        }
        current_zone_temps = None
        current_heating_sp = None
        current_cooling_sp = None

        for row_idx in control_indices:
            row = all_rows[row_idx]
            state = self.state_reader._build_state_from_row(row, row_idx, all_rows)
            if state is None:
                continue

            import dataclasses
            # Override CSV open-loop temps with closed-loop dynamically tracked temps
            if current_zone_temps is None:
                current_zone_temps = dict(state.zone_temps_c)
                current_heating_sp = dict(state.heating_setpoints_c)
                current_cooling_sp = dict(state.cooling_setpoints_c)
            else:
                state = dataclasses.replace(
                    state,
                    zone_temps_c=dict(current_zone_temps),
                    heating_setpoints_c=dict(current_heating_sp),
                    cooling_setpoints_c=dict(current_cooling_sp)
                )

            state_json = json.dumps(state.to_dict(), default=str, indent=2)
            evaluator_summary = build_evaluator_summary(state, self.history, self.config)

            # ── Decide ──
            if mode == "baseline":
                proposal = self.baseline.decide(state)
                llm_response = LLMResponse(
                    action=proposal, model_used="baseline", latency_ms=0.0, source="deterministic"
                )
            else:
                # Try multi-agent first, fall back to single-agent LLM
                import time as _time
                multi_action = None
                if self.multi_agent.available:
                    t0 = _time.monotonic()
                    multi_action = self.multi_agent.decide(state_json, evaluator_summary)
                    latency = (_time.monotonic() - t0) * 1000.0
                    if multi_action is not None:
                        llm_response = LLMResponse(
                            action=multi_action,
                            model_used=f"multi-agent/{self.config.llm.azure_deployment}",
                            latency_ms=latency,
                            source="primary",
                        )
                if multi_action is None:
                    llm_response = self.llm.decide(state_json, evaluator_summary)

            # ── Validate ──
            safety_decision = self.safety.validate(state, llm_response.action)

            # ── Dynamic Simulation Update ──
            # Calculate power consumption for this step and predict temperatures for next step
            action = safety_decision.action
            if action:
                dt_hours = self.config.control.timestep_minutes / 60.0
                total_power = 0.0
                next_temps = {}
                for zone, model in thermal_models.items():
                    sp = action.zone_setpoints.get(zone)
                    if sp:
                        h_sp, c_sp = sp.heating_c, sp.cooling_c
                    else:
                        h_sp, c_sp = 21.0, 24.0
                    
                    t_curr = state.zone_temps_c.get(zone, 21.0)
                    t_out = state.outdoor_temp_c
                    
                    zone_power = model.estimate_power_kw(t_curr, t_out, h_sp, c_sp)
                    total_power += zone_power
                    
                    occ = state.occupancy.get(zone, 0.0)
                    next_t = model.predict(t_curr, t_out, h_sp, c_sp, dt_hours, internal_gains_kw=0.5 * occ)
                    next_temps[zone] = next_t
                
                # Override the logged state values with dynamically calculated ones
                state = dataclasses.replace(
                    state,
                    hvac_power_kw=total_power,
                    facility_power_kw=total_power + sum(state.occupancy.values()) * 0.5,
                    heating_setpoints_c={z: sp.heating_c for z, sp in safety_decision.action.zone_setpoints.items()} if safety_decision.action else current_heating_sp,
                    cooling_setpoints_c={z: sp.cooling_c for z, sp in safety_decision.action.zone_setpoints.items()} if safety_decision.action else current_cooling_sp,
                )
                current_zone_temps = next_temps
                current_heating_sp = state.heating_setpoints_c
                current_cooling_sp = state.cooling_setpoints_c

            # Update history
            state_summary = StateSummary(
                timestamp=state.timestamp,
                facility_power_kw=state.facility_power_kw,
                zone_temps_c=dict(state.zone_temps_c),
            )
            self.history.append(state_summary)
            if len(self.history) > self.config.control.history_window:
                self.history = self.history[-self.config.control.history_window :]

            # Log
            event = ControlEvent(
                step_index=row_idx,
                timestamp=state.timestamp,
                state_before=json.loads(state_json),
                evaluator_summary=evaluator_summary,
                llm_response=llm_response,
                safety_decision=safety_decision,
                state_after=state.to_dict(),
            )
            self.events.append(event)

            append_event(run_dir / "audit.jsonl", {
                "step": row_idx,
                "timestamp": state.timestamp.isoformat(),
                "mode": mode,
                "llm_source": llm_response.source,
                "llm_model": llm_response.model_used,
                "latency_ms": llm_response.latency_ms,
                "safety_status": safety_decision.status,
                "safety_reasons": safety_decision.reasons,
                "action": safety_decision.action.to_dict() if safety_decision.action else None,
                "zone_temps": state.zone_temps_c,
                "zone_rh": state.zone_relative_humidity_pct,
                "zone_pmv": state.zone_pmv,
                "hvac_power_kw": state.hvac_power_kw,
                "outdoor_temp_c": state.outdoor_temp_c,
                "occupancy": state.occupancy,
            })

            print(
                f"  Step {row_idx:3d}: "
                f"temp={min(state.zone_temps_c.values()):.1f}-{max(state.zone_temps_c.values()):.1f} C | "
                f"power={state.hvac_power_kw or 0:.1f} kW | "
                f"LLM={llm_response.source} | safety={safety_decision.status}"
            )

        # Write manifest and metrics
        self._write_manifest(run_dir, mode, sim_dir)
        self._write_metrics(run_dir, mode)
        return self.events

    def compare(self) -> dict:
        """Run baseline and AI on identical setup, return comparison metrics."""
        print("=== Running baseline ===")
        baseline_events = self.run(mode="baseline")

        print("\n=== Running AI controller ===")
        ai_events = self.run(mode="ai")

        return self._compute_comparison(baseline_events, ai_events)

    # ── Private ─────────────────────────────────────────────────────────

    def _run_or_reuse_simulation(self, working_idf: Path | None, run_dir: Path, mode: str) -> Path:
        """Run EnergyPlus simulation or reuse an existing output directory.

        Generates synthetic time-series data from EPW if existing output only
        contains sizing-day constants (no real time variation).
        """
        # Check for existing simulation output in priority order:
        # 1. runtime-spike (the proven callback-spike output)
        # 2. sim-* directories
        if self.config.paths.data_output.exists():
            spike_dir = self.config.paths.data_output / "runtime-spike"
            if spike_dir.exists() and (spike_dir / "epluszsz.csv").exists():
                zsz = spike_dir / "epluszsz.csv"
                if self._has_time_variation(zsz):
                    print(f"  [Controller] Reusing runtime-spike output: {spike_dir}")
                    return spike_dir
                else:
                    print(f"  [Controller] runtime-spike has no time variation, generating synthetic data")

            existing = list(self.config.paths.data_output.glob("sim-*"))
            if existing:
                latest = sorted(existing)[-1]
                zsz = latest / "epluszsz.csv"
                if zsz.exists() and self._has_time_variation(zsz):
                    print(f"  [Controller] Reusing existing simulation: {latest}")
                    return latest

        # Generate synthetic data from EPW
        sim_dir = self.config.paths.data_output / f"sim-{mode}"
        sim_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [Controller] Generating synthetic simulation data from EPW: {self.epw_path.name}")
        generate_synthetic_run(sim_dir, self.epw_path)
        return sim_dir

    @staticmethod
    def _has_time_variation(zsz_path: Path) -> bool:
        """Check if epluszsz.csv has actual time-varying data (not just sizing constants)."""
        try:
            with open(zsz_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [r for r in reader if r.get("Time", "").replace(":", "").isdigit()]
                if len(rows) < 2:
                    return False
                # Check if heating zone temps vary across rows
                temp_key = "SPACE1-1:CHICAGO_IL_USA ANNUAL HEATING 99% DESIGN CONDITIONS DB:Heating Zone Temperature [C]"
                vals = [float(r.get(temp_key, 0)) for r in rows[:10] if r.get(temp_key)]
                if len(vals) < 2:
                    return False
                return max(vals) - min(vals) > 0.01
        except Exception:
            return False

    def _read_all_timesteps(self, sim_dir: Path) -> list[dict]:
        """Read all timestep rows from epluszsz.csv and initialize the state reader's column map."""
        zsz_path = sim_dir / "epluszsz.csv"
        if not zsz_path.exists():
            return []
        # Initialize state reader column map from this CSV
        self.state_reader._epluszsz_col_map = self.state_reader._build_zone_column_map(zsz_path)
        self.state_reader._zsz_path = zsz_path
        rows = []
        with open(zsz_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip non-time rows (e.g. "Peak" summary rows)
                if not row.get("Time", "").replace(":", "").isdigit():
                    continue
                rows.append(row)
        return rows

    # ── Manifest & Metrics ──────────────────────────────────────────────

    def _write_manifest(self, run_dir: Path, mode: str, sim_dir: Path) -> None:
        """Write run manifest."""
        try:
            import energyplus  # type: ignore[import-untyped]
            ep_version = getattr(energyplus, "__version__", "26.1")
        except ImportError:
            ep_version = "26.1 (from runtime spike)"

        manifest = {
            "run_id": run_dir.name,
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "energyplus_version": ep_version,
            "idf_path": str(self.idf_path),
            "epw_path": str(self.epw_path),
            "sim_dir": str(sim_dir),
            "config": {
                "timestep_minutes": self.config.control.timestep_minutes,
                "total_steps": self.config.total_steps,
                "safety_limits": self.config.safety.__dict__,
            },
            "llm": {
                "mode": self.config.llm.mode,
                "primary_model": self.config.llm.primary_model,
                "fallback_model": self.config.llm.fallback_model,
            },
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _write_metrics(self, run_dir: Path, mode: str) -> None:
        """Aggregate metrics from audit log."""
        audit_path = run_dir / "audit.jsonl"
        if not audit_path.exists():
            return

        events = []
        with open(audit_path, encoding="utf-8") as f:
            for line in f:
                events.append(json.loads(line))

        total_steps = len(events)
        llm_primary = sum(1 for e in events if e.get("llm_source") == "primary")
        llm_fallback = sum(1 for e in events if e.get("llm_source") == "fallback")
        llm_det = sum(1 for e in events if e.get("llm_source") == "deterministic")
        safety_accepted = sum(1 for e in events if e.get("safety_status") == "accepted")
        safety_clamped = sum(1 for e in events if e.get("safety_status") == "clamped")

        # Comfort compliance
        occupied_intervals = 0
        comfort_compliant = 0
        for e in events:
            for zone, occ in e.get("state_before", {}).get("occupancy", {}).items():
                if occ > 0:
                    occupied_intervals += 1
                    pmv = e.get("state_before", {}).get("zone_pmv", {}).get(zone)
                    if pmv is not None and -0.5 <= pmv <= 0.5:
                        comfort_compliant += 1

        comfort_pct = (100.0 * comfort_compliant / occupied_intervals) if occupied_intervals else 0.0

        # Energy estimate
        total_energy = 0.0
        for e in events:
            power = e.get("hvac_power_kw", 0.0) or 0.0
            total_energy += power * (self.config.control.timestep_minutes / 60.0)

        metrics = {
            "mode": mode,
            "total_steps": total_steps,
            "llm_calls_primary": llm_primary,
            "llm_calls_fallback": llm_fallback,
            "llm_calls_deterministic": llm_det,
            "safety_accepted": safety_accepted,
            "safety_clamped": safety_clamped,
            "comfort_compliance_pct": round(comfort_pct, 2),
            "occupied_intervals": occupied_intervals,
            "estimated_energy_kwh": round(total_energy, 3),
        }

        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    @staticmethod
    def _compute_comparison(baseline_events: list[ControlEvent], ai_events: list[ControlEvent]) -> dict:
        """Compute comparison metrics between baseline and AI runs."""

        def energy_of(events):
            total = 0.0
            peak = 0.0
            for e in events:
                if e.state_after:
                    power = e.state_after.get("hvac_power_kw", 0.0) or 0.0
                else:
                    power = e.state_before.get("hvac_power_kw", 0.0) or 0.0
                total += power * 0.25  # 15 min = 0.25 h
                peak = max(peak, power)
            return total, peak

        b_kwh, b_peak = energy_of(baseline_events)
        a_kwh, a_peak = energy_of(ai_events)

        from .metrics import compare

        cm = compare(b_kwh, a_kwh, b_peak, a_peak)

        return {
            "baseline_kwh": round(cm.baseline_kwh, 3),
            "ai_kwh": round(cm.ai_kwh, 3),
            "baseline_peak_kw": round(cm.baseline_peak_kw, 3),
            "ai_peak_kw": round(cm.ai_peak_kw, 3),
            "energy_savings_pct": round(cm.energy_savings_pct, 2),
            "peak_reduction_pct": round(cm.peak_reduction_pct, 2),
        }
