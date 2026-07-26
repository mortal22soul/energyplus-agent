"""EnergyPlus Python API runtime adapter and callback-actuation spike."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RuntimeSpikeResult:
    exit_code: int
    observations: list[float] = field(default_factory=list)
    actuator_values: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def energyplus_home() -> Path:
    home = Path(os.environ.get("ECOLOOP_ENERGYPLUS_HOME", r"D:\EnergyPlus"))
    if not (home / "energyplusapi.dll").exists():
        raise FileNotFoundError(f"EnergyPlus API not found under {home}; set ECOLOOP_ENERGYPLUS_HOME")
    return home


def _api():
    home = energyplus_home()
    if str(home) not in sys.path:
        sys.path.insert(0, str(home))
    from pyenergyplus.api import EnergyPlusAPI
    return EnergyPlusAPI()


def run_schedule_actuation_spike(idf_path: str | Path, epw_path: str | Path, output_dir: str | Path) -> RuntimeSpikeResult:
    """Read SPACE1-1 temperature and override Htg-SetP-Sch in a single E+ process.

    This deliberately targets the installed 5ZoneAirCooled example. It is the Phase 0.5
    proof, not the general production controller; later model mappings belong in config.
    """
    api = _api()
    state = api.state_manager.new_state()
    result = RuntimeSpikeResult(exit_code=1)
    handles: dict[str, int] = {}

    def callback(sim_state):
        if not api.exchange.api_data_fully_ready(sim_state):
            return
        if not handles:
            handles["temp"] = api.exchange.get_variable_handle(sim_state, "Zone Air Temperature", "SPACE1-1")
            handles["heat"] = api.exchange.get_actuator_handle(sim_state, "Schedule:Compact", "Schedule Value", "Htg-SetP-Sch")
            if min(handles.values()) < 0:
                result.errors.append(f"invalid EnergyPlus handles: {handles}")
                api.runtime.stop_simulation(sim_state)
                return
        temp = api.exchange.get_variable_value(sim_state, handles["temp"])
        target = 21.5 if len(result.observations) % 2 == 0 else 22.0
        api.exchange.set_actuator_value(sim_state, handles["heat"], target)
        result.observations.append(temp)
        result.actuator_values.append(target)
        if len(result.observations) >= 8:
            api.runtime.stop_simulation(sim_state)

    api.runtime.callback_after_predictor_before_hvac_managers(state, callback)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result.exit_code = api.runtime.run_energyplus(state, ["-d", str(output_dir), "-w", str(epw_path), str(idf_path)])
    api.state_manager.delete_state(state)
    return result
