"""Simple Model Predictive Control (MPC) optimizer for Eco-Loop.

Uses a 3R-2C thermal network model to predict zone temperatures
and optimize setpoints over a 4-hour rolling horizon.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .schemas import ControlAction, ZoneSetpoints
from .config import SafetyLimits


@dataclass(frozen=True)
class ThermalModelParams:
    """Parameters for a simplified 3R-2C thermal network model.

    R_wall: thermal resistance of wall (K/W equivalent, normalized)
    R_window: thermal resistance of window (lower = more heat transfer)
    C_zone: thermal capacitance of zone air (J/K equivalent, normalized)
    C_mass: thermal capacitance of building mass (higher = more thermal lag)
    """
    R_wall: float = 2.0       # hours (thermal time constant contribution)
    R_window: float = 0.8     # hours
    C_zone: float = 1.5       # hours (effective time constant)
    C_mass: float = 4.0       # hours (building mass time constant)
    hvac_efficiency: float = 3.5  # COP for cooling, efficiency for heating


# Default thermal model parameters for 5ZoneAirCooled
ZONE_THERMAL_PARAMS: dict[str, ThermalModelParams] = {
    "SPACE1-1": ThermalModelParams(R_wall=2.5, C_zone=2.0, C_mass=5.0),   # core: heavy mass
    "SPACE2-1": ThermalModelParams(R_wall=1.5, C_zone=1.2, C_mass=3.0),   # perimeter: less mass
    "SPACE3-1": ThermalModelParams(R_wall=2.8, C_zone=1.8, C_mass=4.5),   # interior: moderate
    "SPACE4-1": ThermalModelParams(R_wall=1.3, C_zone=1.0, C_mass=2.8),   # perimeter: light
    "SPACE5-1": ThermalModelParams(R_wall=1.0, C_zone=0.8, C_mass=2.5),   # top floor: lightest
}


class SimpleRCThermalModel:
    """Simplified 3R-2C thermal network model for zone temperature prediction.

    Predicts zone air temperature after dt hours given outdoor temp
    and HVAC setpoints. Uses a first-order exponential decay model
    with building mass as thermal buffer.
    """

    def __init__(self, params: ThermalModelParams | None = None):
        self.params = params or ThermalModelParams()

    def predict(
        self,
        t_current: float,
        t_outdoor: float,
        heating_sp: float,
        cooling_sp: float,
        dt_hours: float = 0.25,  # 15 minutes
        internal_gains_kw: float = 0.5,
    ) -> float:
        """Predict zone air temperature after dt hours.

        Args:
            t_current: current zone air temperature (C)
            t_outdoor: outdoor dry-bulb temperature (C)
            heating_sp: heating setpoint (C)
            cooling_sp: cooling setpoint (C)
            dt_hours: prediction horizon in hours
            internal_gains_kw: internal heat gains (lighting, equipment, people)

        Returns:
            Predicted zone air temperature (C)
        """
        tau = self.params.C_zone * self.params.R_wall
        decay = math.exp(-dt_hours / tau)

        # Natural drift toward outdoor temp + internal gains effect
        t_gain = internal_gains_kw * self.params.R_wall * 0.5  # simplified gain effect
        t_equilibrium = t_outdoor + t_gain

        # Natural evolution without HVAC
        t_natural = t_equilibrium + (t_current - t_equilibrium) * decay

        # HVAC effect: if temp would exceed setpoints, HVAC pulls it back
        if t_natural > cooling_sp:
            # Cooling mode: HVAC pulls temp toward cooling setpoint
            hvac_pull = (t_natural - cooling_sp) * (1 - decay) * self.params.hvac_efficiency / self.params.C_zone
            t_predicted = t_natural - hvac_pull
            t_predicted = max(t_predicted, cooling_sp - 0.5)  # don't over-cool
        elif t_natural < heating_sp:
            # Heating mode: HVAC pulls temp toward heating setpoint
            hvac_pull = (heating_sp - t_natural) * (1 - decay) * self.params.hvac_efficiency / self.params.C_zone
            t_predicted = t_natural + hvac_pull
            t_predicted = min(t_predicted, heating_sp + 0.5)  # don't over-heat
        else:
            # In deadband: no HVAC action needed
            t_predicted = t_natural

        return t_predicted

    def estimate_power_kw(
        self,
        t_current: float,
        t_outdoor: float,
        heating_sp: float,
        cooling_sp: float,
    ) -> float:
        """Estimate HVAC power consumption for maintaining setpoints.

        Returns estimated power in kW.
        """
        if t_current > cooling_sp:
            # Cooling load
            delta = t_current - cooling_sp
            return delta * self.params.C_zone / (self.params.R_wall * self.params.hvac_efficiency)
        elif t_current < heating_sp:
            # Heating load
            delta = heating_sp - t_current
            return delta * self.params.C_zone / (self.params.R_wall * 0.95)  # heating efficiency ~95%
        return 0.0


class MPCOptimizer:
    """Simple MPC optimizer using brute-force setpoint enumeration.

    Optimizes over a 4-hour horizon with 15-minute steps. Enumerates
    setpoint candidates within safety limits and scores each on
    energy cost + comfort penalty.
    """

    def __init__(
        self,
        safety_limits: SafetyLimits | None = None,
        horizon_hours: float = 4.0,
        step_hours: float = 0.25,
        setpoint_resolution: float = 0.5,
    ):
        self.limits = safety_limits or SafetyLimits()
        self.horizon_hours = horizon_hours
        self.step_hours = step_hours
        self.resolution = setpoint_resolution
        self.models: dict[str, SimpleRCThermalModel] = {}

    def _get_model(self, zone: str) -> SimpleRCThermalModel:
        """Get or create thermal model for a zone."""
        if zone not in self.models:
            params = ZONE_THERMAL_PARAMS.get(zone, ThermalModelParams())
            self.models[zone] = SimpleRCThermalModel(params)
        return self.models[zone]

    def optimize(
        self,
        zone_temps: dict[str, float],
        outdoor_temp: float,
        forecast_temps: list[float],
        occupancy: dict[str, float],
        current_heating_sp: dict[str, float],
        current_cooling_sp: dict[str, float],
    ) -> dict[str, dict]:
        """Find optimal setpoints for each zone over the MPC horizon.

        Returns dict mapping zone_name -> {"heating_c": float, "cooling_c": float, "predicted_temp": float, "score": float}
        """
        results = {}
        for zone, t_current in zone_temps.items():
            model = self._get_model(zone)
            is_occupied = occupancy.get(zone, 0) > 0

            best_score = float("inf")
            best_h = current_heating_sp.get(zone, 20.0)
            best_c = current_cooling_sp.get(zone, 26.0)
            best_pred = t_current

            # Generate candidate setpoints
            h_candidates = self._range(
                max(self.limits.heating_min_c, best_h - self.limits.max_delta_per_interval_c),
                min(self.limits.heating_max_c, best_h + self.limits.max_delta_per_interval_c),
            )
            c_candidates = self._range(
                max(self.limits.cooling_min_c, best_c - self.limits.max_delta_per_interval_c),
                min(self.limits.cooling_max_c, best_c + self.limits.max_delta_per_interval_c),
            )

            for h_sp in h_candidates:
                for c_sp in c_candidates:
                    # Enforce deadband
                    if c_sp - h_sp < self.limits.minimum_deadband_c:
                        continue

                    # Simulate forward over horizon
                    score, pred_temp = self._score_trajectory(
                        model, t_current, outdoor_temp, forecast_temps,
                        h_sp, c_sp, is_occupied,
                    )

                    if score < best_score:
                        best_score = score
                        best_h = h_sp
                        best_c = c_sp
                        best_pred = pred_temp

            results[zone] = {
                "heating_c": best_h,
                "cooling_c": best_c,
                "predicted_temp_c": round(best_pred, 2),
                "score": round(best_score, 4),
            }

        return results

    def _score_trajectory(
        self,
        model: SimpleRCThermalModel,
        t_current: float,
        t_outdoor: float,
        forecast_temps: list[float],
        heating_sp: float,
        cooling_sp: float,
        is_occupied: bool,
    ) -> tuple[float, float]:
        """Score a setpoint trajectory over the horizon.

        Returns (total_score, final_predicted_temp).
        Lower score is better.
        """
        steps = int(self.horizon_hours / self.step_hours)
        t = t_current
        total_energy = 0.0
        total_comfort_penalty = 0.0

        for i in range(steps):
            # Outdoor temp from forecast or constant
            t_out = forecast_temps[i] if i < len(forecast_temps) else t_outdoor

            # Predict temperature
            t = model.predict(t, t_out, heating_sp, cooling_sp, self.step_hours)

            # Energy cost
            power = model.estimate_power_kw(t, t_out, heating_sp, cooling_sp)
            total_energy += power * self.step_hours

            # Comfort penalty (only for occupied)
            if is_occupied:
                comfort_mid = (heating_sp + cooling_sp) / 2.0
                deviation = abs(t - comfort_mid)
                if deviation > 2.0:
                    total_comfort_penalty += deviation * 5.0  # heavy penalty
                elif deviation > 1.0:
                    total_comfort_penalty += deviation * 2.0

        # Combined score: energy + comfort penalty
        comfort_weight = 3.0 if is_occupied else 0.5
        score = total_energy + comfort_weight * total_comfort_penalty

        return score, t

    def _range(self, lo: float, hi: float) -> list[float]:
        """Generate float range with given resolution."""
        vals = []
        v = lo
        while v <= hi + 1e-9:
            vals.append(round(v, 1))
            v += self.resolution
        return vals if vals else [round(lo, 1)]

    def to_control_action(
        self,
        results: dict[str, dict],
        mode: str = "normal",
    ) -> ControlAction:
        """Convert MPC results to a ControlAction."""
        zone_setpoints = {
            zone: ZoneSetpoints(
                heating_c=info["heating_c"],
                cooling_c=info["cooling_c"],
            )
            for zone, info in results.items()
        }
        return ControlAction(
            action_id=f"mpc-opt",
            zone_setpoints=zone_setpoints,
            mode=mode,
            rationale="MPC-optimized setpoints over 4h horizon",
            confidence=0.85,
        )


def evaluate_mpc(
    zone_temps: dict[str, float],
    outdoor_temp: float,
    forecast_temps: list[float],
    occupancy: dict[str, float],
    current_heating_sp: dict[str, float],
    current_cooling_sp: dict[str, float],
    safety_limits: SafetyLimits | None = None,
) -> dict:
    """Run MPC optimization and return recommendation as evaluator signal."""
    optimizer = MPCOptimizer(safety_limits=safety_limits)
    results = optimizer.optimize(
        zone_temps, outdoor_temp, forecast_temps,
        occupancy, current_heating_sp, current_cooling_sp,
    )

    return {
        "status": "optimized",
        "horizon_hours": optimizer.horizon_hours,
        "recommendations": {
            zone: {
                "heating_c": info["heating_c"],
                "cooling_c": info["cooling_c"],
                "predicted_temp_c": info["predicted_temp_c"],
            }
            for zone, info in results.items()
        },
    }
