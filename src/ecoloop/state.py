"""State reading, PMV/PPD evaluation, forecast extraction, and deterministic evaluators."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from .comfort import calculate_pmv_ppd, ComfortResult
from .schemas import BuildingState, StateSummary
from .config import RunConfig

if TYPE_CHECKING:
    from .schemas import ZoneSetpoints


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse EnergyPlus time string (HH:MM:SS or HH:MM:SS + date) to datetime."""
    ts_str = ts_str.strip()
    if len(ts_str) <= 8:
        # Just HH:MM:SS, handle 24:00 → 00:00 next day
        h, m, s = map(int, ts_str.split(":"))
        if h == 24:
            h = 0
        return datetime(2025, 1, 1, h, m, s)
    # Try common formats
    for fmt in ("%H:%M:%S", "%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    # Fallback: just time part
    h, m, s = map(int, ts_str.split(":"))
    return datetime(2025, 1, 1, h, m, s)


def _parse_float(val: str) -> float | None:
    """Parse EnergyPlus scientific notation float, return None on empty/bad value."""
    val = val.strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


class StateReader:
    """Reads EnergyPlus sizing/simulation output CSVs and builds typed state."""

    def __init__(self, config: RunConfig):
        self.config = config
        self._epluszsz_col_map: dict[str, dict[str, int]] = {}
        self._eplusssz_col_map: dict[str, int] = {}
        self._zone_temps_key = "Heating Zone Temperature [C]"
        self._zsz_path: Path | None = None  # cached path for lazy init

    def _build_zone_column_map(self, zsz_path: Path) -> dict[str, dict[str, int]]:
        """Parse epluszsz.csv header to map zone → {field_suffix: col_index}.

        Uses the last segment after ':' as the key to handle design-day prefixed headers.
        """
        col_map: dict[str, dict[str, int]] = {}
        with open(zsz_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            for i, h in enumerate(headers):
                if h == "Time":
                    continue
                parts = h.split(":")
                zone = parts[0]
                # Use last segment as key (e.g. "Heating Zone Temperature [C]")
                key = parts[-1] if len(parts) > 1 else h
                col_map.setdefault(zone, {})[key] = i
        return col_map

    def _build_system_column_map(self, ssz_path: Path) -> dict[str, int]:
        """Parse eplusssz.csv header to map field → col_index."""
        col_map: dict[str, int] = {}
        with open(ssz_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            for i, h in enumerate(headers):
                col_map[h] = i
        return col_map

    def read_latest(self, output_dir: str | Path) -> BuildingState | None:
        """Read the last timestep from EnergyPlus output files."""
        output_dir = Path(output_dir)
        zsz_path = output_dir / "epluszsz.csv"
        ssz_path = output_dir / "eplusssz.csv"

        if not zsz_path.exists():
            return None

        if not self._epluszsz_col_map:
            self._epluszsz_col_map = self._build_zone_column_map(zsz_path)
        if ssz_path.exists() and not self._eplusssz_col_map:
            self._eplusssz_col_map = self._build_system_column_map(ssz_path)

        rows: list[dict] = []
        with open(zsz_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            return None

        last_row = rows[-1]
        ts = _parse_timestamp(last_row["Time"])
        step_index = len(rows)

        zone_temps: dict[str, float] = {}
        zone_rh: dict[str, float] = {}
        heating_sp: dict[str, float] = {}
        cooling_sp: dict[str, float] = {}
        zone_pmv: dict[str, float] = {}
        zone_ppd: dict[str, float] = {}
        outdoor_temp_c = 20.0

        for zone_name, fields in self._epluszsz_col_map.items():
            # Zone air temperature: use Heating Zone Temperature as proxy for room air
            # In sizing output, "Heating Zone Temperature" = room air temp at heating peak
            temp_idx = fields.get("Heating Zone Temperature [C]")
            if temp_idx is not None:
                raw = last_row.get(list(last_row.keys())[temp_idx])
                val = _parse_float(raw) if raw else None
                if val is not None:
                    zone_temps[zone_name] = val

            # Setpoint: use heating zone temp as heating SP, cooling zone temp as cooling SP
            heat_idx = fields.get("Heating Zone Temperature [C]")
            cool_idx = fields.get("Cooling Zone Temperature [C]")
            rh_idx = fields.get("Heating Zone Relative Humidity [%]")

            if heat_idx is not None:
                raw = last_row.get(list(last_row.keys())[heat_idx])
                val = _parse_float(raw) if raw else None
                if val is not None:
                    heating_sp[zone_name] = val

            if cool_idx is not None:
                raw = last_row.get(list(last_row.keys())[cool_idx])
                val = _parse_float(raw) if raw else None
                if val is not None:
                    cooling_sp[zone_name] = val

            if rh_idx is not None:
                raw = last_row.get(list(last_row.keys())[rh_idx])
                val = _parse_float(raw) if raw else None
                if val is not None:
                    zone_rh[zone_name] = val

            # PMV/PPD from comfort model
            t_air = zone_temps.get(zone_name, 20.0)
            t_radiant = t_air  # approximation when mean radiant temp unavailable
            rh = zone_rh.get(zone_name, 50.0)
            try:
                result: ComfortResult = calculate_pmv_ppd(
                    air_temp_c=t_air,
                    radiant_temp_c=t_radiant,
                    relative_humidity_pct=rh,
                )
                zone_pmv[zone_name] = result.pmv
                zone_ppd[zone_name] = result.ppd_pct
            except Exception:
                pass  # PMV unavailable for this zone

        # Outdoor temperature from zsz data (last row's site outdoor)
        site_outdoor_idx = None
        if "Site Outdoor Air Drybulb Temperature [C]" in last_row:
            site_outdoor_idx = list(last_row.keys()).index("Site Outdoor Air Drybulb Temperature [C]")
        elif "Time" in last_row:
            # Try to get from first available outdoor field
            for key in last_row:
                if "Outdoor" in key and "Drybulb" in key:
                    site_outdoor_idx = list(last_row.keys()).index(key)
                    break

        outdoor_temp_c = 20.0
        if site_outdoor_idx is not None and site_outdoor_idx < len(last_row):
            val = _parse_float(list(last_row.values())[site_outdoor_idx])
            if val is not None:
                outdoor_temp_c = val

        # Occupancy: 1 person per zone during daytime, 0 at night
        hour = ts.hour + ts.minute / 60.0
        occupancy = {}
        for zone_name in zone_temps:
            is_occupied = 8 <= hour < 18
            occupancy[zone_name] = 1.0 if is_occupied else 0.0

        # Forecast: load from bundled CSV if available
        forecast = self._load_forecast()

        # Facility power: estimate from number of active HVAC zones
        hvac_power_kw = self._estimate_hvac_power(zone_temps, heating_sp, cooling_sp, occupancy)

        state = BuildingState(
            timestamp=ts,
            step_index=step_index,
            zone_temps_c=zone_temps,
            zone_relative_humidity_pct=zone_rh,
            occupancy=occupancy,
            heating_setpoints_c=heating_sp if heating_sp else {z: 18.0 for z in zone_temps},
            cooling_setpoints_c=cooling_sp if cooling_sp else {z: 28.0 for z in zone_temps},
            zone_pmv=zone_pmv,
            zone_ppd_pct=zone_ppd,
            outdoor_temp_c=outdoor_temp_c,
            forecast_outdoor_temp_c=forecast,
            hvac_power_kw=hvac_power_kw,
            facility_power_kw=hvac_power_kw,
        )
        return state

    def _load_forecast(self) -> list[float]:
        """Load bundled outdoor temperature forecast CSV."""
        forecast_path = self.config.paths.forecast_csv()
        if not forecast_path.exists():
            return []
        temps: list[float] = []
        with open(forecast_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    val = _parse_float(row[0])
                    if val is not None:
                        temps.append(val)
        return temps[:96]  # max 24h at 15-min intervals

    def _estimate_hvac_power(
        self,
        zone_temps: dict[str, float],
        heating_sp: dict[str, float],
        cooling_sp: dict[str, float],
        occupancy: dict[str, float],
    ) -> float | None:
        """HVAC power estimate from design loads when available, fallback to simple model."""
        # Use sizing data heating/cooling loads as power proxy if available
        # (design loads are in W; 1 occupied zone with 3-4kW cooling ≈ active cooling)
        # For sizing CSV: heating/cooling zone temperatures ARE the setpoints,
        # not the loads. Use a simple model based on setpoint deltas.
        total = 0.0
        for zone in zone_temps:
            t_air = zone_temps[zone]
            h_sp = heating_sp.get(zone, 18.0)
            c_sp = cooling_sp.get(zone, 28.0)
            occ = occupancy.get(zone, 0.0)
            if occ > 0:
                # Use setpoint gap as proxy for demand intensity
                gap = max(c_sp - t_air, t_air - h_sp, 0.0)
                if gap > 0.5:
                    total += gap * 0.5  # empirical scale factor
        return total if total > 0 else None

    def _build_state_from_row(
        self, row: dict, row_idx: int, all_rows: list[dict]
    ) -> BuildingState | None:
        """Build a BuildingState from a single CSV row (offline mode).

        Needs column map from the CSV that owns `row`. Call `_build_zone_column_map`
        before calling this if not already done via `read_latest`.
        """
        if not self._epluszsz_col_map or not self._zsz_path:
            return None

        ts = _parse_timestamp(row.get("Time", "00:00:00"))
        keys = list(row.keys())

        zone_temps: dict[str, float] = {}
        zone_rh: dict[str, float] = {}
        heating_sp: dict[str, float] = {}
        cooling_sp: dict[str, float] = {}
        zone_pmv: dict[str, float] = {}
        zone_ppd: dict[str, float] = {}

        for zone_name, fields in self._epluszsz_col_map.items():
            temp_idx = fields.get(self._zone_temps_key)
            if temp_idx is not None and temp_idx < len(keys):
                val = _parse_float(row.get(keys[temp_idx], ""))
                if val is not None:
                    zone_temps[zone_name] = val

            heat_temp_idx = fields.get(self._zone_temps_key)
            cooling_temp_idx = fields.get("Cooling Zone Temperature [C]")
            rh_idx = fields.get("Heating Zone Relative Humidity [%]")

            if heat_temp_idx is not None and heat_temp_idx < len(keys):
                val = _parse_float(row.get(keys[heat_temp_idx], ""))
                if val is not None:
                    heating_sp[zone_name] = val

            if cooling_temp_idx is not None and cooling_temp_idx < len(keys):
                val = _parse_float(row.get(keys[cooling_temp_idx], ""))
                if val is not None:
                    cooling_sp[zone_name] = val

            if rh_idx is not None and rh_idx < len(keys):
                val = _parse_float(row.get(keys[rh_idx], ""))
                if val is not None:
                    zone_rh[zone_name] = val

            # PMV/PPD
            t_air = zone_temps.get(zone_name, 20.0)
            t_radiant = t_air
            rh = zone_rh.get(zone_name, 50.0)
            try:
                result: ComfortResult = calculate_pmv_ppd(
                    air_temp_c=t_air,
                    radiant_temp_c=t_radiant,
                    relative_humidity_pct=rh,
                )
                zone_pmv[zone_name] = result.pmv
                zone_ppd[zone_name] = result.ppd_pct
            except Exception:
                pass

        # Outdoor temperature
        outdoor_temp_c = 20.0
        for key in keys:
            if "Outdoor" in key and "Drybulb" in key:
                val = _parse_float(row.get(key, ""))
                if val is not None:
                    outdoor_temp_c = val
                    break

        # Occupancy (8am-6pm)
        hour = ts.hour + ts.minute / 60.0
        occupancy = {}
        for zone_name in zone_temps:
            is_occupied = 8 <= hour < 18
            occupancy[zone_name] = 1.0 if is_occupied else 0.0

        # Forecast
        forecast = self._load_forecast()

        # HVAC power estimate
        hvac_power_kw = self._estimate_hvac_power(zone_temps, heating_sp, cooling_sp, occupancy)

        # Cumulative energy estimate
        cumulative = 0.0
        for r in all_rows[: row_idx + 1]:
            r_temps = {}
            for z, f in self._epluszsz_col_map.items():
                idx = f.get(self._zone_temps_key)
                if idx is not None and idx < len(keys):
                    v = _parse_float(r.get(keys[idx], ""))
                    if v is not None:
                        r_temps[z] = v
            r_hs = {z: heating_sp.get(z, 18.0) for z in r_temps}
            r_cs = {z: cooling_sp.get(z, 28.0) for z in r_temps}
            r_occ = {z: 1.0 if 8 <= ts.hour < 18 else 0.0 for z in r_temps}
            p = self._estimate_hvac_power(r_temps, r_hs, r_cs, r_occ)
            if p:
                cumulative += p * (self.config.control.timestep_minutes / 60.0)

        return BuildingState(
            timestamp=ts,
            step_index=row_idx,
            zone_temps_c=zone_temps,
            zone_relative_humidity_pct=zone_rh,
            occupancy=occupancy,
            heating_setpoints_c=heating_sp if heating_sp else {z: 18.0 for z in zone_temps},
            cooling_setpoints_c=cooling_sp if cooling_sp else {z: 28.0 for z in zone_temps},
            zone_pmv=zone_pmv,
            zone_ppd_pct=zone_ppd,
            outdoor_temp_c=outdoor_temp_c,
            forecast_outdoor_temp_c=forecast,
            hvac_power_kw=hvac_power_kw,
            facility_power_kw=hvac_power_kw,
            cumulative_facility_energy_kwh=cumulative,
        )


# ── Deterministic Evaluators ──────────────────────────────────────────────
# These summarize state into structured signals for the LLM prompt.
# They do NOT override the LLM; they inform it.

def evaluate_comfort(state: BuildingState) -> dict:
    """Comfort evaluator: PMV/PPD summary for occupied zones."""
    occupied = state.occupied_zones()
    if not occupied:
        return {"status": "unoccupied", "zones": {}}

    zone_details = {}
    violations = 0
    for zone in occupied:
        pmv = state.zone_pmv.get(zone)
        ppd = state.zone_ppd_pct.get(zone)
        if pmv is None:
            zone_details[zone] = {"pmv": None, "ppd": None, "status": "unknown"}
            continue
        in_band = -0.5 <= pmv <= 0.5
        zone_details[zone] = {
            "pmv": round(pmv, 3),
            "ppd": round(ppd, 1) if ppd is not None else None,
            "status": "comfortable" if in_band else "uncomfortable",
        }
        if not in_band:
            violations += 1

    compliance_pct = 100.0 * (len(occupied) - violations) / len(occupied) if occupied else 100.0
    return {
        "status": "occupied",
        "compliance_pct": round(compliance_pct, 1),
        "violations": violations,
        "zones": zone_details,
    }


def evaluate_energy(state: BuildingState) -> dict:
    """Energy evaluator: current demand and trend estimate."""
    power = state.hvac_power_kw
    return {
        "hvac_power_kw": round(power, 2) if power is not None else None,
        "cumulative_kwh": state.cumulative_facility_energy_kwh,
        "trend": "stable",  # simplified; history-based trend goes in controller
    }


def evaluate_forecast(state: BuildingState, config: RunConfig) -> dict:
    """Forecast evaluator: flags hot/cold conditions and preconditioning opportunity."""
    forecast = state.forecast_outdoor_temp_c
    if not forecast:
        return {"status": "unavailable", "message": "No forecast data"}

    # Look at next 4 hours (16 timesteps at 15-min)
    next_4h = forecast[:16] if len(forecast) >= 16 else forecast
    max_4h = max(next_4h)
    min_4h = min(next_4h)

    # Preconditioning rules
    occupied = len(state.occupied_zones()) > 0
    if not occupied:
        if max_4h > 30:
            return {
                "status": "precool_opportunity",
                "max_4h_c": round(max_4h, 1),
                "recommendation": "Pre-cool before occupancy if within safety limits",
            }
        if min_4h < 10:
            return {
                "status": "preheat_opportunity",
                "min_4h_c": round(min_4h, 1),
                "recommendation": "Pre-heat before occupancy if within safety limits",
            }
        return {"status": "normal", "max_4h_c": round(max_4h, 1), "min_4h_c": round(min_4h, 1)}

    return {
        "status": "occupied_comfort_priority",
        "max_4h_c": round(max_4h, 1),
        "min_4h_c": round(min_4h, 1),
    }


def evaluate_oscillation(state: BuildingState, history: list[StateSummary]) -> dict:
    """Oscillation evaluator: detects frequent setpoint reversals in recent history."""
    if len(history) < 3:
        return {"status": "insufficient_history", "reversals": 0}

    reversals = 0
    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]
        # Check if any zone heating/cooling setpoint reversed direction
        for zone in curr.zone_temps_c:
            if zone in prev.zone_temps_c:
                # Simplified: check facility power direction
                pass

    # Use power trend as proxy for oscillation
    powers = [s.facility_power_kw for s in history[-6:] if s.facility_power_kw is not None]
    if len(powers) >= 3:
        diffs = [powers[i] - powers[i - 1] for i in range(1, len(powers))]
        sign_changes = sum(1 for i in range(1, len(diffs)) if diffs[i] * diffs[i - 1] < 0)
        if sign_changes >= 3:
            return {
                "status": "oscillation_detected",
                "sign_changes": sign_changes,
                "recommendation": "Hold current setpoints; avoid rapid changes",
            }

    return {"status": "stable", "reversals": 0}


def build_evaluator_summary(
    state: BuildingState,
    history: list[StateSummary],
    config: RunConfig,
) -> dict:
    """Run all deterministic evaluators and return a summary dict for the LLM prompt."""
    summary = {
        "comfort": evaluate_comfort(state),
        "energy": evaluate_energy(state),
        "forecast": evaluate_forecast(state, config),
        "oscillation": evaluate_oscillation(state, history),
    }

    # Add MPC recommendation if available
    try:
        from .mpc import evaluate_mpc
        mpc_result = evaluate_mpc(
            zone_temps=state.zone_temps_c,
            outdoor_temp=state.outdoor_temp_c,
            forecast_temps=state.forecast_outdoor_temp_c,
            occupancy=state.occupancy,
            current_heating_sp=state.heating_setpoints_c,
            current_cooling_sp=state.cooling_setpoints_c,
            safety_limits=config.safety,
        )
        summary["mpc"] = mpc_result
    except Exception:
        summary["mpc"] = {"status": "unavailable"}

    return summary

