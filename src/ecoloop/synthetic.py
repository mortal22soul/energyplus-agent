"""Generate realistic synthetic per-timestep EnergyPlus-style output from EPW data."""
from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path


def generate_synthetic_run(
    output_dir: Path,
    epw_path: Path,
    start_day: int = 180,
    num_days: int = 7,
    timestep_minutes: int = 15,
) -> Path:
    """Generate a synthetic EnergyPlus simulation output directory.

    Produces epluszsz.csv and eplusssz.csv in EnergyPlus format,
    using EPW weather data for realistic outdoor conditions.

    Args:
        output_dir: Where to write the synthetic output
        epw_path: Path to EPW weather file
        start_day: 1-indexed day of year to start (180 = late June)
        num_days: Number of days to simulate
        timestep_minutes: Simulation timestep

    Returns:
        Path to output directory
    """
    # Parse EPW hourly temps
    hourly_temps: list[float] = []
    hourly_rh: list[float] = []
    with open(epw_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10 or not parts[0].isdigit():
                continue
            try:
                hourly_temps.append(float(parts[6]))  # dry bulb C
                hourly_rh.append(float(parts[8]))  # relative humidity %
            except ValueError:
                continue

    output_dir.mkdir(parents=True, exist_ok=True)

    # Interpolate hourly to timestep_minutes
    steps_per_hour = 60 // timestep_minutes
    total_steps = num_days * 24 * steps_per_hour

    # Zone names (from 5ZoneAirCooled)
    zones = [f"SPACE{i}-1" for i in range(1, 6)]

    # Thermal response: sinusoidal model with zone-specific phase lag
    # Outdoor temp drives zone temps with different thermal mass/lag per zone
    zone_base_temps = {
        "SPACE1-1": 20.0,   # core zone, stable
        "SPACE2-1": 19.5,   # perimeter, cooler
        "SPACE3-1": 20.5,   # interior, slightly warm
        "SPACE4-1": 19.0,   # perimeter, more swing
        "SPACE5-1": 19.0,   # top floor, most swing
    }
    zone_params = {
        "SPACE1-1": {"amplitude": 1.5, "phase_lag_hours": 1.0},  # core: small lag
        "SPACE2-1": {"amplitude": 2.5, "phase_lag_hours": 2.0},  # perimeter: bigger swing, more lag
        "SPACE3-1": {"amplitude": 1.0, "phase_lag_hours": 0.5},  # interior: minimal swing
        "SPACE4-1": {"amplitude": 2.8, "phase_lag_hours": 1.5},  # perimeter: significant swing
        "SPACE5-1": {"amplitude": 3.2, "phase_lag_hours": 2.5},  # top floor: max swing, max lag
    }

    # Occupancy: 8am-6pm weekdays, 9am-5pm weekends
    def is_occupied(step_index: int) -> bool:
        ts = datetime(2025, 1, 1) + timedelta(minutes=step_index * timestep_minutes)
        hour = ts.hour + ts.minute / 60.0
        weekday = ts.weekday() < 5
        if weekday:
            return 8.0 <= hour < 18.0
        return 9.0 <= hour < 17.0

    # Write epluszsz.csv (zone sizing/time-series format)
    zsz_path = output_dir / "epluszsz.csv"
    _write_zsz_csv(
        zsz_path, zones, zone_base_temps, zone_params,
        hourly_temps, hourly_rh, start_day, num_days,
        timestep_minutes, steps_per_hour, total_steps
    )

    # Write eplusssz.csv (system sizing format)
    ssz_path = output_dir / "eplusssz.csv"
    _write_ssz_csv(
        ssz_path, hourly_temps, start_day, num_days,
        timestep_minutes, steps_per_hour, total_steps
    )

    # Write manifest
    _write_manifest(output_dir, epw_path, start_day, num_days, timestep_minutes)

    return output_dir


def _write_zsz_csv(
    path: Path,
    zones: list[str],
    zone_base_temps: dict[str, float],
    zone_params: dict[str, dict],
    hourly_temps: list[float],
    hourly_rh: list[float],
    start_day: int,
    num_days: int,
    timestep_minutes: int,
    steps_per_hour: int,
    total_steps: int,
) -> None:
    """Write zone sizing/time-series CSV in EnergyPlus format."""
    # Build header matching the real epluszsz.csv format
    headers = ["Time"]
    for zone in zones:
        headers.extend([
            f"{zone}:CHICAGO_IL_USA ANNUAL HEATING 99% DESIGN CONDITIONS DB:Des Heat Load [W]",
            f"{zone}:CHICAGO_IL_USA ANNUAL COOLING 1% DESIGN CONDITIONS DB/MCWB:Des Sens Cool Load [W]",
            f"{zone}:CHICAGO_IL_USA ANNUAL HEATING 99% DESIGN CONDITIONS DB:Des Heat Mass Flow [kg/s]",
            f"{zone}:CHICAGO_IL_USA ANNUAL COOLING 1% DESIGN CONDITIONS DB/MCWB:Des Cool Mass Flow [kg/s]",
            f"{zone}::Des Latent Heat Load [W]",
            f"{zone}::Des Latent Cool Load [W]",
            f"{zone}::Des Latent Heat Mass Flow [kg/s]",
            f"{zone}::Des Latent Cool Mass Flow [kg/s]",
            f"{zone}::Des Heat Load No DOAS [W]",
            f"{zone}::Des Sens Cool Load No DOAS [W]",
            f"{zone}::Des Latent Heat Load No DOAS [W]",
            f"{zone}::Des Latent Cool Load No DOAS [W]",
            f"{zone}:CHICAGO_IL_USA ANNUAL HEATING 99% DESIGN CONDITIONS DB:Heating Zone Temperature [C]",
            f"{zone}:CHICAGO_IL_USA ANNUAL HEATING 99% DESIGN CONDITIONS DB:Heating Zone Relative Humidity [%]",
            f"{zone}:CHICAGO_IL_USA ANNUAL COOLING 1% DESIGN CONDITIONS DB/MCWB:Cooling Zone Temperature [C]",
            f"{zone}:CHICAGO_IL_USA ANNUAL COOLING 1% DESIGN CONDITIONS DB/MCWB:Cooling Zone Relative Humidity [%]",
        ])

    # Site-level outdoor conditions
    headers.extend([
        "Site Outdoor Air Drybulb Temperature [C]",
        "Site Outdoor Air Dewpoint Temperature [C]",
        "Site Outdoor Air Wetbulb Temperature [C]",
        "Site Outdoor Air Relative Humidity [%]",
    ])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        # Track zone temps for thermal lag
        zone_current_temps = dict(zone_base_temps)

        for step in range(total_steps):
            # Time string
            total_minutes = step * timestep_minutes
            h = (total_minutes // 60) % 24
            m = total_minutes % 60
            time_str = f"{h:02d}:{m:02d}:00"

            # Outdoor temp from EPW (interpolate to 15-min)
            hour_idx = start_day * 24 + step // steps_per_hour
            hour_idx = min(hour_idx, len(hourly_temps) - 1)
            outdoor_temp = hourly_temps[hour_idx]
            outdoor_rh = hourly_rh[hour_idx]

            row = [time_str]

            for zone in zones:
                base = zone_base_temps[zone]
                amp = zone_params[zone]["amplitude"]
                lag = zone_params[zone]["phase_lag_hours"]

                # Hour of day with phase lag
                hour_of_day = (step * timestep_minutes) / 60.0
                lagged_hour = hour_of_day - lag
                # Sinusoidal variation: warmest at ~14:00, coldest at ~05:00
                sin_input = (lagged_hour - 9.5) / 12.0 * 2 * 3.14159  # peak at 9.5+12/2=15.5h
                outdoor_component = amp * (1 + 1.5 * (outdoor_temp - 15.0) / 20.0)

                # Day/night: higher occupancy heat gain during day
                is_day = 8 <= (hour_of_day % 24) < 18
                occupancy_gain = 0.8 if is_day else 0.0

                zt = base + outdoor_component * math.sin(sin_input) + occupancy_gain

                # Small random walk for realism
                zt += (hash((step, zone)) % 100) / 5000.0 - 0.01

                # Clamp to realistic range
                zt = max(15.0, min(32.0, zt))

                # Heating/cooling loads based on difference from comfort band
                heating_load = max(0, 21.0 - zt) * 500 + 500  # W
                cooling_load = max(0, zt - 24.0) * 400 + 300  # W

                # RH: varies with outdoor, moderated indoors
                indoor_rh = 40.0 + (outdoor_rh - 50.0) * 0.2
                indoor_rh = max(25.0, min(65.0, indoor_rh))

                # Zone temps for heating/cooling setpoints (design conditions)
                heating_zone_temp = 21.0 + (22.0 - outdoor_temp) * 0.05
                cooling_zone_temp = 24.0 + (outdoor_temp - 22.0) * 0.05

                row.extend([
                    f"{heating_load:.3E}",
                    f"{cooling_load:.3E}",
                    f"{heating_load * 0.001:.3E}",
                    f"{cooling_load * 0.001:.3E}",
                    "0.000000E+00",
                    "0.000000E+00",
                    "0.000000E+00",
                    "0.000000E+00",
                    f"{heating_load * 0.95:.3E}",
                    f"{cooling_load * 0.95:.3E}",
                    "0.000000E+00",
                    "0.000000E+00",
                    f"{heating_zone_temp:.6E}",
                    f"{indoor_rh:.3E}",
                    f"{cooling_zone_temp:.6E}",
                    f"{indoor_rh:.3E}",
                ])

            # Site outdoor
            writer.writerow(row + [
                f"{outdoor_temp:.6E}",
                f"{outdoor_temp - 2.0:.6E}",
                f"{outdoor_temp - 1.0:.6E}",
                f"{outdoor_rh:.3E}",
            ])


def _write_ssz_csv(
    path: Path,
    hourly_temps: list[float],
    start_day: int,
    num_days: int,
    timestep_minutes: int,
    steps_per_hour: int,
    total_steps: int,
) -> None:
    """Write system sizing CSV in EnergyPlus format."""
    headers = [
        "Time",
        "VAV SYS 1:DesPer 1:Des Heat Mass Flow [kg/s]",
        "VAV SYS 1:DesPer 1:Des Heat Cap [W]",
        "VAV SYS 1:DesPer 1:Des Cool Mass Flow [kg/s]",
        "VAV SYS 1:DesPer 1:Des Sens Cool Cap [W]",
        "VAV SYS 1:DesPer 1:Des Tot Cool Cap [W]",
        "VAV SYS 1:DesPer 2:Des Heat Mass Flow [kg/s]",
        "VAV SYS 1:DesPer 2:Des Heat Cap [W]",
        "VAV SYS 1:DesPer 2:Des Cool Mass Flow [kg/s]",
        "VAV SYS 1:DesPer 2:Des Sens Cool Cap [W]",
        "VAV SYS 1:DesPer 2:Des Tot Cool Cap [W]",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for step in range(total_steps):
            total_minutes = step * timestep_minutes
            h = (total_minutes // 60) % 24
            m = total_minutes % 60
            time_str = f"{h:02d}:{m:02d}:00"

            hour_idx = start_day * 24 + step // steps_per_hour
            hour_idx = min(hour_idx, len(hourly_temps) - 1)
            outdoor_temp = hourly_temps[hour_idx]

            # System sizing based on outdoor temp
            heat_cap = max(0, 25.0 - outdoor_temp) * 1000 + 5000
            cool_cap = max(0, outdoor_temp - 20.0) * 800 + 3000

            writer.writerow([
                time_str,
                f"{heat_cap * 0.5:.3E}",
                f"{heat_cap:.3E}",
                f"{cool_cap * 0.001:.3E}",
                f"{cool_cap:.3E}",
                f"{cool_cap * 1.1:.3E}",
                f"{heat_cap * 0.3:.3E}",
                f"{heat_cap * 0.6:.3E}",
                f"{cool_cap * 0.0005:.3E}",
                f"{cool_cap * 0.5:.3E}",
                f"{cool_cap * 0.55:.3E}",
            ])


def _write_manifest(output_dir: Path, epw_path: Path, start_day: int, num_days: int, timestep_minutes: int) -> None:
    """Write simulation manifest."""
    import json
    manifest = {
        "type": "synthetic_simulation",
        "generated_at": datetime.now().isoformat(),
        "epw_source": str(epw_path),
        "start_day_of_year": start_day,
        "num_days": num_days,
        "timestep_minutes": timestep_minutes,
        "total_steps": num_days * 24 * (60 // timestep_minutes),
        "zones": ["SPACE1-1", "SPACE2-1", "SPACE3-1", "SPACE4-1", "SPACE5-1"],
        "note": "Synthetic per-timestep data generated from EPW weather file. Zone temps use thermal lag model.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
