from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal
from uuid import uuid4


@dataclass(frozen=True)
class ZoneSetpoints:
    heating_c: float
    cooling_c: float


@dataclass(frozen=True)
class ControlAction:
    zone_setpoints: dict[str, ZoneSetpoints]
    mode: Literal["hold", "normal", "precondition", "setback"] = "normal"
    rationale: str = ""
    confidence: float = 0.0
    action_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StateSummary:
    timestamp: datetime
    facility_power_kw: float | None
    zone_temps_c: dict[str, float]


@dataclass(frozen=True)
class BuildingState:
    timestamp: datetime
    step_index: int
    zone_temps_c: dict[str, float]
    zone_relative_humidity_pct: dict[str, float]
    occupancy: dict[str, float]
    heating_setpoints_c: dict[str, float]
    cooling_setpoints_c: dict[str, float]
    zone_pmv: dict[str, float] = field(default_factory=dict)
    zone_ppd_pct: dict[str, float] = field(default_factory=dict)
    outdoor_temp_c: float = 20.0
    forecast_outdoor_temp_c: list[float] = field(default_factory=list)
    zone_co2_ppm: dict[str, float] | None = None
    zone_mean_radiant_temp_c: dict[str, float] | None = None
    hvac_power_kw: float | None = None
    facility_power_kw: float | None = None
    cumulative_facility_energy_kwh: float | None = None
    recent_history: list[StateSummary] = field(default_factory=list)

    def occupied_zones(self) -> set[str]:
        return {zone for zone, people in self.occupancy.items() if people > 0}

    def to_dict(self) -> dict:
        return asdict(self)
