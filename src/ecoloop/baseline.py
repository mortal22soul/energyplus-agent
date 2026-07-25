"""Naive fixed-schedule baseline controller for comparison against AI mode."""
from __future__ import annotations

from datetime import datetime, timedelta

from .schemas import BuildingState, ControlAction, ZoneSetpoints


class BaselineController:
    """Fixed-schedule baseline: standard office setpoints with simple occupied/unoccupied rule.

    Baseline strategy (intentionally naive for comparison):
    - Occupied (Mon-Fri 8am-6pm): heating 20°C, cooling 26°C
    - Unoccupied: setback to 18°C / 30°C
    - No pre-conditioning, no forecast awareness, no adaptive adjustments

    This provides a clear target for the AI to beat.
    """

    # Fixed setpoints for occupied/unoccupied
    OCCUPIED_HEATING_C: float = 20.0
    OCCUPIED_COOLING_C: float = 26.0
    UNOCCUPIED_HEATING_C: float = 18.0
    UNOCCUPIED_COOLING_C: float = 30.0

    def decide(self, state: BuildingState) -> ControlAction:
        """Return fixed setpoints based on simple occupancy schedule."""
        ts = state.timestamp
        hour = ts.hour + ts.minute / 60.0

        # Weekday occupied: 8am-6pm
        # Weekend: consider unoccupied
        is_occupied = ts.weekday() < 5 and 8.0 <= hour < 18.0

        if is_occupied:
            heating_c = self.OCCUPIED_HEATING_C
            cooling_c = self.OCCUPIED_COOLING_C
            rationale = "occupied baseline: standard comfort"
        else:
            heating_c = self.UNOCCUPIED_HEATING_C
            cooling_c = self.UNOCCUPIED_COOLING_C
            rationale = "unoccupied setback"

        zone_setpoints = {
            zone: ZoneSetpoints(heating_c=heating_c, cooling_c=cooling_c)
            for zone in state.zone_temps_c
        }

        return ControlAction(
            action_id=f"baseline-{ts.strftime('%Y%m%d-%H%M')}",
            zone_setpoints=zone_setpoints,
            mode="normal" if is_occupied else "setback",
            rationale=rationale,
            confidence=1.0,
        )
