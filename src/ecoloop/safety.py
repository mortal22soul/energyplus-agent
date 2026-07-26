"""Safety engine with hard constraints and rate limiting for HVAC setpoints."""
from __future__ import annotations
from dataclasses import dataclass
from .schemas import BuildingState, ControlAction, ZoneSetpoints


@dataclass(frozen=True)
class SafetyLimits:
    heating_min_c: float = 18.0
    heating_max_c: float = 24.0
    cooling_min_c: float = 22.0
    cooling_max_c: float = 28.0
    minimum_deadband_c: float = 2.0
    max_delta_per_interval_c: float = 0.5
    max_delta_per_hour_c: float = 2.0
    pmv_comfort_band: tuple[float, float] = (-0.5, 0.5)
    unoccupied_heating_c: float = 18.0
    unoccupied_cooling_c: float = 28.0


@dataclass(frozen=True)
class SafetyDecision:
    action: ControlAction
    status: str
    reasons: list[str]


class SafetyEngine:
    """Enforces hard safety constraints on HVAC setpoint proposals.

    Status logic:
    - "accepted":  proposal is within all limits (rate-limit still applied as smoothing).
    - "clamped":    proposal was outside an absolute bound and had to be clamped.
    - "fallback":   no valid proposal was provided.
    """

    def __init__(self, limits: SafetyLimits | None = None) -> None:
        self.limits = limits or SafetyLimits()

    def fallback(self, state: BuildingState, reason: str) -> SafetyDecision:
        action = ControlAction(
            {
                z: ZoneSetpoints(state.heating_setpoints_c[z], state.cooling_setpoints_c[z])
                for z in state.zone_temps_c
            },
            "hold",
            reason,
            1.0,
        )
        return SafetyDecision(action, "fallback", [reason])

    def validate(
        self, state: BuildingState, proposal: ControlAction | None
    ) -> SafetyDecision:
        """Validate and constrain a ControlAction.

        Always applies rate-limiting and deadband enforcement as smoothing.
        Returns "accepted" unless absolute bounds were violated.
        """
        if proposal is None or not isinstance(proposal, ControlAction):
            return self.fallback(state, "no valid proposal")

        out: dict[str, ZoneSetpoints] = {}
        reasons: list[str] = []
        clamped = False

        for zone, old_h in state.heating_setpoints_c.items():
            old_c = state.cooling_setpoints_c[zone]
            p = proposal.zone_setpoints.get(zone)

            if p is None:
                # Zone not in proposal — hold current (no reason needed)
                out[zone] = ZoneSetpoints(old_h, old_c)
                continue

            # 1. Hard range clamping
            h = min(max(p.heating_c, self.limits.heating_min_c), self.limits.heating_max_c)
            c = min(max(p.cooling_c, self.limits.cooling_min_c), self.limits.cooling_max_c)

            if h != p.heating_c or c != p.cooling_c:
                reasons.append(f"{zone}: setpoints outside absolute range, clamped")
                clamped = True

            # 2. Rate limiting (always applied for smooth transitions)
            h = min(max(h, old_h - self.limits.max_delta_per_interval_c), old_h + self.limits.max_delta_per_interval_c)
            c = min(max(c, old_c - self.limits.max_delta_per_interval_c), old_c + self.limits.max_delta_per_interval_c)

            # 3. Deadband enforcement (hard constraint)
            if c - h < self.limits.minimum_deadband_c:
                c = h + self.limits.minimum_deadband_c
                reasons.append(f"{zone}: deadband enforced")
                clamped = True

            out[zone] = ZoneSetpoints(h, c)

        status = "clamped" if clamped else "accepted"
        return SafetyDecision(
            ControlAction(out, proposal.mode, proposal.rationale, proposal.confidence, proposal.action_id),
            status,
            reasons,
        )
