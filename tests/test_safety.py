"""Tests for the safety engine."""
import unittest
from ecoloop.safety import SafetyEngine, SafetyLimits
from ecoloop.schemas import BuildingState, ControlAction, ZoneSetpoints


def make_state(zone="Z1", h_sp=20.0, c_sp=24.0, temp=22.0, occupied=True):
    occ = {zone: 1.0} if occupied else {zone: 0.0}
    temps = {zone: temp}
    return BuildingState(
        timestamp=__import__("datetime").datetime(2025, 1, 1, 12, 0),
        step_index=1,
        zone_temps_c=temps,
        zone_relative_humidity_pct={zone: 50.0},
        occupancy=occ,
        heating_setpoints_c={zone: h_sp},
        cooling_setpoints_c={zone: c_sp},
    )


class TestSafetyEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SafetyEngine()

    def test_accept_valid_action(self):
        state = make_state()
        action = ControlAction(zone_setpoints={"Z1": ZoneSetpoints(20.0, 24.0)})
        decision = self.engine.validate(state, action)
        self.assertEqual(decision.status, "accepted")
        self.assertEqual(decision.reasons, [])

    def test_clamp_over_max_cooling(self):
        state = make_state(c_sp=24.0)
        action = ControlAction(zone_setpoints={"Z1": ZoneSetpoints(20.0, 35.0)})
        decision = self.engine.validate(state, action)
        # Absolute max is 28, but rate limit from current 24.0 caps at 24.5
        self.assertEqual(decision.action.zone_setpoints["Z1"].cooling_c, 24.5)

    def test_clamp_below_min_heating(self):
        state = make_state(h_sp=20.0)
        action = ControlAction(zone_setpoints={"Z1": ZoneSetpoints(10.0, 24.0)})
        decision = self.engine.validate(state, action)
        # Absolute min is 18, but rate limit from current 20.0 floors at 19.5
        self.assertEqual(decision.action.zone_setpoints["Z1"].heating_c, 19.5)

    def test_rate_limit_heating(self):
        state = make_state(h_sp=20.0)
        # Try to raise heating by 1.0 C in one step (limit is 0.5 C)
        action = ControlAction(zone_setpoints={"Z1": ZoneSetpoints(21.0, 24.0)})
        decision = self.engine.validate(state, action)
        self.assertEqual(decision.action.zone_setpoints["Z1"].heating_c, 20.5)

    def test_deadband_enforced(self):
        state = make_state(h_sp=22.0, c_sp=23.0)
        action = ControlAction(zone_setpoints={"Z1": ZoneSetpoints(22.0, 23.0)})
        decision = self.engine.validate(state, action)
        c = decision.action.zone_setpoints["Z1"].cooling_c
        h = decision.action.zone_setpoints["Z1"].heating_c
        self.assertGreaterEqual(c - h, 2.0)

    def test_none_proposal_returns_fallback(self):
        state = make_state()
        decision = self.engine.validate(state, None)
        self.assertEqual(decision.status, "fallback")
        self.assertEqual(decision.action.mode, "hold")

    def test_invalid_action_type_returns_fallback(self):
        state = make_state()
        decision = self.engine.validate(state, "not an action")
        self.assertEqual(decision.status, "fallback")


if __name__ == "__main__":
    unittest.main()
