"""Tests for the MPC optimizer and thermal model."""
import unittest

from ecoloop.mpc import (
    SimpleRCThermalModel,
    ThermalModelParams,
    MPCOptimizer,
    evaluate_mpc,
)
from ecoloop.config import SafetyLimits


class TestSimpleRCThermalModel(unittest.TestCase):
    """Test the 3R-2C thermal network model."""

    def setUp(self):
        self.model = SimpleRCThermalModel(ThermalModelParams())

    def test_predict_returns_float(self):
        result = self.model.predict(22.0, 30.0, 20.0, 26.0, 0.25)
        self.assertIsInstance(result, float)

    def test_predict_cooling_drift(self):
        """Zone temp should move toward outdoor temp when no HVAC."""
        # Hot outdoor, zone in deadband
        t1 = self.model.predict(23.0, 35.0, 20.0, 26.0, 0.25)
        # Zone should be warmer than before (drifting toward outdoor)
        self.assertGreater(t1, 23.0)

    def test_predict_heating_drift(self):
        """Zone should cool down with cold outdoor and wide deadband."""
        t1 = self.model.predict(22.0, 5.0, 18.0, 28.0, 0.25)
        # Zone should be cooler (drifting toward cold outdoor)
        self.assertLess(t1, 22.0)

    def test_predict_hvac_pulls_back(self):
        """HVAC should limit temperature when exceeding setpoints."""
        # Very hot outdoor, zone at cooling setpoint boundary
        t1 = self.model.predict(27.0, 40.0, 20.0, 26.0, 0.25)
        # HVAC cooling should limit the rise
        self.assertLess(t1, 40.0)

    def test_predict_short_horizon(self):
        """Short prediction horizon should show small change."""
        t1 = self.model.predict(22.0, 22.0, 20.0, 26.0, 0.01)
        self.assertAlmostEqual(t1, 22.0, delta=1.0)

    def test_estimate_power_in_deadband(self):
        """No power needed when in deadband."""
        power = self.model.estimate_power_kw(23.0, 25.0, 20.0, 26.0)
        self.assertEqual(power, 0.0)

    def test_estimate_power_cooling(self):
        """Cooling power when above cooling setpoint."""
        power = self.model.estimate_power_kw(28.0, 35.0, 20.0, 26.0)
        self.assertGreater(power, 0.0)

    def test_estimate_power_heating(self):
        """Heating power when below heating setpoint."""
        power = self.model.estimate_power_kw(17.0, 5.0, 20.0, 26.0)
        self.assertGreater(power, 0.0)


class TestMPCOptimizer(unittest.TestCase):
    """Test the MPC brute-force optimizer."""

    def setUp(self):
        self.optimizer = MPCOptimizer(
            safety_limits=SafetyLimits(),
            horizon_hours=1.0,  # short horizon for fast tests
            setpoint_resolution=1.0,  # coarse for speed
        )

    def test_optimize_returns_all_zones(self):
        zones = {"SPACE1-1": 22.0, "SPACE2-1": 21.0}
        results = self.optimizer.optimize(
            zone_temps=zones,
            outdoor_temp=25.0,
            forecast_temps=[25.0] * 4,
            occupancy={"SPACE1-1": 1.0, "SPACE2-1": 0.0},
            current_heating_sp={"SPACE1-1": 20.0, "SPACE2-1": 18.0},
            current_cooling_sp={"SPACE1-1": 26.0, "SPACE2-1": 28.0},
        )
        self.assertIn("SPACE1-1", results)
        self.assertIn("SPACE2-1", results)

    def test_optimize_respects_safety_limits(self):
        zones = {"SPACE1-1": 22.0}
        results = self.optimizer.optimize(
            zone_temps=zones,
            outdoor_temp=25.0,
            forecast_temps=[25.0] * 4,
            occupancy={"SPACE1-1": 1.0},
            current_heating_sp={"SPACE1-1": 20.0},
            current_cooling_sp={"SPACE1-1": 26.0},
        )
        sp = results["SPACE1-1"]
        limits = SafetyLimits()
        self.assertGreaterEqual(sp["heating_c"], limits.heating_min_c)
        self.assertLessEqual(sp["heating_c"], limits.heating_max_c)
        self.assertGreaterEqual(sp["cooling_c"], limits.cooling_min_c)
        self.assertLessEqual(sp["cooling_c"], limits.cooling_max_c)
        # Deadband
        self.assertGreaterEqual(sp["cooling_c"] - sp["heating_c"], limits.minimum_deadband_c)

    def test_optimize_has_predicted_temp(self):
        zones = {"SPACE1-1": 22.0}
        results = self.optimizer.optimize(
            zone_temps=zones,
            outdoor_temp=25.0,
            forecast_temps=[25.0] * 4,
            occupancy={"SPACE1-1": 1.0},
            current_heating_sp={"SPACE1-1": 20.0},
            current_cooling_sp={"SPACE1-1": 26.0},
        )
        self.assertIn("predicted_temp_c", results["SPACE1-1"])
        self.assertIsInstance(results["SPACE1-1"]["predicted_temp_c"], float)

    def test_to_control_action(self):
        zones = {"SPACE1-1": 22.0}
        results = self.optimizer.optimize(
            zone_temps=zones,
            outdoor_temp=25.0,
            forecast_temps=[25.0] * 4,
            occupancy={"SPACE1-1": 1.0},
            current_heating_sp={"SPACE1-1": 20.0},
            current_cooling_sp={"SPACE1-1": 26.0},
        )
        action = self.optimizer.to_control_action(results)
        self.assertIsNotNone(action)
        self.assertIn("SPACE1-1", action.zone_setpoints)

    def test_rate_limit_respected(self):
        """Optimizer should respect rate limit (max 0.5C per step)."""
        zones = {"SPACE1-1": 22.0}
        results = self.optimizer.optimize(
            zone_temps=zones,
            outdoor_temp=25.0,
            forecast_temps=[25.0] * 4,
            occupancy={"SPACE1-1": 1.0},
            current_heating_sp={"SPACE1-1": 20.0},
            current_cooling_sp={"SPACE1-1": 26.0},
        )
        sp = results["SPACE1-1"]
        # Candidate range is within rate limit of current setpoints
        self.assertLessEqual(abs(sp["heating_c"] - 20.0), 0.5 + 0.01)
        self.assertLessEqual(abs(sp["cooling_c"] - 26.0), 0.5 + 0.01)


class TestEvaluateMPC(unittest.TestCase):
    """Test the MPC evaluator function."""

    def test_evaluate_mpc_returns_dict(self):
        result = evaluate_mpc(
            zone_temps={"SPACE1-1": 22.0},
            outdoor_temp=25.0,
            forecast_temps=[25.0] * 4,
            occupancy={"SPACE1-1": 1.0},
            current_heating_sp={"SPACE1-1": 20.0},
            current_cooling_sp={"SPACE1-1": 26.0},
        )
        self.assertEqual(result["status"], "optimized")
        self.assertIn("recommendations", result)
        self.assertIn("SPACE1-1", result["recommendations"])

    def test_evaluate_mpc_with_custom_limits(self):
        limits = SafetyLimits(heating_min_c=19.0, cooling_max_c=27.0)
        result = evaluate_mpc(
            zone_temps={"SPACE1-1": 22.0},
            outdoor_temp=25.0,
            forecast_temps=[25.0] * 4,
            occupancy={"SPACE1-1": 1.0},
            current_heating_sp={"SPACE1-1": 20.0},
            current_cooling_sp={"SPACE1-1": 26.0},
            safety_limits=limits,
        )
        self.assertEqual(result["status"], "optimized")
        rec = result["recommendations"]["SPACE1-1"]
        self.assertGreaterEqual(rec["heating_c"], 19.0)
        self.assertLessEqual(rec["cooling_c"], 27.0)


if __name__ == "__main__":
    unittest.main()
