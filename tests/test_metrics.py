"""Tests for metrics module."""
import unittest
from ecoloop.metrics import compare


class TestMetrics(unittest.TestCase):

    def test_positive_savings(self):
        """When AI uses less energy than baseline, savings should be positive."""
        result = compare(baseline_kwh=100.0, ai_kwh=85.0, baseline_peak_kw=10.0, ai_peak_kw=8.0)
        self.assertEqual(result.baseline_kwh, 100.0)
        self.assertEqual(result.ai_kwh, 85.0)
        self.assertEqual(result.energy_savings_pct, 15.0)
        self.assertEqual(result.peak_reduction_pct, 20.0)

    def test_no_savings(self):
        """When AI uses same energy, savings should be zero."""
        result = compare(baseline_kwh=100.0, ai_kwh=100.0, baseline_peak_kw=10.0, ai_peak_kw=10.0)
        self.assertEqual(result.energy_savings_pct, 0.0)
        self.assertEqual(result.peak_reduction_pct, 0.0)

    def test_negative_savings(self):
        """When AI uses more energy, savings should be negative."""
        result = compare(baseline_kwh=100.0, ai_kwh=110.0, baseline_peak_kw=10.0, ai_peak_kw=12.0)
        self.assertEqual(result.energy_savings_pct, -10.0)
        self.assertEqual(result.peak_reduction_pct, -20.0)

    def test_invalid_baseline_raises(self):
        with self.assertRaises(ValueError):
            compare(baseline_kwh=0.0, ai_kwh=50.0, baseline_peak_kw=10.0, ai_peak_kw=5.0)


if __name__ == "__main__":
    unittest.main()
