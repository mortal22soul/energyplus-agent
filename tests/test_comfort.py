"""Tests for the comfort module."""
import unittest
from ecoloop.comfort import calculate_pmv_ppd


class TestComfort(unittest.TestCase):
    def test_pmv_comfortable(self):
        """Standard office conditions should yield comfortable PMV."""
        # 0.5 clo is summer clothing; 22°C/50% feels slightly cool per Fanger.
        # Use warmer conditions or higher clo for comfortable zone.
        result = calculate_pmv_ppd(
            air_temp_c=23.5, radiant_temp_c=23.5, relative_humidity_pct=50.0
        )
        self.assertGreaterEqual(result.pmv, -0.5)
        self.assertLessEqual(result.pmv, 0.5)

    def test_pmv_cold(self):
        """Cold conditions should yield negative PMV."""
        result = calculate_pmv_ppd(
            air_temp_c=15.0, radiant_temp_c=15.0, relative_humidity_pct=40.0
        )
        self.assertLess(result.pmv, -1.0)

    def test_pmv_hot(self):
        """Hot conditions should yield positive PMV."""
        result = calculate_pmv_ppd(
            air_temp_c=30.0, radiant_temp_c=30.0, relative_humidity_pct=50.0
        )
        self.assertGreater(result.pmv, 1.0)

    def test_ppd_low_when_comfortable(self):
        """PPD should be low when PMV is near zero."""
        result = calculate_pmv_ppd(
            air_temp_c=22.0, radiant_temp_c=22.0, relative_humidity_pct=50.0
        )
        self.assertLess(result.ppd_pct, 20.0)

    def test_invalid_humidity_raises(self):
        """Humidity outside 0-100 should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_pmv_ppd(
                air_temp_c=22.0, radiant_temp_c=22.0, relative_humidity_pct=150.0
            )

    def test_ppd_high_when_uncomfortable(self):
        """PPD should be high when PMV is extreme."""
        result = calculate_pmv_ppd(
            air_temp_c=35.0, radiant_temp_c=35.0, relative_humidity_pct=70.0
        )
        self.assertGreater(result.ppd_pct, 50.0)


if __name__ == "__main__":
    unittest.main()
