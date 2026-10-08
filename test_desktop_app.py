import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from desktop_app import calibrate_standard


class CalibrationTests(unittest.TestCase):
    def make_standard(self, directory, n):
        p = Path(directory) / 'standard.csv'
        y = np.full(1200, 1000.)
        y[np.arange(100, 100 + n * 100, 100)] += 1000.
        with p.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time', 'signal'])
            w.writerows(zip(np.arange(len(y)) * .003, y))
        return p

    def test_standard_net_integration_and_reference_size(self):
        with tempfile.TemporaryDirectory() as d:
            result = calibrate_standard(self.make_standard(d, 6), 100)
            self.assertEqual(result['reference_event_count'], 6)
            self.assertAlmostEqual(result['reference_median_net_counts'], 3.)
            self.assertEqual(result['reference_diameter_nm'], 100)

    def test_reject_insufficient_reference_events(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                calibrate_standard(self.make_standard(d, 3), 100)


if __name__ == '__main__':
    unittest.main()
