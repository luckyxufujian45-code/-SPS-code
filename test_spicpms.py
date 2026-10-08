import unittest
import numpy as np
from spicpms import events, features, read_trace


class SignalTests(unittest.TestCase):
    def test_event_boundaries_and_net_signal(self):
        t = np.arange(6) * .001
        y = np.array([10., 20., 30., 10., 15., 10.])
        e = events(np.array([0, 1, 1, 0, 1, 0], dtype=bool), t, y,
                   np.full(6, 10.), np.ones(6))
        self.assertEqual(len(e), 2)
        self.assertEqual(e[0]['net_signal_sum'], 30.)
        self.assertEqual(e[0]['end_index'], 2)
        self.assertAlmostEqual(e[0]['duration_s'], .002)
        cps = events(np.array([0, 1, 1, 0, 1, 0], dtype=bool), t, y,
                     np.full(6, 10.), np.ones(6), 'cps')
        self.assertAlmostEqual(cps[0]['net_counts_estimate'], .03)

    def test_constant_background_remains_finite(self):
        x, baseline, z = features(np.full(1000, 20.))
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue((z == 0).all())

    def test_instrument_csv_preserves_time_and_signal(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'instrument.csv'
            rows = ['sep=;', 'Number;Time 107Ag;107Ag']
            rows += [f'{i + 1};{.005 + .003 * i:.6f};50769.5596408722' for i in range(100)]
            p.write_text('\n'.join(rows), encoding='utf-8-sig')
            t, y = read_trace(p)
            self.assertAlmostEqual(float(np.median(np.diff(t))), .003)
            self.assertAlmostEqual(y[0], 50769.5596408722)


if __name__ == '__main__':
    unittest.main()
