import random
import unittest

from sensoranalyzer.preprocess import SensorResult
from sensoranalyzer.resample import Resampler


def result(t: float, v: float) -> SensorResult:
    """Отсчёт, у которого все величины равны v (по осям - v, 2v, 3v)."""
    vec = (v, 2 * v, 3 * v)
    return SensorResult(t, (0, 0, 0), (0, 0, 0), vec, vec, vec, vec, vec, True)


def irregular_times(n: int, seed: int = 1) -> list[float]:
    # Шаг 1.3-3.5 мс, как у дрона
    rnd = random.Random(seed)
    t, out = 0.0013, []
    for _ in range(n):
        out.append(t)
        t += rnd.uniform(0.0013, 0.0035)
    return out


class HoldTest(unittest.TestCase):
    def test_takes_latest_sample_like_firmware_tick(self):
        times = irregular_times(2000)
        r = Resampler(100.0, "hold")
        grid = [p for i, t in enumerate(times) for p in r.push(result(t, float(i)))]
        for p in grid:
            latest = max(i for i, t in enumerate(times) if t <= p.t + 1e-12)
            self.assertEqual(p.acc_g[0], float(latest))
        self.assertEqual([round(p.t * 100) for p in grid], list(range(1, len(grid) + 1)))

    def test_grid_covers_stream(self):
        times = irregular_times(4000)
        r = Resampler(100.0, "hold")
        grid = [p for t in times for p in r.push(result(t, 0.0))]
        self.assertEqual(len(grid), int(times[-1] * 100))
        self.assertAlmostEqual(r.stats.dt_mean, (times[-1] - times[0]) / (len(times) - 1), places=12)
        self.assertGreaterEqual(r.stats.dt_min, 0.0013 - 1e-12)


class LinearTest(unittest.TestCase):
    def test_ramp_is_exact(self):
        times = irregular_times(1000, seed=7)
        r = Resampler(250.0, "linear")
        grid = [p for t in times for p in r.push(result(t, 5.0 * t))]
        self.assertGreater(len(grid), 100)
        for p in grid:
            self.assertAlmostEqual(p.gyr_dps[0], 5.0 * p.t, places=9)
            self.assertAlmostEqual(p.gyr_dps[2], 15.0 * p.t, places=9)


class GapTest(unittest.TestCase):
    def test_points_inside_long_gap_are_skipped(self):
        r = Resampler(100.0, "linear", max_gap=0.05)
        out = r.push(result(0.0, 0.0)) + r.push(result(0.005, 1.0)) + r.push(result(0.5, 2.0))
        self.assertEqual([round(p.t, 3) for p in out], [0.0, 0.5])
        self.assertEqual(r.stats.skipped, 49)

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            Resampler(0)
        with self.assertRaises(ValueError):
            Resampler(100, "cubic")


if __name__ == "__main__":
    unittest.main()
