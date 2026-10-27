import tempfile
import unittest
from pathlib import Path

from sensoranalyzer.pipeline import Pipeline
from sensoranalyzer.resample import Resampler
from sensoranalyzer.sources import RecordingSource

DATA = Path(__file__).parent / "data"


class PipelineGridTest(unittest.TestCase):
    def test_hold_grid_at_firmware_tick_rate(self):
        grid = []
        stats = Pipeline(RecordingSource(DATA / "drone_boot_v2.imus"), speed=0,
                         resampler=Resampler(100.0, "hold"), on_grid=grid.append).run()
        self.assertGreater(stats.samples, 2000)
        # Сетка 100 Гц на всю длину потока, шаг ровно 10 мс
        steps = {round((b.t - a.t) * 1e3, 6) for a, b in zip(grid, grid[1:])}
        self.assertEqual(steps, {10.0})
        self.assertAlmostEqual(len(grid), stats.ticks, delta=3)


class PlotsTest(unittest.TestCase):
    def test_all_plots_written(self):
        from sensoranalyzer.plots import make_all
        with tempfile.TemporaryDirectory() as d:
            files = make_all(str(DATA / "drone_boot_v2.imus"), d, compare={"gyr_cut_freq": 40.0}, t_from=1.0)
            names = sorted(p.name for p in files)
            self.assertEqual(names, ["compare.png", "gyro_zero.png", "response.png", "sampling.png",
                                     "signals.png", "spectrum.png"])
            for p in files:
                self.assertGreater(p.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
