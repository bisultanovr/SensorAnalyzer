import math
import unittest

from dataclasses import replace

from sensoranalyzer.preprocess import (GyroZeroCalibrator, ProcessingParams, Pt2Filter, SensorStage, f32, normalize)
from sensoranalyzer.protocol import FLAG_CALIB_ALLOWED, Sample
from sensoranalyzer.sources import FIRMWARE_DEFAULTS

PARAMS = ProcessingParams.from_config(FIRMWARE_DEFAULTS)


class Pt2Test(unittest.TestCase):
    def test_legacy_coefficient_like_pt2FilterInit(self):
        # FilterPT2.cpp, pt2FilterInit: dt / (rc + dt), rc = 1 / (2 pi fc * 1.553774)
        dt = 1 / 475
        rc = 1 / (2 * math.pi * 25 * 1.553774)
        self.assertAlmostEqual(Pt2Filter.coefficient(25, 475, exact=False), dt / (rc + dt), places=7)  # float32

    def test_legacy_cuts_below_target(self):
        # Формула RC на 475 Гц срезает заметно ниже заданного
        self.assertAlmostEqual(Pt2Filter.magnitude(100, 475, 58.7, exact=False) ** 2, 0.5, delta=0.01)
        self.assertAlmostEqual(Pt2Filter.magnitude(25, 475, 20.4, exact=False) ** 2, 0.5, delta=0.01)

    def test_exact_cutoff_at_target(self):
        # |H|^2 = 1/2 (-3 дБ) ровно на заданной частоте при любой частоте отсчётов
        for fs in (400, 475, 500, 4000):
            for fc in (5, 25, 100, 180):
                if fc < fs / 2:
                    self.assertAlmostEqual(Pt2Filter.magnitude(fc, fs, fc) ** 2, 0.5, places=9, msg=(fc, fs))

    def test_exact_matches_rc_when_sampled_fast(self):
        # Когда отсчёты намного чаще среза, обе формулы дают одно и то же
        self.assertAlmostEqual(Pt2Filter.coefficient(100, 1e6) / Pt2Filter.coefficient(100, 1e6, exact=False),
                               1.0, places=3)

    def test_float32_like_firmware(self):
        # Каждая операция PT2 округляется до float32, как в прошивке без оптимизации
        f = Pt2Filter(0.39710834622383118)
        y = f.apply(4096.0)
        self.assertEqual(y, f32(f32(0.0 + f32(0.39710834622383118 * f32(f32(0.0 + f32(0.39710834622383118 * 4096.0)) - 0.0)))))

    def test_firmware_values(self):
        # Значения pt2FilterCutoffK на 475 Гц (для сверки с прошивкой при отладке)
        self.assertAlmostEqual(Pt2Filter.coefficient(25, 475), 0.3971, places=4)
        self.assertAlmostEqual(Pt2Filter.coefficient(100, 475), 0.8168, places=4)

    def test_step_converges(self):
        f = Pt2Filter(Pt2Filter.coefficient(25, 475))
        for _ in range(200):
            y = f.apply(1000.0)
        self.assertAlmostEqual(y, 1000.0, places=3)


class CalibratorTest(unittest.TestCase):
    def test_ignores_rotation(self):
        c = GyroZeroCalibrator(0.5, 50.0)
        c.update([40.0, 30.0, 1.0])  # |w| > 50
        self.assertEqual(c.shift, [0.0, 0.0, 0.0])
        c.update([10.0, -20.0, 4.0])
        self.assertEqual(c.shift, [5.0, -10.0, 2.0])


class NormalizeTest(unittest.TestCase):
    def test_calibration(self):
        self.assertEqual(normalize(4096, -4096, 4096), 1.0)
        self.assertEqual(normalize(4117, -4086, 4117), 1.0)
        self.assertEqual(normalize(15.5, -4086, 4117), 0.0)


class StageTest(unittest.TestCase):
    def test_static_board(self):
        stage = SensorStage(PARAMS)
        sample = Sample(0, 0, FLAG_CALIB_ALLOWED, (0, 0, 4096), (8, -16, 4))
        for i in range(5000):
            r = stage.process(sample, i / 475)
        self.assertAlmostEqual(r.acc_g[2], 1.0, places=6)
        for s, g in zip(r.gyro_shift, sample.gyr):
            self.assertAlmostEqual(s, g, places=3)  # ноль сошёлся
        for w in r.gyr_dps:
            self.assertAlmostEqual(w, 0.0, places=3)

    def test_no_calibration_when_armed(self):
        stage = SensorStage(PARAMS)
        r = None
        for i in range(500):
            r = stage.process(Sample(i, 0, 0, (0, 0, 4096), (8, -16, 4)), i / 475)
        self.assertEqual(r.gyro_shift, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(r.gyr_dps[0], 8 / 16.4, places=3)

    def test_config_version_selects_formula(self):
        self.assertTrue(PARAMS.exact_cutoff)
        self.assertAlmostEqual(PARAMS.gyro_shift_tau, 0.5, places=6)  # alpha в float32
        legacy = ProcessingParams.from_config(replace(FIRMWARE_DEFAULTS, version=1, gyro_shift_alpha=0.0005 * 4000 / 475))
        self.assertFalse(legacy.exact_cutoff)
        self.assertAlmostEqual(legacy.gyro_shift_tau, 0.4989, places=4)

    def test_rate_override_keeps_tau(self):
        p = PARAMS.with_overrides({"sample_rate": 400.0})
        self.assertAlmostEqual(p.gyro_shift_alpha, 1 - math.exp(-1 / (0.5 * 400)), places=9)
        self.assertIsNone(p.acc_k)  # коэффициенты прошивки для 475 Гц больше не годятся
        p = PARAMS.with_overrides({"gyro_shift_alpha": 0.01})
        self.assertAlmostEqual(p.gyro_shift_tau, -1 / (475 * math.log(0.99)), places=9)

    def test_params_check(self):
        bad = PARAMS.with_overrides({"acc_cut_freq": 300.0, "acc_calib": (10, -10, -1, 1, -1, 1)})
        self.assertEqual(len(bad.check()), 2)
        with self.assertRaises(ValueError):
            PARAMS.with_overrides({"no_such": 1.0})


if __name__ == "__main__":
    unittest.main()
