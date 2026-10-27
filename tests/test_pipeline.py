import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sensoranalyzer.export import export_csv
from sensoranalyzer.pipeline import Pipeline
from sensoranalyzer.preprocess import ProcessingParams, SensorStage
from sensoranalyzer.protocol import FLAG_CALIB_ALLOWED, Sample, Tick, encode
from sensoranalyzer.sources import FIRMWARE_DEFAULTS, CsvSource, RecordingSource

DATA = Path(__file__).parent / "data"
RECORDING = DATA / "drone_idle_5s.imus"  # дрон неподвижен, снято с работающей платы


class RecordedDroneTest(unittest.TestCase):
    def test_model_matches_firmware_gyro_zero(self):
        results, checks = [], []
        stats = Pipeline(RecordingSource(RECORDING), speed=0,
                         on_result=results.append, on_tick=checks.append).run()
        self.assertGreater(stats.samples, 2000)
        self.assertEqual((stats.lost_samples, stats.lost_ticks, stats.buffer_dropped), (0, 0, 0))
        # Модель с float64 против float32 прошивки: расхождение - доли тысячной кода АЦП
        self.assertLess(stats.shift_diff_max, 1e-3)
        last = results[-1]
        self.assertAlmostEqual(sum(a * a for a in last.acc_g) ** 0.5, 1.0, delta=0.01)
        self.assertTrue(all(abs(w) < 0.5 for w in last.gyr_dps))

    def test_override_changes_result(self):
        base, changed = [], []
        Pipeline(RecordingSource(RECORDING), speed=0, on_result=base.append).run()
        Pipeline(RecordingSource(RECORDING), speed=0, overrides={"acc_cut_freq": 5.0},
                 on_result=changed.append).run()
        self.assertEqual(len(base), len(changed))
        self.assertNotEqual(base[-1].acc_filtered, changed[-1].acc_filtered)


class RecordedBootV2Test(unittest.TestCase):
    """Прошивка с CONFIG v2, запись захватывает перезапуск платы: секунда старого сеанса, сброс,
    старт с номинальных 475 Гц, через секунду - переход на измеренные 400 Гц."""

    def test_bit_exact_through_restart_and_rate_change(self):
        rates = []
        pipeline = Pipeline(RecordingSource(DATA / "drone_boot_v2.imus"), speed=0,
                            on_config=lambda config, params: rates.append(config.sample_rate))
        stats = pipeline.run()
        self.assertEqual(stats.restarts, 1)
        self.assertEqual(rates[0], 475.0)              # номинал до первого замера
        self.assertAlmostEqual(rates[-1], 400.0, delta=1.0)  # измеренная
        self.assertTrue(stats.bit_exact, stats)


class SyntheticBootTest(unittest.TestCase):
    """Поток с включения платы: без seed модель повторяет прошивку с первого отсчёта."""

    def test_exact_from_boot(self):
        params = ProcessingParams.from_config(FIRMWARE_DEFAULTS)
        firmware = SensorStage(params)  # та же модель в роли прошивки
        stream = bytearray(encode(FIRMWARE_DEFAULTS))
        t = 0
        for i in range(950):
            s = Sample(i, t, FLAG_CALIB_ALLOWED, (i % 7, -3, 4096), (8 + i % 3, -16, 4))
            firmware.process(s, 0)
            stream += encode(s)
            t += 2105
            if i % 5 == 4:
                shift = tuple(firmware.calibrator.shift)
                stream += encode(Tick(i // 5, t, FLAG_CALIB_ALLOWED, (0, 0, 0, 1), shift, 0.01))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "boot.imus"
            path.write_bytes(stream)
            stats = Pipeline(RecordingSource(path), speed=0, seed=False).run()
        self.assertEqual(stats.ticks, 190)
        self.assertEqual(stats.shift_diff_max, 0.0)  # модель и "прошивка" обе в float32

    def test_rate_change_applies_at_same_sample(self):
        # Прошивка сменила частоту (IMU_SetSampleRate) и в том же такте прислала CONFIG:
        # модель должна пересчитать коэффициенты с того же отсчёта
        params = ProcessingParams.from_config(FIRMWARE_DEFAULTS)
        firmware = SensorStage(params)
        stream = bytearray(encode(FIRMWARE_DEFAULTS))
        t = 0
        for i in range(2000):
            s = Sample(i, t, FLAG_CALIB_ALLOWED, (i % 11, -3, 4096), (8 + (i * 7) % 13, -16, 4))
            firmware.process(s, 0)
            stream += encode(s)
            t += 2500
            if i % 4 == 3:
                shift = tuple(firmware.calibrator.shift)
                stream += encode(Tick(i // 4, t, FLAG_CALIB_ALLOWED, (0, 0, 0, 1), shift, 0.01))
                if i == 999:
                    new = params.with_overrides({"sample_rate": 400.0})
                    firmware.set_params(new)
                    acc_k, gyr_k = new.filter_k()
                    stream += encode(replace(FIRMWARE_DEFAULTS, sample_rate=400.0, gyro_shift_alpha=new.gyro_shift_alpha,
                                             acc_k=acc_k, gyr_k=gyr_k, filter_state=firmware.filter_state,
                                             gyro_shift=shift))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rate.imus"
            path.write_bytes(stream)
            pipeline = Pipeline(RecordingSource(path), speed=0, seed=False)
            stats = pipeline.run()
        self.assertEqual(pipeline.params.sample_rate, 400.0)
        self.assertTrue(stats.bit_exact, stats)


class LiveAttachTest(unittest.TestCase):
    """Подключение к работающей плате: состояние берётся из CONFIG v2, дальше - бит в бит."""

    def test_seed_from_config_state(self):
        params = ProcessingParams.from_config(FIRMWARE_DEFAULTS)
        firmware = SensorStage(params)
        stream = bytearray()
        t = 0
        for i in range(3000):
            s = Sample(i, t, FLAG_CALIB_ALLOWED, (i % 5, 30, 4100), (12 + i % 4, -20, 6))
            firmware.process(s, 0)
            if i >= 700:  # ПК подключился не с начала
                stream += encode(s)
            t += 2105
            if i % 5 == 4 and i >= 700:
                shift = tuple(firmware.calibrator.shift)
                stream += encode(Tick(i // 5, t, FLAG_CALIB_ALLOWED, (0, 0, 0, 1), shift, 0.01))
                if (i // 5) % 100 == 0:
                    stream += encode(replace(FIRMWARE_DEFAULTS, filter_state=firmware.filter_state, gyro_shift=shift))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "attach.imus"
            path.write_bytes(stream)
            stats = Pipeline(RecordingSource(path), speed=0).run()
        self.assertGreater(stats.before_config, 0)
        self.assertEqual(stats.state_checks, 3)  # первый CONFIG - подхват, ещё три - сверка
        self.assertTrue(stats.bit_exact, stats)


class RestartTest(unittest.TestCase):
    def test_restart_resets_model_and_keeps_time(self):
        params = ProcessingParams.from_config(FIRMWARE_DEFAULTS)
        stream = bytearray()
        times = []
        for run in range(2):  # два сеанса платы подряд: номера и время начинаются заново
            firmware = SensorStage(params)
            for i in range(500):
                s = Sample(i, 1000 + i * 2105, FLAG_CALIB_ALLOWED, (run * 50, 0, 4096), (10 + run, -5, 3))
                firmware.process(s, 0)
                stream += encode(s)
                if i % 5 == 4:
                    shift = tuple(firmware.calibrator.shift)
                    stream += encode(Tick(i // 5, 1000 + i * 2105, FLAG_CALIB_ALLOWED, (0, 0, 0, 1), shift, 0.01))
                    if i % 500 == 4:  # прошивка шлёт CONFIG с первого такта после включения
                        stream += encode(replace(FIRMWARE_DEFAULTS, filter_state=firmware.filter_state, gyro_shift=shift))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "restart.imus"
            path.write_bytes(stream)
            stats = Pipeline(RecordingSource(path), speed=0, seed=False,
                             on_result=lambda r: times.append(r.t)).run()
        self.assertEqual(stats.restarts, 1)
        self.assertEqual(stats.shift_diff_max, 0.0)
        self.assertEqual(stats.samples, 1000)
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])))  # время не прыгает назад


class CsvTest(unittest.TestCase):
    def test_export_and_replay(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "rec.csv"
            count = export_csv(RECORDING, out)
            source = CsvSource(out)
            self.assertEqual(source.validate(), [])
            results = []
            stats = Pipeline(source, speed=0, on_result=results.append).run()
            self.assertEqual(stats.samples, count)

    def test_validate_reports_errors(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.csv"
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_ms", "ax_raw", "ay_raw", "az_raw", "gx_raw", "gy_raw", "gz_raw"])
                w.writerow([10, 1, 2, 3, 4, 5, 6])
                w.writerow([10, 1, 2, 3, 4, 5, 6])      # время не возрастает
                w.writerow([20, 1, 2, 40000, 4, 5, 6])  # вне int16
                w.writerow([30, "x", 2, 3, 4, 5, 6])
            errors = CsvSource(path).validate()
            self.assertEqual(len(errors), 3)

    def test_missing_columns(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.csv"
            path.write_text("time,ax\n1,2\n")
            self.assertEqual(len(CsvSource(path).validate()), 2)


if __name__ == "__main__":
    unittest.main()
