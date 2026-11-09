import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sensoranalyzer.export import export_csv
from sensoranalyzer.orientation import (IDENTITY, OrientationStage, accel_shift, delta_axis_angle, pitch_roll_yaw,
                                        qmul, qnorm)
from sensoranalyzer.pipeline import Pipeline
from sensoranalyzer.preprocess import ProcessingParams, SensorStage, f32
from sensoranalyzer.protocol import Sample, Tick, encode
from sensoranalyzer.sources import FIRMWARE_DEFAULTS, CsvSource, RecordingSource

DATA = Path(__file__).parent / "data"
LEVEL = replace(ProcessingParams.from_config(FIRMWARE_DEFAULTS), sample_rate=400.0)  # без поправки установки
TIMER = 50_000_000
STEP = TIMER // 400  # отсчёт 400 Гц в тиках таймера


def _rotation(axis: int, degrees: float) -> tuple:
    q = [0.0, 0.0, 0.0, math.cos(math.radians(degrees) / 2)]
    q[axis] = math.sin(math.radians(degrees) / 2)
    return tuple(q)


class QuaternionMathTest(unittest.TestCase):
    def test_product_and_norm(self):
        q = qnorm((0.1, -0.2, 0.3, 0.9))
        self.assertEqual(qmul(q, IDENTITY), q)
        self.assertAlmostEqual(sum(v * v for v in q), 1.0, places=6)
        self.assertTrue(all(f32(v) == v for v in q))  # результат - float32
        half = _rotation(2, 90)
        self.assertTrue(all(abs(a - b) < 1e-6 for a, b in zip(qmul(half, half), (0, 0, 1, 0))))
        self.assertEqual(qnorm((0.0, 0.0, 0.0, 0.0)), (0.0, 0.0, 0.0, 0.0))

    def test_delta_small_and_large_angle(self):
        q, large = delta_axis_angle((0.01, -0.02, 0.005))
        self.assertFalse(large)
        self.assertEqual(q[:3], (f32(0.005), f32(-0.01), f32(0.0025)))
        q, large = delta_axis_angle((0.2, 0.0, 0.0))  # больше 5 градусов за шаг - через sin/cos
        self.assertTrue(large)
        self.assertAlmostEqual(q[0], math.sin(0.1), places=6)
        self.assertAlmostEqual(q[3], math.cos(0.1), places=6)

    def test_accel_shift(self):
        self.assertEqual(accel_shift(0.0, 0.0), IDENTITY)  # без поправки - точно единичный
        pitch, roll, _ = pitch_roll_yaw(accel_shift(2.0, 3.0))
        self.assertAlmostEqual(pitch, 2.0, places=2)
        self.assertAlmostEqual(roll, 3.0, places=2)

    def test_angle_conventions(self):
        # Как GetPitchRollYaw прошивки: тангаж - поворот вокруг X, крен - вокруг Y, рыскание -
        # против поворота вокруг Z, 0..360
        self.assertEqual(pitch_roll_yaw(IDENTITY), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(pitch_roll_yaw(_rotation(0, 10))[0], 10.0, places=6)
        self.assertAlmostEqual(pitch_roll_yaw(_rotation(1, 10))[1], 10.0, places=6)
        self.assertAlmostEqual(pitch_roll_yaw(_rotation(2, 30))[2], 330.0, places=6)


def _run(stage: OrientationStage, seconds: float, acc, gyr, per_tick: int = 4, t0: int = 0) -> int:
    t = t0
    for i in range(int(seconds * 400)):
        stage.sample(t, acc, gyr)
        t += STEP
        if i % per_tick == per_tick - 1:
            stage.tick()
    return t


class OrientationStageTest(unittest.TestCase):
    def test_level_and_still_stays_identity(self):
        stage = OrientationStage(LEVEL, TIMER)
        _run(stage, 1.0, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        self.assertEqual(stage.quat, IDENTITY)
        self.assertEqual(stage.gain, f32(0.01))  # покой: полная коррекция

    def test_gyro_integration(self):
        # 90 град/с вокруг Z в течение секунды: рыскание 90 градусов (по часовой - 270)
        stage = OrientationStage(LEVEL, TIMER)
        _run(stage, 1.0, (0.0, 0.0, 1.0), (0.0, 0.0, 90.0))
        self.assertAlmostEqual(pitch_roll_yaw(stage.quat)[2], 270.0, delta=0.01)
        self.assertEqual(stage.gain, 0.0)  # быстрое вращение: коррекция выключена
        self.assertEqual(stage.large_steps, 0)

    def test_accel_corrects_tilt(self):
        # Плата наклонена на 10 градусов вокруг X, гироскоп молчит: от единичного кватерниона
        # коррекция за 10 с сводит тангаж к наклону по акселерометру
        a = math.radians(10)
        stage = OrientationStage(LEVEL, TIMER)
        _run(stage, 10.0, (0.0, math.sin(a), math.cos(a)), (0.0, 0.0, 0.0))
        pitch, roll, _ = pitch_roll_yaw(stage.quat)
        self.assertAlmostEqual(pitch, 10.0, delta=0.05)
        self.assertAlmostEqual(roll, 0.0, delta=0.05)

    def test_correction_off_under_acceleration(self):
        stage = OrientationStage(LEVEL, TIMER)
        stage.seed(_rotation(0, 5))
        _run(stage, 1.0, (0.0, 0.0, 1.1), (0.0, 0.0, 0.0))  # |a| = 1.1 g: ускорение, а не вертикаль
        self.assertEqual(stage.gain, 0.0)
        self.assertAlmostEqual(pitch_roll_yaw(stage.quat)[0], 5.0, places=4)

    def test_gap_integrated_with_nominal_step(self):
        # Разрыв в секунду (опрос датчика стоял): шаг берётся 1/частота, а не секунда
        stage = OrientationStage(LEVEL, TIMER)
        stage.sample(0, (0.0, 0.0, 1.0), (100.0, 0.0, 0.0))  # первый отсчёт - тоже 1/частота
        stage.sample(TIMER, (0.0, 0.0, 1.0), (100.0, 0.0, 0.0))
        self.assertAlmostEqual(pitch_roll_yaw(stage.quat)[0], 2 * 100.0 / 400, places=4)

    def test_override_rate_for_sparse_source(self):
        # Отсчёты через 10 мс при номинале 400 Гц: без своей частоты каждый шаг - "разрыв"
        stage = OrientationStage(LEVEL, TIMER, sample_rate=100.0)
        for i in range(101):
            stage.sample(i * TIMER // 100, (0.0, 0.0, 1.0), (0.0, 0.0, 50.0))
        self.assertAlmostEqual(pitch_roll_yaw(stage.quat)[2], 360.0 - 50.0 - 0.5, delta=0.01)


def _firmware_stream(samples: int, drop: int | None = None, attach: int = 0) -> bytes:
    """Поток "прошивки", роль которой играют модели ПК: отсчёты с качанием вокруг Z, такт на
    каждый пятый отсчёт. drop - номер отсчёта, кадр которого не дошёл; attach - с какого отсчёта
    ПК подключился (тогда CONFIG с состоянием - на первом такте после подключения)."""
    params = ProcessingParams.from_config(FIRMWARE_DEFAULTS)
    stage = SensorStage(params)
    orient = OrientationStage(params, FIRMWARE_DEFAULTS.timer_freq)
    stream = bytearray() if attach else bytearray(encode(FIRMWARE_DEFAULTS))
    config_sent = not attach
    t = 0
    for i in range(samples):
        s = Sample(i, t, 0, (i % 7, 300, 4096), (20, -16, int(300 * math.sin(i / 50))))
        r = stage.process(s, 0)
        orient.sample(s.time, r.acc_g, r.gyr_dps)
        if i != drop and i >= attach:
            stream += encode(s)
        t += 2105
        if i % 5 == 4:
            orient.tick()
            if i >= attach:
                shift = tuple(stage.calibrator.shift)
                stream += encode(Tick(i // 5, t, 0, orient.quat, shift, orient.gain))
                if not config_sent:
                    stream += encode(replace(FIRMWARE_DEFAULTS, filter_state=stage.filter_state, gyro_shift=shift))
                    config_sent = True
    return bytes(stream)


def _pipeline_stats(stream: bytes, seed: bool):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "s.imus"
        path.write_bytes(stream)
        return Pipeline(RecordingSource(path), speed=0, seed=seed).run()


class PipelineOrientationTest(unittest.TestCase):
    def test_recorded_bit_exact_through_restart_and_reset(self):
        # Запись с перезапуском платы: секунда старого сеанса, старт с единичного кватерниона,
        # через секунду - его сброс (флаг QuatReset), смена частоты отсчётов
        stats = Pipeline(RecordingSource(DATA / "drone_boot_v2.imus"), speed=0).run()
        self.assertEqual(stats.restarts, 1)
        self.assertGreater(stats.quat_checks, 600)
        self.assertTrue(stats.orientation_exact, stats)

    def test_from_boot(self):
        stats = _pipeline_stats(_firmware_stream(3000), seed=False)
        self.assertEqual(stats.quat_checks, 600)
        self.assertTrue(stats.orientation_exact, stats)

    def test_lost_sample_resyncs(self):
        stats = _pipeline_stats(_firmware_stream(3000, drop=1502), seed=False)
        self.assertEqual(stats.lost_samples, 1)
        self.assertEqual(stats.quat_resyncs, 1)
        self.assertEqual(stats.quat_checks, 599)  # на такте подхвата не сверяется
        # Кватернион подхвачен, но фильтры PT2 на ПК этот отсчёт тоже не видели: остаётся след в
        # младшем разряде float32. Без подхвата пропущенный поворот остался бы навсегда (~1e-4)
        self.assertLess(stats.quat_diff_max, 1e-6)

    def test_attach_mid_stream(self):
        stats = _pipeline_stats(_firmware_stream(3000, attach=1000), seed=True)
        self.assertEqual(stats.quat_resyncs, 0)
        self.assertGreater(stats.quat_checks, 390)
        self.assertTrue(stats.orientation_exact, stats)


class CsvOrientationTest(unittest.TestCase):
    def test_csv_ticks_and_tilt(self):
        # В CSV нет тактов прошивки: конвейер отсчитывает их сам; неподвижная плата - наклоны
        # сходятся к наклону по акселерометру. Калибровка у CSV паспортная, |a| = 1.03 g, и
        # коррекция ослаблена до 0.004 за такт, а ноль гироскопа в первую секунду ещё не оценён и
        # уводит ориентацию: за 5 с остаётся до трети начальной ошибки
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "idle.csv"
            export_csv(DATA / "drone_idle_5s.imus", out)
            source = CsvSource(out)
            results, orientation = [], []
            Pipeline(source, speed=0, on_result=results.append, on_orientation=orientation.append).run()
        duration = results[-1].t - results[0].t
        self.assertAlmostEqual(len(orientation), duration * 100, delta=2)
        self.assertAlmostEqual(source.measured_rate, len(results) / duration, delta=1.0)
        ax, ay, az = results[-1].acc_g
        tilt = (math.degrees(math.atan(ay / math.hypot(ax, az))), math.degrees(math.atan(-ax / math.hypot(ay, az))))
        self.assertLess(max(map(abs, orientation[0].angles[:2])), 0.05)  # старт с единичного кватерниона
        for model, accel in zip(orientation[-1].angles, tilt):
            self.assertLess(abs(model - accel), 0.4 * abs(accel) + 0.05)


class ParamsTest(unittest.TestCase):
    def test_check_orientation_params(self):
        params = ProcessingParams.from_config(FIRMWARE_DEFAULTS)
        self.assertEqual(params.check(), [])
        bad = params.with_overrides({"quat_acc_alpha": 1.5, "quat_gyr_max": 0.0})
        self.assertEqual(len(bad.check()), 2)


if __name__ == "__main__":
    unittest.main()
