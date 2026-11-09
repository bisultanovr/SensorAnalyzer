"""Расчёт ориентации - бит в бит как IRS прошивки (app/INS/IRS.cpp, app/INS/geom/Quaternion.cpp).

Прошивка ведёт кватернион ориентации OriQuat двумя шагами:
- на каждом отсчёте IMU (DoneProcIMU -> UpdateGyroHigh) - поворот на угловую скорость за шаг
  между отсчётами. Шаг меряется по таймеру платы, угол за шаг мал, поэтому поворот - по формуле
  малого угла: dq = (w dt / 2, 1 - |w dt|^2 / 8);
- на такте 100 Гц (TimerUpdateMain -> UpdateGyro, UpdateAccel -> RestoreQuat) - коррекция по
  акселерометру: поворот, который сводит ожидаемое направление вертикали с измеренным, на долю
  gain = quat_acc_alpha * (1 - dyn) * (1 - |w|^2 / quat_gyr_max). dyn - насколько модуль
  ускорения отличается от 1 g, в долях quat_acc_max; w - средняя угловая скорость за такт, град/с.
  Пока плата движется с ускорением или быстро вращается, акселерометр показывает не вертикаль, и
  коррекция ослабевает до нуля.

Это комплементарный фильтр: гироскоп даёт быстрые изменения ориентации, акселерометр не даёт
накопиться дрейфу наклонов. Рыскание по акселерометру не корректируется, а коррекция по
магнитометру в прошивке отключена, поэтому рыскание дрейфует вместе с остаточным нулём гироскопа.

Арифметика - как в preprocess: каждая операция округляется до float32 в порядке вычисления C.
Библиотечные sinf/cosf прошивки нужны только в двух местах: поправка установки платы
(SetAccelShift, раз при старте) и поворот больше ~5 градусов за один отсчёт (около 2000 град/с при
400 Гц - предел шкалы гироскопа). Здесь они берутся из math с округлением до float32; если библиотека прошивки округляет
иначе, расхождение - в младшем разряде, и сверка с кадрами TICK его покажет.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .preprocess import ProcessingParams, f32

Quat = tuple[float, float, float, float]  # X Y Z W - порядок полей Quaternion прошивки
Vec3 = tuple[float, float, float]

IDENTITY: Quat = (0.0, 0.0, 0.0, 1.0)

TICK_RATE = 100.0  # IRS::Freq - такт коррекции, Гц

# Константы IRS.cpp и Quaternion.cpp, как их сворачивает компилятор во float
PI = f32(3.14159265359)
TO_RAD = f32(PI / 180.0)
SMALL_ANGLE_SQ = f32(0.0075)  # DeltaAxisAngleXYZ: квадрат угла за шаг, ниже которого - формула малого угла
QUAT_NORM_MIN = f32(1e-6)
VEC_NORM_MIN = f32(1e-12)
CROSS_MIN = f32(0.001)


# --- арифметика Quaternion и Vector3 прошивки ---------------------------------------------------
def qmul(a: Quat, b: Quat) -> Quat:
    """Quaternion::operator*(Quaternion): произведение Гамильтона."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        f32(f32(f32(f32(aw * bx) + f32(ax * bw)) + f32(ay * bz)) - f32(az * by)),
        f32(f32(f32(f32(aw * by) - f32(ax * bz)) + f32(ay * bw)) + f32(az * bx)),
        f32(f32(f32(f32(aw * bz) + f32(ax * by)) - f32(ay * bx)) + f32(az * bw)),
        f32(f32(f32(f32(aw * bw) - f32(ax * bx)) - f32(ay * by)) - f32(az * bz)),
    )


def qnorm(q: Quat) -> Quat:
    """Quaternion::Norm()."""
    x, y, z, w = q
    norm = f32(math.sqrt(f32(f32(f32(f32(x * x) + f32(y * y)) + f32(z * z)) + f32(w * w))))
    if norm > QUAT_NORM_MIN:
        norm = f32(1.0 / norm)
        return f32(x * norm), f32(y * norm), f32(z * norm), f32(w * norm)
    return 0.0, 0.0, 0.0, 0.0


def conjugate(q: Quat) -> Quat:
    return -q[0], -q[1], -q[2], q[3]


def length_sq(v: Vec3) -> float:
    """Vector3::LengthSquared()."""
    return f32(f32(f32(v[0] * v[0]) + f32(v[1] * v[1])) + f32(v[2] * v[2]))


def length(v: Vec3) -> float:
    """Vector3::Length()."""
    return f32(math.sqrt(length_sq(v)))


def vnorm(v: Vec3) -> Vec3:
    """Vector3::Norm()."""
    n = length(v)
    if n > VEC_NORM_MIN:
        n = f32(1.0 / n)
        return f32(v[0] * n), f32(v[1] * n), f32(v[2] * n)
    return 0.0, 0.0, 0.0


def scale(v: Vec3, k: float) -> Vec3:
    return f32(v[0] * k), f32(v[1] * k), f32(v[2] * k)


def delta_axis_angle(a: Vec3) -> tuple[Quat, bool]:
    """Quaternion::DeltaAxisAngleXYZ: поворот на вектор a (рад) в осях платы. Второе значение -
    True, если понадобились sinf/cosf (угол за шаг больше ~5 градусов)."""
    x, y, z = a
    a_sq = length_sq(a)
    if a_sq < SMALL_ANGLE_SQ:
        return (f32(0.5 * x), f32(0.5 * y), f32(0.5 * z), f32(1.0 - f32(0.125 * a_sq))), False
    angle = f32(math.sqrt(a_sq))
    half = f32(0.5 * angle)
    factor = f32(f32(math.sin(half)) / angle)
    return (f32(x * factor), f32(y * factor), f32(z * factor), f32(math.cos(half))), True


def accel_shift(level_pitch: float, level_roll: float) -> Quat:
    """IRS::SetAccelShift(Pitch, Roll, 0): поправка установки платы (LevelHor), град."""
    h_roll = f32(f32(f32(level_roll) * TO_RAD) / 2.0)
    h_pitch = f32(f32(f32(level_pitch) * TO_RAD) / 2.0)
    q_roll = (0.0, f32(math.sin(h_roll)), 0.0, f32(math.cos(h_roll)))
    q_pitch = (f32(math.sin(h_pitch)), 0.0, 0.0, f32(math.cos(h_pitch)))
    q_yaw = IDENTITY  # Yaw = 0: sinf(0) и cosf(0) точны
    return qnorm(qmul(qmul(q_pitch, q_roll), q_yaw))


# --- углы для показа ------------------------------------------------------------------------------
def _qmul64(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _angle(x: float, y: float, z: float) -> float:
    """IRS::GetAngle: угол вектора к плоскости YZ, град."""
    if y == 0.0 and z == 0.0:
        return math.copysign(90.0, x) if x else 0.0
    return math.degrees(math.atan(x / math.hypot(y, z)))


def pitch_roll_yaw(q: Quat) -> tuple[float, float, float]:
    """Тангаж, крен, рыскание в градусах - по определениям IRS::GetPitchRollYaw, но в double:
    это показ, а не сверка. Рыскание - 0..360, по часовой стрелке, если смотреть сверху."""
    x, y, z, w = q
    prt = _qmul64(_qmul64((-x, -y, -z, w), (0.0, 0.0, 1.0, 0.0)), q)  # вертикаль в осях платы
    yaw = math.degrees(math.atan2(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z)))
    if yaw < 0.0:
        yaw += 360.0
    return _angle(prt[1], prt[0], prt[2]), _angle(-prt[0], prt[1], prt[2]), yaw


# --- блок ориентации --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OrientationResult:
    """Ориентация после такта 100 Гц."""
    t: float           # с от начала потока
    quat: Quat         # X Y Z W
    gain: float        # коэффициент коррекции по акселерометру в этом такте (LastGain)
    gyr_avg: Vec3      # средняя угловая скорость за такт, град/с (IMU.Gyr)
    samples: int       # отсчётов, проинтегрированных за такт
    acc: Vec3          # ускорение, по которому шла коррекция, g (до поправки установки)

    @property
    def angles(self) -> tuple[float, float, float]:
        return pitch_roll_yaw(self.quat)


class OrientationStage:
    """Кватернион ориентации и всё, от чего он зависит в прошивке.

    sample() - на каждом обработанном отсчёте (как DoneProcIMU), tick() - на такте 100 Гц (как
    TimerUpdateMain). Время отсчёта - сырое время таймера платы из кадра: шаг интегрирования
    прошивка считает из него во float, и здесь так же.

    sample_rate - частота отсчётов для правила разрыва, если она не та, что в параметрах тракта:
    у CSV в параметрах номинал прошивки, а отсчёты могут идти реже (imu_log.csv - 100 Гц). Без
    этого каждый шаг CSV длиннее 4 / 475 с считался бы разрывом и интегрировался шагом 1 / 475 с.
    """

    def __init__(self, params: ProcessingParams, timer_freq: int, sample_rate: float | None = None) -> None:
        self.quat: Quat = IDENTITY
        self.gain = 0.0
        self.gyr_avg: Vec3 = (0.0, 0.0, 0.0)
        self.tick_samples = 0
        self.large_steps = 0  # повороты больше ~5 градусов за отсчёт: шли через sinf/cosf
        self._gyr_sum: Vec3 = (0.0, 0.0, 0.0)
        self._gyr_count = 0
        self._acc: Vec3 = (0.0, 0.0, 0.0)  # DataIMU.Acc - ускорение последнего отсчёта, g
        self._last_time: int | None = None
        self._timer_freq = f32(timer_freq)
        self._rate_override = sample_rate
        self.set_params(params)

    def set_params(self, params: ProcessingParams) -> None:
        """Новые параметры: кватернион сохраняется. Частота отсчётов здесь - только для шага
        первого отсчёта и шага после разрыва потока, как IMU_GetSampleRate в DoneProcIMU."""
        self._rate = f32(self._rate_override or params.sample_rate)
        self._shift = accel_shift(params.level_pitch, params.level_roll)
        self._alpha = f32(params.quat_acc_alpha)
        self._acc_max = f32(params.quat_acc_max)
        self._gyr_max = f32(params.quat_gyr_max)

    def seed(self, quat: Quat, gain: float = 0.0) -> None:
        """Продолжить с кватерниона прошивки из кадра TICK: дальше модель повторяет её шаг в шаг."""
        self.quat = tuple(quat)
        self.gain = gain
        self._gyr_sum, self._gyr_count = (0.0, 0.0, 0.0), 0

    # DoneProcIMU
    def sample(self, raw_time: int, acc_g: Vec3, gyr_dps: Vec3) -> bool:
        """Отсчёт. True - поворот за шаг больше ~5 градусов (считался через sinf/cosf)."""
        if self._last_time is None:
            period = f32(1.0 / self._rate)
        else:
            ticks = (raw_time - self._last_time) & 0xFFFFFFFF  # время в кадре - младшие 32 бита
            period = f32(f32(float(ticks)) / self._timer_freq)
            if period > f32(4.0 / self._rate):  # разрыв дольше нескольких отсчётов
                period = f32(1.0 / self._rate)
        self._last_time = raw_time
        large = self._update_gyro_high(gyr_dps, period)
        self._acc = acc_g
        return large

    def _update_gyro_high(self, gyr: Vec3, period: float) -> bool:
        if not all(math.isfinite(v) for v in gyr):
            return False
        s = self._gyr_sum
        self._gyr_sum = (f32(s[0] + gyr[0]), f32(s[1] + gyr[1]), f32(s[2] + gyr[2]))
        self._gyr_count += 1
        delta, large = delta_axis_angle(scale(gyr, f32(period * TO_RAD)))
        self.large_steps += large
        self.quat = qnorm(qmul(self.quat, delta))
        return large

    # TimerUpdateMain
    def tick(self, reset: bool = False) -> None:
        """Такт 100 Гц. reset - прошивка в этом такте сбросила кватернион в единичный (флаг
        QuatReset кадра TICK: через секунду после включения)."""
        if reset:
            self.quat = IDENTITY
        self._update_gyro()
        self._update_accel(self._acc)

    def result(self, t: float) -> OrientationResult:
        return OrientationResult(t, self.quat, self.gain, self.gyr_avg, self.tick_samples, self._acc)

    def _update_gyro(self) -> None:
        count = self._gyr_count
        s = self._gyr_sum
        self._gyr_sum, self._gyr_count = (0.0, 0.0, 0.0), 0
        self.tick_samples = count
        if count == 0:  # в прошивке 0/0 - NaN, и IMU.Gyr остаётся прежним
            return
        n = f32(float(count))
        self.gyr_avg = (f32(s[0] / n), f32(s[1] / n), f32(s[2] / n))

    def _update_accel(self, acc: Vec3) -> None:
        if not all(math.isfinite(v) for v in acc):
            return
        accel = qmul(qmul(self._shift, (acc[0], acc[1], acc[2], 0.0)), conjugate(self._shift))
        self._restore_quat((accel[0], accel[1], accel[2]))

    def _restore_quat(self, acc: Vec3) -> None:
        """IRS::RestoreQuat: коррекция по акселерометру."""
        dyn = f32(abs(f32(length(acc) - 1.0)))
        dyn = 1.0 if dyn > self._acc_max else f32(dyn / self._acc_max)
        gain = f32(self._alpha * f32(1.0 - dyn))

        gyr = length_sq(self.gyr_avg)
        if gyr > self._gyr_max:
            gyr = self._gyr_max
        gyr = f32(1.0 - f32(gyr / self._gyr_max))
        gain = f32(gain * gyr)
        self.gain = gain

        ax, ay, az = vnorm(acc)
        qx, qy, qz, qw = self.quat
        # Куда должна смотреть вертикаль в осях платы при текущем кватернионе
        ex = f32(2.0 * f32(f32(qx * qz) - f32(qw * qy)))
        ey = f32(2.0 * f32(f32(qw * qx) + f32(qy * qz)))
        ez = f32(1.0 - f32(2.0 * f32(f32(qx * qx) + f32(qy * qy))))

        cross = (f32(f32(ay * ez) - f32(az * ey)),
                 f32(f32(az * ex) - f32(ax * ez)),
                 f32(f32(ax * ey) - f32(ay * ex)))
        dot = f32(f32(f32(ax * ex) + f32(ay * ey)) + f32(az * ez))
        if dot < 0.0:  # ошибка больше 90 градусов: поворачивать на полную, по оси ошибки
            error = length(cross)
            cross = (1.0, 0.0, 0.0) if error < CROSS_MIN else scale(cross, f32(1.0 / error))

        axis = scale(cross, f32(gain * 0.5))
        self.quat = qnorm(qmul(self.quat, (axis[0], axis[1], axis[2], 1.0)))
