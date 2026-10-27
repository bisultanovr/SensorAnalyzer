"""Предварительная обработка отсчёта - бит в бит как IMU_CallbackProc прошивки (ICM20948.cpp).

Порядок: фильтр PT2 по каждому из 6 каналов -> ускорение в g по калибровке -> оценка нуля
гироскопа (только пока она разрешена прошивкой) -> угловая скорость в град/с за вычетом нуля.

Прошивка считает в float32 и собрана без оптимизации, поэтому каждая операция в ней - отдельное
округление до float32, без слияния умножения со сложением. Здесь так же: после каждой операции
результат округляется до float32 (f32), в том же порядке, что в C. Для +, -, *, / операндов
float32 вычисление в double с последующим округлением даёт ровно результат float32. Что
повторить нельзя - sinf/expf/sqrtf библиотеки прошивки, - то прошивка присылает готовым
(коэффициенты фильтров и alpha в CONFIG v2).
"""
from __future__ import annotations

import cmath
import math
import struct
from dataclasses import dataclass, fields, replace

from .protocol import Config, Sample

_F32 = struct.Struct("<f")


def f32(x: float) -> float:
    """Округлить до float32."""
    return _F32.unpack(_F32.pack(x))[0]


PT2_CUT_CORRECTION = 1.553774  # 1 / sqrt(2^(1/2) - 1): срез каскада из двух PT1 на заданной частоте
GYRO_LSB_PER_DPS_V1 = 16.384   # прошивка до CONFIG v2: 32768 / 2000

# С этой версии CONFIG: срезы PT2 точные для дискретного фильтра (pt2FilterCutoffK), SampleRate -
# измеренная частота, гироскоп 16.4 ед./(град/с), в кадре - коэффициенты и состояние обработки.
# До неё - формула RC, номинал 475 Гц и 16.384
CONFIG_EXACT_CUTOFF_VERSION = 2


def gyro_shift_alpha(tau: float, sample_rate: float) -> float:
    """Шаг экспоненциального среднего с постоянной времени tau при отсчётах sample_rate."""
    return 1.0 - math.exp(-1.0 / (tau * sample_rate))


def gyro_shift_tau(alpha: float, sample_rate: float) -> float:
    return -1.0 / (sample_rate * math.log(1.0 - alpha))


# Параметры, от которых зависят коэффициенты фильтров: если меняется любой, присланные
# прошивкой коэффициенты больше не годятся и считаются заново
_FILTER_INPUTS = {"sample_rate", "acc_cut_freq", "gyr_cut_freq", "exact_cutoff"}


@dataclass(frozen=True, slots=True)
class ProcessingParams:
    """Параметры тракта, от которых зависит результат. Начальные значения приходят из прошивки
    (кадр CONFIG), любое можно переопределить, чтобы сравнить с прошивкой.

    Частота отсчётов, срезы и постоянная времени нуля связаны: срезы и tau - то, что задано, а
    коэффициенты фильтров и alpha выводятся из них при текущей частоте. Поэтому при смене
    sample_rate или gyro_shift_tau alpha пересчитывается, если её не задали явно, а при смене
    частоты или срезов - коэффициенты фильтров.
    """
    sample_rate: float       # частота отсчётов, под которую считаются фильтры, Гц
    acc_cut_freq: float      # срез PT2 ускорения, Гц
    gyr_cut_freq: float      # срез PT2 угловой скорости, Гц
    exact_cutoff: bool       # срез -3 дБ ровно на заданной частоте (прошивка с CONFIG v2)
    gyro_shift_tau: float    # постоянная времени оценки нуля гироскопа, с
    gyro_shift_alpha: float  # её шаг при sample_rate - как его считает прошивка
    gyro_shift_limit: float  # порог неподвижности для оценки нуля, ед. АЦП
    gyro_scale: float        # чувствительность гироскопа, ед./(град/с)
    acc_calib: tuple[int, int, int, int, int, int]  # X min, X max, Y min, Y max, Z min, Z max
    level_pitch: float       # поправка установки платы, град (нужна ориентации, ЛР3)
    level_roll: float
    acc_k: float | None = None  # коэффициенты PT2, как их посчитала прошивка; None - считать здесь
    gyr_k: float | None = None

    @classmethod
    def from_config(cls, config: Config) -> ProcessingParams:
        values = {f.name: getattr(config, f.name) for f in fields(cls) if hasattr(config, f.name)}
        values["exact_cutoff"] = config.version >= CONFIG_EXACT_CUTOFF_VERSION
        values["gyro_shift_tau"] = gyro_shift_tau(config.gyro_shift_alpha, config.sample_rate)
        if values.get("gyro_scale") is None:
            values["gyro_scale"] = GYRO_LSB_PER_DPS_V1
        return cls(**values)

    def with_overrides(self, overrides: dict[str, object]) -> ProcessingParams:
        unknown = set(overrides) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError("неизвестные параметры: " + ", ".join(sorted(unknown)))
        params = replace(self, **overrides)
        if _FILTER_INPUTS & overrides.keys():
            params = replace(params, acc_k=overrides.get("acc_k"), gyr_k=overrides.get("gyr_k"))
        if {"sample_rate", "gyro_shift_tau"} & overrides.keys() and "gyro_shift_alpha" not in overrides:
            params = replace(params, gyro_shift_alpha=gyro_shift_alpha(params.gyro_shift_tau, params.sample_rate))
        elif "gyro_shift_alpha" in overrides:
            params = replace(params, gyro_shift_tau=gyro_shift_tau(params.gyro_shift_alpha, params.sample_rate))
        return params

    def filter_k(self) -> tuple[float, float]:
        """Коэффициенты PT2 ускорения и угловой скорости: присланные прошивкой или посчитанные."""
        acc = self.acc_k if self.acc_k is not None else Pt2Filter.coefficient(self.acc_cut_freq, self.sample_rate, self.exact_cutoff)
        gyr = self.gyr_k if self.gyr_k is not None else Pt2Filter.coefficient(self.gyr_cut_freq, self.sample_rate, self.exact_cutoff)
        return f32(acc), f32(gyr)

    def check(self) -> list[str]:
        """Параметры, при которых тракт теряет смысл."""
        errors = []
        nyquist = self.sample_rate / 2
        for name in ("acc_cut_freq", "gyr_cut_freq"):
            value = getattr(self, name)
            if not 0 < value < nyquist:
                errors.append(f"{name} = {value}: должен быть в (0, {nyquist})")
        if not 0 < self.gyro_shift_alpha < 1:
            errors.append(f"gyro_shift_alpha = {self.gyro_shift_alpha}: должен быть в (0, 1)")
        if self.gyro_shift_tau <= 0:
            errors.append(f"gyro_shift_tau = {self.gyro_shift_tau}: должна быть положительной")
        if self.gyro_shift_limit < 0:
            errors.append(f"gyro_shift_limit = {self.gyro_shift_limit}: не может быть отрицательным")
        if self.gyro_scale <= 0:
            errors.append(f"gyro_scale = {self.gyro_scale}: должна быть положительной")
        c = self.acc_calib
        if not (c[0] < c[1] and c[2] < c[3] and c[4] < c[5]):
            errors.append(f"acc_calib = {c}: min должен быть меньше max по каждой оси")
        return errors


class Pt2Filter:
    """Два одинаковых звена PT1 подряд (FilterPT2.cpp). Звено: y += k (x - y), то есть
    H1(z) = k / (1 - (1 - k) z^-1); каскад - H1^2."""

    def __init__(self, k: float, state1: float = 0.0, state2: float = 0.0) -> None:
        self.k = k
        self.state1 = state1
        self.state2 = state2

    @staticmethod
    def coefficient(cut_freq: float, sample_rate: float, exact: bool = True) -> float:
        """k для среза cut_freq.

        exact = False - pt2FilterInit, повторённый в float32: формула RC, поправка 1.553774
        выведена для непрерывного фильтра, и дискретный срезает ниже заданного - на 475 Гц
        вместо 100 Гц около 59 Гц, вместо 25 Гц около 20 Гц.

        exact = True - как pt2FilterCutoffK: срез -3 дБ ровно на cut_freq. |H1|^2 = k^2 / (b - b k + k^2),
        b = 4 sin^2(pi fc / fs); условие каскада |H1|^4 = 1/2 даёт (sqrt(2) - 1) k^2 + b k - b = 0.
        Прошивка считает его через sinf/sqrtf своей библиотеки, поэтому её k может отличаться от
        этого на единицу младшего разряда float32; для сверки берётся k из CONFIG.
        """
        if not exact:
            adjusted = f32(f32(cut_freq) * f32(PT2_CUT_CORRECTION))
            dt = f32(1.0 / f32(sample_rate))
            rc = f32(1.0 / f32(f32(2.0 * f32(math.pi)) * adjusted))
            return f32(dt / f32(rc + dt))
        q = math.sqrt(2.0) - 1.0
        b = 4.0 * math.sin(math.pi * cut_freq / sample_rate) ** 2
        return (-b + math.sqrt(b * b + 4.0 * q * b)) / (2.0 * q)

    @classmethod
    def magnitude(cls, cut_freq: float, sample_rate: float, f: float, exact: bool = True) -> float:
        """АЧХ каскада на частоте f, в разах."""
        k = cls.coefficient(cut_freq, sample_rate, exact)
        z = cmath.exp(2j * math.pi * f / sample_rate)
        return abs(k / (1 - (1 - k) / z)) ** 2

    def apply(self, x: float) -> float:
        k = self.k
        self.state1 = f32(self.state1 + f32(k * f32(x - self.state1)))
        self.state2 = f32(self.state2 + f32(k * f32(self.state1 - self.state2)))
        return self.state2


class GyroZeroCalibrator:
    """Ноль гироскопа: экспоненциальное среднее, пока плата не вращается (IMU_Calibrate)."""

    def __init__(self, alpha: float, limit: float) -> None:
        self.alpha = f32(alpha)
        self.limit = f32(limit)
        self.shift = [0.0, 0.0, 0.0]

    def update(self, gyr: list[float]) -> None:
        g0, g1, g2 = gyr
        length = f32(f32(f32(g0 * g0) + f32(g1 * g1)) + f32(g2 * g2))
        if length > f32(self.limit * self.limit):
            return
        a = self.alpha
        keep = f32(1.0 - a)
        self.shift = [f32(f32(s * keep) + f32(a * g)) for s, g in zip(self.shift, gyr)]


def normalize(value: float, lo: int, hi: int) -> float:
    """Код в доли полной шкалы по калибровке min/max (Normalize в MathFunc.cpp, Scale = 1)."""
    half = f32(f32(hi - lo) / 2.0)
    middle = f32(f32(hi + lo) / 2.0)
    return f32(f32(value - middle) / half)


@dataclass(frozen=True, slots=True)
class SensorResult:
    t: float                               # с от начала потока
    raw_acc: tuple[int, int, int]           # коды датчика
    raw_gyr: tuple[int, int, int]
    acc_filtered: tuple[float, float, float]  # после PT2, ед. АЦП
    gyr_filtered: tuple[float, float, float]
    acc_g: tuple[float, float, float]       # ускорение, g
    gyr_dps: tuple[float, float, float]     # угловая скорость за вычетом нуля, град/с
    gyro_shift: tuple[float, float, float]  # оценка нуля, ед. АЦП
    calib_allowed: bool


class SensorStage:
    """Обработка одного отсчёта."""

    def __init__(self, params: ProcessingParams) -> None:
        self.params = params
        acc_k, gyr_k = params.filter_k()
        self._acc = [Pt2Filter(acc_k) for _ in range(3)]
        self._gyr = [Pt2Filter(gyr_k) for _ in range(3)]
        self._gyro_scale = f32(params.gyro_scale)
        self.calibrator = GyroZeroCalibrator(params.gyro_shift_alpha, params.gyro_shift_limit)

    def set_params(self, params: ProcessingParams) -> None:
        """Новые параметры на ходу: состояние фильтров и оценка нуля сохраняются - так же
        прошивка пересчитывает коэффициенты при смене частоты отсчётов (IMU_SetSampleRate)."""
        acc_k, gyr_k = params.filter_k()
        for f in self._acc:
            f.k = acc_k
        for f in self._gyr:
            f.k = gyr_k
        self._gyro_scale = f32(params.gyro_scale)
        self.calibrator.alpha = f32(params.gyro_shift_alpha)
        self.calibrator.limit = f32(params.gyro_shift_limit)
        self.params = params

    @property
    def filter_state(self) -> tuple[float, ...]:
        """Как FilterState в CONFIG: ускорение X Y Z, угловая скорость X Y Z, у каждого state1, state2."""
        return tuple(v for f in self._acc + self._gyr for v in (f.state1, f.state2))

    def seed_state(self, filter_state: tuple[float, ...], gyro_shift: tuple[float, float, float]) -> None:
        """Продолжить с состояния прошивки из CONFIG v2: дальше модель совпадает с ней бит в бит."""
        for i, f in enumerate(self._acc + self._gyr):
            f.state1, f.state2 = filter_state[2 * i], filter_state[2 * i + 1]
        self.calibrator.shift = list(gyro_shift)

    def seed(self, sample: Sample | None = None, gyro_shift: tuple[float, float, float] | None = None) -> None:
        """Подхватить поток с середины, когда состояния прошивки нет (CONFIG v1): фильтры - с
        текущего отсчёта, ноль - как у прошивки. Совпадение тогда приближённое: полсекунды
        после подхвата модель догоняет фильтры."""
        if sample is not None:
            for f, v in zip(self._acc + self._gyr, sample.acc + sample.gyr):
                f.state1 = f.state2 = float(v)
        if gyro_shift is not None:
            self.calibrator.shift = list(gyro_shift)

    def process(self, sample: Sample, t: float) -> SensorResult:
        acc_f = [f.apply(v) for f, v in zip(self._acc, sample.acc)]
        gyr_f = [f.apply(v) for f, v in zip(self._gyr, sample.gyr)]

        c = self.params.acc_calib
        acc_g = (normalize(acc_f[0], c[0], c[1]), normalize(acc_f[1], c[2], c[3]), normalize(acc_f[2], c[4], c[5]))

        if sample.calib_allowed:
            self.calibrator.update(gyr_f)
        shift = self.calibrator.shift
        scale = self._gyro_scale
        gyr_dps = tuple(f32(f32(g - s) / scale) for g, s in zip(gyr_f, shift))

        return SensorResult(t, sample.acc, sample.gyr, tuple(acc_f), tuple(gyr_f), acc_g, gyr_dps,
                            tuple(shift), sample.calib_allowed)
