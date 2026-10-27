"""Блок дискретизации: обработанные отсчёты на равномерную сетку заданной частоты.

Отсчёты с дрона идут неравномерно: датчик опрашивается по I2C из таймера 500 Гц, и шаг между
готовыми отсчётами гуляет от 1.3 до 3.5 мс, а средняя частота - от 400 до 500 Гц. Расчётам на
постоянном такте (ориентация на 100 Гц, ЛР3) и спектральному анализу нужна равномерная сетка.

Два способа:
  hold   - на момент сетки берётся последний готовый отсчёт. Так делает такт 100 Гц прошивки
           (TimerUpdateMain читает последний DataIMU), поэтому это режим для повторения прошивки;
  linear - линейная интерполяция между соседними отсчётами: для графиков и спектра, где
           ступеньки удержания дают ложные гармоники.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .preprocess import SensorResult

MODES = ("hold", "linear")

_VECTOR_FIELDS = ("raw_acc", "raw_gyr", "acc_filtered", "gyr_filtered", "acc_g", "gyr_dps", "gyro_shift")


def _lerp(a: tuple, b: tuple, w: float) -> tuple:
    return tuple(x + (y - x) * w for x, y in zip(a, b))


@dataclass(slots=True)
class ResampleStats:
    inputs: int = 0
    outputs: int = 0
    skipped: int = 0          # точки сетки внутри разрывов длиннее max_gap
    dt_min: float = math.inf  # шаг входных отсчётов, с
    dt_max: float = 0.0
    _dt_sum: float = 0.0
    _dt_sq: float = 0.0

    def add_dt(self, dt: float) -> None:
        self.dt_min = min(self.dt_min, dt)
        self.dt_max = max(self.dt_max, dt)
        self._dt_sum += dt
        self._dt_sq += dt * dt

    @property
    def dt_mean(self) -> float:
        n = self.inputs - 1
        return self._dt_sum / n if n > 0 else 0.0

    @property
    def dt_std(self) -> float:
        n = self.inputs - 1
        if n <= 1:
            return 0.0
        mean = self._dt_sum / n
        return math.sqrt(max(0.0, self._dt_sq / n - mean * mean))

    @property
    def input_rate(self) -> float:
        return 1.0 / self.dt_mean if self.dt_mean else 0.0


class Resampler:
    """Сетка t_k = k / rate (время потока, от его начала). Каждая точка сетки выдаётся, как только
    известны отсчёты по обе стороны от неё, - то есть с задержкой не больше одного отсчёта.

    max_gap - разрыв между отсчётами (с), дольше которого точки сетки внутри него не выдаются, а
    считаются пропущенными: интерполировать через потерю связи или перезапуск платы нельзя.
    None - выдавать всегда (в hold так же, как прошивка держит последний отсчёт).
    """

    def __init__(self, rate: float, mode: str = "hold", max_gap: float | None = None) -> None:
        if rate <= 0:
            raise ValueError("частота сетки должна быть положительной")
        if mode not in MODES:
            raise ValueError(f"режим дискретизации: {', '.join(MODES)}")
        self.rate = rate
        self.mode = mode
        self.max_gap = max_gap
        self.stats = ResampleStats()
        self._prev: SensorResult | None = None
        self._k = 0

    def _grid(self, k: int) -> float:
        return k / self.rate

    def _point(self, t: float, a: SensorResult, b: SensorResult) -> SensorResult:
        if self.mode == "hold" or b.t == a.t:
            src = b if t >= b.t else a
            return SensorResult(t, *(getattr(src, f) for f in _VECTOR_FIELDS), src.calib_allowed)
        w = (t - a.t) / (b.t - a.t)
        return SensorResult(t, *(_lerp(getattr(a, f), getattr(b, f), w) for f in _VECTOR_FIELDS), a.calib_allowed)

    def push(self, result: SensorResult) -> list[SensorResult]:
        """Принять очередной отсчёт; вернуть точки сетки, которые теперь можно выдать."""
        self.stats.inputs += 1
        prev = self._prev
        out: list[SensorResult] = []
        if prev is None:
            self._k = math.ceil(result.t * self.rate - 1e-9)
            prev = result
        else:
            dt = result.t - prev.t
            if dt <= 0:
                return out  # отсчёт не новее предыдущего: сетке нечего добавить
            self.stats.add_dt(dt)
        gap = self.max_gap is not None and result.t - prev.t > self.max_gap
        while (t := self._grid(self._k)) <= result.t:
            if gap and t < result.t:
                self.stats.skipped += 1
            else:
                out.append(self._point(t, prev, result))
            self._k += 1
        self._prev = result
        self.stats.outputs += len(out)
        return out
