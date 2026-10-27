"""Синхронизация выдачи отсчётов записи с их собственным временем."""
from __future__ import annotations

import time
from collections.abc import Callable


class ReplayClock:
    """Выдаёт отсчёты записи в темпе, в котором они снимались, с множителем speed.

    speed = 1 - как в жизни, 2 - вдвое быстрее, 0 - без ожидания (обработка с максимальной
    скоростью, для сравнения параметров на всей записи). Опорная точка - первый отсчёт: если
    обработка отстала, догонять не приходится, ожидание просто становится нулевым.
    """

    def __init__(self, speed: float = 1.0,
                 now: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.speed = speed
        self._now = now
        self._sleep = sleep
        self._origin: tuple[float, float] | None = None  # (время отсчёта, время ПК)

    def wait_until(self, t: float) -> None:
        if self.speed <= 0:
            return
        if self._origin is None:
            self._origin = (t, self._now())
            return
        t0, wall0 = self._origin
        delay = wall0 + (t - t0) / self.speed - self._now()
        if delay > 0:
            self._sleep(delay)
