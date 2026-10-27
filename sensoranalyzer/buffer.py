"""Кольцевой буфер между потоком чтения источника и обработкой."""
from __future__ import annotations

import threading
from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """Буфер фиксированной ёмкости для одного писателя и одного читателя.

    Источник реального времени ждать не умеет: порт продолжает слать данные, пока обработка
    занята. Поэтому при переполнении теряются самые старые элементы, а не новые, и каждая
    потеря считается - по счётчику видно, что обработка не успевает. Файл ждать может: с
    block=True писатель при полном буфере ждёт, пока читатель освободит место.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("ёмкость буфера должна быть положительной")
        self.capacity = capacity
        self._items: deque[T] = deque()
        self._cond = threading.Condition()
        self._closed = False
        self.dropped = 0
        self.max_fill = 0

    def push(self, item: T, block: bool = False) -> None:
        with self._cond:
            if block:
                self._cond.wait_for(lambda: len(self._items) < self.capacity or self._closed)
            if self._closed:
                return
            if len(self._items) >= self.capacity:
                self._items.popleft()
                self.dropped += 1
            self._items.append(item)
            self.max_fill = max(self.max_fill, len(self._items))
            self._cond.notify_all()

    def pop(self, timeout: float | None = None) -> T | None:
        """Следующий элемент; None - если буфер закрыт и пуст или истёк таймаут."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._items or self._closed, timeout):
                return None
            if not self._items:
                return None
            item = self._items.popleft()
            self._cond.notify_all()  # писатель мог ждать места (push с block=True)
            return item

    def close(self) -> None:
        """Писатель закончил: читатель доберёт остаток и получит None."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)
