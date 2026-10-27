"""Выгрузка записи потока в CSV, который потом читает CsvSource."""
from __future__ import annotations

import csv
from pathlib import Path

from .protocol import Config, Sample, StreamTime
from .sources import CSV_RAW_COLUMNS, RecordingSource


def export_csv(recording: str | Path, out: str | Path) -> int:
    """Отсчёты записи в CSV: t_us от начала записи, флаг оценки нуля, коды датчика.

    Время округляется до микросекунды. Для сравнения с прошивкой точнее сама запись .imus:
    в ней время в тактах mtimer и ещё такты прошивки с её кватернионом.
    """
    source = RecordingSource(recording)
    clock: StreamTime | None = None
    count = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("t_us", "calib_allowed", *CSV_RAW_COLUMNS))
        for frame in source.frames():
            if isinstance(frame, Config) and clock is None:
                clock = StreamTime(frame.timer_freq)
            elif isinstance(frame, Sample) and clock is not None:
                t_us = round(clock.seconds(frame.time) * 1_000_000)
                writer.writerow((t_us, int(frame.calib_allowed), *frame.acc, *frame.gyr))
                count += 1
    return count
