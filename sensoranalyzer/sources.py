"""Источники кадров: дрон по USB в реальном времени, запись потока, CSV с отсчётами."""
from __future__ import annotations

import csv
import math
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO

from .preprocess import Pt2Filter, f32
from .protocol import FLAG_CALIB_ALLOWED, TIME_MASK, Config, Frame, FrameParser, Sample

DEFAULT_BAUD = 1_000_000

# Настройки тракта прошивки по умолчанию (IIMU.h, ICM20948.cpp, CONFIG v2): номинал 475 Гц, точные
# срезы 25/100 Гц, tau нуля гироскопа 0.5 с, 16.4 ед./(град/с). CSV их не несёт, поэтому CsvSource
# начинает поток с такого кадра CONFIG; поправки установки и калибровка - паспортные
FIRMWARE_DEFAULTS = Config(
    version=2, timer_freq=1_000_000, sample_rate=475.0, acc_cut_freq=25.0, gyr_cut_freq=100.0,
    gyro_shift_alpha=f32(1.0 - math.exp(-1.0 / (0.5 * 475.0))), gyro_shift_limit=50.0, level_pitch=0.0,
    level_roll=0.0, acc_calib=(-4096, 4096, -4096, 4096, -4096, 4096),
    acc_k=f32(Pt2Filter.coefficient(25.0, 475.0)), gyr_k=f32(Pt2Filter.coefficient(100.0, 475.0)), gyro_scale=f32(16.4),
)


class SignalSource(ABC):
    """Источник кадров ImuStream в порядке их обработки в прошивке."""

    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        """Кадры по одному. Для источника реального времени - пока не вызван stop()."""

    def stop(self) -> None:
        """Прервать frames() из другого потока."""

    @property
    def realtime(self) -> bool:
        """Темп задаёт сам источник (порт), а не синхронизация по времени кадров."""
        return False

    @property
    def stats(self) -> dict[str, int]:
        return {}


class _ParsingSource(SignalSource):
    def __init__(self) -> None:
        self._parser = FrameParser()

    @property
    def stats(self) -> dict[str, int]:
        return {"crc_errors": self._parser.crc_errors, "skipped_bytes": self._parser.skipped_bytes}


class SerialSource(_ParsingSource):
    """Дрон по USB: канал B FT2232H отладочной платы, COM-порт, 1 Мбод.

    Если задан record_to, всё принятое пишется туда как есть - получается запись потока,
    которую потом читает RecordingSource.
    """

    def __init__(self, port: str, baud: int = DEFAULT_BAUD, record_to: BinaryIO | None = None) -> None:
        super().__init__()
        self.port = port
        self.baud = baud
        self._record_to = record_to
        self._stop = threading.Event()
        self.bytes_received = 0

    @property
    def realtime(self) -> bool:
        return True

    def frames(self) -> Iterator[Frame]:
        import serial  # только здесь: остальной код работает и без pyserial

        with serial.Serial(self.port, self.baud, timeout=0.05) as port:
            port.reset_input_buffer()
            while not self._stop.is_set():
                data = port.read(max(1, port.in_waiting))
                if not data:
                    continue
                self.bytes_received += len(data)
                if self._record_to is not None:
                    self._record_to.write(data)
                yield from self._parser.feed(data)

    def stop(self) -> None:
        self._stop.set()


class RecordingSource(_ParsingSource):
    """Запись потока (.imus) - байты ровно как они пришли из порта."""

    CHUNK = 64 * 1024

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self._stop = threading.Event()

    def frames(self) -> Iterator[Frame]:
        with open(self.path, "rb") as f:
            while not self._stop.is_set() and (data := f.read(self.CHUNK)):
                yield from self._parser.feed(data)

    def stop(self) -> None:
        self._stop.set()


CSV_RAW_COLUMNS = ("ax_raw", "ay_raw", "az_raw", "gx_raw", "gy_raw", "gz_raw")
CSV_TIME_COLUMNS = {"t_us": 1_000_000, "t_ms": 1_000}


class CsvSource(SignalSource):
    """CSV с отсчётами: время t_us или t_ms, коды ax_raw..gz_raw, необязательный calib_allowed.

    Подходит и для прежних логов (imu_log.csv), и для выгрузки записи (export_csv). В CSV нет
    тактов прошивки и её настроек: поток начинается с кадра CONFIG по умолчанию, а оценка нуля
    гироскопа без столбца calib_allowed считается разрешённой (дрон на земле).
    """

    def __init__(self, path: str | Path, config: Config = FIRMWARE_DEFAULTS) -> None:
        self.path = Path(path)
        self._config = config
        self._stop = threading.Event()

    def validate(self, max_errors: int = 20) -> list[str]:
        """Проверка формата. Пустой список - файл годен."""
        errors: list[str] = []
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            time_cols = [c for c in CSV_TIME_COLUMNS if c in header]
            if not time_cols:
                errors.append("нет столбца времени (t_us или t_ms)")
            missing = [c for c in CSV_RAW_COLUMNS if c not in header]
            if missing:
                errors.append("нет столбцов: " + ", ".join(missing))
            if errors:
                return errors
            last_t = None
            for line, row in enumerate(reader, start=2):
                try:
                    t = int(row[time_cols[0]])
                    values = [int(row[c]) for c in CSV_RAW_COLUMNS]
                except (TypeError, ValueError):
                    errors.append(f"строка {line}: не целое число")
                else:
                    if any(not -32768 <= v <= 32767 for v in values):
                        errors.append(f"строка {line}: код вне диапазона int16")
                    if last_t is not None and t <= last_t:
                        errors.append(f"строка {line}: время не возрастает")
                    last_t = t
                if len(errors) >= max_errors:
                    errors.append("дальше не проверялось")
                    break
            if last_t is None:
                errors.append("нет ни одного отсчёта")
        return errors

    def frames(self) -> Iterator[Frame]:
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            time_col = next(c for c in CSV_TIME_COLUMNS if c in header)
            timer_freq = CSV_TIME_COLUMNS[time_col]
            yield replace(self._config, timer_freq=timer_freq)
            has_flags = "calib_allowed" in header
            for seq, row in enumerate(reader):
                if self._stop.is_set():
                    return
                v = [int(row[c]) for c in CSV_RAW_COLUMNS]
                calib = int(row["calib_allowed"]) if has_flags else 1
                yield Sample(seq & 0xFFFF, int(row[time_col]) & TIME_MASK,
                             FLAG_CALIB_ALLOWED if calib else 0, (v[0], v[1], v[2]), (v[3], v[4], v[5]))

    def stop(self) -> None:
        self._stop.set()


def open_source(spec: str, record_to: BinaryIO | None = None) -> SignalSource:
    """Источник по строке: имя COM-порта, файл .imus или .csv."""
    if spec.upper().startswith("COM") or spec.startswith("/dev/"):
        return SerialSource(spec, record_to=record_to)
    path = Path(spec)
    if path.suffix.lower() == ".csv":
        return CsvSource(path)
    return RecordingSource(path)
