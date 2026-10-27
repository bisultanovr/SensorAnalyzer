"""Протокол ImuStream: кадры, которые прошивка дрона шлёт в UART0.

Описание кадров - в прошивке, app/extend/ImuStream.h. Кратко: 0xA5 0x5A, тип, длина полезной
части, полезная часть, CRC16-CCITT (0x1021, начало 0xFFFF) по типу, длине и полезной части,
младшим байтом вперёд. Поля little-endian без выравнивания.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

SYNC = b"\xA5\x5A"

TYPE_SAMPLE = 1
TYPE_TICK = 2
TYPE_CONFIG = 3

FLAG_CALIB_ALLOWED = 1 << 0  # идёт оценка нуля гироскопа
FLAG_QUAT_RESET = 1 << 1     # TICK: в этом такте кватернион сброшен в единичный
FLAG_ARMED = 1 << 2          # TICK: моторы включены

_SAMPLE = struct.Struct("<HIB3h3h")
_TICK = struct.Struct("<HIB4f3ff")
_CONFIG_V1 = struct.Struct("<BI7f6h")
_CONFIG_V2 = struct.Struct("<BI7f6h3f12f3f")  # + коэффициенты, чувствительность, состояние

# Допустимые длины полезной части: CONFIG версии 1 и 2 различаются по длине
PAYLOAD_SIZES = {TYPE_SAMPLE: {_SAMPLE.size}, TYPE_TICK: {_TICK.size}, TYPE_CONFIG: {_CONFIG_V1.size, _CONFIG_V2.size}}

TIME_MASK = 0xFFFFFFFF  # время в кадрах - младшие 32 бита mtimer


@dataclass(frozen=True, slots=True)
class Sample:
    """Отсчёт IMU: коды датчика после перестановки байт, до фильтров."""
    seq: int
    time: int
    flags: int
    acc: tuple[int, int, int]
    gyr: tuple[int, int, int]

    @property
    def calib_allowed(self) -> bool:
        return bool(self.flags & FLAG_CALIB_ALLOWED)


@dataclass(frozen=True, slots=True)
class Tick:
    """Такт 100 Гц: состояние прошивки после обновления ориентации."""
    seq: int
    time: int
    flags: int
    quat: tuple[float, float, float, float]  # X Y Z W
    gyro_shift: tuple[float, float, float]   # ноль гироскопа, ед. АЦП
    gain: float                              # коэффициент последней коррекции по акселерометру


@dataclass(frozen=True, slots=True)
class Config:
    """Настройки тракта прошивки, раз в секунду и при смене частоты отсчётов.

    Поля с acc_k - с версии 2. filter_state и gyro_shift - состояние обработки на момент кадра,
    а не настройки, поэтому в сравнении кадров не участвуют: два CONFIG равны, если настройки
    те же.
    """
    version: int
    timer_freq: int
    sample_rate: float
    acc_cut_freq: float
    gyr_cut_freq: float
    gyro_shift_alpha: float
    gyro_shift_limit: float
    level_pitch: float
    level_roll: float
    acc_calib: tuple[int, int, int, int, int, int]  # X min, X max, Y min, Y max, Z min, Z max
    acc_k: float | None = None       # k фильтров PT2, как их посчитала прошивка
    gyr_k: float | None = None
    gyro_scale: float | None = None  # чувствительность гироскопа, ед./(град/с)
    filter_state: tuple[float, ...] | None = field(default=None, compare=False)  # 12: acc XYZ, gyr XYZ по (state1, state2)
    gyro_shift: tuple[float, float, float] | None = field(default=None, compare=False)


Frame = Sample | Tick | Config


def _crc_table() -> list[int]:
    table = []
    for a in range(256):
        crc = a << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
        table.append(crc & 0xFFFF)
    return table


_CRC_TABLE = _crc_table()


def crc16(data: bytes | bytearray | memoryview) -> int:
    crc = 0xFFFF
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[(crc >> 8) ^ b]
    return crc


def decode(frame_type: int, payload: bytes) -> Frame:
    if frame_type == TYPE_SAMPLE:
        seq, time, flags, ax, ay, az, gx, gy, gz = _SAMPLE.unpack(payload)
        return Sample(seq, time, flags, (ax, ay, az), (gx, gy, gz))
    if frame_type == TYPE_TICK:
        seq, time, flags, qx, qy, qz, qw, sx, sy, sz, gain = _TICK.unpack(payload)
        return Tick(seq, time, flags, (qx, qy, qz, qw), (sx, sy, sz), gain)
    if frame_type == TYPE_CONFIG:
        if len(payload) == _CONFIG_V1.size:
            version, freq, rate, fca, fcg, alpha, limit, pitch, roll, *calib = _CONFIG_V1.unpack(payload)
            return Config(version, freq, rate, fca, fcg, alpha, limit, pitch, roll, tuple(calib))
        v = _CONFIG_V2.unpack(payload)
        return Config(*v[:9], tuple(v[9:15]), v[15], v[16], v[17], tuple(v[18:30]), tuple(v[30:33]))
    raise ValueError(f"неизвестный тип кадра {frame_type}")


def encode(frame: Frame) -> bytes:
    """Собрать кадр так же, как прошивка. Нужен для тестов и синтетических данных."""
    if isinstance(frame, Sample):
        frame_type = TYPE_SAMPLE
        payload = _SAMPLE.pack(frame.seq, frame.time, frame.flags, *frame.acc, *frame.gyr)
    elif isinstance(frame, Tick):
        frame_type = TYPE_TICK
        payload = _TICK.pack(frame.seq, frame.time, frame.flags, *frame.quat, *frame.gyro_shift, frame.gain)
    else:
        frame_type = TYPE_CONFIG
        head = (frame.version, frame.timer_freq, frame.sample_rate, frame.acc_cut_freq, frame.gyr_cut_freq,
                frame.gyro_shift_alpha, frame.gyro_shift_limit, frame.level_pitch, frame.level_roll, *frame.acc_calib)
        if frame.acc_k is None:
            payload = _CONFIG_V1.pack(*head)
        else:
            payload = _CONFIG_V2.pack(*head, frame.acc_k, frame.gyr_k, frame.gyro_scale,
                                      *(frame.filter_state or (0.0,) * 12), *(frame.gyro_shift or (0.0,) * 3))
    body = bytes((frame_type, len(payload))) + payload
    crc = crc16(body)
    return SYNC + body + bytes((crc & 0xFF, crc >> 8))


class FrameParser:
    """Потоковый разбор: байты приходят кусками любой длины, кадры отдаются по мере сборки.

    После мусора или битого кадра парсер ищет следующую синхропару, поэтому подключаться к
    потоку можно с любого места.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self.crc_errors = 0
        self.skipped_bytes = 0

    def feed(self, data: bytes) -> list[Frame]:
        buf = self._buf
        buf += data
        frames: list[Frame] = []
        i = 0
        while True:
            j = buf.find(SYNC, i)
            if j < 0:
                # Последний байт может быть началом синхропары из следующего куска
                keep = 1 if buf and buf[-1] == SYNC[0] else 0
                self.skipped_bytes += len(buf) - keep - i
                i = len(buf) - keep
                break
            self.skipped_bytes += j - i
            if len(buf) - j < 4:
                i = j
                break
            frame_type, size = buf[j + 2], buf[j + 3]
            if size not in PAYLOAD_SIZES.get(frame_type, ()):
                self.skipped_bytes += 1
                i = j + 1
                continue
            end = j + 6 + size
            if len(buf) < end:
                i = j
                break
            crc = buf[end - 2] | (buf[end - 1] << 8)
            if crc16(memoryview(buf)[j + 2:end - 2]) != crc:
                self.crc_errors += 1
                self.skipped_bytes += 1
                i = j + 1
                continue
            frames.append(decode(frame_type, bytes(buf[j + 4:end - 2])))
            i = end
        del buf[:i]
        return frames


class StreamTime:
    """Перевод 32-битного времени кадров в секунды от начала потока.

    mtimer на 50 МГц переполняет 32 бита за 86 с. Кадры идут в порядке обработки в прошивке,
    поэтому время в потоке не убывает, и каждое уменьшение - это переполнение.
    """

    def __init__(self, timer_freq: int) -> None:
        self.timer_freq = timer_freq
        self._last: int | None = None
        self._high = 0
        self._start = 0

    def seconds(self, raw: int) -> float:
        if self._last is None:
            self._start = raw
        elif raw < self._last:
            self._high += TIME_MASK + 1
        self._last = raw
        return (self._high + raw - self._start) / self.timer_freq
