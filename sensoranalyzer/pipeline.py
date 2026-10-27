"""Конвейер: источник -> кольцевой буфер -> синхронизация -> предварительная обработка."""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from .buffer import RingBuffer
from .clock import ReplayClock
from .preprocess import ProcessingParams, SensorResult, SensorStage
from .protocol import Config, Frame, Sample, StreamTime, Tick
from .resample import Resampler
from .sources import SignalSource


@dataclass(slots=True)
class TickCheck:
    """Такт прошивки рядом с моделью: насколько оценка нуля гироскопа модели расходится с ней."""
    tick: Tick
    t: float
    model_shift: tuple[float, float, float]
    shift_diff: float  # максимум по осям |модель - прошивка|, ед. АЦП


@dataclass(slots=True)
class PipelineStats:
    samples: int = 0
    ticks: int = 0
    configs: int = 0
    lost_samples: int = 0       # пропуски по номерам кадров: не влезли в буфер UART на плате
    lost_ticks: int = 0
    before_config: int = 0      # кадры до первого CONFIG: без настроек их не обработать
    buffer_dropped: int = 0     # обработка не успевала за источником
    buffer_max_fill: int = 0
    shift_diff_max: float = 0.0  # ноль гироскопа модель/прошивка по TICK, с синхронизации
    shift_diff_last: float = 0.0
    state_diff_max: float = 0.0  # состояние фильтров и ноль модель/прошивка по CONFIG v2
    state_checks: int = 0        # сколько раз состояние сравнивалось
    restarts: int = 0            # перезапуски платы посреди потока

    @property
    def bit_exact(self) -> bool:
        """Модель совпала с прошивкой без единого расхождения (было что сравнить)."""
        return self.state_checks > 0 and self.ticks > 0 and self.shift_diff_max == 0.0 and self.state_diff_max == 0.0


class Pipeline:
    """Источник читается в своём потоке и складывает кадры в буфер, обработка идёт в потоке,
    вызвавшем run().

    seed - подхватить состояние прошивки, если поток начат не с включения платы (обычно так и
    есть при подключении к работающему дрону). С CONFIG v2 состояние берётся из первого CONFIG,
    и дальше модель совпадает с прошивкой бит в бит. С CONFIG v1 состояния в потоке нет: фильтры
    подхватываются с первого отсчёта, ноль - с первого TICK, совпадение приближённое.
    Для записи, снятой с включения платы, seed=False: модель стартует с нулей, как прошивка, а
    отсчёты до первого CONFIG обрабатываются, как только он придёт.

    Перезапуск платы посреди потока (номера кадров начались с нуля) распознаётся сам: прошивка
    тогда стартует с нулей, и модель тоже - с её первого отсчёта, независимо от seed. Время
    потока при этом продолжается, а не прыгает назад.
    """

    def __init__(self, source: SignalSource, *,
                 overrides: dict[str, object] | None = None,
                 speed: float = 1.0,
                 seed: bool = True,
                 buffer_capacity: int = 8192,
                 on_result: Callable[[SensorResult], None] | None = None,
                 on_tick: Callable[[TickCheck], None] | None = None,
                 on_config: Callable[[Config, ProcessingParams], None] | None = None,
                 resampler: Resampler | None = None,
                 on_grid: Callable[[SensorResult], None] | None = None) -> None:
        self.source = source
        self.overrides = dict(overrides or {})
        self.buffer: RingBuffer[Frame] = RingBuffer(buffer_capacity)
        self.clock = None if source.realtime else ReplayClock(speed)
        self.seed = seed
        self.on_result = on_result
        self.on_tick = on_tick
        self.on_config = on_config
        self.resampler = resampler  # блок дискретизации: отсчёты на сетку заданной частоты
        self.on_grid = on_grid
        self.stats = PipelineStats()
        self.params: ProcessingParams | None = None
        self.firmware_config: Config | None = None
        self._stage: SensorStage | None = None
        self._time: StreamTime | None = None
        self._seq: dict[type, int] = {}
        self._filters_seeded = not seed
        self._shift_seeded = not seed
        self._pending: list[Frame] = []  # кадры до первого CONFIG, если модель стартует с нулей
        self._from_boot = not seed       # поток идёт с включения платы: копить кадры до CONFIG
        self._time_offset = 0.0          # время потока до последнего перезапуска платы
        self._last_t = 0.0
        self._reader: threading.Thread | None = None
        self.reader_error: BaseException | None = None

    # --- поток чтения ---------------------------------------------------------------------
    def _read(self) -> None:
        try:
            for frame in self.source.frames():
                self.buffer.push(frame)
        except BaseException as e:  # ошибка порта не должна молча повесить обработку
            self.reader_error = e
        finally:
            self.buffer.close()

    # --- обработка ------------------------------------------------------------------------
    def run(self) -> PipelineStats:
        self._reader = threading.Thread(target=self._read, name="source", daemon=True)
        self._reader.start()
        while True:
            frame = self.buffer.pop(timeout=0.2)
            if frame is None:
                if self.buffer.closed and not len(self.buffer):
                    break
                continue
            self._handle(frame)
        self._reader.join()
        self.stats.buffer_dropped = self.buffer.dropped
        self.stats.buffer_max_fill = self.buffer.max_fill
        if self.reader_error is not None:
            raise self.reader_error
        return self.stats

    def stop(self) -> None:
        self.source.stop()

    def set_overrides(self, overrides: dict[str, object]) -> None:
        """Поменять параметры на ходу. Действует с ближайшего отсчёта."""
        self.overrides = dict(overrides)
        if self.firmware_config is not None:
            self._apply_params()

    def _apply_params(self) -> None:
        params = ProcessingParams.from_config(self.firmware_config).with_overrides(self.overrides)
        errors = params.check()
        if errors:
            raise ValueError("; ".join(errors))
        if self._stage is None:
            self._stage = SensorStage(params)
        else:
            self._stage.set_params(params)
        self.params = params
        if self.on_config:
            self.on_config(self.firmware_config, params)

    def _count_seq(self, frame: Sample | Tick) -> None:
        last = self._seq.get(type(frame))
        if last is not None:
            gap = ((frame.seq - last) & 0xFFFF) - 1
            if gap:
                if isinstance(frame, Sample):
                    self.stats.lost_samples += gap
                else:
                    self.stats.lost_ticks += gap
        self._seq[type(frame)] = frame.seq

    def _handle(self, frame: Frame) -> None:
        if isinstance(frame, Config):
            self._handle_config(frame)
            return

        if self._is_restart(frame):
            self._restart()
        self._count_seq(frame)

        if self._stage is None:
            if not self._from_boot or len(self._pending) >= self.buffer.capacity:
                self.stats.before_config += 1
            else:
                self._pending.append(frame)
            return
        self._process(frame)

    def _is_restart(self, frame: Sample | Tick) -> bool:
        last = self._seq.get(type(frame))
        return frame.seq == 0 and last is not None and last != 0xFFFF

    def _restart(self) -> None:
        """Плата перезапустилась: прошивка начала с нулей - модель тоже, с её первого отсчёта."""
        self.stats.restarts += 1
        self.stats.before_config += len(self._pending)
        self._pending = []
        self._seq = {}
        # Сколько плата перезагружалась, по потоку не узнать: время продолжается через один шаг
        self._time_offset = self._last_t + (1.0 / self.params.sample_rate if self.params else 0.0)
        self._time = None
        self._stage = None
        self.firmware_config = None
        self._from_boot = True
        self._filters_seeded = self._shift_seeded = True

    def _process(self, frame: Sample | Tick) -> None:
        t = self._time_offset + self._time.seconds(frame.time)
        self._last_t = t

        if isinstance(frame, Sample):
            if self.clock is not None:
                self.clock.wait_until(t)
            if not self._filters_seeded:
                self._stage.seed(sample=frame)
                self._filters_seeded = True
            self.stats.samples += 1
            result = self._stage.process(frame, t)
            if self.on_result:
                self.on_result(result)
            if self.resampler is not None:
                for point in self.resampler.push(result):
                    if self.on_grid:
                        self.on_grid(point)
            return

        self.stats.ticks += 1
        synced = self._shift_seeded
        if not self._shift_seeded:
            self._stage.seed(gyro_shift=frame.gyro_shift)
            self._shift_seeded = True
        model = tuple(self._stage.calibrator.shift)
        diff = max(abs(m - f) for m, f in zip(model, frame.gyro_shift))
        if synced:
            self.stats.shift_diff_last = diff
            self.stats.shift_diff_max = max(self.stats.shift_diff_max, diff)
        if self.on_tick:
            self.on_tick(TickCheck(frame, t, model, diff))

    def _handle_config(self, frame: Config) -> None:
        self.stats.configs += 1
        first = self._stage is None
        if frame != self.firmware_config:  # состояние в сравнении не участвует (см. Config)
            self.firmware_config = frame
            if self._time is None:
                self._time = StreamTime(frame.timer_freq)
            self._apply_params()

        if first and self._pending:
            # Кадры до первого CONFIG после включения платы - раньше, чем состояние из него
            pending, self._pending = self._pending, []
            for f in pending:
                self._process(f)

        if frame.filter_state is not None:
            if not self._shift_seeded:
                # Состояние прошивки после этого такта: модель продолжает с него бит в бит
                self._stage.seed_state(frame.filter_state, frame.gyro_shift)
                self._filters_seeded = self._shift_seeded = True
            else:
                model = self._stage.filter_state + tuple(self._stage.calibrator.shift)
                firmware = frame.filter_state + frame.gyro_shift
                diff = max(abs(m - f) for m, f in zip(model, firmware))
                self.stats.state_diff_max = max(self.stats.state_diff_max, diff)
                self.stats.state_checks += 1
