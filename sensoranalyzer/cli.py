"""Командная строка: python -m sensoranalyzer <команда> ..."""
from __future__ import annotations

import argparse
import sys
import threading
import time
from contextlib import nullcontext

from .export import export_csv
from .pipeline import Pipeline, TickCheck
from .preprocess import ProcessingParams, SensorResult
from .protocol import FLAG_ARMED, Config
from .sources import CsvSource, SerialSource, open_source


def parse_overrides(items: list[str]) -> dict[str, object]:
    """name=value из --set. acc_calib - шесть целых через запятую, exact_cutoff - 0 или 1."""
    overrides: dict[str, object] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"ожидается имя=значение: {item}")
        name = name.strip()
        if name == "acc_calib":
            parts = tuple(int(v) for v in value.split(","))
            if len(parts) != 6:
                raise argparse.ArgumentTypeError("acc_calib: нужно шесть чисел")
            overrides[name] = parts
        elif name == "exact_cutoff":
            overrides[name] = value.strip().lower() in ("1", "true", "yes", "да")
        else:
            overrides[name] = float(value)
    return overrides


def cmd_ports(args) -> int:
    from serial.tools import list_ports
    for p in list_ports.comports():
        print(f"{p.device:8} {p.description}  [{p.hwid}]")
    return 0


def _stop_after(pipeline: Pipeline, seconds: float | None) -> None:
    if seconds:
        timer = threading.Timer(seconds, pipeline.stop)
        timer.daemon = True
        timer.start()


def _fmt(v, digits=3) -> str:
    return "(" + ", ".join(f"{x:+.{digits}f}" for x in v) + ")"


class StatusPrinter:
    """Строка состояния раз в секунду времени потока."""

    def __init__(self, pipeline: Pipeline, quiet: bool = False) -> None:
        self.pipeline = pipeline
        self.quiet = quiet
        self.last: SensorResult | None = None
        self.last_tick: TickCheck | None = None
        self._next = 1.0

    def on_config(self, config: Config, params: ProcessingParams) -> None:
        cut = "точные" if params.exact_cutoff else "по формуле RC"
        print(f"CONFIG прошивки v{config.version}: отсчёты {config.sample_rate:.1f} Гц, срезы {config.acc_cut_freq:g}/"
              f"{config.gyr_cut_freq:g} Гц ({cut}), tau нуля {params.gyro_shift_tau:.3f} с, "
              f"калибровка {config.acc_calib}, LevelHor {config.level_pitch:.2f}/{config.level_roll:.2f}")
        if self.pipeline.overrides:
            print(f"  переопределено: {self.pipeline.overrides}")

    def on_result(self, r: SensorResult) -> None:
        self.last = r
        if r.t >= self._next:
            self._next = r.t + 1.0
            if not self.quiet:
                self.print_line()

    def on_tick(self, check: TickCheck) -> None:
        self.last_tick = check

    def print_line(self) -> None:
        s, r, c = self.pipeline.stats, self.last, self.last_tick
        line = (f"t={r.t:7.2f} с  отсч {s.samples:6d}  буфер {len(self.pipeline.buffer):4d}  "
                f"потери {s.lost_samples}/{s.lost_ticks}  a={_fmt(r.acc_g)} g  w={_fmt(r.gyr_dps, 2)} °/с  "
                f"ноль={_fmt(r.gyro_shift, 2)}")
        if c is not None:
            armed = " АРМ" if c.tick.flags & FLAG_ARMED else ""
            line += f"  Δноль={c.shift_diff:.1e}  k={c.tick.gain:.4f}{armed}"
        print(line, flush=True)


def cmd_run(args) -> int:
    overrides = parse_overrides(args.set or [])
    record = open(args.record, "wb") if args.record else nullcontext()
    with record as rec:
        source = open_source(args.source, record_to=rec if args.record else None)
        if args.record and not isinstance(source, SerialSource):
            print("--record пишет только поток из порта", file=sys.stderr)
            return 2
        if isinstance(source, CsvSource):
            errors = source.validate()
            if errors:
                print("CSV не годен:\n  " + "\n  ".join(errors), file=sys.stderr)
                return 1
        pipeline = Pipeline(source, overrides=overrides, speed=args.speed, seed=not args.no_seed)
        status = StatusPrinter(pipeline, quiet=args.quiet)
        pipeline.on_result, pipeline.on_tick, pipeline.on_config = status.on_result, status.on_tick, status.on_config
        _stop_after(pipeline, args.seconds)
        started = time.perf_counter()
        try:
            stats = pipeline.run()
        except KeyboardInterrupt:
            pipeline.stop()
            stats = pipeline.stats
    elapsed = time.perf_counter() - started
    print(f"\nитого за {elapsed:.1f} с: отсчётов {stats.samples}, тактов {stats.ticks}, CONFIG {stats.configs}; "
          f"потеряно на плате {stats.lost_samples}/{stats.lost_ticks}, в буфере ПК {stats.buffer_dropped} "
          f"(макс. заполнение {stats.buffer_max_fill}); до первого CONFIG пропущено {stats.before_config}")
    print(f"разбор: {source.stats}" + (f"; перезапусков платы: {stats.restarts}" if stats.restarts else ""))
    if stats.ticks:
        print(f"ноль гироскопа модель/прошивка: макс. расхождение {stats.shift_diff_max:.2e}, "
              f"последнее {stats.shift_diff_last:.2e} ед. АЦП")
    if stats.state_checks:
        print(f"состояние фильтров и ноль по CONFIG ({stats.state_checks} сверок): макс. расхождение "
              f"{stats.state_diff_max:.2e}" + ("  -> модель совпадает с прошивкой бит в бит" if stats.bit_exact else ""))
    return 0


def cmd_record(args) -> int:
    with open(args.out, "wb") as rec:
        source = SerialSource(args.port, record_to=rec)
        pipeline = Pipeline(source)
        status = StatusPrinter(pipeline)
        pipeline.on_result, pipeline.on_tick = status.on_result, status.on_tick
        _stop_after(pipeline, args.seconds)
        try:
            stats = pipeline.run()
        except KeyboardInterrupt:
            pipeline.stop()
            stats = pipeline.stats
    print(f"\nзаписано {source.bytes_received} байт в {args.out}: отсчётов {stats.samples}, тактов {stats.ticks}, "
          f"потеряно на плате {stats.lost_samples}/{stats.lost_ticks}")
    return 0


def cmd_export(args) -> int:
    count = export_csv(args.recording, args.out)
    print(f"{count} отсчётов -> {args.out}")
    return 0


def cmd_validate(args) -> int:
    errors = CsvSource(args.csv).validate()
    if errors:
        print("\n".join(errors))
        return 1
    print("формат годен")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sensoranalyzer", description="Приём и обработка телеметрии IMU дрона")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("ports", help="список COM-портов").set_defaults(func=cmd_ports)

    r = sub.add_parser("run", help="обработка из порта, записи .imus или CSV")
    r.add_argument("source", help="COM-порт (COM6), запись .imus или .csv")
    r.add_argument("--speed", type=float, default=1.0, help="темп воспроизведения записи; 0 - без ожидания")
    r.add_argument("--set", action="append", metavar="ИМЯ=ЗНАЧЕНИЕ",
                   help="переопределить параметр тракта (можно несколько раз)")
    r.add_argument("--no-seed", action="store_true",
                   help="не подхватывать состояние прошивки (запись снята с включения платы)")
    r.add_argument("--record", metavar="ФАЙЛ.imus", help="заодно записать поток из порта")
    r.add_argument("--seconds", type=float, help="остановиться через столько секунд")
    r.add_argument("--quiet", action="store_true", help="без строки состояния, только итог")
    r.set_defaults(func=cmd_run)

    rec = sub.add_parser("record", help="записать поток из порта в .imus")
    rec.add_argument("port")
    rec.add_argument("out")
    rec.add_argument("--seconds", type=float)
    rec.set_defaults(func=cmd_record)

    e = sub.add_parser("export", help="отсчёты записи .imus в CSV")
    e.add_argument("recording")
    e.add_argument("out")
    e.set_defaults(func=cmd_export)

    v = sub.add_parser("validate", help="проверить формат CSV")
    v.add_argument("csv")
    v.set_defaults(func=cmd_validate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
