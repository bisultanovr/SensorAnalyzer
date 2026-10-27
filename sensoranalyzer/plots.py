"""Графики тракта: сигнал до и после фильтрации, спектр, АЧХ фильтров, ноль гироскопа модели и
прошивки, неравномерность отсчётов и сетка дискретизации, сравнение двух наборов параметров.

Картинки - для проверки работы и для отчёта, поэтому статичные PNG на белом фоне. Оси X, Y, Z
разнесены по отдельным графикам, а не раскрашены тремя цветами; на каждом графике не больше трёх
рядов, цвета - эталонная категориальная палитра в фиксированном порядке слотов.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # только файлы, без окна
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .pipeline import Pipeline, PipelineStats, TickCheck  # noqa: E402
from .preprocess import ProcessingParams, Pt2Filter, SensorResult  # noqa: E402
from .protocol import Config  # noqa: E402
from .resample import Resampler  # noqa: E402
from .sources import CsvSource, open_source  # noqa: E402

# Эталонная палитра, светлая тема: слоты категориальных рядов в фиксированном порядке и чернила
SLOT = ("#2a78d6", "#eb6834", "#1baf7a")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

AXES = ("X", "Y", "Z")


def _style() -> None:
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.4,
        "savefig.dpi": 150,
    })


@dataclass
class Run:
    """Результат прогона тракта по источнику."""
    name: str
    results: list[SensorResult] = field(default_factory=list)
    checks: list[TickCheck] = field(default_factory=list)
    configs: list[tuple[Config, ProcessingParams]] = field(default_factory=list)
    stats: PipelineStats | None = None

    @property
    def params(self) -> ProcessingParams:
        return self.configs[-1][1]

    def window(self, t_from: float | None, t_to: float | None) -> Run:
        lo = -math.inf if t_from is None else t_from
        hi = math.inf if t_to is None else t_to
        return Run(self.name, [r for r in self.results if lo <= r.t <= hi],
                   [c for c in self.checks if lo <= c.t <= hi], self.configs, self.stats)

    def arrays(self) -> dict[str, np.ndarray]:
        r = self.results
        a = {name: np.array([getattr(x, name) for x in r], dtype=float)
             for name in ("raw_acc", "raw_gyr", "acc_filtered", "gyr_filtered", "acc_g", "gyr_dps", "gyro_shift")}
        a["t"] = np.array([x.t for x in r])
        p = self.params
        c = np.array(p.acc_calib, dtype=float).reshape(3, 2)
        # Сырые коды в тех же единицах, что результат: ускорение - по калибровке, угловая
        # скорость - за вычетом той же оценки нуля, чтобы до/после отличались только фильтром
        a["raw_acc_g"] = (a["raw_acc"] - c.mean(axis=1)) / ((c[:, 1] - c[:, 0]) / 2)
        a["raw_gyr_dps"] = (a["raw_gyr"] - a["gyro_shift"]) / p.gyro_scale
        return a


def collect(source: str, overrides: dict[str, object] | None = None, seed: bool = True,
            name: str | None = None) -> Run:
    src = open_source(source)
    if isinstance(src, CsvSource):
        errors = src.validate()
        if errors:
            raise ValueError("CSV не годен: " + "; ".join(errors))
    run = Run(name or Path(source).name)
    pipeline = Pipeline(src, overrides=overrides, speed=0, seed=seed,
                        on_result=run.results.append, on_tick=run.checks.append,
                        on_config=lambda c, p: run.configs.append((c, p)))
    run.stats = pipeline.run()
    if not run.results:
        raise ValueError("в источнике нет ни одного обработанного отсчёта")
    return run


def _grid(run: Run, rate: float, mode: str = "linear") -> Run:
    r = Resampler(rate, mode, max_gap=4.0 / rate)
    points = [p for x in run.results for p in r.push(x)]
    return Run(run.name, points, [], run.configs, run.stats)


def _save(fig, out: Path) -> Path:
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _cutoff(cut_freq: float, rate: float, exact: bool) -> float:
    """Частота, где АЧХ каскада PT2 падает до -3 дБ."""
    lo, hi = 1e-3, rate / 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if Pt2Filter.magnitude(cut_freq, rate, mid, exact) ** 2 > 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _subtitle(run: Run) -> str:
    p = run.params
    cut = "точные срезы" if p.exact_cutoff else "срезы по формуле RC"
    return (f"{run.name}: {p.sample_rate:.0f} Гц, PT2 {p.acc_cut_freq:g}/{p.gyr_cut_freq:g} Гц ({cut}), "
            f"{len(run.results)} отсчётов")


# --- до и после фильтрации ----------------------------------------------------------------
def plot_signals(run: Run, out: Path) -> Path:
    a = run.arrays()
    p = run.params
    fig, axs = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
    for i, ax_name in enumerate(AXES):
        for col, (raw, filt, unit, cut, what) in enumerate((
                (a["raw_acc_g"], a["acc_g"], "g", p.acc_cut_freq, "Ускорение"),
                (a["raw_gyr_dps"], a["gyr_dps"], "°/с", p.gyr_cut_freq, "Угловая скорость"))):
            ax = axs[i, col]
            ax.plot(a["t"], raw[:, i], color=SLOT[0], lw=0.8, label="до фильтрации")
            ax.plot(a["t"], filt[:, i], color=SLOT[1], lw=1.4, label=f"после PT2 {cut:g} Гц")
            ax.set_title(f"{what} {ax_name}, {unit}", loc="left")
    for ax in axs[-1]:
        ax.set_xlabel("время, с")
    axs[0, 0].legend(loc="upper right", ncol=2)
    fig.suptitle("Сигнал до и после фильтрации\n" + _subtitle(run), x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    return _save(fig, out)


# --- спектр -----------------------------------------------------------------------------------
def _asd(x: np.ndarray, rate: float, seg: int) -> tuple[np.ndarray, np.ndarray]:
    """Амплитудная спектральная плотность по Уэлчу: окно Ханна, перекрытие 50 %."""
    x = x - x.mean()
    seg = min(seg, len(x))
    win = np.hanning(seg)
    step = seg // 2
    starts = range(0, len(x) - seg + 1, step)
    psd = np.mean([np.abs(np.fft.rfft(x[s:s + seg] * win)) ** 2 for s in starts], axis=0)
    psd *= 2.0 / (rate * (win ** 2).sum())
    return np.fft.rfftfreq(seg, 1.0 / rate), np.sqrt(psd)


def plot_spectrum(run: Run, out: Path, rate: float) -> Path:
    g = _grid(run, rate).arrays()
    p = run.params
    seg = 2 ** int(math.log2(max(16, min(len(g["t"]) // 4, 4096))))
    fig, axs = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
    for i, ax_name in enumerate(AXES):
        for col, (raw, filt, unit, cut, what) in enumerate((
                (g["raw_acc_g"], g["acc_g"], "g/√Гц", p.acc_cut_freq, "Ускорение"),
                (g["raw_gyr_dps"], g["gyr_dps"], "°/с/√Гц", p.gyr_cut_freq, "Угловая скорость"))):
            ax = axs[i, col]
            f, s_raw = _asd(raw[:, i], rate, seg)
            _, s_filt = _asd(filt[:, i], rate, seg)
            ax.semilogy(f[1:], s_raw[1:], color=SLOT[0], lw=0.9, label="до фильтрации")
            ax.semilogy(f[1:], s_filt[1:], color=SLOT[1], lw=1.4, label="после PT2")
            ax.axvline(cut, color=MUTED, lw=0.8, ls="--")
            ax.set_title(f"{what} {ax_name}, {unit}  ·  пунктир - срез {cut:g} Гц", loc="left")
    for ax in axs[-1]:
        ax.set_xlabel("частота, Гц")
        ax.set_xlim(0, rate / 2)
    axs[0, 0].legend(loc="lower left", ncol=2)
    fig.suptitle(f"Амплитудный спектр (Уэлч, окно {seg} точек, равномерная сетка {rate:g} Гц)\n" + _subtitle(run),
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    return _save(fig, out)


# --- АЧХ фильтров -----------------------------------------------------------------------------
def plot_response(params: ProcessingParams, out: Path) -> Path:
    rate = params.sample_rate
    f = np.linspace(0.2, rate / 2, 1200)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (cut, what) in zip(axs, ((params.acc_cut_freq, "ускорения"), (params.gyr_cut_freq, "угловой скорости"))):
        exact = [20 * math.log10(Pt2Filter.magnitude(cut, rate, x, True)) for x in f]
        legacy = [20 * math.log10(Pt2Filter.magnitude(cut, rate, x, False)) for x in f]
        c_legacy = _cutoff(cut, rate, False)
        ax.plot(f, exact, color=SLOT[0], label=f"точный k: срез {_cutoff(cut, rate, True):.1f} Гц")
        ax.plot(f, legacy, color=SLOT[1], ls="--", label=f"формула RC: срез {c_legacy:.1f} Гц")
        ax.axhline(-3, color=MUTED, lw=0.8)
        ax.axvline(cut, color=MUTED, lw=0.8, ls=":")
        ax.annotate("−3 дБ", (rate / 2, -3), xytext=(-4, 3), textcoords="offset points", ha="right",
                    color=INK2, fontsize=8)
        ax.set_title(f"PT2 {what}, задан срез {cut:g} Гц", loc="left")
        ax.set_xlabel("частота, Гц")
        ax.set_xlim(0, rate / 2)
        ax.set_ylim(-40, 2)
        ax.legend(loc="lower left")
    axs[0].set_ylabel("АЧХ, дБ")
    fig.suptitle(f"АЧХ фильтров PT2 при частоте отсчётов {rate:.0f} Гц", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    return _save(fig, out)


# --- ноль гироскопа: модель и прошивка ----------------------------------------------------------
def plot_gyro_zero(run: Run, out: Path) -> Path | None:
    if not run.checks:
        return None
    t = np.array([c.t for c in run.checks])
    model = np.array([c.model_shift for c in run.checks])
    firmware = np.array([c.tick.gyro_shift for c in run.checks])
    diff = np.abs(model - firmware).max(axis=1)
    fig, axs = plt.subplots(4, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 0.8]})
    for i, ax_name in enumerate(AXES):
        ax = axs[i]
        ax.plot(t, model[:, i], color=SLOT[0], label="модель на ПК")
        ax.plot(t, firmware[:, i], color=SLOT[1], ls="--", label="прошивка (кадр TICK)")
        ax.set_title(f"Ноль гироскопа {ax_name}, ед. АЦП", loc="left")
    axs[0].legend(loc="upper right", ncol=2)
    ax = axs[3]
    if diff.max() > 0:
        ax.semilogy(t, np.where(diff > 0, diff, np.nan), color=SLOT[0], marker="o", ms=3, lw=0)
        ax.set_title("Расхождение модели и прошивки, максимум по осям, ед. АЦП", loc="left")
    else:
        ax.plot(t, diff, color=SLOT[0])
        ax.set_title("Расхождение модели и прошивки, ед. АЦП", loc="left")
        ax.text(0.5, 0.62, f"0 на всех {len(t)} тактах: модель совпадает с прошивкой бит в бит",
                transform=ax.transAxes, ha="center", va="bottom", color=INK2)
    ax.set_xlabel("время, с")
    fig.suptitle("Оценка нуля гироскопа: модель на ПК и прошивка\n" + _subtitle(run), x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    return _save(fig, out)


# --- неравномерность отсчётов и сетка дискретизации ----------------------------------------------
def plot_sampling(run: Run, out: Path, rate: float, window: float = 0.1) -> Path:
    t = np.array([r.t for r in run.results])
    dt = np.diff(t) * 1e3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})

    ax1.hist(dt, bins=np.arange(0, max(4.0, dt.max()) + 0.1, 0.1), color=SLOT[0], rwidth=0.85)
    ax1.axvline(dt.mean(), color=INK2, lw=0.9, ls="--")
    ax1.annotate(f"средний {dt.mean():.2f} мс ({1e3 / dt.mean():.0f} Гц)\nмин {dt.min():.2f}, "
                 f"макс {dt.max():.2f}, СКО {dt.std():.2f} мс",
                 (dt.mean(), 1), xycoords=("data", "axes fraction"), xytext=(6, -6), textcoords="offset points",
                 va="top", color=INK2, fontsize=8)
    ax1.set_title("Шаг между отсчётами, мс", loc="left")
    ax1.set_xlabel("шаг, мс")
    ax1.set_ylabel("отсчётов")

    mid = t[len(t) // 2]
    sel = [r for r in run.results if mid <= r.t <= mid + window]
    hold = _grid(Run(run.name, sel, [], run.configs), rate, "hold").results
    linear = _grid(Run(run.name, sel, [], run.configs), rate, "linear").results
    ax2.plot([r.t for r in sel], [r.gyr_dps[0] for r in sel], color=SLOT[0], lw=0, marker="o", ms=4,
             label="отсчёты с дрона")
    ax2.step([r.t for r in hold], [r.gyr_dps[0] for r in hold], where="post", color=SLOT[1],
             label=f"сетка {rate:g} Гц, удержание")
    ax2.plot([r.t for r in linear], [r.gyr_dps[0] for r in linear], color=SLOT[2], marker="s", ms=4,
             label=f"сетка {rate:g} Гц, интерполяция")
    ax2.set_title(f"Дискретизация на равномерную сетку: угловая скорость X, {window * 1e3:.0f} мс записи", loc="left")
    ax2.set_xlabel("время, с")
    ax2.set_ylabel("°/с")
    ax2.legend(loc="best")
    fig.suptitle("Неравномерность отсчётов и блок дискретизации\n" + _subtitle(run), x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    return _save(fig, out)


# --- сравнение параметров -----------------------------------------------------------------
def plot_compare(base: Run, alt: Run, alt_label: str, out: Path) -> Path:
    a, b = base.arrays(), alt.arrays()
    fig, axs = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
    for i, ax_name in enumerate(AXES):
        for col, (key, unit, what) in enumerate((("acc_g", "g", "Ускорение"), ("gyr_dps", "°/с", "Угловая скорость"))):
            ax = axs[i, col]
            ax.plot(a["t"], a[key][:, i], color=SLOT[0], label="параметры прошивки")
            ax.plot(b["t"], b[key][:, i], color=SLOT[1], lw=1.1, label=alt_label)
            ax.set_title(f"{what} {ax_name}, {unit}", loc="left")
    for ax in axs[-1]:
        ax.set_xlabel("время, с")
    axs[0, 0].legend(loc="upper right", ncol=2)
    fig.suptitle(f"Сравнение параметров тракта на одной записи: {alt_label}\n" + _subtitle(base),
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    return _save(fig, out)


def make_all(source: str, out_dir: str | Path, overrides: dict[str, object] | None = None,
             compare: dict[str, object] | None = None, seed: bool = True,
             t_from: float | None = None, t_to: float | None = None,
             rate: float | None = None) -> list[Path]:
    """Все графики по источнику в каталог out_dir."""
    _style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    full = collect(source, overrides, seed)
    run = full.window(t_from, t_to)
    if len(run.results) < 16:
        raise ValueError("в выбранном отрезке слишком мало отсчётов")
    grid_rate = rate or round(full.params.sample_rate)
    files = [
        plot_signals(run, out / "signals.png"),
        plot_spectrum(run, out / "spectrum.png", grid_rate),
        plot_response(run.params, out / "response.png"),
        plot_sampling(run, out / "sampling.png", rate or 100.0),
    ]
    zero = plot_gyro_zero(run, out / "gyro_zero.png")
    if zero:
        files.append(zero)
    if compare:
        alt = collect(source, {**(overrides or {}), **compare}, seed).window(t_from, t_to)
        label = ", ".join(f"{k}={v}" for k, v in compare.items())
        files.append(plot_compare(run, alt, label, out / "compare.png"))
    return files
