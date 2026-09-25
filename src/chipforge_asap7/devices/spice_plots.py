"""Plot the SPICE benches' results: one PNG per run, next to its ``results.json``.

The benches write their curves with ngspice's ``wrdata`` (a header line of
vector names, then columns) and their verdict to ``results.json``.  This module
reads both and draws the same views every time, with the bench's own pass
criteria drawn on them, so a failing run shows where it failed:

* :func:`plot_finfet_dc` -- ``Id``-``Vg`` on a log axis (off current, on
  current and their ratio), ``Id``-``Vd``, and the resistor-loaded switch with
  the output levels it must reach.
* :func:`plot_sense_amp` -- one column per input polarity: the ``SAPRECHN`` /
  ``SAE`` controls, the ``SA``/``SAN`` inputs, the ``QA``/``QAN`` outputs with
  the 20 % / 80 % thresholds and the measured resolution point, and the supply
  current.

matplotlib comes with the ``plot`` extra and is imported only when drawing.
``python -m chipforge_asap7.devices.spice_plots <run directory>`` re-plots a
finished run without simulating it again.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["plot_finfet_dc", "plot_run", "plot_sense_amp", "read_wrdata"]

# The reference categorical palette's first three slots (light mode), which
# stay distinguishable to every common colour-vision deficiency as a set.
_BLUE, _ORANGE, _AQUA = "#2a78d6", "#eb6834", "#1baf7a"
_INK, _INK_MUTED, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def read_wrdata(path: Path) -> dict[str, list[float]]:
    """Read an ngspice ``wrdata`` file (``set wr_vecnames``) into named columns.

    Names are lowercased (``v(saprechn)``): SPICE nets are case-insensitive, and
    ngspice echoes whatever case the deck happened to use.
    """
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")
    names = lines[0].lower().split()
    columns: dict[str, list[float]] = {name: [] for name in names}
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != len(names):
            continue
        for name, field in zip(names, fields, strict=True):
            columns[name].append(float(field))
    if not columns[names[0]]:
        raise ValueError(f"{path} has no numeric rows")
    return columns


def _require_matplotlib() -> Any:
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "plotting requires matplotlib; install it with "
            "'pip install chipforge-asap7[plot]'"
        ) from exc
    return Figure


def _figure(rows: int, columns: int, *, width: float, height: float) -> Any:
    figure = _require_matplotlib()(
        figsize=(width, height), layout="constrained", facecolor=_SURFACE
    )
    axes = figure.subplots(rows, columns, squeeze=False)
    for axis in axes.flat:
        axis.set_facecolor(_SURFACE)
        axis.grid(True, color=_GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=_INK_MUTED, labelsize=8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(_INK_MUTED)
    return figure, axes


def _label(axis: Any, *, x: str, y: str, title: str | None = None) -> None:
    axis.set_xlabel(x, color=_INK_MUTED, fontsize=9)
    axis.set_ylabel(y, color=_INK_MUTED, fontsize=9)
    if title:
        axis.set_title(title, color=_INK, fontsize=10, loc="left")


def _threshold(axis: Any, y: float, text: str) -> None:
    axis.axhline(y, color=_INK_MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    axis.annotate(
        text,
        (1, y),
        xycoords=("axes fraction", "data"),
        xytext=(-2, 2),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7,
        color=_INK_MUTED,
    )


def _si(value: float, unit: str) -> str:
    """``1.32e-4, "A"`` -> ``"132 µA"``."""
    for scale, prefix in ((1, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p")):
        if abs(value) >= scale:
            return f"{value / scale:.3g} {prefix}{unit}"
    return f"{value / 1e-15:.3g} f{unit}"


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def plot_finfet_dc(run_dir: Path, *, out: Path | None = None) -> Path:
    """Draw a single-device run (`chipforge_asap7.devices.spice`) to a PNG."""
    result = json.loads((run_dir / "results.json").read_text())
    transfer = read_wrdata(run_dir / "transfer.dat")
    output = read_wrdata(run_dir / "output.dat")
    switch = read_wrdata(run_dir / "switch.dat")
    vdd = max(max(transfer["v-sweep"]), max(output["v-sweep"]))
    is_nmos = result["model"].startswith("n")
    # A pFET's source sits at VDD.  Plotting both flavours against |Vgs| and
    # |Vds| puts off current on the left and on current on the right for both.
    vgs = [v if is_nmos else vdd - v for v in transfer["v-sweep"]]
    vds = [v if is_nmos else vdd - v for v in output["v-sweep"]]

    figure, axes = _figure(1, 3, width=12, height=3.8)
    idvg, idvd, sw = axes[0]

    current = [abs(i) for i in transfer["drain_current_a"]]
    idvg.semilogy(vgs, current, color=_BLUE, linewidth=2)
    # The bench reads Ioff at the first sweep point and Ion at the last.  Each
    # label sits on the side of its point the curve leaves from below.
    off, on = (0, -1) if vgs[0] < vgs[-1] else (-1, 0)
    for index, name, offset, align in (
        (off, "Ioff", (8, 0), ("left", "center")),
        (on, "Ion", (-8, 6), ("right", "bottom")),
    ):
        idvg.plot(vgs[index], current[index], "o", color=_BLUE, markersize=7)
        idvg.annotate(
            f"{name} {_si(current[index], 'A')}",
            (vgs[index], current[index]),
            xytext=offset,
            textcoords="offset points",
            ha=align[0],
            va=align[1],
            fontsize=8,
            color=_INK,
        )
    idvg.set_ylim(top=max(current) * 10)
    _label(
        idvg,
        x="|Vgs| (V)",
        y="|Id| (A)",
        title=f"Id-Vg at |Vds| = {vdd:g} V, on/off {result['on_off_ratio']:.3g}",
    )

    idvd.plot(
        vds,
        [abs(i) * 1e6 for i in output["drain_current_a"]],
        color=_BLUE,
        linewidth=2,
    )
    _label(idvd, x="|Vds| (V)", y="|Id| (µA)", title=f"Id-Vd at |Vgs| = {vdd:g} V")

    sw.plot(switch["v-sweep"], switch["v(d)"], color=_BLUE, linewidth=2)
    if is_nmos:
        _threshold(sw, 0.9 * vdd, "off: >= 0.9 VDD")
        _threshold(sw, 0.5 * vdd, "on: <= 0.5 VDD")
    else:
        _threshold(sw, 0.5 * vdd, "on: >= 0.5 VDD")
        _threshold(sw, 0.1 * vdd, "off: <= 0.1 VDD")
    _label(sw, x="Vg (V)", y="Vout (V)", title="Resistor-loaded switch")
    sw.set_ylim(-0.05 * vdd, 1.05 * vdd)

    figure.suptitle(
        f"{result['cell_name']}  ({result['model']})  {_verdict(result['passed'])}",
        color=_INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    return _save(figure, out or run_dir / "dc.png")


def _crossing(
    times: Sequence[float], values: Sequence[float], level: float
) -> float | None:
    """First rising crossing of `level`, or ``None`` if none was observed."""
    for index in range(1, len(values)):
        v0, v1 = values[index - 1], values[index]
        if v0 < level <= v1:
            t0, t1 = times[index - 1], times[index]
            return t0 + (level - v0) / (v1 - v0) * (t1 - t0)
    return None


@dataclass(frozen=True)
class _SenseAmpTiming:
    times_ns: list[float]
    window_ns: tuple[float, float]
    enable_ns: float | None
    resolution_ns: float | None
    delay_ps: float | None


def _sense_amp_timing(
    data: dict[str, list[float]], verdict: dict[str, Any], vdd: float
) -> _SenseAmpTiming:
    """Locate the control edges and choose a window around the decision."""
    times_ns = [time * 1e9 for time in data["time"]]
    release_ns = _crossing(times_ns, data["v(saprechn)"], 0.5 * vdd)
    enable_ns = _crossing(times_ns, data["v(sae)"], 0.5 * vdd)
    delay_s = verdict["resolution_delay_s"]
    delay_ns = delay_s * 1e9 if delay_s is not None else None

    # A failed simulation may never switch a control. Show its available data
    # instead of inventing an edge at the first sample.
    start = release_ns - 0.05 if release_ns is not None else times_ns[0]
    end = (
        enable_ns + max(0.1, 6 * (delay_ns or 0.0))
        if enable_ns is not None
        else times_ns[-1]
    )
    if end <= start:
        start, end = times_ns[0], times_ns[-1]
    resolution_ns = (
        enable_ns + delay_ns if enable_ns is not None and delay_ns is not None else None
    )
    return _SenseAmpTiming(
        times_ns=times_ns,
        window_ns=(start, end),
        enable_ns=enable_ns,
        resolution_ns=resolution_ns,
        delay_ps=delay_s * 1e12 if delay_s is not None else None,
    )


def _plot_sense_amp_case(
    axes: Any, data: dict[str, list[float]], verdict: dict[str, Any], vdd: float
) -> None:
    """Draw the controls, inputs, outputs, and current for one input polarity."""
    timing = _sense_amp_timing(data, verdict, vdd)
    ns, window, enable = timing.times_ns, timing.window_ns, timing.enable_ns
    controls, inputs, outputs, supply = axes
    sa, san = verdict["sa_v"], verdict["san_v"]
    heading = "SA > SAN, QA should rise" if sa > san else "SAN > SA, QAN should rise"
    _label(controls, x="", y="V", title=f"{heading}  {_verdict(verdict['passed'])}")
    for axis, pairs in (
        (
            controls,
            (("SAPRECHN", "v(saprechn)", _BLUE), ("SAE", "v(sae)", _ORANGE)),
        ),
        (inputs, (("SA", "v(sa)", _BLUE), ("SAN", "v(san)", _ORANGE))),
        (outputs, (("QA", "v(qa)", _BLUE), ("QAN", "v(qan)", _ORANGE))),
    ):
        for name, key, colour in pairs:
            axis.plot(ns, data[key], color=colour, linewidth=2, label=name)
        axis.legend(loc="center right", fontsize=8, frameon=False, labelcolor=_INK)
        axis.set_xlim(*window)
        if enable is not None:
            axis.axvline(enable, color=_INK_MUTED, linewidth=0.8, linestyle=":")

    inputs.set_ylim(
        min(sa, san) - 0.5 * abs(sa - san), max(sa, san) + 0.5 * abs(sa - san)
    )
    _label(inputs, x="", y="V")
    inputs.annotate(
        f"{abs(sa - san) * 1e3:.0f} mV",
        (window[0], max(sa, san)),
        xytext=(4, -10),
        textcoords="offset points",
        fontsize=8,
        color=_INK,
    )

    _threshold(outputs, 0.8 * vdd, "80 % VDD")
    _threshold(outputs, 0.2 * vdd, "20 % VDD")
    if timing.resolution_ns is not None and timing.delay_ps is not None:
        outputs.axvline(timing.resolution_ns, color=_INK, linewidth=0.8, linestyle=":")
        outputs.annotate(
            f"resolved {timing.delay_ps:.1f} ps after SAE",
            (timing.resolution_ns, 0.5 * vdd),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color=_INK,
        )
    _label(outputs, x="", y="V")

    current_ua = [current * 1e6 for current in data["supply_current_a"]]
    supply.plot(ns, current_ua, color=_AQUA, linewidth=2)
    supply.set_xlim(*window)
    # Ignore the initial-condition spike when samples lie in the decision window.
    shown = [
        current
        for time, current in zip(ns, current_ua, strict=True)
        if window[0] <= time <= window[1]
    ] or current_ua
    margin = 0.08 * (max(shown) - min(shown) or 1.0)
    supply.set_ylim(min(shown) - margin, max(shown) + margin)
    if enable is not None:
        supply.axvline(enable, color=_INK_MUTED, linewidth=0.8, linestyle=":")
    _label(
        supply,
        x="time (ns)",
        y="supply current (µA)",
        title=f"energy after precharge {verdict['evaluation_energy_j'] * 1e15:.2f} fJ",
    )


def plot_sense_amp(run_dir: Path, *, out: Path | None = None) -> Path:
    """Draw a sense-amplifier run (`...devices.sense_amp_spice`) to a PNG.

    The time axis is zoomed on the decision: from just before precharge
    releases to well after the outputs have resolved.
    """
    result = json.loads((run_dir / "results.json").read_text())
    vdd = result["vdd_v"]
    cases = ("sa_high", "san_high")
    figure, axes = _figure(4, 2, width=11, height=9)

    for column, case in enumerate(cases):
        data = read_wrdata(run_dir / f"{case}.dat")
        _plot_sense_amp_case(axes[:, column], data, result[case], vdd)

    figure.suptitle(
        f"{result['topology']}  n{result['n_fins']} / p{result['p_fins']} fins, "
        f"{result['nmos_model']} / {result['pmos_model']}, VDD {vdd:g} V  "
        f"{_verdict(result['passed'])}",
        color=_INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    return _save(figure, out or run_dir / "transient.png")


def _save(figure: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=_SURFACE)
    return path


def plot_run(run_dir: Path, *, out: Path | None = None) -> Path:
    """Plot whichever bench `run_dir` holds, telling them apart by their files."""
    if (run_dir / "transfer.dat").is_file():
        return plot_finfet_dc(run_dir, out=out)
    if (run_dir / "sa_high.dat").is_file():
        return plot_sense_amp(run_dir, out=out)
    raise FileNotFoundError(f"{run_dir} holds neither a FinFET nor a sense-amp run")


def main(argv: Sequence[str] | None = None) -> None:
    """Re-plot finished runs: ``python -m chipforge_asap7.devices.spice_plots DIR...``."""
    parser = argparse.ArgumentParser(description="Plot finished SPICE bench runs.")
    parser.add_argument(
        "run_dirs", nargs="+", type=Path, help="Bench output directories."
    )
    args = parser.parse_args(argv)
    for run_dir in args.run_dirs:
        print(f"{run_dir} -> {plot_run(run_dir)}")


if __name__ == "__main__":  # pragma: no cover
    main()
