"""Transient bench of one 1R1W column: write through port A, read through port B.

The deck is the column's own CDL (`column_netlist`, the netlist LVS compares
the layout with), its devices turned into BSIM-CMG OSDI instances, plus a
lumped wire capacitance on each bitline.  The controls are driven directly,
one access of each port per cycle:

::

    cycle start  precharge both ports' bitlines, the sense amplifier reset
    + pre        precharge off; WWL[i] and RWL[j] rise, the write driver on
    + sense      SAE rises: the read row's bitline difference is latched to Q
    + rwl, wwl   the wordlines fall, then the write driver turns off

Each cell starts at ``(wordline + row) % 2``.  The default sequence writes
both values, reads both values, reads a cell written the cycle before,
and in one cycle writes and reads two words on the same wordline (different mux rows).
Every cell is watched throughout: a cell no write targets must
keep its value with margin, through every half-selected write (its WWL high,
its write bitlines floating at the precharge) and every read of its
wordline, including the cycle in which both of its access pairs are on.

Usage::

    python -m chipforge_asap7.sram1r1w.column_bench --wordlines 32 --out build/sram1r1w/bench
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..devices.rom_timing import CORNERS, WIRE_FF_PER_UM, RomCorner, default_osdi
from ..devices.spice import _osdi_command_path, translate_asap7_model
from .array import MUX_ROWS, TAP_SLOTS, default_tap_pitch
from .cells import load_cell_library
from .column import ColumnSpec, column_netlist, column_pins

__all__ = [
    "Access",
    "BenchResult",
    "BenchTiming",
    "CycleResult",
    "DEFAULT_SEQUENCE",
    "render_deck",
    "run_bench",
    "spice_netlist",
]

#: One wordline slot, um: the bitlines' length per wordline.
SLOT_UM = 0.108


@dataclass(frozen=True)
class Access:
    """One cycle: write `data` to (`write` wordline, row), read (`read` wordline, row)."""

    write: tuple[int, int] | None = None
    data: int = 0
    read: tuple[int, int] | None = None


DEFAULT_SEQUENCE = (
    Access(write=(0, 0), data=1, read=(1, 0)),
    Access(write=(1, 1), data=1, read=(0, 0)),  # reads the cell written last cycle
    Access(write=(2, 0), data=1, read=(2, 2)),  # one wordline, both ports
    Access(write=(3, 3), data=1, read=(1, 1)),
    Access(write=(0, 0), data=0, read=(3, 3)),
    Access(read=(0, 0)),
)


@dataclass(frozen=True)
class BenchTiming:
    """When each control moves within a cycle, ns from the cycle's start."""

    period: float = 2.0
    precharge_end: float = 0.3
    wordline_rise: float = 0.4
    sense: float = 0.7
    read_wordline_fall: float = 0.8
    write_wordline_fall: float = 0.9
    write_enable_fall: float = 0.95
    select_rise: float = 0.05
    select_fall: float = 1.9
    slew: float = 0.02
    #: Where in the cycle Q is sampled.
    sample: float = 1.85
    max_step_ps: float = 2.0


@dataclass(frozen=True)
class CycleResult:
    cycle: int
    write: tuple[int, int] | None
    data: int
    write_ok: bool | None
    write_ps: float | None
    read: tuple[int, int] | None
    expected: int | None
    q_v: float | None
    read_ok: bool | None
    bitline_mv: float | None
    sense_to_q_ps: float | None


@dataclass(frozen=True)
class BenchResult:
    corner: str
    wordlines: int
    cycles: list[CycleResult]
    #: The least a cell no write targeted held its value by, mV (q - qb, signed
    #: by the value it holds), and which cell and cycle.
    hold_margin_mv: float
    hold_margin_cell: tuple[int, int]
    hold_margin_cycle: int
    final_errors: list[tuple[int, int]]
    passed: bool
    details: dict[str, float] = field(default_factory=dict)


def _node(name: str) -> str:
    """``WBL[2]`` -> ``wbl_2``: ngspice reads brackets as vector syntax."""
    return re.sub(r"\[(\d+)\]", r"_\1", name).lower()


_DEVICE = re.compile(r"^M(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$")


def spice_netlist(cdl: str) -> tuple[str, set[str]]:
    """The CDL with OSDI instances and plain node names; and the models it uses."""
    models: set[str] = set()
    lines = []
    for line in cdl.splitlines():
        if line.strip().upper() == ".END":
            continue
        match = _DEVICE.match(line)
        if match is None:
            lines.append(" ".join(_node(t) if "[" in t else t for t in line.split()))
            continue
        name, d, g, s, b, model, params = match.groups()
        values = dict(
            (k.lower(), v) for k, v in re.findall(r"(\w+)\s*=\s*(\S+)", params)
        )
        length = values["l"]
        length = "20n" if length in ("2e-08", "20n") else length
        models.add(model)
        nodes = " ".join(_node(n) for n in (d, g, s, b))
        lines.append(f"N{name} {nodes} {model} L={length} NFIN={values['nfin']} NF=1")
    return "\n".join(lines) + "\n", models


def _pwl(points: list[tuple[float, int]], vdd: float, slew: float) -> str:
    """A PWL source through (ns, level) steps, each `slew` long."""
    out = [(0.0, points[0][1])]
    for t, level in points[1:]:
        if level == out[-1][1]:
            continue
        out.append((t, out[-1][1]))
        out.append((t + slew, level))
    return "PWL(" + " ".join(f"{t:.4f}n {v * vdd:g}" for t, v in out) + ")"


def _schedule(
    spec: ColumnSpec, sequence: Sequence[Access], timing: BenchTiming
) -> dict[str, list[tuple[float, int]]]:
    """Each control's (ns, level) steps."""
    pins = [p for p in column_pins(spec) if p not in ("VDD", "VSS", "Q")]
    steps: dict[str, list[tuple[float, int]]] = {p: [(0.0, 0)] for p in pins}
    for p in ("wprechn", "rprechn", "wrenan", "oe"):
        steps[p] = [(0.0, 1)]
    steps["wprechn"] = steps["rprechn"] = [(0.0, 0)]
    for k in range(MUX_ROWS):
        steps[f"wyseln[{k}]"] = [(0.0, 1)]
        steps[f"ryseln[{k}]"] = [(0.0, 1)]

    def pulse(pin: str, start: float, end: float, active: int = 1) -> None:
        steps[pin] += [(start, active), (end, 1 - active)]

    for n, access in enumerate(sequence):
        t0 = n * timing.period
        for p in ("wprechn", "rprechn"):
            pulse(p, t0 + timing.precharge_end, t0 + timing.period)
        if access.write is not None:
            wordline, row = access.write
            pulse(
                f"WWL[{wordline}]",
                t0 + timing.wordline_rise,
                t0 + timing.write_wordline_fall,
            )
            pulse(f"wysel[{row}]", t0 + timing.select_rise, t0 + timing.select_fall)
            pulse(f"wyseln[{row}]", t0 + timing.select_rise, t0 + timing.select_fall, 0)
            pulse("wrena", t0 + timing.wordline_rise, t0 + timing.write_enable_fall)
            pulse("wrenan", t0 + timing.wordline_rise, t0 + timing.write_enable_fall, 0)
            steps["D"].append((t0 + timing.select_rise, access.data))
        if access.read is not None:
            wordline, row = access.read
            pulse(
                f"RWL[{wordline}]",
                t0 + timing.wordline_rise,
                t0 + timing.read_wordline_fall,
            )
            pulse(f"rysel[{row}]", t0 + timing.select_rise, t0 + timing.select_fall)
            pulse(f"ryseln[{row}]", t0 + timing.select_rise, t0 + timing.select_fall, 0)
            pulse("sae", t0 + timing.sense, t0 + timing.select_fall)
    steps["oeb"] = [(0.0, 0)]
    for pin in steps:
        steps[pin].sort(key=lambda step: step[0])
    return steps


def bitline_ff(spec: ColumnSpec) -> float:
    """Each bitline's wire, lumped: across the array and the IO block (about 1.5 um)."""
    slots = spec.wordlines + TAP_SLOTS * (
        spec.wordlines // default_tap_pitch(spec.wordlines)
    )
    return WIRE_FF_PER_UM * (slots * SLOT_UM + 1.5)


def initial_state(wordline: int, row: int) -> int:
    return (wordline + row) % 2


def render_deck(
    spec: ColumnSpec,
    sequence: Sequence[Access],
    corner: RomCorner,
    osdi: Path,
    timing: BenchTiming,
    netlist: str,
) -> tuple[str, list[str]]:
    """The deck and the vectors it writes to ``waveform.dat``, in order."""
    vdd = corner.vdd
    pins = column_pins(spec)
    wire_ff = bitline_ff(spec)
    lines = [
        f"* 1R1W column, {spec.wordlines} wordlines x {MUX_ROWS} rows, {corner.name} corner",
        ".include models.pm",
        netlist,
        f".temp {corner.temperature_c:g}",
        ".options reltol=1e-3 abstol=1e-12 vntol=1e-6 method=gear rshunt=1e10",
        f"Vsupply vdd 0 {vdd:g}",
        "Xcol "
        + " ".join(
            "vdd" if p == "VDD" else "0" if p == "VSS" else _node(p) for p in pins
        )
        + f" {spec.name}",
        "CQ q 0 1f",
    ]
    for pin, steps in _schedule(spec, sequence, timing).items():
        lines.append(f"V_{_node(pin)} {_node(pin)} 0 {_pwl(steps, vdd, timing.slew)}")
    for prefix in ("wbl", "wbln", "rbl", "rbln"):
        for k in range(MUX_ROWS):
            lines.append(f"C{prefix}{k} xcol.{prefix}_{k} 0 {wire_ff:.3f}f")
    for k in range(MUX_ROWS):
        for i in range(spec.wordlines):
            value = initial_state(i, k)
            lines.append(
                f".ic v(xcol.xcell_{k}_{i}.q)={value * vdd:g} "
                f"v(xcol.xcell_{k}_{i}.qb)={(1 - value) * vdd:g}"
            )
    watch = ["q", "sae", "rwl_0"]
    watch += [f"xcol.{p}_{k}" for p in ("rbl", "rbln") for k in range(MUX_ROWS)]
    watch += [
        f"xcol.xcell_{k}_{i}.{n}"
        for k in range(MUX_ROWS)
        for i in range(spec.wordlines)
        for n in ("q", "qb")
    ]
    vectors = " ".join(f"v({w})" for w in watch)
    stop = len(sequence) * timing.period
    lines += [
        ".control",
        "set num_threads=1",
        f"pre_osdi {_osdi_command_path(osdi)}",
        "set wr_singlescale",
        "set wr_vecnames",
        "set numdgt=7",
        f"save {vectors}",
        f"tran {timing.max_step_ps:g}p {stop:g}n 0 {timing.max_step_ps:g}p",
        f"wrdata waveform.dat {vectors}",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines), watch


def _crossing(
    time: np.ndarray, trace: np.ndarray, level: float, start: float, rising: bool
):
    inside = time >= start
    t, v = time[inside], trace[inside]
    above = v >= level
    hits = np.flatnonzero(
        above[1:] & ~above[:-1] if rising else ~above[1:] & above[:-1]
    )
    if not len(hits):
        return None
    i = hits[0]
    return float(t[i] + (level - v[i]) * (t[i + 1] - t[i]) / (v[i + 1] - v[i]))


def run_bench(
    spec: ColumnSpec,
    corner: RomCorner,
    osdi: Path,
    out: Path,
    *,
    sequence: Sequence[Access] = DEFAULT_SEQUENCE,
    timing: BenchTiming | None = None,
    timeout_s: float = 3600,
) -> BenchResult:
    """Simulate `sequence` at `corner` in `out` and check every cell and read."""
    timing = timing or BenchTiming()
    for access in sequence:
        for site in (access.write, access.read):
            if site is not None and not (
                0 <= site[0] < spec.wordlines and 0 <= site[1] < MUX_ROWS
            ):
                raise ValueError(
                    f"{site} is outside a {spec.wordlines}-wordline column"
                )
    out.mkdir(parents=True, exist_ok=True)
    netlist, models = spice_netlist(column_netlist(load_cell_library(), spec))
    card = corner.model_card().read_text()
    (out / "models.pm").write_text(
        "\n".join(translate_asap7_model(card, m) for m in sorted(models))
    )
    deck, watch = render_deck(spec, sequence, corner, osdi, timing, netlist)
    (out / "column.sp").write_text(deck)
    simulator = shutil.which("ngspice")
    if simulator is None:
        raise FileNotFoundError("ngspice was not found")
    waveform = out / "waveform.dat"
    waveform.unlink(missing_ok=True)
    done = subprocess.run(
        [simulator, "-b", "column.sp"], cwd=out, capture_output=True, text=True,
        timeout=timeout_s, check=False,
    )  # fmt: skip
    (out / "column.log").write_text(done.stdout + done.stderr)
    if not waveform.is_file():
        raise RuntimeError(f"ngspice wrote no waveform; see {out / 'column.log'}")
    data = np.loadtxt(waveform, skiprows=1)
    time = data[:, 0] * 1e9  # ns
    traces = {w: data[:, j + 1] for j, w in enumerate(watch)}
    vdd = corner.vdd

    def at(net: str, t: float) -> float:
        return float(np.interp(t, time, traces[net]))

    state = {
        (i, k): initial_state(i, k)
        for k in range(MUX_ROWS)
        for i in range(spec.wordlines)
    }
    cycles = []
    worst = (float("inf"), (0, 0), 0)
    for n, access in enumerate(sequence):
        t0, t1 = n * timing.period, (n + 1) * timing.period
        window = (time >= t0) & (time < t1)
        for (i, k), value in state.items():
            if access.write == (i, k):
                continue
            diff = traces[f"xcol.xcell_{k}_{i}.q"] - traces[f"xcol.xcell_{k}_{i}.qb"]
            margin = float(np.min((diff if value else -diff)[window])) * 1000
            if margin < worst[0]:
                worst = (margin, (i, k), n)
        write_ok = write_ps = None
        if access.write is not None:
            i, k = access.write
            q = f"xcol.xcell_{k}_{i}.q"
            write_ok = (at(q, t1 - 0.05) > vdd / 2) == bool(access.data)
            if access.data != state[(i, k)]:
                hit = _crossing(time, traces[q], vdd / 2, t0, bool(access.data))
                if hit is not None and hit < t1:
                    write_ps = (hit - t0 - timing.wordline_rise) * 1000
            state[(i, k)] = access.data if write_ok else 1 - access.data
        expected = q_v = read_ok = bitline_mv = sense_ps = None
        if access.read is not None:
            i, k = access.read
            expected = state[(i, k)] if access.read != access.write else None
            q_v = at("q", t0 + timing.sample)
            if expected is not None:
                read_ok = (q_v > vdd / 2) == bool(expected)
            t_sense = t0 + timing.sense
            bitline_mv = (
                at(f"xcol.rbl_{k}", t_sense) - at(f"xcol.rbln_{k}", t_sense)
            ) * 1000
            previous = at("q", t_sense) > vdd / 2
            if expected is not None and previous != bool(expected):
                hit = _crossing(time, traces["q"], vdd / 2, t_sense, bool(expected))
                if hit is not None and hit < t1:
                    sense_ps = (hit - t_sense) * 1000
        cycles.append(
            CycleResult(
                n,
                access.write,
                access.data,
                write_ok,
                write_ps,
                access.read,
                expected,
                q_v,
                read_ok,
                bitline_mv,
                sense_ps,
            )  # fmt: skip
        )
    end = len(sequence) * timing.period - 0.05
    errors = [
        (i, k)
        for (i, k), value in state.items()
        if (at(f"xcol.xcell_{k}_{i}.q", end) > vdd / 2) != bool(value)
    ]
    passed = (
        not errors
        and worst[0] > 0
        and all(c.write_ok is not False and c.read_ok is not False for c in cycles)
    )
    return BenchResult(
        corner.name, spec.wordlines, cycles, worst[0], worst[1], worst[2], errors, passed,
        {"bitline_ff": bitline_ff(spec)},
    )  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wordlines", type=int, default=32)
    parser.add_argument(
        "--corners", nargs="+", choices=sorted(CORNERS), default=["typical"]
    )
    parser.add_argument(
        "--output-dir",
        "--out",
        dest="out",
        type=Path,
        default=Path("build/sram1r1w/bench"),
    )
    args = parser.parse_args(argv)
    spec = ColumnSpec(args.wordlines)
    osdi = default_osdi(args.out)
    failed = False
    report = {}
    for name in args.corners:
        result = run_bench(spec, CORNERS[name], osdi, args.out / name)
        report[name] = asdict(result)
        failed |= not result.passed
        print(
            f"{name}: {'pass' if result.passed else 'FAIL'}; hold margin "
            f"{result.hold_margin_mv:.0f} mV at cell {result.hold_margin_cell} "
            f"cycle {result.hold_margin_cycle}; final errors {result.final_errors}"
        )
        for c in result.cycles:
            print(f"  {c}")
    (args.out / "bench.json").write_text(json.dumps(report, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
