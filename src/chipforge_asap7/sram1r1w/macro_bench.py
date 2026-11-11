"""Clocked transient bench of the 1R1W macro netlist (`periphery.macro_cdl`).

Only ``CLK``, the enables, the addresses and ``DIN`` are driven.  CLK rises
at the start of each cycle (50% duty, the first rise one period in), and the
inputs change half a period before each rise: a setup and hold of half a
period.  A word's bit ``b`` starts at ``(wordline + row + b) % 2``.

Each cycle can hold a write, a read, or both.  The bench checks four things:

* every read's ``DOUT`` just before the next rise.  A read of the word
  written in the same cycle is the contract's X and is not checked;
* every written word's cells at the end of its cycle;
* every cell's final value;
* each write's margin: the time from the written cell's flip to its
  wordline's fall.

It also reports ``CLK`` to ``DOUT`` for every read that changes ``DOUT``.

Usage::

    python -m chipforge_asap7.sram1r1w.macro_bench --corners typical --period 1.0
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..devices.rom_timing import CORNERS, RomCorner, default_osdi
from ..devices.spice import _osdi_command_path, translate_asap7_model
from .array import MUX_ROWS
from .cells import load_cell_library
from .column_bench import bitline_ff
from .periphery import MacroSpec, macro_cdl, macro_pins

__all__ = ["Cycle", "MacroResult", "DEFAULT_CYCLES", "run_macro_bench"]


@dataclass(frozen=True)
class Cycle:
    """One clock cycle: write `data` (an int over the bits) to `write`; read `read`."""

    write: int | None = None
    data: int = 0
    read: int | None = None


DEFAULT_CYCLES = (
    Cycle(write=5, data=0b10, read=12),
    Cycle(write=77, data=0b01, read=5),  # reads last cycle's write
    Cycle(write=40, data=0b11, read=41),  # one wordline, both ports
    Cycle(write=100, data=0b00, read=100),  # the same word: X
    Cycle(read=77),
    Cycle(read=40),
    Cycle(write=5, data=0b01, read=100),
    Cycle(read=5),
    Cycle(),
    Cycle(write=127, data=0b10, read=126),
    Cycle(read=127),
)


@dataclass(frozen=True)
class MacroResult:
    corner: str
    period_ns: float
    reads: list[dict]
    writes: list[dict]
    final_errors: list[tuple[int, int]]
    clk_to_dout_ps: float | None
    min_write_margin_ps: float | None
    passed: bool


def _node(name: str) -> str:
    return re.sub(r"[\[<](\d+)[\]>]", r"_\1", name).lower()


def initial_bit(address: int, bit: int) -> int:
    return ((address >> 2) + (address & 3) + bit) % 2


def _cell(address: int, bit: int) -> str:
    return f"xm.xbit{bit}.xcell_{address & 3}_{address >> 2}"


def _osdi_netlist(cdl: str) -> tuple[str, set[str]]:
    """OSDI instances and plain node names (as `column_bench.spice_netlist`)."""
    models: set[str] = set()
    lines = []
    for line in cdl.splitlines():
        if line.strip().upper() == ".END":
            continue
        match = re.match(
            r"^M(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$", line
        )
        if match is None:
            lines.append(
                " ".join(_node(t) if t[0] not in ".*" else t for t in line.split())
            )
            continue
        name, d, g, s, b, model, params = match.groups()
        values = {k.lower(): v for k, v in re.findall(r"(\w+)\s*=\s*(\S+)", params)}
        length = "20n" if values["l"] in ("2e-08", "20n") else values["l"]
        models.add(model)
        nodes = " ".join(_node(n) for n in (d, g, s, b))
        lines.append(
            f"N{_node(name)} {nodes} {model} L={length} NFIN={values['nfin']} NF=1"
        )
    return "\n".join(lines) + "\n", models


def _pwl(points: list[tuple[float, int]], vdd: float, slew: float) -> str:
    out = [(0.0, points[0][1])]
    for t, level in points[1:]:
        if level == out[-1][1]:
            continue
        out += [(t, out[-1][1]), (t + slew, level)]
    return "PWL(" + " ".join(f"{t:.4f}n {v * vdd:g}" for t, v in out) + ")"


def _stimulus(
    spec: MacroSpec, cycles: Sequence[Cycle], period: float
) -> dict[str, list[tuple[float, int]]]:
    """Each input's (ns, level) steps; cycle n's edge is at (n + 1) * period."""
    a = spec.address_bits
    steps: dict[str, list[tuple[float, int]]] = {"CLK": [(0.0, 0)]}
    for name in ("WEB", "REB"):
        steps[name] = [(0.0, 1)]
    for k in range(a):
        steps[f"WADDR<{k}>"] = [(0.0, 0)]
        steps[f"RADDR<{k}>"] = [(0.0, 0)]
    for b in range(spec.bits):
        steps[f"DIN<{b}>"] = [(0.0, 0)]
    for n, cycle in enumerate(cycles):
        edge = (n + 1) * period
        steps["CLK"] += [(edge, 1), (edge + period / 2, 0)]
        t = edge - period / 2
        steps["WEB"].append((t, 0 if cycle.write is not None else 1))
        steps["REB"].append((t, 0 if cycle.read is not None else 1))
        if cycle.write is not None:
            for k in range(a):
                steps[f"WADDR<{k}>"].append((t, cycle.write >> k & 1))
            for b in range(spec.bits):
                steps[f"DIN<{b}>"].append((t, cycle.data >> b & 1))
        if cycle.read is not None:
            for k in range(a):
                steps[f"RADDR<{k}>"].append((t, cycle.read >> k & 1))
    end = (len(cycles) + 1) * period
    steps["WEB"].append((end - period / 2, 1))
    steps["REB"].append((end - period / 2, 1))
    return steps


def render_deck(
    spec: MacroSpec, cycles: Sequence[Cycle], corner: RomCorner, osdi: Path, period: float,
    netlist: str, *, slew: float = 0.02, max_step_ps: float = 2.0,
) -> tuple[str, list[str]]:  # fmt: skip
    vdd = corner.vdd
    pins = macro_pins(spec)
    wire = bitline_ff(spec.column)
    lines = [
        f"* {spec.name} at the {corner.name} corner, {period:g} ns",
        ".include models.pm",
        netlist,
        f".temp {corner.temperature_c:g}",
        # A tighter charge tolerance and a looser truncation-error factor: with
        # the defaults the typical corner's timestep collapses where a write
        # and a read switch together (at 3.6 ns of the 0.5 ns bench).
        ".options reltol=1e-3 abstol=1e-12 vntol=1e-6 chgtol=1e-16 trtol=3 "
        "method=gear rshunt=1e10",
        f"Vsupply vdd 0 {vdd:g}",
        "Xm "
        + " ".join(
            "vdd" if p == "VDD" else "0" if p == "VSS" else _node(p) for p in pins
        )
        + f" {spec.name}",
        *(f"CDOUT{b} dout_{b} 0 2f" for b in range(spec.bits)),
        f"CRBLR xm.rblr 0 {wire:.3f}f",
    ]
    for pin, steps in _stimulus(spec, cycles, period).items():
        lines.append(f"V_{_node(pin)} {_node(pin)} 0 {_pwl(steps, vdd, slew)}")
    for b in range(spec.bits):
        for prefix in ("wbl", "wbln", "rbl", "rbln"):
            for k in range(MUX_ROWS):
                lines.append(f"C{prefix}{b}_{k} xm.xbit{b}.{prefix}_{k} 0 {wire:.3f}f")
    for half, bits in spec.halves.items():
        wordline_ff = len(bits) * 2.376 * spec.model.wire_cap_fF_per_um
        for i in range(spec.wordlines):
            for wl in ("wwl", "rwl"):
                net = f"{wl}{half.lower()}_{i}"
                lines.append(f"C{net} xm.{net} 0 {wordline_ff:.3f}f")
    # Each cell's state is set on qb alone; its inverter sets q.  With both
    # held, the fast corner's operating point does not converge in hours.
    for address in range(spec.words):
        for b in range(spec.bits):
            v = initial_bit(address, b)
            lines.append(f".ic v({_cell(address, b)}.qb)={(1 - v) * vdd:g}")
    for net, level in idle_state(spec).items():
        lines.append(f".ic v({net})={level * vdd:g}")
    watch = [*(f"dout_{b}" for b in range(spec.bits)), "clk"]
    watch += [
        f"xm.{n}" for n in ("sae", "ren", "wen", "wrena", "wprechn", "rprechn", "rblr")
    ]
    watch += [
        f"xm.wwl{h.lower()}_{i}" for h in spec.halves for i in range(spec.wordlines)
    ]
    watch += [
        f"{_cell(a, b)}.{n}"
        for a in range(spec.words)
        for b in range(spec.bits)
        for n in ("q", "qb")
    ]
    vectors = " ".join(f"v({w})" for w in watch)
    stop = (len(cycles) + 1.5) * period
    lines += [
        ".control",
        "set num_threads=1",
        f"pre_osdi {_osdi_command_path(osdi)}",
        "set wr_singlescale",
        "set wr_vecnames",
        "set numdgt=7",
        f"save {vectors}",
        f"tran {max_step_ps:g}p {stop:g}n 0 {max_step_ps:g}p",
        f"wrdata waveform.dat {vectors}",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines), watch


def idle_state(spec: MacroSpec) -> dict[str, int]:
    """Every controller's nets, and each bit's output latch, between cycles.

    The controllers' request loops hold one DC state with CLK low, but a
    solver started anywhere can wander through their rings first; this is
    where they rest, for the operating point."""
    state: dict[str, int] = {}
    for port, delay, outs in (
        ("W", spec.write_delay, ("wprechn", "wrena", "WEN")),
        ("R", spec.read_delay, ("rprechn", "REN", "sae")),
    ):
        local = {"CLKB": 1, "ARMED": 1, "ARMB": 0, "SET": 0, "REQB": 1, "REQ": 0,
                 "EN": 1, "PN": 1, "ERN": 1, "EV0": 0, "SN": 1, "DONE": 0}  # fmt: skip
        local |= {f"DL{i}": i % 2 for i in range(delay - 1)}
        # The write controller's EVB is the wrenan pin; the read's is its own.
        local |= {"EVB": 1, "READY": 0} if port == "R" else {}
        state |= {f"xm.c{port.lower()}_{n.lower()}": v for n, v in local.items()}
        state |= {f"xm.{n.lower()}": 0 for n in outs}
    state["xm.wrenan"] = 1
    # One node of each output latch, as of each cell: with both held, the
    # fast corner's operating point does not converge.
    for b in range(spec.bits):
        state[f"xm.xbit{b}.xread.ol_y2"] = 0
    return state


def _crossings(time: np.ndarray, v: np.ndarray, level: float) -> np.ndarray:
    above = v >= level
    hits = np.flatnonzero(above[1:] != above[:-1])
    return time[hits] + (level - v[hits]) * (time[hits + 1] - time[hits]) / (
        v[hits + 1] - v[hits]
    )


def run_macro_bench(
    spec: MacroSpec, corner: RomCorner, osdi: Path, out: Path, *,
    cycles: Sequence[Cycle] = DEFAULT_CYCLES, period: float = 2.0, timeout_s: float = 7200,
) -> MacroResult:  # fmt: skip
    """Simulate `cycles` at `period` ns and check every read, write and cell."""
    for c in cycles:
        for a in (c.write, c.read):
            if a is not None and not 0 <= a < spec.words:
                raise ValueError(f"address {a} is outside {spec.words} words")
    out.mkdir(parents=True, exist_ok=True)
    netlist, models = _osdi_netlist(macro_cdl(spec, load_cell_library()))
    card = corner.model_card().read_text()
    (out / "models.pm").write_text(
        "\n".join(translate_asap7_model(card, m) for m in sorted(models))
    )
    deck, watch = render_deck(spec, cycles, corner, osdi, period, netlist)
    (out / "macro.sp").write_text(deck)
    simulator = shutil.which("ngspice")
    if simulator is None:
        raise FileNotFoundError("ngspice was not found")
    waveform = out / "waveform.dat"
    waveform.unlink(missing_ok=True)
    done = subprocess.run(
        [simulator, "-b", "macro.sp"], cwd=out, capture_output=True, text=True,
        timeout=timeout_s, check=False,
    )  # fmt: skip
    (out / "macro.log").write_text(done.stdout + done.stderr)
    if not waveform.is_file():
        raise RuntimeError(f"ngspice wrote no waveform; see {out / 'macro.log'}")
    data = np.loadtxt(waveform, skiprows=1)
    time = data[:, 0] * 1e9
    trace = {w: data[:, j + 1] for j, w in enumerate(watch)}
    vdd = corner.vdd
    half = vdd / 2

    def at(net: str, t: float) -> float:
        return float(np.interp(t, time, trace[net]))

    def stored(address: int, b: int, t: float) -> int:
        cell = _cell(address, b)
        return int(at(f"{cell}.q", t) > at(f"{cell}.qb", t))

    memory = {
        (a, b): initial_bit(a, b) for a in range(spec.words) for b in range(spec.bits)
    }
    reads, writes = [], []
    clk_q = []
    margins = []
    for n, cycle in enumerate(cycles):
        edge = (n + 1) * period
        end = edge + period - 0.02
        if cycle.read is not None:
            same = cycle.read == cycle.write
            expected = (
                None
                if same
                else sum(memory[(cycle.read, b)] << b for b in range(spec.bits))
            )
            got = sum(int(at(f"dout_{b}", end) > half) << b for b in range(spec.bits))
            delays = []
            for b in range(spec.bits):
                hits = _crossings(time, trace[f"dout_{b}"], half)
                hits = hits[(hits > edge) & (hits < end)]
                if len(hits):
                    delays.append((hits[0] - edge) * 1000)
            clk_q += delays
            reads.append({
                "cycle": n, "address": cycle.read, "expected": expected, "dout": got,
                "ok": None if expected is None else got == expected,
                "clk_to_dout_ps": max(delays) if delays else None,
            })  # fmt: skip
        if cycle.write is not None:
            for b in range(spec.bits):
                memory[(cycle.write, b)] = cycle.data >> b & 1
            landed = sum(stored(cycle.write, b, end) << b for b in range(spec.bits))
            i = cycle.write >> 2
            margin = None
            for b in range(spec.bits):
                line = f"xm.wwl{spec.half_of(b).lower()}_{i}"
                falls = _crossings(time, trace[line], half)
                falls = falls[(falls > edge) & (falls < end)]
                q = (
                    trace[f"{_cell(cycle.write, b)}.q"]
                    - trace[f"{_cell(cycle.write, b)}.qb"]
                )
                flips = _crossings(time, q, 0.0)
                flips = flips[(flips > edge) & (flips < end)]
                if len(flips) and len(falls) >= 2:
                    m = (falls[1] - flips[-1]) * 1000
                    margin = m if margin is None else min(margin, m)
            if margin is not None:
                margins.append(margin)
            writes.append({
                "cycle": n, "address": cycle.write, "data": cycle.data, "stored": landed,
                "ok": landed == cycle.data, "margin_ps": margin,
            })  # fmt: skip
    final = (len(cycles) + 1.4) * period
    errors = [key for key, v in memory.items() if stored(*key, final) != v]
    passed = (
        not errors
        and all(r["ok"] is not False for r in reads)
        and all(w["ok"] for w in writes)
    )
    return MacroResult(
        corner.name, period, reads, writes, errors,
        max(clk_q) if clk_q else None, min(margins) if margins else None, passed,
    )  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wordlines", type=int, default=32)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--period", type=float, default=2.0, help="ns")
    parser.add_argument(
        "--corners", nargs="+", choices=sorted(CORNERS), default=["typical"]
    )
    parser.add_argument("--out", type=Path, default=Path("build/sram1r1w/macro"))
    args = parser.parse_args(argv)
    spec = MacroSpec(args.wordlines, args.bits)
    osdi = default_osdi(args.out)
    report, failed = {}, False
    for name in args.corners:
        result = run_macro_bench(spec, CORNERS[name], osdi, args.out / f"{name}_{args.period:g}ns",
                                 period=args.period)  # fmt: skip
        report[name] = asdict(result)
        failed |= not result.passed
        print(f"{name} {args.period:g} ns: {'pass' if result.passed else 'FAIL'}; "
              f"CLK->DOUT {result.clk_to_dout_ps}; write margin {result.min_write_margin_ps}; "
              f"final errors {result.final_errors}")  # fmt: skip
        for r in result.reads:
            print("  read ", r)
        for w in result.writes:
            print("  write", w)
    corners = "_".join(args.corners)
    (args.out / f"macro_{corners}_{args.period:g}ns.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
