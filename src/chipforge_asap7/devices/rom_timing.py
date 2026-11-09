"""Simulate a NOR ROM macro's read: ngspice, BSIM-CMG, a reduced netlist.

A 1024 x 148 macro has a quarter of a million transistors, nearly all of them
idle in any one read, so `read_path` keeps what one read uses and replaces the
rest with what the kept nets feel of them:

- a removed gate on a kept net becomes gate load, one merged device per model
  and net, its drain and source on the rail;
- a removed drain or source on a kept net becomes an off device of the same
  fins.  Its other side is the rail when the original's was, so it leaks as
  the original does; otherwise it joins the net, for its capacitance alone.

What stays: the controller, the address latches, the predecoders, the decoder
gates and bits of the words read, and the lane leaves of one data bit and the
replica.  The worst data bit is bit 0, the far end of its wordline and of the
PRECHN and SAMPLE columns, and the worst words are the two furthest from the
lane, so `read_sites` reads those with every other word of that bit programmed.

**Wires are estimates.**  `wire_estimates` gives the long nets a resistance and
capacitance from their drawn length, the stack's sheet values and a flat
capacitance per micron, as a ladder.  Nothing is extracted, and the supplies are
ideal.  A result is therefore evidence about the circuit and an estimate of its
delay, not sign-off: a report's ``timing_qualified`` stays false.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .rom_decoder import RomDecoderSpec
from .rom_netlist import Transistor, rom_macro_devices
from .rom_views import LIBERTY_LOADS_FF, estimate_timing
from .spice import (
    _osdi_command_path,
    compile_bsimcmg107,
    find_default_bsimcmg107_source,
    find_default_model_card,
    translate_asap7_model,
)

__all__ = [
    "CORNERS",
    "LOADS_FF",
    "CornerTiming",
    "ReadPath",
    "RomCorner",
    "Stimulus",
    "Waveforms",
    "characterize",
    "characterize_corner",
    "circuit_digest",
    "default_osdi",
    "load_report",
    "read_errors",
    "read_path",
    "read_sites",
    "read_words",
    "render_deck",
    "search_limit",
    "simulate",
    "wire_estimates",
]

RAILS = {"VDD", "VSS"}
#: How short of its stop time a transient may end and still count as finished.
STOP_TOLERANCE_PS = 0.01
#: Segments of a wire's ladder.
LADDER = 4
#: M2 and M3 are 18 nm wide, M4 and M5 24 nm: ohms per micron from the stack's
#: sheet resistances (`chipforge_asap7.verification.stack`).
OHM_PER_UM = {
    "M2": 3.03147 / 0.018,
    "M3": 3.03147 / 0.018,
    "M4": 2.27360 / 0.024,
    "M5": 2.27360 / 0.024,
}
#: As the B-ROM's wire estimates assume (`chipforge_asap7.brom.wire_rc`).
WIRE_FF_PER_UM = 0.2


@dataclass(frozen=True)
class RomCorner:
    """A process corner: which ASAP7 card, at what supply and temperature."""

    name: str
    card: str
    vdd: float
    temperature_c: float

    def model_card(self) -> Path:
        default = find_default_model_card()
        if default is None:
            raise FileNotFoundError("no ASAP7 model card: set ASAP7_MODEL_CARD")
        return default.with_name(self.card)


#: The corners the B-ROM's characterization reports, by Liberty corner name.
CORNERS = {
    "typical": RomCorner("typical", "7nm_TT_160803.pm", 0.70, 25),
    "fast": RomCorner("fast", "7nm_FF_160803.pm", 0.77, -40),
    "slow": RomCorner("slow", "7nm_SS_160803.pm", 0.63, 125),
}


# ── The reduced netlist ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class Load:
    """Merged devices standing in for removed ones on `net`."""

    model: str
    length: int
    net: str
    kind: str  # "gate", "leak" (off, to the rail) or "cap" (off, both ends on net)
    fins: int


@dataclass(frozen=True)
class ReadPath:
    """The devices one read needs, and the rest as loads on them."""

    devices: tuple[Transistor, ...]
    loads: tuple[Load, ...]
    removed: int


def read_words(spec: RomDecoderSpec) -> tuple[int, int]:
    """The two words furthest from the lane: the first and the last."""
    return 0, spec.array.words - 1


def read_sites(spec: RomDecoderSpec) -> list[tuple[int, int]]:
    """The worst image for bit 0: set in every word but the last.

    The first word reads a 1 down a fully loaded bitline; the last reads a 0
    while every other word's device leaks on it.  The other bits never matter.
    """
    last = spec.array.words - 1
    return [(w, b) for w in range(spec.array.words) for b in range(spec.array.bits)
            if not (b == 0 and w == last)]  # fmt: skip


_ARRAY = re.compile(r"^MA[LR]_W\d+_B(\d+)$")
_DECODER = re.compile(r"^MDW(\d+)_")
_LEAF = re.compile(r"^MS_\w+_B(\d+)$")
_ROW = re.compile(r"^MS_\w+_R(\d+)$")
_WORD = re.compile(r"^WL<(\d+)>$")


def read_path(
    spec: RomDecoderSpec,
    programmed: Iterable[tuple[int, int]],
    words: Sequence[int],
    bits: Sequence[int],
) -> ReadPath:
    """Reduce the macro to reads of `words`, watching data `bits`.

    The replica's leaf and the bits' rows always stay.
    """
    lane = spec.sense_lane
    kept_bits = {*bits, spec.replica_bit}
    rows = {lane.plan[b].row for b in kept_bits}
    lines = {f"WL<{w}>" for w in words}

    def keep(d: Transistor) -> bool:
        if match := _ARRAY.match(d.name):
            return d.gate in lines and int(match[1]) in kept_bits
        if d.name.startswith(("MAL_ISO", "MAR_ISO")):
            return False
        if match := _DECODER.match(d.name):
            return int(match[1]) in words
        if match := _LEAF.match(d.name):
            return int(match[1]) in kept_bits
        if match := _ROW.match(d.name):
            return int(match[1]) in rows
        return True

    every = rom_macro_devices(spec, programmed)
    kept = tuple(d for d in every if keep(d))
    live = {n for d in kept for n in (d.drain, d.gate, d.source)} - RAILS
    fins: Counter[tuple[str, int, str, str]] = Counter()
    for d in every:
        if keep(d):
            continue
        if d.gate in live:
            fins[d.model, d.length, d.gate, "gate"] += d.fins
        for net, other in ((d.drain, d.source), (d.source, d.drain)):
            if net in live:
                kind = "leak" if other in RAILS else "cap"
                fins[d.model, d.length, net, kind] += d.fins
    loads = tuple(
        Load(model, length, net, kind, count)
        for (model, length, net, kind), count in sorted(fins.items())
    )
    return ReadPath(kept, loads, len(every) - len(kept))


# ── Wires ─────────────────────────────────────────────────────────────────────
def _wire(segments: Iterable[tuple[str, float]]) -> tuple[float, float]:
    ohms = ff = 0.0
    for layer, um in segments:
        ohms += OHM_PER_UM[layer] * um
        ff += WIRE_FF_PER_UM * um
    return ohms, ff


def wire_estimates(spec: RomDecoderSpec) -> dict[str, tuple[float, float]]:
    """Ohms and fF of each long net's patterns, by a regular expression on its name.

    Each is the run from the net's driver to its far end, along the layer it
    is drawn on:

    - a bitline: M2 across a half, then M4 across half the lane;
    - a wordline: M3 down the array from the decoder;
    - a decoder line, PA or PB: M4 from the lane across a half;
    - an address line and EVAL: M4 across half the lane to the predecoders;
    - PRECHN and SAMPLE: M3 down the lane;
    - a row's SAMPLEB and SAMPLEL: M2 along it;
    - READY, and the CLK, A and Q pins: an M5 column.
    """
    half_um = spec.half.width / 1000
    lane_um = (spec.right_x - spec.half.width) / 1000
    array_um = spec.cells.height / 1000
    macro_um = array_um + spec.height / 1000
    return {
        r"BL[LR]<\d+>": _wire((("M2", half_um), ("M4", lane_um / 2))),
        r"WL<\d+>": _wire((("M3", array_um),)),
        r"P[AB]<\d+>": _wire((("M4", half_um + lane_um / 2),)),
        r"ALN?<\d+>|EVAL": _wire((("M4", lane_um / 2),)),
        r"PRECHN|SAMPLE": _wire((("M3", array_um),)),
        r"S_SAMPLE[BL]_R\d+": _wire((("M2", lane_um / 2),)),
        r"READY": _wire((("M5", array_um),)),
        r"CLK|A\[\d+\]": _wire((("M5", macro_um),)),
        r"Q\[\d+\]": _wire((("M5", array_um),)),
    }


def _along(spec: RomDecoderSpec, d: Transistor, net: str) -> float:
    """How far along `net`'s wire `d` stands, from the driver: 0 to 1."""
    half = spec.half.words
    if net.startswith(("BLL", "BLR")):
        if d.drain != net and d.source != net or not d.name.startswith("MA"):
            return 0.0
        match = re.match(r"^MA([LR])_W(\d+)_", d.name)
        word = int(match[2])
        return 1 - word / half if match[1] == "L" else (word + 1) / half
    if net.startswith("WL<"):
        if not d.name.startswith("MA"):
            return 0.0
        bit = int(_ARRAY.match(d.name)[1])
        return 1 - bit / spec.cells.bits
    if net.startswith(("PA<", "PB<")):
        word = int(_DECODER.match(d.name)[1]) if _DECODER.match(d.name) else None
        if word is None:
            return 0.0
        side = word - half if word >= half else half - 1 - word
        return (side + 1) / half
    if net in ("PRECHN", "SAMPLE"):
        return _lane_row(spec, d)
    if net.startswith(("S_SAMPLEB", "S_SAMPLEL")):
        return 1.0 if _LEAF.match(d.name) else 0.0
    if net == "CLK" or net.startswith(("A[", "READY")):
        return 1.0
    if net.startswith("Q["):
        return 0.0
    return 1.0 if d.gate == net else 0.0


def _lane_row(spec: RomDecoderSpec, d: Transistor) -> float:
    """Leaves and clock inverters by their row, down from the controller."""
    lane = spec.sense_lane
    if match := _LEAF.match(d.name):
        row = lane.plan[int(match[1])].row
    elif match := _ROW.match(d.name):
        row = int(match[1])
    else:
        return 0.0
    return 1 - row / lane.rows


def _node(net: str, step: int) -> str:
    return net if step == 0 else f"{net}@{step}"


def _ladders(
    spec: RomDecoderSpec, path: ReadPath
) -> tuple[list[tuple[str, str, str, str]], tuple[Transistor, ...], list[Load]]:
    """Each ladder element ``(name, net, net, value)``, and devices and loads
    moved onto the ladders' nodes."""
    wires = {re.compile(rf"^(?:{k})$"): v for k, v in wire_estimates(spec).items()}
    nets = {n for d in path.devices for n in (d.drain, d.gate, d.source)} - RAILS
    lines, wired = [], {}
    for net in sorted(nets):
        for pattern, (ohms, ff) in wires.items():
            if pattern.match(net):
                wired[net] = ohms, ff
                break
    for i, (net, (ohms, ff)) in enumerate(wired.items()):
        for step in range(LADDER):
            near, far = _node(net, step), _node(net, step + 1)
            lines.append((f"RW{i}_{step}", near, far, f"{ohms / LADDER:.3f}"))
            lines.append((f"CW{i}_{step}", far, "VSS", f"{ff / LADDER:.4f}f"))

    def moved(d: Transistor) -> Transistor:
        def at(net: str) -> str:
            if net not in wired:
                return net
            return _node(net, round(_along(spec, d, net) * LADDER))

        return replace(d, drain=at(d.drain), gate=at(d.gate), source=at(d.source))

    loads = []
    for load in path.loads:
        if load.net not in wired:
            loads.append(load)
            continue
        share, rest = divmod(load.fins, LADDER)
        for step in range(LADDER):
            fins = share + (step < rest)
            if fins:
                loads.append(replace(load, net=_node(load.net, step + 1), fins=fins))
    return lines, tuple(moved(d) for d in path.devices), loads


# ── The deck ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Stimulus:
    """Reads of `reads`, each a CLK cycle, low first; times in ps.

    The address arrives `setup` before CLK falls and, with `hold` given, turns
    to its complement that long after.  `tail` is more time after the last
    high phase, for an output still on its way.
    """

    reads: tuple[int, ...]
    low: float = 600.0
    high: float = 600.0
    setup: float = 200.0
    hold: float | None = None
    slew: float = 5.0
    load_ff: float = 1.0
    max_step: float = 1.0
    tail: float = 0.0

    @property
    def first_fall(self) -> float:
        """Reset, released while CLK is high, leaves one high phase to recover."""
        return 2 * self.high + 2 * max(self.setup, 0) + 100.0

    def fall(self, read: int) -> float:
        return self.first_fall + read * (self.low + self.high)

    def rise(self, read: int) -> float:
        return self.fall(read) + self.low

    @property
    def stop(self) -> float:
        return self.fall(len(self.reads)) + self.high + self.tail


def _pwl(points: list[tuple[float, int]], vdd: float, slew: float) -> str:
    out, level = [f"0 {points[0][1] * vdd:g}"], points[0][1]
    for t, value in sorted(points[1:]):
        if value != level:
            out += [f"{t:.3f}p {level * vdd:g}", f"{t + slew:.3f}p {value * vdd:g}"]
            level = value
    return "PWL(" + " ".join(out) + ")"


def _clock(s: Stimulus) -> list[tuple[float, int]]:
    points = [(0.0, 1)]
    for read in range(len(s.reads)):
        points += [(s.fall(read), 0), (s.rise(read), 1)]
    return points


def _address(s: Stimulus, bit: int) -> list[tuple[float, int]]:
    value = lambda read: s.reads[read] >> bit & 1
    points = [(0.0, value(0))]
    for read in range(len(s.reads)):
        centre = s.fall(read) + s.slew / 2
        points.append((centre - s.setup - s.slew / 2, value(read)))
        if s.hold is not None:
            points.append((centre + s.hold - s.slew / 2, 1 - value(read)))
    return points


def render_deck(
    spec: RomDecoderSpec,
    path: ReadPath,
    stimulus: Stimulus,
    corner: RomCorner,
    osdi: Path,
    watch: Sequence[str],
    *,
    wires: bool = True,
) -> tuple[str, list[str]]:
    """The deck, and the nodes it writes to ``waveform.dat`` in order."""
    if wires:
        rc, devices, loads = _ladders(spec, path)
    else:
        rc, devices, loads = [], path.devices, list(path.loads)
    nets = {n for d in devices for n in (d.drain, d.gate, d.source, d.bulk)}
    nets |= {load.net for load in loads}
    nets |= {net for _, a, b, _ in rc for net in (a, b)}
    names = {net: f"n{i}" for i, net in enumerate(sorted(nets))}
    # An output's load, and what its reader sees, are at the pin: the wire's end.
    for net in [n for n in names if n.startswith("Q[") and "@" not in n]:
        names[net] = names.get(_node(net, LADDER), names[net])
    names.update(VSS="0", VDD="vdd")
    vdd = corner.vdd
    lines = [
        f"* NOR ROM read at the {corner.name} corner",
        ".include models.pm",
        f".temp {corner.temperature_c:g}",
        ".options reltol=1e-3 abstol=1e-12 vntol=1e-6 method=gear rshunt=1e10",
        f"Vsupply vdd 0 {vdd:g}",
        f"VCLK {names['CLK']} 0 {_pwl(_clock(stimulus), vdd, stimulus.slew)}",
        f"VRESETN {names['RESETN']} 0 "
        + _pwl([(0.0, 0), (stimulus.high, 1)], vdd, stimulus.slew),
    ]
    for bit in range(spec.address_bits):
        points = _address(stimulus, bit)
        lines.append(
            f"VA{bit} {names[f'A[{bit}]']} 0 {_pwl(points, vdd, stimulus.slew)}"
        )
    for b in range(spec.array.bits):
        if f"Q[{b}]" in names and stimulus.load_ff > 0:
            lines.append(f"CQ{b} {names[f'Q[{b}]']} 0 {stimulus.load_ff:g}f")
    lines += [f"{name} {names[a]} {names[b]} {value}" for name, a, b, value in rc]
    for d in devices:
        lines.append(
            f"N{d.name} {names[d.drain]} {names[d.gate]} {names[d.source]} "
            f"{names[d.bulk]} {d.model} L={d.length}n NFIN={d.fins} NF=1"
        )
    for i, load in enumerate(loads):
        rail = "0" if load.model.startswith("n") else "vdd"
        net = names[load.net]
        terminals = {
            "gate": (rail, net, rail),
            "leak": (net, rail, rail),
            "cap": (net, rail, net),
        }[load.kind]
        lines.append(
            f"NLOAD{i} {terminals[0]} {terminals[1]} {terminals[2]} {rail} "
            f"{load.model} L={load.length}n NFIN={load.fins} NF=1"
        )
    nodes = [names[w] for w in watch]
    vectors = [f"v({node})" for node in nodes]
    lines += [
        ".control",
        "set num_threads=1",
        f"pre_osdi {_osdi_command_path(osdi)}",
        "set wr_singlescale",
        "set wr_vecnames",
        "set numdgt=9",
        "save " + " ".join(vectors),
        f"tran {stimulus.max_step:g}p {stimulus.stop:.3f}p 0 {stimulus.max_step:g}p",
        "wrdata waveform.dat " + " ".join(vectors),
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines), nodes


# ── Running it ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Waveforms:
    """Time in ps and each watched net's volts."""

    time: np.ndarray
    traces: dict[str, np.ndarray]
    vdd: float

    def at(self, net: str, t: float) -> float:
        return float(np.interp(t, self.time, self.traces[net]))

    def crossing(
        self, net: str, start: float, end: float, *, rising: bool, level: float = 0.5
    ) -> float | None:
        """When `net` first crosses `level` of VDD in the given direction."""
        threshold = level * self.vdd
        inside = (self.time >= start) & (self.time <= end)
        t, v = self.time[inside], self.traces[net][inside]
        above = v >= threshold
        hits = np.flatnonzero(
            above[1:] & ~above[:-1] if rising else ~above[1:] & above[:-1]
        )
        if not len(hits):
            return None
        i = hits[0]
        return float(t[i] + (threshold - v[i]) * (t[i + 1] - t[i]) / (v[i + 1] - v[i]))


def simulate(
    spec: RomDecoderSpec,
    path: ReadPath,
    stimulus: Stimulus,
    corner: RomCorner,
    osdi: Path,
    watch: Sequence[str],
    out: Path,
    *,
    wires: bool = True,
    timeout_s: float = 3600,
) -> Waveforms:
    """Run one transient in `out` and read the watched nets back."""
    out.mkdir(parents=True, exist_ok=True)
    card = corner.model_card().read_text()
    models = sorted({d.model for d in path.devices} | {x.model for x in path.loads})
    (out / "models.pm").write_text(
        "\n".join(translate_asap7_model(card, model) for model in models)
    )
    deck, _ = render_deck(spec, path, stimulus, corner, osdi, watch, wires=wires)
    (out / "read.sp").write_text(deck)
    waveform = out / "waveform.dat"
    waveform.unlink(missing_ok=True)
    simulator = shutil.which("ngspice")
    if simulator is None:
        raise FileNotFoundError("ngspice was not found")
    done = subprocess.run(
        [simulator, "-b", "read.sp"],
        cwd=out,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    (out / "read.log").write_text(done.stdout + done.stderr)
    if done.returncode or not waveform.is_file():
        raise RuntimeError(
            f"ngspice failed (exit {done.returncode}); see {out}/read.log"
        )
    data = np.loadtxt(waveform, skiprows=1)
    if data.ndim != 2 or data[-1, 0] * 1e12 < stimulus.stop - STOP_TOLERANCE_PS:
        raise RuntimeError(f"incomplete transient; see {out}/read.log")
    traces = {name: data[:, i + 1] for i, name in enumerate(watch)}
    return Waveforms(data[:, 0] * 1e12, traces, corner.vdd)


# ── Characterizing ────────────────────────────────────────────────────────────
#: A report's format, so that a reader can refuse one it doesn't know.
REPORT_FORMAT = "chipforge-asap7-rom-timing/1"
#: The output loads ``CLK`` to ``Q`` is measured at, in fF.
LOADS_FF = LIBERTY_LOADS_FF
#: The load every other measurement runs at, in fF.
NOMINAL_LOAD_FF = 1.0
#: How long the droop run holds ``CLK`` low, in ps.
DROOP_LOW_PS = 6000.0
#: The fraction of VDD `max_low` extrapolates a bitline down to.  The leaf's NAND
#: hasn't flipped by then: at 1024 words and the slow corner, a bitline at 0.39 V
#: of 0.63 V after 20 ns left ``D`` at 7% of VDD.
DROOP_LIMIT = 2 / 3
#: A logic level is good within this fraction of VDD of its rail.
LEVEL_TOLERANCE = 0.2


@dataclass(frozen=True)
class CornerTiming:
    """One corner's measurements in ps, with no margin added.

    ``clk_q_*`` and ``q_*_slew`` hold one value per load in `LOADS_FF`; a slew is
    20% to 80% of the swing.  ``max_low`` is an extrapolation from `droop`,
    the worst bitline's fall in mV/ns while ``CLK`` is low, and is ``None`` if it
    did not fall.
    """

    clk_q_rise: tuple[float, ...]
    clk_q_fall: tuple[float, ...]
    q_rise_slew: tuple[float, ...]
    q_fall_slew: tuple[float, ...]
    setup: float
    hold: float
    min_low: float
    min_high: float
    droop: float
    max_low: float | None
    aborted: int = 0


def read_errors(
    w: Waveforms, stimulus: Stimulus, expected: Sequence[int], net: str = "Q[0]"
) -> list[str]:
    """What is wrong with `net` over `stimulus`'s reads; empty if nothing is.

    Each read's value has to be there when the next ``CLK`` rise comes, so a
    value that passes through the register early shows as the wrong word then.
    """
    errors = []
    swing = LEVEL_TOLERANCE * w.vdd
    for k, want in enumerate(expected):
        last = k + 1 == len(expected)
        when = stimulus.stop if last else stimulus.rise(k + 1)
        value = w.at(net, when)
        if abs(value - want * w.vdd) > swing:
            errors.append(
                f"read {k}: {net} is {value:.3f} V at {when:g} ps, not {want}"
            )
    return errors


def search_limit(
    passes: Callable[[Sequence[float]], Sequence[bool]],
    low: float,
    high: float,
    *,
    points: int = 6,
    resolution: float = 5.0,
) -> float:
    """The smallest value found to pass, `low` taken to fail and `high` to pass.

    Each round tries `points` values at once.  A failure above a pass moves the
    bracket above that failure, so a limit that isn't monotonic comes out on the
    safe side.
    """
    while high - low > resolution:
        step = (high - low) / (points + 1)
        values = [low + step * (i + 1) for i in range(points)]
        results = list(passes(values))
        failed = [v for v, ok in zip(values, results, strict=True) if not ok]
        if failed:
            low = max(failed)
        passed = [v for v, ok in zip(values, results, strict=True) if ok and v > low]
        if passed:
            high = min(passed)
    return high


def default_osdi(build: Path) -> Path:
    """BSIM-CMG's OSDI library under `build`, compiled the first time."""
    output = build / "BSIMCMG107.osdi"
    if output.is_file():
        return output.resolve()
    source = find_default_bsimcmg107_source()
    if source is None:
        raise FileNotFoundError(
            "no BSIM-CMG 107 source: set BSIMCMG107_SOURCE or pass an OSDI library"
        )
    return compile_bsimcmg107(source, output)


def _edge(
    w: Waveforms, start: float, end: float, *, rising: bool
) -> tuple[float | None, float | None]:
    """``Q[0]``'s 50% crossing after `start`, and its 20% to 80% time."""
    mid = w.crossing("Q[0]", start, end, rising=rising)
    first = w.crossing("Q[0]", start, end, rising=rising, level=0.2 if rising else 0.8)
    second = w.crossing("Q[0]", start, end, rising=rising, level=0.8 if rising else 0.2)
    if mid is None or first is None or second is None:
        return None, None
    return mid - start, second - first


def characterize_corner(
    spec: RomDecoderSpec,
    corner: RomCorner,
    osdi: Path,
    out: Path,
    *,
    jobs: int = 6,
    resolution: float = 10.0,
    timeout_s: float = 3600,
    path: ReadPath | None = None,
) -> CornerTiming:
    """Measure one corner, running up to `jobs` simulations at a time.

    A read of the two furthest words back and forth takes ``Q[0]`` both ways.
    Every limit is searched with the others relaxed, so they don't add: a
    period at both minimums together has not been run.
    """
    a, b = read_words(spec)
    reads = (a, b, a)
    expected = (1, 0, 1)
    path = path or read_path(spec, read_sites(spec), (a, b), (0,))
    aborted = []
    numbers = itertools.count()

    def run(stimulus: Stimulus) -> Waveforms:
        return simulate(
            spec, path, stimulus, corner, osdi, ("Q[0]",),
            out / f"run{next(numbers):03d}", timeout_s=timeout_s,
        )  # fmt: skip

    def good(stimulus: Stimulus) -> bool:
        try:
            return not read_errors(run(stimulus), stimulus, expected)
        except (RuntimeError, subprocess.TimeoutExpired):
            aborted.append(stimulus)
            return False

    def many(make: Callable[[float], Stimulus]) -> Callable[..., list[bool]]:
        def passes(values: Sequence[float]) -> list[bool]:
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                return list(pool.map(lambda v: good(make(v)), values))

        return passes

    # A cycle long enough to read, from the estimate and then by doubling.
    guess = estimate_timing(spec).at(corner.name)
    low = max(400.0, round(guess.min_low))
    high = max(400.0, round(guess.min_high))
    for _ in range(4):
        if good(Stimulus(reads, low=low, high=high)):
            break
        low, high = 2 * low, 2 * high
    else:
        raise RuntimeError(
            f"no read succeeded at the {corner.name} corner, up to CLK low {low:g} ps"
        )
    base = Stimulus(reads, low=low, high=high)

    # CLK to Q at each load.
    def measure(load: float) -> tuple[tuple[float, float], tuple[float, float]]:
        stimulus = replace(base, load_ff=load)
        w = run(stimulus)
        errors = read_errors(w, stimulus, expected)
        if errors:
            raise RuntimeError(f"at {load:g} fF: {errors[0]}")
        settle = stimulus.slew / 2
        rise = _edge(w, stimulus.rise(2) + settle, stimulus.stop, rising=True)
        fall = _edge(w, stimulus.rise(1) + settle, stimulus.rise(2), rising=False)
        if None in rise or None in fall:
            raise RuntimeError(f"Q[0] did not switch twice at {load:g} fF")
        return rise, fall

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        edges = list(pool.map(measure, LOADS_FF))

    # Each limit, one at a time.  Setup can be negative: the address latches
    # close only after PRECHN has risen.
    setup_floor = -low / 2
    if many(lambda s: replace(base, setup=s))([setup_floor])[0]:
        setup = setup_floor
    else:
        setup = search_limit(
            many(lambda s: replace(base, setup=s)),
            setup_floor,
            base.setup,
            points=jobs,
            resolution=resolution,
        )
    if many(lambda h: replace(base, hold=h))([0.0])[0]:
        hold = 0.0
    else:
        if not many(lambda h: replace(base, hold=h))([low / 2])[0]:
            raise RuntimeError(f"address hold of {low / 2:g} ps does not read")
        hold = search_limit(
            many(lambda h: replace(base, hold=h)),
            0.0,
            low / 2,
            points=jobs,
            resolution=resolution,
        )
    min_low = search_limit(
        many(lambda v: replace(base, low=v)),
        base.slew,
        low,
        points=jobs,
        resolution=resolution,
    )
    min_high = search_limit(
        many(lambda v: replace(base, high=v, tail=high)),
        base.slew,
        high,
        points=jobs,
        resolution=resolution,
    )
    droop, max_low = _droop(spec, path, corner, osdi, out / "droop", timeout_s)
    return CornerTiming(
        clk_q_rise=tuple(round(e[0][0], 2) for e in edges),
        clk_q_fall=tuple(round(e[1][0], 2) for e in edges),
        q_rise_slew=tuple(round(e[0][1], 2) for e in edges),
        q_fall_slew=tuple(round(e[1][1], 2) for e in edges),
        setup=round(setup, 2),
        hold=round(hold, 2),
        min_low=round(min_low, 2),
        min_high=round(min_high, 2),
        droop=round(droop, 3),
        max_low=None if max_low is None else round(max_low),
        aborted=len(aborted),
    )


def _droop(
    spec: RomDecoderSpec,
    path: ReadPath,
    corner: RomCorner,
    osdi: Path,
    out: Path,
    timeout_s: float,
) -> tuple[float, float | None]:
    """The worst bitline's fall in mV/ns with ``CLK`` held low, and how long
    ``CLK`` can stay low before it reaches `DROOP_LIMIT` of VDD, in ps.

    The read selects a word with no bit here and every other word has one, so
    nothing holds the line up and all of them leak.  The rate falls with the
    voltage, so a straight line from the end of the run is on the safe side.
    """
    _, word = read_words(spec)
    nets = ("BLL<0>", "BLR<0>")
    stimulus = Stimulus((word,), low=DROOP_LOW_PS, high=1500.0)
    w = simulate(spec, path, stimulus, corner, osdi, nets, out, timeout_s=timeout_s)
    start = stimulus.fall(0)
    middle, end = start + DROOP_LOW_PS / 2, start + DROOP_LOW_PS
    level, rate = w.vdd, 0.0
    for net in nets:
        now = w.at(net, end)
        slope = (w.at(net, middle) - now) / (end - middle)  # V/ps
        level = min(level, now)
        rate = max(rate, slope)
    if rate <= 0:
        return 0.0, None
    return rate * 1e6, DROOP_LOW_PS + (level - DROOP_LIMIT * w.vdd) / rate


def characterize(
    spec: RomDecoderSpec,
    corners: Sequence[str],
    osdi: Path,
    out: Path,
    *,
    jobs: int = 6,
    resolution: float = 10.0,
    timeout_s: float = 3600,
) -> dict:
    """Measure `corners` and return the report ``--timing`` reads.

    A corner whose simulations cannot finish is recorded under ``failed``
    instead of ending the rest.
    """
    a, b = read_words(spec)
    path = read_path(spec, read_sites(spec), (a, b), (0,))
    measured, failed = {}, {}
    for name in corners:
        try:
            measured[name] = asdict(
                characterize_corner(
                    spec,
                    CORNERS[name],
                    osdi,
                    out / name,
                    jobs=jobs,
                    resolution=resolution,
                    timeout_s=timeout_s,
                    path=path,
                )
            )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            failed[name] = str(exc)
    return {
        "format": REPORT_FORMAT,
        "macro": spec.macro_name,
        "circuit_sha256": circuit_digest(spec),
        "words": spec.array.words,
        "bits": spec.array.bits,
        "bit": 0,
        "read_words": [a, b],
        "programming": "worst case: every bit of the measured bit's column but one",
        "devices_simulated": len(path.devices),
        "devices_removed": path.removed,
        "input_slew_ps": Stimulus((0,)).slew,
        "loads_ff": list(LOADS_FF),
        "nominal_load_ff": NOMINAL_LOAD_FF,
        "wires": "estimated",
        "extracted": False,
        "timing_qualified": False,
        "corners": measured,
        "failed": failed,
    }


def circuit_digest(spec: RomDecoderSpec) -> str:
    """A hash of the netlist a report measures, so a changed circuit refuses it.

    A macro's name only says its shape.  Programming is `read_sites`, the same
    for every image.
    """
    devices = sorted(
        (d.name, d.drain, d.gate, d.source, d.bulk, d.model, d.length, d.fins)
        for d in rom_macro_devices(spec, read_sites(spec))
    )
    return hashlib.sha256(repr(devices).encode()).hexdigest()


def load_report(path: Path, spec: RomDecoderSpec) -> dict[str, CornerTiming]:
    """The measured corners of the report at `path`, which must be of `spec` as it
    is now."""
    report = json.loads(Path(path).read_text())
    if report.get("format") != REPORT_FORMAT:
        raise ValueError(f"{path} is not a {REPORT_FORMAT} report")
    if report.get("macro") != spec.macro_name:
        raise ValueError(
            f"{path} measured {report.get('macro')}, not {spec.macro_name}"
        )
    if report.get("circuit_sha256") != circuit_digest(spec):
        raise ValueError(f"{path} measured a different circuit: characterize it again")
    return {
        name: CornerTiming(
            **{
                key: tuple(value) if isinstance(value, list) else value
                for key, value in corner.items()
            }
        )
        for name, corner in report["corners"].items()
    }
