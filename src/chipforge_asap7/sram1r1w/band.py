"""The ``PS`` rank between the two halves' strip pairs, and the wordline core.

Both halves' strip pairs face the centre band with their bottom edge, where
each slice takes its two ``SEL`` lines (`chipforge_asap7.sram1r1w.strips`):
port B's (read) on the slice's own M3, port A's (write) on an M5 track.  The
rank is two `GateRowSpec` rows between them, one gate pair a slice, each
slice's gates lined up under it:

::

    ── upper strip pair (its bottom edge: the slice's top rail) ─────────
       write rank: NAND, gap, INV
       read rank, mirrored: NAND, gap, INV  (sharing the write row's rail)
    ── lower strip pair, mirrored ───────────────────────────────────────

Each row's outer rail is the slices' top rail, VDD or VSS, so the rows turn
over with it.  A slice is 8 gate columns.  Each row holds its port's
``PSN = NAND(a...)`` and ``PS = INV(PSN)`` there, so ``PS``'s inverter has
one finger:

* the read row's ``PS`` output column is the slices' ``SEL`` x, and runs on
  M3 the band's full height, straight into both pairs' port B ``SEL``;
* the write row leaves that column free for it.  Its ``PS`` goes up a V3 to
  an M4 jog, and up a V4 to the slice's port A M5 track, which runs the band's
  full height into both pairs.

The address literals come in on each row's M2 tracks from its right edge.
A rank of up to three address bits (32 wordlines) fits a row's seven M2
levels: six literals and the ``PSN`` nets.

Three grids constrain the band, all absolute:

* the pairs' M1 ties their sources to the rail they share with the band
  every 108 nm, and a V0 where one meets a rank row's supply stub would sit
  in a T of M1 (``V0.M1.AUX.3``): a NAND moves a column left, or turns over,
  to keep its outer supply contacts off those x;
* the lower pair is the upper one mirrored about the band; both stay on the
  M4 tracks (12 nm past a multiple of 48) only if the band's height is
  ``-2 P - 24`` modulo 48 nm, ``P`` the pair's height, so the rows' bands
  are sized for it (`rank_bands`);
* the fins (27 nm) and those tracks set where the band's foot sits in the
  core's frame (`band_foot`).

Units are um; the rows are drawn in nm by `build_gate_row` and read back.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..devices.gate_row import Gate, GateRowSpec, build_gate_row
from ..devices.rom_control import LEVELS
from ..layout.layers import LAYERS
from ..devices.rom_latch import M3_EDGE
from ..devices.rowcell import PAD
from ..layout.transform import edge_boundary, rect, require_gdstk
from .cells import CellLibrary
from .column import build_column
from .periphery import RANK_BITS, MacroSpec, rank_row
from .strips import (
    _snap,
    _via_cells,
    build_strip_pair,
    strip_pair_netlist,
    strip_pair_pins,
)

__all__ = [
    "RankBand",
    "WordlineCore",
    "build_rank_band",
    "build_wordline_core",
    "rank_layout_row",
]

M2 = LAYERS["M2"]["layer"]
M3 = LAYERS["M3"]["layer"]
M4 = LAYERS["M4"]["layer"]
M5 = LAYERS["M5"]["layer"]
PIN_TEXTTYPE = 251
COLUMN = 0.054
#: Where a row's inverter stands, from the read line's column: the write
#: row's one column past it (the line's column is its gap), the read row's
#: one column before it (its output is the line).
INV_AT = {"W": 1, "R": -1}
#: The pairs' M1 stubs onto their bottom rail: every 108 nm, from 0.
STUB_PITCH = 108
#: The smallest band of a rank row, nm: room for its seven M2 levels.
MIN_BAND = 243


def rank_bands(pair_height: float) -> tuple[int, int]:
    """``(n, p)`` bands, nm, whole fin pitches, for rows that keep a mirrored
    pair of `pair_height` (um) on its M4 tracks: twice their sum is
    ``-2 P - 24`` modulo 48 nm."""
    p_nm = round(pair_height * 1000)
    want = (-2 * p_nm - 24) % 48
    sizes = range(MIN_BAND, MIN_BAND + 27 * 16, 27)
    fits = [
        (n + p, abs(n - p), n, p)
        for n in sizes
        for p in sizes
        if (2 * (n + p)) % 48 == want
    ]
    if not fits:
        raise ValueError(f"no rank rows keep a {p_nm} nm pair on its tracks")
    _, _, n, p = min(fits)
    return n, p


def band_foot(pair_height: float) -> float:
    """Where the band's foot sits in the core's frame, um: on the fin grid
    (27 nm), and ``P + 24`` modulo 48 nm, which puts the lower pair's M4 on
    its tracks."""
    p_nm = round(pair_height * 1000)
    return next(y for y in range(0, 432, 27) if (y - p_nm - 24) % 48 == 0) / 1000


def _sel_pins(pair: Any) -> dict[str, list[float]]:
    """Each port's ``SEL`` x for every slice, from the pair's labels."""
    xs: dict[str, dict[int, float]] = {"A": {}, "B": {}}
    for lb in pair.labels:
        if lb.text.startswith("SEL_"):
            port, k = lb.text[4], int(lb.text[6:-1])
            xs[port][k] = float(lb.origin[0])
    return {p: [v[k] for k in sorted(v)] for p, v in xs.items()}


def rank_layout_row(
    spec: MacroSpec,
    port: str,
    read_xs: list[float],
    x_left: float,
    x_right: float,
    outer: str = "p",
    bands: tuple[int, int] = (270, 270),
) -> GateRowSpec:
    """`rank_row` placed under the slices whose read ``SEL`` lines are at
    `read_xs`, as a row from `x_left` to `x_right` whose `outer` band
    (``"p"`` or ``"n"``) meets the pair."""
    if not 1 < spec.upper_bits <= RANK_BITS:
        raise NotImplementedError(
            f"the rank's layout takes 2 to {RANK_BITS} address bits, not {spec.upper_bits}"
        )
    row = rank_row(spec, port)

    def x_of(column: int) -> int:  # nm, absolute
        return round((x_left + (column + 1) * COLUMN) * 1000)

    def clear(gate: Gate, first: int) -> bool:
        cols = gate.bands[outer]
        return all(x_of(first + i) % STUB_PITCH for i, c in enumerate(cols) if c == "S")

    gates, at = [], []
    for s, x in enumerate(read_xs):
        # The read line's column: column c is at x_left + (c + 1) * COLUMN.
        line = round((x - x_left) / COLUMN) - 1
        if abs(x_left + (line + 1) * COLUMN - x) > 1e-6:
            raise ValueError(f"slice {s}'s SEL is off the row's gate grid")
        nand, inv = row.gates[2 * s], row.gates[2 * s + 1]
        inv_at = line + INV_AT[port]
        if not clear(inv, inv_at):
            raise RuntimeError(f"slice {s}: the PS inverter's supply is on a stub")
        options = [
            (inv_at - 1 - nand.columns - shift, replace(nand, mirrored=m))
            for shift in (0, 1)
            for m in (False, True)
        ]
        first, nand = next(((c, g) for c, g in options if clear(g, c)), (None, None))
        if nand is None:
            raise RuntimeError(f"slice {s}: no place for its PS NAND off the stubs")
        gates += [nand, inv]
        at += [first, inv_at]
    width = round((x_right - x_left) / COLUMN)
    # A tap at the right end: where the slices' strips are VSS-edged, the
    # rows' n-wells meet each other in the band's middle and no other well.
    row = replace(
        row,
        gates=tuple(gates),
        at=tuple(at),
        right=row.inputs,
        span=width - 1,
        tap=True,
        band_height=bands,
        name=f"{row.name}_rank",
    )
    if row.columns != width - 1:
        raise RuntimeError(f"no room for the {port} rank row's tap")
    if not x_of(width - 2) % STUB_PITCH:
        raise RuntimeError(f"the {port} rank row's tap tie is on a stub")
    return row


def _to_um(spec: GateRowSpec, library: Any, name: str) -> Any:
    """`build_gate_row`'s cell, read back into `library` in um."""
    import gdspy

    gdstk = require_gdstk()
    for cell in library.cells:
        if cell.name == name:
            return cell
    nm = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    previous = gdspy.current_library
    gdspy.current_library = nm
    try:
        build_gate_row(spec, name=name, lib=nm)
    finally:
        gdspy.current_library = previous
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "row.gds"
        nm.write_gds(str(path))
        um = gdstk.read_gds(str(path), unit=1e-6)
    cell = next(c for c in um.cells if c.name == name)
    library.add(cell)
    return cell


@dataclass(frozen=True)
class RankBand:
    """The two rank rows under a strip pair: the cell and the rows' specs."""

    cell: Any
    rows: dict[str, GateRowSpec]
    height: float


def build_rank_band(spec: MacroSpec, pair: Any, library: Any, foot: float) -> RankBand:
    """Both ports' rank rows for `pair`, in its x, drawn from y = 0 up.

    `foot` is where the band's foot sits in the frame whose M4 tracks the
    write lines' jogs keep to (12 nm past a multiple of 48).
    """
    gdstk = require_gdstk()
    x_left, _, x_right, ph = edge_boundary(pair)
    sel = _sel_pins(pair)
    edge = spec.slice.rails[-1][1]  # the pair's bottom rail
    outer = "p" if edge == "VDD" else "n"
    bands = rank_bands(ph)
    rows = {
        p: rank_layout_row(spec, p, sel["B"], x_left, x_right, outer, bands)
        for p in ("W", "R")
    }
    cells = {p: _to_um(r, library, r.name) for p, r in rows.items()}
    h = rows["W"].height / 1000
    top = 2 * h
    # Each row's outer rail against its pair, the two sharing the other rail
    # in the middle: a row's VDD is its top, so with VDD outside the write
    # row stands upright and the read row is turned over, and with VSS the
    # other way round.
    flips = {"W": outer == "n", "R": outer == "p"}
    base = {"W": h, "R": 0.0}

    def at(port: str, y_nm: float) -> float:
        y = y_nm / 1000
        return base[port] + (h - y if flips[port] else y)

    band = library.new_cell(f"sram1r1w_rank_{spec.words}x{spec.bits}")
    for port, cell in cells.items():
        origin = (x_left, base[port] + (h if flips[port] else 0))
        band.add(gdstk.Reference(cell, origin, x_reflection=flips[port]))
    via34, via45 = _via_cells(library)
    w = rows["W"]
    lv = w.levels
    # The jog's track: inside every write PS output's M3, which runs from
    # under the n via row to the row's top, clear of its ends.
    lo, hi = sorted((at("W", lv["n"] - PAD), at("W", w.height - M3_EDGE)))
    y = _snap(foot + at("W", (lv["n"] + lv[LEVELS - 1]) / 2)) - foot
    if not lo + 0.017 <= y <= hi - 0.017:
        raise RuntimeError("no M4 track on the write PS outputs' M3")
    for s, (xr, xw) in enumerate(zip(sel["B"], sel["A"], strict=True)):
        # The read line, the full band.
        rect(band, (xr - 0.009, 0, xr + 0.009, top), M3)
        # The write line: its INV's output, up a V3 to the M4 jog, over to
        # the M5 track, which runs the full band.
        xo = x_left + w.out_x(w.gates[2 * s + 1]) / 1000
        band.add(gdstk.Reference(via34, (xo, y)), gdstk.Reference(via45, (xw, y)))
        rect(band, (min(xo, xw) - 0.020, y - 0.012, max(xo, xw) + 0.020, y + 0.012), M4)
        rect(band, (xw - 0.012, 0, xw + 0.012, top), M5)
    for port, row in rows.items():
        for net in row.right:
            _, (x, yy) = row.pin_positions[net]
            origin = (x_left + x / 1000, at(port, yy))
            band.add(gdstk.Label(net, origin, layer=M2, texttype=PIN_TEXTTYPE))
    rect(band, (x_left, 0, x_right, top), LAYERS["BOUNDARY"]["layer"])
    return RankBand(band, rows, top)


def _row_cdl(row: GateRowSpec) -> list[str]:
    lines = [f".SUBCKT {row.name} {' '.join(row.pins)}"]
    for device, d, g, s, flavor, fins in row.devices:
        model = row.bands[flavor].spec.model
        bulk = "VSS" if flavor == "n" else "VDD"
        lines.append(f"M{device} {d} {g} {s} {bulk} {model} nfin={fins} l=20n nf=1 m=1")
    return [*lines, f".ENDS {row.name}"]


def _pair_net(pin: str, half: str) -> str:
    """The macro's net on a strip pair's pin, in half `half`.  Each half's B
    rails are their own nets until the select rows' risers join them."""
    if pin.startswith("SEL_"):
        return ("W" if pin[4] == "A" else "R") + "PS" + pin[5:]
    if pin.startswith(("WWL", "RWL")):
        return pin[:3] + half + pin[3:]
    if pin.startswith("B_"):
        return ("W" if pin[2] == "A" else "R") + "B" + half + pin[3:]
    return pin


@dataclass(frozen=True)
class WordlineCore:
    """Both halves' strip pairs and the rank between them, and the column
    the pairs were built for."""

    cell: Any
    column: Any
    pair: Any
    band: RankBand
    netlist: str


def build_wordline_core(spec: MacroSpec, cells: CellLibrary) -> WordlineCore:
    """The rank band from `band_foot`, the upper pair on it, the lower pair
    under it, mirrored, labelled with the macro's nets.

    The upper half's columns stand on the upper pair's top, the lower half's
    hang, mirrored, from the lower pair's bottom.  The core's frame is the
    macro's: its own tracks and fins hold where its origin is on both, a
    multiple of 432 nm.
    """
    gdstk = require_gdstk()
    library = cells.library
    column = build_column(cells, spec.column)
    pair = build_strip_pair(cells, column, spec.slice)
    x_left, _, x_right, ph = edge_boundary(pair)
    foot = band_foot(ph)
    band = build_rank_band(spec, pair, library, foot)
    h = band.height
    core = library.new_cell(f"sram1r1w_core_{spec.words}x{spec.bits}")
    core.add(
        gdstk.Reference(pair, (0, foot + h)),
        gdstk.Reference(band.cell, (0, foot)),
        gdstk.Reference(pair, (0, foot), x_reflection=True),
    )
    rect(core, (x_left, foot - ph, x_right, foot + h + ph), LAYERS["BOUNDARY"]["layer"])
    for half, flip in (("U", False), ("L", True)):
        for lb in pair.labels:
            if lb.text.startswith("SEL_") or (flip and lb.text in ("VDD", "VSS")):
                continue
            x, y = lb.origin
            origin = (x, foot - y if flip else foot + h + y)
            core.add(gdstk.Label(_pair_net(lb.text, half), origin, layer=lb.layer,
                                 texttype=PIN_TEXTTYPE))  # fmt: skip
    for lb in band.cell.labels:
        x, y = lb.origin
        core.add(
            gdstk.Label(lb.text, (x, foot + y), layer=lb.layer, texttype=PIN_TEXTTYPE)
        )
    pins = strip_pair_pins(spec.slices, spec.wordlines)
    outer = [
        _pair_net(p, half)
        for half in "UL"
        for p in pins
        if not p.startswith("SEL_") and p not in ("VDD", "VSS")
    ]
    literals = sorted({n for row in band.rows.values() for n in row.right})
    pair_cdl = strip_pair_netlist(spec.slice, spec.wordlines, pair.name)
    lines = [pair_cdl.replace("\n.END\n", "\n")]
    for row in band.rows.values():
        lines += _row_cdl(row)
    lines.append(f".SUBCKT {core.name} {' '.join([*outer, *literals, 'VDD', 'VSS'])}")
    for half in "UL":
        nets = " ".join(_pair_net(p, half) for p in pins)
        lines.append(f"X{half} {nets} {pair.name}")
    for port, row in band.rows.items():
        lines.append(f"X{port} {' '.join(row.pins)} {row.name}")
    lines += [f".ENDS {core.name}", ".END", ""]
    return WordlineCore(core, column, pair, band, "\n".join(lines))
