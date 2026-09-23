"""Drawing helpers shared by the dense-row IO cells.

A *dense-row cell* is one standard-cell row (`RowStack` with one ``(n, p)``
entry): every poly stripe is one gate through both bands, cut on the rails;
diffusion islands sit on the 54 nm column grid; gate contacts go on the seam
between the bands; local wiring is M1 on the columns, M2 on horizontal tracks
and M3 on vertical ones.  `chipforge_asap7.devices.bitline_mux`,
`write_driver` and `output_latch` all draw this way, and what they share
lives here so that a fix to the frame, a rail contact or a via stack lands in
all of them.

Everything is in nanometres.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..layout.grid import FIN_WIDTH, GATE_PITCH, GATE_WIDTH
from ..layout.layers import LAYERS, box, require_gdspy
from .finfet import (
    CONTACT_SIZE,
    DEVICE_GATE_CUT_HEIGHT,
    GATE_LIG_HEIGHT,
    LI_RAIL_HEIGHT,
    M1_MIN_SPACE,
    M1_V0_ENCLOSURE,
    M1_WIDTH,
    POLY_OVERHANG,
    SD_BAR_WIDTH,
    VT_LAYERS,
    build_device_band,
)
from .inverter import ACTIVE_ABUT_OVERHANG, M2_V1_ENCLOSURE
from .row import RowBand, RowStack

__all__ = [
    "CAP",
    "HALF",
    "M4_HALF",
    "M4_PITCH",
    "M4_X_GRID",
    "PAD",
    "PIN_LAYERS",
    "TRACK",
    "V3_M3_CAP",
    "V3_M4_CAP",
    "draw_frame",
    "draw_rails",
    "gate_contact",
    "island",
    "label",
    "landing",
    "m2_track",
    "m3_column",
    "m4_track",
    "sd_contact",
    "square",
    "stack_to_m3",
    "supply_contact",
    "via_y",
]

HALF = CONTACT_SIZE // 2  # 9: half a via, half an 18 nm track
CAP = HALF + M1_V0_ENCLOSURE  # 14: metal past a via along its own track
PAD = HALF + M2_V1_ENCLOSURE  # 17: M2 past a V1
TRACK = M1_WIDTH + M1_MIN_SPACE  # 36: the M2 (and M3) track pitch
HALF_SD = SD_BAR_WIDTH // 2
HALF_LIG = GATE_LIG_HEIGHT // 2
HALF_CUT = DEVICE_GATE_CUT_HEIGHT // 2
HALF_RAIL = LI_RAIL_HEIGHT // 2
HALF_GATE = GATE_WIDTH // 2
LIG_PAST_GATE = 1
ISLAND_OVERHANG = ACTIVE_ABUT_OVERHANG  # ACTIVE past its end columns

M4_HALF = 12  # M4 is 24 nm wide, and a V3 is exactly as tall as its M4
M4_PITCH = 48
M4_X_GRID = 24  # the public deck holds M4 vertices to a 24 nm grid in x
V3_M3_CAP = M4_HALF + 5  # M3 past a V3, along the column
V3_M4_CAP = HALF + 11  # M4 past a V3, along the track

PIN_LAYERS = {
    metal: (LAYERS[f"{metal}_PIN"]["layer"], LAYERS[f"{metal}_PIN"]["datatype"])
    for metal in ("M1", "M2", "M3", "M4")
}


def square(cell: Any, layer: str, x: float, y: float) -> None:
    """One via, centred on (x, y)."""
    box(cell, layer, x - HALF, y - HALF, x + HALF, y + HALF)


def label(cell: Any, text: str, metal: str, origin: tuple[float, float]) -> None:
    """A pin label on `metal`'s pin layer."""
    layer, texttype = PIN_LAYERS[metal]
    cell.add(require_gdspy().Label(text, origin, layer=layer, texttype=texttype))


def draw_frame(cell: Any, stack: RowStack, width: float, vt: str) -> None:
    """Implants, wells, the VT layer, the fin and gate grids and the gate cuts.

    FIN and GATE are manufacturing grids: every stripe is one gate through
    both bands, cut on the rails.  Which band has diffusion under a stripe is
    what makes it an nFET's gate, a pFET's, or nobody's.
    """
    height = stack.height
    for band in stack.bands():
        box(cell, band.implant, 0, band.y0, width, band.y1)
        if band.in_nwell:
            box(cell, "NWELL", 0, band.y0, width, band.y1)
    if vt_layer := VT_LAYERS[vt]:
        box(cell, vt_layer, 0, 0, width, height)
    for y_fin in stack.fin_grid_ys:
        box(cell, "FIN", 0, y_fin, width, y_fin + FIN_WIDTH)
    for x_gate in range(GATE_PITCH // 2, int(width), GATE_PITCH):
        box(
            cell,
            "GATE",
            x_gate - HALF_GATE,
            -POLY_OVERHANG,
            x_gate + HALF_GATE,
            height + POLY_OVERHANG,
        )
    for y_rail, _ in stack.rails:
        box(cell, "GATE_CUT", 0, y_rail - HALF_CUT, width, y_rail + HALF_CUT)


def draw_rails(cell: Any, stack: RowStack, width: float) -> None:
    """An LI rail under an M1 rail on every row boundary, the full width."""
    for y_rail, _ in stack.rails:
        box(cell, "LIG", 0, y_rail - HALF_RAIL, width, y_rail + HALF_RAIL)
        box(cell, "M1", 0, y_rail - HALF, width, y_rail + HALF)


def island(
    cell: Any,
    band: RowBand,
    columns: Sequence[float],
    contacted: Sequence[float] | None = None,
) -> None:
    """One diffusion island over `columns`, with an SDT + LISD bar on the `contacted` ones.

    A series stack's inner nodes are columns without a bar: the diffusion
    alone joins the two channels.  ACTIVE runs `ISLAND_OVERHANG` past the end
    columns, which leaves 9 nm to the next gate.
    """
    xs = sorted(columns)
    build_device_band(
        cell,
        band.spec,
        y0=band.y0,
        sd_xs=sorted(xs if contacted is None else contacted),
        active_x=(xs[0] - ISLAND_OVERHANG, xs[-1] + ISLAND_OVERHANG),
    )


def supply_contact(cell: Any, band: RowBand, x: float) -> None:
    """A source tied to its band's rail: LISD carried on to the rail, V0 on the rail.

    The rail's LI carries the 18 nm V0 (V0.AUX.1-2) and its M1 is the net, so
    the column's own via row stays free for a signal on the other band.
    """
    lo, hi = band.active_span
    far = hi if band.rail_below else lo  # the ACTIVE edge away from the rail
    box(
        cell,
        "LISD",
        x - HALF_SD,
        min(far, band.rail_y),
        x + HALF_SD,
        max(far, band.rail_y),
    )
    square(cell, "V0", x, band.rail_y)


def via_y(band: RowBand) -> int:
    """Y of `band`'s via row: its contact row, moved off the rail by a fin if need be.

    A via's M1 pad runs `CAP` past it, and the released cells keep every M1
    pad a full space (18 nm) from the M1 rail; a contact row one fin from the
    rail would put the pad 13 nm from it.  Moving the via up a fin costs
    nothing, the LISD bar spans the whole active.
    """
    y = int(band.contact_y)
    clearance = HALF + M1_MIN_SPACE + CAP
    if band.rail_below:
        return max(y, int(band.rail_y) + clearance)
    return min(y, int(band.rail_y) - clearance)


def landing(cell: Any, band: RowBand, x: float) -> int:
    """`stack_to_m3` on `band`'s via row at column `x`; returns the via y.

    On a row whose rail is below, the landing lies along the track (18 nm
    tall) so that the first track above it keeps its place; on one whose
    rail is above it stands along the column, as `stack_to_m3` does, because
    there neighbouring columns may both carry one.
    """
    y = via_y(band)
    if band.rail_below:
        square(cell, "V1", x, y)
        box(cell, "M2", x - PAD, y - HALF, x + PAD, y + HALF)
        square(cell, "V2", x, y)
    else:
        stack_to_m3(cell, x, y)
    return y


def sd_contact(cell: Any, band: RowBand, x: float, reach: float | None = None) -> float:
    """A V0 on `band`'s via row at column `x`, under an M1 pad.

    The pad covers the via and, with `reach`, runs on to that y (toward the
    seam, or past it) so a track or a bar can land on it.  Returns the via y.
    """
    y = via_y(band)
    ends = [y - CAP, y + CAP] + ([reach] if reach is not None else [])
    square(cell, "V0", x, y)
    box(cell, "M1", x - HALF, min(ends), x + HALF, max(ends))
    return y


def gate_contact(
    cell: Any, seam: float, gates: Sequence[float], x_via: float, reach: float
) -> None:
    """One LIG pad over a run of same-net gates, a V0 on it at `x_via`, and an M1 bar to `reach`."""
    xs = sorted(gates)
    box(cell, "LIG", xs[0] - HALF_GATE - LIG_PAST_GATE, seam - HALF_LIG,
        xs[-1] + HALF_GATE + LIG_PAST_GATE, seam + HALF_LIG)  # fmt: skip
    square(cell, "V0", x_via, seam)
    box(
        cell,
        "M1",
        x_via - HALF,
        min(seam - CAP, reach),
        x_via + HALF,
        max(seam + CAP, reach),
    )


def m2_track(
    cell: Any, y: float, x0: float, x1: float, vias: Sequence[float] = ()
) -> None:
    """An M2 track from `x0` to `x1` (the ends past the outermost via), with a V1 at each of `vias`."""
    box(cell, "M2", min(x0, x1), y - HALF, max(x0, x1), y + HALF)
    for x in vias:
        square(cell, "V1", x, y)


def m3_column(
    cell: Any, x: float, y0: float, y1: float, vias: Sequence[float] = ()
) -> None:
    """An M3 column from `y0` to `y1`, with a V2 at each of `vias`."""
    box(cell, "M3", x - HALF, min(y0, y1), x + HALF, max(y0, y1))
    for y in vias:
        square(cell, "V2", x, y)


def stack_to_m3(cell: Any, x: float, y: float) -> None:
    """V1, an M2 landing turned *along the column*, and V2, on an M1 pad at (x, y).

    Neighbouring columns are 54 nm apart.  Two M2 landings lying along the
    track would face each other tip to tip at 20 nm, where two short edges
    need 31; standing up, they face side to side at 36.
    """
    square(cell, "V1", x, y)
    box(cell, "M2", x - HALF, y - PAD, x + HALF, y + PAD)
    square(cell, "V2", x, y)


def m4_track(cell: Any, y: float, xs: Sequence[float]) -> tuple[float, float]:
    """A horizontal M4 wire over every column in `xs`, with an 18 x 24 V3 on each.

    M4 only runs horizontally, 24 nm wide, and a V3 is exactly as tall as it;
    the wire ends `V3_M4_CAP` past the outer vias, rounded out to the 24 nm
    grid the public deck holds M4 vertices to in x.  (A 594 nm row is not a
    whole number of M4 pitches, so nothing holds the wire's y to a grid.)  The M3 under each via has to reach
    `V3_M3_CAP` past it, which is the caller's business.  Returns the wire's
    x extent.
    """
    x0 = min(xs) - V3_M4_CAP
    x1 = max(xs) + V3_M4_CAP
    x0 -= x0 % M4_X_GRID
    x1 += -x1 % M4_X_GRID
    box(cell, "M4", x0, y - M4_HALF, x1, y + M4_HALF)
    for x in xs:
        box(cell, "V3", x - HALF, y - M4_HALF, x + HALF, y + M4_HALF)
    return x0, x1
