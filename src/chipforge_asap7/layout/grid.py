"""ASAP7 fin / gate substrate grids for hand-built array macros.

Extracted from the array compilers in chipforge (`scripts/brom_compiler.py`,
`scripts/rom_compiler.py`, `scripts/pseudo_nmos_compiler.py`,
`scripts/gen_rom_bitcell.py`), which each re-derived the same four things:

  1. a horizontal **fin grid** — 7 nm stripes on a 27 nm pitch, drawn across the
     whole array so every abutted bitcell sits on continuous active fins;
  2. a vertical **gate grid** — 20 nm poly stripes on the 54 nm contacted-poly
     pitch (CPP), run past the array top and bottom so poly ends are cut, not
     drawn, at the boundary;
  3. **gate cuts** on the row boundaries, which is what turns one continuous
     poly stripe into one independent transistor gate per row;
  4. the **column/gate alignment offset** (`COLUMN_X_SHIFT`) that makes a 2-CPP
     bitcell's gate land exactly on a gate-grid track.

The numbers here are the ones already taped out through the NOR-ROM flow; they
are reproduced exactly so a compiler can call these helpers instead of its local
loops and emit byte-identical geometry.

Coordinates are integers in **nanometres** (gdspy libraries in this flow use
``unit=1e-9, precision=1e-10``).  Y grows upward, X to the right.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .layers import box

__all__ = [
    "BITCELL_DRAIN_DX",
    "BITCELL_GATE_DX",
    "BITCELL_SOURCE_DX",
    "CANDIDATE_PITCH",
    "COLUMN_PITCH",
    "COLUMN_X_SHIFT",
    "FIN_P",
    "FIN_PITCH",
    "FIN_WIDTH",
    "GATE_CUT_HEIGHT",
    "GATE_P",
    "GATE_PITCH",
    "GATE_WIDTH",
    "PITCH_X",
    "PITCH_Y",
    "STD_CELL_H",
    "STD_CELL_HEIGHT",
    "WL_PITCH",
    "WORDLINE_PITCH",
    "X_SHIFT",
    "centered_fin_ys",
    "column_gate_track",
    "column_x",
    "draw_fin_grid",
    "draw_gate_cuts",
    "draw_gate_grid",
    "fin_ys",
    "gate_center_x",
    "gate_track_xs",
    "row_boundary_ys",
    "snap_to_fin",
    "snap_to_gate",
    "spans_excluding",
]

# ── Grid constants (nm) ───────────────────────────────────────────────────────
FIN_PITCH = 27  # ASAP7 fin pitch
FIN_WIDTH = 7  # drawn fin stripe width (the FIN layer is a marker, not silicon)
GATE_PITCH = 54  # contacted poly pitch (CPP)
GATE_WIDTH = 20  # drawn poly stripe width (gen_rom_bitcell poly_width)
GATE_CUT_HEIGHT = 24  # GATE_CUT stripe height at a row boundary

COLUMN_PITCH = 108  # bitcell column pitch = 2 CPP
COLUMN_X_SHIFT = 46  # gate-grid alignment offset, see column_gate_track()

WORDLINE_PITCH = 81  # NOR-ROM row pitch = 3 fin pitch (2 active + 1 isolation)
CANDIDATE_PITCH = 108  # B-ROM candidate row pitch = 4 fin pitch
STD_CELL_HEIGHT = 270  # ASAP7 7.5-track std cell height

# X anchors inside the 2-CPP `ROM_BitCell` (gen_rom_bitcell.ASAP7), measured
# from the column's bitline X.  They are what COLUMN_PITCH / COLUMN_X_SHIFT have
# to satisfy, so they live with the grid rather than with the cell.
BITCELL_GATE_DX = 36  # gate (wordline) sits at column_x - 36
BITCELL_SOURCE_DX = 72  # source (VSS / programming site) at column_x - 72
BITCELL_DRAIN_DX = 0  # drain (bitline) is the column X itself

# Short aliases matching the names the chipforge scripts already use.
FIN_P = FIN_PITCH
GATE_P = GATE_PITCH
PITCH_X = COLUMN_PITCH
PITCH_Y = CANDIDATE_PITCH
WL_PITCH = WORDLINE_PITCH
X_SHIFT = COLUMN_X_SHIFT
STD_CELL_H = STD_CELL_HEIGHT


# ── Fin grid ──────────────────────────────────────────────────────────────────
def fin_ys(y0: float, y1: float, *, pitch: int = FIN_PITCH) -> list[float]:
    """Bottom Y of every fin stripe covering ``[y0, y1]``.

    Fins are anchored at `y0` and stepped by `pitch`; the last fin is placed at
    or past `y1` so an array whose height is not a whole fin count is still fully
    covered (this matches ``range(int(height / FIN_P) + 1)`` in the compilers).
    """
    if y1 < y0:
        y0, y1 = y1, y0
    count = int((y1 - y0) / pitch) + 1
    return [y0 + i * pitch for i in range(count)]


def centered_fin_ys(cell_h: int, fins: int, *, pitch: int = FIN_PITCH) -> list[float]:
    """Bottom Y of `fins` active fins centred in a `cell_h`-tall cell.

    The single-device version of the array grid: it is how `gen_rom_bitcell` and
    the pseudo-nMOS load cell place active fins so that abutting two cells leaves
    at least one empty fin pitch of isolation between their diffusions.
    """
    if cell_h % pitch:
        raise ValueError(f"cell_h {cell_h} is not a multiple of fin pitch {pitch}")
    if (fins + 1) * pitch > cell_h:
        raise ValueError(
            f"{fins} fins plus isolation need >= {(fins + 1) * pitch} nm; "
            f"cell_h={cell_h} is too short"
        )
    y0 = ((cell_h - fins * pitch) // pitch // 2) * pitch + pitch // 2
    return [y0 + i * pitch for i in range(fins)]


def snap_to_fin(y: float, *, pitch: int = FIN_PITCH) -> int:
    """Round `y` to the nearest fin track."""
    return int((y + pitch // 2) // pitch) * pitch


def draw_fin_grid(
    cell: Any,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    pitch: int = FIN_PITCH,
    width: float = FIN_WIDTH,
) -> int:
    """Draw the horizontal fin grid across ``[x0, x1] x [y0, y1]``.

    Returns the number of fin stripes drawn.
    """
    ys = fin_ys(y0, y1, pitch=pitch)
    for y in ys:
        box(cell, "FIN", x0, y, x1, y + width)
    return len(ys)


# ── Gate grid ─────────────────────────────────────────────────────────────────
def gate_track_xs(x0: float, x1: float, *, pitch: int = GATE_PITCH) -> list[float]:
    """Left edge X of every gate (poly) track covering ``[x0, x1]``."""
    if x1 < x0:
        x0, x1 = x1, x0
    count = int((x1 - x0) / pitch) + 1
    return [x0 + j * pitch for j in range(count)]


def gate_center_x(track_x: float, *, width: float = GATE_WIDTH) -> float:
    """Center X of the poly stripe whose left edge is at `track_x`.

    The array grid draws poly *right of* the track (``[j*54, j*54 + 20]``) while
    a bitcell draws it *centred* on its own gate X, so bridging the two costs
    this half-width — get it wrong and every bitcell is 10 nm off its gate.
    """
    return track_x + width / 2


def snap_to_gate(x: float, *, pitch: int = GATE_PITCH) -> int:
    """Round `x` to the nearest gate track."""
    return int((x + pitch // 2) // pitch) * pitch


def draw_gate_grid(
    cell: Any,
    x0: float,
    x1: float,
    y_bottom: float,
    y_top: float,
    *,
    pitch: int = GATE_PITCH,
    width: float = GATE_WIDTH,
) -> int:
    """Draw the vertical poly grid over ``[x0, x1]``, spanning y_bottom..y_top.

    `y_bottom` is normally *below* the array (the compilers run poly down into
    the peripheral rows) so no poly line ends inside the array; the row-by-row
    separation comes from `draw_gate_cuts`, not from ending the stripes.

    Returns the number of gate tracks drawn.
    """
    xs = gate_track_xs(x0, x1, pitch=pitch)
    for x in xs:
        box(cell, "GATE", x, y_bottom, x + width, y_top)
    return len(xs)


# ── Column <-> gate-grid alignment ────────────────────────────────────────────
def column_x(
    index: int,
    x0: float = 0.0,
    *,
    pitch: int = COLUMN_PITCH,
    shift: int = COLUMN_X_SHIFT,
) -> float:
    """X of bit column `index` (its bitline / bitcell drain) in an array at `x0`.

    One column pitch of margin is left at the array's left edge, matching the
    ``(index + 1) * PITCH_X + X_SHIFT`` used by every chipforge array compiler.
    """
    return x0 + (index + 1) * pitch + shift


def column_gate_track(
    index: int,
    x0: float = 0.0,
    *,
    pitch: int = COLUMN_PITCH,
    shift: int = COLUMN_X_SHIFT,
    gate_pitch: int = GATE_PITCH,
    gate_dx: int = BITCELL_GATE_DX,
    gate_width: float = GATE_WIDTH,
) -> int:
    """Index of the gate track that column `index`'s bitcell gate lands on.

    This is the invariant `COLUMN_X_SHIFT` exists to satisfy: with the ASAP7
    2-CPP bitcell (gate at ``column_x - 36``), a shift of 46 nm puts every
    column's gate exactly on the centre of gate track ``2 * index + 2``.  Raises
    if the parameters do not align, which is the failure the compilers would
    otherwise only show as a DRC/LVS surprise after hardening.
    """
    gate_center = column_x(index, x0, pitch=pitch, shift=shift) - gate_dx
    offset = gate_center - x0 - gate_width / 2
    track, rem = divmod(offset, gate_pitch)
    if rem:
        raise ValueError(
            f"column {index} gate at x={gate_center} is {rem} nm off the "
            f"{gate_pitch} nm gate grid (shift={shift}, pitch={pitch})"
        )
    return int(track)


# ── Row boundaries and gate cuts ──────────────────────────────────────────────
def row_boundary_ys(rows: int, pitch: int) -> list[float]:
    """Y of each boundary between (and outside) `rows` rows on `pitch`.

    Rows are centred on ``(r + 1) * pitch``, so the boundaries — where the gate
    cuts go — fall on the half-pitches ``(r + 0.5) * pitch``.
    """
    return [(r + 0.5) * pitch for r in range(rows + 1)]


def spans_excluding(
    x0: float, x1: float, blocks: Iterable[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Split ``[x0, x1]`` around `blocks`, dropping empty pieces.

    Used wherever a full-width feature (gate cuts, wordline rails) has to step
    over mid-array buffer strips.  `blocks` must be sorted and non-overlapping.
    """
    spans: list[tuple[float, float]] = []
    cursor = x0
    for bx0, bx1 in blocks:
        if bx0 > cursor:
            spans.append((cursor, bx0))
        cursor = max(cursor, bx1)
    if x1 > cursor:
        spans.append((cursor, x1))
    return spans


def draw_gate_cuts(
    cell: Any,
    spans: Sequence[tuple[float, float]],
    ys: Iterable[float],
    *,
    height: float = GATE_CUT_HEIGHT,
) -> int:
    """Draw GATE_CUT stripes at each y in `ys`, over each span in `spans`.

    Cuts are centred on their y (``y ± height / 2``).  Returns the stripe count.
    """
    drawn = 0
    for y in ys:
        for x_start, x_end in spans:
            if x_end <= x_start:
                continue
            box(cell, "GATE_CUT", x_start, y - height / 2, x_end, y + height / 2)
            drawn += 1
    return drawn
