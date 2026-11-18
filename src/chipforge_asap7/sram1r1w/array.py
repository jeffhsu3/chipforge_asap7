"""Tile the 8T cell into a column's array: wordlines across x, mux rows up y.

As in ASAP7's academic SRAM library and OpenFinRAM's
``generate_asap7_wordline_arrays.py``, which this follows: a *row*
(``sramcol``) is one bitline set (``BLA``/``BLAN``/``BLB``/``BLBN``) running
through one cell per wordline, the cells 108 nm apart and every other one
mirrored in x.  Four rows stacked, every other one mirrored in y, make the
array one data bit's IO column serves through its 4:1 mux.

A tap takes two slots after every `tap_pitch` cells and hands every bitline
and rail straight through, so it can sit mid-row; it bounds the distance
from any cell to a well and substrate tie.

Coordinates are um (the cells' unit).
"""

from __future__ import annotations

import math
from typing import Any

from ..layout.transform import (
    clone_label,
    direct_label,
    edge_boundary,
    require_gdstk,
)
from .cells import (
    BITCELL,
    BITCELL_MIRRORED,
    BITCELL_MIRRORED_END,
    TAP,
    CellLibrary,
)

__all__ = [
    "BITLINES",
    "MUX_ROWS",
    "TAP_SLOTS",
    "WORDLINES",
    "array_name",
    "build_array",
    "build_row",
    "default_tap_pitch",
    "slot_layout",
]

BOUNDARY = 100
MUX_ROWS = 4
#: Bitcell slots a tap takes.
TAP_SLOTS = 2
WORDLINES = ("WLA", "WLB")
BITLINES = ("BLA", "BLAN", "BLB", "BLBN")


def default_tap_pitch(wordlines: int) -> int:
    """A tap after every gcd(wordlines, 16) cells, as OpenFinRAM's compiler places them."""
    return math.gcd(wordlines, 16)


def slot_layout(wordlines: int, tap_pitch: int) -> tuple[int, list[int], list[int]]:
    """``(slots, slot of each wordline's cell, first slot of each tap)``."""
    if wordlines < 1:
        raise ValueError(f"a row needs at least one wordline, got {wordlines}")
    if tap_pitch and wordlines % tap_pitch:
        raise ValueError(f"tap pitch {tap_pitch} does not divide {wordlines} wordlines")
    if not tap_pitch:
        return wordlines, list(range(wordlines)), []
    cells = [i + TAP_SLOTS * (i // tap_pitch) for i in range(wordlines)]
    taps = [cells[i] + 1 for i in range(wordlines) if (i + 1) % tap_pitch == 0]
    return wordlines + TAP_SLOTS * (wordlines // tap_pitch), cells, taps


def _point(
    point: Any,
    origin: tuple[float, float],
    *,
    mirror_x: bool = False,
    mirror_y: bool = False,
) -> tuple[float, float]:
    """Where a reference at `origin`, mirrored in x (a half turn and a y
    reflection) or in y (a y reflection), puts a point of its cell."""
    x, y = float(point[0]), float(point[1])
    if mirror_x:
        x = -x
    if mirror_y:
        y = -y
    return origin[0] + x, origin[1] + y


def _supply_label(cell: Any, prefix: str) -> Any:
    labels = [lb for lb in cell.labels if lb.text.lower().startswith(prefix)]
    if not labels:
        raise RuntimeError(f"{cell.name}: no {prefix} label")
    return labels[0]  # the first, as OpenFinRAM's arrays take it


def row_name(wordlines: int, tap_pitch: int) -> str:
    return f"sramcol_x{wordlines}{f'_tap{tap_pitch}' if tap_pitch else ''}_sram_8t"


def array_name(wordlines: int, tap_pitch: int) -> str:
    return f"array_x{wordlines}x{MUX_ROWS}{f'_tap{tap_pitch}' if tap_pitch else ''}_sram_8t"


def build_row(library: Any, cells: CellLibrary, wordlines: int, tap_pitch: int) -> Any:
    """One bitline set through `wordlines` cells, labelled ``WLA[i]``/``WLB[i]``
    at each cell and its bitlines and supplies at the last one."""
    gdstk = require_gdstk()
    name = row_name(wordlines, tap_pitch)
    if name in {c.name for c in library.cells}:
        return next(c for c in library.cells if c.name == name)
    bitcell = cells.cells[BITCELL]
    x0, y0, x1, y1 = edge_boundary(bitcell)
    width, height = x1 - x0, y1 - y0
    for master in (BITCELL_MIRRORED, BITCELL_MIRRORED_END):
        if edge_boundary(cells.cells[master]) != (x0, y0, x1, y1):
            raise RuntimeError(f"{master} does not share the bitcell's boundary")
    slots, cell_slots, tap_slots = slot_layout(wordlines, tap_pitch)
    tx0, ty0 = 0.0, 0.0
    if tap_pitch:
        tx0, ty0, tx1, ty1 = edge_boundary(cells.cells[TAP])
        if abs(tx1 - tx0 - TAP_SLOTS * width) > 1e-6 or abs(ty1 - ty0 - height) > 1e-6:
            raise RuntimeError(f"the tap is not {TAP_SLOTS} slots on the bitcell's row")

    row = library.new_cell(name)
    origins = []
    for index, slot in enumerate(cell_slots):
        mirror = index % 2 == 1
        origin = ((slot + 1) * width + x0 if mirror else slot * width - x0, -y0)
        master = cells.cells[BITCELL_MIRRORED if mirror else BITCELL]
        if mirror and index == wordlines - 1 and not tap_pitch:
            master = cells.cells[BITCELL_MIRRORED_END]
        row.add(
            gdstk.Reference(
                master,
                origin=origin,
                rotation=math.pi if mirror else 0,
                x_reflection=mirror,
            )
        )
        origins.append((origin, mirror))
        for pin in WORDLINES:
            label = direct_label(bitcell, pin)
            point = _point(tuple(label.origin), origin, mirror_x=mirror)
            row.add(clone_label(label, f"{pin}[{index}]", point))
    for slot in tap_slots:
        row.add(gdstk.Reference(cells.cells[TAP], origin=(slot * width - tx0, -ty0)))

    # The shared bitlines and supplies are exposed at the last cell.
    origin, mirror = origins[-1]
    for pin in BITLINES:
        label = direct_label(bitcell, pin)
        row.add(
            clone_label(
                label, pin, _point(tuple(label.origin), origin, mirror_x=mirror)
            )
        )
    for prefix, text in (("vdd", "VDD"), ("vss", "VSS")):
        label = _supply_label(bitcell, prefix)
        row.add(
            clone_label(
                label, text, _point(tuple(label.origin), origin, mirror_x=mirror)
            )
        )
    row.add(gdstk.rectangle((0, 0), (slots * width, height), layer=BOUNDARY))
    return row


def build_array(
    library: Any, cells: CellLibrary, wordlines: int, tap_pitch: int | None = None
) -> Any:
    """`MUX_ROWS` rows, every other mirrored in y: one data bit's array.

    Labels ``WLA[i]``/``WLB[i]`` (from the first row; the wordlines run through
    every row), each row's bitlines ``BLA[r]``... and its supplies.
    """
    gdstk = require_gdstk()
    tap_pitch = default_tap_pitch(wordlines) if tap_pitch is None else tap_pitch
    name = array_name(wordlines, tap_pitch)
    if name in {c.name for c in library.cells}:
        return next(c for c in library.cells if c.name == name)
    row = build_row(library, cells, wordlines, tap_pitch)
    x0, y0, x1, y1 = edge_boundary(row)
    width, height = x1 - x0, y1 - y0
    array = library.new_cell(name)
    placed = []
    for r in range(MUX_ROWS):
        mirror = r % 2 == 1
        origin = (-x0, (r + 1) * height + y0 if mirror else r * height - y0)
        array.add(gdstk.Reference(row, origin=origin, x_reflection=mirror))
        placed.append((origin, mirror))
    for pin in WORDLINES:
        for index in range(wordlines):
            label = direct_label(row, f"{pin}[{index}]")
            array.add(
                clone_label(label, origin=_point(tuple(label.origin), placed[0][0]))
            )
    for r, (origin, mirror) in enumerate(placed):
        for pin in BITLINES:
            label = direct_label(row, pin)
            point = _point(tuple(label.origin), origin, mirror_y=mirror)
            array.add(clone_label(label, f"{pin}[{r}]", point))
        for text in ("VDD", "VSS"):
            label = direct_label(row, text)
            array.add(
                clone_label(
                    label, origin=_point(tuple(label.origin), origin, mirror_y=mirror)
                )
            )
    array.add(gdstk.rectangle((0, 0), (width, MUX_ROWS * height), layer=BOUNDARY))
    return array
