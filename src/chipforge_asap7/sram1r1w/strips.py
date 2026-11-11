"""The wordline driver strips under a stack of 1R1W columns.

One `DriverSliceSpec` slice drives four wordlines on the array's 108 nm pitch,
``WL[4k + j] = SEL[k] . B[j]``.  A port's strip is one slice per four
wordlines, aligned to the column's wordlines, with a filler in every other
slot under the array, and a filler then a tap past its left end.  As in
OpenFinRAM's 2RW compiler (``compile_asap7_2rw.py``, `build_wordline_strips`),
which this follows, the two strips are a pair:

::

    ── column stack above: WWL[i] on M3, RWL[i] on M5 ───────────────────
       port A strip: drivers, then NANDs        WWL on M3 straight up
    ══ shared VSS rail ════════════════════════════════════════════════
       port B strip, flipped: NANDs, then drivers
       RWL: V3 + V4 on the driver's output, M5 straight up
    ── bottom ─────────────────────────────────────────────────────────

The pair abuts the column's bottom with no margin: the column's fins already
sit on a slice's grid.  Each slice's ``SEL`` and ``B[j]`` are its own pins,
labelled ``SEL_A[k]`` and ``B_A[k][j]`` (``_B`` for port B): the macro's
router joins them to the decoder.

Coordinates are um.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

from ..devices.driver_slice import (
    DRIVER_SLICE_PINS,
    DriverSliceSpec,
    build_driver_slice,
)
from ..devices.row_support import RowSupportSpec, build_row_support
from ..layout.layers import LAYERS
from ..layout.transform import edge_boundary, indexed_labels, rect, require_gdstk
from .cells import CellLibrary

__all__ = ["build_strip_pair", "strip_pair_netlist", "strip_pair_pins"]

M3 = LAYERS["M3"]["layer"]
M4 = LAYERS["M4"]["layer"]
M5 = LAYERS["M5"]["layer"]
V3 = LAYERS["V3"]["layer"]
V4 = LAYERS["V4"]["layer"]
BOUNDARY = LAYERS["BOUNDARY"]["layer"]
PIN_TEXTTYPE = 251
WORDLINE_PITCH = 0.108
SLICE_WIDTH = 4 * WORDLINE_PITCH
#: Support columns, one slot each.  A slice's diffusion and implant overhang
#: its edge for an abutting neighbour, so a tap beside one breaks the tap's
#: select enclosure: the tap stands past a filler, and fillers fill the gaps.
SUPPORT = {"filler": ("filler", 1), "tap": ("tap", 1)}


def _support(spec: DriverSliceSpec, kind: str, slots: int, name: str, lib: Any) -> Any:
    """`build_driver_slice_support` `slots` wordline pitches wide: a column of
    three row-support cells, the NAND stack's, the same mirrored, the drivers'."""
    import gdspy

    width = 2 * slots  # contacted poly pitches
    cell = lib.new_cell(name)
    nand = build_row_support(
        RowSupportSpec(stack=spec.nand.stack, kind=kind, width_cpp=width),
        name=f"{name}__nand",
        lib=lib,
    )
    driver = build_row_support(
        RowSupportSpec(stack=spec.inverter.stack, kind=kind, width_cpp=width),
        name=f"{name}__drv", lib=lib,
    )  # fmt: skip
    cell.add(gdspy.CellReference(nand, origin=(0, 0)))
    cell.add(gdspy.CellReference(nand, origin=(0, spec.driver_y0), x_reflection=True))
    cell.add(gdspy.CellReference(driver, origin=(0, spec.driver_y0)))
    return cell


def _slice_cells(spec: DriverSliceSpec, library: Any) -> dict[str, Any]:
    """The slice and its support columns, drawn by chipforge's generators, read back in um.

    ``"slice"``, and ``(kind, slots)`` for each support column `SUPPORT` uses."""
    import gdspy

    gdstk = require_gdstk()
    name = spec.cell_name
    names = {"slice": name} | {
        key: f"{name}_{key[0]}{key[1]}" for key in set(SUPPORT.values())
    }
    have = {c.name: c for c in library.cells}
    if all(n in have for n in names.values()):
        return {key: have[n] for key, n in names.items()}
    nm = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    previous = gdspy.current_library
    gdspy.current_library = nm
    try:
        build_driver_slice(spec, lib=nm)
        for key, cell_name in names.items():
            if key != "slice":
                _support(spec, key[0], key[1], cell_name, nm)
    finally:
        gdspy.current_library = previous
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "slice.gds"
        nm.write_gds(str(path))
        um = gdstk.read_gds(str(path), unit=1e-6)
    cells = {c.name: c for c in um.cells}
    for cell in um.cells:
        if cell.name not in have:
            library.add(cell)
    return {key: cells[n] for key, n in names.items()}


def strip_pair_pins(slices: int, wordlines: int) -> list[str]:
    pins = []
    for port in ("A", "B"):
        for k in range(slices):
            pins.append(f"SEL_{port}[{k}]")
            pins += [f"B_{port}[{k}][{j}]" for j in range(4)]
    pins += [f"WWL[{i}]" for i in range(wordlines)]
    pins += [f"RWL[{i}]" for i in range(wordlines)]
    return [*pins, "VDD", "VSS"]


def build_strip_pair(cells: CellLibrary, column: Any, spec: DriverSliceSpec) -> Any:
    """The pair under `column` (a `build_column` cell), in its x; returns the gdstk cell.

    Its top edge abuts the column's bottom.
    """
    gdstk = require_gdstk()
    library = cells.library
    parts = _slice_cells(spec, library)
    slice_cell = parts["slice"]
    write = {i: float(lb.origin[0]) for i, lb in indexed_labels(column, "WWL").items()}
    read = {i: float(lb.origin[0]) for i, lb in indexed_labels(column, "RWL").items()}
    wordlines = len(write)
    if wordlines % 4 or set(read) != set(write):
        raise ValueError(f"{column.name}: wordlines must come in fours, both ports")
    _, _, _, height = edge_boundary(slice_cell)
    pins = {
        lb.text: (float(lb.origin[0]), float(lb.origin[1]), lb.layer)
        for lb in slice_cell.labels
    }

    strip = library.new_cell(f"wl_strip_{spec.cell_name}_x{wordlines // 4}")
    starts = []
    for k in range(wordlines // 4):
        x0 = write[4 * k] - WORDLINE_PITCH / 2
        for j in range(4):
            if abs(write[4 * k + j] - (x0 + WORDLINE_PITCH * (j + 0.5))) > 1e-6:
                raise RuntimeError(
                    f"slice {k}: WWL[{4 * k + j}] is off the slice's output"
                )
        strip.add(gdstk.Reference(slice_cell, (x0, 0)))
        starts.append(x0)
    spans = [(x0, x0 + SLICE_WIDTH) for x0 in starts]
    # The array: its first cell is the first wordline's, and a tapped array
    # ends in a tap past the last slice; the strip runs under all of it.
    array_x0 = write[0] - WORDLINE_PITCH / 2
    slots = round((_array_right(column, array_x0) - array_x0) / WORDLINE_PITCH)
    array_x1 = array_x0 + slots * WORDLINE_PITCH
    fill = [
        (a[1], b[0])
        for a, b in zip(spans, spans[1:], strict=False)
        if b[0] - a[1] > 1e-6
    ]
    # Fillers run on under the array's last tap, so its gates end on poly; the
    # one tap is at the left end.  ACTIVE.LUP.1 allows 30 um to a tap, a strip
    # of some 256 wordlines.  A tap under the array meets the cells' SRAM
    # marker (SRAM.LISD.AUX.1), and past it the read IO block's rails.
    fill.append((spans[-1][1], max(array_x1, spans[-1][1] + WORDLINE_PITCH)))
    columns = [
        (array_x0 - 2 * WORDLINE_PITCH, "tap"),
        (array_x0 - WORDLINE_PITCH, "filler"),
    ]
    for g0, g1 in fill:
        count = round((g1 - g0) / WORDLINE_PITCH)
        if abs(g1 - g0 - count * WORDLINE_PITCH) > 1e-6:
            raise RuntimeError(
                f"a {(g1 - g0) * 1000:.0f} nm gap between slices is not whole slots"
            )
        columns += [(g0 + c * WORDLINE_PITCH, "filler") for c in range(count)]
    for x, kind in columns:
        strip.add(gdstk.Reference(parts[SUPPORT[kind]], (x, 0)))
    x_left, x_right = array_x0 - 2 * WORDLINE_PITCH, fill[-1][1]
    if spans[-1][1] - x_left > 28.0:
        raise ValueError("a strip this long needs a second tap (ACTIVE.LUP.1, 30 um)")

    via34 = _via_cell(library, "wl_via34", ((-0.020, -0.012, 0.020, 0.012, M4),
                                            (-0.009, -0.017, 0.009, 0.017, M3),
                                            (-0.009, -0.012, 0.009, 0.012, V3)))  # fmt: skip
    via45 = _via_cell(library, "wl_via45", ((-0.012, -0.023, 0.012, 0.023, M5),
                                            (-0.023, -0.012, 0.023, 0.012, M4),
                                            (-0.012, -0.012, 0.012, 0.012, V4)))  # fmt: skip

    pair = library.new_cell(f"wl_strips_{spec.cell_name}_x{wordlines // 4}")
    # No margin to the column: OpenFinRAM's tile needs half a fin pitch, but
    # the 1R1W column's fins (its IO blocks', 13.5 nm past a multiple of 27
    # from its bottom) already share a grid with a slice abutted below.
    top = 2 * height
    seam = height  # the shared VSS rail
    pair.add(gdstk.Reference(strip, (0, seam)))
    pair.add(gdstk.Reference(strip, (0, seam), x_reflection=True))
    for port, sign in (("A", 1), ("B", -1)):
        for k, x0 in enumerate(starts):
            sx, sy, layer = pins["SEL"]
            pair.add(gdstk.Label(f"SEL_{port}[{k}]", (x0 + sx, seam + sign * sy), layer=layer,
                                 texttype=PIN_TEXTTYPE))  # fmt: skip
            for j in range(4):
                bx, by, layer = pins[f"B{j}"]
                pair.add(gdstk.Label(f"B_{port}[{k}][{j}]", (x0 + bx, seam + sign * by),
                                     layer=layer, texttype=PIN_TEXTTYPE))  # fmt: skip
    # Port B's via row: the first M4 track (12 nm past a multiple of 48 in the
    # column's frame, the pair abutting its bottom) at or above the drivers'
    # output pins, where the V3 lands on their M3.
    nm = round((0.025 - top) * 1000)  # the pins are 25 nm above the pair's bottom
    via_row = top + (12 + 48 * math.ceil((nm - 12) / 48)) / 1000
    for i in range(wordlines):
        x, x5 = write[i], read[i]
        pair.add(
            gdstk.Label(f"WWL[{i}]", (x, top - 0.005), layer=M3, texttype=PIN_TEXTTYPE)
        )
        y_via = via_row
        pair.add(gdstk.Reference(via34, (x, y_via)))
        pair.add(gdstk.Reference(via45, (x5, y_via)))
        rect(pair, (x5 - 0.012, y_via - 0.023, x5 + 0.012, top), M5)
        pair.add(
            gdstk.Label(f"RWL[{i}]", (x5, top - 0.005), layer=M5, texttype=PIN_TEXTTYPE)
        )
    rect(pair, (x_left, 0, x_right, top), BOUNDARY)
    return pair


def _array_right(column: Any, array_x0: float) -> float:
    """The array's right edge in `column`: where its read IO block begins."""
    refs = [r for r in column.references if r.cell.name.startswith("array_")]
    if len(refs) != 1:
        raise RuntimeError(f"{column.name}: expected one array")
    x0, _, x1, _ = edge_boundary(refs[0].cell)
    return float(refs[0].origin[0]) + x1  # the array is placed with its x0 at array_x0


def _via_cell(library: Any, name: str, shapes: tuple) -> Any:
    gdstk = require_gdstk()
    for cell in library.cells:
        if cell.name == name:
            return cell
    cell = library.new_cell(name)
    for x0, y0, x1, y1, layer in shapes:
        cell.add(gdstk.rectangle((x0, y0), (x1, y1), layer=layer))
    return cell


def strip_pair_netlist(spec: DriverSliceSpec, wordlines: int, name: str) -> str:
    """The pair's CDL: one slice instance per four wordlines and port."""
    slices = wordlines // 4
    lines = [spec.netlist().replace("\n.END\n", "\n").rstrip()]
    lines.append(f".SUBCKT {name} {' '.join(strip_pair_pins(slices, wordlines))}")
    for port, wl in (("A", "WWL"), ("B", "RWL")):
        for k in range(slices):
            nets = {"SEL": f"SEL_{port}[{k}]", "VDD": "VDD", "VSS": "VSS"}
            nets |= {f"B{j}": f"B_{port}[{k}][{j}]" for j in range(4)}
            nets |= {f"WL{j}": f"{wl}[{4 * k + j}]" for j in range(4)}
            lines.append(
                f"X{port}{k} {' '.join(nets[p] for p in DRIVER_SLICE_PINS)} {spec.cell_name}"
            )
    lines += [f".ENDS {name}", ".END", ""]
    return "\n".join(lines)
