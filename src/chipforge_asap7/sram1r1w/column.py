"""One data bit of the 1R1W SRAM: a write port, the array, a read port.

::

    write IO | edge cap | array, N wordlines ... tap | read IO
     (port A, mirrored)                                 (port B)

The array is OpenFinRAM's 8T cell (`chipforge_asap7.sram1r1w.cells`): its
port A is the write port, its port B the read port.  Each port's IO is an
`IoColumnSpec` block at the bitcell's own bitline heights, as OpenFinRAM's
2RW column builds it (``generate_asap7_8t_iocolumn.py``, which this
follows), with the block's `role` cut to what its port does: the write
block keeps the bitline mux and the write driver, the read block the mux,
the sense amplifier and the output latch.  Port A's bitlines leave the array
on M2 and port B's on M4, so neither crosses the other's IO.

The column's pins:

====================  ======================================================
``WWL[i]``, ``RWL[i]``  write and read wordline of word slot ``i`` (M3, M5)
``wysel[k]``...         write mux select of row ``k`` and its complement
``wprechn``             write bitline precharge, active low
``wrena``, ``wrenan``   write driver enable and its complement
``D``                   the bit written
``rysel[k]``...         read mux select of row ``k`` and its complement
``rprechn``             read bitline precharge, active low
``sae``                 sense enable (one phase: low precharges the amplifier)
``oe``, ``oeb``         output latch enable and its complement
``Q``                   the bit read
====================  ======================================================

Coordinates are um, the cells' unit.
"""

from __future__ import annotations

import math
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..devices.io_column import (
    IoColumnSpec,
    block_netlist,
    build_io_column,
    io_column_pins,
)
from ..layout.layers import LAYERS
from ..layout.straps import add_supply_straps, tie_seam_supplies
from ..layout.transform import (
    bbox,
    clone_label,
    edge_boundary,
    indexed_labels,
    rect,
    require_gdstk,
    topbot_name,
)
from ..verification.sram import io_block_specs
from .array import MUX_ROWS, build_array
from .cells import FILLER, CellLibrary

__all__ = [
    "ColumnSpec",
    "build_column",
    "column_netlist",
    "column_pins",
    "io_specs",
]

BOUNDARY = LAYERS["BOUNDARY"]["layer"]
GATE = LAYERS["GATE"]["layer"]
LIG = LAYERS["LIG"]["layer"]
V0 = LAYERS["V0"]["layer"]
M1 = LAYERS["M1"]["layer"]
M2 = LAYERS["M2"]["layer"]
M3 = LAYERS["M3"]["layer"]
M4 = LAYERS["M4"]["layer"]
M5 = LAYERS["M5"]["layer"]
PIN_TEXTTYPE = 251
#: Where a rail's LIG stops short of a block's array-facing edge: the end
#: tile's dummy row puts a WLB gate contact 17 nm past that edge, level with
#: the block's top rail, and two short LIG edges want 31 nm (LIG.S.4-5).
RAIL_LIG_INSET = 0.014

#: Each port's block pin -> the column's, beside the per-row bitline and select
#: buses, which `_pin_name` maps.
_CONTROLS = {
    "write": {"PRECHN": "wprechn", "D": "D", "WRENA": "wrena", "WRENAN": "wrenan"},
    "read": {"PRECHN": "rprechn", "SAE": "sae", "OE": "oe", "OEB": "oeb", "Q": "Q"},
}
_BUS = {
    "write": {"BL": "WBL", "BLN": "WBLN", "YSEL": "wysel", "YSELN": "wyseln"},
    "read": {"BL": "RBL", "BLN": "RBLN", "YSEL": "rysel", "YSELN": "ryseln"},
}
#: The array's bitlines each port's block meets, and the metal they arrive on.
_ARRAY_BITLINES = {
    "write": {"BL": ("BLA", M2), "BLN": ("BLAN", M2)},
    "read": {"BL": ("BLB", M4), "BLN": ("BLBN", M4)},
}


@dataclass(frozen=True)
class ColumnSpec:
    """One bit's column: `wordlines` word slots, 4:1 muxed.

    Args:
        wordlines: word slots across the array (a word is a slot and a mux row).
        tap_pitch: a tap after every this many cells; None: gcd(wordlines, 16).
    """

    wordlines: int = 32
    tap_pitch: int | None = None

    def __post_init__(self) -> None:
        if self.wordlines < 1:
            raise ValueError(f"wordlines must be positive, got {self.wordlines}")

    @property
    def words(self) -> int:
        return self.wordlines * MUX_ROWS

    @property
    def name(self) -> str:
        tap = "" if self.tap_pitch is None else f"_tap{self.tap_pitch}"
        return f"col1r1w_x{self.wordlines}x{MUX_ROWS}{tap}_sram_8t"


def io_specs(cells: CellLibrary) -> dict[str, IoColumnSpec]:
    """The write block (port A, M2 bitlines) and the read block (port B, M4)."""
    base = io_block_specs(cells.bitcell, MUX_ROWS)
    return {
        "write": replace(base["A"], role="write"),
        "read": replace(base["B"], role="read"),
    }


def _pin_name(role: str, block_pin: str) -> str:
    """``BL[2]`` -> ``WBL[2]``, ``PRECHN`` -> ``rprechn``, supplies unchanged."""
    match = re.fullmatch(r"(BL|BLN|YSEL|YSELN)\[(\d+)\]", block_pin)
    if match:
        return f"{_BUS[role][match.group(1)]}[{match.group(2)}]"
    if block_pin in ("VDD", "VSS"):
        return block_pin
    return _CONTROLS[role][block_pin]


def column_pins(spec: ColumnSpec) -> list[str]:
    """The column's pins, in the order of its subcircuit."""
    rows = range(MUX_ROWS)
    return [
        *(f"WWL[{i}]" for i in range(spec.wordlines)),
        *(f"RWL[{i}]" for i in range(spec.wordlines)),
        *(f"wysel[{k}]" for k in rows),
        *(f"wyseln[{k}]" for k in rows),
        "wprechn",
        "wrena",
        "wrenan",
        "D",
        *(f"rysel[{k}]" for k in rows),
        *(f"ryseln[{k}]" for k in rows),
        "rprechn",
        "sae",
        "oe",
        "oeb",
        "Q",
        "VDD",
        "VSS",
    ]


def _trim_rail_lig(blocks: list[Any]) -> int:
    """Start each mux row's rail LIG `RAIL_LIG_INSET` in from its array-facing edge.

    The row cells run an LI line under every rail, full width, so a V0 on the
    rail lands on local interconnect; at the edge that meets the array nothing
    lands that close, and the top rail's line would sit 17 nm from the end
    row's WLB contact.
    """
    rows = {
        c.name: c
        for block in blocks
        for c in block.dependencies(True)
        if c.name.endswith("__row")
    }
    trimmed = 0
    for row in rows.values():
        inner = row.get_polygons(depth=None)
        for poly in [q for q in row.polygons if q.layer == LIG]:
            x0, y0, x1, y1 = bbox(poly)
            if y1 - y0 > 0.017 or x1 - x0 < 0.3 or abs(x0) > 1e-6:
                continue  # not a rail line reaching the array-facing edge
            lo = x0 + RAIL_LIG_INSET
            for q in inner:
                if q.layer in (V0, GATE):
                    a, b, c, d = bbox(q)
                    if b < y1 and d > y0 and a < lo and c > x0:
                        raise RuntimeError(
                            f"{row.name}: a via or gate uses the rail LIG's end"
                        )
            row.remove(poly)
            rect(row, (lo, y0, x1, y1), LIG)
            trimmed += 1
    if not trimmed:
        raise RuntimeError("no rail LIG reaches the IO blocks' array-facing edge")
    return trimmed


def _build_blocks(specs: dict[str, IoColumnSpec]) -> dict[str, Any]:
    """Draw both blocks with chipforge's cell generators (gdspy, nm) and read them back in um."""
    import gdspy

    gdstk = require_gdstk()
    nm = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    previous = gdspy.current_library
    gdspy.current_library = nm
    try:
        names = {
            role: build_io_column(spec, lib=nm, draw_pin_labels=False).name
            for role, spec in specs.items()
        }
    finally:
        gdspy.current_library = previous
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "blocks.gds"
        nm.write_gds(str(path))
        um = gdstk.read_gds(str(path), unit=1e-6)
    cells = {cell.name: cell for cell in um.cells}
    _trim_rail_lig([cells[name] for name in names.values()])
    return {role: cells[name] for role, name in names.items()}


def _port_io(library: Any, role: str, block: Any, spec: IoColumnSpec) -> Any:
    """One port's IO: its block, its bitline entries facing the array.

    The write block stands left of the array, mirrored in x; the read block
    right of it as drawn.  Each sits `grid_offset` up the column so its fins
    fall on the array's.
    """
    gdstk = require_gdstk()
    width, height = spec.width / 1000, spec.height / 1000
    offset = spec.mux.grid_offset / 1000
    mirrored = role == "write"
    cell = library.new_cell(f"io1r1w_{role}_sram_8t")
    if mirrored:
        # A reflection in y then a half turn is a mirror in x.
        cell.add(
            gdstk.Reference(
                block, origin=(width, offset), rotation=math.pi, x_reflection=True
            )
        )
    else:
        cell.add(gdstk.Reference(block, origin=(0.0, offset)))
    layers = {"M1": M1, "M2": M2, "M3": M3, "M4": M4}
    for pin, (metal, (x, y)) in spec.pin_positions.items():
        base = pin.split(".")[0]
        px = (width - x / 1000) if mirrored else x / 1000
        py = offset + y / 1000
        if re.fullmatch(r"BLN?\[\d+\]", base):
            px = width - 0.006 if mirrored else 0.006  # on the array-facing edge
        cell.add(
            gdstk.Label(
                _pin_name(role, base),
                (px, py),
                layer=layers[metal],
                texttype=PIN_TEXTTYPE,
            )
        )
    rect(cell, (0.0, 0.0, width, height), BOUNDARY)
    # The write block's outer edge is its west one, the read block's its east.
    add_supply_straps(cell, "low" if mirrored else "high")
    return cell


def _cap_array(library: Any, cells: CellLibrary) -> Any:
    """The edge cap beside the write block: the column-group filler and one side cap per row."""
    gdstk = require_gdstk()
    name = "col_cap_x4_sram_8t"
    existing = [c for c in library.cells if c.name == name]
    if existing:
        return existing[0]
    filler = cells.cells[FILLER]
    x0, y0, x1, y1 = edge_boundary(cells.cells[topbot_name()])
    width, height = x1 - x0, y1 - y0
    fx0, fy0, fx1, fy1 = edge_boundary(filler)
    if abs(fy1 - fy0 - MUX_ROWS * height) > 1e-7:
        raise RuntimeError(
            "the column-group filler does not match the side caps' height"
        )
    cap = library.new_cell(name)
    cap.add(gdstk.Reference(filler, origin=(-fx0, -fy0)))
    for row in range(MUX_ROWS):
        master = cells.cells[topbot_name(False, bool(row % 2))]
        cap.add(gdstk.Reference(master, origin=(fx1 - fx0 - x0, row * height - y0)))
    rect(cap, (0.0, 0.0, width + fx1 - fx0, MUX_ROWS * height), BOUNDARY)
    return cap


def build_column(cells: CellLibrary, spec: ColumnSpec | None = None) -> Any:
    """Assemble one bit's column into `cells.library`; returns the gdstk cell."""
    gdstk = require_gdstk()
    spec = spec or ColumnSpec()
    library = cells.library
    array = build_array(library, cells, spec.wordlines, spec.tap_pitch)
    specs = io_specs(cells)
    blocks = _build_blocks(specs)
    for block in blocks.values():
        for dep in (block, *block.dependencies(True)):
            if dep.name not in {c.name for c in library.cells}:
                library.add(dep)
    io = {
        role: _port_io(library, role, blocks[role], specs[role])
        for role in ("write", "read")
    }
    cap = _cap_array(library, cells)

    arx0, ary0, arx1, ary1 = edge_boundary(array)
    wx0, wy0, wx1, _ = edge_boundary(io["write"])
    rx0, ry0, rx1, _ = edge_boundary(io["read"])
    cx0, cy0, cx1, _ = edge_boundary(cap)
    height = ary1 - ary0
    for part in (io["write"], io["read"], cap):
        _, y0, _, y1 = edge_boundary(part)
        if abs(y1 - y0 - height) > 1e-6:
            raise RuntimeError(f"{part.name} is not the array's height")
    cap_x = wx1 - wx0
    array_x = cap_x + cx1 - cx0
    read_x = array_x + arx1 - arx0
    width = read_x + rx1 - rx0

    column = library.new_cell(spec.name)
    column.add(gdstk.Reference(io["write"], origin=(-wx0, -wy0)))
    column.add(gdstk.Reference(cap, origin=(cap_x - cx0, -cy0)))
    column.add(gdstk.Reference(array, origin=(array_x - arx0, -ary0)))
    column.add(gdstk.Reference(io["read"], origin=(read_x - rx0, -ry0)))

    for source, name, layer in (("WLA", "WWL", M3), ("WLB", "RWL", M5)):
        for index, label in indexed_labels(array, source).items():
            x, y = float(label.origin[0]), float(label.origin[1])
            column.add(
                clone_label(
                    label, f"{name}[{index}]", (array_x + x - arx0, y - ary0), layer
                )
            )

    # Each port's bitlines must arrive where its block takes them; the write
    # port's cross the metal-free filler to the cap on M2.
    filler = next(r.cell for r in cap.references if r.cell.name == FILLER)
    if any(p.layer in (M2, M4) for p in filler.polygons):
        raise RuntimeError("the edge filler is no longer free of bitline metal")
    fx0, _, fx1, _ = edge_boundary(filler)
    for role, (io_cell, oy) in (
        ("write", (io["write"], wy0)),
        ("read", (io["read"], ry0)),
    ):
        for bus, (source, layer) in _ARRAY_BITLINES[role].items():
            ours = indexed_labels(array, source)
            theirs = indexed_labels(io_cell, _BUS[role][bus])
            if set(ours) != set(theirs):
                raise RuntimeError(f"{spec.name}: incomplete {source} interface")
            half = 0.009 if layer == M2 else 0.012
            for index, label in ours.items():
                y = float(label.origin[1]) - ary0
                io_y = float(theirs[index].origin[1]) - oy
                if theirs[index].layer != layer or abs(io_y - y) > half + 1e-6:
                    raise RuntimeError(
                        f"{spec.name}: {theirs[index].text} misses {source}[{index}]"
                    )
                if role == "write":
                    rect(
                        column,
                        (
                            cap_x - 0.018,
                            io_y - half,
                            cap_x + fx1 - fx0 + 0.018,
                            io_y + half,
                        ),
                        layer,
                    )

    for io_cell, io_x, (ox, oy) in (
        (io["write"], 0.0, (wx0, wy0)),
        (io["read"], read_x, (rx0, ry0)),
    ):
        for label in io_cell.labels:
            if re.fullmatch(r"[WR]BLN?\[\d+\]", label.text):
                continue  # internal: block to array
            if (
                label.text in ("VDD", "VSS")
                and label.layer == M1
                and io_cell is io["write"]
            ):
                continue  # one rail label per supply; the read block's are kept
            point = (io_x + float(label.origin[0]) - ox, float(label.origin[1]) - oy)
            column.add(clone_label(label, origin=point))

    # The write block meets the cap, the read block the array's last tap (or,
    # untapped, its last cell, with no room for every tie).
    tie_seam_supplies(column, cap_x, "left")
    tie_seam_supplies(column, read_x, "right", every=False)
    rect(column, (0.0, 0.0, width, height), BOUNDARY)
    return column


# ── Netlist ───────────────────────────────────────────────────────────────────
BITCELL_NETLIST = """\
* OpenFinRAM's ASAP7 8T cell (tech/spice/sram_cell_8t.sp): port A writes, port B reads.
.SUBCKT sram_cell_8t WLA WLB BLA BLAN BLB BLBN VDD VSS
M0 Q  QB VSS VSS nmos_sram L=2e-08 W=5.4e-08 nfin=2
M1 QB Q  VSS VSS nmos_sram L=2e-08 W=5.4e-08 nfin=2
M2 Q  QB VDD VDD pmos_sram L=2e-08 W=2.7e-08 nfin=1
M3 QB Q  VDD VDD pmos_sram L=2e-08 W=2.7e-08 nfin=1
M4 Q  WLA BLA  VSS nmos_sram L=2e-08 W=5.4e-08 nfin=2
M5 QB WLA BLAN VSS nmos_sram L=2e-08 W=5.4e-08 nfin=2
M6 Q  WLB BLB  VSS nmos_sram L=2e-08 W=5.4e-08 nfin=2
M7 QB WLB BLBN VSS nmos_sram L=2e-08 W=5.4e-08 nfin=2
.ENDS sram_cell_8t
"""


def column_netlist(cells: CellLibrary, spec: ColumnSpec | None = None) -> str:
    """The column's CDL: the bitcell, both blocks, and the column of their instances."""
    spec = spec or ColumnSpec()
    specs = io_specs(cells)
    lines = [BITCELL_NETLIST]
    blocks = {}
    for role, io_spec in specs.items():
        name = f"io1r1w_{role}_block"
        blocks[role] = (name, io_column_pins(io_spec))
        lines.append(block_netlist(io_spec, name=name).replace(".END\n", ""))
    lines.append(f".SUBCKT {spec.name} {' '.join(column_pins(spec))}")
    for k in range(MUX_ROWS):
        for i in range(spec.wordlines):
            lines.append(
                f"Xcell_{k}_{i} WWL[{i}] RWL[{i}] WBL[{k}] WBLN[{k}] RBL[{k}] RBLN[{k}] "
                "VDD VSS sram_cell_8t"
            )
    for role, (name, pins) in blocks.items():
        lines.append(f"X{role} {' '.join(_pin_name(role, p) for p in pins)} {name}")
    lines += [f".ENDS {spec.name}", ".END", ""]
    return "\n".join(lines)
