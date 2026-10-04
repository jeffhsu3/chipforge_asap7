"""Column IO for the 6T cell's 270 nm rows, its leaves in one column.

`StaggeredIoColumnSpec` puts the 297 nm `BitlineMuxSpec` leaves in two
staggered columns, because no leaf of that row style fits one 270 nm row.
`SidewaysMuxSpec` does (the released ASAP7 leaf's arrangement), so here one
column of them serves every row, each leaf beside its own row::

        ←── array rows (BL/BLN on M2) enter here, at x = 0
        │ leaves       │gap│ write  │ sense │ output│tap│
        │ leaf 3 (sw)  │   │ driver │ amp   │ latch │   │  row 3
        │ leaf 2       │   │        │       │  ▼    │   │  row 2
        │ leaf 1 (sw)  │   │        │       │       │   │  row 1
        │ leaf 0       │   │                          │  row 0
        └──────────────┴───┴──────────────────────────┘

The leaves are a `build_sideways_mux_group` raised `grid_offset` (half a fin
pitch) onto the array's fin grid, so the group overhangs the block's top by
that much, and the next block's starts where it ends; odd leaves are swapped
to meet the array's mirrored odd rows.  The bitlines enter each leaf on M2 at
the array's own heights: no lift, no M4.

The logic is `StaggeredIoColumnSpec`'s: the 594 nm row pairs of the write
driver, sense amplifier and output latch (mirrored) side by side, ``SA`` and
``SAN`` reaching them on M4 from the leaves' tracks, ``QA``/``QAN`` crossing
on M4 above them.  Supplies: every rail runs into the gap, to a VSS and a VDD
M3 strap there that run the block's height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..layout.grid import GATE_PITCH
from ..layout.layers import box, require_gdspy
from ..layout.rules import M4_PITCH
from .bitline_mux import BitlineMuxSpec
from .bitline_mux_270 import ROW, SidewaysMuxSpec, build_sideways_mux_group
from .cli import parse_spec, write_gds
from .io_column import _block_devices, io_column_pins
from .output_latch import OutputLatchSpec, build_output_latch
from .row_support import RowSupportSpec, build_row_support
from .rowcell import CAP, HALF, PAD, V3_M3_CAP, label, m3_column, m4_track, square
from .sense_amp_row import SenseAmpRowSpec, build_sense_amp_row
from .write_driver import WriteDriverSpec, build_write_driver

__all__ = ["SidewaysIoColumnSpec", "build_sideways_io_column", "sideways_block_netlist"]

#: The gap between the leaves and the logic: room for a VSS and a VDD strap,
#: a multiple of 108 nm so the logic starts on both the gate and the M3 grid.
_GAP = 162
_ORIGIN_GRID = 108
_STRAP_OFFSETS = (36, 108)  # from the gap's left edge: VSS, VDD
_RAIL_OVERHANG = 5
_PAD_HALF_X = 15
FIN_PITCH = 27


def _ceil_to(value: float, grid: int) -> int:
    return int(-(-value // grid) * grid)


@dataclass(frozen=True)
class SidewaysIoColumnSpec:
    """One data bit's column IO for 270 nm (6T) rows, one leaf per row in one column.

    Args:
        selects: the mux ratio, one leaf per array row.
        bitline_entry: ``(y_BL, y_BLN)`` of an unmirrored array row's M2
            bitlines, from the row's bottom.  The 6T cell's are the default.
        grid_offset: how far up the leaves sit on the array's fin grid.
        sense_amp, write_driver, output_latch: as in `IoColumnSpec`.
        tap: a tap at the end of the logic row.
        one_sense_phase: ``SAPRECHN`` strapped to ``SAE``; pin ``SAE`` only.
    """

    selects: int = 4
    bitline_entry: tuple[float, float] = (186.5, 83.5)
    grid_offset: float = 13.5
    sense_amp: SenseAmpRowSpec = field(default_factory=SenseAmpRowSpec)
    write_driver: WriteDriverSpec = field(default_factory=WriteDriverSpec)
    output_latch: OutputLatchSpec = field(default_factory=OutputLatchSpec)
    tap: bool = True
    one_sense_phase: bool = True
    two_sided = False
    row_pitch = ROW

    def __post_init__(self) -> None:
        object.__setattr__(self, "bitline_entry", tuple(self.bitline_entry))
        if self.selects < 2 or self.selects % 2:
            raise ValueError(f"selects must be even and at least 2, got {self.selects}")
        rows = {(spec.band_height, spec.vt) for spec in (self.sense_amp, self.write_driver, self.output_latch)}
        if len(rows) != 1:
            raise ValueError(f"the three logic cells are not on one row: {sorted(rows)}")
        self.leaves  # the leaf rejects an entry it cannot take
        if self.logic_top + 2 * M4_PITCH + V3_M3_CAP > self.height:
            raise ValueError(
                f"the logic pair and its output lines need {self.logic_top + 2 * M4_PITCH + V3_M3_CAP} nm; "
                f"the block is {self.height}"
            )

    # ── The parts ─────────────────────────────────────────────────────────────
    @property
    def leaves(self) -> SidewaysMuxSpec:
        return SidewaysMuxSpec(selects=self.selects, bitline_entry=self.bitline_entry, grid_offset=self.grid_offset)

    @property
    def mux(self) -> BitlineMuxSpec:
        """The mux as `IoColumnSpec`'s netlist helpers enumerate it: the leaves' six devices a select."""
        return BitlineMuxSpec(selects=self.selects)

    # ── Sizes ─────────────────────────────────────────────────────────────────
    @property
    def height(self) -> int:
        return self.selects * ROW

    @property
    def gap_x(self) -> int:
        return self.leaves.width

    @property
    def straps(self) -> dict[str, float]:
        near, far = _STRAP_OFFSETS
        return {"VSS": self.gap_x + near, "VDD": self.gap_x + far}

    @property
    def logic_x(self) -> int:
        return _ceil_to(self.gap_x + _GAP, _ORIGIN_GRID)

    @property
    def leaf_rails(self) -> list[tuple[float, str]]:
        """``(y, net)`` of every rail of the leaves: VSS under leaf 0, alternating up."""
        return [(self.grid_offset + k * ROW, "VDD" if k % 2 else "VSS") for k in range(self.selects + 1)]

    @property
    def logic_y(self) -> float:
        """Bottom of the logic pair: on the leaves' fin grid, its rails three fin pitches from theirs."""
        y = self.grid_offset + 2 * FIN_PITCH
        rails = [r for r, _ in self.leaf_rails]
        while any(abs(y + k * self.write_driver.stack.height / 2 - r) < 3 * FIN_PITCH for r in rails for k in range(3)):
            y += FIN_PITCH
        return y

    @property
    def logic_rails(self) -> list[tuple[float, str]]:
        return [(self.logic_y + y, net) for y, net in self.write_driver.stack.rails]

    @property
    def logic_top(self) -> float:
        return self.logic_y + self.sense_amp.height

    @property
    def placements(self) -> dict[str, tuple[float, float]]:
        x, y = self.logic_x, self.logic_y
        wd, sa = self.write_driver.width, self.sense_amp.width
        return {
            "write_driver": (x, y),
            "sense_amp": (x + wd, y),
            "output_latch": (x + wd + sa, y),  # mirrored: this is its bottom-left
            "tap": (x + wd + sa + self.output_latch.width, y),
        }

    @property
    def width(self) -> int:
        tap = 2 * GATE_PITCH if self.tap else 0
        return int(self.placements["tap"][0] + tap)

    @property
    def cell_name(self) -> str:
        sa, wd, ol = self.sense_amp, self.write_driver, self.output_latch
        y_bl, y_bln = (round(10 * y) for y in self.bitline_entry)
        return (
            f"iocol270_x{self.selects}_in{y_bl}x{y_bln}_sa{sa.n_fingers}t{sa.tail_fingers}_wd{wd.keeper_fins}"
            f"_ol{ol.fingers}{'' if self.tap else '_notap'}{'_1ph' if self.one_sense_phase else ''}"
        )

    # ── Where things are ──────────────────────────────────────────────────────
    def bitline_ys(self, row: int) -> dict[str, float]:
        """Where row `row`'s bitlines cross the block's edge (and enter its leaf)."""
        base = row * ROW
        y_bl, y_bln = self.bitline_entry
        if row % 2:
            return {"BL": base + ROW - y_bl, "BLN": base + ROW - y_bln}
        return {"BL": base + y_bl, "BLN": base + y_bln}

    def track_x(self, net: str) -> float:
        return self.leaves.track_x[net]

    @property
    def sense_line_ys(self) -> tuple[float, float]:
        """y of the ``SA`` and ``SAN`` lines to the logic, in the pair's bottom row (whole nm)."""
        y = math.ceil(self.logic_y)
        return (y + 2 * M4_PITCH, y + 3 * M4_PITCH)

    @property
    def routes(self) -> dict[str, list[tuple[float, list[float]]]]:
        """``net -> [(y, [x, ...])]``: the block's M4 lines, each over the M3 it joins."""
        place = self.placements
        wd_x, sa_x, ol_x = (place[k][0] for k in ("write_driver", "sense_amp", "output_latch"))
        sa, wd, ol = self.sense_amp.track_x, self.write_driver.track_x, self.output_latch.track_x
        y_sa, y_san = self.sense_line_ys
        top_line = math.ceil(self.logic_top)
        return {
            "SA": [(y_sa, [self.track_x("SA"), wd_x + wd["SA"], sa_x + sa["SA"]])],
            "SAN": [(y_san, [self.track_x("SAN"), wd_x + wd["SAN"], sa_x + sa["SAN"]])],
            "QA": [(top_line + M4_PITCH, [sa_x + sa["QA"], ol_x + ol["QA"]])],
            "QAN": [(top_line + 2 * M4_PITCH, [sa_x + sa["QAN"], ol_x + ol["QAN"]])],
        }

    @property
    def sense_phase_strap(self) -> tuple[float, float, float] | None:
        if not self.one_sense_phase:
            return None
        sa = self.sense_amp
        x0, y0 = self.placements["sense_amp"]
        _, tie_mid, _ = sa.tie_levels
        return (x0 + sa.track_x["SAE"], y0 + tie_mid, y0 + sa.stack.seam_ys[1] - 17)

    @property
    def pin_positions(self) -> dict[str, tuple[str, tuple[float, float]]]:
        pins: dict[str, tuple[str, tuple[float, float]]] = {}
        for row in range(self.selects):
            for net, y in self.bitline_ys(row).items():
                pins[f"{net}[{row}]"] = ("M2", (PAD, y))
        group = self.leaves.group_pin_positions()
        for row in range(self.selects):
            for net in ("YSEL", "YSELN"):
                metal, (x, y) = group[f"{net}[{row}]"]
                pins[f"{net}[{row}]"] = (metal, (x, y + self.grid_offset))
        metal, (x, y) = group["PRECHN"]
        pins["PRECHN"] = (metal, (x, y + self.grid_offset))
        place = self.placements

        def take(spec, key, names, mirrored=False):
            ox, oy = place[key]
            for pin, (metal, (x, y)) in spec.pin_positions.items():
                if pin in names:
                    pins[pin] = (metal, (ox + x, oy + (spec.height - y if mirrored else y)))

        if self.one_sense_phase:
            metal, (x, y) = self.sense_amp.pin_positions["SAPRECHN"]
            ox, oy = place["sense_amp"]
            pins["SAE"] = (metal, (ox + x, oy + y))
        else:
            take(self.sense_amp, "sense_amp", ("SAE", "SAPRECHN"))
        take(self.write_driver, "write_driver", ("D", "WRENA", "WRENAN"))
        take(self.output_latch, "output_latch", ("OE", "OEB", "Q"), mirrored=True)
        for net, x in self.straps.items():
            pins[net] = ("M3", (x, self.height / 2))
        return pins


def sideways_block_netlist(spec: SidewaysIoColumnSpec, name: str | None = None) -> str:
    """A flat BSIM-CMG subcircuit of the block, on `IoColumnSpec`'s pins and nets."""
    title = name or spec.cell_name
    n_band, p_band = spec.leaves.bands
    lines = [
        "* ASAP7 column IO, 270 nm leaves in one column: bitline mux, sense amplifier, write driver, output latch",
        f".SUBCKT {title} {' '.join(io_column_pins(spec))}",  # pyright: ignore[reportArgumentType, reportCallIssue]
    ]
    for device, d, g, s, flavor, fins, _ in _block_devices(spec):  # pyright: ignore[reportArgumentType]
        band = n_band if flavor == "n" else p_band
        bulk = "VSS" if flavor == "n" else "VDD"
        lines.append(f"{device} {d} {g} {s} {bulk} {band.spec.model} nfin={fins} l={band.spec.gate_length}n nf=1 m=1")
    lines += [f".ENDS {title}", ".END", ""]
    return "\n".join(lines)


def build_sideways_io_column(
    spec: SidewaysIoColumnSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Place the leaf column and the logic, route, strap the supplies; return the gdspy Cell."""
    spec = spec or SidewaysIoColumnSpec()
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = lib.new_cell(cell_name) if lib is not None else gdspy.Cell(cell_name, exclude_from_current=True)

    def reuse(name_: str, build):
        existing = lib.cells.get(name_) if lib is not None else None
        return existing if existing is not None else build()

    # ── Leaves: one group, half a fin pitch up ────────────────────────────────
    leaves = spec.leaves
    group = reuse(leaves.group_cell_name, lambda: build_sideways_mux_group(leaves, lib=lib, draw_pin_labels=False))
    cell.add(gdspy.CellReference(group, origin=(0, spec.grid_offset)))
    # The array's bitline bars end at the block's edge; carry each to its leaf's.
    for row in range(spec.selects):
        for y in spec.bitline_ys(row).values():
            box(cell, "M2", 0, y - HALF, PAD + HALF, y + HALF)

    # ── Logic ─────────────────────────────────────────────────────────────────
    place = spec.placements
    logic = {
        "write_driver": reuse(spec.write_driver.cell_name,
                              lambda: build_write_driver(spec.write_driver, lib=lib, draw_pin_labels=False)),
        "sense_amp": reuse(spec.sense_amp.cell_name,
                           lambda: build_sense_amp_row(spec.sense_amp, lib=lib, draw_pin_labels=False)),
        "output_latch": reuse(spec.output_latch.cell_name,
                              lambda: build_output_latch(spec.output_latch, lib=lib, draw_pin_labels=False)),
    }  # fmt: skip
    for key, placed in logic.items():
        x, y = place[key]
        if key == "output_latch":
            cell.add(gdspy.CellReference(placed, origin=(x, y + spec.output_latch.height), x_reflection=True))
        else:
            cell.add(gdspy.CellReference(placed, origin=(x, y)))
    if spec.tap:
        tap_spec = RowSupportSpec(stack=spec.write_driver.stack, kind="tap", width_cpp=2)
        tap = reuse(tap_spec.cell_name, lambda: build_row_support(tap_spec, lib=lib))
        cell.add(gdspy.CellReference(tap, origin=place["tap"]))

    # ── Routes ────────────────────────────────────────────────────────────────
    top = spec.logic_top
    for net, lines in spec.routes.items():
        for y, xs in lines:
            if net in ("QA", "QAN"):
                # The amplifier's risers and the mirrored latch's stubs end on
                # the pair's top edge: carry both up to the line.
                for x in xs:
                    m3_column(cell, x, top - CAP, y + V3_M3_CAP)
            m4_track(cell, y, xs)
    if strap := spec.sense_phase_strap:
        x, y0, y1 = strap
        m3_column(cell, x, y0, y1)

    # ── Supplies: every rail into the gap, to its net's strap ─────────────────
    straps = spec.straps
    # The leaves' top rail is half a fin pitch above the block: the next
    # block's bottom one, strapped there.
    inside = [(y, net) for y, net in spec.leaf_rails if y < spec.height]
    rails = {"leaves": (inside, spec.gap_x, +1), "logic": (spec.logic_rails, spec.logic_x, -1)}
    extensions: dict[str, list[tuple[float, float, float]]] = {"VSS": [], "VDD": []}
    vias: dict[float, list[float]] = {}
    for found, x_edge, sign in rails.values():
        for y, net in found:
            x_strap = straps[net]
            reach = x_strap + sign * (HALF + _RAIL_OVERHANG)
            x0, x1 = sorted((x_edge, reach))
            box(cell, "M1", x0, y - HALF, x1, y + HALF)
            extensions[net].append((x0, x1, y))
            if any(abs(y - other) < 4 * HALF for other in vias.setdefault(x_strap, [])):
                continue
            vias[x_strap].append(y)
            square(cell, "V1", x_strap, y)
            box(cell, "M2", x_strap - _PAD_HALF_X, y - HALF, x_strap + _PAD_HALF_X, y + HALF)
            square(cell, "V2", x_strap, y)
    for net, found in extensions.items():
        for i, (a0, a1, ya) in enumerate(found):
            for b0, b1, yb in found[i + 1 :]:
                lo, hi = max(a0, b0), min(a1, b1)
                if hi > lo and 0 < abs(ya - yb) < 4 * HALF:
                    box(cell, "M1", lo, min(ya, yb), hi, max(ya, yb))
    for x in straps.values():
        m3_column(cell, x, 0, spec.height)

    if draw_pin_labels:
        for pin, (metal, origin) in spec.pin_positions.items():
            label(cell, pin, metal, origin)
    box(cell, "BOUNDARY", 0, 0, spec.width, spec.height)
    return cell


def main(argv: list[str] | None = None) -> None:
    """Write a single-column 270 nm column-IO GDS: ``asap7-io-column-270 --selects 4``."""
    args = parse_spec(
        SidewaysIoColumnSpec,
        argv,
        description="Generate one data bit's column IO for an array of 270 nm rows (ASAP7 6T), leaves in one column.",
    )
    spec = args.spec
    cell, out = write_gds(lambda lib: build_sideways_io_column(spec, lib=lib), args.out)
    print(f"✓ {cell.name}: {spec.width} x {spec.height} nm -> {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
