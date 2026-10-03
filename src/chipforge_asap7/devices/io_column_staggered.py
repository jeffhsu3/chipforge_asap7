"""Column IO for an array whose rows are shorter than a leaf: the 6T cell's 270 nm.

`IoColumnSpec` stacks one bitline leaf per array row, which needs a row at
least as tall as the leaf (297 nm on the dense row; neither band of the leaf
or the logic cells goes below 135 + 162).  The released 6T cell is 270 nm
tall, so its leaves are staggered instead: the even rows' leaves stand in one
column, the odd rows' in a second, each leaf on the 540 nm of its row *pair*::

        ←── array rows (BL/BLN on M2) enter here
        │lift│ column E │gap│ column O │gap│ write  │ sense │ output│tap│
        │    │          │   │ leaf 3 ▼ │   │ driver │ amp   │ latch │   │  row 3
        │    │ leaf 2   │   │          │   │        │       │  ▼    │   │  row 2
        │    │          │   │ leaf 1 ▼ │   │        │       │       │   │  row 1
        │    │ leaf 0   │   │          │   │                          │  row 0
        └────┴──────────┴───┴──────────┴───┴────────────────────────────┘

Each leaf is the array row it serves, raised by `leaf_offset`: an odd number
of half fin pitches (`GRID_OFFSET`), so that its fins are on the array's
grid, and more than one so that the rails of the first even leaf and the last
odd one stay clear of the next block's at a stacking seam.  The odd leaves
are the even ones mirrored about the seam of their pair, as the array's rows
are.  The bitlines climb from M2
to M4 in the *lift* at the block's edge and run on M4 to their leaf, so the
odd rows' cross column E over its leaves; on M2 they would have to share the
leaf's own M2 tracks, which no entry height clears.

Each column is a `selects / 2`-to-one group: leaf ``k`` of column E answers to
``YSEL[2k]`` and leaf ``k`` of column O to ``YSEL[2k + 1]``, so each column
carries only its own selects.  Every M3 track runs the block's height, bridged
over the gaps between a column's leaves; ``SA``, ``SAN`` and ``PRECHN`` of the
two columns are joined on M4 at a height the odd rows' bitlines leave free.

The logic cells are the 594 nm row pairs of `IoColumnSpec`, side by side in
one pair instead of stacked: write driver, sense amplifier, and the output
latch mirrored so its ``QA``/``QAN`` stubs face up.  ``SA``/``SAN`` reach the
driver's and the amplifier's columns on M4 inside the pair; ``QA``/``QAN``
climb above it and cross on M4 to the latch.

Supplies: every rail of a column runs on into the gap beside it, to a VSS and
a VDD M3 strap there; M2 bridges across column O, where it has no leaf, join
the two gaps' straps.  So the block has one VDD and one VSS, and full-height
M3 straps for a grid to land on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from ..layout.grid import GATE_PITCH
from ..layout.layers import box, require_gdspy
from ..layout.rules import M4_PITCH, M4_X_GRID
from .bitline_mux import BitlineMuxSpec, build_bitline_mux
from .cli import parse_spec, write_gds
from .io_column import _block_devices, io_column_pins
from .output_latch import OutputLatchSpec, build_output_latch
from .row_support import RowSupportSpec, build_row_support
from .rowcell import (
    CAP,
    HALF,
    M4_HALF,
    PAD,
    V3_M3_CAP,
    V3_M4_CAP,
    label,
    m3_column,
    m4_track,
    square,
)
from .sense_amp_row import SenseAmpRowSpec, build_sense_amp_row
from .write_driver import WriteDriverSpec, build_write_driver

__all__ = [
    "GRID_OFFSET",
    "StaggeredIoColumnSpec",
    "build_staggered_io_column",
    "staggered_block_netlist",
]

#: Half a fin pitch: a leaf this far (or this plus whole pitches) above the
#: bottom of an array row has its fins (a space centred on its edge) on the
#: array's (a fin centred on its boundary).
GRID_OFFSET = 13.5
#: x of the lift's M3 stubs, and the lift's width.
_LIFT_X, _LIFT_WIDTH = 72, 108
#: Gaps between columns: room for a VSS and a VDD strap, and multiples of
#: 108 nm so that every column starts on both the gate grid and the M3 grid.
_GAP = 162
_ORIGIN_GRID = 108
_STRAP_OFFSETS = (36, 108)  # from a gap's left edge: the near and the far strap
_RAIL_OVERHANG = 5
_PAD_HALF_X = 15


def _ceil_to(value: float, grid: int) -> int:
    return int(-(-value // grid) * grid)


@dataclass(frozen=True)
class StaggeredIoColumnSpec:
    """One data bit's column IO under an array of short rows.

    Args:
        leaf: the bitline leaf all of them are drawn from (fins, bands, VT);
            its ``selects``, ``select`` and entry are set here.
        selects: the mux ratio, even; half the leaves go in each column.
        row_pitch: the array's row height.
        bitline_entry: ``(y_BL, y_BLN)`` of an unflipped array row's M2
            bitlines, from the row's bottom.  The 6T cell's are the default.
        leaf_offset: how far an even leaf sits above its row's bottom (and an
            odd one below its row's top).
        sense_amp, write_driver, output_latch: as in `IoColumnSpec`.
        tap: a tap at the end of the logic row.
        one_sense_phase: ``SAPRECHN`` strapped to ``SAE``; pin ``SAE`` only.
    """

    leaf: BitlineMuxSpec = field(default_factory=BitlineMuxSpec)
    selects: int = 4
    row_pitch: int = 270
    bitline_entry: tuple[float, float] = (186.5, 83.5)
    leaf_offset: float = 3 * GRID_OFFSET
    sense_amp: SenseAmpRowSpec = field(default_factory=SenseAmpRowSpec)
    write_driver: WriteDriverSpec = field(default_factory=WriteDriverSpec)
    output_latch: OutputLatchSpec = field(default_factory=OutputLatchSpec)
    tap: bool = True
    one_sense_phase: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "bitline_entry", tuple(self.bitline_entry))
        if self.selects < 2 or self.selects % 2:
            raise ValueError(f"selects must be even and at least 2, got {self.selects}")
        rows = {
            (spec.band_height, spec.vt)
            for spec in (self.leaf, self.sense_amp, self.write_driver, self.output_latch)
        }
        if len(rows) != 1:
            raise ValueError(f"the four cells are not on one row: {sorted(rows)}")
        if self.leaf.rows != 1:
            raise ValueError("a staggered leaf is one row")
        if (self.leaf_offset / GRID_OFFSET) % 2 != 1:
            raise ValueError(
                f"leaf_offset must be an odd multiple of {GRID_OFFSET} nm, got {self.leaf_offset}"
            )
        if self.leaf_height + self.leaf_offset > 2 * self.row_pitch - self.leaf_offset:
            raise ValueError(
                f"a {self.leaf_height} nm leaf does not fit a {2 * self.row_pitch} nm row pair"
            )
        if self.logic_top + 2 * M4_PITCH + V3_M3_CAP > self.height:
            raise ValueError(
                f"the logic pair and its output lines need {self.logic_top + 2 * M4_PITCH} nm; "
                f"the block is {self.height}"
            )
        self.column_leaf(0)  # the leaf rejects an entry it cannot take

    # ── The mux as `IoColumnSpec`'s netlist helpers see it ─────────────────────
    @property
    def mux(self) -> BitlineMuxSpec:
        """The leaf with the whole block's mux ratio: what the netlist enumerates."""
        return replace(self.leaf, selects=self.selects, select=0, bitline_entry=None)

    two_sided = False

    @property
    def per_column(self) -> int:
        return self.selects // 2

    def column_leaf(self, k: int) -> BitlineMuxSpec:
        """Leaf ``k`` of either column: local select ``k`` of the column's, bitlines on M4.

        The M4 runs on a whole nanometre half a nanometre off a bitline the
        array draws on a half (the 6T cell's), so that a macro router never
        meets half-nanometre metal: the lift's M3 stub takes up the half.
        """
        y_bl, y_bln = (y - self.leaf_offset + (0.5 if y % 1 else 0.0) for y in self.bitline_entry)
        return replace(
            self.leaf,
            selects=self.per_column,
            select=k,
            bitline_entry=(y_bl, y_bln),
            bitline_layer="M4",
            grid_offset=0.0,
        )

    # ── Sizes ─────────────────────────────────────────────────────────────────
    @property
    def leaf_height(self) -> int:
        return self.leaf.row_height

    @property
    def leaf_width(self) -> int:
        return self.column_leaf(0).width

    @property
    def height(self) -> int:
        return self.selects * self.row_pitch

    @property
    def column_x(self) -> tuple[int, int]:
        """Left edges of column E and column O."""
        x_e = _ceil_to(_LIFT_WIDTH, _ORIGIN_GRID)
        x_o = _ceil_to(x_e + self.leaf_width + _GAP, _ORIGIN_GRID)
        return x_e, x_o

    @property
    def gaps(self) -> tuple[int, int]:
        """Left edges of the gap after column E and the gap after column O."""
        x_e, x_o = self.column_x
        return x_e + self.leaf_width, x_o + self.leaf_width

    @property
    def logic_x(self) -> int:
        return _ceil_to(self.gaps[1] + _GAP, _ORIGIN_GRID)

    @property
    def logic_y(self) -> float:
        """Bottom of the logic pair: on the leaves' fin grid, its rails clear of column O's."""
        pitch = 2 * GRID_OFFSET
        rails = [y for y, _ in self.leaf_rails("O")]
        y = GRID_OFFSET + pitch * 8
        while any(
            abs(y + k * self.leaf_height - r) < 3 * pitch for r in rails for k in range(3)
        ):
            y += pitch
        return y

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
        lf, sa, wd, ol = self.leaf, self.sense_amp, self.write_driver, self.output_latch
        y_bl, y_bln = (round(10 * y) for y in self.bitline_entry)
        return (
            f"iocol6_x{self.selects}_p{self.row_pitch}_{lf.n_fins}n{lf.p_fins}p"
            f"_h{lf.band_height[0]}x{lf.band_height[1]}{'' if lf.vt == 'rvt' else '_' + lf.vt}"
            f"_in{y_bl}x{y_bln}{'' if self.leaf_offset == 3 * GRID_OFFSET else f'_o{round(10 * self.leaf_offset)}'}"
            f"_sa{sa.n_fingers}t{sa.tail_fingers}_wd{wd.keeper_fins}"
            f"_ol{ol.fingers}{'' if self.tap else '_notap'}{'_1ph' if self.one_sense_phase else ''}"
        )

    # ── Where things are ──────────────────────────────────────────────────────
    def leaf_origin(self, row: int) -> tuple[int, float, bool]:
        """``(x, y, mirrored)`` of the reference that places row `row`'s leaf."""
        x_e, x_o = self.column_x
        pair = row // 2 * 2 * self.row_pitch
        if row % 2 == 0:
            return x_e, pair + self.leaf_offset, False
        # The even leaf mirrored about the pair's seam: its bottom is up top.
        return x_o, pair + 2 * self.row_pitch - self.leaf_offset, True

    def leaf_span(self, row: int) -> tuple[float, float]:
        _, y, mirrored = self.leaf_origin(row)
        return (y - self.leaf_height, y) if mirrored else (y, y + self.leaf_height)

    def rows_of(self, column: str) -> list[int]:
        return list(range(0 if column == "E" else 1, self.selects, 2))

    def leaf_rails(self, column: str) -> list[tuple[float, str]]:
        """``(y, net)`` of every M1 rail of a column's leaves."""
        rails = []
        for row in self.rows_of(column):
            y0, y1 = self.leaf_span(row)
            _, _, mirrored = self.leaf_origin(row)
            bottom, top = ("VDD", "VSS") if mirrored else ("VSS", "VDD")
            rails += [(y0, bottom), (y1, top)]
        return rails

    def bitline_ys(self, row: int) -> dict[str, float]:
        """Where row `row`'s bitlines cross the block's edge."""
        base = row * self.row_pitch
        y_bl, y_bln = self.bitline_entry
        if row % 2:
            return {"BL": base + self.row_pitch - y_bl, "BLN": base + self.row_pitch - y_bln}
        return {"BL": base + y_bl, "BLN": base + y_bln}

    def m4_ys(self, row: int) -> dict[str, float]:
        """Where row `row`'s bitlines run on M4, from the lift to their leaf: its entry as placed."""
        _, y, mirrored = self.leaf_origin(row)
        entry = self.column_leaf(0).entry_y
        return {net: (y - entry[net]) if mirrored else (y + entry[net]) for net in ("BL", "BLN")}

    def track_x(self, column: str, net: str) -> float:
        """Block x of a column's M3 track (`net` as the leaf names it, e.g. ``YSEL[0]``)."""
        x_e, x_o = self.column_x
        return (x_e if column == "E" else x_o) + self.column_leaf(0).track_x[net]

    def global_select(self, column: str, k: int) -> int:
        return 2 * k + (0 if column == "E" else 1)

    @property
    def joins(self) -> dict[str, float]:
        """y of the M4 lines joining column E's SA, SAN and PRECHN to column O's.

        They cross column E, as the odd rows' bitlines do, and run into
        column O beside the logic's sense lines, so they take heights at
        least an M4 pitch from all of those, where both columns' sense
        tracks exist.
        """
        crossing = [y for row in self.rows_of("O") for y in self.m4_ys(row).values()]
        crossing += self.sense_line_ys
        lo = max(self.leaf_span(0)[0], self.leaf_span(1)[0]) + V3_M3_CAP
        hi = min(self.leaf_span(self.selects - 2)[1], self.leaf_span(self.selects - 1)[1])
        hi -= V3_M3_CAP
        # Whole nanometres: the leaves sit half a fin pitch off the row grid,
        # and a macro router landing on a line half a nanometre off its own
        # grid leaves a bend in the merged M4.
        found, y = [], math.ceil(lo)
        while len(found) < 3 and y <= hi:
            if all(abs(y - b) >= M4_PITCH for b in crossing) and all(
                y - f >= M4_PITCH for f in found
            ):
                found.append(y)
                y += M4_PITCH
            else:
                y += 1
        if len(found) < 3:
            raise ValueError("no room to join the two columns' sense and precharge tracks")
        return dict(zip(("SA", "SAN", "PRECHN"), found, strict=True))

    @property
    def sense_line_ys(self) -> tuple[float, float]:
        """y of the ``SA`` and ``SAN`` lines to the logic: in the pair's bottom row, as in `IoColumnSpec`.

        On whole nanometres, as every M4 line of the block is (see `joins`).
        """
        y = math.ceil(self.logic_y)
        return (y + 2 * M4_PITCH, y + 3 * M4_PITCH)

    @property
    def routes(self) -> dict[str, list[tuple[float, list[float]]]]:
        """``net -> [(y, [x, ...])]``: the block's M4 lines, each over the M3 it joins."""
        place = self.placements
        wd_x, sa_x, ol_x = (place[k][0] for k in ("write_driver", "sense_amp", "output_latch"))
        sa, wd, ol = self.sense_amp.track_x, self.write_driver.track_x, self.output_latch.track_x
        top = self.logic_top
        routes = {
            net: [(y_join, [self.track_x("E", net), self.track_x("O", net)])]
            for net, y_join in self.joins.items()
        }
        y_sa, y_san = self.sense_line_ys
        routes["SA"].append((y_sa, [self.track_x("O", "SA"), wd_x + wd["SA"], sa_x + sa["SA"]]))
        routes["SAN"].append(
            (y_san, [self.track_x("O", "SAN"), wd_x + wd["SAN"], sa_x + sa["SAN"]])
        )
        top_line = math.ceil(top)
        routes["QA"] = [(top_line + M4_PITCH, [sa_x + sa["QA"], ol_x + ol["QA"]])]
        routes["QAN"] = [(top_line + 2 * M4_PITCH, [sa_x + sa["QAN"], ol_x + ol["QAN"]])]
        return routes

    @property
    def straps(self) -> dict[str, tuple[float, float]]:
        """``net -> (x in the first gap, x in the second)`` of the M3 supply straps.

        In each gap the strap whose net has the rail nearest the gap's left
        column sits nearer that column, so that no rail crosses a rail of the
        other net it lies close to.
        """
        g1, g2 = self.gaps
        near, far = _STRAP_OFFSETS
        # Gap 1: column E's rails reach right, column O's left; with VSS near
        # E, E's VDD rails cross the VSS strap and O's VSS rails the VDD one.
        return {"VSS": (g1 + near, g2 + far), "VDD": (g1 + far, g2 + near)}

    @property
    def bridges(self) -> dict[str, float]:
        """y of the M2 lines that join the two gaps' straps across column O, below its first leaf."""
        lo, hi = 0, self.leaf_span(1)[0]
        ys = [y for y, _ in self.leaf_rails("E")] + [y for y, _ in self.leaf_rails("O")]
        found, y = [], lo + 2 * HALF
        while len(found) < 2 and y < hi - 2 * HALF:
            if all(abs(y - r) >= 4 * HALF for r in ys) and all(y - f >= 4 * HALF for f in found):
                found.append(y)
            y += 1
        if len(found) < 2:
            raise ValueError("no room below column O's first leaf for the supply bridges")
        return dict(zip(("VSS", "VDD"), found, strict=True))

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
        for column in ("E", "O"):
            for k, row in enumerate(self.rows_of(column)):
                y0, y1 = self.leaf_span(row)
                for net in ("YSEL", "YSELN"):
                    pins[f"{net}[{row}]"] = ("M3", (self.track_x(column, f"{net}[{k}]"), (y0 + y1) / 2))
        y0, y1 = self.leaf_span(1)
        pins["PRECHN"] = ("M3", (self.track_x("O", "PRECHN"), (y0 + y1) / 2))
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
        for net, (x, _) in self.straps.items():
            pins[net] = ("M3", (x, self.height / 2))
        return pins


def staggered_block_netlist(spec: StaggeredIoColumnSpec, name: str | None = None) -> str:
    """A flat BSIM-CMG subcircuit of the block, on `IoColumnSpec`'s pins and nets."""
    title = name or spec.cell_name
    n_band, p_band = spec.leaf.bands
    lines = [
        "* ASAP7 column IO, staggered leaves: bitline mux, sense amplifier, write driver, output latch",
        f".SUBCKT {title} {' '.join(io_column_pins(spec))}",  # pyright: ignore[reportArgumentType]
    ]
    # `IoColumnSpec`'s helpers read only `mux`, `two_sided`, `one_sense_phase`
    # and the three logic specs, which this spec has.
    for device, d, g, s, flavor, fins, _ in _block_devices(spec):  # pyright: ignore[reportArgumentType]
        band = n_band if flavor == "n" else p_band
        bulk = "VSS" if flavor == "n" else "VDD"
        lines.append(
            f"{device} {d} {g} {s} {bulk} {band.spec.model} nfin={fins} l={band.spec.gate_length}n nf=1 m=1"
        )
    lines += [f".ENDS {title}", ".END", ""]
    return "\n".join(lines)


def build_staggered_io_column(
    spec: StaggeredIoColumnSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Place the leaves and the logic, lift and route, strap the supplies; return the gdspy Cell."""
    spec = spec or StaggeredIoColumnSpec()
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = lib.new_cell(cell_name) if lib is not None else gdspy.Cell(cell_name, exclude_from_current=True)
    height = spec.height

    def reuse(name_: str, build):
        existing = lib.cells.get(name_) if lib is not None else None
        return existing if existing is not None else build()

    # ── Leaves ────────────────────────────────────────────────────────────────
    for column in ("E", "O"):
        for k, row in enumerate(spec.rows_of(column)):
            leaf_spec = spec.column_leaf(k)
            leaf = reuse(
                leaf_spec.cell_name,
                lambda leaf_spec=leaf_spec: build_bitline_mux(leaf_spec, lib=lib, draw_pin_labels=False),
            )
            x, y, mirrored = spec.leaf_origin(row)
            cell.add(gdspy.CellReference(leaf, origin=(x, y), x_reflection=mirrored))

    # Every M3 track of a column, bridged over its gaps: the selects and the
    # precharge the block's height (they are the same nets in the next bit's
    # block), the sense lines from the column's first leaf to its last.
    leaf0 = spec.column_leaf(0)
    for column in ("E", "O"):
        spans = sorted(spec.leaf_span(row) for row in spec.rows_of(column))
        for net in leaf0.track_x:
            x = spec.track_x(column, net)
            if net in ("SA", "SAN"):
                lo, hi = spans[0][0], spans[-1][1]
            else:
                lo, hi = 0, height
            edges = [lo, *(y for span in spans for y in span), hi]
            for y0, y1 in zip(edges[::2], edges[1::2], strict=True):
                if y1 > y0:
                    m3_column(cell, x, y0, y1)

    # ── Bitlines: M2 in, up to M4 in the lift, across to the leaf's entry ──
    for row in range(spec.selects):
        x_leaf, _, _ = spec.leaf_origin(row)
        entry = spec.column_leaf(0).entry_x
        m4 = spec.m4_ys(row)
        for net, y_m2 in spec.bitline_ys(row).items():
            y = m4[net]
            box(cell, "M2", 0, y_m2 - HALF, _LIFT_X + PAD, y_m2 + HALF)
            square(cell, "V2", _LIFT_X, y_m2)
            box(cell, "V3", _LIFT_X - HALF, y - M4_HALF, _LIFT_X + HALF, y + M4_HALF)
            m3_column(cell, _LIFT_X, min(y, y_m2) - V3_M3_CAP, max(y, y_m2) + V3_M3_CAP)
            # Past the leaf's own M4, so the wire's far end is the block's, on grid.
            end = x_leaf + entry[net] + V3_M4_CAP
            x0 = _LIFT_X - V3_M4_CAP
            box(cell, "M4", x0 - x0 % M4_X_GRID, y - M4_HALF, _ceil_to(end, M4_X_GRID), y + M4_HALF)

    # ── Logic ─────────────────────────────────────────────────────────────────
    place = spec.placements
    logic = {
        "write_driver": reuse(
            spec.write_driver.cell_name,
            lambda: build_write_driver(spec.write_driver, lib=lib, draw_pin_labels=False),
        ),
        "sense_amp": reuse(
            spec.sense_amp.cell_name,
            lambda: build_sense_amp_row(spec.sense_amp, lib=lib, draw_pin_labels=False),
        ),
        "output_latch": reuse(
            spec.output_latch.cell_name,
            lambda: build_output_latch(spec.output_latch, lib=lib, draw_pin_labels=False),
        ),
    }
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

    # ── Supplies ──────────────────────────────────────────────────────────────
    straps = spec.straps
    g1, g2 = spec.gaps
    _, x_o = spec.column_x
    rails = {
        # (rails, x where they end, which way they run into the gap)
        "E": (spec.leaf_rails("E"), g1, +1),
        "O_left": (spec.leaf_rails("O"), x_o, -1),
        "O_right": (spec.leaf_rails("O"), g2, +1),
        "logic": (
            [(spec.logic_y + k * spec.leaf_height, "VDD" if k % 2 else "VSS") for k in range(3)],
            spec.logic_x,
            -1,
        ),
    }
    gap_of = {"E": 0, "O_left": 0, "O_right": 1, "logic": 1}
    extensions: dict[str, list[tuple[float, float, float]]] = {"VSS": [], "VDD": []}
    vias: dict[float, list[float]] = {}
    for key, (found, x_edge, sign) in rails.items():
        for y, net in found:
            x_strap = straps[net][gap_of[key]]
            reach = x_strap + sign * (HALF + _RAIL_OVERHANG)
            x0, x1 = sorted((x_edge, reach))
            box(cell, "M1", x0, y - HALF, x1, y + HALF)
            extensions[net].append((x0, x1, y))
            # Two rails of the net this close are filled between (below) and
            # share the first one's via.
            if any(abs(y - other) < 4 * HALF for other in vias.setdefault(x_strap, [])):
                continue
            vias[x_strap].append(y)
            square(cell, "V1", x_strap, y)
            box(cell, "M2", x_strap - _PAD_HALF_X, y - HALF, x_strap + _PAD_HALF_X, y + HALF)
            square(cell, "V2", x_strap, y)
    # Rails of one net that end up within a space of each other are filled
    # between, where both run.
    for net, found in extensions.items():
        for i, (a0, a1, ya) in enumerate(found):
            for b0, b1, yb in found[i + 1 :]:
                lo, hi = max(a0, b0), min(a1, b1)
                if hi > lo and 0 < abs(ya - yb) < 4 * HALF:
                    box(cell, "M1", lo, min(ya, yb), hi, max(ya, yb))
    for net, xs in straps.items():
        for x in xs:
            m3_column(cell, x, 0, height)
    for net, y in spec.bridges.items():
        xa, xb = straps[net]
        box(cell, "M2", xa - _PAD_HALF_X, y - HALF, xb + _PAD_HALF_X, y + HALF)
        square(cell, "V2", xa, y)
        square(cell, "V2", xb, y)

    if draw_pin_labels:
        for pin, (metal, origin) in spec.pin_positions.items():
            label(cell, pin, metal, origin)
    box(cell, "BOUNDARY", 0, 0, spec.width, height)
    return cell


def main(argv: list[str] | None = None) -> None:
    """Write a staggered column-IO GDS: ``asap7-io-column-staggered --selects 4``."""
    args = parse_spec(
        StaggeredIoColumnSpec,
        argv,
        description="Generate one data bit's column IO for an array of 270 nm rows (ASAP7 6T).",
    )
    spec = args.spec
    cell, out = write_gds(lambda lib: build_staggered_io_column(spec, lib=lib), args.out)
    print(f"✓ {cell.name}: {spec.width} x {spec.height} nm -> {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
