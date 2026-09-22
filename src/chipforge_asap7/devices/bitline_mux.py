"""Parametric bitline leaf: precharge and a transmission-gate column mux, one per bitline pair.

This is the part of an SRAM's column IO that repeats at the *bitline* pitch,
and so the part a reused hard core gets wrong when the bitcell changes.  The
released ASAP7 bank draws it as ``sram_prech_ymux_6t112`` inside a 1080 nm core
whose four pairs sit on the 270 nm pitch of the 6T cell; under an 8T array,
whose rows are 594 nm, OpenFinRAM has to fan every pair in through a pitch
adapter and cannot change the mux ratio at all.  `BitlineMuxSpec` draws the
same six transistors on whatever row the array has::

    BL  ──┬── pFET(PRECHN) ── VDD          BLN ──┬── pFET(PRECHN) ── VDD
          └── TG(YSEL/YSELN) ── SA               └── TG(YSEL/YSELN) ── SAN

Two things decide the floorplan.  In the dense row style every poly stripe
runs through both bands, so an nFET and the pFET above it share a gate -- and
here nothing does: the transmission gate wants ``YSEL`` below and ``YSELN``
above, and the precharge pFETs have no nFET at all.  So n-only and p-only
devices get gate columns of their own, and the other band simply has no
diffusion there::

    column      xa   Ga   x0   G0   x1   G1   x2   G2   x3   G3   x4   Gb   xb
    p band            .   SA  YSELN BL  PRECHN VDD PRECHN BLN YSELN SAN   .
    n band      BL  YSEL  SA    .    .    .    .    .    .    .   SAN YSEL BLN

And what a *group* of these leaves shares has to run straight through them:
``SA``, ``SAN`` and ``PRECHN`` are full-height M3 tracks over their own
columns, and every select of the group -- this leaf's and its neighbours' --
passes either side on M3, so stacking `selects` leaves in Y makes a
`selects`-to-one mux with no routing of its own.  The bitlines cross on M2,
which is where an array delivers them.

The default row is 297 nm, half an 8T bitcell row (`RowStack` with
``band_height=(135, 162)``).  Two ports' leaves cannot share a strip -- their
M3 tracks would land on each other -- so each port gets a strip of its own, at
its own end of the bitlines, and the whole 594 nm of bitcell row per pair.
``rows=2`` spends it: the six transistors drawn twice, mirrored about the VDD
rail and wired in parallel.  Where the array delivers its bitlines is
`bitline_entry`: two M3 columns at the left edge take them from there, on M2
or M4, to the rows' own tracks.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

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
    SELECT_X_ENC,
    VT_LAYERS,
    build_device_band,
)
from .inverter import ACTIVE_ABUT_OVERHANG, M2_V1_ENCLOSURE
from .row import RowBand, RowStack

__all__ = [
    "BITLINE_MUX_PINS",
    "BitlineMuxSpec",
    "bitline_mux_group_pins",
    "build_bitline_mux",
    "build_bitline_mux_group",
]

BITLINE_MUX_PINS = ("BL", "BLN", "SA", "SAN", "PRECHN", "YSEL", "YSELN", "VDD", "VSS")

_HALF = CONTACT_SIZE // 2  # 9: half a via, half an 18 nm track
_CAP = _HALF + M1_V0_ENCLOSURE  # 14: metal past a via along its own track
_PAD = _HALF + M2_V1_ENCLOSURE  # 17: M2 past a V1
_TRACK = M1_WIDTH + M1_MIN_SPACE  # 36
_HALF_SD = SD_BAR_WIDTH // 2
_HALF_LIG = GATE_LIG_HEIGHT // 2
_HALF_CUT = DEVICE_GATE_CUT_HEIGHT // 2
_HALF_RAIL = LI_RAIL_HEIGHT // 2
_LIG_PAST_GATE = 1
_DEVICE_COLUMNS = 8  # gate pitches the six devices and their two end dummies take

_M1_PIN = (LAYERS["M1_PIN"]["layer"], LAYERS["M1_PIN"]["datatype"])
_M2_PIN = (LAYERS["M2_PIN"]["layer"], LAYERS["M2_PIN"]["datatype"])
_M3_PIN = (LAYERS["M3_PIN"]["layer"], LAYERS["M3_PIN"]["datatype"])
_M4_PIN = (LAYERS["M4_PIN"]["layer"], LAYERS["M4_PIN"]["datatype"])
_PIN_LAYERS = {"M1": _M1_PIN, "M2": _M2_PIN, "M3": _M3_PIN, "M4": _M4_PIN}

_M4_HALF = 12  # M4 is 24 nm wide, and a V3 is exactly as tall as its M4
_M4_PITCH = 48
_M4_X_GRID = 24  # the public deck holds M4 vertices to a 24 nm grid in x
_V3_M3_CAP = _M4_HALF + 5  # M3 past a V3, along the column
_V3_M4_CAP = _HALF + 11  # M4 past a V3, along the track


@dataclass(frozen=True)
class BitlineMuxSpec:
    """Precharge plus transmission-gate column select for one bitline pair.

    Args:
        n_fins: fins of the two select nFETs.
        p_fins: fins of the four pFETs -- two precharge, two select.  They
            share one band, so they share a fin count.
        selects: how many leaves share this one's ``SA``/``SAN``: the mux
            ratio.  Every select of the group runs through every leaf, so the
            leaf gets wider with it (36 nm of M3 a side per select).
        select: which of them this leaf answers to.
        band_height: ``(n_band, p_band)`` in nm.  The default, 135 + 162, is
            half a 594 nm 8T bitcell row, which no even split reaches on the
            27 nm fin grid.
        vt: threshold flavor of all six devices.
        rows: 1, or 2 for the six transistors drawn twice, the second copy
            mirrored about the VDD rail and in parallel with the first.  Twice
            the height, twice the fins, the same width but for the entry
            columns.
        bitline_entry: ``(y_BL, y_BLN)``, where the array delivers the pair at
            the leaf's left edge, from the leaf's bottom.  ``None`` means on
            the bottom row's own tracks.  Half nanometres are fine: the 8T
            cell's M2 bitlines sit on them.
        bitline_layer: the metal they arrive on.  M2 and M4 both run along the
            bitline; an 8T array has port A on one and port B on the other.
    """

    n_fins: int = 3
    p_fins: int = 3
    selects: int = 4
    select: int = 0
    band_height: tuple[int, int] = (135, 162)
    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"
    rows: int = 1
    bitline_entry: tuple[float, float] | None = None
    bitline_layer: Literal["M2", "M4"] = "M2"

    def __post_init__(self) -> None:
        for name in ("selects", "select", "rows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {value!r}")
        if self.selects < 1:
            raise ValueError(f"selects must be >= 1, got {self.selects}")
        if not 0 <= self.select < self.selects:
            raise ValueError(f"select must be 0..{self.selects - 1}, got {self.select}")
        if self.rows not in (1, 2):
            raise ValueError(f"rows must be 1 or 2, got {self.rows}")
        if self.bitline_layer not in ("M2", "M4"):
            raise ValueError(
                f"bitline_layer must be M2 or M4, got {self.bitline_layer!r}"
            )
        if self.bitline_layer == "M4" and self.bitline_entry is None:
            raise ValueError(
                "bitlines on M4 need a bitline_entry: the leaf has no M4 of its own"
            )
        object.__setattr__(self, "band_height", tuple(self.band_height))
        if self.bitline_entry is not None:
            entry = tuple(self.bitline_entry)
            if len(entry) != 2:
                raise ValueError(
                    f"bitline_entry is (y_BL, y_BLN), got {self.bitline_entry!r}"
                )
            object.__setattr__(self, "bitline_entry", entry)
        # RowStack and FinFETSpec reject what cannot be drawn.
        n_band, p_band = self.bands
        # Six M2 tracks have to fit between the two via rows; see `tracks_y`.
        if self.tracks_y["BLN"] + _HALF + M1_MIN_SPACE > self.tracks_y["YSEL"] - _HALF:
            raise ValueError(
                f"an n band of {n_band.height} nm leaves no room for both bitline "
                "tracks under the gate contacts; use at least 135 nm"
            )
        if self.tracks_y["YSELN"] + _HALF + M1_MIN_SPACE > p_band.contact_y - _PAD:
            raise ValueError(
                f"a p band of {p_band.height} nm leaves no room for the select tie "
                "under the via row; use at least 162 nm"
            )
        if self.bitline_entry is not None:
            self._check_entry()

    def _check_entry(self) -> None:
        """Reject an entry that would put two nets' metal within a space of each other."""
        ys = self.entry_y
        half = _HALF if self.bitline_layer == "M2" else _M4_HALF
        for net, y in ys.items():
            if not half + _HALF <= y <= self.height - half - _HALF:
                raise ValueError(
                    f"{net} enters at y={y}, outside the {self.height} nm leaf"
                )
        if self.bitline_layer == "M4":
            if abs(ys["BL"] - ys["BLN"]) < _M4_PITCH:
                raise ValueError(
                    f"M4 bitlines need {_M4_PITCH} nm between them, got {ys}"
                )
            return
        # On M2 the entry shares the left margin with the other bitline's
        # tracks and landings, and with the YSEL tie of the group's last select.
        ties = [self.tracks_y["YSEL"]]
        if self.rows == 2:
            ties.append(self.height - ties[0])
        for net, other in (("BL", "BLN"), ("BLN", "BL")):
            y = ys[net]
            for y_other in (*self.track_ys(other), ys[other], *ties):
                if abs(y - y_other) < _TRACK:
                    raise ValueError(
                        f"{net} entering on M2 at y={y} is within {_TRACK} nm of other "
                        f"M2 at y={y_other}"
                    )
            for y_own in self.track_ys(net):
                if 0 < abs(y - y_own) < _TRACK:
                    raise ValueError(
                        f"{net} entering on M2 at y={y} crowds its own track at y={y_own}; "
                        "enter on the track or a full pitch from it"
                    )

    # ── Rows ──────────────────────────────────────────────────────────────────
    @property
    def row_stack(self) -> RowStack:
        """The row the six transistors are drawn in."""
        return RowStack(
            rows=((self.n_fins, self.p_fins),), vt=self.vt, band_height=self.band_height
        )

    @property
    def stack(self) -> RowStack:
        """The whole leaf: what a tap or a filler beside it has to match."""
        return replace(self.row_stack, rows=((self.n_fins, self.p_fins),) * self.rows)

    @property
    def bands(self) -> tuple[RowBand, RowBand]:
        """The n and the p band of the bottom row."""
        n_band, p_band = self.row_stack.bands()
        return n_band, p_band

    @property
    def seam_y(self) -> int:
        return self.row_stack.seam_y(0)

    @property
    def row_height(self) -> int:
        return self.row_stack.height

    @property
    def height(self) -> int:
        return self.rows * self.row_height

    @property
    def rails(self) -> tuple[tuple[int, str], ...]:
        return self.stack.rails

    @property
    def cell_name(self) -> str:
        return f"{self._size_tag}_s{self.select}of{self.selects}{self._variant_tag}"

    @property
    def _size_tag(self) -> str:
        rows_tag = "" if self.rows == 1 else f"_r{self.rows}"
        return (
            f"blmux_{self.n_fins}n{self.p_fins}p"
            f"_h{self.band_height[0]}x{self.band_height[1]}{rows_tag}"
        )

    @property
    def _variant_tag(self) -> str:
        tag = "" if self.vt == "rvt" else f"_{self.vt}"
        if self.bitline_entry is not None:
            # Tenths of a nanometre, so that 245.5 stays a legal cell name.
            y_bl, y_bln = (round(10 * y) for y in self.bitline_entry)
            tag += f"_{self.bitline_layer.lower()}in{y_bl}x{y_bln}"
        return tag

    # ── Columns ───────────────────────────────────────────────────────────────
    @property
    def has_entry_columns(self) -> bool:
        """Whether the bitlines take an M3 column each on their way in.

        Two rows always do, to reach the upper copy; one row does when the
        bitlines arrive anywhere but on its own tracks.
        """
        return self.rows == 2 or self.bitline_entry is not None

    @property
    def columns_left(self) -> int:
        """Empty gate pitches left of the devices: ``YSEL`` tracks, entry columns."""
        tracks = self.selects + (2 if self.has_entry_columns else 0)
        return max(0, math.ceil((_TRACK * tracks - 81) / GATE_PITCH))

    @property
    def columns_right(self) -> int:
        """Empty gate pitches right of the devices, for the ``YSELN`` tracks."""
        return max(0, math.ceil((_TRACK * self.selects - 27) / GATE_PITCH))

    @property
    def width(self) -> int:
        return (_DEVICE_COLUMNS + self.columns_left + self.columns_right) * GATE_PITCH

    @property
    def gate_grid_xs(self) -> list[int]:
        return list(range(GATE_PITCH // 2, self.width, GATE_PITCH))

    @property
    def sd_x(self) -> dict[str, int]:
        """Source/drain columns by name; see the module docstring's table."""
        first = (1 + self.columns_left) * GATE_PITCH
        names = ("xa", "x0", "x1", "x2", "x3", "x4", "xb")
        return {name: first + i * GATE_PITCH for i, name in enumerate(names)}

    @property
    def gate_x(self) -> dict[str, int]:
        x = self.sd_x
        half = GATE_PITCH // 2
        return {"Ga": x["xa"] + half, "G0": x["x0"] + half, "G1": x["x1"] + half,
                "G2": x["x2"] + half, "G3": x["x3"] + half, "Gb": x["x4"] + half}  # fmt: skip

    @property
    def track_x(self) -> dict[str, int]:
        """X of every full-height M3 track: the shared nets, and every select of the group."""
        x = self.sd_x
        tracks = {"SA": x["x0"], "SAN": x["x4"], "PRECHN": x["x2"]}
        for j in range(self.selects):
            tracks[f"YSEL[{j}]"] = x["x0"] - _TRACK * (j + 1)
            tracks[f"YSELN[{j}]"] = x["xb"] + _TRACK * (j + 1)
        return tracks

    @property
    def entry_x(self) -> dict[str, int]:
        """X of the two entry columns, outside every select track; ``BL`` outermost."""
        x0 = self.sd_x["x0"]
        return {
            "BL": x0 - _TRACK * (self.selects + 2),
            "BLN": x0 - _TRACK * (self.selects + 1),
        }

    @property
    def entry_y(self) -> dict[str, float]:
        """Y at which each bitline crosses the left edge."""
        if self.bitline_entry is None:
            return {net: self.tracks_y[net] for net in ("BL", "BLN")}
        return dict(zip(("BL", "BLN"), self.bitline_entry, strict=True))

    def track_ys(self, net: str) -> tuple[int, ...]:
        """Y of `net`'s M2 track in every row; the upper row is the lower one mirrored."""
        y = self.tracks_y[net]
        return (y,) if self.rows == 1 else (y, self.height - y)

    @property
    def tracks_y(self) -> dict[str, int]:
        """Y of the horizontal M2 tracks, bottom to top.

        Below the seam, over the n band: the two bitlines, one M2 pitch apart,
        the lower one a pitch above the via row's landing pads.  Above it: the
        ``YSEL`` tie on the gate contacts and the ``YSELN`` tie a pitch higher.
        """
        n_band, _ = self.bands
        # 80 on the default row.
        bl = int(n_band.contact_y) + _PAD + M1_MIN_SPACE + _HALF
        # 152: its V1 sits on the gate contact's M1 bar, above the LIG.
        ysel = self.seam_y + _PAD
        return {"BL": bl, "BLN": bl + _TRACK, "YSEL": ysel, "YSELN": ysel + _TRACK}

    @property
    def pin_positions(self) -> dict[str, tuple[str, tuple[float, float]]]:
        """``pin -> (metal, (x, y))``.  Bitlines enter at the left edge; rails are named once."""
        tx, entry = self.track_x, self.entry_y
        mid = self.height / 2
        rails = {net: y for y, net in reversed(self.rails)}
        return {
            "BL": (self.bitline_layer, (_PAD, entry["BL"])),
            "BLN": (self.bitline_layer, (_PAD, entry["BLN"])),
            "SA": ("M3", (tx["SA"], mid)),
            "SAN": ("M3", (tx["SAN"], mid)),
            "PRECHN": ("M3", (tx["PRECHN"], mid)),
            "YSEL": ("M3", (tx[f"YSEL[{self.select}]"], mid)),
            "YSELN": ("M3", (tx[f"YSELN[{self.select}]"], mid)),
            "VDD": ("M1", (self.width / 2, rails["VDD"])),
            "VSS": ("M1", (self.width / 2, rails["VSS"])),
        }

    # ── Netlist ───────────────────────────────────────────────────────────────
    @property
    def devices(self) -> tuple[tuple[str, str, str, str, str, int], ...]:
        """``(name, drain, gate, source, flavor, fins)`` for the six transistors."""
        n, p = self.rows * self.n_fins, self.rows * self.p_fins
        return (
            ("MNT", "BL", "YSEL", "SA", "n", n),
            ("MNC", "BLN", "YSEL", "SAN", "n", n),
            ("MPT", "SA", "YSELN", "BL", "p", p),
            ("MPC", "SAN", "YSELN", "BLN", "p", p),
            ("MPPT", "BL", "PRECHN", "VDD", "p", p),
            ("MPPC", "BLN", "PRECHN", "VDD", "p", p),
        )

    def netlist(self, name: str | None = None) -> str:
        """A BSIM-CMG subcircuit sized by ``nfin``."""
        title = name or self.cell_name
        n_band, p_band = self.bands
        lines = [
            "* ASAP7 bitline leaf: precharge and transmission-gate column select",
            f".SUBCKT {title} {' '.join(BITLINE_MUX_PINS)}",
        ]
        for device, drain, gate, source, flavor, fins in self.devices:
            band = n_band if flavor == "n" else p_band
            bulk = "VSS" if flavor == "n" else "VDD"
            lines.append(
                f"{device} {drain} {gate} {source} {bulk} {band.spec.model} "
                f"nfin={fins} l={band.spec.gate_length}n nf=1 m=1"
            )
        lines += [f".ENDS {title}", ".END", ""]
        return "\n".join(lines)


# ── Layout ────────────────────────────────────────────────────────────────────
def _square(cell: Any, layer: str, x: float, y: float) -> None:
    box(cell, layer, x - _HALF, y - _HALF, x + _HALF, y + _HALF)


def _stack_to_m3(cell: Any, x: float, y: float) -> None:
    """V1, an M2 landing turned *along the column*, and V2, on an M1 pad at (x, y).

    Neighbouring columns are 54 nm apart.  Two M2 landings lying along the
    track would face each other tip to tip at 20 nm, where two short edges need
    31; standing up, they face side to side at 36.
    """
    _square(cell, "V1", x, y)
    box(cell, "M2", x - _HALF, y - _PAD, x + _HALF, y + _PAD)
    _square(cell, "V2", x, y)


def _new_cell(name: str, lib: Any) -> Any:
    if lib is not None:
        return lib.new_cell(name)
    return require_gdspy().Cell(name, exclude_from_current=True)


def _draw_row(cell: Any, spec: BitlineMuxSpec) -> None:
    """The six transistors and their wiring, in the bottom row of `spec`."""
    n_band, p_band = spec.bands
    width, height, seam = spec.width, spec.row_height, spec.seam_y
    x, g, tx, ty = spec.sd_x, spec.gate_x, spec.track_x, spec.tracks_y
    half_gate = GATE_WIDTH // 2

    # Implant and well tile the row band by band, the whole width of the cell.
    for band in (n_band, p_band):
        box(cell, band.implant, 0, band.y0, width, band.y1)
        if band.in_nwell:
            box(cell, "NWELL", 0, band.y0, width, band.y1)
    if vt_layer := VT_LAYERS[spec.vt]:
        box(cell, vt_layer, 0, 0, width, height)

    # FIN and GATE are manufacturing grids; every stripe is one gate through
    # both bands, cut on the rails.  Which band has diffusion under it is what
    # makes it an nFET's gate, a pFET's, or nobody's.
    for y_fin in spec.row_stack.fin_grid_ys:
        box(cell, "FIN", 0, y_fin, width, y_fin + FIN_WIDTH)
    for x_gate in spec.gate_grid_xs:
        box(
            cell,
            "GATE",
            x_gate - half_gate,
            -POLY_OVERHANG,
            x_gate + half_gate,
            height + POLY_OVERHANG,
        )
    for y_rail, _ in spec.row_stack.rails:
        box(cell, "GATE_CUT", 0, y_rail - _HALF_CUT, width, y_rail + _HALF_CUT)

    # Diffusion.  One p island under the four pFETs; two n islands, one under
    # each select nFET, 200 nm apart with the pFETs' gates passing between them.
    over = ACTIVE_ABUT_OVERHANG
    islands = (
        (p_band, ("x0", "x1", "x2", "x3", "x4")),
        (n_band, ("xa", "x0")),
        (n_band, ("x4", "xb")),
    )
    for band, columns in islands:
        xs = [x[column] for column in columns]
        build_device_band(
            cell,
            band.spec,
            y0=band.y0,
            sd_xs=xs,
            active_x=(xs[0] - over, xs[-1] + over),
        )
    # The implant encloses the outermost diffusion.
    assert x["xa"] - over >= SELECT_X_ENC

    # Rails.  The precharge source is the only supply contact: its LISD carries
    # on to the VDD rail and the V0 sits on the rail, over an LI rail.
    for y_rail, _ in spec.row_stack.rails:
        box(cell, "LIG", 0, y_rail - _HALF_RAIL, width, y_rail + _HALF_RAIL)
        box(cell, "M1", 0, y_rail - _HALF, width, y_rail + _HALF)
    p_lo, _ = p_band.active_span
    box(cell, "LISD", x["x2"] - _HALF_SD, p_lo, x["x2"] + _HALF_SD, p_band.rail_y)
    _square(cell, "V0", x["x2"], p_band.rail_y)

    # Every other diffusion column: a V0 and an M1 pad on its band's via row.
    # The two bitline columns of the n band carry their pad up to the BL track.
    y_n, y_p = int(n_band.contact_y), int(p_band.contact_y)
    for column in ("x0", "x1", "x3", "x4"):
        _square(cell, "V0", x[column], y_p)
        box(cell, "M1", x[column] - _HALF, y_p - _CAP, x[column] + _HALF, y_p + _CAP)
    for column in ("x0", "x4"):
        _square(cell, "V0", x[column], y_n)
        box(cell, "M1", x[column] - _HALF, y_n - _CAP, x[column] + _HALF, y_n + _CAP)
    for column in ("xa", "xb"):
        _square(cell, "V0", x[column], y_n)
        box(
            cell,
            "M1",
            x[column] - _HALF,
            y_n - _CAP,
            x[column] + _HALF,
            ty["BL"] + _CAP,
        )
        _square(cell, "V1", x[column], ty["BL"])

    # Shared nets, straight through on M3.  SA and SAN tap both bands of their
    # own column; PRECHN taps its gate contact (below).
    for net in ("SA", "SAN"):
        box(cell, "M3", tx[net] - _HALF, 0, tx[net] + _HALF, height)
        for y_via in (y_n, y_p):
            _stack_to_m3(cell, tx[net], y_via)

    # Bitlines.  BL comes in on its track, meets the n column it passes first,
    # and rides a short M3 jumper over the seam to its p column.  BLN crosses
    # the whole cell one track higher: its n column is on the far side.  Each
    # track starts at the cell edge, or at its entry column if there is one.
    start = {net: 0 for net in ("BL", "BLN")}
    if spec.has_entry_columns:
        start = {net: x_entry - _PAD for net, x_entry in spec.entry_x.items()}
    box(cell, "M2", start["BL"], ty["BL"] - _HALF, x["x1"] + _PAD, ty["BL"] + _HALF)
    box(cell, "M2", start["BLN"], ty["BLN"] - _HALF, x["xb"] + _PAD, ty["BLN"] + _HALF)
    box(cell, "M2", x["xb"] - _PAD, ty["BL"] - _HALF, x["xb"] + _PAD, ty["BL"] + _HALF)
    for column, track in (("x1", "BL"), ("x3", "BLN")):
        _square(cell, "V2", x[column], ty[track])
        box(
            cell,
            "M3",
            x[column] - _HALF,
            ty[track] - _CAP,
            x[column] + _HALF,
            y_p + _CAP,
        )
        _stack_to_m3(cell, x[column], y_p)
    # BLN's n column, up from the BL track's height to its own.
    for y_via in (ty["BL"], ty["BLN"]):
        _square(cell, "V2", x["xb"], y_via)
    box(cell, "M3", x["xb"] - _HALF, ty["BL"] - _CAP, x["xb"] + _HALF, ty["BLN"] + _CAP)

    # Gate contacts, on the seam.  One LIG pad per run of same-net gates.
    def pad(first: str, last: str) -> None:
        box(cell, "LIG", g[first] - half_gate - _LIG_PAST_GATE, seam - _HALF_LIG,
            g[last] + half_gate + _LIG_PAST_GATE, seam + _HALF_LIG)  # fmt: skip

    # PRECHN: one pad under both precharge gates, contacted between them, where
    # the n band has nothing; the M1 bar drops to the n via row to meet its M3.
    pad("G1", "G2")
    _square(cell, "V0", x["x2"], seam)
    box(cell, "M1", x["x2"] - _HALF, y_n - _CAP, x["x2"] + _HALF, seam + _CAP)
    box(cell, "M3", tx["PRECHN"] - _HALF, 0, tx["PRECHN"] + _HALF, height)
    _stack_to_m3(cell, x["x2"], y_n)

    # YSEL on the two outer gates, YSELN on the two inner ones.  Each is tied on
    # its own M2 track and taps its own select among the group's M3 tracks.
    for role, gates in (("YSEL", ("Ga", "Gb")), ("YSELN", ("G0", "G3"))):
        y_tie = ty[role]
        track = tx[f"{role}[{spec.select}]"]
        for gate in gates:
            pad(gate, gate)
            _square(cell, "V0", g[gate], seam)
            box(cell, "M1", g[gate] - _HALF, seam - _CAP, g[gate] + _HALF, y_tie + _CAP)
            _square(cell, "V1", g[gate], y_tie)
        reach = [g[gates[0]], g[gates[1]], track]
        box(
            cell,
            "M2",
            min(reach) - _PAD,
            y_tie - _HALF,
            max(reach) + _PAD,
            y_tie + _HALF,
        )
        _square(cell, "V2", track, y_tie)
    for j in range(spec.selects):
        for role in ("YSEL", "YSELN"):
            track = tx[f"{role}[{j}]"]
            box(cell, "M3", track - _HALF, 0, track + _HALF, height)


def _draw_entry(cell: Any, spec: BitlineMuxSpec) -> None:
    """Each bitline from the left edge, along an M3 column, to its track in every row."""
    for net, x_entry in spec.entry_x.items():
        y_in = spec.entry_y[net]
        via_ys = set(spec.track_ys(net))
        if spec.bitline_layer == "M2":
            via_ys.add(y_in)
            box(cell, "M2", 0, y_in - _HALF, x_entry + _PAD, y_in + _HALF)
            ends = [(y - _CAP, y + _CAP) for y in via_ys]
        else:
            reach = x_entry + _V3_M4_CAP
            reach += -reach % _M4_X_GRID
            box(cell, "M4", 0, y_in - _M4_HALF, reach, y_in + _M4_HALF)
            box(
                cell,
                "V3",
                x_entry - _HALF,
                y_in - _M4_HALF,
                x_entry + _HALF,
                y_in + _M4_HALF,
            )
            ends = [(y - _CAP, y + _CAP) for y in via_ys]
            ends.append((y_in - _V3_M3_CAP, y_in + _V3_M3_CAP))
        for y_via in via_ys:
            _square(cell, "V2", x_entry, y_via)
        box(cell, "M3", x_entry - _HALF, min(lo for lo, _ in ends),
            x_entry + _HALF, max(hi for _, hi in ends))  # fmt: skip


def build_bitline_mux(
    spec: BitlineMuxSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Draw `spec` and return the gdspy Cell (nanometre coordinates).

    One row is drawn flat.  Two rows are one row cell placed twice, the second
    time mirrored about the VDD rail: everything that runs the height of a row
    on M3 meets itself there, which is all the wiring the copy needs apart from
    its bitlines.
    """
    spec = spec or BitlineMuxSpec()
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = _new_cell(cell_name, lib)
    if spec.rows == 1:
        _draw_row(cell, spec)
    else:
        row = _new_cell(f"{cell_name}__row", lib)
        _draw_row(row, spec)
        cell.add(gdspy.CellReference(row))
        cell.add(gdspy.CellReference(row, origin=(0, spec.height), x_reflection=True))
    if spec.has_entry_columns:
        _draw_entry(cell, spec)

    if draw_pin_labels:
        for pin, (metal, origin) in spec.pin_positions.items():
            if pin not in ("VDD", "VSS"):
                layer, texttype = _PIN_LAYERS[metal]
                cell.add(gdspy.Label(pin, origin, layer=layer, texttype=texttype))
        # Every rail is its own conductor until a power grid joins them.
        for y_rail, net in spec.rails:
            cell.add(
                gdspy.Label(
                    net, (spec.width / 2, y_rail), layer=_M1_PIN[0], texttype=_M1_PIN[1]
                )
            )
    box(cell, "BOUNDARY", 0, 0, spec.width, spec.height)
    return cell


def bitline_mux_group_pins(spec: BitlineMuxSpec) -> tuple[str, ...]:
    """Pins of a `spec.selects`-to-one group: per-leaf bitlines and selects, shared everything else."""
    per_leaf = [
        f"{pin}[{i}]"
        for i in range(spec.selects)
        for pin in ("BL", "BLN", "YSEL", "YSELN")
    ]
    return (*per_leaf, "SA", "SAN", "PRECHN", "VDD", "VSS")


def build_bitline_mux_group(
    spec: BitlineMuxSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Stack `spec.selects` leaves into one column mux; return the gdspy Cell.

    Leaf ``i`` answers to select ``i`` and sits at ``y = i * height``, every
    other one flipped, as standard-cell rows are so that neighbours share a
    rail -- and as an array's rows are, so a flipped leaf finds its bitlines
    where its `bitline_entry`, flipped, says.  Nothing is routed here: ``SA``,
    ``SAN``, ``PRECHN`` and the selects are the leaves' own M3 tracks meeting
    end to end.  `spec.select` is ignored.
    """
    spec = spec or BitlineMuxSpec()
    gdspy = require_gdspy()
    cell_name = name or f"{spec._size_tag}_group{spec.selects}{spec._variant_tag}"
    cell = _new_cell(cell_name, lib)
    height, tx, entry = spec.height, spec.track_x, spec.entry_y
    layers = _PIN_LAYERS

    def label(text: str, metal: str, origin: tuple[float, float]) -> None:
        if draw_pin_labels:
            cell.add(
                gdspy.Label(
                    text, origin, layer=layers[metal][0], texttype=layers[metal][1]
                )
            )

    for i in range(spec.selects):
        leaf_spec = replace(spec, select=i)
        leaf = build_bitline_mux(
            leaf_spec, name=f"{cell_name}__leaf{i}", lib=lib, draw_pin_labels=False
        )
        flipped = bool(i % 2)
        y0 = (i + 1) * height if flipped else i * height
        cell.add(gdspy.CellReference(leaf, origin=(0, y0), x_reflection=flipped))
        for pin in ("BL", "BLN"):
            y_pin = y0 - entry[pin] if flipped else y0 + entry[pin]
            label(f"{pin}[{i}]", spec.bitline_layer, (_PAD, y_pin))
        for pin in ("YSEL", "YSELN"):
            label(f"{pin}[{i}]", "M3", (tx[f"{pin}[{i}]"], i * height + height / 2))
    for pin in ("SA", "SAN", "PRECHN"):
        label(pin, "M3", (tx[pin], height / 2))
    # Every rail is its own conductor until a power grid joins them: name them all.
    for k in range(spec.rows * spec.selects + 1):
        label("VDD" if k % 2 else "VSS", "M1", (spec.width / 2, k * spec.row_height))
    box(cell, "BOUNDARY", 0, 0, spec.width, spec.selects * height)
    return cell


def main(argv: list[str] | None = None) -> None:
    """Write a bitline-leaf GDS: ``python -m chipforge_asap7.devices.bitline_mux``."""
    parser = argparse.ArgumentParser(
        description="Generate an ASAP7 precharge + column-mux leaf for one bitline pair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-fins", type=int, default=3)
    parser.add_argument("--p-fins", type=int, default=3)
    parser.add_argument(
        "--selects", type=int, default=4, help="Mux ratio: leaves sharing one SA/SAN."
    )
    parser.add_argument(
        "--select", type=int, default=0, help="Which select this leaf answers to."
    )
    parser.add_argument(
        "--band-height", default="135,162", help="n,p band heights in nm."
    )
    parser.add_argument("--vt", choices=("rvt", "lvt", "slvt", "sram"), default="rvt")
    parser.add_argument(
        "--rows", type=int, choices=(1, 2), default=1, help="2 doubles the devices."
    )
    parser.add_argument(
        "--bitline-entry", default=None, help="y_BL,y_BLN at the left edge, in nm."
    )
    parser.add_argument("--bitline-layer", choices=("M2", "M4"), default="M2")
    parser.add_argument(
        "--group", action="store_true", help="Draw the whole stacked mux, not one leaf."
    )
    parser.add_argument("--out", type=Path, default=None, help="Output GDS path.")
    args = parser.parse_args(argv)
    entry = args.bitline_entry
    spec = BitlineMuxSpec(
        n_fins=args.n_fins, p_fins=args.p_fins, selects=args.selects, select=args.select,
        band_height=tuple(int(v) for v in args.band_height.split(",")), vt=args.vt,
        rows=args.rows, bitline_layer=args.bitline_layer,
        bitline_entry=entry and tuple(float(v) for v in entry.split(",")),
    )  # fmt: skip
    library = require_gdspy().GdsLibrary(unit=1e-9, precision=1e-10)
    require_gdspy().current_library = library
    cell = (build_bitline_mux_group if args.group else build_bitline_mux)(
        spec, lib=library
    )
    output = args.out or Path(f"{cell.name}.gds")
    output.parent.mkdir(parents=True, exist_ok=True)
    library.write_gds(str(output))
    leaves = spec.selects if args.group else 1
    print(f"✓ {cell.name}: {spec.width} x {leaves * spec.height} nm -> {output}")


if __name__ == "__main__":  # pragma: no cover
    main()
