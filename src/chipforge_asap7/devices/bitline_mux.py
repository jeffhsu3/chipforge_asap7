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

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

from ..layout.grid import GATE_PITCH
from ..layout.layers import box, require_gdspy
from ..layout.rules import M1_MIN_SPACE, M4_PITCH, M4_X_GRID, TRACK_PITCH
from .cli import Option, parse_spec, write_gds
from .finfet import SELECT_X_ENC
from .row import RowBand, RowStack
from .rowcell import (
    RUNSET_RAIL_SPACE,
    CAP,
    HALF,
    ISLAND_OVERHANG,
    M4_HALF,
    PAD,
    V3_M3_CAP,
    V3_M4_CAP,
    draw_frame,
    draw_rails,
    gate_contact,
    island,
    label,
    landing,
    m2_track,
    m3_column,
    sd_contact,
    square,
    stack_to_m3,
    supply_contact,
    via_y,
)

__all__ = [
    "BITLINE_MUX_PINS",
    "BitlineMuxSpec",
    "bitline_mux_group_pins",
    "build_bitline_mux",
    "build_bitline_mux_group",
]

#: How far a group's sense lines stop short of its outer ends: half of M3's
#: tip-to-tip space for two short edges (M3.S.4-5, 31 nm), rounded up, so
#: two groups stacked end to end keep their sense lines apart.
SENSE_END_CLEARANCE = 16

BITLINE_MUX_PINS = ("BL", "BLN", "SA", "SAN", "PRECHN", "YSEL", "YSELN", "VDD", "VSS")

_DEVICE_COLUMNS = 8  # gate pitches the six devices and their two end dummies take


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
        grid_offset: nm the leaf's bottom sits above the bottom of the array
            row that delivers its bitlines, ``bitline_entry`` staying measured
            from the row.  A leaf centres a fin *space* on its edge, as ASAP7's
            standard cells do; an SRAM bitcell centres a *fin* there.  Placed
            ``13.5`` (half the fin pitch) up, the leaf's fins fall on the
            array's grid and the two can abut, which is how the released ASAP7
            bank places its periphery.  In a group the flipped leaves sit that
            far *down* from their rows, so their landings move the other way;
            `for_row` draws each leaf accordingly.
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
    grid_offset: float = 0.0

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
        elif self.grid_offset:
            raise ValueError(
                "grid_offset places the array's bitlines, so it needs a bitline_entry"
            )
        # RowStack and FinFETSpec reject what cannot be drawn.
        n_band, p_band = self.bands
        # Six M2 tracks have to fit between the two via rows; see `tracks_y`.
        if self.tracks_y["BLN"] + HALF + M1_MIN_SPACE > self.tracks_y["YSEL"] - HALF:
            raise ValueError(
                f"an n band of {n_band.height} nm leaves no room for both bitline "
                "tracks under the gate contacts; use at least 135 nm"
            )
        if self.tracks_y["YSELN"] + HALF + M1_MIN_SPACE > via_y(p_band, RUNSET_RAIL_SPACE) - PAD:
            raise ValueError(
                f"a p band of {p_band.height} nm leaves no room for the select tie "
                "under the via row; use at least 162 nm"
            )
        if self.bitline_entry is not None:
            self._check_entry()

    def _check_entry(self) -> None:
        """Reject an entry that would put two nets' metal within a space of each other."""
        ys = self.entry_y
        half = HALF if self.bitline_layer == "M2" else M4_HALF
        for net, y in ys.items():
            if not half + HALF <= y <= self.height - half - HALF:
                raise ValueError(
                    f"{net} enters at y={y}, outside the {self.height} nm leaf"
                )
        if self.bitline_layer == "M4":
            if abs(ys["BL"] - ys["BLN"]) < M4_PITCH:
                raise ValueError(
                    f"M4 bitlines need {M4_PITCH} nm between them, got {ys}"
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
                if abs(y - y_other) < TRACK_PITCH:
                    raise ValueError(
                        f"{net} entering on M2 at y={y} is within {TRACK_PITCH} nm of other "
                        f"M2 at y={y_other}"
                    )
            for y_own in self.track_ys(net):
                if 0 < abs(y - y_own) < TRACK_PITCH:
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
            if self.grid_offset:
                tag += f"g{round(10 * self.grid_offset)}"
        return tag

    def for_row(self, index: int) -> BitlineMuxSpec:
        """The leaf drawn for row `index` of a group: its select, and its entry on its own edge.

        Odd rows are flipped, so with a `grid_offset` their bitlines cross
        their edge as far *above* the entry as the even rows' cross below it.
        The result has no offset of its own: it is what is drawn.
        """
        spec = replace(self, select=index)
        if not self.grid_offset:
            return spec
        sign = 1 if index % 2 else -1
        entry = tuple(y + sign * self.grid_offset for y in self.bitline_entry)
        return replace(spec, bitline_entry=entry, grid_offset=0.0)

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
        return max(0, math.ceil((TRACK_PITCH * tracks - 81) / GATE_PITCH))

    @property
    def columns_right(self) -> int:
        """Empty gate pitches right of the devices, for the ``YSELN`` tracks."""
        return max(0, math.ceil((TRACK_PITCH * self.selects - 27) / GATE_PITCH))

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
            tracks[f"YSEL[{j}]"] = x["x0"] - TRACK_PITCH * (j + 1)
            tracks[f"YSELN[{j}]"] = x["xb"] + TRACK_PITCH * (j + 1)
        return tracks

    @property
    def entry_x(self) -> dict[str, int]:
        """X of the two entry columns, outside every select track; ``BL`` outermost."""
        x0 = self.sd_x["x0"]
        return {
            "BL": x0 - TRACK_PITCH * (self.selects + 2),
            "BLN": x0 - TRACK_PITCH * (self.selects + 1),
        }

    @property
    def entry_y(self) -> dict[str, float]:
        """Y at which each bitline crosses the left edge of this leaf as drawn (unflipped)."""
        if self.bitline_entry is None:
            return {net: self.tracks_y[net] for net in ("BL", "BLN")}
        return {
            net: y - self.grid_offset
            for net, y in zip(("BL", "BLN"), self.bitline_entry, strict=True)
        }

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
        # The landing on the via row lies along the track, 18 nm tall.
        bl = via_y(n_band, RUNSET_RAIL_SPACE) + HALF + M1_MIN_SPACE + HALF + 3
        # 152: its V1 sits on the gate contact's M1 bar, above the LIG.
        ysel = self.seam_y + PAD
        return {
            "BL": bl,
            "BLN": bl + TRACK_PITCH,
            "YSEL": ysel,
            "YSELN": ysel + TRACK_PITCH,
        }

    @property
    def pin_positions(self) -> dict[str, tuple[str, tuple[float, float]]]:
        """``pin -> (metal, (x, y))``.  Bitlines enter at the left edge; rails are named once."""
        tx, entry = self.track_x, self.entry_y
        mid = self.height / 2
        rails = {net: y for y, net in reversed(self.rails)}
        return {
            "BL": (self.bitline_layer, (PAD, entry["BL"])),
            "BLN": (self.bitline_layer, (PAD, entry["BLN"])),
            "SA": ("M3", (tx["SA"], mid)),
            "SAN": ("M3", (tx["SAN"], mid)),
            "PRECHN": ("M3", (tx["PRECHN"], mid)),
            "YSEL": ("M3", (tx[f"YSEL[{self.select}]"], mid)),
            "YSELN": ("M3", (tx[f"YSELN[{self.select}]"], mid)),
            "VDD": ("M1", (self.width / 2, rails["VDD"])),
            "VSS": ("M1", (self.width / 2, rails["VSS"])),
        }

    @property
    def group_cell_name(self) -> str:
        return f"{self._size_tag}_group{self.selects}{self._variant_tag}"

    def group_pin_positions(self) -> dict[str, tuple[str, tuple[float, float]]]:
        """``pin -> (metal, (x, y))`` of a `selects`-to-one group of this leaf.

        Bitlines enter at the left edge of each leaf, the flipped ones' at
        their entry mirrored; selects are on their M3 tracks; the shared nets
        once; and every rail once, since each is its own conductor until a
        power grid joins them.
        """
        height, tx = self.height, self.track_x
        pins: dict[str, tuple[str, tuple[float, float]]] = {}
        for i in range(self.selects):
            y0 = (i + 1) * height if i % 2 else i * height
            entry = self.for_row(i).entry_y
            for pin in ("BL", "BLN"):
                y_pin = y0 - entry[pin] if i % 2 else y0 + entry[pin]
                pins[f"{pin}[{i}]"] = (self.bitline_layer, (PAD, y_pin))
            for pin in ("YSEL", "YSELN"):
                pins[f"{pin}[{i}]"] = (
                    "M3",
                    (tx[f"{pin}[{i}]"], i * height + height / 2),
                )
        for pin in ("SA", "SAN", "PRECHN"):
            pins[pin] = ("M3", (tx[pin], height / 2))
        for k in range(self.rows * self.selects + 1):
            pins[f"{'VDD' if k % 2 else 'VSS'}{'' if k < 2 else f'.{k}'}"] = (
                "M1",
                (self.width / 2, k * self.row_height),
            )
        return pins

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
def _new_cell(name: str, lib: Any) -> Any:
    if lib is not None:
        return lib.new_cell(name)
    return require_gdspy().Cell(name, exclude_from_current=True)


def _draw_row(cell: Any, spec: BitlineMuxSpec) -> None:
    """The six transistors and their wiring, in the bottom row of `spec`."""
    n_band, p_band = spec.bands
    width, height, seam = spec.width, spec.row_height, spec.seam_y
    x, g, tx, ty = spec.sd_x, spec.gate_x, spec.track_x, spec.tracks_y

    draw_frame(cell, spec.row_stack, width, spec.vt)

    # Diffusion.  One p island under the four pFETs; two n islands, one under
    # each select nFET, 200 nm apart with the pFETs' gates passing between them.
    island(cell, p_band, [x[c] for c in ("x0", "x1", "x2", "x3", "x4")])
    island(cell, n_band, [x["xa"], x["x0"]])
    island(cell, n_band, [x["x4"], x["xb"]])
    # The implant encloses the outermost diffusion.
    assert x["xa"] - ISLAND_OVERHANG >= SELECT_X_ENC

    # Rails.  The precharge source is the only supply contact.
    draw_rails(cell, spec.row_stack, width)
    supply_contact(cell, p_band, x["x2"])

    # Every other diffusion column: a V0 and an M1 pad on its band's via row.
    # The two bitline columns of the n band carry their pad up to the BL track.
    for column in ("x0", "x1", "x3", "x4"):
        y_p = sd_contact(cell, p_band, x[column], rail_space=RUNSET_RAIL_SPACE)
    for column in ("x0", "x4"):
        y_n = sd_contact(cell, n_band, x[column], rail_space=RUNSET_RAIL_SPACE)
    for column in ("xa", "xb"):
        sd_contact(cell, n_band, x[column], reach=ty["BL"] + CAP, rail_space=RUNSET_RAIL_SPACE)
        square(cell, "V1", x[column], ty["BL"])

    # Shared nets, straight through on M3.  SA and SAN tap both bands of their
    # own column (their tracks are drawn by the leaf, which knows whether its
    # ends are a group's outer ends); PRECHN taps its gate contact (below).
    for net in ("SA", "SAN"):
        landing(cell, n_band, tx[net], rail_space=RUNSET_RAIL_SPACE)
        landing(cell, p_band, tx[net], rail_space=RUNSET_RAIL_SPACE)

    # Bitlines.  BL comes in on its track, meets the n column it passes first,
    # and rides a short M3 jumper over the seam to its p column.  BLN crosses
    # the whole cell one track higher: its n column is on the far side.  Each
    # track starts at the cell edge, or at its entry column if there is one.
    start = {net: 0 for net in ("BL", "BLN")}
    if spec.has_entry_columns:
        start = {net: x_entry - PAD for net, x_entry in spec.entry_x.items()}
    m2_track(cell, ty["BL"], start["BL"], x["x1"] + PAD)
    m2_track(cell, ty["BLN"], start["BLN"], x["xb"] + PAD)
    m2_track(cell, ty["BL"], x["xb"] - PAD, x["xb"] + PAD)
    for column, track in (("x1", "BL"), ("x3", "BLN")):
        m3_column(cell, x[column], ty[track] - CAP, y_p + CAP, vias=[ty[track]])
        stack_to_m3(cell, x[column], y_p)
    # BLN's n column, up from the BL track's height to its own.
    m3_column(
        cell, x["xb"], ty["BL"] - CAP, ty["BLN"] + CAP, vias=[ty["BL"], ty["BLN"]]
    )

    # Gate contacts, on the seam.  PRECHN: one pad under both precharge gates,
    # contacted between them, where the n band has nothing; the M1 bar drops
    # to the n via row to meet its M3.
    gate_contact(cell, seam, [g["G1"], g["G2"]], x["x2"], y_n - CAP)
    m3_column(cell, tx["PRECHN"], 0, height)
    landing(cell, n_band, x["x2"], rail_space=RUNSET_RAIL_SPACE)

    # YSEL on the two outer gates, YSELN on the two inner ones.  Each is tied on
    # its own M2 track and taps its own select among the group's M3 tracks.
    for role, gates in (("YSEL", ("Ga", "Gb")), ("YSELN", ("G0", "G3"))):
        y_tie = ty[role]
        track = tx[f"{role}[{spec.select}]"]
        for gate in gates:
            gate_contact(cell, seam, [g[gate]], g[gate], y_tie + CAP)
        reach = [g[gates[0]], g[gates[1]], track]
        m2_track(
            cell,
            y_tie,
            min(reach) - PAD,
            max(reach) + PAD,
            vias=[g[gates[0]], g[gates[1]]],
        )
        square(cell, "V2", track, y_tie)
    for j in range(spec.selects):
        for role in ("YSEL", "YSELN"):
            m3_column(cell, tx[f"{role}[{j}]"], 0, height)


def _draw_entry(cell: Any, spec: BitlineMuxSpec) -> None:
    """Each bitline from the left edge, along an M3 column, to its track in every row."""
    for net, x_entry in spec.entry_x.items():
        y_in = spec.entry_y[net]
        via_ys = set(spec.track_ys(net))
        if spec.bitline_layer == "M2":
            via_ys.add(y_in)
            box(cell, "M2", 0, y_in - HALF, x_entry + PAD, y_in + HALF)
            ends = [(y - CAP, y + CAP) for y in via_ys]
        else:
            reach = x_entry + V3_M4_CAP
            reach += -reach % M4_X_GRID
            box(cell, "M4", 0, y_in - M4_HALF, reach, y_in + M4_HALF)
            box(
                cell,
                "V3",
                x_entry - HALF,
                y_in - M4_HALF,
                x_entry + HALF,
                y_in + M4_HALF,
            )
            ends = [(y - CAP, y + CAP) for y in via_ys]
            ends.append((y_in - V3_M3_CAP, y_in + V3_M3_CAP))
        for y_via in via_ys:
            square(cell, "V2", x_entry, y_via)
        box(cell, "M3", x_entry - HALF, min(lo for lo, _ in ends),
            x_entry + HALF, max(hi for _, hi in ends))  # fmt: skip


def build_bitline_mux(
    spec: BitlineMuxSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
    open_ends: tuple[bool, bool] = (False, False),
) -> Any:
    """Draw `spec` and return the gdspy Cell (nanometre coordinates).

    One row is drawn flat.  Two rows are one row cell placed twice, the second
    time mirrored about the VDD rail: everything that runs the height of a row
    on M3 meets itself there, which is all the wiring the copy needs apart from
    its bitlines.

    ``SA`` and ``SAN`` run the leaf's height, except that an end marked in
    `open_ends` (``(bottom, top)``) stops `SENSE_END_CLEARANCE` short: at a
    group's outer ends, so that a group stacked on another's (the next bit's
    column in an array of them) does not join their sense lines.  The select
    and precharge tracks do run on; they are the same nets in every column
    of a bank.
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
    bottom_open, top_open = open_ends
    for net in ("SA", "SAN"):
        m3_column(
            cell,
            spec.track_x[net],
            SENSE_END_CLEARANCE if bottom_open else 0,
            spec.height - (SENSE_END_CLEARANCE if top_open else 0),
        )
    if spec.has_entry_columns:
        _draw_entry(cell, spec)

    if draw_pin_labels:
        for pin, (metal, origin) in spec.pin_positions.items():
            if pin not in ("VDD", "VSS"):
                label(cell, pin, metal, origin)
        # Every rail is its own conductor until a power grid joins them.
        for y_rail, net in spec.rails:
            label(cell, net, "M1", (spec.width / 2, y_rail))
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
    cell_name = name or spec.group_cell_name
    cell = _new_cell(cell_name, lib)
    height = spec.height

    for i in range(spec.selects):
        leaf_spec = spec.for_row(i)
        flipped = bool(i % 2)
        # The group's outer ends, in the leaf's own frame (a flipped leaf's
        # bottom is on the group's top side).
        outer = (i == 0, i == spec.selects - 1)
        leaf = build_bitline_mux(
            leaf_spec,
            name=f"{cell_name}__leaf{i}",
            lib=lib,
            draw_pin_labels=False,
            open_ends=outer[::-1] if flipped else outer,
        )
        y0 = (i + 1) * height if flipped else i * height
        cell.add(gdspy.CellReference(leaf, origin=(0, y0), x_reflection=flipped))
    if draw_pin_labels:
        for pin, (metal, origin) in spec.group_pin_positions().items():
            label(cell, pin.split(".")[0], metal, origin)
    box(cell, "BOUNDARY", 0, 0, spec.width, spec.selects * height)
    return cell


def main(argv: list[str] | None = None) -> None:
    """Write a bitline-leaf GDS: ``asap7-bitline-mux --selects 4 --group``."""
    args = parse_spec(
        BitlineMuxSpec,
        argv,
        description="Generate an ASAP7 precharge + column-mux leaf for one bitline pair.",
        options=(
            Option("group", bool, False, "Draw the whole stacked mux, not one leaf."),
        ),
    )
    spec = args.spec
    build = build_bitline_mux_group if args.group else build_bitline_mux
    cell, out = write_gds(lambda lib: build(spec, lib=lib), args.out)
    leaves = spec.selects if args.group else 1
    print(f"✓ {cell.name}: {spec.width} x {leaves * spec.height} nm -> {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
