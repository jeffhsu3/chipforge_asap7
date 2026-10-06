"""A bitline precharge + column-mux leaf one ASAP7 6T row (270 nm) tall.

`BitlineMuxSpec` stacks an n band under a p band, which needs a 297 nm row:
the 6T bitcell's row is 270 nm, so its IO had to stagger two columns of
leaves (`StaggeredIoColumnSpec`).  The released ASAP7 SRAM's own leaf
(``sram_prech_ymux_6t112_v1`` in ``srambank_32b``) fits 270 nm another way,
and this draws that arrangement:

* the pFETs and nFETs stand **side by side** in x, each region the full row
  tall; the gates run vertically through two 3-fin strips, the lower strip
  carrying BLN and the upper BL, so every gate serves both bitlines;
* the gate contacts sit in the 54 nm channel between the strips;
* the six devices are `BitlineMuxSpec`'s six (precharge, and a transmission
  gate per bitline), so its netlist and LVS references apply unchanged::

        x:  0        54   81  108  135  162      216     270  297  324
            |  pFETs (NWELL, PSELECT)               | nFETs (NSELECT)  |
    BL  ->  | VDD  PRECHN  BL  YSELN  SA            | SA  YSEL  BL     |  upper strip
    BLN ->  | VDD  PRECHN  BLN YSELN  SAN           | SAN YSEL  BLN    |  lower strip

The bitlines enter on M2 from the left edge at the strips' heights (70 and
173 nm, i.e. the 6T cell's 83.5 and 186.5 nm with the leaf half a fin pitch
up on the array's grid), and the sense lines leave on M2 to the right.  M3
tracks run the leaf's height, left to right: PRECHN, ``YSELN[0..selects)``,
``YSEL[0..selects)``, SAN, SA; each leaf ties its own select pair and both
sense lines to them with a V2, so a stacked group needs no other wiring.

With ``local_ysel`` a leaf makes its own ``YSEL`` from ``YSELN`` with an
inverter -- a plain 270 nm one, its nFET on the strip by the VSS rail and its
pFET on the one by VDD, implants split at the strips' seam as a standard
cell's -- standing past the nFETs, under the select tracks.  Only ``YSELN``
then crosses the column: from 8:1 the track bus, not the devices, sets the
leaf's width, and it loses half its select tracks (756 -> 540 nm at 8:1,
1350 -> 756 at 16:1; at 4:1 the inverter would widen the leaf).

With ``predecode`` the column carries two one-hot groups instead, ``YPA`` (4
lines) and ``YPB`` (``selects / 4``), and leaf ``i`` makes ``YSELN =
NAND(YPA[i % 4], YPB[i // 4])`` and ``YSEL`` from it, a NAND2 and an
inverter on the same 270 nm row (4 columns past the nFETs).  At 16:1 the
column then carries 8 select tracks, not 16, and the gates set the leaf's
width: 648 nm against 756.

Coordinates are the released cell's, made whole-nanometre (its one off-grid
V0 is the cell's only DRC finding beyond latch-up); everything else is
pure arithmetic on the spec, like `BitlineMuxSpec`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from ..layout.grid import GATE_PITCH
from ..layout.layers import box, require_gdspy
from .bitline_mux import BITLINE_MUX_PINS, SENSE_END_CLEARANCE
from .cli import Option, parse_spec, write_gds
from .row import RowBand, RowStack
from .rowcell import HALF, label, square

__all__ = [
    "SidewaysMuxSpec",
    "build_sideways_mux",
    "build_sideways_mux_group",
    "sideways_mux_group_pins",
]

ROW = 270
P_REGION = 216  # the pFETs' NWELL/PSELECT, from the left edge
N_REGION_END = 378  # the nFETs' NSELECT, from P_REGION
#: The two fin strips (ACTIVE), lower (BLN) and upper (BL): 3 fins each.
STRIPS = ((27, 108), (162, 243))
P_ACTIVE, N_ACTIVE = (46, 170), (262, 332)
#: Source/drain columns: pFET VDD, bitline, sense line; nFET sense line, bitline.
SD = {"P_VDD": 54, "P_BL": 108, "P_SA": 162, "N_SA": 270, "N_BL": 324}
#: Active gates; every other gate on the grid is a dummy.
GATES = {"PRECHN": 81, "YSELN": 135, "YSEL": 297}
GATE_CONTACT_Y = (124, 146)  # LIG between the strips
#: M2 bars: the bitline entries, the sense lines out, the select ties.
#: (PRECHN's tie sits 3 nm above the released cell's 106: V1.S.4 wants 27 nm
#: corner to corner from BLN's V1, which has a 5 nm M2 end cap.)
M2_Y = {"BLN": 70, "BL": 173, "SAN": 89, "SA": 181, "PRECHN": 109, "YSELN": 135, "YSEL": 217}
TRACK_PITCH = 36
YSELN_V1, YSEL_V1 = 176, 360  # the select gate ties' V1s, up to their M2 bars
PRECHN_TRACK = 51
FIRST_SELECT_TRACK = 123  # the released cell's: 87 would meet the PRECHN tie's M2
FIRST_YSEL_TRACK = 267  # the released cell's first YSEL track
SD_HALF = 12
#: The local YSEL inverter: its source and drain columns, gate, output bar
#: (over the dummy gate past it), and the input's V1 on the YSELN bar.
INV_SOURCE, INV_DRAIN, INV_GATE, INV_OUT = 432, 486, 459, 513
INV_IN_V1 = 437
INV_REGION = 378  # its implants and well, from the nFETs' end
INV_END = 540  # the leaf's width with it
INV_VIA_Y = (54, 216)  # its drains' V0s: lower strip, upper strip
#: The predecode gates (NAND2 then inverter), past the nFETs: columns, the
#: inputs' gates, the inverter's, the two strip crossings over the dummy
#: gates either side, the M2 rows the inputs take, the via rows.
PRE_COLS = (432, 486, 540, 594)  # n: YSELN, (series), VSS, YSEL; p: VDD, YSELN, VDD, YSEL
PRE_GATE_A, PRE_GATE_B, PRE_GATE_INV = 459, 513, 567
PRE_YSELN_X, PRE_YSEL_X = 405, 621
PRE_ROW_A, PRE_ROW_B = 89, 181
PRE_VIA_Y = (41, 229)  # by the lower rail, by the upper one
PRE_END = 648
PRE_SENSE_V1 = 162  # the sense lines' ties, left of the select tracks
#: The sense lines' tracks: their ties' M2 has to start past the bitline
#: entries' (x <= 122) on rows 173/181 (a swapped leaf's SAN is on 181).
PRE_SENSE_TRACKS = (159, 195)
FIRST_PREDECODE_TRACK = 267
RAIL_V1_FIRST = 27  # rail V1s every two gate pitches from here, between the rail V0s
M2_PAST_V2 = 17


@dataclass(frozen=True)
class SidewaysMuxSpec:
    """One 270 nm leaf: precharge and column select for one bitline pair.

    `select` is the leaf's index among `selects`.  `bitline_entry` is where
    an unmirrored array row delivers (BL, BLN), from the row's bottom, and
    `grid_offset` how far up the leaf sits on the array's grid.

    A `swapped` leaf serves a mirrored array row (the odd rows), whose BL is
    low and BLN high: it is the same layout with the nets swapped -- BL and
    SA on the lower strip, BLN and SAN on the upper, VDD on the bottom rail
    and VSS on the top -- as the released ``_v2`` leaf is.  Reflecting the
    leaf instead would put it half a fin pitch on the wrong side of its row
    (it sits `grid_offset` up), 27 nm into the next leaf.
    """

    selects: int = 4
    select: int = 0
    fins: int = 3
    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"
    bitline_entry: tuple[float, float] = (186.5, 83.5)
    grid_offset: float = 13.5
    swapped: bool = False
    local_ysel: bool = False
    predecode: bool = False
    _stack: RowStack = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.selects < 1:
            raise ValueError("a mux selects at least one leaf")
        if not 0 <= self.select < self.selects:
            raise ValueError(f"select {self.select} is not one of {self.selects}")
        if self.predecode and (self.selects < 8 or self.selects % 4):
            raise ValueError(f"predecoded selects come in fours, 8 or more; got {self.selects}")
        if self.fins != 3:
            raise ValueError("the 270 nm leaf's strips are three fins (the 6T row's); use fins=3")
        local = tuple(round(y - self.grid_offset, 3) for y in self.bitline_entry)
        if local != (M2_Y["BL"], M2_Y["BLN"]):
            raise ValueError(
                f"bitlines entering at {self.bitline_entry} (grid offset {self.grid_offset}) "
                f"land at {local}; the leaf's BL/BLN bars are at {(M2_Y['BL'], M2_Y['BLN'])}"
            )
        # The device model and gate length come from the ASAP7 7.5-track row.
        object.__setattr__(self, "_stack", RowStack(rows=((3, 3),), vt=self.vt, band_height=135))

    # ── Size ─────────────────────────────────────────────────────────────────
    @property
    def height(self) -> int:
        return ROW

    @property
    def track_x(self) -> dict[str, int]:
        """Each M3 track's centre: PRECHN, the selects' bars, then the sense lines."""
        tracks = {"PRECHN": PRECHN_TRACK}
        if self.predecode:
            # The sense lines first, tied near the leaf's own columns; then
            # the two groups.
            tracks["SAN"], tracks["SA"] = PRE_SENSE_TRACKS
            x = FIRST_PREDECODE_TRACK
            for net in self.group_nets:
                tracks[net] = x
                x += TRACK_PITCH
            return tracks
        x = FIRST_SELECT_TRACK
        for role in ("YSELN",) if self.local_ysel else ("YSELN", "YSEL"):
            if role == "YSEL":
                # Over the nFETs at the earliest, where their gate's tie is;
                # the sense lines' tracks then clear the sense-line M1 bars.
                x = max(x, FIRST_YSEL_TRACK)
            for j in range(self.selects):
                tracks[f"{role}[{j}]"] = x
                x += TRACK_PITCH
        tracks["SAN"], tracks["SA"] = x, x + TRACK_PITCH
        return tracks

    @property
    def width(self) -> int:
        """Whole gate pitches, the devices at least; the last M3 track inside the edge."""
        tracks = GATE_PITCH * math.ceil((max(self.track_x.values()) + HALF) / GATE_PITCH)
        devices = PRE_END if self.predecode else INV_END if self.local_ysel else N_REGION_END
        return max(tracks, devices)

    @property
    def bands(self) -> tuple[RowBand, RowBand]:
        """The n and p device flavours (model, gate length), for netlists and LVS references."""
        n_band, p_band = self._stack.bands()
        return n_band, p_band

    @property
    def cell_name(self) -> str:
        vt = "" if self.vt == "rvt" else f"_{self.vt}"
        local = "_pd" if self.predecode else "_ly" if self.local_ysel else ""
        return f"bitline_mux270_s{self.select}of{self.selects}{vt}{local}{'_sw' if self.swapped else ''}"

    @property
    def group_cell_name(self) -> str:
        vt = "" if self.vt == "rvt" else f"_{self.vt}"
        local = "_pd" if self.predecode else "_ly" if self.local_ysel else ""
        return f"bitline_mux270_x{self.selects}{vt}{local}"

    def for_select(self, index: int, swapped: bool | None = None) -> SidewaysMuxSpec:
        swapped = self.swapped if swapped is None else swapped
        return SidewaysMuxSpec(self.selects, index, self.fins, self.vt, self.bitline_entry, self.grid_offset,
                               swapped, self.local_ysel, self.predecode)  # fmt: skip

    @property
    def group_nets(self) -> tuple[str, ...]:
        """The predecoded select lines every leaf of a group shares (none without `predecode`)."""
        if not self.predecode:
            return ()
        return (*(f"YPA[{k}]" for k in range(4)), *(f"YPB[{k}]" for k in range(self.selects // 4)))

    @property
    def select_inputs(self) -> tuple[str, str]:
        """This leaf's two predecode lines."""
        return f"YPA[{self.select % 4}]", f"YPB[{self.select // 4}]"

    @property
    def leaf_nets(self) -> tuple[str, ...]:
        """A leaf's own pins in a group (``BL[i]``...): ``YSEL`` is internal with `local_ysel`, both
        selects with `predecode`."""
        if self.predecode:
            return ("BL", "BLN")
        return ("BL", "BLN", "YSELN") if self.local_ysel else ("BL", "BLN", "YSEL", "YSELN")

    @property
    def pins(self) -> tuple[str, ...]:
        if self.predecode:
            base = tuple(p for p in BITLINE_MUX_PINS if p not in ("YSEL", "YSELN"))
            return (*base[:-2], *self.select_inputs, *base[-2:])
        return tuple(p for p in BITLINE_MUX_PINS if not (self.local_ysel and p == "YSEL"))

    def leaf_devices(self, index: int) -> tuple[tuple[str, str, str, str, str, int], ...]:
        """Leaf `index`'s devices, as a group names them."""
        return self.for_select(index).devices

    @property
    def strip_nets(self) -> tuple[tuple[str, str], tuple[str, str]]:
        """``((bitline, sense line) of the lower strip, of the upper strip)``."""
        low, high = ("BLN", "SAN"), ("BL", "SA")
        return (high, low) if self.swapped else (low, high)

    @property
    def rails(self) -> tuple[tuple[int, str], tuple[int, str]]:
        return ((0, "VDD"), (ROW, "VSS")) if self.swapped else ((0, "VSS"), (ROW, "VDD"))

    # ── Pins ──────────────────────────────────────────────────────────────────
    @property
    def pin_positions(self) -> dict[str, tuple[str, tuple[float, float]]]:
        tx = self.track_x
        (bl_low, sa_low), (bl_high, sa_high) = self.strip_nets
        pins = {
            bl_low: ("M2", (HALF, M2_Y["BLN"])),
            bl_high: ("M2", (HALF, M2_Y["BL"])),
            sa_low: ("M3", (tx[sa_low], M2_Y["SAN"])),
            sa_high: ("M3", (tx[sa_high], M2_Y["SA"])),
            "PRECHN": ("M3", (tx["PRECHN"], M2_Y["PRECHN"])),
        }
        if self.predecode:
            for net, y in zip(self.select_inputs, (PRE_ROW_A, PRE_ROW_B)):
                pins[net] = ("M3", (tx[net], y))
        else:
            pins["YSELN"] = ("M3", (tx[f"YSELN[{self.select}]"], M2_Y["YSELN"]))
        if not (self.local_ysel or self.predecode):
            pins["YSEL"] = ("M3", (tx[f"YSEL[{self.select}]"], M2_Y["YSEL"]))
        for y, net in self.rails:
            pins[net] = ("M1", (self.width / 2, y))
        return pins

    def group_pin_positions(self) -> dict[str, tuple[str, tuple[float, float]]]:
        """Pins of the stacked group: leaf ``i`` at ``y = i * 270``, odd leaves swapped."""
        pins: dict[str, tuple[str, tuple[float, float]]] = {}
        tx = self.track_x
        for i in range(self.selects):
            leaf = self.for_select(i, swapped=bool(i % 2))
            for net in ("BL", "BLN"):
                metal, (x, y) = leaf.pin_positions[net]
                pins[f"{net}[{i}]"] = (metal, (x, i * ROW + y))
            if self.predecode:
                continue
            if not self.local_ysel:
                pins[f"YSEL[{i}]"] = ("M3", (tx[f"YSEL[{i}]"], i * ROW + M2_Y["YSEL"]))
            pins[f"YSELN[{i}]"] = ("M3", (tx[f"YSELN[{i}]"], i * ROW + M2_Y["YSELN"]))
        for net in self.group_nets:
            pins[net] = ("M3", (tx[net], ROW / 2))
        pins["SA"] = ("M3", (tx["SA"], M2_Y["SA"]))
        pins["SAN"] = ("M3", (tx["SAN"], M2_Y["SAN"]))
        pins["PRECHN"] = ("M3", (tx["PRECHN"], M2_Y["PRECHN"]))
        for k in range(self.selects + 1):
            net = "VDD" if k % 2 else "VSS"
            pins[f"{net}.{k}" if k > 1 else net] = ("M1", (self.width / 2, k * ROW))
        return pins

    # ── Netlist ───────────────────────────────────────────────────────────────
    @property
    def devices(self) -> tuple[tuple[str, str, str, str, str, int], ...]:
        """``(name, drain, gate, source, flavor, fins)``: `BitlineMuxSpec`'s six, and the
        local YSEL inverter's two."""
        f = self.fins
        inverter = (("MNI", "YSEL", "YSELN", "VSS", "n", f), ("MPI", "YSEL", "YSELN", "VDD", "p", f))
        if self.predecode:
            a, b = self.select_inputs
            series = f"YN[{self.select}]"  # the NAND's nFET stack, contacted: one net
            inverter = (("MNA", "YSELN", a, series, "n", f), ("MNB", series, b, "VSS", "n", f),
                        ("MPA", "YSELN", a, "VDD", "p", f), ("MPB", "YSELN", b, "VDD", "p", f),
                        *inverter)  # fmt: skip
        return (*(inverter if self.local_ysel or self.predecode else ()),
            ("MNT", "BL", "YSEL", "SA", "n", f),
            ("MNC", "BLN", "YSEL", "SAN", "n", f),
            ("MPT", "SA", "YSELN", "BL", "p", f),
            ("MPC", "SAN", "YSELN", "BLN", "p", f),
            ("MPPT", "BL", "PRECHN", "VDD", "p", f),
            ("MPPC", "BLN", "PRECHN", "VDD", "p", f),
        )

    def netlist(self, name: str | None = None) -> str:
        """A BSIM-CMG subcircuit sized by ``nfin``, pins as `BitlineMuxSpec`'s."""
        title = name or self.cell_name
        n_band, p_band = self.bands
        lines = [
            "* ASAP7 270 nm bitline leaf: precharge and transmission-gate column select",
            f".SUBCKT {title} {' '.join(self.pins)}",
        ]
        for device, drain, gate, source, flavor, fins in self.devices:
            band = n_band if flavor == "n" else p_band
            bulk = "VSS" if flavor == "n" else "VDD"
            lines.append(f"{device} {drain} {gate} {source} {bulk} {band.spec.model} "
                         f"nfin={fins} l={band.spec.gate_length}n nf=1 m=1")  # fmt: skip
        lines += [f".ENDS {title}", ".END", ""]
        return "\n".join(lines)


# ── Layout ────────────────────────────────────────────────────────────────────
def _new_cell(name: str, lib: Any) -> Any:
    if lib is not None:
        return lib.new_cell(name)
    return require_gdspy().Cell(name, exclude_from_current=True)


def _v0(cell: Any, x: float, y: float) -> None:
    square(cell, "V0", x, y)


def _draw_leaf(cell: Any, spec: SidewaysMuxSpec, sense_tracks: bool) -> None:
    w, tx = spec.width, spec.track_x

    # Implants and well: pFETs left, nFETs right, each the row's height.
    box(cell, "NWELL", 0, 0, P_REGION, ROW)
    box(cell, "PSELECT", 0, 0, P_REGION, ROW)
    box(cell, "NSELECT", P_REGION, 0, N_REGION_END, ROW)
    if spec.predecode:
        _draw_predecode(cell, spec)
    elif spec.local_ysel:
        _draw_inverter(cell, spec)
    # Fin and gate grids across the whole leaf; gates cut on both rails.
    for k in range(10):
        box(cell, "FIN", 0, 10 + 27 * k, w, 17 + 27 * k)
    for x in range(GATE_PITCH // 2, w, GATE_PITCH):
        box(cell, "GATE", x - 10, -5, x + 10, ROW + 5)
    box(cell, "GATE_CUT", 0, -22, w, 22)
    box(cell, "GATE_CUT", 0, ROW - 22, w, ROW + 22)

    # Diffusion: two strips a side; S/D contacts (LISD, and SDT over ACTIVE).
    for lo, hi in STRIPS:
        box(cell, "ACTIVE", *P_ACTIVE[:1], lo, P_ACTIVE[1], hi)
        box(cell, "ACTIVE", N_ACTIVE[0], lo, N_ACTIVE[1], hi)
        for x in SD.values():
            box(cell, "SDT", x - SD_HALF, lo, x + SD_HALF, hi)
            box(cell, "LISD", x - SD_HALF, lo, x + SD_HALF, hi)
    # The precharge source runs on to the VDD rail: the top one, or a swapped
    # leaf's bottom one.
    if spec.swapped:
        box(cell, "LISD", SD["P_VDD"] - SD_HALF, 0, SD["P_VDD"] + SD_HALF, STRIPS[0][0])
    else:
        box(cell, "LISD", SD["P_VDD"] - SD_HALF, STRIPS[1][1], SD["P_VDD"] + SD_HALF, ROW)

    # Rails: LIG and M1 on both edges, M2 over them; V0 every gate pitch.
    # (Edge to edge, as a standard cell's: the released leaf's start 16 nm in,
    # which leaves a LIG gap against a neighbouring filler's.)
    box(cell, "LIG", 0, -8, w, 8)
    box(cell, "LIG", 0, ROW - 8, w, ROW + 8)
    for y in (0, ROW):
        box(cell, "M1", 0, y - HALF, w, y + HALF)
        box(cell, "M2", 0, y - HALF, w, y + HALF)
        for x in range(SD["P_VDD"], w - HALF, GATE_PITCH):
            _v0(cell, x, y)
        # The M2 rail is the M1 rail's twin, not floating metal: abutted
        # leaves (and blocks) otherwise merge two isolated pieces of it.
        for x in range(RAIL_V1_FIRST, w - HALF, 2 * GATE_PITCH):
            square(cell, "V1", x, y)

    # VDD: both strips' precharge source, joined by an M1 strap at x = 36.
    _v0(cell, SD["P_VDD"], 39)
    _v0(cell, SD["P_VDD"], 189)
    box(cell, "M1", 27, 30, 45, 198)
    box(cell, "M1", 27, 30, 68, 48)
    box(cell, "M1", 27, 180, 68, 198)

    # Gate contacts in the channel, on M1, up to their M2 ties.
    box(cell, "LIG", 54, *GATE_CONTACT_Y[:1], 92, GATE_CONTACT_Y[1])
    box(cell, "LIG", 124, GATE_CONTACT_Y[0], 161, GATE_CONTACT_Y[1])
    box(cell, "LIG", 285, GATE_CONTACT_Y[0], 324, GATE_CONTACT_Y[1])
    _v0(cell, 72, 135)  # PRECHN
    box(cell, "M1", 63, 92, 81, 149)
    square(cell, "V1", 72, M2_Y["PRECHN"])
    box(cell, "M2", 37, M2_Y["PRECHN"] - HALF, 86, M2_Y["PRECHN"] + HALF)
    square(cell, "V2", PRECHN_TRACK, M2_Y["PRECHN"])
    _v0(cell, 145, 135)  # YSELN
    box(cell, "M1", 136, 126, 207, 144)
    square(cell, "V1", YSELN_V1, M2_Y["YSELN"])
    _v0(cell, 314, 135)  # YSEL
    box(cell, "M1", 300, 126, 366, 144)
    box(cell, "M1", 351, 126, 369, 231)
    square(cell, "V1", YSEL_V1, M2_Y["YSEL"])
    # Each select's M2 bar reaches its whole column of tracks; the leaf taps its own.
    # (Each bar covers its gate tie's V1 too: with one or two selects the
    # tracks alone would stop short of it.)  A local YSEL comes from the
    # inverter: YSELN's bar reaches its input, YSEL's its output.
    if spec.predecode:
        yseln, ysel = [PRE_YSELN_X, PRE_GATE_INV], [PRE_YSEL_X]
    else:
        yseln = [tx[f"YSELN[{j}]"] for j in range(spec.selects)] + ([INV_IN_V1] if spec.local_ysel else [])
        ysel = [INV_OUT] if spec.local_ysel else [tx[f"YSEL[{j}]"] for j in range(spec.selects)]
    box(cell, "M2", 109, M2_Y["YSELN"] - HALF, max(*yseln, YSELN_V1) + M2_PAST_V2, M2_Y["YSELN"] + HALF)
    box(cell, "M2", min(YSEL_V1, *ysel) - M2_PAST_V2, M2_Y["YSEL"] - HALF,
        max(YSEL_V1, *ysel) + M2_PAST_V2, M2_Y["YSEL"] + HALF)  # fmt: skip
    if not spec.predecode:
        square(cell, "V2", tx[f"YSELN[{spec.select}]"], M2_Y["YSELN"])
    if not (spec.local_ysel or spec.predecode):
        square(cell, "V2", tx[f"YSEL[{spec.select}]"], M2_Y["YSEL"])

    # Bitlines: the pFET column, along M1, to the nFET column; in on M2.
    for net, (lo, hi), m1, n_v0 in (("BLN", STRIPS[0], (27, 105), 68),
                                    ("BL", STRIPS[1], (159, 243), 213)):  # fmt: skip
        y_pv0 = 91 if net == "BLN" else 194
        _v0(cell, SD["P_BL"], y_pv0)
        box(cell, "M1", 99, m1[0], 117, m1[1])
        bar = (27, 45) if net == "BLN" else (225, 243)
        box(cell, "M1", 99, bar[0], 333, bar[1])
        _v0(cell, SD["N_BL"], n_v0)
        box(cell, "M1", 315, min(bar[0], n_v0 - HALF - 5), 333, max(bar[1], n_v0 + HALF + 5))
        square(cell, "V1", 108, M2_Y[net])
        box(cell, "M2", 0, M2_Y[net] - HALF, 122, M2_Y[net] + HALF)

    # Sense lines: both sides' columns on one M1 bar, out on M2 to their tracks.
    (_, sa_low), (_, sa_high) = spec.strip_nets
    for net, y in ((sa_low, M2_Y["SAN"]), (sa_high, M2_Y["SA"])):
        _v0(cell, SD["P_SA"], y)
        _v0(cell, SD["N_SA"], y)
        box(cell, "M1", 148, y - HALF, 284, y + HALF)
        x_v1 = PRE_SENSE_V1 if spec.predecode else 251
        square(cell, "V1", x_v1, y)
        box(cell, "M2", min(x_v1, tx[net]) - M2_PAST_V2, y - HALF, max(x_v1, tx[net]) + M2_PAST_V2, y + HALF)
        square(cell, "V2", tx[net], y)

    # M3: the precharge and select tracks the leaf's height (the sense lines'
    # are the leaf's own only when drawn alone; a group runs them itself).
    for net, x in tx.items():
        if net in ("SA", "SAN") and not sense_tracks:
            continue
        box(cell, "M3", x - HALF, 0, x + HALF, ROW)


def _draw_inverter(cell: Any, spec: SidewaysMuxSpec) -> None:
    """YSEL = !YSELN past the nFETs: nFET on the strip by the VSS rail, pFET on the one by VDD."""
    w = spec.width
    seam = ROW // 2
    (vss_y, vdd_y) = (ROW, 0) if spec.swapped else (0, ROW)
    n_strip, p_strip = (STRIPS[1], STRIPS[0]) if spec.swapped else (STRIPS[0], STRIPS[1])
    n_half, p_half = ((seam, ROW), (0, seam)) if spec.swapped else ((0, seam), (seam, ROW))
    box(cell, "NSELECT", INV_REGION, n_half[0], w, n_half[1])
    box(cell, "PSELECT", INV_REGION, p_half[0], w, p_half[1])
    box(cell, "NWELL", INV_REGION, p_half[0], w, p_half[1])
    for (lo, hi), rail_y in ((n_strip, vss_y), (p_strip, vdd_y)):
        box(cell, "ACTIVE", INV_SOURCE - 8, lo, INV_DRAIN + 8, hi)
        for x in (INV_SOURCE, INV_DRAIN):
            box(cell, "SDT", x - SD_HALF, lo, x + SD_HALF, hi)
            box(cell, "LISD", x - SD_HALF, lo, x + SD_HALF, hi)
        # The source on to its rail (whose V0 stands on this column).
        box(cell, "LISD", INV_SOURCE - SD_HALF, min(rail_y, lo), INV_SOURCE + SD_HALF, max(rail_y, hi))
    # Output: each drain's V0 on a flag to a bar over the dummy gate past it.
    for y in INV_VIA_Y:
        _v0(cell, INV_DRAIN, y)
        box(cell, "M1", INV_DRAIN - 14, y - HALF, INV_OUT + HALF, y + HALF)
    box(cell, "M1", INV_OUT - HALF, INV_VIA_Y[0] - HALF, INV_OUT + HALF, M2_Y["YSEL"] + 14)
    square(cell, "V1", INV_OUT, M2_Y["YSEL"])
    # Input: a contact between the strips, an M1 pad left to its V1 on YSELN's bar.
    box(cell, "LIG", INV_GATE - 11, GATE_CONTACT_Y[0], INV_GATE + 11, GATE_CONTACT_Y[1])
    _v0(cell, INV_GATE, seam)
    box(cell, "M1", INV_IN_V1 - 14, seam - HALF, INV_GATE + 14, seam + HALF)
    square(cell, "V1", INV_IN_V1, M2_Y["YSELN"])


def _draw_predecode(cell: Any, spec: SidewaysMuxSpec) -> None:
    """YSELN = NAND(YPA[a], YPB[b]) and YSEL = !YSELN past the nFETs, one 270 nm row.

    nFETs on the strip by the VSS rail (the NAND's series pair, the
    inverter's), pFETs on the one by VDD; each output crosses between the
    strips on M1 over the dummy gate beside it.
    """
    w = spec.width
    seam = ROW // 2
    (vss_y, vdd_y) = (ROW, 0) if spec.swapped else (0, ROW)
    n_strip, p_strip = (STRIPS[1], STRIPS[0]) if spec.swapped else (STRIPS[0], STRIPS[1])
    n_half, p_half = ((seam, ROW), (0, seam)) if spec.swapped else ((0, seam), (seam, ROW))
    n_via, p_via = (PRE_VIA_Y[1], PRE_VIA_Y[0]) if spec.swapped else PRE_VIA_Y
    box(cell, "NSELECT", INV_REGION, n_half[0], w, n_half[1])
    box(cell, "PSELECT", INV_REGION, p_half[0], w, p_half[1])
    box(cell, "NWELL", INV_REGION, p_half[0], w, p_half[1])
    c0, c1, c2, c3 = PRE_COLS
    for (lo, hi), rail_y, sources in ((n_strip, vss_y, (c2,)), (p_strip, vdd_y, (c0, c2))):
        box(cell, "ACTIVE", c0 - 8, lo, c3 + 8, hi)
        for x in PRE_COLS:
            box(cell, "SDT", x - SD_HALF, lo, x + SD_HALF, hi)
            box(cell, "LISD", x - SD_HALF, lo, x + SD_HALF, hi)
        for x in sources:
            box(cell, "LISD", x - SD_HALF, min(rail_y, lo), x + SD_HALF, max(rail_y, hi))

    def flag(x_via: float, y: float, x_bar: float) -> None:
        _v0(cell, x_via, y)
        box(cell, "M1", min(x_via, x_bar) - 14, y - HALF, max(x_via, x_bar) + 14, y + HALF)

    # YSELN: the NAND's n drain and p drain, over the dummy gate on the left.
    flag(c0, n_via, PRE_YSELN_X)
    flag(c1, p_via, PRE_YSELN_X)
    # YSEL: the inverter's drains, over the dummy gate on the right.
    flag(c3, n_via, PRE_YSEL_X)
    flag(c3, p_via, PRE_YSEL_X)
    lo, hi = min(PRE_VIA_Y) - HALF, max(PRE_VIA_Y) + HALF
    for x in (PRE_YSELN_X, PRE_YSEL_X):
        box(cell, "M1", x - HALF, lo, x + HALF, hi)
    square(cell, "V1", PRE_YSELN_X, M2_Y["YSELN"])
    square(cell, "V1", PRE_YSEL_X, M2_Y["YSEL"])
    # Gate contacts between the strips: the inputs' bars on to their rows,
    # the inverter's on YSELN's.
    for x, y_v1 in ((PRE_GATE_A, PRE_ROW_A), (PRE_GATE_B, PRE_ROW_B), (PRE_GATE_INV, M2_Y["YSELN"])):
        box(cell, "LIG", x - 11, GATE_CONTACT_Y[0], x + 11, GATE_CONTACT_Y[1])
        _v0(cell, x, seam)
        box(cell, "M1", x - HALF, min(seam, y_v1) - 14, x + HALF, max(seam, y_v1) + 14)
        square(cell, "V1", x, y_v1)
    # The inputs: each on its row from its group's track.
    tx = spec.track_x
    for net, x_gate, y in zip(spec.select_inputs, (PRE_GATE_A, PRE_GATE_B), (PRE_ROW_A, PRE_ROW_B)):
        box(cell, "M2", min(tx[net], x_gate) - M2_PAST_V2, y - HALF, max(tx[net], x_gate) + M2_PAST_V2, y + HALF)
        square(cell, "V2", tx[net], y)


def build_sideways_mux(
    spec: SidewaysMuxSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Draw one leaf (nanometre coordinates) and return the gdspy Cell."""
    spec = spec or SidewaysMuxSpec()
    cell = _new_cell(name or spec.cell_name, lib)
    _draw_leaf(cell, spec, sense_tracks=True)
    if draw_pin_labels:
        for pin, (metal, origin) in spec.pin_positions.items():
            label(cell, pin, metal, origin)
    box(cell, "BOUNDARY", 0, 0, spec.width, ROW)
    return cell


def sideways_mux_group_pins(spec: SidewaysMuxSpec) -> tuple[str, ...]:
    """Pins of a `spec.selects`-to-one group: per-leaf bitlines and selects, shared everything else."""
    per_leaf = [f"{pin}[{i}]" for i in range(spec.selects) for pin in spec.leaf_nets]
    return (*per_leaf, *spec.group_nets, "SA", "SAN", "PRECHN", "VDD", "VSS")


def build_sideways_mux_group(
    spec: SidewaysMuxSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Stack `spec.selects` leaves, leaf ``i`` at ``y = 270 i``, odd ones swapped.

    As the array's rows alternate mirrored, the leaves alternate swapped
    (see `SidewaysMuxSpec`), so each meets its row's bitlines and every two
    neighbours share a rail of one net.  The precharge and select tracks meet end to end.  ``SA`` and ``SAN`` run
    the group's height but stop `SENSE_END_CLEARANCE` short of its ends, so a
    group stacked on another's (the next bit's) keeps its own sense lines.
    `spec.select` is ignored.
    """
    spec = spec or SidewaysMuxSpec()
    gdspy = require_gdspy()
    cell_name = name or spec.group_cell_name
    cell = _new_cell(cell_name, lib)
    for i in range(spec.selects):
        leaf = _new_cell(f"{cell_name}__leaf{i}", lib)
        _draw_leaf(leaf, spec.for_select(i, swapped=bool(i % 2)), sense_tracks=False)
        box(leaf, "BOUNDARY", 0, 0, spec.width, ROW)
        cell.add(gdspy.CellReference(leaf, origin=(0, i * ROW)))
    top = spec.selects * ROW
    for net in ("SA", "SAN"):
        x = spec.track_x[net]
        box(cell, "M3", x - HALF, SENSE_END_CLEARANCE, x + HALF, top - SENSE_END_CLEARANCE)
    if draw_pin_labels:
        for pin, (metal, origin) in spec.group_pin_positions().items():
            label(cell, pin.split(".")[0], metal, origin)
    box(cell, "BOUNDARY", 0, 0, spec.width, top)
    return cell


def main(argv: list[str] | None = None) -> None:
    """Write a 270 nm bitline-leaf GDS: ``asap7-bitline-mux-270 --selects 4 --group``."""
    args = parse_spec(
        SidewaysMuxSpec,
        argv,
        description="Generate an ASAP7 270 nm precharge + column-mux leaf for the 6T row.",
        options=(Option("group", bool, False, "Draw the whole stacked mux, not one leaf."),),
    )
    build = build_sideways_mux_group if args.group else build_sideways_mux
    cell, out = write_gds(lambda lib: build(args.spec, lib=lib), args.out)
    print(f"wrote {cell.name} to {out}")


if __name__ == "__main__":
    main()
