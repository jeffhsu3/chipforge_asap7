"""`WriteDriverSpec`'s twelve transistors on one 270 nm (7.5-track) row.

The same circuit -- an inverter makes ``DN``; nFET pass gates load ``(D,
DN)`` into a latch with one-fin pFET keepers while write enable is low; two
transmission gates put the latch on ``SA``/``SAN`` while it is high -- in
702 x 270 nm instead of two 297 nm rows, so that it fits under the sense
amplifier in a 4:1 column group (1080 nm).

Every poly stripe is one gate through both bands (no cut at the seam), so an
nFET and a pFET share a stripe only when they share a gate net.  The nFETs
make three islands and the pFETs four, broken (92 nm, ACTIVE.S.2A) where a
stripe is one band's gate only::

    col   0    1    2    3    4    5    6    7    8    9   10   11
    p     SA   W    .    .    W    VDD  WN   SAN  .    VDD  DN   .
    n     D    W    SA   .    W    VSS  WN   DN   VSS  .    WN   SAN
    gate     ~W   W    .    .    WN   W    ~W   D    .    D    W
             (~W: WRENAN, W: WRENA; keepers are the 1-fin WN/W pair)

The keepers' one-fin island and the transmission gate's three fins meet on
column 6 (a jogged ACTIVE).  The breaks double as the seam crossings M1 can
make (a contacted gate's bar stands 27 nm from every column): ``SA`` crosses
over the dummy stripe after column 2, ``W`` over the one before column 4,
``DN`` over the one after column 8.  The rest climbs to M2 and M3; ``WRENA``
and ``WRENAN`` each join their two gates on a short M4 line inside the cell.
Contacts sit on the via rows nearest the rails (41 nm in from each), which
leaves two M2 tracks a gate bar can reach and two more beside the via rows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from ..layout.grid import FIN_WIDTH, GATE_PITCH
from ..layout.layers import box, require_gdspy
from .cli import parse_spec, write_gds
from .row import RowBand, RowStack
from .rowcell import (
    CAP,
    HALF,
    HALF_SD,
    PAD,
    draw_frame,
    draw_rails,
    gate_contact,
    island,
    label,
    m4_track,
    square,
    supply_contact,
)
from .rowcell import V3_M3_CAP as M3_PAST_V3
from .write_driver import WRITE_DRIVER_PINS, WriteDriverSpec

__all__ = ["WriteDriver270Spec", "build_write_driver_270", "render_write_driver_270_lvs_schematic"]

ROW = 270
COLUMNS = 12
#: M2 tracks: the via rows, one between each and the gate tracks, and the two gate tracks.
Y_N, Y_77, Y_GATE_LO, Y_GATE_HI, Y_185, Y_P = 41, 77, 113, 149, 185, 229
#: M4 lines of the two write enables.  Half a nanometre off the cell's grid
#: so that, mirrored on a column IO's half-nanometre fin-grid row, they land
#: on whole nanometres: a router's on-grid patch over one then leaves no
#: half-nanometre step (M4.AUX.3).
Y_M4 = {"WRENA": 29.5, "WRENAN": 240.5}
#: The control pins are M3 a router lands on from M4 tracks (M4 is
#: right-way and on-grid only): each runs past its own net's M4 line, over
#: rows a block keeps free.
TOP_OF_WRENA = 180


def col(i: int) -> int:
    return (i + 1) * GATE_PITCH


def gate(i: int) -> int:
    """The stripe between columns `i` and `i + 1`."""
    return col(i) + GATE_PITCH // 2


#: ``(gate index, net, M2 track of its V1)``: every contacted stripe.
GATES = (
    (0, "WRENAN", Y_GATE_HI),
    (1, "WRENA", Y_GATE_LO),
    (4, "WN", Y_GATE_LO),
    (5, "W", Y_GATE_HI),
    (6, "WRENAN", Y_GATE_LO),
    (7, "D", Y_GATE_HI),
    (9, "D", Y_GATE_HI),
    (10, "WRENA", Y_GATE_LO),
)
#: M3 columns, by net (x, y0, y1); a net's pin is its first.
M3 = {
    "SA": [(col(0), 0, Y_P)],  # to the edge: a sense amplifier's SA track above takes it
    "W": [(col(1), Y_N, Y_P)],
    "WRENAN": [(col(2), Y_GATE_HI, Y_M4["WRENAN"]), (gate(6), Y_GATE_LO, Y_M4["WRENAN"])],
    "WRENA": [(col(3), Y_M4["WRENA"], TOP_OF_WRENA), (472, Y_M4["WRENA"], Y_GATE_LO)],
    "WN": [(col(5), Y_N, Y_P), (col(10), Y_N, Y_185)],
    "D": [(gate(8), Y_77, 228)],
    "SAN": [(col(11), 0, Y_P)],
}
PIN_NETS = ("D", "WRENA", "WRENAN", "SA", "SAN")


@dataclass(frozen=True)
class WriteDriver270Spec:
    """`WriteDriverSpec` (3-fin nFETs and transmission gates, 1-fin keepers) on one 270 nm row."""

    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"

    @property
    def reference(self) -> WriteDriverSpec:
        """The two-row cell with the same devices: its netlist is this one's."""
        return WriteDriverSpec(n_fins=3, p_fins=3, keeper_fins=1, vt=self.vt)

    @property
    def stack(self) -> RowStack:
        return RowStack(rows=((3, 3),), vt=self.vt, band_height=135)

    @property
    def bands(self) -> dict[str, RowBand]:
        n, p = self.stack.bands()
        return {"n": n, "p": p, "k": replace(p, spec=replace(p.spec, fins=1))}

    @property
    def width(self) -> int:
        return (COLUMNS + 1) * GATE_PITCH

    @property
    def height(self) -> int:
        return ROW

    @property
    def rails(self) -> tuple[tuple[int, str], ...]:
        return self.stack.rails

    @property
    def cell_name(self) -> str:
        return f"wrdrv270{'' if self.vt == 'rvt' else '_' + self.vt}"

    @property
    def track_x(self) -> dict[str, int]:
        """x of each pin's M3."""
        return {net: M3[net][0][0] for net in PIN_NETS}

    @property
    def pin_positions(self) -> dict[str, tuple[str, tuple[float, float]]]:
        pins: dict[str, tuple[str, tuple[float, float]]] = {}
        for net in PIN_NETS:
            x, y0, y1 = M3[net][0]
            pins[net] = ("M3", (x, (y0 + y1) / 2))

        pins["VSS"] = ("M1", (self.width / 2, 0))
        pins["VDD"] = ("M1", (self.width / 2, ROW))
        return pins

    @property
    def devices(self) -> tuple[tuple[str, str, str, str, str, int], ...]:
        return self.reference.devices

    def netlist(self, name: str | None = None) -> str:
        return self.reference.netlist(name or self.cell_name)


def render_write_driver_270_lvs_schematic(spec: WriteDriver270Spec, *, cell_name: str | None = None) -> str:
    """One unit-fin MOS per FIN x GATE channel (every node is contacted)."""
    name = cell_name or spec.cell_name
    bands = spec.bands
    lines = []
    for device, d, g, s, flavor, fins in spec.devices:
        band = bands["n"] if flavor == "n" else bands["p"]
        bulk = "VSS" if flavor == "n" else "VDD"
        for fin in range(fins):
            lines.append(f"{device}_f{fin} {d} {g} {s} {bulk} {band.spec.model} "
                         f"L={band.spec.gate_length}n W={FIN_WIDTH}n")  # fmt: skip
    body = "\n".join(lines)
    return (
        "* ASAP7 270 nm write driver LVS reference: one MOS per FIN x GATE channel\n"
        f".SUBCKT {name} {' '.join(WRITE_DRIVER_PINS)}\n{body}\n.ENDS {name}\n.END\n"
    )


def build_write_driver_270(
    spec: WriteDriver270Spec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Draw `spec` and return the gdspy Cell (nanometre coordinates)."""
    spec = spec or WriteDriver270Spec()
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = lib.new_cell(cell_name) if lib is not None else gdspy.Cell(cell_name, exclude_from_current=True)
    bands, stack, width = spec.bands, spec.stack, spec.width
    n, p, k = bands["n"], bands["p"], bands["k"]
    seam = stack.seam_y(0)

    draw_frame(cell, stack, width, spec.vt)
    for cols in ((0, 1, 2), (4, 5, 6, 7, 8), (10, 11)):
        island(cell, n, [col(i) for i in cols])
    island(cell, p, [col(0), col(1)])
    island(cell, k, [col(4), col(5), col(6)])  # the keepers, one fin
    island(cell, p, [col(6), col(7)])  # MPTWN, three fins from the keepers' WN on
    island(cell, p, [col(9), col(10)])
    draw_rails(cell, stack, width)
    for i in (5, 8):
        supply_contact(cell, n, col(i))
    supply_contact(cell, k, col(5))
    supply_contact(cell, p, col(9))
    # The keepers' W column: its bar carried up to the via row.
    k_lo, _ = k.active_span
    box(cell, "LISD", col(4) - HALF_SD, k_lo, col(4) + HALF_SD, Y_P + CAP)

    def pad(x: float, y: float, reach: float | None = None) -> None:
        """A drain contact: V0 under an M1 pad along the column (to `reach`)."""
        square(cell, "V0", x, y)
        ends = (y - CAP, y + CAP, *(() if reach is None else (reach,)))
        box(cell, "M1", x - HALF, min(ends), x + HALF, max(ends))

    def flag(y: float, x0: float, x1: float) -> None:
        box(cell, "M1", min(x0, x1) - CAP, y - HALF, max(x0, x1) + CAP, y + HALF)

    def crossing(x: float, y0: float, y1: float) -> None:
        box(cell, "M1", x - HALF, y0 - HALF, x + HALF, y1 + HALF)

    # A contact under a flag is the flag's alone: a V0's M1 is exactly as
    # wide as the via across the wire (V0.M1.AUX.3), so no pad along the column.
    flagged = {Y_N: (2, 4, 6, 7), Y_P: (1, 4, 6, 10)}
    for y, columns in ((Y_N, (0, 1, 2, 4, 6, 7, 10, 11)), (Y_P, (0, 1, 4, 6, 7, 10))):
        for i in columns:
            if i in flagged[y]:
                square(cell, "V0", col(i), y)
            else:
                pad(col(i), y, reach=Y_77 + CAP if (i, y) == (0, Y_N) else None)  # D climbs to its track

    # M1: W over the dummy before column 4 (p: columns 1 to 4), SA after
    # column 2 (up to its track), DN after column 8; WN's pads to its M3.
    flag(Y_P, col(1), col(4))
    crossing(gate(3), Y_N, Y_P)
    flag(Y_N, gate(3), col(4))
    flag(Y_N, col(2), gate(2))
    crossing(gate(2), Y_N, Y_185 + CAP - HALF)  # under its V1
    flag(Y_N, col(7), gate(8))
    crossing(gate(8), Y_N, Y_P)
    flag(Y_P, gate(8), col(10))
    for y in (Y_N, Y_P):
        flag(y, col(5), col(6))

    # Gates: one contact each on the seam, the bar on to its track.
    for i, _, y in GATES:
        gate_contact(cell, seam, [gate(i)], gate(i), y + (CAP if y > seam else -CAP))

    def via1(x: float, y: float) -> None:
        square(cell, "V1", x, y)

    def m2(y: float, x0: float, x1: float) -> None:
        box(cell, "M2", min(x0, x1) - PAD, y - HALF, max(x0, x1) + PAD, y + HALF)

    def standing(x: float, y: float) -> None:
        """A V1 under an M2 landing turned along the column (beside a neighbour's)."""
        via1(x, y)
        box(cell, "M2", x - HALF, y - PAD, x + HALF, y + PAD)

    # V1s and M2, net by net.
    for i, net, y in GATES:
        via1(gate(i), y)
    m2(Y_GATE_HI, gate(0), col(2))  # WRENAN: gate 0 to its M3
    m2(Y_GATE_LO, gate(6), gate(6))  # WRENAN: gate 6 under its M3
    m2(Y_GATE_LO, gate(1), col(3))  # WRENA: gate 1 to its M3
    m2(Y_GATE_LO, 472, gate(10))  # WRENA: gate 10 to its M3
    m2(Y_GATE_LO, gate(4), col(5))  # WN: its gate to its M3
    via1(gate(3), Y_GATE_HI)
    m2(Y_GATE_HI, gate(3), gate(5))  # W: the crossing to its gate
    m2(Y_GATE_HI, gate(7), gate(9))  # D: both gates
    via1(col(0), Y_77)
    m2(Y_77, col(0), gate(8))  # D: the pass gate's column to the gates' M3
    via1(gate(2), Y_185)
    m2(Y_185, col(0), gate(2))  # SA: the crossing to its M3
    m2(Y_185, col(5), col(10))  # WN: the transmission gate's nFET
    for x in (col(1), col(5), col(10)):
        via1(x, Y_N)
        m2(Y_N, x, x)
    for x in (col(1), col(5)):
        via1(x, Y_P)
        m2(Y_P, x, x)
    standing(col(0), Y_P)  # SA
    standing(col(11), Y_N)  # SAN
    via1(col(7), Y_P)
    m2(Y_P, col(7), col(11))  # SAN: the transmission gate's pFET to its M3

    # M3 and V2s, M4 and V3s.
    v2 = {
        "SA": [(col(0), Y_185), (col(0), Y_P)],
        "W": [(col(1), Y_N), (col(1), Y_P)],
        "WRENAN": [(col(2), Y_GATE_HI), (gate(6), Y_GATE_LO)],
        "WRENA": [(col(3), Y_GATE_LO), (472, Y_GATE_LO)],
        "WN": [(col(5), Y_N), (col(5), Y_GATE_LO), (col(5), Y_185), (col(5), Y_P), (col(10), Y_N), (col(10), Y_185)],
        "D": [(gate(8), Y_77), (gate(8), Y_GATE_HI)],
        "SAN": [(col(11), Y_N), (col(11), Y_P)],
    }
    for net, columns in M3.items():
        for x, y0, y1 in columns:
            lo, hi = min(y0, y1), max(y0, y1)
            on_m4 = net in Y_M4
            box(cell, "M3", x - HALF, max(0, lo - (M3_PAST_V3 if on_m4 and lo == Y_M4.get(net) else PAD)),
                x + HALF, min(ROW, hi + (M3_PAST_V3 if on_m4 and hi == Y_M4.get(net) else PAD)))  # fmt: skip
        for x, y in v2.get(net, []):
            square(cell, "V2", x, y)
    for net, y in Y_M4.items():
        m4_track(cell, y, [x for x, _, _ in M3[net]])

    if draw_pin_labels:
        for pin, (metal, origin) in spec.pin_positions.items():
            label(cell, pin, metal, origin)
    box(cell, "BOUNDARY", 0, 0, width, ROW)
    return cell


def main(argv: list[str] | None = None) -> None:
    """Write the one-row write driver: ``asap7-write-driver-270``."""
    args = parse_spec(WriteDriver270Spec, argv, description="Generate the write driver on one 270 nm row.")
    cell, out = write_gds(lambda lib: build_write_driver_270(args.spec, lib=lib), args.out)
    print(f"wrote {cell.name} to {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
