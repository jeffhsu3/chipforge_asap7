"""The column IO's output latch on four 270 nm (7.5-track) rows.

`OutputLatchSpec`'s function -- an SR latch set and reset by the sense
amplifier's ``QA``/``QAN``, an inverter, and a tristate driver onto ``Q`` --
built from 270 nm row cells and stacked as the released ASAP7 bank's
``srlatch_sram_6t122`` is (432 x 1080 nm, one 4:1 column group tall)::

    row 3 (mirrored)  tristate  Q = !Y2N while OE         | tap
    row 2             inverter  Y2N = !Y2                 | filler tap
    row 1 (mirrored)  NAND2     Y2 = NAND(QAN, Y1)        | filler tap
    row 0             NAND2     Y1 = NAND(QA, Y2)         | filler tap

Rows alternate mirrored so neighbours share a rail.  Every net between rows
climbs from its cell's M1 pin on a V1, an M2 jog and a V2 to an M3 column of
its own; ``QA``/``QAN`` are M3 pins at the bottom (where the sense
amplifier's outputs arrive), ``OE``/``OEB``/``Q`` on the top.  The cells' M1
gate bars are lengthened (within their via rows' clearance) so that each pin
of a row takes its V1 at its own height, 36 nm or more from the next pin's
M2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..layout.grid import FIN_WIDTH
from ..layout.layers import PIN_LAYERS, box, require_gdspy
from .cli import parse_spec, write_gds
from .inverter import InverterSpec, build_inverter
from .nand import NandSpec, build_nand
from .row import RowBand, RowStack
from .row_support import RowSupportSpec, build_row_support
from .rowcell import CAP, HALF, PAD, square
from .tristate import TristateSpec, build_tristate

__all__ = [
    "OUTPUT_LATCH_270_PINS",
    "OutputLatch270Spec",
    "build_output_latch_270",
    "output_latch_270_units",
    "render_output_latch_270_lvs_schematic",
]

OUTPUT_LATCH_270_PINS = ("QA", "QAN", "OE", "OEB", "Q", "VDD", "VSS")
ROW = 270
#: Where the QA/QAN pins' M3 starts, above the bottom edge.
INPUT_BOTTOM = 31
WIDTH = 432  # the widest cell (the tristate, 324) and a tap
TAP_X = 324
#: M3 columns, by net.
COLUMN = {"QA": 81, "QAN": 351, "Y1": 243, "Y2": 189, "Y2N": 27, "OE": 135, "OEB": 243, "Q": 297}
#: Each pin's V1: (row, x, y in the cell's own frame).  Gate bars run 68-202
#: at the most, clear of their via rows; output bars the cell's height.
PIN_VIA = {
    ("NAND1", "B"): (0, 81, 100),   # QA
    ("NAND1", "A"): (0, 135, 170),  # Y2
    ("NAND1", "Y"): (0, 189, 100),  # Y1
    ("NAND2", "A"): (1, 135, 100),  # QAN
    ("NAND2", "B"): (1, 81, 170),   # Y1
    ("NAND2", "Y"): (1, 189, 210),  # Y2
    ("INV", "A"): (2, 135, 150),    # Y2
    ("INV", "Y"): (2, 90, 100),     # Y2N
    ("TRI", "A"): (3, 81, 100),     # Y2N
    ("TRI", "EN"): (3, 135, 170),   # OE
    ("TRI", "ENB"): (3, 243, 170),  # OEB
    ("TRI", "Q"): (3, 297, 100),    # Q
}
#: The gate bars these pins sit on (``(row, x)``): lengthened to their via.
GATE_BARS = {("NAND1", "B"), ("NAND1", "A"), ("NAND2", "A"), ("NAND2", "B"),
             ("TRI", "A"), ("TRI", "EN"), ("TRI", "ENB")}  # fmt: skip
NET = {
    ("NAND1", "B"): "QA", ("NAND1", "A"): "Y2", ("NAND1", "Y"): "Y1",
    ("NAND2", "A"): "QAN", ("NAND2", "B"): "Y1", ("NAND2", "Y"): "Y2",
    ("INV", "A"): "Y2", ("INV", "Y"): "Y2N",
    ("TRI", "A"): "Y2N", ("TRI", "EN"): "OE", ("TRI", "ENB"): "OEB", ("TRI", "Q"): "Q",
}  # fmt: skip


@dataclass(frozen=True)
class OutputLatch270Spec:
    """The output latch on four 270 nm rows (3-fin devices, the inverter two fingers)."""

    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"

    @property
    def nand(self) -> NandSpec:
        return NandSpec(rows=((3, 3),), fingers=1, vt=self.vt, abut=False, band_height=135)

    @property
    def inverter(self) -> InverterSpec:
        return InverterSpec(rows=((3, 3),), fingers=2, vt=self.vt, abut=False)

    @property
    def tristate(self) -> TristateSpec:
        return TristateSpec(fins=3, vt=self.vt)

    @property
    def stack(self) -> RowStack:
        return RowStack(rows=((3, 3),), vt=self.vt, band_height=135)

    @property
    def bands(self) -> tuple[RowBand, RowBand]:
        n_band, p_band = self.stack.bands()
        return n_band, p_band

    @property
    def width(self) -> int:
        return WIDTH

    @property
    def height(self) -> int:
        return 4 * ROW

    @property
    def cell_name(self) -> str:
        return f"output_latch270{'' if self.vt == 'rvt' else '_' + self.vt}"

    @property
    def rails(self) -> list[tuple[int, str]]:
        return [(k * ROW, "VDD" if k % 2 else "VSS") for k in range(5)]

    @property
    def track_x(self) -> dict[str, int]:
        """x of the M3 pins: ``QA``/``QAN`` at the bottom edge, ``OE``/``OEB``/``Q`` at the top."""
        return {net: COLUMN[net] for net in ("QA", "QAN", "OE", "OEB", "Q")}

    @property
    def pin_positions(self) -> dict[str, tuple[str, tuple[float, float]]]:
        tx = self.track_x
        pins: dict[str, tuple[str, tuple[float, float]]] = {net: ("M3", (tx[net], 40)) for net in ("QA", "QAN")}
        pins.update({net: ("M3", (tx[net], self.height - 40)) for net in ("OE", "OEB", "Q")})
        pins["VSS"] = ("M1", (WIDTH / 2, 0))
        pins["VDD"] = ("M1", (WIDTH / 2, ROW))
        return pins

    @property
    def devices(self) -> tuple[tuple[str, str, str, str, str, int], ...]:
        """``(name, drain, gate, source, flavor, fins)`` -- `OutputLatchSpec`'s function."""
        return (
            # Y1 = NAND(QA, Y2): QA on the rail side (B), Y2 on the output side (A).
            ("MN1A", "Y1", "Y2", "M1", "n", 3), ("MN1B", "M1", "QA", "VSS", "n", 3),
            ("MP1A", "Y1", "Y2", "VDD", "p", 3), ("MP1B", "Y1", "QA", "VDD", "p", 3),
            # Y2 = NAND(QAN, Y1): Y1 on the rail side (B), QAN on the output side (A).
            ("MN2A", "Y2", "QAN", "M2", "n", 3), ("MN2B", "M2", "Y1", "VSS", "n", 3),
            ("MP2A", "Y2", "QAN", "VDD", "p", 3), ("MP2B", "Y2", "Y1", "VDD", "p", 3),
            # Y2N = !Y2, two fingers.
            ("MNI0", "Y2N", "Y2", "VSS", "n", 3), ("MNI1", "Y2N", "Y2", "VSS", "n", 3),
            ("MPI0", "Y2N", "Y2", "VDD", "p", 3), ("MPI1", "Y2N", "Y2", "VDD", "p", 3),
            # Q = !Y2N while OE.
            ("MNA", "N2", "Y2N", "VSS", "n", 3), ("MNE", "Q", "OE", "N2", "n", 3),
            ("MPA", "N1", "Y2N", "VDD", "p", 3), ("MPE", "Q", "OEB", "N1", "p", 3),
        )  # fmt: skip

    @property
    def series_nodes(self) -> frozenset[str]:
        """Uncontacted diffusion between two channels: one net per fin in the extraction."""
        return frozenset({"M1", "M2", "N2"})

    def netlist(self, name: str | None = None) -> str:
        title = name or self.cell_name
        n_band, p_band = self.bands
        lines = ["* ASAP7 output latch on four 270 nm rows", f".SUBCKT {title} {' '.join(OUTPUT_LATCH_270_PINS)}"]
        for device, d, g, s, flavor, fins in self.devices:
            band = n_band if flavor == "n" else p_band
            bulk = "VSS" if flavor == "n" else "VDD"
            lines.append(f"{device} {d} {g} {s} {bulk} {band.spec.model} nfin={fins} l={band.spec.gate_length}n nf=1 m=1")
        lines += [f".ENDS {title}", ".END", ""]
        return "\n".join(lines)


def output_latch_270_units(spec: OutputLatch270Spec, *, rename=lambda net: net, prefix: str = "M") -> list[str]:
    """One unit-fin MOS per FIN x GATE channel; series nodes (uncontacted diffusion) are one net per fin.

    `rename` maps the latch's nets, for a parent that embeds it.
    """
    n_band, p_band = spec.bands
    lines = []
    series = {"M1", "M2", "N2"}  # N1 (the tristate's p node) is contacted
    for device, d, g, s, flavor, fins in spec.devices:
        band = n_band if flavor == "n" else p_band
        bulk = "VSS" if flavor == "n" else "VDD"
        for fin in range(fins):
            nets = [rename(f"{net}_f{fin}") if net in series else rename(net) for net in (d, g, s)]
            lines.append(f"{prefix}{device}_f{fin} {' '.join(nets)} {rename(bulk)} {band.spec.model} "
                         f"L={band.spec.gate_length}n W={FIN_WIDTH}n")  # fmt: skip
    return lines


def render_output_latch_270_lvs_schematic(spec: OutputLatch270Spec, *, cell_name: str | None = None) -> str:
    name = cell_name or spec.cell_name
    body = "\n".join(output_latch_270_units(spec))
    return (
        "* ASAP7 270 nm output latch LVS reference: one MOS per FIN x GATE channel\n"
        f".SUBCKT {name} {' '.join(OUTPUT_LATCH_270_PINS)}\n{body}\n.ENDS {name}\n.END\n"
    )


def build_output_latch_270(
    spec: OutputLatch270Spec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Place the four rows, lengthen the gate bars, route every net on M2/M3; return the gdspy Cell."""
    spec = spec or OutputLatch270Spec()
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = lib.new_cell(cell_name) if lib is not None else gdspy.Cell(cell_name, exclude_from_current=True)

    def reuse(name_: str, build):
        existing = lib.cells.get(name_) if lib is not None else None
        return existing if existing is not None else build()

    nand = reuse(spec.nand.cell_name, lambda: build_nand(spec.nand, lib=lib, draw_pin_labels=False))
    inv = reuse(spec.inverter.cell_name, lambda: build_inverter(spec.inverter, lib=lib, draw_pin_labels=False))
    tri = reuse(spec.tristate.cell_name, lambda: build_tristate(spec.tristate, lib=lib, draw_pin_labels=False))
    support = {kind: RowSupportSpec(stack=spec.stack, kind=kind) for kind in ("filler", "tap")}
    filler, tap = (reuse(s.cell_name, lambda s=s: build_row_support(s, lib=lib)) for s in support.values())

    def y_of(row: int, y_local: float) -> float:
        """A cell-frame y in the latch, on a row placed as drawn (even) or mirrored (odd)."""
        return row * ROW + (ROW - y_local if row % 2 else y_local)

    rows = [nand, nand, inv, tri]
    for row, placed in enumerate(rows):
        mirrored = bool(row % 2)
        origin_y = (row + 1) * ROW if mirrored else row * ROW
        cell.add(gdspy.CellReference(placed, origin=(0, origin_y), x_reflection=mirrored))
        x = spec.nand.width if placed is nand else spec.inverter.width if placed is inv else spec.tristate.width
        while x < TAP_X:
            cell.add(gdspy.CellReference(filler, origin=(x, origin_y), x_reflection=mirrored))
            x += 108
        cell.add(gdspy.CellReference(tap, origin=(TAP_X, origin_y), x_reflection=mirrored))

    # Pins: a gate bar lengthened to its via, a V1, the M2 jog to the net's column, a V2.
    ends: dict[str, list[float]] = {}
    for (inst, pin), (row, x, y_local) in PIN_VIA.items():
        net, y = NET[(inst, pin)], y_of(row, y_local)
        if (inst, pin) in GATE_BARS:
            seam = y_of(row, 135)
            box(cell, "M1", x - HALF, min(y - CAP, seam - CAP), x + HALF, max(y + CAP, seam + CAP))
        column = COLUMN[net]
        square(cell, "V1", x, y)
        square(cell, "V2", column, y)
        box(cell, "M2", min(x, column) - PAD, y - HALF, max(x, column) + PAD, y + HALF)
        ends.setdefault(net, []).append(y)
    # The M3 columns: between their vias, and out to the pins at the edges.
    for net, ys in ends.items():
        lo, hi = min(ys), max(ys)
        x = COLUMN[net]
        if net in ("QA", "QAN"):
            # Down to just inside the bottom edge: a cell abutting below may
            # bring its own M3 up to its top (M3.S.6, corner to corner).
            box(cell, "M3", x - HALF, INPUT_BOTTOM, x + HALF, hi + CAP)
            continue
        if net in ("OE", "OEB", "Q"):
            hi = spec.height
        box(cell, "M3", x - HALF, lo - CAP, x + HALF, hi + (CAP if hi < spec.height else 0))

    if draw_pin_labels:
        for pin, (metal, origin) in spec.pin_positions.items():
            layer, texttype = PIN_LAYERS[metal]
            cell.add(gdspy.Label(pin, origin, layer=layer, texttype=texttype))
    box(cell, "BOUNDARY", 0, 0, WIDTH, spec.height)
    return cell


def main(argv: list[str] | None = None) -> None:
    """Write the 270 nm output latch: ``asap7-output-latch-270``."""
    args = parse_spec(OutputLatch270Spec, argv, description="Generate the column IO's output latch on four 270 nm rows.")
    cell, out = write_gds(lambda lib: build_output_latch_270(args.spec, lib=lib), args.out)
    print(f"wrote {cell.name} to {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
