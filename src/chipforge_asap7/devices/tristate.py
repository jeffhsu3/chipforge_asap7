"""A tristate inverter on the 270 nm (7.5-track) standard-cell row.

``Q = !A`` while ``EN`` is high (``ENB`` low), floating otherwise: a series
nFET pair ``VSS -A- n2 -EN- Q`` and a series pFET pair ``VDD -A- N1 -ENB- Q``.
The two output-side gates carry different signals (EN on the nFET, ENB on
the pFET), so they cannot share one poly line the way a NAND's inputs do.
Instead of cutting the gate at the seam, each gets its own track, with only
its own band's diffusion under it::

    y=270 ─────────────────────── vdd rail ───────────────────────
            p:  VDD  A   N1   ·    ·   N1  ENB   Q              (broken: 92 nm)
    y=135 ─ ─ ─ ─ ─ ─ ─ ─ seam: A, EN, ENB contacts ─ ─ ─ ─ ─ ─ ─ ─
            n:  VSS  A   ·   EN   Q                              (ends after EN)
    y=  0 ─────────────────────── vss rail ───────────────────────
             x:  54  81 108 135  162 189  216 243  270    297 (Q bar)

The p band breaks under EN and the empty track after it: diffusion of two
transistors must stand 92 nm apart (ACTIVE.S.2A).  Its middle node ``N1`` is
two contacted columns joined on its via row; the n band's (``·``) is
uncontacted diffusion.  ``Q`` joins the n drain and the p drain with a
vertical M1 bar over the right dummy track.  324 x 270 nm, self-contained
(dummy gates at both edges): `RowSupportSpec` taps and fillers abut it.
Each drain contact is a V0 under one horizontal M1 bar, exactly as wide as
the via across it (V0.M1.AUX.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..layout.grid import FIN_WIDTH, GATE_PITCH
from ..layout.layers import PIN_LAYERS, box, require_gdspy
from .cli import parse_spec, write_gds
from .row import RowBand, RowStack
from .rowcell import CAP, HALF, draw_frame, draw_rails, gate_contact, island, square, supply_contact, via_y

__all__ = ["TRISTATE_PINS", "TristateSpec", "build_tristate", "render_tristate_lvs_schematic"]

TRISTATE_PINS = ("A", "EN", "ENB", "Q", "VDD", "VSS")
ROW = 270
#: Source/drain columns and gate tracks (see the module docstring).
COL = {"rail": 54, "mid": 108, "out_n": 162, "mid_p": 216, "out_p": 270}
GATE = {"A": 81, "EN": 135, "ENB": 243}
Q_BAR = 297
#: A gate bar runs this far above its seam V0 (18 x 41 nm, clear of M1.A.1).
_BAR_ABOVE = 27
_M1_PIN_LAYER, _M1_PIN_TEXTTYPE = PIN_LAYERS["M1"]


@dataclass(frozen=True)
class TristateSpec:
    """One tristate inverter: `fins` per device, on the 270 nm row."""

    fins: int = 3
    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"

    def __post_init__(self) -> None:
        if not 1 <= self.fins <= 3:
            raise ValueError(f"a 135 nm band carries 1 to 3 fins, got {self.fins}")

    @property
    def stack(self) -> RowStack:
        return RowStack(rows=((self.fins, self.fins),), vt=self.vt, band_height=135)

    @property
    def bands(self) -> tuple[RowBand, RowBand]:
        n_band, p_band = self.stack.bands()
        return n_band, p_band

    @property
    def width(self) -> int:
        return 6 * GATE_PITCH

    @property
    def height(self) -> int:
        return ROW

    @property
    def seam_y(self) -> int:
        return self.stack.seam_y(0)

    @property
    def rails(self) -> tuple[tuple[int, str], ...]:
        return self.stack.rails

    @property
    def cell_name(self) -> str:
        vt = "" if self.vt == "rvt" else f"_{self.vt}"
        return f"tristate_inv_{self.fins}n{self.fins}p{vt}"

    @property
    def pin_positions(self) -> dict[str, tuple[float, float]]:
        seam = self.seam_y
        rails = {net: y for y, net in self.rails}
        return {
            "A": (GATE["A"], seam),
            "EN": (GATE["EN"], seam),
            "ENB": (GATE["ENB"], seam),
            "Q": (Q_BAR, seam),
            "VDD": (self.width / 2, rails["VDD"]),
            "VSS": (self.width / 2, rails["VSS"]),
        }

    @property
    def devices(self) -> tuple[tuple[str, str, str, str, str, int], ...]:
        """``(name, drain, gate, source, flavor, fins)``."""
        f = self.fins
        return (
            ("MNA", "n2", "A", "VSS", "n", f),
            ("MNE", "Q", "EN", "n2", "n", f),
            ("MPA", "N1", "A", "VDD", "p", f),
            ("MPE", "Q", "ENB", "N1", "p", f),
        )

    def netlist(self, name: str | None = None) -> str:
        title = name or self.cell_name
        n_band, p_band = self.bands
        lines = ["* ASAP7 tristate inverter, 270 nm row", f".SUBCKT {title} {' '.join(TRISTATE_PINS)}"]
        for device, d, g, s, flavor, fins in self.devices:
            band = n_band if flavor == "n" else p_band
            bulk = "VSS" if flavor == "n" else "VDD"
            lines.append(f"{device} {d} {g} {s} {bulk} {band.spec.model} nfin={fins} l={band.spec.gate_length}n nf=1 m=1")
        lines += [f".ENDS {title}", ".END", ""]
        return "\n".join(lines)


def render_tristate_lvs_schematic(spec: TristateSpec, *, cell_name: str | None = None) -> str:
    """One unit-fin MOS per FIN x GATE channel; the n series node is one net per fin (uncontacted)."""
    name = cell_name or spec.cell_name
    n_band, p_band = spec.bands
    lines = []
    for device, d, g, s, flavor, fins in spec.devices:
        band = n_band if flavor == "n" else p_band
        bulk = "VSS" if flavor == "n" else "VDD"
        for fin in range(fins):
            nets = [f"n2_f{fin}" if net == "n2" else net for net in (d, g, s)]
            lines.append(f"{device}_f{fin} {' '.join(nets)} {bulk} {band.spec.model} "
                         f"L={band.spec.gate_length}n W={FIN_WIDTH}n")  # fmt: skip
    body = "\n".join(lines)
    return (
        "* ASAP7 tristate inverter LVS reference: one MOS per FIN x GATE channel\n"
        f".SUBCKT {name} {' '.join(TRISTATE_PINS)}\n{body}\n.ENDS {name}\n.END\n"
    )


def build_tristate(
    spec: TristateSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Draw `spec` and return the gdspy Cell (nanometre coordinates)."""
    spec = spec or TristateSpec()
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = lib.new_cell(cell_name) if lib is not None else gdspy.Cell(cell_name, exclude_from_current=True)
    n_band, p_band = spec.bands
    stack, width, seam = spec.stack, spec.width, spec.seam_y

    draw_frame(cell, stack, width, spec.vt)
    # n: one island under A and EN, its middle node uncontacted.  p: two
    # islands, broken under EN, so that gate drives the nFET alone.
    island(cell, n_band, [COL["rail"], COL["mid"], COL["out_n"]], contacted=[COL["rail"], COL["out_n"]])
    island(cell, p_band, [COL["rail"], COL["mid"]])
    island(cell, p_band, [COL["mid_p"], COL["out_p"]])

    draw_rails(cell, stack, width)
    supply_contact(cell, n_band, COL["rail"])
    supply_contact(cell, p_band, COL["rail"])

    y_n, y_p = via_y(n_band), via_y(p_band)

    def flag(y: float, x_via: float, x0: float, x1: float, vias: tuple[float, ...] = ()) -> None:
        """Horizontal M1 from `x0` to `x1` at `y`, with a V0 at `x_via` (and each of `vias`)."""
        for x in (x_via, *vias):
            square(cell, "V0", x, y)
        box(cell, "M1", x0, y - HALF, x1, y + HALF)

    # N1: the p band's two middle columns, one bar on its via row.
    flag(y_p, COL["mid"], COL["mid"] - CAP, COL["mid_p"] + CAP, vias=(COL["mid_p"],))
    # Q: the n drain's and the p drain's bars, joined by one over the right
    # dummy track (36 nm clear of the ENB gate's bar).
    flag(y_n, COL["out_n"], COL["out_n"] - CAP, Q_BAR + HALF)
    flag(y_p, COL["out_p"], COL["out_p"] - CAP, Q_BAR + HALF)
    box(cell, "M1", Q_BAR - HALF, y_n - HALF, Q_BAR + HALF, y_p + HALF)

    # Inputs: one contact per gate on the seam, each its own M1 bar.
    for x in GATE.values():
        gate_contact(cell, seam, [x], x, seam + _BAR_ABOVE)

    if draw_pin_labels:
        for pin, origin in spec.pin_positions.items():
            cell.add(gdspy.Label(pin, origin, layer=_M1_PIN_LAYER, texttype=_M1_PIN_TEXTTYPE))
    box(cell, "BOUNDARY", 0, 0, width, spec.height)
    return cell


def main(argv: list[str] | None = None) -> None:
    """Write a tristate-inverter GDS: ``asap7-tristate --fins 3``."""
    args = parse_spec(TristateSpec, argv, description="Generate an ASAP7 tristate inverter on the 270 nm row.")
    cell, out = write_gds(lambda lib: build_tristate(args.spec, lib=lib), args.out)
    print(f"wrote {cell.name} to {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
