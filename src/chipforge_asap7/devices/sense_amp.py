"""Parametric differential latch sense amplifier for ASAP7.

The circuit is OpenFinRAM's 16-transistor ``sense_amp_sram``: a symmetric
regenerative latch with differential ``SA/SAN`` inputs, active-high ``SAE``
evaluation, active-low ``SAPRECHN`` precharge, and complementary ``QA/QAN``
outputs.  The electrical topology is kept here as the single source of truth
for layout, SPICE, and LVS.

This first physical implementation deliberately favors inspectability and
verification over density.  Each logical transistor is a standalone,
DRC-clean ASAP7 FinFET tile.  Tiles are placed in matched order, their fins
remain on one continuous manufacturing grid, and M1/V1/M2/V2/M3 escape routing
connects every terminal to one named M2 net track.  A later compaction pass can
merge diffusion without changing :class:`SenseAmpSpec` or the reference
topology.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..layout.grid import FIN_PITCH, FIN_WIDTH, GATE_PITCH
from ..layout.layers import PIN_LAYERS, box, require_gdspy
from ..layout.rules import CONTACT_SIZE, TRACK_PITCH
from .finfet import ACTIVE_ENC, FinFETSpec
from .finfet import build_finfet as build_leaf_finfet

__all__ = [
    "DEFAULT_N_FINS",
    "DEFAULT_P_FINS",
    "SENSE_AMP_PINS",
    "SenseAmpPlacement",
    "SenseAmpSpec",
    "SenseAmpTransistor",
    "build_sense_amp",
    "sense_amp_transistors",
]

DEFAULT_N_FINS = 12
DEFAULT_P_FINS = 4

ROUTE_WIDTH = CONTACT_SIZE  # wires as wide as the V1/V2 landing on them
ROUTE_PITCH = TRACK_PITCH
ROUTE_ENDCAP = ROUTE_WIDTH // 2 + 5  # V1.M2.EN.2 / V2.M3.EN.2
DEVICE_SLOT_PITCH = 8 * GATE_PITCH
DEVICE_LEFT_MARGIN = GATE_PITCH
PIN_ACCESS_LENGTH = 2 * ROUTE_PITCH

SENSE_AMP_PINS = (
    "SA",
    "SAN",
    "SAE",
    "SAPRECHN",
    "QA",
    "QAN",
    "VDD",
    "VSS",
)

# Horizontal M2 track order.  Internal nodes intentionally have no pin labels.
_NET_ORDER = (
    "VSS",
    "VDD",
    "SA",
    "SAN",
    "SAE",
    "SAPRECHN",
    "QA",
    "QAN",
    "N52",
    "N57",
    "N58",
    "N59",
)

# Within each 8-CPP slot, terminal M2 escapes run to distinct M3 columns.  B
# and G share a Y coordinate in the leaf device, so they escape in opposite
# directions and cannot overlap.
_ESCAPE_X = {
    "B": 0,
    "S": 2 * GATE_PITCH,
    "G": 6 * GATE_PITCH,
    "D": 7 * GATE_PITCH,
}

# Each transistor pair is equidistant from the center of its polarity bank.
_PLACEMENT_ORDER = (
    "P14",
    "P0",
    "P1",
    "P4",
    "P5",
    "P2",
    "P3",
    "P15",
    "N12",
    "N6",
    "N8",
    "N10",
    "N11",
    "N9",
    "N7",
    "N13",
)

_M2_PIN_LAYER, _M2_PIN_TEXTTYPE = PIN_LAYERS["M2"]


@dataclass(frozen=True)
class SenseAmpTransistor:
    """One logical transistor in the sense-amplifier topology."""

    name: str
    flavor: Literal["n", "p"]
    drain: str
    gate: str
    source: str
    bulk: str

    @property
    def terminals(self) -> dict[str, str]:
        """Map leaf FinFET terminal names to circuit nets."""

        return {
            "D": self.drain,
            "G": self.gate,
            "S": self.source,
            "B": self.bulk,
        }


_TRANSISTORS = (
    SenseAmpTransistor("P0", "p", "QAN", "SAPRECHN", "VDD", "VDD"),
    SenseAmpTransistor("P1", "p", "N59", "SAPRECHN", "QAN", "VDD"),
    SenseAmpTransistor("P2", "p", "QA", "SAPRECHN", "N59", "VDD"),
    SenseAmpTransistor("P3", "p", "VDD", "SAPRECHN", "QA", "VDD"),
    SenseAmpTransistor("P4", "p", "VDD", "QA", "QAN", "VDD"),
    SenseAmpTransistor("P5", "p", "QA", "QAN", "VDD", "VDD"),
    SenseAmpTransistor("N6", "n", "N52", "SA", "N57", "VSS"),
    SenseAmpTransistor("N7", "n", "N58", "SAN", "N52", "VSS"),
    SenseAmpTransistor("N8", "n", "N57", "QA", "QAN", "VSS"),
    SenseAmpTransistor("N9", "n", "QA", "QAN", "N58", "VSS"),
    SenseAmpTransistor("N10", "n", "VSS", "SAE", "N52", "VSS"),
    SenseAmpTransistor("N11", "n", "N52", "SAE", "VSS", "VSS"),
    SenseAmpTransistor("N12", "n", "QAN", "VSS", "VSS", "VSS"),
    SenseAmpTransistor("N13", "n", "VSS", "VSS", "QA", "VSS"),
    SenseAmpTransistor("P14", "p", "QAN", "VDD", "VDD", "VDD"),
    SenseAmpTransistor("P15", "p", "VDD", "VDD", "QA", "VDD"),
)


def sense_amp_transistors() -> tuple[SenseAmpTransistor, ...]:
    """Return the canonical OpenFinRAM 16-transistor topology."""

    return _TRANSISTORS


@dataclass(frozen=True)
class SenseAmpPlacement:
    """A logical transistor and its leaf-cell origin in nanometres."""

    transistor: SenseAmpTransistor
    origin: tuple[int, int]


@dataclass(frozen=True)
class SenseAmpSpec:
    """Sizing and threshold parameters for the differential sense amplifier.

    Args:
        n_fins: fins on each of the eight nFETs.
        p_fins: fins on each of the eight pFETs.
        vt: ASAP7 threshold flavor shared by the matched devices.
    """

    n_fins: int = DEFAULT_N_FINS
    p_fins: int = DEFAULT_P_FINS
    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"

    def __post_init__(self) -> None:
        # FinFETSpec owns the legal ASAP7 device grid and produces consistent
        # TypeError/ValueError messages for both polarities.
        FinFETSpec(flavor="n", fins=self.n_fins, vt=self.vt)
        FinFETSpec(flavor="p", fins=self.p_fins, vt=self.vt)

    @property
    def cell_name(self) -> str:
        vt_tag = "" if self.vt == "rvt" else f"_{self.vt}"
        return f"sense_amp_sram{vt_tag}_n{self.n_fins}_p{self.p_fins}"

    @property
    def nmos(self) -> FinFETSpec:
        return FinFETSpec(flavor="n", fins=self.n_fins, vt=self.vt)

    @property
    def pmos(self) -> FinFETSpec:
        return FinFETSpec(flavor="p", fins=self.p_fins, vt=self.vt)

    def device_spec(self, flavor: Literal["n", "p"]) -> FinFETSpec:
        if flavor == "n":
            return self.nmos
        if flavor == "p":
            return self.pmos
        raise ValueError(f"flavor must be 'n' or 'p', got {flavor!r}")

    @property
    def placements(self) -> tuple[SenseAmpPlacement, ...]:
        by_name = {device.name: device for device in _TRANSISTORS}
        return tuple(
            SenseAmpPlacement(
                transistor=by_name[name],
                origin=(DEVICE_LEFT_MARGIN + index * DEVICE_SLOT_PITCH, 0),
            )
            for index, name in enumerate(_PLACEMENT_ORDER)
        )

    @property
    def width(self) -> int:
        last_escape = (
            DEVICE_LEFT_MARGIN
            + (len(_PLACEMENT_ORDER) - 1) * DEVICE_SLOT_PITCH
            + _ESCAPE_X["D"]
        )
        raw_width = last_escape + 4 * ROUTE_WIDTH
        return ((raw_width + GATE_PITCH - 1) // GATE_PITCH) * GATE_PITCH

    @property
    def route_track_ys(self) -> dict[str, int]:
        first = max(self.nmos.height, self.pmos.height) + 2 * ROUTE_PITCH
        return {
            net: first + index * ROUTE_PITCH for index, net in enumerate(_NET_ORDER)
        }

    @property
    def height(self) -> int:
        raw_height = max(self.route_track_ys.values()) + 2 * ROUTE_PITCH
        return ((raw_height + FIN_PITCH - 1) // FIN_PITCH) * FIN_PITCH

    @property
    def fin_grid_ys(self) -> list[int]:
        """Full horizontal manufacturing-fin grid for the macro boundary."""

        return [
            ACTIVE_ENC + index * FIN_PITCH for index in range(self.height // FIN_PITCH)
        ]

    @property
    def pin_positions(self) -> dict[str, tuple[float, float]]:
        """Top-level M2 pin-label positions."""

        return {
            pin: (PIN_ACCESS_LENGTH / 2, self.route_track_ys[pin])
            for pin in SENSE_AMP_PINS
        }


def _centered_box(cell: Any, layer_name: str, x: float, y: float) -> None:
    half = ROUTE_WIDTH / 2
    box(cell, layer_name, x - half, y - half, x + half, y + half)


def build_sense_amp(
    spec: SenseAmpSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
) -> Any:
    """Build a routed hierarchical ASAP7 sense-amplifier GDS cell."""

    spec = spec or SenseAmpSpec()
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = (
        lib.new_cell(cell_name)
        if lib is not None
        else gdspy.Cell(cell_name, exclude_from_current=True)
    )

    leaf_cells = {}
    for flavor, device_spec in (("p", spec.pmos), ("n", spec.nmos)):
        leaf_cells[flavor] = build_leaf_finfet(
            device_spec,
            name=f"{cell_name}__{flavor}fet",
            lib=lib,
            draw_pin_labels=False,
        )

    for placement in spec.placements:
        cell.add(
            gdspy.CellReference(
                leaf_cells[placement.transistor.flavor],
                origin=placement.origin,
            )
        )

    # Fill the spaces between leaf tiles so the top-level macro carries the
    # same continuous FIN mask grid as an abutted ASAP7 row.
    for y_fin in spec.fin_grid_ys:
        box(cell, "FIN", 0, y_fin, spec.width, y_fin + FIN_WIDTH)

    # Join the eight pFET wells into one symmetric polarity bank.  The nFET
    # bodies share the continuous substrate supplied by BOUNDARY.
    p_placements = [p for p in spec.placements if p.transistor.flavor == "p"]
    p_x0 = min(p.origin[0] for p in p_placements)
    p_x1 = max(p.origin[0] for p in p_placements) + spec.pmos.width
    box(cell, "NWELL", p_x0, 0, p_x1, spec.pmos.height)

    # One horizontal M2 track per electrical net.  Only the eight interface
    # nets receive pin shapes and labels; N52/N57/N58/N59 remain internal.
    for net, y_track in spec.route_track_ys.items():
        half = ROUTE_WIDTH / 2
        box(cell, "M2", 0, y_track - half, spec.width, y_track + half)
        if net in SENSE_AMP_PINS:
            box(
                cell,
                "M2_PIN",
                0,
                y_track - half,
                PIN_ACCESS_LENGTH,
                y_track + half,
            )
            cell.add(
                gdspy.Label(
                    net,
                    spec.pin_positions[net],
                    layer=_M2_PIN_LAYER,
                    texttype=_M2_PIN_TEXTTYPE,
                )
            )

    # Escape every D/G/S/B landing through M2 to its own M3 column, then rise
    # to the assigned net track.  Crossings with other M2 tracks are harmless:
    # V2 exists only at the destination track.
    for placement in spec.placements:
        device = placement.transistor
        device_spec = spec.device_spec(device.flavor)
        origin_x, origin_y = placement.origin
        for terminal, net in device.terminals.items():
            local_x, local_y = device_spec.pin_positions[terminal]
            pin_x = origin_x + local_x
            pin_y = origin_y + local_y
            escape_x = origin_x + _ESCAPE_X[terminal]
            track_y = spec.route_track_ys[net]
            half = ROUTE_WIDTH / 2

            _centered_box(cell, "V1", pin_x, pin_y)
            box(
                cell,
                "M2",
                min(pin_x, escape_x) - ROUTE_ENDCAP,
                pin_y - half,
                max(pin_x, escape_x) + ROUTE_ENDCAP,
                pin_y + half,
            )
            _centered_box(cell, "V2", escape_x, pin_y)
            box(
                cell,
                "M3",
                escape_x - half,
                min(pin_y, track_y) - ROUTE_ENDCAP,
                escape_x + half,
                max(pin_y, track_y) + ROUTE_ENDCAP,
            )
            _centered_box(cell, "V2", escape_x, track_y)

    box(cell, "BOUNDARY", 0, 0, spec.width, spec.height)
    return cell


def main(argv: list[str] | None = None) -> None:
    """Generate a parametric sense-amplifier GDS from the command line."""

    parser = argparse.ArgumentParser(
        description="Generate an ASAP7 differential latch sense amplifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-fins", type=int, default=DEFAULT_N_FINS)
    parser.add_argument("--p-fins", type=int, default=DEFAULT_P_FINS)
    parser.add_argument("--vt", choices=("rvt", "lvt", "slvt", "sram"), default="rvt")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    spec = SenseAmpSpec(n_fins=args.n_fins, p_fins=args.p_fins, vt=args.vt)
    output = args.out or Path(f"{spec.cell_name}.gds")
    output.parent.mkdir(parents=True, exist_ok=True)
    gdspy = require_gdspy()
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    build_sense_amp(spec, lib=library)
    library.write_gds(str(output))
    print(f"\u2713 {spec.cell_name}: {spec.width} x {spec.height} nm -> {output}")
    print(f"  16 transistors: 8 x {spec.n_fins}-fin nFET, 8 x {spec.p_fins}-fin pFET")
    print(f"  pins: {', '.join(SENSE_AMP_PINS)}")


if __name__ == "__main__":  # pragma: no cover
    main()
