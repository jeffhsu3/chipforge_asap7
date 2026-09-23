"""Four-wordline driver slice: post-decode NANDs under their wordline drivers.

The released row decoder repeats one 432 nm slice per four wordlines
(`post_Decode_and_size_reduced_x1_...` in ``srambank_32b.gds``).  Its upper
3.24 um is what this module draws, from cells this package already has::

    y=3240 ── WL0 ── WL1 ── WL2 ── WL3 ──   M3, on the 108 nm wordline pitch
            four `InverterSpec` drivers,     dec_inv_62f_halved_AND, mirrored
            interleaved as ASU place them    and sharing every other gate
    y=1350 ────────── vss rail ──────────
            NAND1            NAND3           `NandSpec`, mirrored in Y
    y= 675 ────────── vdd rail ──────────
            NAND0            NAND2           dec_nand_12f_12f_..._P1N1
    y=   0 ────────── vss rail ──────────

``WL<i> = SEL . B<i>``: the four NANDs share SEL on their output-side A
gates -- in the decoder the late, wordline-enable-gated ``PA.WLENA`` -- and
each takes its own ``B<i>`` (``PB.PC<i>``) on the rail side.  A two-finger
NAND is two wordline pitches wide and a two-finger driver owns one, so two
NANDs per row under four interleaved drivers is exactly four wordlines wide,
and the mirrored NAND row puts a VSS rail under the drivers' own.

Routing the slice adds to its leaf cells:

* SEL: a V1 on each NAND's A bar (over its V0), an M2 bar along each NAND
  row's seam, and one M3 at the tile boundary joining the two rows.
* NAND to driver: M1 cannot cross the VSS rail between them, so each output
  goes up on M3.  The lower NAND's output jogs on M2 from its bar to the track
  of driver ``2k``; the upper NAND's bar is already under driver ``2k + 1``.
  The two landings on the drivers' input straps are staggered by one M2 track
  so their pads pass corner to corner rather than end to end.
* WL: a via stack on each driver's top drain contact -- which sits on the
  wordline pitch, the point of the "halved" interleave -- and M3 to the top
  edge, where the array's vertical wordline picks it up.

`DriverSliceSpec.from_sizing` builds the slice that
`chipforge_asap7.devices.sizing.size_decoder` asked for.  The small gates
that make SEL and B<i> from the predecode lines are standard cells in the
released slice and are not drawn here.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from ..layout.layers import LAYERS, box, require_gdspy
from ..layout.rules import (
    CONTACT_SIZE,
    M1_MIN_SPACE,
    M1_V0_ENCLOSURE,
    M1_WIDTH,
    M2_V1_ENCLOSURE,
)
from .finfet import MAX_FINS
from .inverter import MIN_INPUT_REACH, InverterSpec, build_inverter
from .nand import NandSpec, build_nand
from .row_support import RowSupportSpec, build_row_support
from .sizing import WORDLINES_PER_SLICE, DecoderSizing

__all__ = [
    "DRIVER_SLICE_PINS",
    "DriverSliceSpec",
    "PinMap",
    "Router",
    "build_driver_slice",
    "build_driver_slice_support",
    "route_driver_slice",
]

DRIVER_SLICE_PINS = (
    "SEL",
    *(f"B{i}" for i in range(WORDLINES_PER_SLICE)),
    *(f"WL{i}" for i in range(WORDLINES_PER_SLICE)),
    "VDD",
    "VSS",
)

_HALF = CONTACT_SIZE // 2  # 9: half a via, half an 18 nm track
_CAP = _HALF + M1_V0_ENCLOSURE  # 14: metal past a via along its own track
_M2_PAD = _HALF + M2_V1_ENCLOSURE  # 17
_TRACK = M1_WIDTH + M1_MIN_SPACE  # 36
#: The smallest leaf cells that leave room for the slice's own routing.
_MIN_NAND_ROW = (4, 2)
_MIN_DRIVER_FINS = 2


@dataclass(frozen=True)
class DriverSliceSpec:
    """Four post-decode NANDs and the four wordline drivers they feed.

    Args:
        nand: the post-decode NAND.  Two fingers and abutting, so that it is
            two wordline pitches wide and butts against its neighbour.
        inverter: the wordline driver.  Two fingers and abutting, so that four
            interleave into four wordline pitches.
        trim_driver_input: draw each driver with the input metal this slice's
            own router lands on and no more -- one landing, in row 0, reaching
            from the lower of the two staggered landings up to the gate
            contact -- instead of the released cell's tall strap in every row.
            It applies only when `inverter` leaves its own input knobs unset.
            Turn it off for a router that wants the whole strap to choose from;
            `input_strap_y` always reports what is drawn.
    """

    nand: NandSpec = field(default_factory=lambda: NandSpec(rows=((14, 7),), fingers=2))
    inverter: InverterSpec = field(
        default_factory=lambda: InverterSpec(rows=((18, 18), (13, 13)), fingers=2)
    )
    trim_driver_input: bool = True

    def __post_init__(self) -> None:
        for cell, what in ((self.nand, "NAND"), (self.inverter, "wordline driver")):
            if not cell.abut or cell.fingers != 2:
                raise ValueError(
                    f"the {what} must be abutting with two fingers to sit on the "
                    f"108 nm wordline pitch; got fingers={cell.fingers}, abut={cell.abut}"
                )
        if self.nand.vt != self.inverter.vt:
            raise ValueError(
                f"one slice, one threshold flavor: NAND is {self.nand.vt!r}, "
                f"driver is {self.inverter.vt!r}"
            )
        assert 2 * self.nand.width == self.inverter.row_width(WORDLINES_PER_SLICE)

        n_band, p_band = self.nand.n_band, self.nand.p_band
        if p_band.contact_y - self.nand.seam_y < _TRACK + 3 * _HALF:
            raise ValueError(
                "the NAND's p band is too short: the output's M2 jog needs a track "
                "clear of the SEL bar on the seam; use more p fins or band_height"
            )
        if self.nand.seam_y - n_band.contact_y < 2 * _TRACK + 3 * _HALF:
            raise ValueError(
                "the NAND's n band is too short: the output's via needs a track "
                "clear of the B tie; use more n fins or band_height"
            )
        if 0 not in self.driver.input_landing_rows:
            raise ValueError(
                "the slice lands on the driver's row-0 input; "
                f"got input_rows={self.inverter.input_rows}"
            )
        lo, hi = self.input_strap_y
        if self.landing_ys[0] - _CAP < lo or self.landing_ys[1] + _CAP > hi:
            raise ValueError(
                f"the driver's input strap ({hi - lo} nm) does not hold two "
                "staggered landings; use at least two fins per band, or leave "
                "the inverter's input_rows and input_reach to the slice"
            )

    @classmethod
    def from_sizing(
        cls, sizing: DecoderSizing, *, vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"
    ) -> DriverSliceSpec:
        """The slice `size_decoder` asked for, floored at what can be routed."""
        ((n_fins, p_fins),) = sizing.nand_rows
        if n_fins > MAX_FINS:
            raise ValueError(
                f"the post-decode NAND needs {n_fins} fins per device, over the "
                f"{MAX_FINS}-fin band limit; buffer the wordline or split it"
            )
        nand_row = (max(n_fins, _MIN_NAND_ROW[0]), max(p_fins, _MIN_NAND_ROW[1]))
        driver_rows = tuple(
            (max(n, _MIN_DRIVER_FINS), max(p, _MIN_DRIVER_FINS))
            for n, p in sizing.driver_rows
        )
        return cls(
            nand=NandSpec(rows=(nand_row,), fingers=2, vt=vt),
            inverter=InverterSpec(rows=driver_rows, fingers=2, vt=vt),
        )

    # ── Frame ─────────────────────────────────────────────────────────────────
    @property
    def cell_name(self) -> str:
        vt_tag = "" if self.nand.vt == "rvt" else f"_{self.nand.vt}"
        return f"wl_slice_nand{self.nand.code}_inv{self.inverter.code}{vt_tag}"

    @property
    def width(self) -> int:
        return 2 * self.nand.width

    @property
    def driver_y0(self) -> int:
        """Bottom of the driver rows: the VSS rail above the mirrored NAND row."""
        return 2 * self.nand.height

    @property
    def height(self) -> int:
        return self.driver_y0 + self.inverter.height

    @property
    def rails(self) -> tuple[tuple[int, str], ...]:
        nand_rails = ((0, "VSS"), (self.nand.height, "VDD"), (self.driver_y0, "VSS"))
        return nand_rails + tuple(
            (self.driver_y0 + y, net) for y, net in self.inverter.rails[1:]
        )

    @property
    def drc_verifiable(self) -> bool:
        return self.nand.drc_verifiable and self.inverter.drc_verifiable

    # ── Placement ─────────────────────────────────────────────────────────────
    @property
    def nand_placements(self) -> tuple[tuple[int, int, bool], ...]:
        """``(wordline, origin_x, mirrored_in_y)``: NAND ``2k`` below NAND ``2k + 1``."""
        return tuple(
            (2 * column + int(top), column * self.nand.width, top)
            for column in range(2)
            for top in (False, True)
        )

    def nand_point(
        self, origin_x: int, top: bool, x: float, y: float
    ) -> tuple[float, float]:
        """A NAND-local point in slice coordinates."""
        return origin_x + x, (self.driver_y0 - y if top else y)

    def driver_x(self, index: int, local_x: float) -> float:
        origin, mirrored = self.inverter.row_placements(WORDLINES_PER_SLICE)[index]
        return origin - local_x if mirrored else origin + local_x

    @property
    def input_xs(self) -> list[float]:
        """X of each driver's input strap: 81, 135, 297, 351."""
        return [
            self.driver_x(i, self.inverter.input_gate_x)
            for i in range(WORDLINES_PER_SLICE)
        ]

    @property
    def wordline_xs(self) -> list[float]:
        """X of each wordline's M3: the drivers' drain columns, 54 + 108 i."""
        return [
            self.driver_x(i, self.inverter.drain_xs[0])
            for i in range(WORDLINES_PER_SLICE)
        ]

    # ── Routing rows ──────────────────────────────────────────────────────────
    @property
    def output_bar_x(self) -> float:
        """Center X of a NAND's output bar, NAND-local."""
        x0, x1 = self.nand.output_bar_x(self.nand.columns("n", "Y")[0])
        return (x0 + x1) / 2

    def output_tap_y(self, top: bool) -> float:
        """NAND-local Y of the V1 on the output bar, one track in from the end nearest the drivers.

        A flag crosses each end of the bar, so a via a track inside it has M1
        on both sides along the bar (V1.M1.EN.1), which one on the flag row
        itself would not.
        """
        band = self.nand.n_band if top else self.nand.p_band
        return band.contact_y + (3 * _HALF if top else -3 * _HALF)

    @property
    def driver(self) -> InverterSpec:
        """The wordline driver as this slice draws it: `inverter`, input trimmed.

        The layout reducer, run on a whole slice, deletes the row-1 landing of
        every driver and cuts the row-0 strap back to the router's landings and
        the gate contact.  This asks the generator for that cell directly.  The
        devices are `inverter`'s, so its netlist and LVS reference stand.
        """
        inverter = self.inverter
        if (
            not self.trim_driver_input
            or inverter.input_rows is not None
            or inverter.input_reach is not None
        ):
            return inverter
        below, most_above = inverter.full_input_reach(0)
        top_landing = inverter.seam_y(0) - below + _CAP + _TRACK
        above = max(MIN_INPUT_REACH, top_landing + _CAP - inverter.seam_y(0))
        if (
            above > most_above
        ):  # too short for the landings at all; __post_init__ says so
            return inverter
        return replace(inverter, input_rows=(0,), input_reach=(below, above))

    @property
    def input_strap_y(self) -> tuple[float, float]:
        """Slice Y extent of the drivers' row-0 M1 input straps, as drawn."""
        lo, hi = self.driver.input_strap_y(0)
        return self.driver_y0 + lo, self.driver_y0 + hi

    @property
    def landing_ys(self) -> tuple[float, float]:
        """Y of the even and odd drivers' input landings, one M2 track apart.

        Counted up from the bottom of the full strap, so that trimming the
        strap above them does not move them.
        """
        below, _ = self.inverter.full_input_reach(0)
        first = self.driver_y0 + self.inverter.seam_y(0) - below + _CAP
        return first, first + _TRACK

    @property
    def wordline_via_y(self) -> float:
        return self.driver_y0 + self.inverter.bands[-1].contact_y

    @property
    def pin_positions(self) -> dict[str, tuple[str, tuple[float, float]]]:
        """``pin -> (metal, (x, y))``: SEL and WL on M3, B on M2, supplies on M1."""
        # SEL's M3 runs seam to seam across the VDD rail; label it below the
        # rail so it does not sit on the rail's own label.
        sel_y = (self.nand.seam_y + self.nand.height) / 2
        pins: dict[str, tuple[str, tuple[float, float]]] = {
            "SEL": ("M3", (self.nand.width, sel_y))
        }
        tie_y = self.nand.tie_ys["B"]
        for wordline, origin_x, top in self.nand_placements:
            pins[f"B{wordline}"] = (
                "M2",
                self.nand_point(origin_x, top, self.nand.width / 2, tie_y),
            )
        for wordline, x in enumerate(self.wordline_xs):
            pins[f"WL{wordline}"] = ("M3", (x, self.height - _TRACK))
        rails = {net: y for y, net in reversed(self.rails)}
        pins["VDD"] = ("M1", (self.width / 2, rails["VDD"]))
        pins["VSS"] = ("M1", (self.width / 2, rails["VSS"]))
        return pins

    # ── Netlist ───────────────────────────────────────────────────────────────
    def netlist(self, name: str | None = None) -> str:
        """Hierarchical SPICE: four NAND2 + driver pairs sharing SEL."""
        title = name or self.cell_name
        lines = [
            "* ASAP7 four-wordline driver slice: WL<i> = SEL . B<i>",
            f".SUBCKT {title} {' '.join(DRIVER_SLICE_PINS)}",
        ]
        for i in range(WORDLINES_PER_SLICE):
            lines.append(f"XN{i} SEL B{i} N{i} VDD VSS {self.nand.cell_name}")
            lines.append(f"XD{i} N{i} WL{i} VDD VSS {self.inverter.cell_name}")
        lines.append(f".ENDS {title}")
        for leaf in (self.nand.netlist(), self.inverter.netlist()):
            lines += [line for line in leaf.splitlines() if line.strip() != ".END"]
        lines += [".END", ""]
        return "\n".join(lines)


# ── Layout ────────────────────────────────────────────────────────────────────
def _square(cell: Any, layer_name: str, x: float, y: float) -> None:
    box(cell, layer_name, x - _HALF, y - _HALF, x + _HALF, y + _HALF)


def _via12(cell: Any, x: float, y: float) -> None:
    """A V1 with the M2 landing the runset wants (8 nm past the via along the track)."""
    _square(cell, "V1", x, y)
    box(cell, "M2", x - _M2_PAD, y - _HALF, x + _M2_PAD, y + _HALF)


def _m3(cell: Any, x: float, y0: float, y1: float, *, cap: bool = True) -> None:
    extra = _CAP if cap else 0
    box(cell, "M3", x - _HALF, min(y0, y1) - extra, x + _HALF, max(y0, y1) + extra)


PinMap = dict[str, tuple[str, tuple[float, float]]]
#: ``router(cell, spec)`` draws the slice's own M1/M2/M3 and vias into `cell`,
#: over leaf cells that are already placed, and may return the pins it made.
Router = Callable[[Any, "DriverSliceSpec"], "PinMap | None"]


def route_driver_slice(cell: Any, spec: DriverSliceSpec) -> None:
    """The slice's routing: SEL, NAND output to driver input, and the wordlines.

    This is the replaceable part of `build_driver_slice`.  Everything it has
    to connect is published by `spec` -- A bars, output bars, input straps,
    drain contacts -- so a different router (a search-generated one, say)
    needs no knowledge of how the leaf cells are drawn.
    """
    nand = spec.nand

    # SEL.  One V1 per A bar, over its V0; one M2 bar per NAND row along the
    # seam, flush with its outer vias; one M3 on the tile boundary.
    a_x = nand.pad_gate_xs("A")[0]
    sel_x = nand.width
    sel_ys = []
    for top in (False, True):
        _, y = spec.nand_point(0, top, a_x, nand.seam_y)
        for origin_x in (0, nand.width):
            _square(cell, "V1", origin_x + a_x, y)
        box(cell, "M2", a_x - _HALF, y - _HALF, nand.width + a_x + _HALF, y + _HALF)
        _square(cell, "V2", sel_x, y)
        sel_ys.append(y)
    _m3(cell, sel_x, sel_ys[0], sel_ys[1])

    # NAND output to driver input, over the VSS rail on M3.
    for wordline, origin_x, top in spec.nand_placements:
        tap_x, tap_y = spec.nand_point(
            origin_x, top, spec.output_bar_x, spec.output_tap_y(top)
        )
        in_x = spec.input_xs[wordline]
        land_y = spec.landing_ys[wordline % 2]
        _square(cell, "V1", tap_x, tap_y)
        box(
            cell,
            "M2",
            min(tap_x, in_x) - _M2_PAD,
            tap_y - _HALF,
            max(tap_x, in_x) + _M2_PAD,
            tap_y + _HALF,
        )
        _square(cell, "V2", in_x, tap_y)
        _m3(cell, in_x, tap_y, land_y)
        _via12(cell, in_x, land_y)
        _square(cell, "V2", in_x, land_y)

    # Wordlines: up from each driver's top drain contact to the slice edge.
    for x in spec.wordline_xs:
        _via12(cell, x, spec.wordline_via_y)
        _square(cell, "V2", x, spec.wordline_via_y)
        box(cell, "M3", x - _HALF, spec.wordline_via_y - _CAP, x + _HALF, spec.height)


def build_driver_slice(
    spec: DriverSliceSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
    router: Router | None = None,
) -> Any:
    """Draw `spec` and return the gdspy Cell (hierarchical: two leaves, eight references).

    `router` replaces `route_driver_slice`.  If it returns a pin map, those
    pins are labelled instead of `spec.pin_positions` (supplies excepted), so a
    router that moves SEL or a wordline can say where it put them.
    """
    spec = spec or DriverSliceSpec()
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = (
        lib.new_cell(cell_name)
        if lib is not None
        else gdspy.Cell(cell_name, exclude_from_current=True)
    )
    nand, inverter = spec.nand, spec.driver
    nand_leaf = build_nand(
        nand, name=f"{cell_name}__nand", lib=lib, draw_pin_labels=False
    )
    inv_leaf = build_inverter(
        inverter, name=f"{cell_name}__inv", lib=lib, draw_pin_labels=False
    )

    for _wordline, origin_x, top in spec.nand_placements:
        cell.add(
            gdspy.CellReference(
                nand_leaf,
                origin=(origin_x, spec.driver_y0 if top else 0),
                x_reflection=top,
            )
        )
    for origin_x, mirrored in inverter.row_placements(WORDLINES_PER_SLICE):
        cell.add(
            gdspy.CellReference(
                inv_leaf,
                origin=(origin_x, spec.driver_y0),
                x_reflection=mirrored,
                rotation=180 if mirrored else None,
            )
        )

    routed_pins = (router or route_driver_slice)(cell, spec)
    pins = {**spec.pin_positions, **(routed_pins or {})}

    if draw_pin_labels:
        for pin, (metal, origin) in pins.items():
            if pin in ("VDD", "VSS"):
                continue
            layer = LAYERS[f"{metal}_PIN"]
            cell.add(
                gdspy.Label(
                    pin, origin, layer=layer["layer"], texttype=layer["datatype"]
                )
            )
        # Every rail is its own conductor until a power grid joins them.
        m1_pin = LAYERS["M1_PIN"]
        for y_rail, net in spec.rails:
            cell.add(
                gdspy.Label(
                    net,
                    (spec.width / 2, y_rail),
                    layer=m1_pin["layer"],
                    texttype=m1_pin["datatype"],
                )
            )
    box(cell, "BOUNDARY", 0, 0, spec.width, spec.height)
    return cell


def build_driver_slice_support(
    spec: DriverSliceSpec,
    kind: Literal["filler", "tap", "decap"] = "filler",
    *,
    name: str | None = None,
    lib: Any = None,
) -> Any:
    """A filler, tap or decap the full height of the slice, one per row stack.

    The NAND rows and the driver rows are different stacks, so the column that
    terminates a slice is three row-support cells: the NAND stack's, the same
    mirrored, and the driver stack's.
    """
    gdspy = require_gdspy()
    cell_name = name or f"{spec.cell_name}_{kind}"
    cell = (
        lib.new_cell(cell_name)
        if lib is not None
        else gdspy.Cell(cell_name, exclude_from_current=True)
    )
    nand_support = build_row_support(
        RowSupportSpec(stack=spec.nand.stack, kind=kind),
        name=f"{cell_name}__nand",
        lib=lib,
    )
    driver_support = build_row_support(
        RowSupportSpec(stack=spec.inverter.stack, kind=kind),
        name=f"{cell_name}__drv",
        lib=lib,
    )
    cell.add(gdspy.CellReference(nand_support, origin=(0, 0)))
    cell.add(
        gdspy.CellReference(nand_support, origin=(0, spec.driver_y0), x_reflection=True)
    )
    cell.add(gdspy.CellReference(driver_support, origin=(0, spec.driver_y0)))
    return cell


def main(argv: list[str] | None = None) -> None:
    """Size and write a slice: ``python -m chipforge_asap7.devices.driver_slice``."""
    from .sizing import size_decoder

    parser = argparse.ArgumentParser(
        description="Size and draw a four-wordline ASAP7 driver slice.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--wl-load-ff",
        type=float,
        default=22.8,
        help="Wordline load in fF (22.8 reproduces the released slice).",
    )
    parser.add_argument("--depth", type=int, default=32, help="Number of wordlines.")
    parser.add_argument("--stage-effort", type=float, default=4.43)
    parser.add_argument("--vt", choices=("rvt", "lvt", "slvt", "sram"), default="rvt")
    parser.add_argument("--out", type=Path, default=None, help="Output GDS path.")
    args = parser.parse_args(argv)

    sizing = size_decoder(args.wl_load_ff, args.depth, stage_effort=args.stage_effort)
    spec = DriverSliceSpec.from_sizing(sizing, vt=args.vt)
    output = args.out or Path(f"{spec.cell_name}.gds")
    output.parent.mkdir(parents=True, exist_ok=True)
    library = require_gdspy().GdsLibrary(unit=1e-9, precision=1e-10)
    build_driver_slice(spec, lib=library)
    library.write_gds(str(output))
    print(sizing.summary())
    print(f"✓ {spec.cell_name}: {spec.width} x {spec.height} nm -> {output}")


if __name__ == "__main__":  # pragma: no cover
    main()
