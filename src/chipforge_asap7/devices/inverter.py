"""Parametric ASAP7 inverter: a tapless, stacked-band, shared-diffusion cell.

`chipforge_asap7.devices.finfet` draws one transistor as a self-contained
island: its own body tap, a dummy gate track on each side, ACTIVE inset far
enough for a 46 nm select enclosure, and a routing band above the channel for
gate access.  That is the right shape for a device you place on its own and
check on its own.  It is the wrong shape for a dense peripheral cell, and the
released ASAP7 SRAM says so — `dec_inv_62f_halved_AND` in
``asap7_sram_0p0/gds/srambank_32b.gds`` is built the other way round::

    y=1890 ────────── vss rail ──────────
            13-fin nFET   ┐ row 1, flipped
    y=1485                │
            13-fin pFET   ┘
    y=1080 ────────── vdd rail ──────────
            18-fin pFET   ┐ row 0
    y= 540                │
            18-fin nFET   ┘
    y=   0 ────────── vss rail ──────────

Four bands on one continuous poly (``A``) and one output strap (``Y``), two
fingers each: 18+18+13+13 = the 62 fins the cell is named after, giving 31
fins per finger of pull-up and as many of pull-down.  No tap — body ties come
from separate tap rows.  ACTIVE runs 8 nm past both tile edges so abutted
instances merge into one diffusion strip, and each tile draws one more gate
than it owns, the extra landing exactly on the mirrored neighbour's finger.
ASU place four instances, alternately mirrored, whose *tiles* overlap by that
one gate track — which is what "halved" means.  The result is a single
54 nm-pitch array of eight fingers sharing every other source column, with
four independent drains: four wordline drivers.

`InverterSpec` reproduces that cell exactly as
``InverterSpec(rows=((18, 18), (13, 13)), fingers=2)``, and generalises it:
any number of rows, any legal fin count per band, and `abut` to choose
between the released array style and a self-contained island that the public
KLayout runset can check on its own.

Rows alternate orientation the way standard-cell rows do — row 0 has its nFET
at the bottom, row 1 is flipped — so every rail lies between two bands of the
same polarity and carries one supply.  `chipforge_asap7.devices.finfet` still
owns the device arithmetic: each band is a `FinFETSpec`, and `build_device_band`
draws the diffusion stack that a band and a standalone tile have in common.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..layout.grid import FIN_WIDTH, GATE_PITCH, GATE_WIDTH
from ..layout.layers import LAYERS, box, require_gdspy
from .finfet import (
    ACTIVE_ENC,
    CONTACT_SIZE,
    DEVICE_GATE_CUT_HEIGHT,
    GATE_LIG_HEIGHT,
    M1_MIN_SPACE,
    M1_V0_ENCLOSURE,
    M1_WIDTH,
    MAX_VERIFIABLE_FINS,
    POLY_OVERHANG,
    SELECT_X_ENC,
    VT_LAYERS,
    build_device_band,
)
from .row import RowBand, RowStack

__all__ = [
    "ACTIVE_ABUT_OVERHANG",
    "GATE_STRAP_CLEARANCE",
    "INVERTER_PINS",
    "M2_V1_ENCLOSURE",
    "MIN_INPUT_REACH",
    "SELECT_ABUT_OVERHANG",
    "SHORT_M1_EDGE",
    "InverterSpec",
    "build_inverter",
    "build_inverter_row",
]

#: ACTIVE past each tile edge, so abutted instances share one diffusion strip.
#: The released decoder inverter draws ACTIVE -8..170 across a 0..162 tile.
ACTIVE_ABUT_OVERHANG = 8
#: Implant and well overhang in the same style (-17..179 on that cell).
SELECT_ABUT_OVERHANG = 17
#: M2 end-cap around a V1.  V1.M2.EN.2 asks for 5 nm, but the runset builds
#: its 5 nm ring with ``.sized(-2.5.nm).sized(2.5.nm)``, which erases a strip
#: that is exactly 5 nm wide -- the neighbouring V1.M1.EN.1 keeps its 1 dbu of
#: slack and this one does not.  Landing 8 nm clears the rule as written and as
#: implemented; the released decoder inverter draws exactly 5 and trips it.
M2_V1_ENCLOSURE = 8
#: How far the input strap keeps clear of the S/D via rows, along its own
#: track.  Only 4 nm separates it from a drain landing pad in X, so the
#: Euclidean M1 spacing is bought entirely in Y: sqrt(4^2 + 23^2) = 23.3 nm
#: against a rule of 18.  Shrink this and the cell stops being DRC clean.
GATE_STRAP_CLEARANCE = M1_MIN_SPACE + M1_V0_ENCLOSURE
#: The least M1 an input landing can have either side of its V0: the via's own
#: pad.  Two of them make 28 x 18 nm, which is exactly the M1 minimum area
#: (M1.A.1, 504 nm2), so no legal reach can fall under it.  A one-fin band
#: leaves a nanometre less than this on its side of the seam, and there the
#: via rows' clearance is the limit instead.
MIN_INPUT_REACH = CONTACT_SIZE // 2 + M1_V0_ENCLOSURE
#: M1.S.2: an edge under 36 nm needs 25 nm to its neighbour where a long one
#: needs 18.  `GATE_STRAP_CLEARANCE` buys 23.3 nm from a drain pad, so a landing
#: that short has to stop 2 nm further from the via rows than the strap does.
SHORT_M1_EDGE = 36
_SHORT_EDGE_SETBACK = 2

INVERTER_PINS = ("A", "Y", "VDD", "VSS")

_M1_PIN_LAYER = LAYERS["M1_PIN"]["layer"]
_M1_PIN_TEXTTYPE = LAYERS["M1_PIN"]["datatype"]


@dataclass(frozen=True)
class InverterSpec:
    """A CMOS inverter folded into stacked, rail-sharing transistor bands.

    Args:
        rows: ``(n_fins, p_fins)`` per standard-cell row, bottom to top.  Row 0
            puts its nFET at the bottom and every later row flips, so each
            rail separates two bands of one polarity.  Both bands of a row are
            driven by the same gate and drive the same output; more rows is
            how a device gets wider than one legal fin height.
        fingers: gates per band, sharing source/drain columns.  Must be even so
            that both outer columns are sources — that is what lets every
            source tie straight to a rail and, when abutting, what makes the
            shared edge column a source rather than a shorted output.
        vt: threshold-voltage model shared by every band.
        abut: draw the released array style — ACTIVE and implant overhanging
            the tile so neighbours merge, and one shared edge gate.  Set False
            for a self-contained island with its own dummy gates and a full
            46 nm select enclosure, which the public runset can check alone.
        input_rows: rows that get an M1 input landing and the V0 under it.
            ``None`` is every row, as the released cell draws it, which suits a
            parent that wants to choose its access point.  A parent that lands
            once wants ``(0,)``: every other landing is a dead-end stub 27 nm
            from the output strap, and the uncut poly carries the input between
            rows anyway.  Every row keeps its LIG strap regardless -- it is
            what ties the fingers' gates together at that seam, and without it
            the worst gate resistance of a two-row cell rises by half.
        input_reach: ``(below, above)`` nm of M1 from the seam along the gate
            track, for every landing.  ``None`` reaches as far as the via rows
            allow, the released cell's tall strap; ``(MIN_INPUT_REACH,) * 2``
            is the bare via pad.  A landing under `SHORT_M1_EDGE` long must
            also keep clear of the via rows, which a one-fin band has no room
            for; make it 36 nm there.  Give a parent the reach its landings need
            and no more: on a (4, 6)+(3, 3) cell, one bare landing instead of
            two tall straps takes a tenth off the switched wire capacitance
            and a seventh off the input-to-output coupling, at no resistance.

    Both input knobs came out of `chipforge_asap7.verification.reduce`, which
    found the shapes; they change no device, so the netlist and the LVS
    reference are the same for every setting.
    """

    rows: tuple[tuple[int, int], ...] = ((4, 6),)
    fingers: int = 2
    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"
    abut: bool = True
    input_rows: tuple[int, ...] | None = None
    input_reach: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        # RowStack owns row validation and normalisation, so an inverter and
        # the tap or filler placed beside it cannot disagree about the stack.
        object.__setattr__(self, "rows", RowStack(rows=self.rows, vt=self.vt).rows)
        if isinstance(self.fingers, bool) or not isinstance(self.fingers, int):
            raise TypeError(f"fingers must be an integer, got {self.fingers!r}")
        if self.fingers < 2 or self.fingers % 2:
            raise ValueError(
                "fingers must be a positive even number so both outer S/D "
                f"columns are sources; got {self.fingers}"
            )
        if self.input_rows is not None:
            rows = tuple(self.input_rows)
            if (
                not rows
                or any(isinstance(r, bool) or not isinstance(r, int) for r in rows)
                or list(rows) != sorted(set(rows))
                or not 0 <= rows[0] <= rows[-1] < len(self.rows)
            ):
                raise ValueError(
                    "input_rows must be a non-empty ascending selection of rows "
                    f"0..{len(self.rows) - 1}; got {self.input_rows!r}"
                )
            object.__setattr__(self, "input_rows", rows)
        if self.input_reach is not None:
            reach = tuple(self.input_reach)
            if len(reach) != 2 or any(
                isinstance(r, bool) or not isinstance(r, int) for r in reach
            ):
                raise TypeError(
                    f"input_reach must be two integers (below, above); got {self.input_reach!r}"
                )
            object.__setattr__(self, "input_reach", reach)
            short = sum(reach) < SHORT_M1_EDGE
            for row in self.input_landing_rows:
                most = self.full_input_reach(row)
                for got, limit, side in zip(reach, most, ("below", "above")):
                    if not min(MIN_INPUT_REACH, limit) <= got <= limit:
                        raise ValueError(
                            f"input_reach {side} the seam of row {row} must be "
                            f"{min(MIN_INPUT_REACH, limit)}..{limit} nm; got {got}"
                        )
                    if short and got > limit - _SHORT_EDGE_SETBACK:
                        raise ValueError(
                            f"a landing under {SHORT_M1_EDGE} nm long needs 25 nm to "
                            f"the via rows (M1.S.2), so at most "
                            f"{limit - _SHORT_EDGE_SETBACK} nm {side} the seam of row "
                            f"{row}; got input_reach={reach}.  Make it "
                            f"{SHORT_M1_EDGE} nm long or pull it in"
                        )

    # ── Band stack ────────────────────────────────────────────────────────────
    @property
    def stack(self) -> RowStack:
        """The row geometry this inverter shares with its row-support cells."""
        return RowStack(rows=self.rows, vt=self.vt)

    @property
    def code(self) -> str:
        """Per-row fin code, e.g. ``"18n18p_13n13p"``."""
        return self.stack.code

    @property
    def cell_name(self) -> str:
        """Default cell name, e.g. ``"inv_fin_18n18p_13n13p_2f"``."""
        vt_tag = "" if self.vt == "rvt" else f"_{self.vt}"
        style_tag = "" if self.abut else "_iso"
        input_tag = ""
        if self.input_rows is not None:
            input_tag += "_in" + "".join(str(row) for row in self.input_rows)
        if self.input_reach is not None:
            input_tag += "_reach{}x{}".format(*self.input_reach)
        return f"inv_fin_{self.code}_{self.fingers}f{vt_tag}{style_tag}{input_tag}"

    @property
    def bands(self) -> tuple[RowBand, ...]:
        """Every drawn band, bottom to top."""
        return self.stack.bands(self.fingers)

    def row_bands(self, row: int) -> tuple[RowBand, ...]:
        """The two bands of `row`, bottom to top."""
        return self.stack.row_bands(row, self.fingers)

    @property
    def row_ys(self) -> tuple[int, ...]:
        """Y of every row boundary, which is also where a rail runs."""
        return self.stack.row_ys

    @property
    def rails(self) -> tuple[tuple[int, str], ...]:
        """``(y, net)`` of every power rail, bottom to top."""
        return self.stack.rails

    def seam_y(self, row: int) -> int:
        """Y of the n/p boundary inside `row`, where the gate is contacted."""
        return self.stack.seam_y(row)

    @property
    def height(self) -> int:
        return self.stack.height

    # ── Columns ───────────────────────────────────────────────────────────────
    @property
    def width(self) -> int:
        """Tile width.  Abutting cells share their edge gate with a neighbour."""
        extra_tracks = 1 if self.abut else 2
        return (self.fingers + extra_tracks) * GATE_PITCH

    @property
    def gate_grid_xs(self) -> list[int]:
        """Center X of every drawn poly track, active fingers and dummies."""
        return list(range(GATE_PITCH // 2, self.width, GATE_PITCH))

    @property
    def gate_xs(self) -> list[int]:
        """Center X of the electrically active gate fingers.

        Abutting, the tile's last track belongs to the mirrored neighbour; as
        an island, the outermost track on each side is a dummy.
        """
        tracks = self.gate_grid_xs
        return tracks[: self.fingers] if self.abut else tracks[1 : self.fingers + 1]

    @property
    def sd_xs(self) -> list[int]:
        """Center X of each source/drain column, left to right."""
        first = 0 if self.abut else GATE_PITCH
        return [first + i * GATE_PITCH for i in range(self.fingers + 1)]

    @property
    def source_xs(self) -> list[int]:
        """Source columns.  An even finger count puts one at each outer edge."""
        return self.sd_xs[0::2]

    @property
    def drain_xs(self) -> list[int]:
        """Drain columns — every column the output strap has to reach."""
        return self.sd_xs[1::2]

    @property
    def input_gate_x(self) -> int:
        """Gate finger the input strap contacts and runs on."""
        return self.gate_xs[-1]

    @property
    def input_landing_rows(self) -> tuple[int, ...]:
        """Rows that carry an M1 input landing: `input_rows`, or every row."""
        if self.input_rows is not None:
            return self.input_rows
        return tuple(range(len(self.rows)))

    def full_input_reach(self, row: int) -> tuple[int, int]:
        """The most M1 `row`'s landing can have ``(below, above)`` its seam.

        The strap shares a track pitch with the drain landing pads, so it has
        to stop `GATE_STRAP_CLEARANCE` short of both via rows.
        """
        lower, upper = self.row_bands(row)
        keep_out = CONTACT_SIZE // 2 + GATE_STRAP_CLEARANCE
        seam = self.seam_y(row)
        # Via rows sit half a contact inside an integer ACTIVE edge: whole nm.
        return (
            int(seam - lower.contact_y - keep_out),
            int(upper.contact_y - seam - keep_out),
        )

    def input_strap_y(self, row: int) -> tuple[int, int]:
        """Y extent of the M1 input landing in `row`, which must be an input row."""
        if row not in self.input_landing_rows:
            raise ValueError(
                f"row {row} has no input landing; input_rows={self.input_rows}"
            )
        below, above = self.input_reach or self.full_input_reach(row)
        return self.seam_y(row) - below, self.seam_y(row) + above

    def output_strap_x(self, drain_x: int) -> tuple[float, float]:
        """M1 track for the output strap beside `drain_x`.

        The strap cannot sit on the drain column: the input strap runs on a
        gate half a CPP away, and two 18 nm M1 tracks 27 nm apart leave 9 nm,
        half of `M1_MIN_SPACE`.  Tucking the output between the drain via and
        the far gate — the released cell's arrangement — gives 27 nm instead.
        """
        edge = CONTACT_SIZE / 2
        if self.input_gate_x > drain_x:
            return drain_x - edge - M1_WIDTH, drain_x - edge
        return drain_x + edge, drain_x + edge + M1_WIDTH

    @property
    def device_pitch(self) -> int:
        """X pitch between abutting instances.

        An abutting tile is one gate track wider than the fingers it owns, so
        instances repeat every ``fingers * GATE_PITCH`` and overlap by that
        extra track -- not every `width`.
        """
        return self.fingers * GATE_PITCH

    def row_placements(self, count: int) -> tuple[tuple[int, bool], ...]:
        """``(origin_x, mirrored)`` for `count` abutting instances.

        An abutting tile draws one more gate than it owns and puts its outer
        S/D columns on its own edges, so neighbours interleave rather than
        simply butt.  Alternate instances are mirrored: each one's shared gate
        lands on its neighbour's outermost finger and its outer source column
        on the neighbour's, so `count` tiles become one continuous 54 nm array
        of ``count * fingers`` fingers, sharing every other source column, with
        `count` independent inputs and outputs.

        For the released decoder inverter -- two fingers, a 162 nm tile -- this
        returns ``(0, False), (216, True), (216, False), (432, True)``, which
        is exactly how ASU place its four wordline drivers.  A single tile
        holding half of each finger pair is what "halved" in
        `dec_inv_62f_halved_AND` means.

        An even `count` terminates itself: the last instance is mirrored, so
        the gate it does not own faces inward and its outer column is a
        contacted source.  An odd `count` ends on an unmirrored tile whose
        extra gate has no neighbour to claim it, leaving one finger on
        floating poly -- real devices, which
        `chipforge_asap7.verification.lvs.render_inverter_row_lvs_schematic`
        models rather than ignores.  A placed design ends the row with a
        filler that supplies the missing half.
        """
        if not self.abut:
            raise ValueError(
                "an isolated inverter is self-contained; abutting placement "
                "needs abut=True"
            )
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        pitch = self.device_pitch
        return tuple(
            ((index + 1) * pitch, True) if index % 2 else (index * pitch, False)
            for index in range(count)
        )

    def row_width(self, count: int) -> int:
        """Drawn width of `count` abutting instances, dangling edge track included."""
        placements = self.row_placements(count)
        return max(
            origin if mirrored else origin + self.width
            for origin, mirrored in placements
        )

    @property
    def active_x(self) -> tuple[float, float]:
        if self.abut:
            return -ACTIVE_ABUT_OVERHANG, self.width + ACTIVE_ABUT_OVERHANG
        return SELECT_X_ENC, self.width - SELECT_X_ENC

    @property
    def select_x(self) -> tuple[float, float]:
        if self.abut:
            return -SELECT_ABUT_OVERHANG, self.width + SELECT_ABUT_OVERHANG
        return 0, self.width

    @property
    def fin_x(self) -> tuple[float, float]:
        """X span of the fin grid.

        ACTIVE.FIN.EX.1 wants 10 nm of ACTIVE past FIN, which an abutting cell
        cannot give at its own edge: its ACTIVE deliberately overhangs.  Run
        FIN past the overhang instead, so FIN crosses the ACTIVE boundary
        rather than ending just inside it.  Abutted neighbours then overlap
        identical stripes on the same fin tracks, which merge to the same
        shape the array would have drawn anyway.
        """
        ax0, ax1 = self.active_x
        return min(0, ax0 - ACTIVE_ENC), max(self.width, ax1 + ACTIVE_ENC)

    @property
    def fin_grid_ys(self) -> list[int]:
        """Bottom Y of the full manufacturing fin grid across the cell."""
        return self.stack.fin_grid_ys

    # ── Sizing and verification ───────────────────────────────────────────────
    @property
    def pull_down_fins(self) -> int:
        return sum(n for n, _ in self.rows) * self.fingers

    @property
    def pull_up_fins(self) -> int:
        return sum(p for _, p in self.rows) * self.fingers

    @property
    def total_fins(self) -> int:
        return self.pull_down_fins + self.pull_up_fins

    @property
    def drc_verifiable(self) -> bool:
        """True when the public KLayout runset can check every band's height.

        The deck enumerates ACTIVE/SDT heights 1x27..12x27 instead of testing
        for a multiple of 27, so taller bands draw a false ACTIVE.W.2 /
        SDT.W.3 -- see `chipforge_asap7.devices.finfet.MAX_VERIFIABLE_FINS`.
        The released decoder inverter's 18- and 13-fin bands are both above it.
        """
        return max(band.fins for band in self.bands) <= MAX_VERIFIABLE_FINS

    @property
    def pin_positions(self) -> dict[str, tuple[float, float]]:
        """One M1 access point per pin, in cell-local nanometres.

        Composite generators use these to place upper-metal vias without
        rediscovering the cell's internal geometry.  A supply appears once,
        on the lowest rail that carries it; `rails` has them all.
        """
        strap_x0, strap_x1 = self.output_strap_x(self.drain_xs[0])
        rails = {net: y for y, net in reversed(self.rails)}
        return {
            "A": (self.input_gate_x, self.seam_y(self.input_landing_rows[0])),
            "Y": ((strap_x0 + strap_x1) / 2, self.bands[0].contact_y),
            "VDD": (self.width / 2, rails["VDD"]),
            "VSS": (self.width / 2, rails["VSS"]),
        }

    # ── Netlist ───────────────────────────────────────────────────────────────
    def netlist(self, name: str | None = None) -> str:
        """A BSIM-CMG subcircuit, one instance per band, sized by ``nfin``."""
        title = name or self.cell_name
        header = (
            f"* ASAP7 inverter: {self.pull_down_fins} pull-down / "
            f"{self.pull_up_fins} pull-up fins in {len(self.rows)} row(s)"
        )
        lines = [header, f".SUBCKT {title} {' '.join(INVERTER_PINS)}"]
        for index, band in enumerate(self.bands):
            rail = band.rail_net
            lines.append(
                f"M{index} Y A {rail} {rail} {band.spec.model} "
                f"nfin={band.fins} l={band.spec.gate_length}n "
                f"nf={self.fingers} m=1"
            )
        lines += [f".ENDS {title}", ".END", ""]
        return "\n".join(lines)


# ── Layout ────────────────────────────────────────────────────────────────────
def _square(cell: Any, layer_name: str, x: float, y: float, size: float) -> None:
    half = size / 2
    box(cell, layer_name, x - half, y - half, x + half, y + half)


def build_inverter(
    spec: InverterSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Draw `spec` and return the gdspy Cell.

    Args:
        spec: the inverter to draw; defaults to `InverterSpec()`.
        name: cell name; defaults to `spec.cell_name`.
        lib: a `gdspy.GdsLibrary` to create the cell in.  When omitted the cell
            is standalone, so building several in one process cannot collide in
            gdspy's global library.
        draw_pin_labels: emit the ``A/Y/VDD/VSS`` labels.  A larger composite
            disables these and exposes only its own top-level pins.
    """
    spec = spec or InverterSpec()
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = (
        lib.new_cell(cell_name)
        if lib is not None
        else gdspy.Cell(cell_name, exclude_from_current=True)
    )

    half_ct = CONTACT_SIZE / 2  # via
    half_m1 = M1_WIDTH / 2  # metal
    m1_pad = half_ct + M1_V0_ENCLOSURE  # M1 half-extent along its own track
    half_gate = GATE_WIDTH / 2
    half_cut = DEVICE_GATE_CUT_HEIGHT / 2
    sx0, sx1 = spec.select_x
    bands = spec.bands

    # Implant tiles the whole cell, one band at a time; a pFET band and its
    # well are coincident, and adjacent p bands merge into one continuous
    # NWELL exactly as they do in the released cell.
    for band in bands:
        implant = "NSELECT" if band.flavor == "n" else "PSELECT"
        box(cell, implant, sx0, band.y0, sx1, band.y0 + band.height)
        if band.flavor == "p":
            box(cell, "NWELL", sx0, band.y0, sx1, band.y0 + band.height)
    if vt_layer := VT_LAYERS[spec.vt]:
        box(cell, vt_layer, sx0, 0, sx1, spec.height)

    # FIN and GATE are manufacturing grids, not shapes cropped to a channel.
    # One uncut poly stripe runs the full height of the cell, so a single
    # input drives every band -- there is no gate cut between the rows.
    fx0, fx1 = spec.fin_x
    for y_fin in spec.fin_grid_ys:
        box(cell, "FIN", fx0, y_fin, fx1, y_fin + FIN_WIDTH)
    for x_gate in spec.gate_grid_xs:
        box(
            cell,
            "GATE",
            x_gate - half_gate,
            -POLY_OVERHANG,
            x_gate + half_gate,
            spec.height + POLY_OVERHANG,
        )
    for y_cut in (0, spec.height):
        box(cell, "GATE_CUT", sx0, y_cut - half_cut, sx1, y_cut + half_cut)

    # Diffusion.  Every band shares the same columns and the same ACTIVE
    # extent, which is what makes the fingers of stacked bands one device.
    for band in bands:
        build_device_band(
            cell,
            band.spec,
            y0=band.y0,
            sd_xs=spec.sd_xs,
            active_x=spec.active_x,
        )

    # Power rails, and a stub from each source column up or down to the rail
    # its own band faces.
    for y_rail, _net in spec.rails:
        box(cell, "M1", 0, y_rail - half_m1, spec.width, y_rail + half_m1)
    for band in bands:
        y_via = band.contact_y
        above = y_via > band.rail_y
        end = y_via + (m1_pad if above else -m1_pad)
        # Stop on the rail's *far* edge.  Stopping on its centre line leaves a
        # 9 nm concave step where the stub meets the rail, and on the column
        # sitting at x=0 of an abutting cell that step is a real M1.W.1.
        rail_far = band.rail_y + (-half_m1 if above else half_m1)
        for x_source in spec.source_xs:
            _square(cell, "V0", x_source, y_via, CONTACT_SIZE)
            box(cell, "M1", x_source - half_m1, rail_far, x_source + half_m1, end)

    # Output.  One vertical M1 strap per drain column per row, landing on that
    # row's two drain via rows.  The strap stops short of both rails, so
    # crossing one costs an M1-M2-M3 jog rather than a short.
    for row in range(len(spec.rows)):
        lower, upper = spec.row_bands(row)
        y_lo, y_hi = lower.contact_y, upper.contact_y
        for x_drain in spec.drain_xs:
            strap_x0, strap_x1 = spec.output_strap_x(x_drain)
            box(cell, "M1", strap_x0, y_lo - half_ct, strap_x1, y_hi + half_ct)
            for y_via in (y_lo, y_hi):
                _square(cell, "V0", x_drain, y_via, CONTACT_SIZE)
                box(
                    cell,
                    "M1",
                    min(strap_x0, x_drain - m1_pad),
                    y_via - half_m1,
                    max(strap_x1, x_drain + m1_pad),
                    y_via + half_m1,
                )
        # Folding past two fingers gives a row more than one drain column;
        # tie them together on M2, clear of both via rows.
        if len(spec.drain_xs) > 1:
            y_tie = spec.seam_y(row)
            centers = [sum(spec.output_strap_x(x)) / 2 for x in spec.drain_xs]
            m2_pad = half_ct + M2_V1_ENCLOSURE
            box(
                cell,
                "M2",
                min(centers) - m2_pad,
                y_tie - half_m1,
                max(centers) + m2_pad,
                y_tie + half_m1,
            )
            for center in centers:
                _square(cell, "V1", center, y_tie, CONTACT_SIZE)

    # Cross each interior rail on M3.  M1 cannot: the rail owns that track.
    for row in range(len(spec.rows) - 1):
        y_lo = spec.row_bands(row)[1].contact_y
        y_hi = spec.row_bands(row + 1)[0].contact_y
        for x_drain in spec.drain_xs:
            for y_via in (y_lo, y_hi):
                _square(cell, "V1", x_drain, y_via, CONTACT_SIZE)
                box(
                    cell,
                    "M2",
                    x_drain - half_ct - M2_V1_ENCLOSURE,
                    y_via - half_m1,
                    x_drain + half_ct + M2_V1_ENCLOSURE,
                    y_via + half_m1,
                )
                _square(cell, "V2", x_drain, y_via, CONTACT_SIZE)
            box(
                cell,
                "M3",
                x_drain - half_m1,
                y_lo - m1_pad,
                x_drain + half_m1,
                y_hi + m1_pad,
            )

    # Input.  One LIG strap joins every active finger at each row's n/p seam
    # -- the same place the released standard cells take gate access, and the
    # only Y in the cell where no LISD bar is in the way.  A single 22 nm pad
    # on the contacted finger would leave the others on floating poly.  The
    # link between rows is carried by the uncut poly itself, not by metal, so
    # the LIG is drawn in every row and the M1 landing only where it is asked
    # for: a row's LIG still ties its fingers' gates without a via above it.
    gate_x = spec.input_gate_x
    half_lig = GATE_LIG_HEIGHT / 2
    for row in range(len(spec.rows)):
        y_seam = spec.seam_y(row)
        box(
            cell,
            "LIG",
            spec.gate_xs[0] - half_lig,
            y_seam - half_lig,
            spec.gate_xs[-1] + half_lig,
            y_seam + half_lig,
        )
        if row not in spec.input_landing_rows:
            continue
        strap_y0, strap_y1 = spec.input_strap_y(row)
        _square(cell, "V0", gate_x, y_seam, CONTACT_SIZE)
        box(cell, "M1", gate_x - half_m1, strap_y0, gate_x + half_m1, strap_y1)

    if draw_pin_labels:
        positions = spec.pin_positions
        labels = [("A", positions["A"]), ("Y", positions["Y"])]
        # Name *every* rail, not just one per supply.  A standalone cell has
        # one M1 rail per row boundary and they are separate conductors until
        # the placed design's power grid joins them, so LVS needs the names to
        # know that the top VSS rail is the same net as the bottom one.
        labels += [(net, (spec.width / 2, y)) for y, net in spec.rails]
        for pin, origin in labels:
            cell.add(
                gdspy.Label(
                    pin,
                    origin,
                    layer=_M1_PIN_LAYER,
                    texttype=_M1_PIN_TEXTTYPE,
                )
            )

    for row, y0 in enumerate(spec.row_ys[:-1]):
        box(cell, "BOUNDARY", 0, y0, spec.width, spec.row_ys[row + 1])
    return cell


def build_inverter_row(
    spec: InverterSpec,
    count: int,
    *,
    name: str | None = None,
    lib: Any = None,
    leaf_name: str | None = None,
) -> Any:
    """Place `count` abutting inverters as one continuous-diffusion row.

    See `InverterSpec.row_placements` for the interleave this relies on, and
    for why an odd `count` leaves one unpaired edge gate.  The leaf cell is
    drawn once and referenced, with its own pin labels suppressed; the row
    labels ``A0/Y0 .. A<n-1>/Y<n-1>`` and one label per rail, so each driver's
    input and output are separately identifiable to LVS.

    Args:
        spec: the inverter to repeat.  Must have `abut` set.
        count: how many drivers to place.
        name: row cell name; defaults to ``<cell_name>_x<count>``.
        lib: a `gdspy.GdsLibrary` to create both cells in.
        leaf_name: name for the repeated cell; defaults to `spec.cell_name`.
    """
    gdspy = require_gdspy()
    placements = spec.row_placements(count)
    row_name = name or f"{spec.cell_name}_x{count}"
    row = (
        lib.new_cell(row_name)
        if lib is not None
        else gdspy.Cell(row_name, exclude_from_current=True)
    )
    leaf = build_inverter(
        spec, name=leaf_name or spec.cell_name, lib=lib, draw_pin_labels=False
    )

    def label(text: str, origin: tuple[float, float]) -> None:
        row.add(
            gdspy.Label(text, origin, layer=_M1_PIN_LAYER, texttype=_M1_PIN_TEXTTYPE)
        )

    pins = spec.pin_positions
    for index, (origin_x, mirrored) in enumerate(placements):
        row.add(
            gdspy.CellReference(
                leaf,
                origin=(origin_x, 0),
                x_reflection=mirrored,
                rotation=180 if mirrored else None,
            )
        )
        for pin in ("A", "Y"):
            x, y = pins[pin]
            label(f"{pin}{index}", (origin_x - x if mirrored else origin_x + x, y))

    width = spec.row_width(count)
    for y_rail, net in spec.rails:
        label(net, (width / 2, y_rail))
    for row_index, y0 in enumerate(spec.row_ys[:-1]):
        box(row, "BOUNDARY", 0, y0, width, spec.row_ys[row_index + 1])
    return row


def _parse_rows(text: str) -> tuple[tuple[int, int], ...]:
    """Parse ``"18:18,13:13"`` into ``((18, 18), (13, 13))``."""
    rows = []
    for field in text.split(","):
        n, _, p = field.partition(":")
        if not p:
            raise argparse.ArgumentTypeError(
                f"row {field!r} must be 'n_fins:p_fins', e.g. '18:18'"
            )
        try:
            rows.append((int(n), int(p)))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"row {field!r} must be 'n_fins:p_fins', e.g. '18:18'"
            ) from None
    return tuple(rows)


def main(argv: list[str] | None = None) -> None:
    """Write an inverter GDS: ``python -m chipforge_asap7.devices.inverter``."""
    parser = argparse.ArgumentParser(
        description="Generate a stacked-band ASAP7 inverter, sized in fins.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rows",
        type=_parse_rows,
        default="4:6",
        help="Comma-separated n_fins:p_fins per row, bottom to top.",
    )
    parser.add_argument("--fingers", type=int, default=2, help="Gates per band.")
    parser.add_argument("--vt", choices=("rvt", "lvt", "slvt", "sram"), default="rvt")
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="Draw a self-contained island instead of the abutting array style.",
    )
    parser.add_argument(
        "--input-rows",
        type=lambda text: tuple(int(row) for row in text.split(",")),
        default=None,
        help="Rows that get an M1 input landing, e.g. '0'; every row when omitted.",
    )
    parser.add_argument(
        "--input-reach",
        type=lambda text: tuple(int(nm) for nm in text.split(",")),
        default=None,
        help=f"M1 'below,above' each landing's seam in nm, e.g. "
        f"'{MIN_INPUT_REACH},{MIN_INPUT_REACH}' for the bare via pad; the full strap when omitted.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output GDS path.")
    args = parser.parse_args(argv)

    rows = args.rows if isinstance(args.rows, tuple) else _parse_rows(args.rows)
    spec = InverterSpec(
        rows=rows,
        fingers=args.fingers,
        vt=args.vt,
        abut=not args.isolated,
        input_rows=args.input_rows,
        input_reach=args.input_reach,
    )
    output = args.out or Path(f"{spec.cell_name}.gds")
    output.parent.mkdir(parents=True, exist_ok=True)
    library = require_gdspy().GdsLibrary(unit=1e-9, precision=1e-10)
    build_inverter(spec, lib=library)
    library.write_gds(str(output))
    print(f"✓ {spec.cell_name}: {spec.width} x {spec.height} nm -> {output}")
    print(
        f"  {spec.pull_down_fins} pull-down / {spec.pull_up_fins} pull-up fins, "
        f"{len(spec.rows)} row(s) x {spec.fingers} fingers"
    )
    print(f"  DRC-verifiable by the public runset: {spec.drc_verifiable}")


if __name__ == "__main__":  # pragma: no cover
    main()
