"""Tap, filler and decap: the cells a row of logic needs but cannot contain.

An abutting logic row drawn by `chipforge_asap7.devices.inverter` is as clean
as a logic cell can be on its own, and still reports eight DRC violations,
because two of the rules it has to satisfy are not properties of a cell at all:

    ACTIVE.LUP.1              every device needs a body tie within 30 um, and
                              a logic cell carries no tap -- the released
                              `dec_inv_62f_halved_AND` carries none either
    NSELECT/PSELECT/WELL      an abutting cell's implant stops 9 nm past its
      .ACTIVE.EN.1            own ACTIVE instead of 46 nm, because the rest of
                              the enclosure comes from the neighbour

Both are closed by cells with almost no geometry in them.  Measured with the
public runset on a four-wide inverter row::

    row alone                        8    2x LUP, 2x WELL.EN, 2x NSELECT.EN,
                                          2x PSELECT.EN
    filler + row + filler            2    2x LUP
    filler + row + filler + tap      0

The released cells say the same thing.  `FILLER_ASAP7_75t_R` carries
implant, well, the FIN and GATE manufacturing grids, the two rails and nothing
else -- no ACTIVE whatsoever.  `TAPCELL_ASAP7_75t_R` is that same shell with
the implant polarity *inverted* per band, plus one 16 nm ACTIVE tie in each,
wired to the rail it faces.  Both report zero violations from the public
runset on their own, which is more than a logic cell can manage.

The ordering matters and is not interchangeable.  A filler carries the *same*
implant as the row, which is what gives the row's last ACTIVE its 46 nm
enclosure; a tap carries the opposite.  Put a tap straight against a logic
cell and the enclosure violation stays exactly where it was.  So a terminated
row reads ``filler, logic..., filler, tap, filler`` -- never ``logic, tap``.

`decap` is the third of the trio and the only one with devices in it: a MOS
capacitor pair between the rails, drawn as a series stack.  The two bands of a
row share one gate conductor, which floats, so their capacitors sit back to
back between VDD and VSS and neither oxide sees the full supply.  Stacked rows
each get their own plate net, because the poly is cut at every rail.  That halves the
capacitance against a pair of grounded-gate caps, and it is what fits in a
cell whose two bands span the same columns: the only Y with no source/drain
bar across it is the n/p seam, and one strap there can reach one gate net, not
two.  The released `DECAPx4_ASAP7_75t_R` buys the other arrangement by
staggering its two ACTIVE regions horizontally so each band's strap has a
column of its own; that is a bigger cell for the same job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..layout.grid import FIN_WIDTH, GATE_PITCH, GATE_WIDTH
from ..layout.layers import PIN_LAYERS, box, require_gdspy
from ..layout.rules import CONTACT_SIZE, M1_WIDTH
from .cli import parse_spec, write_gds
from .finfet import (
    DEVICE_GATE_CUT_HEIGHT,
    GATE_LIG_HEIGHT,
    LI_RAIL_HEIGHT,
    MAX_VERIFIABLE_FINS,
    POLY_OVERHANG,
    SD_BAR_WIDTH,
    SELECT_X_ENC,
    SELECT_Y_ENC,
    TAP_ACTIVE_WIDTH,
    VT_LAYERS,
)
from .row import RowBand, RowStack

__all__ = [
    "DEFAULT_WIDTH_CPP",
    "MINIMUM_WIDTH_CPP",
    "ROW_SUPPORT_KINDS",
    "RowSupportSpec",
    "build_row_support",
]

ROW_SUPPORT_KINDS = ("filler", "tap", "decap")
#: Released widths: FILLER and TAPCELL are 2 CPP, DECAPx4 is 10.  Six is the
#: smallest decap whose plate is wider than its two edge dummies.
DEFAULT_WIDTH_CPP = {"filler": 2, "tap": 2, "decap": 6}
#: A tap needs one interior column to put its tie on, and a decap at least one
#: gate between two of them.  Two is also the floor for a filler, even though
#: it has no geometry to fit: FIN, NWELL, NSELECT and PSELECT all carry a
#: 108 nm minimum horizontal width, so a one-pitch cell is 54 nm short on four
#: rules at once.  The released `FILLERxp5_ASAP7_75t_R` is exactly that width
#: and gets away with it only because its layers merge with the cells it is
#: placed between; on its own it does not pass.
MINIMUM_WIDTH_CPP = {"filler": 2, "tap": 2, "decap": 3}

_M1_PIN_LAYER, _M1_PIN_TEXTTYPE = PIN_LAYERS["M1"]


@dataclass(frozen=True)
class RowSupportSpec:
    """A tap, filler or decap sized to the row stack it will be placed in.

    Args:
        stack: the row geometry.  Pass the very same `RowStack` the logic cells
            in the row were built from -- ``InverterSpec.stack`` returns it --
            and the two cannot disagree about band heights or rail positions.
        kind: ``"filler"``, ``"tap"`` or ``"decap"``.
        width_cpp: cell width in contacted-poly pitches.  Defaults to the
            released width for the kind.
    """

    stack: RowStack = field(default_factory=RowStack)
    kind: Literal["filler", "tap", "decap"] = "filler"
    width_cpp: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ROW_SUPPORT_KINDS:
            raise ValueError(
                f"kind must be one of {ROW_SUPPORT_KINDS}, got {self.kind!r}"
            )
        if self.width_cpp is None:
            object.__setattr__(self, "width_cpp", DEFAULT_WIDTH_CPP[self.kind])
        if isinstance(self.width_cpp, bool) or not isinstance(self.width_cpp, int):
            raise TypeError(f"width_cpp must be an integer, got {self.width_cpp!r}")
        minimum = MINIMUM_WIDTH_CPP[self.kind]
        if self.width_cpp < minimum:
            raise ValueError(
                f"a {self.kind} needs width_cpp >= {minimum} to fit its "
                f"geometry on the 54 nm gate grid; got {self.width_cpp}"
            )

    # ── Naming ────────────────────────────────────────────────────────────────
    @property
    def cell_name(self) -> str:
        """e.g. ``"tap_fin_18n18p_13n13p_2cpp"``."""
        vt_tag = "" if self.stack.vt == "rvt" else f"_{self.stack.vt}"
        return f"{self.kind}_fin_{self.stack.code}_{self.width_cpp}cpp{vt_tag}"

    # ── Geometry ──────────────────────────────────────────────────────────────
    @property
    def width(self) -> int:
        return self.width_cpp * GATE_PITCH

    @property
    def height(self) -> int:
        return self.stack.height

    @property
    def bands(self) -> tuple[RowBand, ...]:
        return self.stack.bands()

    @property
    def rails(self) -> tuple[tuple[int, str], ...]:
        return self.stack.rails

    @property
    def gate_grid_xs(self) -> list[int]:
        """Center X of every drawn poly track."""
        return list(range(GATE_PITCH // 2, self.width, GATE_PITCH))

    @property
    def gate_xs(self) -> list[int]:
        """Gate tracks a decap uses as capacitor plates; the rest are dummies."""
        return self.gate_grid_xs[1:-1] if self.kind == "decap" else []

    @property
    def tie_xs(self) -> list[int]:
        """Interior source/drain columns.

        The tracks *between* two gates, so a contact on one is enclosed by
        poly on both sides exactly as it is in a logic cell.  The released
        TAPCELL puts its single tie on the one such column a 2 CPP cell has.
        A filler has nothing to tie but still places a contact there, stitching
        its local-interconnect rail to the power rail above rather than leaving
        a floating strip of LI across the cell.
        """
        return [(index + 1) * GATE_PITCH for index in range(self.width_cpp - 1)]

    @property
    def active_x(self) -> tuple[float, float]:
        """X extent of a decap's plate diffusion, inset for select enclosure."""
        return SELECT_X_ENC, self.width - SELECT_X_ENC

    def active_y(self, band: RowBand) -> tuple[float, float]:
        """Y extent of this band's ACTIVE, whatever the cell holds.

        A tap fills the band right up to its select enclosure -- more tie area
        for the same cell -- while a decap keeps to the fin-quantized span a
        device in that band would use, so its plate really is a MOS capacitor
        on the fin grid.
        """
        if self.kind == "tap":
            return band.y0 + SELECT_Y_ENC, band.y1 - SELECT_Y_ENC
        return band.active_span

    def tie_fins(self, band: RowBand) -> int:
        """Fin height of `band`'s tie, which the runset checks like any ACTIVE."""
        lo, hi = self.active_y(band)
        return round((hi - lo) / 27)

    @property
    def drc_verifiable(self) -> bool:
        """True when the public runset can check every ACTIVE height here.

        A tap fills its band, so on a stack of tall bands its tie is taller
        than the deck's enumerated height list even though a tap has no fins
        to speak of -- see
        `chipforge_asap7.devices.finfet.MAX_VERIFIABLE_FINS`.  A filler has no
        ACTIVE at all and is always checkable.
        """
        if self.kind == "filler":
            return True
        return all(self.tie_fins(band) <= MAX_VERIFIABLE_FINS for band in self.bands)

    # ── Electrical ────────────────────────────────────────────────────────────
    @property
    def devices(self) -> tuple[tuple[RowBand, int], ...]:
        """``(band, unit_fin_count)`` for every capacitor a decap contains."""
        if self.kind != "decap":
            return ()
        return tuple((band, band.fins * len(self.gate_xs)) for band in self.bands)

    @property
    def pin_positions(self) -> dict[str, tuple[float, float]]:
        rails = {net: y for y, net in reversed(self.rails)}
        return {net: (self.width / 2, y) for net, y in rails.items()}


# ── Layout ────────────────────────────────────────────────────────────────────
def _square(cell: Any, layer_name: str, x: float, y: float, size: float) -> None:
    half = size / 2
    box(cell, layer_name, x - half, y - half, x + half, y + half)


def build_row_support(
    spec: RowSupportSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Draw `spec` and return the gdspy Cell."""
    spec = spec or RowSupportSpec()
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = (
        lib.new_cell(cell_name)
        if lib is not None
        else gdspy.Cell(cell_name, exclude_from_current=True)
    )

    width = spec.width
    half_m1 = M1_WIDTH / 2
    half_sd = SD_BAR_WIDTH / 2
    half_gate = GATE_WIDTH / 2
    half_cut = DEVICE_GATE_CUT_HEIGHT / 2
    half_lig = GATE_LIG_HEIGHT / 2
    half_rail = LI_RAIL_HEIGHT / 2
    stack = spec.stack

    # Implant, band by band.  This one inversion -- a tap doped opposite to the
    # devices it serves -- is the whole difference between the released TAPCELL
    # and FILLER, and the reason a tap cannot sit straight against a logic cell.
    for band in spec.bands:
        implant = band.tap_implant if spec.kind == "tap" else band.implant
        box(cell, implant, 0, band.y0, width, band.y1)
        if band.in_nwell:
            box(cell, "NWELL", 0, band.y0, width, band.y1)
    if vt_layer := VT_LAYERS[stack.vt]:
        box(cell, vt_layer, 0, 0, width, spec.height)

    for y_fin in stack.fin_grid_ys:
        box(cell, "FIN", 0, y_fin, width, y_fin + FIN_WIDTH)
    for x_gate in spec.gate_grid_xs:
        box(
            cell,
            "GATE",
            x_gate - half_gate,
            -POLY_OVERHANG,
            x_gate + half_gate,
            spec.height + POLY_OVERHANG,
        )

    # Cut the poly at every rail; it is dummy on both sides of one.
    for y_rail, _net in spec.rails:
        box(cell, "GATE_CUT", 0, y_rail - half_cut, width, y_rail + half_cut)
    # At a seam, cut whatever is dummy there.  In a tap or filler that is the
    # whole width; in a decap only the two edge tracks, because the plates have
    # to stay continuous through the seam to share one gate conductor -- the
    # same split the released DECAPx4 draws.
    seam_spans = (
        [(0, width)]
        if spec.kind != "decap"
        else [(0, GATE_PITCH), (width - GATE_PITCH, width)]
    )
    for y_seam in stack.seam_ys:
        for x0, x1 in seam_spans:
            box(cell, "GATE_CUT", x0, y_seam - half_cut, x1, y_seam + half_cut)

    # Rails run the full width so an abutted row's power is continuous.  The
    # LI rail under each one is not decoration: V0 is 18 nm and V0.AUX.1-2
    # oversizes LIG by 1 nm before asking whether the contact landed on local
    # interconnect, so a 16 nm rail is exactly enough -- and without it every
    # contact sitting on a power rail hangs over the end of its LISD bar.
    for y_rail, _net in spec.rails:
        box(cell, "LIG", 0, y_rail - half_rail, width, y_rail + half_rail)
        box(cell, "M1", 0, y_rail - half_m1, width, y_rail + half_m1)

    # Ties.  A tap's ACTIVE ties the well or substrate; a decap's is one plate
    # of a capacitor.  Both reach their rail the same way the released TAPCELL
    # does: LISD carries on past the diffusion to the rail, and the contact
    # sits on the rail itself.  There is no stub to draw -- the rail is already
    # 18 nm of M1 running the full width, which is exactly the landing
    # V0.M1.AUX.3 asks for.
    for band in spec.bands:
        if spec.kind == "filler":
            continue
        act_lo, act_hi = spec.active_y(band)
        lisd_lo = band.rail_y if band.rail_below else act_lo
        lisd_hi = act_hi if band.rail_below else band.rail_y
        if spec.kind == "decap":
            box(cell, "ACTIVE", spec.active_x[0], act_lo, spec.active_x[1], act_hi)
        for x_tie in spec.tie_xs:
            if spec.kind == "tap":
                half_tie = TAP_ACTIVE_WIDTH / 2
                box(cell, "ACTIVE", x_tie - half_tie, act_lo, x_tie + half_tie, act_hi)
            box(cell, "SDT", x_tie - half_sd, act_lo, x_tie + half_sd, act_hi)
            box(cell, "LISD", x_tie - half_sd, lisd_lo, x_tie + half_sd, lisd_hi)
    # One contact per rail per column, not one per band: two stacked rows share
    # the rail between them, and so would their vias.
    for y_rail, _net in spec.rails:
        for x_tie in spec.tie_xs:
            _square(cell, "V0", x_tie, y_rail, CONTACT_SIZE)

    # A decap's two plates share one gate conductor, strapped where no S/D bar
    # crosses.  It carries no via: the node between the two capacitors floats,
    # which is what keeps either oxide off the full supply.
    if spec.gate_xs:
        for y_seam in stack.seam_ys:
            box(
                cell,
                "LIG",
                spec.gate_xs[0] - half_lig,
                y_seam - half_lig,
                spec.gate_xs[-1] + half_lig,
                y_seam + half_lig,
            )

    if draw_pin_labels:
        for y_rail, net in spec.rails:
            cell.add(
                gdspy.Label(
                    net,
                    (width / 2, y_rail),
                    layer=_M1_PIN_LAYER,
                    texttype=_M1_PIN_TEXTTYPE,
                )
            )

    for row, y0 in enumerate(stack.row_ys[:-1]):
        box(cell, "BOUNDARY", 0, y0, width, stack.row_ys[row + 1])
    return cell


def main(argv: list[str] | None = None) -> None:
    """Write a row-support GDS: ``asap7-row-support --kind tap --stack.rows 4 6``."""
    args = parse_spec(
        RowSupportSpec,
        argv,
        description="Generate an ASAP7 tap, filler or decap for a row stack.",
    )
    spec = args.spec
    cell, out = write_gds(lambda lib: build_row_support(spec, lib=lib), args.out)
    print(f"✓ {cell.name}: {spec.width} x {spec.height} nm -> {out}")
    if spec.kind == "decap":
        fins = sum(count for _band, count in spec.devices)
        print(f"  {len(spec.gate_xs)} plates per band, {fins} unit-fin capacitors")
    elif spec.kind == "tap":
        print(f"  {len(spec.tie_xs)} tie column(s) x {len(spec.bands)} bands")


if __name__ == "__main__":  # pragma: no cover
    main()
