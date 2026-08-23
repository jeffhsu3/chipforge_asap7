"""Parametric ASAP7 FinFET, sized in fins rather than microns.

The planar generator this mirrors (OpenFASOC glayout's `primitives/fet.py`)
takes a continuous ``width`` in microns and snaps it to grid.  On a FinFET that
knob does not exist: drive strength comes in whole fins, so the device is
parameterised by three **discrete counts** and named after them —

    nmos_fin_122   ->  1 fin, 2 fingers, 2 multipliers

the same fin-count nomenclature as OpenFinRAM's ``sram_cell_6t_122``
(``nfin_pu=1, nfin_pd=2, nfin_pg=2``), instead of the planar ``W=3u`` style.
`FinFETSpec.code` builds that digit string; counts of 10 or more switch to
``_``-separated fields (``12_2_1``) so the name stays unambiguous.

Layout is a multi-finger device on the shared ASAP7 grid from
`chipforge_asap7.layout` — fins on the 27 nm fin pitch, gates on the 54 nm
contacted-poly pitch, source/drain columns midway between gates::

    +--------------------------------+   multipliers are folded into fingers
    |   S     G     D     G     S    |   fins run in X, stacked in Y
    |  ###   |||   ###   |||   ###   |   gates run in Y
    +--------------------------------+

Fingers share source/drain columns and alternate S, D, S, ... from the left, so
an even physical finger count ends on a source (as drawn above) and an odd one
ends on a drain.  Multipliers are folded into additional parallel fingers.  A horizontal LIG strap joins every
gate, while standard-cell-style M1 shapes tie all source columns and all drain
columns.  The cell also contains the opposite-polarity well/substrate tap, so
the fourth (body) terminal is real rather than just a netlist placeholder.

The geometry follows the released ASAP7 7.5-track standard cells: 27 nm-tall
ACTIVE increments with 10 nm fin enclosure, 46/27 nm select enclosure, dummy
gates on both sides, 24 nm SDT/LISD, 18 nm V0, and the same rail/contact shapes used
by ``INVxp33_ASAP7_75t_R`` and ``TAPCELL_ASAP7_75t_R``.  The extra tap column
and dummy geometry are intentional; a bare channel is neither standalone DRC
clean nor a complete four-terminal transistor.

Everything except the drawing itself is pure arithmetic and needs no GDS
library: `FinFETSpec` alone answers size, pin-location and netlist questions.

The supported parameter space is deliberately the legal ASAP7 device grid:
20 nm gates at one CPP, one through `MAX_FINS` fins, and no arbitrary S/D
offset.  Above `MAX_VERIFIABLE_FINS` the public KLayout runset can no longer
check the fin-height rules -- see the note on those two constants -- so
`FinFETSpec.drc_verifiable` reports which side of that line a device is on.
Rejecting an off-grid request is preferable to silently emitting a known DRC
violation.  DRC regression is performed with the public ASAP7 KLayout runset;
electrical regression checks the four independently routed terminals and every
FIN/ACTIVE/GATE channel intersection.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ..layout.grid import (
    FIN_PITCH,
    FIN_WIDTH,
    GATE_PITCH,
    GATE_WIDTH,
)
from ..layout.layers import LAYERS, box, require_gdspy

__all__ = [
    "ACTIVE_ENC",
    "CONTACT_SIZE",
    "DEVICE_GATE_CUT_HEIGHT",
    "GATE_CUT_MIN_SPACE",
    "GATE_LIG_HEIGHT",
    "GATE_SD_SPACE",
    "GATE_V0_DX",
    "LIG_GATE_EXTENSION",
    "LI_RAIL_HEIGHT",
    "M1_BOUNDARY_INSET",
    "M1_MIN_SPACE",
    "M1_V0_ENCLOSURE",
    "M1_WIDTH",
    "MAX_FINS",
    "MAX_VERIFIABLE_FINS",
    "POLY_OVERHANG",
    "ROUTING_BAND_HEIGHT",
    "SD_BAR_WIDTH",
    "SELECT_X_ENC",
    "SELECT_Y_ENC",
    "TAP_ACTIVE_WIDTH",
    "TAP_COLUMN_WIDTH",
    "V0_LISD_ENCLOSURE",
    "VT_LAYERS",
    "FinFETSpec",
    "build_device_band",
    "build_finfet",
    "nmos_fin",
    "pmos_fin",
]

# ── Device rules (nm), matching the released ASAP7 7.5-track cells ─────
CONTACT_SIZE = 18  # V0 square between local interconnect and M1
# V0.M1.AUX.3 requires M1 to be exactly as wide as the V0 landing on it,
# measured across the M1 track, so these two are equal by rule and not by
# coincidence.  Metal shapes are still drawn from M1_WIDTH and vias from
# CONTACT_SIZE, so neither one silently resizes the other.
M1_WIDTH = CONTACT_SIZE  # M1 landing pad width
M1_V0_ENCLOSURE = 5  # V0.M1.EN.1: M1 end-cap past V0 along the track
M1_BOUNDARY_INSET = 18  # M1.S.1 clearance the drain pad keeps from the cell edge
SD_BAR_WIDTH = 24  # LISD source/drain bar width (x_sd +/- 12)
GATE_SD_SPACE = 5  # gate-edge to LISD-edge spacing in the ASAP7 cells
POLY_OVERHANG = 5  # released cells run GATE 5 nm beyond placement boundaries
ACTIVE_ENC = 10  # makes ACTIVE height exactly nfin * 27 nm

SELECT_X_ENC = 46
SELECT_Y_ENC = 27
TAP_COLUMN_WIDTH = 2 * GATE_PITCH
TAP_ACTIVE_WIDTH = 16  # ACTIVE.W.3 minimum horizontal width
DEVICE_GATE_CUT_HEIGHT = 44
LI_RAIL_HEIGHT = 16  # LIG source/body rail; LIG.LISD.OV.1 wants 8 nm of overlap
GATE_LIG_HEIGHT = 22  # LIG strap joining every active gate finger
LIG_GATE_EXTENSION = 2  # LIG.GATE.EX.1 minimum is 1 nm
GATE_V0_DX = 10  # gate V0 offset from the first S/D column, toward the gate
# GCUT.S.3 needs 35 nm between the gate-access cut and the top boundary cut, so
# the routing band can never be shorter than DEVICE_GATE_CUT_HEIGHT + 35 = 79
# nm.  Three fin pitches is the next height on the grid and leaves 2 nm spare;
# this constant is load-bearing, not just a comfortable amount of room.
ROUTING_BAND_HEIGHT = 3 * FIN_PITCH

# Two different ceilings, for two different reasons.
#
# The public KLayout runset expresses "ACTIVE/SDT vertical height is an integer
# multiple of 27 nm" by enumerating 1x27 .. 12x27 -- KLayout's DRC language has
# no integer-multiple predicate -- so ACTIVE.W.2 and SDT.W.3 fire on anything
# taller even though it *is* a multiple of 27.  Twelve is where that
# enumeration stops; it is not a process limit.  The released ASAP7 SRAM banks
# ship 13-, 14- and 18-fin devices (`dec_inv_62f_halved_AND` has both an 18-fin
# and a 13-fin pair), and running the runset on that cell reports exactly those
# two rules against ASU's own collateral.
MAX_VERIFIABLE_FINS = 12
# The tallest device in the released ASAP7 collateral.  No process maximum is
# documented anywhere in the PDK, so this is an evidence-based bound rather
# than a rule; raise it if a design needs more.
MAX_FINS = 18

# ── Runset limits the fixed geometry above has no slack against ───────────
# The drawn cell sits exactly on these three rules.  `test_finfet` asserts the
# drawn margins against them so a geometry edit fails immediately instead of
# waiting for the next machine that happens to have KLayout installed.
GATE_CUT_MIN_SPACE = 35  # GCUT.S.3
M1_MIN_SPACE = 18  # M1.S.1, both edges > 36 nm
V0_LISD_ENCLOSURE = 3  # V0.LISD.EN.2, on at least two opposite sides

_ISOLATION_FINS = 2  # one select-enclosure fin pitch above and below ACTIVE
#: Threshold flavor -> its layout marker layer, or None for the RVT default.
VT_LAYERS: dict[str, str | None] = {
    "rvt": None,
    "lvt": "LVT",
    "slvt": "SLVT",
    "sram": "SRAMVT",
}
_M1_PIN_LAYER = LAYERS["M1_PIN"]["layer"]
_M1_PIN_TEXTTYPE = LAYERS["M1_PIN"]["datatype"]


@dataclass(frozen=True)
class FinFETSpec:
    """A FinFET sized in fins, fingers and multipliers.

    Args:
        flavor: ``"n"`` or ``"p"``.
        fins: active fins per finger — the only width knob a FinFET has.
        fingers: gates sharing source/drain columns.
        multipliers: parallel copies, folded into the active finger array.
        finger_pitch_cpp: gate pitch in CPP.  ASAP7 DRC requires exactly one.
        gate_length: drawn gate width.  ASAP7 DRC requires exactly 20 nm.
        sd_dx: retained for API compatibility; non-``None`` values are rejected
            because an off-grid S/D column cannot be DRC clean.
        row_height: height of the ACTIVE/select band.  Defaults to
            ``(fins + 2) * FIN_PITCH`` and may be enlarged by whole fin pitches.
        vt: threshold-voltage model: ``"rvt"``, ``"lvt"``, ``"slvt"`` or
            ``"sram"``.  Non-RVT choices also add the corresponding marker.
    """

    flavor: Literal["n", "p"] = "n"
    fins: int = 1
    fingers: int = 1
    multipliers: int = 1
    finger_pitch_cpp: int = 1
    gate_length: int = GATE_WIDTH
    sd_dx: int | None = None
    row_height: int | None = None
    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"

    def __post_init__(self) -> None:
        if self.flavor not in ("n", "p"):
            raise ValueError(f"flavor must be 'n' or 'p', got {self.flavor!r}")
        if self.vt not in VT_LAYERS:
            raise ValueError(f"vt must be one of {tuple(VT_LAYERS)}, got {self.vt!r}")

        required_ints = (
            "fins",
            "fingers",
            "multipliers",
            "finger_pitch_cpp",
            "gate_length",
        )
        optional_ints = ("sd_dx", "row_height")
        for field in required_ints:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an integer, got {value!r}")
        for field in optional_ints:
            value = getattr(self, field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"{field} must be an integer or None, got {value!r}")

        for field in ("fins", "fingers", "multipliers", "finger_pitch_cpp"):
            value = getattr(self, field)
            if value < 1:
                raise ValueError(f"{field} must be >= 1, got {value}")
        if self.fins > MAX_FINS:
            raise ValueError(
                f"fins must be <= {MAX_FINS}, the tallest device in the "
                f"released ASAP7 collateral; got {self.fins}"
            )
        if self.finger_pitch_cpp != 1:
            raise ValueError("ASAP7 DRC requires finger_pitch_cpp=1 (54 nm CPP)")
        if self.gate_length != GATE_WIDTH:
            raise ValueError(
                f"ASAP7 DRC requires gate_length={GATE_WIDTH} nm, got "
                f"{self.gate_length}"
            )
        if self.sd_dx is not None:
            raise ValueError(
                "sd_dx overrides are not DRC-safe; ASAP7 S/D columns are fixed "
                "halfway between adjacent 54 nm gate tracks"
            )

        band_height = self.height_per_row
        if band_height % FIN_PITCH:
            raise ValueError(
                f"row_height {band_height} is not a multiple of fin pitch {FIN_PITCH}"
            )
        minimum = self.default_height_per_row
        if band_height < minimum:
            raise ValueError(
                f"row_height {band_height} is too short; {self.fins} fins plus "
                f"27 nm select enclosure on each side and a legal body tap "
                f"need >= {minimum} nm"
            )
        maximum = (MAX_FINS + _ISOLATION_FINS) * FIN_PITCH
        if band_height > maximum:
            raise ValueError(
                f"row_height {band_height} is too tall; the body-tap ACTIVE "
                f"would exceed the {MAX_FINS}-fin device limit "
                f"({maximum} nm select band)"
            )

    # ── Naming ────────────────────────────────────────────────────────────────
    @property
    def code(self) -> str:
        """Fin-count code: fins, fingers, multipliers — e.g. ``"122"``."""
        counts = (self.fins, self.fingers, self.multipliers)
        joiner = "" if all(c < 10 for c in counts) else "_"
        return joiner.join(str(c) for c in counts)

    @property
    def cell_name(self) -> str:
        """Default cell name, e.g. ``"nmos_fin_122"``.

        A non-default `row_height` changes the drawn cell without changing any
        of the three counts, so it is appended as ``_h<nm>``.  Without it two
        different geometries would claim one name and collide the moment they
        met in the same library.
        """
        vt_tag = "" if self.vt == "rvt" else f"_{self.vt}"
        band_tag = (
            ""
            if self.height_per_row == self.default_height_per_row
            else f"_h{self.height_per_row}"
        )
        return f"{self.flavor}mos{vt_tag}_fin_{self.code}{band_tag}"

    # ── Geometry (pure arithmetic, no GDS library needed) ─────────────────────
    @property
    def finger_pitch(self) -> int:
        """Gate-to-gate pitch in nm."""
        return self.finger_pitch_cpp * GATE_PITCH

    @property
    def gate_sd_clearance(self) -> float:
        """Gate-edge to S/D-bar-edge spacing (SDT.GATE.S.2, LISD clearance).

        Fixing the CPP and the gate length pins this at `GATE_SD_SPACE`, the
        value the released cells use; composite generators that route past a
        finger need the number rather than the constant behind it.
        """
        return self.finger_pitch / 2 - self.gate_length / 2 - SD_BAR_WIDTH / 2

    @property
    def default_height_per_row(self) -> int:
        """Height of the ACTIVE/select band when `row_height` is not given."""
        return max(self.fins + _ISOLATION_FINS, 4) * FIN_PITCH

    @property
    def height_per_row(self) -> int:
        """Height of the ACTIVE/select band in nm."""
        if self.row_height is not None:
            return self.row_height
        return self.default_height_per_row

    @property
    def tap_fins(self) -> int:
        """Fin height of the body-tap ACTIVE, which the runset also checks."""
        return (self.height_per_row - 2 * SELECT_Y_ENC) // FIN_PITCH

    @property
    def drc_verifiable(self) -> bool:
        """True when the public KLayout runset can check this device's heights.

        A taller device is still legal ASAP7 -- the released SRAM is full of
        them -- but the runset's enumerated height list stops at
        `MAX_VERIFIABLE_FINS`, so it reports a false ACTIVE.W.2 / SDT.W.3
        against anything above it.  The DRC regression uses this to keep its
        matrix inside what the deck can actually decide.
        """
        return max(self.fins, self.tap_fins) <= MAX_VERIFIABLE_FINS

    @property
    def layout_fingers(self) -> int:
        """Physical gate count after folding multipliers into the finger row."""
        return self.fingers * self.multipliers

    @property
    def device_x0(self) -> int:
        """Left edge of the transistor portion, after the body-tap column."""
        return TAP_COLUMN_WIDTH

    @property
    def device_width(self) -> int:
        """Width of the transistor portion, including two dummy gate tracks."""
        return (self.layout_fingers + 2) * GATE_PITCH

    @property
    def width(self) -> int:
        """Full DRC-clean cell width, including the body-tap column."""
        return TAP_COLUMN_WIDTH + self.device_width

    @property
    def height(self) -> int:
        """Full cell height, including the gate-access routing band."""
        return self.height_per_row + ROUTING_BAND_HEIGHT

    @property
    def gate_xs(self) -> list[int]:
        """Center X of every electrically active physical gate finger."""
        first = self.device_x0 + 3 * GATE_PITCH // 2
        return [first + i * GATE_PITCH for i in range(self.layout_fingers)]

    @property
    def gate_grid_xs(self) -> list[int]:
        """All gate-grid centers, including tap and edge dummy gates."""
        return list(range(GATE_PITCH // 2, self.width, GATE_PITCH))

    @property
    def sd_xs(self) -> list[int]:
        """Center X of each source/drain column, left to right.

        There is one more column than there are fingers — adjacent fingers
        share the column between them, which is the point of fingering.
        """
        first = self.device_x0 + GATE_PITCH
        return [first + j * GATE_PITCH for j in range(self.layout_fingers + 1)]

    @property
    def source_xs(self) -> list[int]:
        """Source columns — the even ones, counting from the left."""
        return self.sd_xs[0::2]

    @property
    def drain_xs(self) -> list[int]:
        """Drain columns — the odd ones."""
        return self.sd_xs[1::2]

    def fin_ys(self, row: int = 0) -> list[float]:
        """Bottom Y of each active fin (multipliers are folded in X)."""
        if row != 0:
            raise IndexError("multipliers are folded into the physical finger row")
        spare_pitches = self.height_per_row // FIN_PITCH - self.fins
        active_lo = (spare_pitches // 2) * FIN_PITCH
        return [active_lo + ACTIVE_ENC + i * FIN_PITCH for i in range(self.fins)]

    @property
    def fin_grid_ys(self) -> list[int]:
        """Bottom Y of the full manufacturing fin grid across the cell."""
        return [ACTIVE_ENC + i * FIN_PITCH for i in range(self.height // FIN_PITCH)]

    def active_span(self, row: int = 0) -> tuple[float, float]:
        """Bottom/top Y of the fin-quantized ACTIVE region."""
        ys = self.fin_ys(row)
        return ys[0] - ACTIVE_ENC, ys[-1] + FIN_WIDTH + ACTIVE_ENC

    def gate_contact_y(self, row: int = 0) -> int:
        """Y of the gate access at the select/routing-band boundary."""
        if row != 0:
            raise IndexError("multipliers are folded into the physical finger row")
        return self.height_per_row

    def source_contact_y(self, row: int = 0) -> int:
        """Y of source V0s on the lower source/body rail."""
        if row != 0:
            raise IndexError("multipliers are folded into the physical finger row")
        return 0

    def drain_contact_y(self, row: int = 0) -> float:
        """Y of drain V0s at the lower edge of ACTIVE."""
        act_lo, _ = self.active_span(row)
        return act_lo + CONTACT_SIZE / 2

    def sd_contact_y(self, row: int = 0) -> float:
        """Deprecated alias for `drain_contact_y`.

        The name reads as the source-or-drain accessor next to
        `source_contact_y`, which it never was.
        """
        return self.drain_contact_y(row)

    @property
    def pin_positions(self) -> dict[str, tuple[float, float]]:
        """M1 access point for each terminal, in cell-local nanometres.

        Composite generators use these coordinates to place upper-metal vias
        without having to rediscover the leaf device's internal geometry.
        """

        act_lo, act_hi = self.active_span()
        return {
            "B": (GATE_PITCH, self.height_per_row),
            "S": (self.source_xs[0], 0),
            "D": (self.drain_xs[0], (act_lo + act_hi) / 2),
            "G": (self.sd_xs[0] + GATE_V0_DX, self.gate_contact_y()),
        }

    @property
    def total_fins(self) -> int:
        """Fins in the whole device — what actually sets drive strength."""
        return self.fins * self.fingers * self.multipliers

    # ── Netlist ───────────────────────────────────────────────────────────────
    @property
    def model(self) -> str:
        """ASAP7 BSIM-CMG model-card name."""
        return f"{self.flavor}mos_{self.vt}"

    def netlist(
        self,
        instance: str = "M0",
        nodes: tuple[str, str, str, str] = ("D", "G", "S", "B"),
    ) -> str:
        """A BSIM-CMG style instance line, sized by `nfin` rather than by width."""
        d, g, s, b = nodes
        return (
            f"{instance} {d} {g} {s} {b} {self.model} "
            f"nfin={self.fins} l={self.gate_length}n "
            f"nf={self.fingers} m={self.multipliers}"
        )


# ── Layout ────────────────────────────────────────────────────────────────────
def build_device_band(
    cell: Any,
    spec: FinFETSpec,
    *,
    y0: float = 0.0,
    sd_xs: Sequence[float] | None = None,
    active_x: tuple[float, float] | None = None,
    source_lisd_y: float | None = None,
) -> tuple[float, float]:
    """Draw one transistor band's diffusion stack; return its ACTIVE span.

    A *band* is the part of a FinFET that is the same whether the device is a
    standalone tile (`build_finfet`) or one of several rows stacked on shared
    rails inside a composite cell (`chipforge_asap7.devices.inverter`): one
    fin-quantized ACTIVE rectangle and a `SD_BAR_WIDTH`-wide SDT + LISD bar on
    every source/drain column.

    Everything else -- body taps, vias, straps, rails, implant, wells and gate
    cuts -- stays with the caller, because that is exactly where an isolated
    tile and an abutting array cell disagree.  Sharing more than this would
    mean making the tile's tap, dummy gates and routing band optional, which
    is three new ways for the standalone device to come out wrong.

    Args:
        cell: target gdspy Cell.
        spec: the band's device.  Only `fins`, the band height and the column
            arithmetic are read; `flavor`, `vt` and wells stay with the caller.
        y0: Y of the band's select-band bottom, in cell coordinates.
        sd_xs: source/drain column centers, overriding `spec.sd_xs`.  A
            composite cell places its own columns.
        active_x: ``(x0, x1)`` for the ACTIVE rectangle, overriding the inset
            island `build_finfet` draws.  An abutting cell passes an overhang
            here so its diffusion merges with its neighbour's.
        source_lisd_y: when given, LISD on the source columns runs down to
            this Y instead of stopping at the ACTIVE edge -- how `build_finfet`
            reaches the source rail below its channel.

    Returns:
        ``(act_lo, act_hi)`` in cell coordinates.
    """
    columns = list(spec.sd_xs if sd_xs is None else sd_xs)
    if active_x is None:
        active_x = (spec.device_x0 + SELECT_X_ENC, spec.width - SELECT_X_ENC)
    rel_lo, rel_hi = spec.active_span()
    act_lo, act_hi = rel_lo + y0, rel_hi + y0

    box(cell, "ACTIVE", active_x[0], act_lo, active_x[1], act_hi)
    half_sd = SD_BAR_WIDTH / 2
    for index, x_sd in enumerate(columns):
        box(cell, "SDT", x_sd - half_sd, act_lo, x_sd + half_sd, act_hi)
        # Columns alternate S, D, S, ... from the left, so the even ones are
        # the sources that may have to reach a rail below the band.
        lisd_lo = act_lo if source_lisd_y is None or index % 2 else source_lisd_y
        box(cell, "LISD", x_sd - half_sd, lisd_lo, x_sd + half_sd, act_hi)
    return act_lo, act_hi


def build_finfet(
    spec: FinFETSpec,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Draw `spec` and return the gdspy Cell.

    Args:
        spec: the device to draw.
        name: cell name; defaults to `spec.cell_name`.
        lib: a `gdspy.GdsLibrary` to create the cell in.  When omitted the cell
            is standalone (`exclude_from_current=True`), so building several
            devices in one process cannot collide in gdspy's global library.
        draw_pin_labels: emit the leaf-level ``D/G/S/B`` labels.  Composite
            cells disable these and expose only their own top-level pins.
    """
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = (
        lib.new_cell(cell_name)
        if lib is not None
        else gdspy.Cell(cell_name, exclude_from_current=True)
    )

    half_gate = spec.gate_length / 2
    half_sd = SD_BAR_WIDTH / 2
    half_ct = CONTACT_SIZE / 2  # via
    half_m1 = M1_WIDTH / 2  # metal
    half_rail = LI_RAIL_HEIGHT / 2
    m1_pad = half_ct + M1_V0_ENCLOSURE  # M1 half-extent along its own track
    act_lo, act_hi = spec.active_span()
    select_hi = spec.height_per_row
    device_x0 = spec.device_x0
    device_x1 = spec.width
    body_x = GATE_PITCH

    device_select = "NSELECT" if spec.flavor == "n" else "PSELECT"
    tap_select = "PSELECT" if spec.flavor == "n" else "NSELECT"
    # NSELECT and PSELECT tile the whole cell with no gap, exactly as they do
    # in every released cell (INVxp33, NAND2xp33 and TAPCELL all split the full
    # boundary between the two).  Stopping them at the ACTIVE band would leave
    # the gate-access rows implant-less and make the cell unabuttable.
    box(cell, tap_select, 0, 0, TAP_COLUMN_WIDTH, spec.height)
    box(cell, device_select, device_x0, 0, device_x1, spec.height)

    # A pFET and its n-type body tap share one continuous n-well.  Its edge is
    # deliberately on the placement boundary, just as in the released cells;
    # ACTIVE and uncut GATE remain enclosed by at least 27 nm and 17 nm.
    if spec.flavor == "p":
        box(cell, "NWELL", 0, 0, spec.width, spec.height)
    if vt_layer := VT_LAYERS[spec.vt]:
        box(cell, vt_layer, device_x0, 0, device_x1, spec.height)

    # FIN and GATE are manufacturing grids, not shapes cropped to the channel.
    # Keeping the dummy tracks is required by the fixed-pitch ASAP7 rules.
    for y_fin in spec.fin_grid_ys:
        box(cell, "FIN", 0, y_fin, spec.width, y_fin + FIN_WIDTH)
    for x_gate in spec.gate_grid_xs:
        box(
            cell,
            "GATE",
            x_gate - half_gate,
            -POLY_OVERHANG,
            x_gate + half_gate,
            spec.height + POLY_OVERHANG,
        )

    half_gate_cut = DEVICE_GATE_CUT_HEIGHT / 2
    for y_cut in (0, spec.height):
        box(
            cell,
            "GATE_CUT",
            0,
            y_cut - half_gate_cut,
            spec.width,
            y_cut + half_gate_cut,
        )

    # At the gate-access band, cut every tap/dummy gate but leave all active
    # fingers continuous through their LIG strap.  This is the split central
    # GCUT pattern used by the ASAP7 inverter cells.
    for x0, x1 in (
        (0, device_x0 + GATE_PITCH),
        (device_x1 - GATE_PITCH, device_x1),
    ):
        box(
            cell,
            "GATE_CUT",
            x0,
            select_hi - half_gate_cut,
            x1,
            select_hi + half_gate_cut,
        )

    # The transistor ACTIVE extends 25 nm beyond the two outer active gate
    # edges.  This leaves a 9 nm gap to each edge dummy and gives exactly the
    # 46 nm select enclosure used by the standard cells.  Source LISD carries
    # on down to the y=0 rail; drain LISD stops at the ACTIVE edge.
    build_device_band(cell, spec, source_lisd_y=0)

    # The opposite-polarity tap occupies the gap between the tap column's two
    # dummy gates.  It is at least 864 nm² even for a one-fin transistor.
    tap_act_lo = SELECT_Y_ENC
    tap_act_hi = select_hi - SELECT_Y_ENC
    half_tap_act = TAP_ACTIVE_WIDTH / 2
    box(
        cell,
        "ACTIVE",
        body_x - half_tap_act,
        tap_act_lo,
        body_x + half_tap_act,
        tap_act_hi,
    )

    # Source columns reach the lower LI/M1 rail.  Drain columns contact at the
    # ACTIVE edge and are joined by one contiguous M1 landing.  This topology
    # avoids crossing alternating S/D columns while physically tying every
    # folded finger to the same S and D terminals.
    box(cell, "LIG", device_x0, -half_rail, device_x1, half_rail)
    box(cell, "M1", device_x0, -half_m1, device_x1, half_m1)
    for x_source in spec.source_xs:
        box(
            cell,
            "V0",
            x_source - half_ct,
            -half_ct,
            x_source + half_ct,
            half_ct,
        )

    y_drain = spec.drain_contact_y()
    for x_drain in spec.drain_xs:
        box(
            cell,
            "V0",
            x_drain - half_ct,
            y_drain - half_ct,
            x_drain + half_ct,
            y_drain + half_ct,
        )
    # One contiguous pad ties every drain column, ending on the V0.M1.EN.1
    # end-cap past the outermost drain via at each side.  Running it out to the
    # cell edge instead would drag the drain conductor across every source
    # column -- no short, but a large and entirely avoidable Cds.
    box(
        cell,
        "M1",
        spec.drain_xs[0] - m1_pad,
        act_lo,
        spec.drain_xs[-1] + m1_pad,
        act_hi,
    )

    # All active gates meet one LIG strap.  Its V0 is offset off the gate
    # centre toward the first source column, the same trick the released INV
    # cells use -- though they shift their V0 7 nm from the gate centre and
    # this one lands GATE_V0_DX past the S/D column, 17 nm off.  Either way the
    # M1 pin above it stays clear of the drain conductor.
    y_gate = spec.gate_contact_y()
    half_gate_lig = GATE_LIG_HEIGHT / 2
    gate_lig_x0 = spec.sd_xs[0]
    gate_lig_x1 = spec.gate_xs[-1] + half_gate + LIG_GATE_EXTENSION
    box(
        cell,
        "LIG",
        gate_lig_x0,
        y_gate - half_gate_lig,
        gate_lig_x1,
        y_gate + half_gate_lig,
    )
    gate_v0_x = spec.sd_xs[0] + GATE_V0_DX
    box(
        cell,
        "V0",
        gate_v0_x - half_ct,
        y_gate - half_ct,
        gate_v0_x + half_ct,
        y_gate + half_ct,
    )
    box(
        cell,
        "M1",
        gate_v0_x - half_m1,
        y_gate - m1_pad,
        gate_v0_x + half_m1,
        y_gate + m1_pad,
    )

    # Four-terminal body access.  ACTIVE/LISD reaches a top LI rail in the tap
    # column; V0 and M1 expose B without shorting it to the lower source rail.
    box(
        cell,
        "SDT",
        body_x - half_sd,
        tap_act_lo,
        body_x + half_sd,
        tap_act_hi,
    )
    box(cell, "LISD", body_x - half_sd, tap_act_lo, body_x + half_sd, select_hi)
    box(cell, "LIG", 0, select_hi - half_rail, TAP_COLUMN_WIDTH, select_hi + half_rail)
    box(
        cell,
        "V0",
        body_x - half_ct,
        select_hi - half_ct,
        body_x + half_ct,
        select_hi + half_ct,
    )
    box(cell, "M1", 0, select_hi - half_m1, TAP_COLUMN_WIDTH, select_hi + half_m1)

    if draw_pin_labels:
        for pin, origin in spec.pin_positions.items():
            cell.add(
                gdspy.Label(
                    pin,
                    origin,
                    layer=_M1_PIN_LAYER,
                    texttype=_M1_PIN_TEXTTYPE,
                )
            )

    box(cell, "BOUNDARY", 0, 0, spec.width, spec.height)
    return cell


def nmos_fin(
    fins: int = 1, fingers: int = 1, multipliers: int = 1, **kwargs: Any
) -> Any:
    """Build an n-type FinFET.

    See `FinFETSpec` for the sizing arguments and `build_finfet` for `name`,
    `lib` and `draw_pin_labels`; all of them pass straight through.
    """
    return _build("n", fins, fingers, multipliers, **kwargs)


def pmos_fin(
    fins: int = 1, fingers: int = 1, multipliers: int = 1, **kwargs: Any
) -> Any:
    """Build a p-type FinFET; see `nmos_fin` for the argument list."""
    return _build("p", fins, fingers, multipliers, **kwargs)


def _build(
    flavor: Literal["n", "p"],
    fins: int,
    fingers: int,
    multipliers: int,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
    **spec_kwargs: Any,
) -> Any:
    spec = FinFETSpec(
        flavor=flavor,
        fins=fins,
        fingers=fingers,
        multipliers=multipliers,
        **spec_kwargs,
    )
    return build_finfet(spec, name=name, lib=lib, draw_pin_labels=draw_pin_labels)


def main(argv: list[str] | None = None) -> None:
    """Write a device GDS: ``python -m chipforge_asap7.devices --fins 2``."""
    parser = argparse.ArgumentParser(
        description="Generate a parametric ASAP7 FinFET, sized in fins.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--flavor", choices=("n", "p"), default="n")
    parser.add_argument(
        "--vt", choices=tuple(VT_LAYERS), default="rvt", help="Threshold flavor."
    )
    parser.add_argument("--fins", type=int, default=1, help="Active fins per finger.")
    parser.add_argument("--fingers", type=int, default=1, help="Gates sharing S/D.")
    parser.add_argument(
        "--multipliers",
        type=int,
        default=1,
        help="Parallel copies folded into additional fingers.",
    )
    parser.add_argument(
        "--finger-pitch-cpp",
        type=int,
        default=1,
        help="Gate pitch in CPP; ASAP7 DRC requires 1.",
    )
    parser.add_argument(
        "--gate-length",
        type=int,
        default=GATE_WIDTH,
        help="Drawn gate width in nm; ASAP7 DRC requires 20.",
    )
    parser.add_argument("--out", default=None, help="Output GDS path.")
    args = parser.parse_args(argv)

    spec = FinFETSpec(
        flavor=args.flavor,
        vt=args.vt,
        fins=args.fins,
        fingers=args.fingers,
        multipliers=args.multipliers,
        finger_pitch_cpp=args.finger_pitch_cpp,
        gate_length=args.gate_length,
    )
    lib = require_gdspy().GdsLibrary(unit=1e-9, precision=1e-10)
    build_finfet(spec, lib=lib)
    out = args.out or f"{spec.cell_name}.gds"
    lib.write_gds(out)
    print(f"✓ {spec.cell_name}: {spec.width} x {spec.height} nm -> {out}")
    print(f"  {spec.total_fins} fins total ({spec.fins} per finger)")
    print(f"  {spec.netlist()}")


if __name__ == "__main__":  # pragma: no cover
    main()
