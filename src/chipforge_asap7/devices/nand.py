"""Parametric ASAP7 NAND2: the post-decode gate of the released row decoder.

`dec_nand_12f_12f_for_and_size_reduced_post_decode_P1N1` in
``asap7_sram_0p0/gds/srambank_32b.gds`` is the last NAND of the wordline AND
tree -- ``WL<i> = INV62(NAND(PA·WLENA, PB·PC<i>))`` -- and is drawn the same way
as `dec_inv_62f_halved_AND`: one tapless row whose ACTIVE, implant and edge
source columns overhang the tile so that neighbours merge.  Its column
pattern is a NAND's, not an inverter's::

    y=675 ─────────────── vdd rail ───────────────
            7-fin pFETs   VDD  B   Y   A  VDD  A   Y   B  VDD
    y=432 ─ ─ ─ ─ ─ ─ ─ ─ ─ seam: gate contacts ─ ─ ─ ─ ─
           14-fin nFETs   VSS  B   ·   A   Y   A   ·   B  VSS
    y=  0 ─────────────── vss rail ───────────────
                           0  27  54  81  108 135 162 189 216

Two series nFET stacks in parallel and four parallel pFETs: two fingers per
input.  The ``·`` columns are the uncontacted series nodes.  The A gates sit
on the output side of every stack and share one LIG pad; in the decoder they
carry the late, wordline-enable-gated select, which is why they are there.
The B gates are on the rail side and, with the A pair between them, cannot
share a pad.  The released cell leaves them for its parent to join on M2;
`build_nand` joins them itself with an M2 bar, so the cell is one NAND2 on
its own and LVS can say so.

`NandSpec(rows=((14, 7),), fingers=2)` is that tile, 216 x 675 nm.
``fingers`` is per input.  ``abut`` picks the released array style (even
fingers, so both edge columns are sources shared with the neighbours) or a
self-contained island with edge dummies; ``NandSpec(rows=((3, 3),),
fingers=1, abut=False)`` is the NAND2xp33 footprint on the 270 nm row.  One
row only for now: the M3 rail crossing the inverter uses for stacked rows
has no released NAND to validate against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..layout.grid import FIN_WIDTH, GATE_PITCH, GATE_WIDTH
from ..layout.layers import PIN_LAYERS, box, require_gdspy
from ..layout.rules import (
    CONTACT_SIZE,
    M1_MIN_SPACE,
    M1_V0_ENCLOSURE,
    M1_WIDTH,
    TRACK_PITCH,
)
from .cli import parse_spec, write_gds
from .finfet import (
    ACTIVE_ENC,
    DEVICE_GATE_CUT_HEIGHT,
    GATE_LIG_HEIGHT,
    LI_RAIL_HEIGHT,
    MAX_VERIFIABLE_FINS,
    POLY_OVERHANG,
    SD_BAR_WIDTH,
    SELECT_X_ENC,
    VT_LAYERS,
    build_device_band,
)
from .inverter import ACTIVE_ABUT_OVERHANG, SELECT_ABUT_OVERHANG
from .row import RowBand, RowStack

__all__ = [
    "LIG_PAD_EXTENSION",
    "NAND_PINS",
    "TIE_PITCH",
    "NandSpec",
    "build_nand",
    "build_nand_row",
]

NAND_PINS = ("A", "B", "Y", "VDD", "VSS")
#: An input whose gates cannot share one LIG pad is joined on an M2 bar with a
#: V1 on each of its M1 gate bars.  The bars sit below the seam, one M1 track
#: apart, so the gate bars that carry them stay clear of the drain-via rows.
TIE_PITCH = TRACK_PITCH
#: LIG past the gate edge on a contact pad.  LIG.GATE.EX.1 asks for 1 nm and
#: that is what the released decoder and standard cells draw; the 2 nm the
#: standalone FinFET tile uses would put two 22 nm pads on adjacent gates
#: 30 nm apart, one under LIG.S.4-5.
LIG_PAD_EXTENSION = 1

_HALF_CT = CONTACT_SIZE // 2
_HALF_M1 = M1_WIDTH // 2
_HALF_SD = SD_BAR_WIDTH // 2
_HALF_LIG = GATE_LIG_HEIGHT // 2
_HALF_CUT = DEVICE_GATE_CUT_HEIGHT // 2
_HALF_RAIL = LI_RAIL_HEIGHT // 2
_M1_PAD = _HALF_CT + M1_V0_ENCLOSURE  # 14: M1 past a via along its own track
#: A gate bar runs this far above its seam V0.  With 14 nm below, a bar that
#: carries no tie is 18 x 41 nm, clear of the 504 nm2 M1.A.1 minimum that an
#: 18 x 28 pad would sit exactly on.
_BAR_CAP_ABOVE = _HALF_M1 + M1_MIN_SPACE  # 27
_M1_PIN_LAYER, _M1_PIN_TEXTTYPE = PIN_LAYERS["M1"]

Role = Literal["A", "B"]


@dataclass(frozen=True)
class NandSpec:
    """A CMOS NAND2 with fingered inputs on one rail-sharing row.

    Args:
        rows: one ``(n_fins, p_fins)`` pair.  The released post-decode NAND is
            ``(14, 7)``: 14-fin series stacks against 7-fin pull-ups, two per
            input, which is the ``P1N1`` (equal drive) sizing.
        fingers: gates per input.  Gates run ``B A A B B A A B ...`` so every
            A gate is on the output side of its stack and every B gate on the
            rail side; an abutting cell needs an even count so that both edge
            columns are sources.
        vt: threshold-voltage model shared by both bands.
        abut: the released array style -- ACTIVE, implant and the edge source
            columns overhang the tile and neighbours butt together -- or a
            self-contained island with a dummy gate on each side.
        band_height: pin both bands to one height (see `RowStack`), e.g. 135
            for the 270 nm standard-cell row.
    """

    rows: tuple[tuple[int, int], ...] = ((14, 7),)
    fingers: int = 2
    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"
    abut: bool = True
    band_height: int | None = None

    def __post_init__(self) -> None:
        stack = RowStack(rows=self.rows, vt=self.vt, band_height=self.band_height)
        object.__setattr__(self, "rows", stack.rows)
        if len(self.rows) != 1:
            raise ValueError(
                "NandSpec draws one row; stacked rows are not supported yet"
            )
        if isinstance(self.fingers, bool) or not isinstance(self.fingers, int):
            raise TypeError(f"fingers must be an integer, got {self.fingers!r}")
        if self.fingers < 1:
            raise ValueError(f"fingers must be >= 1, got {self.fingers}")
        if self.abut and self.fingers % 2:
            raise ValueError(
                "an abutting NAND needs an even finger count so both edge "
                f"columns are sources; got {self.fingers} (or set abut=False)"
            )
        lo, hi = self.gate_bar_y
        if lo < self.n_band.contact_y + _HALF_M1 + M1_MIN_SPACE:
            raise ValueError(
                f"the {self.rows[0][0]}-fin n band is too short for the gate "
                f"contacts and {len(self.tie_ys)} input tie(s); use taller bands "
                "(band_height) or fewer fingers"
            )
        if hi > self.p_band.contact_y - _HALF_M1 - M1_MIN_SPACE:
            raise ValueError(
                f"the {self.rows[0][1]}-fin p band is too short for the gate "
                "contacts; use taller bands (band_height)"
            )

    # ── Band stack ────────────────────────────────────────────────────────────
    @property
    def stack(self) -> RowStack:
        """The row geometry this NAND shares with its row-support cells."""
        return RowStack(rows=self.rows, vt=self.vt, band_height=self.band_height)

    @property
    def code(self) -> str:
        return self.stack.code

    @property
    def cell_name(self) -> str:
        """e.g. ``"nand2_fin_14n7p_2f"`` for the released post-decode tile."""
        vt_tag = "" if self.vt == "rvt" else f"_{self.vt}"
        style_tag = "" if self.abut else "_iso"
        return f"nand2_fin_{self.code}_{self.fingers}f{vt_tag}{style_tag}"

    @property
    def bands(self) -> tuple[RowBand, ...]:
        """The n band then the p band, each carrying ``2 * fingers`` gates."""
        return self.stack.bands(2 * self.fingers)

    @property
    def n_band(self) -> RowBand:
        return self.bands[0]

    @property
    def p_band(self) -> RowBand:
        return self.bands[1]

    @property
    def rails(self) -> tuple[tuple[int, str], ...]:
        return self.stack.rails

    @property
    def seam_y(self) -> int:
        """The n/p boundary: the only Y no S/D bar crosses, where gates are contacted."""
        return self.stack.seam_y(0)

    @property
    def height(self) -> int:
        return self.stack.height

    # ── Columns and gates ─────────────────────────────────────────────────────
    @property
    def tracks(self) -> int:
        return 2 * self.fingers + (0 if self.abut else 2)

    @property
    def width(self) -> int:
        return self.tracks * GATE_PITCH

    @property
    def gate_grid_xs(self) -> list[int]:
        """Center X of every drawn poly track."""
        return list(range(GATE_PITCH // 2, self.width, GATE_PITCH))

    @property
    def gate_xs(self) -> list[int]:
        """Center X of the ``2 * fingers`` active gates."""
        tracks = self.gate_grid_xs
        return tracks if self.abut else tracks[1:-1]

    @property
    def gate_roles(self) -> tuple[Role, ...]:
        """``B A A B B A A B ...``: every stack reads rail, B, A, output."""
        roles: list[Role] = []
        for j in range(2 * self.fingers):
            half, position = divmod(j, 2)
            roles.append("B" if (half % 2 == 0) == (position == 0) else "A")
        return tuple(roles)

    @property
    def sd_xs(self) -> list[int]:
        """Center X of the ``2 * fingers + 1`` source/drain columns."""
        first = 0 if self.abut else GATE_PITCH
        return [first + i * GATE_PITCH for i in range(2 * self.fingers + 1)]

    def column_roles(self, flavor: str) -> tuple[str, ...]:
        """``"S"`` (rail), ``"Y"`` (output) or ``"x"`` (uncontacted series node) per column.

        The n band contacts every fourth column to a rail and the one between
        each pair of A gates to Y; the p band alternates rail and Y.
        """
        count = 2 * self.fingers + 1
        if flavor == "n":
            return tuple(
                "S" if i % 4 == 0 else "Y" if i % 4 == 2 else "x" for i in range(count)
            )
        return tuple("S" if i % 2 == 0 else "Y" for i in range(count))

    def columns(self, flavor: str, role: str) -> list[int]:
        return [x for x, r in zip(self.sd_xs, self.column_roles(flavor)) if r == role]

    @property
    def pads(self) -> tuple[tuple[Role, tuple[int, ...]], ...]:
        """Runs of consecutive same-input gates, each contacted by one LIG pad."""
        pads: list[tuple[Role, list[int]]] = []
        for x, role in zip(self.gate_xs, self.gate_roles):
            if pads and pads[-1][0] == role:
                pads[-1][1].append(x)
            else:
                pads.append((role, [x]))
        return tuple((role, tuple(xs)) for role, xs in pads)

    def pad_gate_xs(self, role: Role) -> list[int]:
        """The gate of each pad of `role` that carries its V0 and M1 bar."""
        return [xs[0] for r, xs in self.pads if r == role]

    @property
    def tie_ys(self) -> dict[Role, int]:
        """Y of the M2 bar joining the pads of every input that has more than one."""
        ties: dict[Role, int] = {}
        for role in ("B", "A"):
            if len(self.pad_gate_xs(role)) > 1:
                ties[role] = self.seam_y - TIE_PITCH * (len(ties) + 1)
        return ties

    @property
    def gate_bar_y(self) -> tuple[int, int]:
        """Y extent of every input's M1 bars: the seam V0 plus the tie rows below it."""
        lowest = min(self.tie_ys.values(), default=self.seam_y)
        return lowest - _M1_PAD, self.seam_y + _BAR_CAP_ABOVE

    def output_bar_x(self, n_drain_x: int) -> tuple[int, int]:
        """The vertical Y bar beside `n_drain_x`, over the next gate track.

        That gate is either the second A finger of the pad (no M1 bar of its
        own) or an edge dummy, so the track is free; the bars of the gates on
        either side are one M1 pitch away.  The bar is one 18 nm track centred
        on that gate, which also keeps an isolated tile's bar 18 nm inside
        its boundary.
        """
        start = n_drain_x + _HALF_CT + _HALF_M1
        return start, start + M1_WIDTH

    # ── Overhang ──────────────────────────────────────────────────────────────
    @property
    def active_x(self) -> tuple[int, int]:
        if self.abut:
            return -ACTIVE_ABUT_OVERHANG, self.width + ACTIVE_ABUT_OVERHANG
        return SELECT_X_ENC, self.width - SELECT_X_ENC

    @property
    def select_x(self) -> tuple[int, int]:
        if self.abut:
            return -SELECT_ABUT_OVERHANG, self.width + SELECT_ABUT_OVERHANG
        return 0, self.width

    @property
    def fin_x(self) -> tuple[int, int]:
        ax0, ax1 = self.active_x
        return min(0, ax0 - ACTIVE_ENC), max(self.width, ax1 + ACTIVE_ENC)

    @property
    def fin_grid_ys(self) -> list[int]:
        return self.stack.fin_grid_ys

    # ── Sizing and verification ───────────────────────────────────────────────
    @property
    def nfet_fins(self) -> int:
        """Fins in the pull-down: two series devices per stack, one stack per finger."""
        return 2 * self.fingers * self.rows[0][0]

    @property
    def pfet_fins(self) -> int:
        return 2 * self.fingers * self.rows[0][1]

    @property
    def total_fins(self) -> int:
        return self.nfet_fins + self.pfet_fins

    @property
    def drc_verifiable(self) -> bool:
        """True when the public runset can check both bands' ACTIVE heights."""
        return max(self.rows[0]) <= MAX_VERIFIABLE_FINS

    @property
    def pin_positions(self) -> dict[str, tuple[float, float]]:
        """One M1 point per pin, in cell-local nanometres."""
        y_bar = self.output_bar_x(self.columns("n", "Y")[0])
        rails = {net: y for y, net in reversed(self.rails)}
        return {
            "A": (self.pad_gate_xs("A")[0], self.seam_y),
            "B": (self.pad_gate_xs("B")[0], self.seam_y),
            "Y": ((y_bar[0] + y_bar[1]) / 2, self.seam_y),
            "VDD": (self.width / 2, rails["VDD"]),
            "VSS": (self.width / 2, rails["VSS"]),
        }

    # ── Netlist ───────────────────────────────────────────────────────────────
    def netlist(self, name: str | None = None) -> str:
        """A BSIM-CMG subcircuit: one series pair and two pull-ups per finger."""
        title = name or self.cell_name
        n_spec, p_spec = self.n_band.spec, self.p_band.spec
        lines = [
            (
                f"* ASAP7 NAND2: {self.fingers} finger(s) per input, "
                f"{n_spec.fins}-fin nFET stacks, {p_spec.fins}-fin pFETs"
            ),
            f".SUBCKT {title} {' '.join(NAND_PINS)}",
        ]
        for s in range(self.fingers):
            size_n = f"nfin={n_spec.fins} l={n_spec.gate_length}n nf=1 m=1"
            size_p = f"nfin={p_spec.fins} l={p_spec.gate_length}n nf=1 m=1"
            lines += [
                f"MNa{s} Y A n{s} VSS {n_spec.model} {size_n}",
                f"MNb{s} n{s} B VSS VSS {n_spec.model} {size_n}",
                f"MPa{s} Y A VDD VDD {p_spec.model} {size_p}",
                f"MPb{s} Y B VDD VDD {p_spec.model} {size_p}",
            ]
        lines += [f".ENDS {title}", ".END", ""]
        return "\n".join(lines)


# ── Layout ────────────────────────────────────────────────────────────────────
def _square(cell: Any, layer_name: str, x: float, y: float, size: float) -> None:
    half = size / 2
    box(cell, layer_name, x - half, y - half, x + half, y + half)


def build_nand(
    spec: NandSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Draw `spec` and return the gdspy Cell (nanometre coordinates)."""
    spec = spec or NandSpec()
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = (
        lib.new_cell(cell_name)
        if lib is not None
        else gdspy.Cell(cell_name, exclude_from_current=True)
    )
    half_gate = GATE_WIDTH // 2
    sx0, sx1 = spec.select_x
    n_band, p_band = spec.n_band, spec.p_band
    seam = spec.seam_y

    # Implant tiles the row band by band; the p band and its well coincide.
    for band in (n_band, p_band):
        box(cell, band.implant, sx0, band.y0, sx1, band.y1)
        if band.in_nwell:
            box(cell, "NWELL", sx0, band.y0, sx1, band.y1)
    if vt_layer := VT_LAYERS[spec.vt]:
        box(cell, vt_layer, sx0, 0, sx1, spec.height)

    # FIN and GATE are manufacturing grids.  Every stripe is one continuous
    # gate through both bands; the poly is cut on both rails.
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
    for y_rail, _ in spec.rails:
        box(cell, "GATE_CUT", sx0, y_rail - _HALF_CUT, sx1, y_rail + _HALF_CUT)

    # Diffusion: one fin-quantized ACTIVE per band, SDT + LISD on every
    # contacted column.  The n band's series nodes get neither.
    for band in (n_band, p_band):
        contacted = [
            x for x, r in zip(spec.sd_xs, spec.column_roles(band.flavor)) if r != "x"
        ]
        build_device_band(
            cell, band.spec, y0=band.y0, sd_xs=contacted, active_x=spec.active_x
        )

    # Rails, and the source ties.  A source is contacted on its band's via
    # row and reaches the rail with an M1 stub -- at a tile edge the stub
    # straddles the boundary and merges with the neighbour's, as the
    # inverter's does.  The exception is an interior pFET source: the output
    # flag runs along the p via row over it, so as in the released cell its
    # LISD carries on to the rail and the contact sits on the rail, over a
    # 16 nm LI rail that gives an 18 nm V0 something to land on (V0.AUX.1-2).
    for y_rail, _ in spec.rails:
        box(cell, "LIG", 0, y_rail - _HALF_RAIL, spec.width, y_rail + _HALF_RAIL)
        box(cell, "M1", 0, y_rail - _HALF_M1, spec.width, y_rail + _HALF_M1)
    for band in (n_band, p_band):
        act_lo, _ = band.active_span
        y_via = band.contact_y
        rail_far = band.rail_y + (-_HALF_M1 if band.rail_below else _HALF_M1)
        for x in spec.columns(band.flavor, "S"):
            at_edge = spec.abut and x in (spec.sd_xs[0], spec.sd_xs[-1])
            if band.flavor == "p" and not at_edge:
                box(cell, "LISD", x - _HALF_SD, act_lo, x + _HALF_SD, band.rail_y)
                _square(cell, "V0", x, band.rail_y, CONTACT_SIZE)
            else:
                _square(cell, "V0", x, y_via, CONTACT_SIZE)
                end = y_via + (_M1_PAD if band.rail_below else -_M1_PAD)
                box(cell, "M1", x - _HALF_M1, rail_far, x + _HALF_M1, end)
        for x in spec.columns(band.flavor, "Y"):
            _square(cell, "V0", x, y_via, CONTACT_SIZE)

    # Output.  One 18 nm flag along the p drain-via row spans every pFET
    # drain; each nFET drain gets a flag on its own via row and a vertical bar
    # up to the top flag, over the gate track beside it.
    y_n, y_p = n_band.contact_y, p_band.contact_y
    n_drains, p_drains = spec.columns("n", "Y"), spec.columns("p", "Y")
    bars = [spec.output_bar_x(x) for x in n_drains]
    box(
        cell,
        "M1",
        min(p_drains[0] - _M1_PAD, bars[0][0]),
        y_p - _HALF_M1,
        max(p_drains[-1] + _M1_PAD, bars[-1][1]),
        y_p + _HALF_M1,
    )
    for x_drain, (bx0, bx1) in zip(n_drains, bars):
        box(cell, "M1", x_drain - _M1_PAD, y_n - _HALF_M1, bx1, y_n + _HALF_M1)
        box(cell, "M1", bx0, y_n - _HALF_M1, bx1, y_p + _HALF_M1)

    # Inputs.  One LIG pad per run of same-input gates, contacted on the run's
    # first gate by a V0 and a vertical M1 bar; an input with several runs is
    # joined below the seam on M2, one V1 per bar.  The M2 bar stops flush
    # with its outer vias so abutting tiles keep 36 nm between their ties.
    bar_lo, bar_hi = spec.gate_bar_y
    for role, xs in spec.pads:
        box(
            cell,
            "LIG",
            xs[0] - half_gate - LIG_PAD_EXTENSION,
            seam - _HALF_LIG,
            xs[-1] + half_gate + LIG_PAD_EXTENSION,
            seam + _HALF_LIG,
        )
        _square(cell, "V0", xs[0], seam, CONTACT_SIZE)
        box(cell, "M1", xs[0] - _HALF_M1, bar_lo, xs[0] + _HALF_M1, bar_hi)
    for role, y_tie in spec.tie_ys.items():
        gate_xs = spec.pad_gate_xs(role)
        for x in gate_xs:
            _square(cell, "V1", x, y_tie, CONTACT_SIZE)
        box(
            cell,
            "M2",
            gate_xs[0] - _HALF_CT,
            y_tie - _HALF_M1,
            gate_xs[-1] + _HALF_CT,
            y_tie + _HALF_M1,
        )

    if draw_pin_labels:
        for pin, origin in spec.pin_positions.items():
            cell.add(
                gdspy.Label(pin, origin, layer=_M1_PIN_LAYER, texttype=_M1_PIN_TEXTTYPE)
            )
    box(cell, "BOUNDARY", 0, 0, spec.width, spec.height)
    return cell


def build_nand_row(
    spec: NandSpec,
    count: int,
    *,
    name: str | None = None,
    lib: Any = None,
    leaf_name: str | None = None,
) -> Any:
    """Butt `count` abutting NANDs into one continuous-diffusion row.

    The tile is symmetric (``B A A B``) with a source on each edge, so
    instances simply repeat every `width` with no mirroring -- the released
    decoder places its four post-decode NANDs the same way.  Labels are
    ``A<k>/B<k>/Y<k>`` plus one per rail.
    """
    if not spec.abut:
        raise ValueError("an isolated NAND is self-contained; a row needs abut=True")
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    gdspy = require_gdspy()
    row_name = name or f"{spec.cell_name}_x{count}"
    row = (
        lib.new_cell(row_name)
        if lib is not None
        else gdspy.Cell(row_name, exclude_from_current=True)
    )
    leaf = build_nand(
        spec, name=leaf_name or spec.cell_name, lib=lib, draw_pin_labels=False
    )

    def label(text: str, origin: tuple[float, float]) -> None:
        row.add(
            gdspy.Label(text, origin, layer=_M1_PIN_LAYER, texttype=_M1_PIN_TEXTTYPE)
        )

    pins = spec.pin_positions
    for index in range(count):
        origin_x = index * spec.width
        row.add(gdspy.CellReference(leaf, origin=(origin_x, 0)))
        for pin in ("A", "B", "Y"):
            x, y = pins[pin]
            label(f"{pin}{index}", (origin_x + x, y))
    width = count * spec.width
    for y_rail, net in spec.rails:
        label(net, (width / 2, y_rail))
    box(row, "BOUNDARY", 0, 0, width, spec.height)
    return row


def main(argv: list[str] | None = None) -> None:
    """Write a NAND2 GDS: ``asap7-nand --rows 14 7``."""
    args = parse_spec(
        NandSpec,
        argv,
        description="Generate an ASAP7 NAND2 in the released decoder's array style.",
    )
    spec = args.spec
    cell, out = write_gds(lambda lib: build_nand(spec, lib=lib), args.out)
    print(f"✓ {cell.name}: {spec.width} x {spec.height} nm -> {out}")
    print(
        f"  {spec.nfet_fins} nFET / {spec.pfet_fins} pFET fins, "
        f"{spec.fingers} finger(s) per input, {len(spec.tie_ys)} M2 input tie(s)"
    )
    print(f"  DRC-verifiable by the public runset: {spec.drc_verifiable}")


if __name__ == "__main__":  # pragma: no cover
    main()
