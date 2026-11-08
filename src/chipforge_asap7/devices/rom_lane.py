"""The NOR ROM's sense lane: read-leaf rows between the half arrays, and their bitlines.

`chipforge_asap7.devices.rom_decoder` splits the array in two and leaves a
lane between the halves.  Above the array it holds the predecoders; beside
the array it is empty, and this module fills it with rows of read leaves
(`chipforge_asap7.devices.rom_sense`), one leaf per bit::

    left half ]== BLL<b> ─┐                               ┌─ BLR<b> ==[ right half
                     lift │ M4 ═════════ c0 [leaf b] c4 ═══ M4 │ lift
                          └ filler [PRECHN | leaves] filler tap filler ┘

The rows stand on the array's own fin grid, 351 nm each by default, and
alternate as standard-cell rows do: odd rows are the even rows mirrored about
their shared rail.  Each half's bitline ``BL<b>`` is horizontal M2 at the bit
row's contact height; in the lane it climbs at a *lift*, an M3 segment in a
strip beside the half, onto an M4 track, and the track crosses the rows to
the leaf's bitline column (c0 for ``BLL``, c4 for ``BLR``).  A bit's two
bitlines share one track: ``BLL`` stops at its leaf's c0 and ``BLR`` starts at
its c4.

`RomLaneSpec.plan` gives each bit a row, a slot and a track: the free track
nearest its bitline, in the row that holds it or up to two rows either side,
so the lift stays short.  A track carries one bit, and a row's ends leave
none, so about four tracks in five are usable: enough for the 81 nm bit
pitch of a 2-fin array and up, not for a 1-fin array's 54 nm.  The lifts of one side are
packed into as few M3 columns as their spans allow.  The row with the most
bits sets the leaves per row, and every row is padded to the same width.

With ``sample`` each leaf has a sample latch beside it, and the rows are
taller (`chipforge_asap7.devices.rom_sense`); ``SAMPLE`` is a second
full-height M3 column beside ``PRECHN``'s, and each bit's latch output is
``Q<b>``.  With ``register`` too, a register latch follows each sample latch
and ``Q<b>`` changes only as ``SAMPLE`` falls.

With ``replica`` the half's last bit is the replica bit, programmed in every
word, whose leaf output is the controller's ``READY``
(`chipforge_asap7.devices.rom_control`).  Its row keeps one more M4 track,
`RomLaneSpec.ready_track`, on which ``READY`` leaves the leaf for an M5
riser (`RomLaneSpec.ready_x`).

The supply grid is an M3 VSS and VDD strap on the outer filler of each row,
with a via stack on every rail of its net.  The VSS strap runs up to the M2
bar that `build_rom` lays along the array's top edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal

from ..layout.grid import FIN_WIDTH, GATE_PITCH
from ..layout.layers import PIN_LAYERS, box, require_gdspy
from ..layout.rules import M4_PITCH, M4_WIDTH, M4_X_GRID
from .rom_bitcell import RomArraySpec
from .rom_sense import RomSenseRowSpec, build_rom_sense_row
from .row_support import RowSupportSpec, build_row_support
from .rowcell import CAP, HALF, M4_HALF, PAD, V3_M3_CAP, V3_M4_CAP, square

__all__ = ["LanePlacement", "RomLaneSpec", "build_rom_lane"]

M4_TRACK_OFFSET = M4_WIDTH // 2  # M4.AUX.2: tracks at 12 + 48k
#: M5 runs vertically, 24 nm wide, on tracks at 12 + 48k (M5.AUX.1-2).
M5_WIDTH = 24
M5_PITCH = 2 * M5_WIDTH
M5_TRACK_OFFSET = M5_WIDTH // 2
#: M4 and M5 past a 24 nm V4, across the other metal (V4.M4.EN.1, V4.M5.EN.2).
V4_CAP = M5_WIDTH // 2 + 11
#: Two lift segments on one M3 column, end to end (two short edges).
LIFT_END_SPACE = 32
#: Lift column 0 stands where the half's bitline pin is, a gate pitch past it.
LIFT_INSET = GATE_PITCH
#: Lift columns a gate pitch apart: at 36 nm, two segments' ends that pass
#: each other by less than 20 nm meet corner to corner (M3.S.6).
LIFT_PITCH = GATE_PITCH
#: The supply straps, a track apart on the outer filler.
STRAP_PITCH = 36
#: Gate pitches of filler on each side of the leaves, and of the tap.
END_CPP = 2
#: The lane's top rail under the array's VSS bar: two half tracks and the
#: 18 nm between them, unless it is a VSS rail on the bar.
TOP_CLEAR = 36


@dataclass(frozen=True)
class LanePlacement:
    """Where bit `bit`'s leaf is: lane row, slot in the row, M4 track y and lift column."""

    bit: int
    row: int
    slot: int
    track: int
    lift: int


@dataclass(frozen=True)
class RomLaneSpec:
    """The sense lane beside `half`, one of the two half arrays.

    Args:
        half: one half array; the left one stands at x = 0, the right one at
            `right_x`.
        right_x: x of the right half.  ``None`` is the narrowest lane,
            `min_right_x`.
        n_fins, p_fins, band_height, vt, sample, register: the leaves'
            devices and rows, their sample latches and register latches
            (see `RomSenseRowSpec`).
        replica: the half's last bit is the replica bit; its leaf's ``D`` is
            ``READY``.
    """

    half: RomArraySpec
    right_x: int | None = None
    n_fins: int | None = None
    p_fins: int | None = None
    band_height: tuple[int, int] | None = None
    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"
    sample: bool = False
    register: bool = False
    replica: bool = False

    def __post_init__(self) -> None:
        rs = self.row_spec(1)
        object.__setattr__(self, "n_fins", rs.n_fins)
        object.__setattr__(self, "p_fins", rs.p_fins)
        object.__setattr__(self, "band_height", rs.band_height)
        if self.rows < 1:
            raise ValueError(
                f"a {self.half.height} nm array has no room for a {self.row_height} nm "
                "sense row"
            )
        if self.right_x is not None and self.right_x < self.min_right_x:
            raise ValueError(
                f"the sense lane needs the right half at x >= {self.min_right_x}, "
                f"got {self.right_x}"
            )

    # ── Rows ──────────────────────────────────────────────────────────────────
    def row_spec(self, leaves: int) -> RomSenseRowSpec:
        return RomSenseRowSpec(
            leaves=leaves,
            n_fins=self.n_fins,
            p_fins=self.p_fins,
            band_height=self.band_height,
            vt=self.vt,
            sample=self.sample,
            register=self.register,
        )

    @cached_property
    def row_height(self) -> int:
        return self.row_spec(1).height

    @cached_property
    def rows(self) -> int:
        """Rows that fit under the array's top edge, where the VSS bar runs.

        The top rail's strap pad is M2 too: a VSS rail may meet the bar, a
        VDD rail has to stand `TOP_CLEAR` under it.
        """
        top, h = self.half.height, self.row_height
        rows = top // h
        while rows and not (
            top - rows * h >= TOP_CLEAR or (top == rows * h and rows % 2 == 0)
        ):
            rows -= 1
        return rows

    @property
    def height(self) -> int:
        return self.rows * self.row_height

    def row_y(self, row: int) -> int:
        return row * self.row_height

    def bitline_y(self, bit: int) -> int:
        return self.half.row_y(bit) + self.half.bitcell.contact_y

    def tracks(self, row: int) -> list[int]:
        """M4 track centres a bitline column in `row` can take a V3 at."""
        lo, hi = self.row_spec(1).track_span
        y0 = self.row_y(row)
        first = y0 + lo + (M4_TRACK_OFFSET - y0 - lo) % M4_PITCH
        return list(range(first, y0 + hi + 1, M4_PITCH))

    # ── Plan ──────────────────────────────────────────────────────────────────
    @staticmethod
    def lift_span(y_bitline: int, y_track: int) -> tuple[int, int]:
        """M3 extent of a lift: a V2 on the bitline, a V3 on the track."""
        return (
            min(y_bitline - CAP, y_track - V3_M3_CAP),
            max(y_bitline + CAP, y_track + V3_M3_CAP),
        )

    @cached_property
    def plan(self) -> tuple[LanePlacement, ...]:
        """Row, slot, track and lift column of every bit, in bit order."""
        free = {r: self.tracks(r) for r in range(self.rows)}
        used: dict[int, int] = {r: 0 for r in range(self.rows)}
        placed = []
        for bit in self._plan_order:
            y = self.bitline_y(bit)
            home = min(max(y // self.row_height, 0), self.rows - 1)
            best = None
            for r in (home, home - 1, home + 1, home - 2, home + 2):
                if free.get(r):
                    t = min(free[r], key=lambda t: (abs(t - y), t))
                    key = (abs(t - y), used[r])
                    if best is None or key < best[0]:
                        best = (key, r, t)
            if best is None:
                raise ValueError(
                    f"bit {bit} at y = {y} finds no free M4 track in rows "
                    f"{home - 2}..{home + 2}: a {self.half.bitcell.height} nm bit "
                    f"pitch is denser than the lane's tracks ({M4_PITCH} nm, less "
                    "each row's ends)"
                )
            _, r, t = best
            free[r].remove(t)
            if self.replica and bit == self.replica_bit:
                # READY's own track: a free one that its D column reaches, not
                # beside the replica's own (their tips would meet, M4.S.3).
                lv = self.row_spec(1).levels
                lo, hi = lv["n"] + 3, lv["p"] - 3  # V3_M3_CAP inside the column
                y0 = self.row_y(r)
                ok = [
                    y
                    for y in free[r]
                    if abs(y - t) >= 2 * M4_PITCH
                    and lo <= (self.row_height - (y - y0) if r % 2 else y - y0) <= hi
                ]
                if not ok:
                    raise ValueError(f"row {r} has no M4 track left for READY")
                ready = min(ok, key=lambda y: (abs(y - t), y))
                free[r].remove(ready)
                object.__setattr__(self, "_ready_track", ready)
            placed.append((bit, r, used[r], t))
            used[r] += 1
        # Pack each lift onto the first column whose last segment it clears.
        ends: list[int] = []
        plan = []
        for bit, r, slot, t in sorted(placed, key=lambda p: self.bitline_y(p[0])):
            lo, hi = self.lift_span(self.bitline_y(bit), t)
            column = next(
                (c for c, end in enumerate(ends) if lo - end >= LIFT_END_SPACE),
                len(ends),
            )
            if column == len(ends):
                ends.append(hi)
            else:
                ends[column] = hi
            plan.append(LanePlacement(bit, r, slot, t, column))
        return tuple(sorted(plan, key=lambda p: p.bit))

    @property
    def replica_bit(self) -> int:
        """The replica's bit: the half's last."""
        if not self.replica:
            raise ValueError("this lane has no replica bit (replica=False)")
        return self.half.bits - 1

    @property
    def _plan_order(self) -> list[int]:
        """Bits in planning order: the replica first, so that its row keeps
        a track for READY."""
        bits = list(range(self.half.bits))
        if self.replica:
            bits.remove(self.replica_bit)
            bits.insert(0, self.replica_bit)
        return bits

    @property
    def ready_track(self) -> int:
        """The M4 track READY leaves the replica's leaf on."""
        self.plan  # noqa: B018
        return self._ready_track  # type: ignore[attr-defined]

    @property
    def ready_x(self) -> int:
        """X of READY's M5 riser: the M5 track nearest the replica's D column."""
        x = self.leaf_x(self.plan[self.replica_bit].slot, 2)
        return M5_TRACK_OFFSET + round((x - M5_TRACK_OFFSET) / M5_PITCH) * M5_PITCH

    @cached_property
    def leaves(self) -> int:
        """Leaves on the fullest row: every row is padded to it."""
        return max(p.slot for p in self.plan) + 1

    @cached_property
    def lifts(self) -> int:
        return max(p.lift for p in self.plan) + 1

    def row_leaves(self, row: int) -> int:
        return sum(1 for p in self.plan if p.row == row)

    # ── X ─────────────────────────────────────────────────────────────────────
    def lift_x(self, side: int, column: int) -> int:
        """X of lift column `column` on side 0 (left) or 1 (right)."""
        if side == 0:
            return self.half.width + LIFT_INSET + column * LIFT_PITCH
        return self.x_right - LIFT_INSET - column * LIFT_PITCH

    @property
    def x_right(self) -> int:
        return self.min_right_x if self.right_x is None else self.right_x

    @cached_property
    def x0(self) -> int:
        """X of the rows' left edge: a gate pitch past the last left lift, on the gate grid."""
        edge = self.lift_x(0, self.lifts - 1) + 2 * GATE_PITCH
        return -(-edge // GATE_PITCH) * GATE_PITCH

    @property
    def sense_x(self) -> int:
        """X of the leaf rows, past the left filler."""
        return self.x0 + END_CPP * GATE_PITCH

    @cached_property
    def row_width(self) -> int:
        return (
            self.row_spec(self.leaves).width
            + 3 * END_CPP * GATE_PITCH
            + END_CPP * GATE_PITCH
        )

    @property
    def x1(self) -> int:
        return self.x0 + self.row_width

    @cached_property
    def min_right_x(self) -> int:
        """The narrowest right half: its lifts a gate pitch clear of the rows, on a whole tile."""
        need = self.x1 + 2 * GATE_PITCH + (self.lifts - 1) * LIFT_PITCH + LIFT_INSET
        tile = self.half.bitcell.width
        return -(-need // tile) * tile

    @property
    def strap_xs(self) -> dict[str, int]:
        """The supply straps, a track apart on the outer filler."""
        f = self.x1 - END_CPP * GATE_PITCH
        return {"VSS": f + STRAP_PITCH, "VDD": f + 2 * STRAP_PITCH}

    @property
    def prechn_x(self) -> int:
        return self.sense_x + self.row_spec(1).prechn_x

    @property
    def sample_x(self) -> int:
        """X of ``SAMPLE``'s column, which runs the lane's height."""
        return self.sense_x + self.row_spec(1).sample_x

    def leaf_x(self, slot: int, column: int) -> int:
        return self.sense_x + self.row_spec(self.leaves).leaf_x(slot, column)

    def m4_ends(self, side: int) -> int:
        """X of every M4 wire's outer end on `side`, past lift column 0, on the M4 grid.

        Tips on adjacent tracks have to be aligned or 40 nm apart (M4.S.4),
        and lifts on different columns would leave them 54 nm apart, so every
        wire runs out to the same x.
        """
        if side == 0:
            x = self.lift_x(0, 0) - V3_M4_CAP
            return x - x % M4_X_GRID
        x = self.lift_x(1, 0) + V3_M4_CAP
        return x + -x % M4_X_GRID

    def to_absolute(self, row: int, y_local: float) -> float:
        """Y of a point at `y_local` in `row`'s unflipped cell."""
        y0 = self.row_y(row)
        return y0 + (self.row_height - y_local if row % 2 else y_local)

    def d_pin(self, bit: int) -> tuple[float, float]:
        p = self.plan[bit]
        rs = self.row_spec(self.leaves)
        x = self.leaf_x(p.slot, 2)
        return x, self.to_absolute(p.row, rs.levels["n"])

    def q_pin(self, bit: int) -> tuple[float, float]:
        """Where bit `bit`'s latch output ``Q<b>`` is labelled: its M3 at the Q track."""
        p = self.plan[bit]
        rs = self.row_spec(self.leaves)
        x = self.sense_x + rs.q_x(p.slot)
        return x, self.to_absolute(p.row, rs.levels["Q"])

    @property
    def cell_name(self) -> str:
        vt_tag = "" if self.vt == "rvt" else f"_{self.vt}"
        return (
            f"rom_lane{'_q' if self.sample else ''}{'r' if self.register else ''}"
            f"_{self.half.cell_name.removeprefix('rom_array_')}"
            f"_x{self.x_right}_{self.n_fins}n{self.p_fins}p"
            f"_h{self.band_height[0]}x{self.band_height[1]}{vt_tag}"
        )

    # ── Netlist ───────────────────────────────────────────────────────────────
    def devices(self) -> list[tuple[str, str, str, str, str, int]]:
        """Every leaf's transistors, named and netted by bit, then each row's own.

        A row's ``SAMPLEB`` inverter drives that row's latches only, so its
        output is ``SAMPLEB_R<row>``.
        """
        rs = self.row_spec(1)
        out = []
        for p in self.plan:
            b = p.bit
            for device, drain, gate, source, flavor, fins in rs.leaf_devices(0):
                rename = {
                    "BLL<0>": f"BLL<{b}>",
                    "BLR<0>": f"BLR<{b}>",
                    "D<0>": self.d_net(b),
                    "Q<0>": f"Q<{b}>",
                    "X0": f"X{b}",
                    "LX0": f"LX{b}",
                    "LY0": f"LY{b}",
                    "LQ0": f"LQ{b}",
                    "RX0": f"RX{b}",
                    "RY0": f"RY{b}",
                    "SAMPLEB": f"SAMPLEB_R{p.row}",
                }
                out.append(
                    (
                        f"{device[:-1]}_B{b}",
                        rename.get(drain, drain),
                        rename.get(gate, gate),
                        rename.get(source, source),
                        flavor,
                        fins,
                    )
                )
        for r in range(self.rows):
            if not self.row_leaves(r):
                continue
            for device, drain, gate, source, flavor, fins in rs.lead_devices:
                rename = {"SAMPLEB": f"SAMPLEB_R{r}"}
                out.append(
                    (
                        f"{device}_R{r}",
                        rename.get(drain, drain),
                        gate,
                        rename.get(source, source),
                        flavor,
                        fins,
                    )
                )
        return out

    def d_net(self, bit: int) -> str:
        """The net of `bit`'s leaf output: ``READY`` for the replica."""
        return "READY" if self.replica and bit == self.replica_bit else f"D<{bit}>"

    @property
    def data_bits(self) -> range:
        """The bits that are data, not the replica."""
        return range(self.half.bits - 1 if self.replica else self.half.bits)

    @property
    def pins(self) -> tuple[str, ...]:
        bits = range(self.half.bits)
        return (
            *(f"BLL<{b}>" for b in bits),
            *(f"BLR<{b}>" for b in bits),
            *(self.d_net(b) for b in bits),
            *((f"Q<{b}>" for b in self.data_bits) if self.sample else ()),
            "PRECHN",
            *(("SAMPLE",) if self.sample else ()),
            "VDD",
            "VSS",
        )

    def lvs_reference(self, name: str | None = None) -> str:
        """The lane alone as an LVS reference: one MOS per FIN x GATE channel."""
        title = name or self.cell_name
        rs = self.row_spec(1)
        lines = []
        for device, drain, gate, source, flavor, fins in self.devices():
            model = rs.bands[flavor].spec
            bulk = "VSS" if flavor == "n" else "VDD"
            for fin in range(fins):
                d, s = (
                    f"{net}_f{fin}" if net.startswith("X") else net
                    for net in (drain, source)
                )
                lines.append(
                    f"{device}_f{fin} {d} {gate} {s} {bulk} {model.model} "
                    f"L={model.gate_length}n W={FIN_WIDTH}n"
                )
        return "\n".join(
            [
                "* ASAP7 NOR ROM sense lane LVS reference: one MOS per FIN x GATE channel",
                f".SUBCKT {title} {' '.join(self.pins)}",
                *lines,
                f".ENDS {title}",
                ".END",
                "",
            ]
        )


def _label(cell: Any, text: str, metal: str, origin: tuple[float, float]) -> None:
    layer, texttype = PIN_LAYERS[metal]
    cell.add(require_gdspy().Label(text, origin, layer=layer, texttype=texttype))


def _shared(lib: Any, name: str, build: Any) -> Any:
    if lib is not None and name in lib.cells:
        return lib.cells[name]
    return build()


def build_rom_lane(
    spec: RomLaneSpec,
    *,
    top: int | None = None,
    name: str | None = None,
    lib: Any = None,
) -> Any:
    """Draw the lane in macro coordinates (left half at the origin).

    `top` is the y of the array's VSS bar; the VSS strap reaches it with a V2.
    """
    gdspy = require_gdspy()
    cell = (
        lib.new_cell(name or spec.cell_name)
        if lib is not None
        else gdspy.Cell(name or spec.cell_name, exclude_from_current=True)
    )
    h, k = spec.row_height, spec.leaves
    stack = spec.row_spec(1).stack
    rows = {}
    for leaves in sorted({spec.row_leaves(r) for r in range(spec.rows)} - {0}):
        rs = spec.row_spec(leaves)
        rows[leaves] = _shared(
            lib,
            rs.cell_name,
            lambda rs=rs: build_rom_sense_row(rs, lib=lib, draw_pin_labels=False),
        )

    def support(kind: str, cpp: int) -> Any:
        s = RowSupportSpec(stack=stack, kind=kind, width_cpp=cpp)  # type: ignore[arg-type]
        return _shared(lib, s.cell_name, lambda: build_row_support(s, lib=lib))

    full = spec.row_spec(k).width
    for r in range(spec.rows):
        x = spec.x0
        n = spec.row_leaves(r)
        if n:
            width = spec.row_spec(n).width
            pad = END_CPP + (full - width) // GATE_PITCH
            items = [
                (support("filler", END_CPP), END_CPP),
                (rows[n], width // GATE_PITCH),
            ]
        else:  # a row with no bit on it is all filler
            pad = 2 * END_CPP + full // GATE_PITCH
            items = []
        items += [
            (support("filler", pad), pad),
            (support("tap", END_CPP), END_CPP),
            (support("filler", END_CPP), END_CPP),
        ]
        flip = bool(r % 2)
        y = spec.row_y(r) + (h if flip else 0)
        for item, cpp in items:
            cell.add(gdspy.CellReference(item, (x, y), x_reflection=flip))
            x += cpp * GATE_PITCH
        assert x == spec.x1

    # Bitlines: M2 on to the lift, M3 up to the track, M4 to the leaf.
    for p in spec.plan:
        y_bl = spec.bitline_y(p.bit)
        lo, hi = spec.lift_span(y_bl, p.track)
        leaf = (spec.leaf_x(p.slot, 0), spec.leaf_x(p.slot, 4))
        for side in (0, 1):
            x = spec.lift_x(side, p.lift)
            edge = spec.half.width if side == 0 else spec.x_right
            box(
                cell,
                "M2",
                min(edge, x - PAD),
                y_bl - HALF,
                max(edge, x + PAD),
                y_bl + HALF,
            )
            square(cell, "V2", x, y_bl)
            _label(cell, f"{('BLL', 'BLR')[side]}<{p.bit}>", "M2", (x, y_bl))
            box(cell, "M3", x - HALF, lo, x + HALF, hi)
            inner = leaf[side] + (V3_M4_CAP if side == 0 else -V3_M4_CAP)
            inner += -inner % M4_X_GRID if side == 0 else -(inner % M4_X_GRID)
            outer = spec.m4_ends(side)
            box(cell, "M4", min(outer, inner), p.track - M4_HALF,
                max(outer, inner), p.track + M4_HALF)  # fmt: skip
            for xv in (x, leaf[side]):
                box(
                    cell,
                    "V3",
                    xv - HALF,
                    p.track - M4_HALF,
                    xv + HALF,
                    p.track + M4_HALF,
                )
        _label(cell, spec.d_net(p.bit), "M3", spec.d_pin(p.bit))
        if spec.sample and p.bit in spec.data_bits:
            _label(cell, f"Q<{p.bit}>", "M3", spec.q_pin(p.bit))

    if spec.replica:
        # READY: a V3 off the replica's D column onto its own track, and a V4
        # up to the M5 riser, which the parent carries on up.
        p = spec.plan[spec.replica_bit]
        x_d, y = spec.leaf_x(p.slot, 2), spec.ready_track
        x5 = spec.ready_x
        x0 = min(x_d - V3_M4_CAP, x5 - V4_CAP)
        x1 = max(x_d + V3_M4_CAP, x5 + V4_CAP)
        x0 -= x0 % M4_X_GRID
        x1 += -x1 % M4_X_GRID
        box(cell, "M4", x0, y - M4_HALF, x1, y + M4_HALF)
        box(cell, "V3", x_d - HALF, y - M4_HALF, x_d + HALF, y + M4_HALF)
        half5 = M5_WIDTH // 2
        box(cell, "V4", x5 - half5, y - half5, x5 + half5, y + half5)
        box(cell, "M5", x5 - half5, y - V4_CAP, x5 + half5, spec.height)

    # PRECHN, and SAMPLE beside it: the rows' first columns, through the rows
    # without leaves too.
    columns = {"PRECHN": spec.prechn_x}
    if spec.sample:
        columns["SAMPLE"] = spec.sample_x
    for net, x in columns.items():
        box(cell, "M3", x - HALF, 0, x + HALF, spec.height)
        _label(cell, net, "M3", (x, spec.height - h / 2))

    # Supplies: a via stack on every rail of the strap's net.
    for net, x in spec.strap_xs.items():
        y1 = spec.height if net == "VDD" or top is None else top
        box(cell, "M3", x - HALF, -CAP, x + HALF, y1 + CAP)
        for r in range(spec.rows + 1):
            rail_net = "VSS" if r % 2 == 0 else "VDD"
            if rail_net != net:
                continue
            y = spec.row_y(r)
            square(cell, "V1", x, y)
            box(cell, "M2", x - PAD, y - HALF, x + PAD, y + HALF)
            square(cell, "V2", x, y)
        if net == "VSS" and top is not None:
            square(cell, "V2", x, top)
        _label(cell, net, "M3", (x, spec.height - h / 2))
    return cell
