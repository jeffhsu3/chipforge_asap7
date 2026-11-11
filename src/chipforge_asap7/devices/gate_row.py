"""A dense row of static CMOS gates, with every net on an M2 track.

The NOR ROM's controller row (`chipforge_asap7.devices.rom_control`) made
general: any gates, any input pins and any outputs.  As there, each gate is
one diffusion island on one dense row (`chipforge_asap7.devices.rowcell`),
with a gap column after it:

* a gate's inputs are its poly stripes, each run of one net with a gate
  contact on the seam and an M1 bar to that net's M2 track;
* its output is the drains joined along the via rows, and one M3 column with
  a V2 on the output's track;
* `assign_levels` gives every net a track by interval colouring of the spans
  its terminals cover, `LEVELS` of them between the via rows.

An input pin is an M3 column from its track to the row's top, in a gap
column of its own before the first gate it drives.  An output pin's M3 column
runs on to the row's top.  A net in `left` keeps a track from the row's left
edge, to meet a row abutted there (the address latches' ``PRECHN``), and one in
`right` to the right edge.  An input on an edge has no M3 pin: its track's end
is the pin.

By default the gates follow one another, each input's pin column before the
first gate it drives.  `at` places each gate's first column instead, to line
gates up with cells beside the row (a wordline decoder's slices); each input
pin then takes the nearest free column left of the first gate it drives.

With `tap` the row ends in a 2 CPP well and substrate tap
(`chipforge_asap7.devices.row_support`), for a row whose wells meet no
tapped row's.  The row's own implant runs on past its last gate, the filler
a tap needs between it and logic (ACTIVE.LUP.1, and the implants' 46 nm
enclosure of ACTIVE).

Gate kinds: ``inv``, ``nand2``, ``nor2``, ``nand3`` and ``nor3``.  An
inverter's fingers alternate source and output columns; a two-input gate's
fingers mirror, so ``nand2`` with two fingers is the p band ``Y S Y S Y`` and
the n band ``S . Y . S`` on stripes ``A B B A``.

Units are nanometres.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal

from ..layout.grid import FIN_WIDTH, GATE_PITCH
from ..layout.layers import box, require_gdspy
from .finfet import SELECT_X_ENC
from .rom_control import LEVELS, assign_levels
from .rom_latch import LANDING_CLEAR, M3_EDGE
from .rom_sense import _reach
from .row import RowBand, RowStack
from .row_support import RowSupportSpec, build_row_support
from .rowcell import (
    ISLAND_OVERHANG,
    PAD,
    draw_frame,
    draw_rails,
    gate_contact,
    island,
    label,
    landing,
    m2_track,
    m3_column,
    sd_contact,
    stack_to_m3,
    supply_contact,
    via_y,
)

__all__ = ["Gate", "GateRowSpec", "build_gate_row"]

#: The shortest M2 track: past M2.S.2's 36.25 nm, so its sides are long edges.
MIN_TRACK = 40
#: A tap's width, in columns.
TAP_CPP = 2

Kind = Literal["inv", "nand2", "nor2", "nand3", "nor3"]

# One finger of each three-input kind, p band and n band: a supply ("S"), the
# output ("Y") or an uncontacted series node (None).  Stripe i joins columns
# i and i + 1.
_THREE = {
    "nand3": (("Y", "S", "Y", "S"), ("S", None, None, "Y")),
    "nor3": (("S", None, None, "Y"), ("Y", "S", "Y", "S")),
}


def _parallel(fingers: int) -> tuple[str, ...]:
    return tuple("Y" if i % 2 == 0 else "S" for i in range(2 * fingers + 1))


def _series(fingers: int) -> tuple[str | None, ...]:
    return tuple(
        None if i % 2 else ("S" if i % 4 == 0 else "Y") for i in range(2 * fingers + 1)
    )


@dataclass(frozen=True)
class Gate:
    """``out = kind(*ins)``, with `fingers` copies side by side; `mirrored`
    draws it right to left."""

    out: str
    kind: Kind
    ins: tuple[str, ...]
    fingers: int = 1
    mirrored: bool = False

    def __post_init__(self) -> None:
        arity = {"inv": 1, "nand2": 2, "nor2": 2, "nand3": 3, "nor3": 3}[self.kind]
        if len(self.ins) != arity:
            raise ValueError(f"{self.out}: a {self.kind} takes {arity} inputs")
        if self.fingers < 1 or (arity == 3 and self.fingers != 1):
            raise ValueError(f"{self.out}: {self.fingers} fingers of a {self.kind}")

    @property
    def stripes(self) -> tuple[str, ...]:
        """The net on each poly stripe, left to right."""
        if self.kind == "inv":
            stripes = self.ins * self.fingers
        elif self.kind in _THREE:
            stripes = self.ins
        else:
            a, b = self.ins
            stripes = tuple(
                n
                for k in range(self.fingers)
                for n in ((a, b) if k % 2 == 0 else (b, a))
            )
        return stripes[::-1] if self.mirrored else stripes

    @property
    def bands(self) -> dict[str, tuple[str | None, ...]]:
        """Column nets of the p and n bands, left to right."""
        if self.kind == "inv":
            cols = tuple("S" if i % 2 == 0 else "Y" for i in range(self.fingers + 1))
            bands = {"p": cols, "n": cols}
        elif self.kind in _THREE:
            p, n = _THREE[self.kind]
            bands = {"p": p, "n": n}
        else:
            par, ser = _parallel(self.fingers), _series(self.fingers)
            bands = (
                {"p": par, "n": ser} if self.kind == "nand2" else {"p": ser, "n": par}
            )
        return {b: c[::-1] for b, c in bands.items()} if self.mirrored else bands

    @property
    def columns(self) -> int:
        return len(self.bands["p"])

    @property
    def out_column(self) -> int:
        """The column of the output's M3: an output column of both bands if
        there is one, else the rightmost output column."""
        ys = {
            b: {i for i, n in enumerate(c) if n == "Y"} for b, c in self.bands.items()
        }
        common = ys["p"] & ys["n"]
        return min(common) if common else max(ys["p"] | ys["n"])


@dataclass(frozen=True)
class GateRowSpec:
    """One row of `gates`, in order.

    Args:
        gates: the gates, left to right; each output is a distinct net.
        inputs: nets driven from outside, each an M3 pin to the row's top.
        outputs: gate outputs that are pins, their M3 run to the row's top.
        left: nets that keep a track from the row's left edge, top level down.
        right: nets that keep a track to the row's right edge, on the levels
            under `left`'s.
        at: each gate's first column; empty to place them one after another.
        span: the fewest columns the row has.
        tap: end the row in a well and substrate tap.
        name: the cell's name.
        n_fins, p_fins: fins of every nFET and pFET.
        band_height: ``(n_band, p_band)`` in nm, the address-latch row's.
        vt: threshold flavor of every device.
    """

    gates: tuple[Gate, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...] = ()
    left: tuple[str, ...] = ()
    right: tuple[str, ...] = ()
    at: tuple[int, ...] = ()
    span: int = 0
    tap: bool = False
    name: str = "gate_row"
    n_fins: int = 4
    p_fins: int = 4
    band_height: tuple[int, int] = (243, 243)
    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"

    def __post_init__(self) -> None:
        object.__setattr__(self, "gates", tuple(self.gates))
        object.__setattr__(self, "band_height", tuple(self.band_height))
        object.__setattr__(self, "at", tuple(self.at))
        outs = [g.out for g in self.gates]
        if len(set(outs)) != len(outs):
            raise ValueError("two gates drive one net")
        if set(self.inputs) & set(outs):
            raise ValueError(
                f"inputs driven inside the row: {set(self.inputs) & set(outs)}"
            )
        used = {n for g in self.gates for n in g.ins}
        if missing := used - set(outs) - set(self.inputs) - {"VDD", "VSS"}:
            raise ValueError(f"undriven nets: {sorted(missing)}")
        if extra := set(self.outputs) - set(outs):
            raise ValueError(f"outputs no gate drives: {sorted(extra)}")
        if both := set(self.left) & set(self.right):
            raise ValueError(f"nets on both edges: {sorted(both)}")
        if self.at:
            if len(self.at) != len(self.gates):
                raise ValueError("`at` needs one column per gate")
            end = 1  # column 0 is the gap at the row's start
            for gate, column in zip(self.gates, self.at, strict=True):
                if column < end:
                    raise ValueError(f"{gate.out} at column {column} overlaps")
                end = column + gate.columns + 1
        lv = self.levels
        if lv[LEVELS - 1] > lv["p"] - LANDING_CLEAR:
            raise ValueError(
                f"bands of {self.band_height} nm leave no room for {LEVELS} M2 levels"
            )

    # ── Rows ──────────────────────────────────────────────────────────────────
    @property
    def stack(self) -> RowStack:
        return RowStack(
            rows=((self.n_fins, self.p_fins),), vt=self.vt, band_height=self.band_height
        )

    @property
    def bands(self) -> dict[str, RowBand]:
        n, p = self.stack.bands()
        return {"n": n, "p": p}

    @property
    def height(self) -> int:
        return self.stack.height

    @property
    def rails(self) -> tuple[tuple[int, str], ...]:
        return self.stack.rails

    @property
    def levels(self) -> dict[Any, int]:
        """Y of the via rows (``"n"``, ``"p"``) and of the M2 levels 0 up."""
        y_n, y_p = via_y(self.bands["n"]), via_y(self.bands["p"])
        first = y_n + LANDING_CLEAR
        return {"n": y_n, "p": y_p, **{i: first + 36 * i for i in range(LEVELS)}}

    # ── Columns ───────────────────────────────────────────────────────────────
    @property
    def edge_nets(self) -> frozenset[str]:
        return frozenset(self.left) | frozenset(self.right)

    @cached_property
    def placement(self) -> tuple[dict[str, int], dict[str, int]]:
        """``(first column of each gate, pin column of each input)``: a gap
        column follows every gate; an input takes the gap before the first
        gate it drives, or the row's end if none does.  An edge input has no
        pin column."""
        if self.at:
            return self._placed()
        first: dict[str, int] = {}
        pins: dict[str, int] = {}
        column = 0
        for gate in self.gates:
            for net in gate.ins:
                if net in self.inputs and net not in pins and net not in self.edge_nets:
                    pins[net] = column
                    column += 1
            column = max(column, 1)  # column 0 is the gap at the row's start
            first[gate.out] = column
            column += gate.columns + 1
        for net in self.inputs:
            if net not in pins and net not in self.edge_nets:
                pins[net] = column
                column += 1
        return first, pins

    def _placed(self) -> tuple[dict[str, int], dict[str, int]]:
        """`placement` with the gates at `at`: each pin in the nearest column
        left of its first gate that is neither a gate's nor its gap."""
        first = {g.out: c for g, c in zip(self.gates, self.at, strict=True)}
        taken = {
            c
            for g in self.gates
            for c in range(first[g.out], first[g.out] + g.columns + 1)
        }
        uses: dict[str, int] = {}
        for gate in self.gates:
            for net in gate.ins:
                uses.setdefault(net, first[gate.out])
        pins: dict[str, int] = {}
        for net in self.inputs:
            if net in self.edge_nets:
                continue
            column = uses.get(net, self.span) - 1
            while column >= 0 and column in taken:
                column -= 1
            if column < 0:
                raise ValueError(f"no free column left of {net}'s first gate")
            pins[net] = column
            taken.add(column)
        return first, pins

    @property
    def columns(self) -> int:
        first, pins = self.placement
        ends = [first[g.out] + g.columns for g in self.gates]
        # A tap: past the last gate's gap column, one column of filler.
        tap = max(ends) + 1 + TAP_CPP if self.tap else 0
        return max(*ends, max(pins.values(), default=0) + 1, self.span, tap)

    @property
    def width(self) -> int:
        return (self.columns + 1) * GATE_PITCH

    @staticmethod
    def column_x(column: int) -> int:
        return (column + 1) * GATE_PITCH

    def gate_column_x(self, gate: Gate, column: int) -> int:
        return self.column_x(self.placement[0][gate.out] + column)

    def stripe_x(self, gate: Gate, stripe: int) -> int:
        return self.gate_column_x(gate, stripe) + GATE_PITCH // 2

    def out_x(self, gate: Gate) -> int:
        return self.gate_column_x(gate, gate.out_column)

    def pin_x(self, pin: str) -> int:
        return self.column_x(self.placement[1][pin])

    def contacts(self, gate: Gate) -> list[tuple[str, list[int]]]:
        """``(net, stripe xs)`` of each run of same-net stripes."""
        runs: list[tuple[str, list[int]]] = []
        for i, net in enumerate(gate.stripes):
            x = self.stripe_x(gate, i)
            if runs and runs[-1][0] == net:
                runs[-1][1].append(x)
            else:
                runs.append((net, [x]))
        return runs

    # ── Nets ──────────────────────────────────────────────────────────────────
    @cached_property
    def terminals(self) -> dict[str, dict[str, list[int]]]:
        """Each net's track terminals: ``"v1"`` gate xs and ``"v2"`` column xs.
        A supply input is tied at its gate contact and has no track."""
        nets: dict[str, dict[str, list[int]]] = {}
        for gate in self.gates:
            nets.setdefault(gate.out, {"v1": [], "v2": []})["v2"].append(
                self.out_x(gate)
            )
            for net, run in self.contacts(gate):
                nets.setdefault(net, {"v1": [], "v2": []})["v1"].append(run[0])
        for pin in self.inputs:
            terminal = nets.setdefault(pin, {"v1": [], "v2": []})
            if pin not in self.edge_nets:
                terminal["v2"].append(self.pin_x(pin))
        for supply in ("VDD", "VSS"):
            if supply in nets:
                raise ValueError(f"tie {supply} inputs outside the row, not on a gate")
        return nets

    @cached_property
    def spans(self) -> dict[str, tuple[int, int]]:
        """Each net's M2 track extent, its terminals' outermost plus a pad, at
        least `MIN_TRACK` long.  A net with one terminal, an output's own
        column, has no track: a stub that short has only short edges
        (M2.S.2's, up to 36.25 nm), which need 25 nm to the next level's."""
        spans = {}
        for net, t in self.terminals.items():
            xs = t["v1"] + t["v2"]
            if len(xs) == 1 and net not in self.edge_nets:
                continue
            lo, hi = min(xs) - PAD, max(xs) + PAD
            spans[net] = (lo, max(hi, lo + MIN_TRACK))
        for net in self.left:
            spans[net] = (0, spans[net][1])
        for net in self.right:
            spans[net] = (spans[net][0], self.width)
        return spans

    @cached_property
    def net_levels(self) -> dict[str, int]:
        """The M2 level of every net; `left` nets take the top levels and
        `right` nets the next.  Raises if they do not fit, which only the
        layout needs."""
        edge = (*self.left, *self.right)
        fixed = {net: LEVELS - 1 - i for i, net in enumerate(edge)}
        return assign_levels(self.spans, LEVELS, fixed=fixed)

    @property
    def pin_positions(self) -> dict[str, tuple[str, tuple[float, float]]]:
        top = self.height - M3_EDGE - 9
        lv = self.levels
        pins: dict[str, tuple[str, tuple[float, float]]] = {
            pin: ("M3", (self.pin_x(pin), top))
            for pin in self.inputs
            if pin not in self.edge_nets
        }
        for net in self.left:
            pins.setdefault(net, ("M2", (PAD, lv[self.net_levels[net]])))
        for net in self.right:
            pins.setdefault(net, ("M2", (self.width - PAD, lv[self.net_levels[net]])))
        for gate in self.gates:
            if gate.out in self.outputs:
                pins[gate.out] = ("M3", (self.out_x(gate), top))
        pins["VDD"] = ("M1", (self.width / 2, self.rails[1][0]))
        pins["VSS"] = ("M1", (self.width / 2, self.rails[0][0]))
        return pins

    @property
    def pins(self) -> tuple[str, ...]:
        return tuple(self.pin_positions)

    # ── Netlist ───────────────────────────────────────────────────────────────
    @property
    def devices(self) -> tuple[tuple[str, str, str, str, str, int], ...]:
        """``(name, drain, gate, source, flavor, fins)``: stripe i of a band
        joins its columns i and i + 1; series nodes are ``<out>_<band><i>``."""
        out = []
        for gate in self.gates:
            for band, cols in gate.bands.items():
                fins = self.p_fins if band == "p" else self.n_fins
                supply = "VDD" if band == "p" else "VSS"

                def net(i: int, band: str = band, cols: Any = cols) -> str:
                    c = cols[i]
                    if c == "S":
                        return supply
                    if c == "Y":
                        return gate.out
                    return f"{gate.out}_{band}{i}"

                for i, g in enumerate(gate.stripes):
                    out.append(
                        (
                            f"M{band.upper()}_{gate.out}_{i}",
                            net(i),
                            g,
                            net(i + 1),
                            band,
                            fins,
                        )
                    )
        return tuple(out)

    @property
    def series_nodes(self) -> frozenset[str]:
        return frozenset(
            f"{gate.out}_{band}{i}"
            for gate in self.gates
            for band, cols in gate.bands.items()
            for i, c in enumerate(cols)
            if c is None
        )

    @property
    def local_nets(self) -> frozenset[str]:
        """Gate outputs that are not pins: the row's own nets."""
        return frozenset(g.out for g in self.gates) - set(self.outputs) - self.edge_nets

    def lvs_reference(self, name: str | None = None) -> str:
        """The LVS reference: one MOS per FIN x GATE channel."""
        lines = []
        for device, drain, gate, source, flavor, fins in self.devices:
            model = self.bands[flavor].spec
            bulk = "VSS" if flavor == "n" else "VDD"
            for fin in range(fins):
                d, s = (
                    f"{n}_f{fin}" if n in self.series_nodes else n
                    for n in (drain, source)
                )
                lines.append(
                    f"{device}_f{fin} {d} {gate} {s} {bulk} {model.model} "
                    f"L={model.gate_length}n W={FIN_WIDTH}n"
                )
        title = name or self.name
        return "\n".join(
            ["* ASAP7 gate row LVS reference: one MOS per FIN x GATE channel",
             f".SUBCKT {title} {' '.join(self.pins)}", *lines, f".ENDS {title}", ".END", ""]
        )  # fmt: skip


# ── Layout ────────────────────────────────────────────────────────────────────
def _draw_gate(cell: Any, spec: GateRowSpec, gate: Gate) -> None:
    """`gate`'s islands, its output's via-row joins and M3, and its gate contacts."""
    bands = spec.bands
    lv = spec.levels
    seam = spec.stack.seam_y(0)
    xs = [spec.gate_column_x(gate, i) for i in range(gate.columns)]
    x_out = spec.out_x(gate)
    for name, cols in gate.bands.items():
        band = bands[name]
        island(
            cell,
            band,
            xs,
            contacted=[x for x, c in zip(xs, cols, strict=True) if c is not None],
        )
        ys = []
        for x, c in zip(xs, cols, strict=True):
            if c == "S":
                supply_contact(cell, band, x)
            elif c == "Y":
                sd_contact(cell, band, x)
                ys.append(x)
        cover = sorted({*ys, x_out})
        y = lv[name]
        if cover == [x_out]:
            if band.rail_below:
                landing(cell, band, x_out)
            else:
                stack_to_m3(cell, x_out, y)
        else:
            m2_track(cell, y, cover[0] - PAD, cover[-1] + PAD, vias=ys)
            box(cell, "V2", x_out - 9, y - 9, x_out + 9, y + 9)
    tracked = gate.out in spec.net_levels
    top = spec.height - M3_EDGE if gate.out in spec.outputs else lv["p"] + PAD
    m3_column(
        cell,
        x_out,
        lv["n"] - PAD,
        top,
        vias=[lv[spec.net_levels[gate.out]]] if tracked else [],
    )
    for net, run in spec.contacts(gate):
        gate_contact(cell, seam, run, run[0], _reach(lv[spec.net_levels[net]], seam))


def build_gate_row(spec: GateRowSpec, *, name: str | None = None, lib: Any = None,
                   draw_pin_labels: bool = True) -> Any:  # fmt: skip
    """Draw `spec` and return the gdspy Cell (nanometre coordinates)."""
    gdspy = require_gdspy()
    cell_name = name or spec.name
    cell = (
        lib.new_cell(cell_name)
        if lib is not None
        else gdspy.Cell(cell_name, exclude_from_current=True)
    )
    lv = spec.levels
    logic = spec.width - (TAP_CPP * GATE_PITCH if spec.tap else 0)
    draw_frame(cell, spec.stack, logic, spec.vt)
    draw_rails(cell, spec.stack, logic)
    if spec.tap:
        tap = build_row_support(
            RowSupportSpec(stack=spec.stack, kind="tap", width_cpp=TAP_CPP),
            draw_pin_labels=False,
        )
        for (layer, datatype), shapes in tap.get_polygons(by_spec=True).items():
            for points in shapes:
                cell.add(
                    gdspy.Polygon(points + (logic, 0), layer=layer, datatype=datatype)
                )
    assert spec.column_x(1) - ISLAND_OVERHANG >= SELECT_X_ENC
    for gate in spec.gates:
        _draw_gate(cell, spec, gate)
    for net, (lo, hi) in spec.spans.items():
        m2_track(cell, lv[spec.net_levels[net]], lo, hi, vias=spec.terminals[net]["v1"])
    for pin in spec.inputs:
        if pin in spec.edge_nets:
            continue
        y = lv[spec.net_levels[pin]]
        m3_column(cell, spec.pin_x(pin), y - PAD, spec.height - M3_EDGE, vias=[y])
    if draw_pin_labels:
        for pin, (metal, origin) in spec.pin_positions.items():
            label(cell, pin, metal, origin)
    box(cell, "BOUNDARY", 0, 0, spec.width, spec.height)
    return cell
