"""The column IO of one port for one data bit: mux group, sense amplifier, write driver, output latch.

`IoColumnSpec` places a mux group beside a logic column of three cells. The
default block has one group::

    ┌──────────────┬───────────────────┐  ← top = the mux group's height
    │              │ write driver  fill│  row pair 3
    │  bitline mux │ tap        fillers│  row pair 2
    │  group       │ output latch  fill│  row pair 1
    │  (selects    │ sense amp     fill│  row pair 0
    │   leaves)    │                   │
    └──────────────┴───────────────────┘
    ↑ bitlines enter here, from the array

With ``two_sided=True``, a mirrored group occupies the other side::

    ← left array bitlines                         right array bitlines →
    ┌──────────────┬───────────────────┬──────────────┐
    │              │ write driver  fill│              │  row pair 3
    │ left bitline │ tap        fillers│ right bitline│  row pair 2
    │ mux group    │ output latch  fill│ mux group    │  row pair 1
    │              │ sense amp     fill│ (mirrored)   │  row pair 0
    └──────────────┴───────────────────┴──────────────┘

The groups share the logic column and its ``SA``/``SAN`` lines. Each group has
its own bitlines, selects and ``PRECHN``; the right group's pins carry ``_R``
(``BL_R[i]``, ``YSEL_R[i]``, ``PRECHN_R``). This serves two facing arrays of one
port, only one of which a cycle accesses.

The groups' ``PRECHN`` and select tracks run their full height on M3;
``SA``/``SAN`` stop short at the outer ends so stacked blocks stay separate.
The logic cells' pins are M3 stubs and columns. What joins them is
drawn here on M4, the one layer none of the cells use: a horizontal wire
with a V3 on each M3 it meets, at a height where both exist.  ``SA``/``SAN``
reach the sense amplifier's stubs and the write driver's columns from the
group's tracks; ``QA``/``QAN`` climb from the amplifier's risers into the
latch's stubs directly above.  Nothing else is routed: every other pin is
the cell's own metal, labelled once here.

A tap in the spare row pair ties the block's wells and substrate, so the
block is DRC clean on its own (the latch-up rule reaches 30 um).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..layout.grid import GATE_PITCH
from ..layout.layers import box, require_gdspy
from ..layout.rules import M4_PITCH
from .bitline_mux import BitlineMuxSpec, build_bitline_mux_group
from .cli import parse_spec, write_gds
from .output_latch import OutputLatchSpec, build_output_latch
from .row_support import RowSupportSpec, build_row_support
from .rowcell import CAP, V3_M3_CAP, label, m3_column, m4_track
from .sense_amp_row import SenseAmpRowSpec, build_sense_amp_row
from .write_driver import WriteDriverSpec, build_write_driver

__all__ = [
    "IoColumnSpec",
    "block_netlist",
    "block_series_nodes",
    "build_io_column",
    "io_column_pins",
]

_MIN_SUPPORT = 2 * GATE_PITCH  # a filler or a tap is at least two pitches wide
_ROUTE_OFFSET = (
    2 * M4_PITCH
)  # first M4 line above a cell's bottom rail: 96 nm, clear of the entries
_SA_TIE_ABOVE = (
    17  # the amplifier's SAPRECHN tie, below its top row's seam (its _TIE_OFFSETS[1])
)
_MUX_LEAF_NETS = ("BL", "BLN", "YSEL", "YSELN")


def _default_mux() -> BitlineMuxSpec:
    return BitlineMuxSpec(rows=2, bitline_entry=(348.5, 245.5))


@dataclass(frozen=True)
class IoColumnSpec:
    """One port's column IO for one data bit.

    Args:
        mux: the bitline leaf; ``selects`` leaves are stacked into the group.
        sense_amp, write_driver, output_latch: the three logic cells.  All
            four have to be drawn on the same row (band heights and VT).
        tap: put a tap in the spare row pair.
        one_sense_phase: strap the amplifier's ``SAPRECHN`` to its ``SAE`` on
            M3, so that one phase precharges low and evaluates high, and the
            block has an ``SAE`` pin only.  That is how the released
            composite drives it.
        two_sided: a second group, mirrored, on the logic column's right,
            sharing the amplifier, driver and latch (pins suffixed ``_R``).
    """

    mux: BitlineMuxSpec = field(default_factory=_default_mux)
    sense_amp: SenseAmpRowSpec = field(default_factory=SenseAmpRowSpec)
    write_driver: WriteDriverSpec = field(default_factory=WriteDriverSpec)
    output_latch: OutputLatchSpec = field(default_factory=OutputLatchSpec)
    tap: bool = True
    one_sense_phase: bool = False
    two_sided: bool = False

    def __post_init__(self) -> None:
        rows = {
            (spec.band_height, spec.vt)
            for spec in (self.mux, self.sense_amp, self.write_driver, self.output_latch)
        }
        if len(rows) != 1:
            raise ValueError(f"the four cells are not on one row: {sorted(rows)}")
        if self.height % self.pair_height:
            raise ValueError(
                f"a group of {self.mux.selects} leaves of {self.mux.rows} row(s) is not whole row pairs"
            )
        if self.pairs < 4:
            raise ValueError(
                f"the logic needs four row pairs ({4 * self.pair_height} nm) and the group is "
                f"{self.height} nm; stack more leaves"
            )

    # ── Sizes ─────────────────────────────────────────────────────────────────
    @property
    def pair_height(self) -> int:
        """Two of the leaf's rows: the height of every logic cell."""
        return 2 * self.mux.row_height

    @property
    def height(self) -> int:
        return self.mux.selects * self.mux.height

    @property
    def pairs(self) -> int:
        return self.height // self.pair_height

    @property
    def logic_width(self) -> int:
        """Width of the logic column: the widest cell, or more so every spare is a whole support cell."""
        widths = [
            self.sense_amp.width,
            self.write_driver.width,
            self.output_latch.width,
        ]
        width = max(widths)
        while any(0 < width - w < _MIN_SUPPORT for w in widths):
            width += GATE_PITCH
        return width

    @property
    def width(self) -> int:
        return self.mux.width * (2 if self.two_sided else 1) + self.logic_width

    @property
    def right_group_x(self) -> int:
        """Left edge of the mirrored second group (its original right edge)."""
        return self.mux.width + self.logic_width

    def right_x(self, x: float) -> float:
        """A point of the group, in the block, as the mirrored second group puts it."""
        return self.right_group_x + (self.mux.width - x)

    @property
    def placements(self) -> dict[str, tuple[int, int]]:
        """Origin of each cell: the group at the left, the logic column beside it, a pair each."""
        x = self.mux.width
        pair = self.pair_height
        return {
            "mux": (0, 0),
            "sense_amp": (x, 0),
            "output_latch": (x, pair),
            "write_driver": (x, 3 * pair),
        }

    @property
    def support_pairs(self) -> list[int]:
        """Row pairs of the logic column with no cell: the tap's, and any above the write driver."""
        return [2, *range(4, self.pairs)]

    # ── Routes ────────────────────────────────────────────────────────────────
    @property
    def routes(self) -> dict[str, list[tuple[int, list[int]]]]:
        """``net -> [(y, [x, ...]), ...]``: the M4 lines, each over the M3 columns it joins.

        ``SA``/``SAN`` from the group's tracks to the amplifier's stubs (in
        the amplifier's bottom row) and to the driver's columns (in the
        driver's); ``QA``/``QAN`` from the amplifier's risers to the latch's
        stubs, just inside the latch.  Lines on one cell sit an M4 pitch
        apart.
        """
        place = self.placements
        x_logic = place["sense_amp"][0]
        sa_y, ol_y, wd_y = (
            place["sense_amp"][1],
            place["output_latch"][1],
            place["write_driver"][1],
        )
        mux_x, sa_x, wd_x, ol_x = (
            self.mux.track_x,
            self.sense_amp.track_x,
            self.write_driver.track_x,
            self.output_latch.track_x,
        )

        def group(net: str) -> list[float]:
            """The net's track in each group."""
            xs = [mux_x[net]]
            if self.two_sided:
                xs.append(self.right_x(mux_x[net]))
            return xs

        return {
            "SA": [
                (sa_y + _ROUTE_OFFSET, [*group("SA"), x_logic + sa_x["SA"]]),
                (wd_y + _ROUTE_OFFSET, [*group("SA"), x_logic + wd_x["SA"]]),
            ],
            "SAN": [
                (
                    sa_y + _ROUTE_OFFSET + M4_PITCH,
                    [*group("SAN"), x_logic + sa_x["SAN"]],
                ),
                (
                    wd_y + _ROUTE_OFFSET + M4_PITCH,
                    [*group("SAN"), x_logic + wd_x["SAN"]],
                ),
            ],
            "QA": [(ol_y + M4_PITCH, [x_logic + sa_x["QA"], x_logic + ol_x["QA"]])],
            "QAN": [
                (ol_y + 2 * M4_PITCH, [x_logic + sa_x["QAN"], x_logic + ol_x["QAN"]])
            ],
        }

    @property
    def cell_name(self) -> str:
        mux, sa, wd, ol = self.mux, self.sense_amp, self.write_driver, self.output_latch
        return (
            f"iocol_x{mux.selects}_r{mux.rows}_h{mux.band_height[0]}x{mux.band_height[1]}"
            f"{mux._variant_tag}_sa{sa.n_fingers}t{sa.tail_fingers}_wd{wd.keeper_fins}_ol{ol.fingers}"
            f"{'' if self.tap else '_notap'}{'_1ph' if self.one_sense_phase else ''}"
            f"{'_2s' if self.two_sided else ''}"
        )

    @property
    def sense_phase_strap(self) -> tuple[int, int, int] | None:
        """``(x, y0, y1)`` of the M3 joining ``SAE`` to ``SAPRECHN``, with `one_sense_phase`.

        The two are stubs on one column of the amplifier, from its bottom and
        top edges; the strap overlaps both by a via's cap.
        """
        if not self.one_sense_phase:
            return None
        sa = self.sense_amp
        x0, y0 = self.placements["sense_amp"]
        _, tie_mid, _ = sa.tie_levels
        seam1 = sa.stack.seam_ys[1]
        return (x0 + sa.track_x["SAE"], y0 + tie_mid, y0 + seam1 - _SA_TIE_ABOVE)

    @property
    def pin_positions(self) -> dict[str, tuple[str, tuple[float, float]]]:
        """Every pin, on the cell metal that is it."""
        place = self.placements
        pins: dict[str, tuple[str, tuple[float, float]]] = {}

        def take(spec, origin, names):
            ox, oy = origin
            for pin, (metal, (x, y)) in spec.items():
                if pin in names:
                    pins[pin] = (metal, (ox + x, oy + y))

        group = self.mux.group_pin_positions()
        group_pins = [p for p in group if p.startswith(("BL", "YSEL")) or p == "PRECHN"]
        take(group, place["mux"], group_pins)
        if self.two_sided:
            for pin in group_pins:
                metal, (x, y) = group[pin]
                pins[_right_pin(pin)] = (metal, (self.right_x(x), y))
        if self.one_sense_phase:
            # One net: pin it on the top stub, clear of the SA/SAN lines that
            # cross the amplifier's bottom row.
            metal, (x, y) = self.sense_amp.pin_positions["SAPRECHN"]
            ox, oy = place["sense_amp"]
            pins["SAE"] = (metal, (ox + x, oy + y))
        else:
            take(self.sense_amp.pin_positions, place["sense_amp"], ("SAE", "SAPRECHN"))
        take(
            self.write_driver.pin_positions,
            place["write_driver"],
            ("D", "WRENA", "WRENAN"),
        )
        take(self.output_latch.pin_positions, place["output_latch"], ("OE", "OEB", "Q"))
        # Every rail is its own conductor until a power grid joins them.
        for k in range(2 * self.pairs + 1):
            pins[f"{'VDD' if k % 2 else 'VSS'}{'' if k < 2 else f'.{k}'}"] = (
                "M1", (self.width / 2, k * self.mux.row_height),
            )  # fmt: skip
        return pins


def _mux_net(
    net: str, leaf_index: int | None = None, *, side: Literal["left", "right"]
) -> str:
    """Name a mux net in the block; sense and supply nets stay shared."""
    suffix = "_R" if side == "right" else ""
    if net in _MUX_LEAF_NETS:
        if leaf_index is None:
            raise ValueError(f"{net} needs a leaf index")
        return f"{net}{suffix}[{leaf_index}]"
    if net == "PRECHN":
        return f"PRECHN{suffix}"
    return net


def _right_pin(pin: str) -> str:
    """Map a group pin to its right-side name using the mux naming rule."""
    net, bracket, index = pin.partition("[")
    leaf_index = int(index.removesuffix("]")) if bracket else None
    return _mux_net(net, leaf_index, side="right")


def _block_devices(
    spec: IoColumnSpec,
) -> list[tuple[str, str, str, str, str, int, str]]:
    """Every transistor of the block on the block's nets: ``(name, d, g, s, flavor, fins, tag)``.

    Leaf ``i`` gets its own bitlines and selects; ``SA``/``SAN`` and
    ``QA``/``QAN`` are the nets the block routes; any other internal net is
    prefixed with its cell's tag (`block_series_nodes` names the ones that
    are uncontacted diffusion).
    """
    devices = []
    sides = ("left", "right") if spec.two_sided else ("left",)
    for i in range(spec.mux.selects):
        for name, d, g, s, flavor, fins in spec.mux.devices:
            for side in sides:
                suffix = "R" if side == "right" else ""
                nets = [_mux_net(net, i, side=side) for net in (d, g, s)]
                devices.append(
                    (f"M{i}{suffix}_{name[1:]}", *nets, flavor, fins, f"mux{suffix}{i}")
                )
    shared = {"SA", "SAN", "QA", "QAN", "VDD", "VSS"}
    pins = {"SAE", "SAPRECHN", "D", "WRENA", "WRENAN", "OE", "OEB", "Q"}
    for tag, cell in (
        ("sa", spec.sense_amp),
        ("wd", spec.write_driver),
        ("ol", spec.output_latch),
    ):
        for name, d, g, s, flavor, fins in cell.devices:
            nets = [n if n in shared or n in pins else f"{tag}_{n}" for n in (d, g, s)]
            if spec.one_sense_phase:
                nets = ["SAE" if n == "SAPRECHN" else n for n in nets]
            devices.append((f"M{tag}_{name[1:]}", *nets, flavor, fins, tag))
    return devices


def block_series_nodes(spec: IoColumnSpec) -> frozenset[str]:
    """The block's uncontacted series nodes, on the block's net names."""
    nodes = set()
    for tag, cell in (("sa", spec.sense_amp), ("ol", spec.output_latch)):
        nodes |= {f"{tag}_{n}" for n in cell.series_nodes}
    return frozenset(nodes)


def block_netlist(spec: IoColumnSpec, name: str | None = None) -> str:
    """A flat BSIM-CMG subcircuit of the whole block, sized by ``nfin``."""
    title = name or spec.cell_name
    n_band, p_band = spec.mux.bands
    lines = [
        "* ASAP7 column IO: bitline mux group, sense amplifier, write driver, output latch",
        f".SUBCKT {title} {' '.join(io_column_pins(spec))}",
    ]
    for device, d, g, s, flavor, fins, _ in _block_devices(spec):
        band = n_band if flavor == "n" else p_band
        bulk = "VSS" if flavor == "n" else "VDD"
        lines.append(
            f"{device} {d} {g} {s} {bulk} {band.spec.model} nfin={fins} l={band.spec.gate_length}n nf=1 m=1"
        )
    lines += [f".ENDS {title}", ".END", ""]
    return "\n".join(lines)


def io_column_pins(spec: IoColumnSpec) -> tuple[str, ...]:
    """The block's pins: the group's per-leaf ones, its precharge, the three cells' controls, supplies."""
    per_leaf = [
        _mux_net(pin, i, side="left")
        for i in range(spec.mux.selects)
        for pin in getattr(spec.mux, "leaf_nets", _MUX_LEAF_NETS)
    ]
    phases = ("SAE",) if spec.one_sense_phase else ("SAE", "SAPRECHN")
    right = []
    if spec.two_sided:
        right = [
            _mux_net(pin, i, side="right")
            for i in range(spec.mux.selects)
            for pin in _MUX_LEAF_NETS
        ] + [_mux_net("PRECHN", side="right")]
    return (
        *per_leaf,
        "PRECHN",
        *right,
        *phases,
        "D",
        "WRENA",
        "WRENAN",
        "OE",
        "OEB",
        "Q",
        "VDD",
        "VSS",
    )


def build_io_column(
    spec: IoColumnSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Place the four cells and the support, route on M4, label the pins; return the gdspy Cell."""
    spec = spec or IoColumnSpec()
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = (
        lib.new_cell(cell_name)
        if lib is not None
        else gdspy.Cell(cell_name, exclude_from_current=True)
    )
    place = spec.placements
    pair = spec.pair_height
    x_logic, width_logic = spec.mux.width, spec.logic_width

    # A library may already hold a cell of the same name from another block
    # (two ports share their logic cells and supports): reuse it.
    def reuse(cell_name: str, build):
        existing = lib.cells.get(cell_name) if lib is not None else None
        return existing if existing is not None else build()

    cells = {
        "mux": reuse(
            spec.mux.group_cell_name,
            lambda: build_bitline_mux_group(spec.mux, lib=lib, draw_pin_labels=False),
        ),
        "sense_amp": reuse(
            spec.sense_amp.cell_name,
            lambda: build_sense_amp_row(spec.sense_amp, lib=lib, draw_pin_labels=False),
        ),
        "output_latch": reuse(
            spec.output_latch.cell_name,
            lambda: build_output_latch(
                spec.output_latch, lib=lib, draw_pin_labels=False
            ),
        ),
        "write_driver": reuse(
            spec.write_driver.cell_name,
            lambda: build_write_driver(
                spec.write_driver, lib=lib, draw_pin_labels=False
            ),
        ),
    }
    for key, placed in cells.items():
        cell.add(gdspy.CellReference(placed, origin=place[key]))
    if spec.two_sided:
        # A reflection in y then a half turn is a mirror in x.
        cell.add(
            gdspy.CellReference(
                cells["mux"], origin=(spec.width, 0), rotation=180, x_reflection=True
            )
        )

    # Support: fillers beside the narrower cells, and in the spare pairs a tap
    # first, then fillers.  A support cell is two pitches or more, which is
    # what `logic_width` guarantees every spare is.
    stack = spec.write_driver.stack
    made: dict[tuple[str, int], Any] = {}

    def support(kind: Literal["filler", "tap"], pitches: int) -> Any:
        key = (kind, pitches)
        if key not in made:
            support_spec = RowSupportSpec(stack=stack, kind=kind, width_cpp=pitches)
            made[key] = reuse(
                support_spec.cell_name, lambda: build_row_support(support_spec, lib=lib)
            )
        return made[key]

    def fill(x0: int, x1: int, y: int, *, tap: bool = False) -> None:
        x = x0
        if tap and x1 - x >= _MIN_SUPPORT:
            cell.add(gdspy.CellReference(support("tap", 2), origin=(x, y)))
            x += _MIN_SUPPORT
        if x1 - x >= _MIN_SUPPORT:
            cell.add(
                gdspy.CellReference(
                    support("filler", (x1 - x) // GATE_PITCH), origin=(x, y)
                )
            )

    for key, logic in (
        ("sense_amp", spec.sense_amp),
        ("output_latch", spec.output_latch),
        ("write_driver", spec.write_driver),
    ):
        fill(x_logic + logic.width, x_logic + width_logic, place[key][1])
    for k, row_pair in enumerate(spec.support_pairs):
        fill(x_logic, x_logic + width_logic, row_pair * pair, tap=spec.tap and k == 0)

    # Routes.  Each M4 line needs M3 under both its vias: the group's tracks
    # and the driver's columns run the full cell height, the amplifier's
    # stubs and the latch's stubs the bottom of theirs, and the amplifier's
    # risers end on its top edge, so those are carried on to the line.
    sa_top = place["sense_amp"][1] + spec.sense_amp.height
    for net, lines in spec.routes.items():
        for y, xs in lines:
            if net in ("QA", "QAN"):
                riser_x, stub_x = xs
                if riser_x == stub_x:
                    # Riser and stub are one column: bridge the boundary on M3.
                    m3_column(cell, riser_x, sa_top - CAP, y + V3_M3_CAP)
                    continue
                m3_column(cell, riser_x, sa_top - CAP, y + V3_M3_CAP)
            m4_track(cell, y, xs)

    if strap := spec.sense_phase_strap:
        x, y0, y1 = strap
        m3_column(cell, x, y0, y1)

    if draw_pin_labels:
        for pin, (metal, origin) in spec.pin_positions.items():
            label(cell, pin.split(".")[0], metal, origin)
    box(cell, "BOUNDARY", 0, 0, spec.width, spec.height)
    return cell


def main(argv: list[str] | None = None) -> None:
    """Write a column-IO GDS: ``asap7-io-column --mux.selects 8``."""
    args = parse_spec(
        IoColumnSpec,
        argv,
        description="Generate one port's column IO for one data bit on the ASAP7 8T row.",
    )
    spec = args.spec
    cell, out = write_gds(lambda lib: build_io_column(spec, lib=lib), args.out)
    print(f"✓ {cell.name}: {spec.width} x {spec.height} nm -> {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
