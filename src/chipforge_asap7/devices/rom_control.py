"""The NOR ROM's self-timed read controller: CLK and RESETN to PRECHN, EVAL and SAMPLE.

CLK high asks for a precharge and CLK low for a read, as in the B-ROM
(`chipforge_asap7.brom.controller`)::

    REQ    = RESETN . !CLK
    PRECHN = REQ + EVAL               precharge off while a read is pending
    EVAL   = (REQ + SAMPLE) . PRECHN  the wordline gate, once precharge is off
    SAMPLE = REQ . EVAL . READY       the sample latches, once the replica reads

``READY`` is the replica bit's read leaf: high once the replica bitline,
programmed in every word, has discharged.  A read opens the sample latches
only after the slowest bitline could have.  When CLK rises the phases fall
in order -- SAMPLE, then EVAL (held by SAMPLE), then PRECHN (held by EVAL)
-- so the latches close before the wordline falls and the bitlines
precharge only after it has.  RESETN low is CLK high: it drains a read the
same way and does not touch the latched data.  Each gate watches the
drivers' outputs, not their inputs.

In static CMOS, eleven gates::

    RN  = !RESETN               REQ = NOR(CLK, RN)
    EN  = NOR(REQ, EVAL)        PRECHN = !EN        PN = !PRECHN
    ERN = NOR(REQ, SAMPLE)      EV0 = NOR(ERN, PN)  EVAL = !!EV0
    SN  = NAND(REQ, EVAL, READY)                    SAMPLE = !SN

`RomControlSpec` draws them on one dense row (`chipforge_asap7.devices.rowcell`),
each gate one diffusion island and a gap.  A gate's inputs are its poly
stripes, each with a gate contact on the seam and an M1 bar to its net's M2
track.  Its output is the drains joined along the via rows and one M3
column.  `assign_levels` gives every net a track by interval colouring of the
spans its terminals cover; ``PRECHN`` keeps the top track from the row's
left edge, where it meets the address latches' ``PRECHN`` track
(`chipforge_asap7.devices.rom_latch`, which this row matches).  ``EVAL`` and
``SAMPLE`` climb to the row's top on their output columns; ``CLK``,
``RESETN`` and ``READY`` are M3 pin columns in the gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal

from ..layout.grid import FIN_WIDTH, GATE_PITCH
from ..layout.layers import box, require_gdspy
from .cli import parse_spec, write_gds
from .finfet import SELECT_X_ENC
from .row import RowBand, RowStack
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
from .rom_latch import LANDING_CLEAR, M3_EDGE
from .rom_sense import _reach

__all__ = [
    "CONTROL_INPUTS",
    "CONTROL_OUTPUTS",
    "GateSpec",
    "RomControlSpec",
    "assign_levels",
    "build_rom_control_row",
]

CONTROL_INPUTS = ("CLK", "RESETN", "READY")
CONTROL_OUTPUTS = ("PRECHN", "EVAL", "SAMPLE")
#: M2 levels between the via rows, as on the address-latch row.
LEVELS = 7
#: Two tracks on one level, end to end.
TRACK_END_SPACE = 36

# Each kind's columns, p band and n band: a supply ("S"), the output ("Y") or
# an uncontacted series node (None).  Stripe i joins columns i and i + 1.
_KINDS: dict[str, tuple[tuple[str | None, ...], tuple[str | None, ...]]] = {
    "nor2": (("S", None, "Y"), ("Y", "S", "Y")),
    "nand3": (("Y", "S", "Y", "S"), ("S", None, None, "Y")),
}


@dataclass(frozen=True)
class GateSpec:
    """One static CMOS gate: ``out = kind(*ins)``; an ``inv`` has `fingers` stripes."""

    out: str
    kind: Literal["inv", "nor2", "nand3"]
    ins: tuple[str, ...]
    fingers: int = 1

    @property
    def stripes(self) -> tuple[str, ...]:
        """The net on each poly stripe, left to right."""
        return self.ins * self.fingers if self.kind == "inv" else self.ins

    @property
    def bands(self) -> dict[str, tuple[str | None, ...]]:
        """Column nets of the p and n bands."""
        if self.kind == "inv":
            cols = tuple("S" if i % 2 == 0 else "Y" for i in range(self.fingers + 1))
            return {"p": cols, "n": cols}
        p, n = _KINDS[self.kind]
        return {"p": p, "n": n}

    @property
    def columns(self) -> int:
        return len(self.bands["p"])

    @property
    def out_column(self) -> int:
        """The column of the output's M3: an output column of both bands if
        there is one, else the rightmost output column."""
        ys = {b: {i for i, n in enumerate(c) if n == "Y"} for b, c in self.bands.items()}
        common = ys["p"] & ys["n"]
        return min(common) if common else max(ys["p"] | ys["n"])


def controller_gates(fingers: int) -> tuple[GateSpec, ...]:
    """The controller's gates in row order; the three drivers have `fingers` stripes."""
    return (
        GateSpec("RN", "inv", ("RESETN",)),
        GateSpec("REQ", "nor2", ("CLK", "RN")),
        GateSpec("EN", "nor2", ("REQ", "EVAL")),
        GateSpec("PRECHN", "inv", ("EN",), fingers),
        GateSpec("PN", "inv", ("PRECHN",)),
        GateSpec("ERN", "nor2", ("REQ", "SAMPLE")),
        GateSpec("EV0", "nor2", ("ERN", "PN")),
        GateSpec("EVB", "inv", ("EV0",)),
        GateSpec("EVAL", "inv", ("EVB",), fingers),
        GateSpec("SN", "nand3", ("REQ", "EVAL", "READY")),
        GateSpec("SAMPLE", "inv", ("SN",), fingers),
    )


def assign_levels(
    spans: dict[str, tuple[int, int]],
    levels: int,
    fixed: dict[str, int] | None = None,
) -> dict[str, int]:
    """A level for every net's x span, none two overlapping on one level.

    Two spans on one level keep `TRACK_END_SPACE` between them.  `fixed`
    nets keep their level.  Spans go in left to right, each on the lowest
    level free for it.
    """
    taken: dict[int, list[tuple[int, int]]] = {lv: [] for lv in range(levels)}
    result = dict(fixed or {})
    for net, level in result.items():
        taken[level].append(spans[net])
    for net in sorted(set(spans) - result.keys(), key=lambda n: spans[n]):
        lo, hi = spans[net]
        for level in range(levels):
            if all(
                lo - b >= TRACK_END_SPACE or a - hi >= TRACK_END_SPACE
                for a, b in taken[level]
            ):
                taken[level].append((lo, hi))
                result[net] = level
                break
        else:
            raise ValueError(f"no M2 level is free for {net} over x = {lo}..{hi}")
    return result


@dataclass(frozen=True)
class RomControlSpec:
    """The controller row.

    Args:
        fingers: stripes of each of the three output drivers.
        n_fins: fins of every nFET.
        p_fins: fins of every pFET.
        band_height: ``(n_band, p_band)`` in nm: the address-latch row's.
        vt: threshold flavor of every device.
    """

    fingers: int = 4
    n_fins: int = 4
    p_fins: int = 4
    band_height: tuple[int, int] = (243, 243)
    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"

    def __post_init__(self) -> None:
        object.__setattr__(self, "band_height", tuple(self.band_height))
        if isinstance(self.fingers, bool) or not isinstance(self.fingers, int):
            raise TypeError(f"fingers must be an integer, got {self.fingers!r}")
        if self.fingers < 1:
            raise ValueError(f"fingers must be >= 1, got {self.fingers}")
        lv = self.levels
        if lv[LEVELS - 1] > lv["p"] - LANDING_CLEAR:
            raise ValueError(
                f"bands of {self.band_height} nm leave {lv['p'] - lv['n']} nm between "
                f"the via rows; the seven M2 levels need {2 * LANDING_CLEAR + 6 * 36}"
            )
        self.net_levels  # noqa: B018 - fail early if the nets do not fit

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

    @property
    def cell_name(self) -> str:
        vt_tag = "" if self.vt == "rvt" else f"_{self.vt}"
        return (
            f"rom_control_f{self.fingers}_{self.n_fins}n{self.p_fins}p"
            f"_h{self.band_height[0]}x{self.band_height[1]}{vt_tag}"
        )

    # ── Columns ───────────────────────────────────────────────────────────────
    @property
    def gates(self) -> tuple[GateSpec, ...]:
        return controller_gates(self.fingers)

    @cached_property
    def placement(self) -> tuple[dict[str, int], dict[str, int]]:
        """``(first column of each gate, pin column of each input pin)``.

        A gap column follows every gate, and an input pin takes the gap
        before the first gate it drives.
        """
        first: dict[str, int] = {}
        pins: dict[str, int] = {}
        column = 0
        for gate in self.gates:
            for net in gate.ins:
                if net in CONTROL_INPUTS and net not in pins:
                    pins[net] = column
                    column += 1
            column = max(column, 1)  # column 0 is the gap at the row's start
            first[gate.out] = column
            column += gate.columns + 1
        return first, pins

    @property
    def columns(self) -> int:
        last = self.gates[-1]
        return self.placement[0][last.out] + last.columns

    @property
    def width(self) -> int:
        return (self.columns + 1) * GATE_PITCH

    @staticmethod
    def column_x(column: int) -> int:
        return (column + 1) * GATE_PITCH

    def gate_column_x(self, gate: GateSpec, column: int) -> int:
        return self.column_x(self.placement[0][gate.out] + column)

    def stripe_x(self, gate: GateSpec, stripe: int) -> int:
        return self.gate_column_x(gate, stripe) + GATE_PITCH // 2

    def out_x(self, gate: GateSpec) -> int:
        return self.gate_column_x(gate, gate.out_column)

    def pin_x(self, pin: str) -> int:
        return self.column_x(self.placement[1][pin])

    def contacts(self, gate: GateSpec) -> list[tuple[str, list[int]]]:
        """``(net, stripe xs)`` of each run of same-net stripes: one gate
        contact, on the first, serves the run."""
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
        """Each net's track terminals: ``"v1"`` gate xs and ``"v2"`` column xs."""
        nets: dict[str, dict[str, list[int]]] = {}
        for gate in self.gates:
            nets.setdefault(gate.out, {"v1": [], "v2": []})["v2"].append(self.out_x(gate))
            for net, run in self.contacts(gate):
                nets.setdefault(net, {"v1": [], "v2": []})["v1"].append(run[0])
        for pin in CONTROL_INPUTS:
            nets[pin]["v2"].append(self.pin_x(pin))
        return nets

    @cached_property
    def spans(self) -> dict[str, tuple[int, int]]:
        """Each net's M2 track extent, its terminals' outermost plus a pad."""
        spans = {}
        for net, t in self.terminals.items():
            xs = t["v1"] + t["v2"]
            spans[net] = (min(xs) - PAD, max(xs) + PAD)
        spans["PRECHN"] = (0, spans["PRECHN"][1])  # on to the address latches' track
        return spans

    @cached_property
    def net_levels(self) -> dict[str, int]:
        """The M2 level of every net; ``PRECHN`` has the top one."""
        return assign_levels(self.spans, LEVELS, fixed={"PRECHN": LEVELS - 1})

    @property
    def pin_positions(self) -> dict[str, tuple[str, tuple[float, float]]]:
        top = self.height - M3_EDGE - 9
        lv = self.levels
        pins: dict[str, tuple[str, tuple[float, float]]] = {
            pin: ("M3", (self.pin_x(pin), top)) for pin in CONTROL_INPUTS
        }
        pins["PRECHN"] = ("M2", (PAD, lv[self.net_levels["PRECHN"]]))
        for gate in self.gates:
            if gate.out in ("EVAL", "SAMPLE"):
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
                        (f"M{band.upper()}_{gate.out}_{i}", net(i), g, net(i + 1),
                         band, fins)  # fmt: skip
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

    def _lines(self, unit_fins: bool) -> list[str]:
        lines = []
        for device, drain, gate, source, flavor, fins in self.devices:
            model = self.bands[flavor].spec
            bulk = "VSS" if flavor == "n" else "VDD"
            if not unit_fins:
                lines.append(
                    f"{device} {drain} {gate} {source} {bulk} {model.model} "
                    f"nfin={fins} l={model.gate_length}n nf=1 m=1"
                )
                continue
            for fin in range(fins):
                d, s = (
                    f"{n}_f{fin}" if n in self.series_nodes else n
                    for n in (drain, source)
                )
                lines.append(
                    f"{device}_f{fin} {d} {gate} {s} {bulk} {model.model} "
                    f"L={model.gate_length}n W={FIN_WIDTH}n"
                )
        return lines

    def _subckt(self, title: str, comment: str, unit_fins: bool) -> str:
        return "\n".join(
            [
                comment,
                f".SUBCKT {title} {' '.join(self.pins)}",
                *self._lines(unit_fins),
                f".ENDS {title}",
                ".END",
                "",
            ]
        )

    def netlist(self, name: str | None = None) -> str:
        """A BSIM-CMG subcircuit sized by ``nfin``."""
        return self._subckt(
            name or self.cell_name, "* ASAP7 NOR ROM read controller", unit_fins=False
        )

    def lvs_reference(self, name: str | None = None) -> str:
        """The LVS reference: one MOS per FIN x GATE channel."""
        return self._subckt(
            name or self.cell_name,
            "* ASAP7 NOR ROM read controller LVS reference: one MOS per FIN x GATE channel",
            unit_fins=True,
        )


# ── Layout ────────────────────────────────────────────────────────────────────
def _draw_gate(cell: Any, spec: RomControlSpec, gate: GateSpec) -> None:
    """`gate`'s islands, its output's via-row joins and M3, and its gate contacts."""
    bands = spec.bands
    lv = spec.levels
    seam = spec.stack.seam_y(0)
    xs = [spec.gate_column_x(gate, i) for i in range(gate.columns)]
    x_out = spec.out_x(gate)
    for name, cols in gate.bands.items():
        band = bands[name]
        island(cell, band, xs, contacted=[x for x, c in zip(xs, cols) if c is not None])
        ys = []
        for x, c in zip(xs, cols):
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
            vias = ys
            m2_track(cell, y, cover[0] - PAD, cover[-1] + PAD, vias=vias)
            box(cell, "V2", x_out - 9, y - 9, x_out + 9, y + 9)
    level = lv[spec.net_levels[gate.out]]
    top = spec.height - M3_EDGE if gate.out in ("EVAL", "SAMPLE") else lv["p"] + PAD
    m3_column(cell, x_out, lv["n"] - PAD, top, vias=[level])
    for net, run in spec.contacts(gate):
        gate_contact(cell, seam, run, run[0], _reach(lv[spec.net_levels[net]], seam))


def build_rom_control_row(
    spec: RomControlSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Draw `spec` and return the gdspy Cell (nanometre coordinates)."""
    spec = spec or RomControlSpec()
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = (
        lib.new_cell(cell_name)
        if lib is not None
        else gdspy.Cell(cell_name, exclude_from_current=True)
    )
    lv = spec.levels
    draw_frame(cell, spec.stack, spec.width, spec.vt)
    draw_rails(cell, spec.stack, spec.width)
    assert spec.column_x(1) - ISLAND_OVERHANG >= SELECT_X_ENC
    for gate in spec.gates:
        _draw_gate(cell, spec, gate)
    # Every net's track: a V1 on each stripe it drives, the V2s are the columns'.
    for net, (lo, hi) in spec.spans.items():
        m2_track(cell, lv[spec.net_levels[net]], lo, hi, vias=spec.terminals[net]["v1"])
    # The input pins: an M3 column from their track to the row's top.
    for pin in CONTROL_INPUTS:
        y = lv[spec.net_levels[pin]]
        m3_column(cell, spec.pin_x(pin), y - PAD, spec.height - M3_EDGE, vias=[y])
    if draw_pin_labels:
        for pin, (metal, origin) in spec.pin_positions.items():
            label(cell, pin, metal, origin)
    box(cell, "BOUNDARY", 0, 0, spec.width, spec.height)
    return cell


def main(argv: list[str] | None = None) -> None:
    """Write a controller row GDS: ``asap7-rom-control --fingers 4``."""
    args = parse_spec(
        RomControlSpec, argv, description="Generate the ASAP7 NOR ROM read controller."
    )
    spec = args.spec
    cell, out = write_gds(lambda lib: build_rom_control_row(spec, lib=lib), args.out)
    print(f"✓ {cell.name}: {spec.width} x {spec.height} nm -> {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
