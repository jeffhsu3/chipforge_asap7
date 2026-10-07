"""Parametric output latch: a NAND SR latch on the sense amplifier, an inverter and a tristate driver.

The released bank ends each column with an SR latch of two NAND2s set and
reset by the sense amplifier's ``QA``/``QAN`` (both high while it precharges,
which is the latch's hold state), an inverter, and a large tristate inverter
onto the data pin, enabled by ``OE``/``OEB``.  `OutputLatchSpec` draws that on
two dense rows of the bitline leaf's row::

    row 1     .    .   Y2N   S    .    Qp  n1/S  S/n2  Qn  S/n2  n1/S  Qp      inverter, tristate
    stripe             Y2        ENB    A     EN    EN    A    ENB
    row 0    Y1  m1/Y1  S   m2/Y2  Y2                                        SR latch
    stripe      Y2    QA    QAN   Y1

The latch is one five-column island in each band: the two NAND2s share the
supply column in the middle, the series nodes ``m1``/``m2`` are uncontacted,
and the p band's pull-ups sit two to a drain.  The tristate is drawn in
pairs of fingers: the pFET enable stripes and the nFET enable stripes cannot
share poly, so each pair takes six stripes, ``ENB A EN EN A ENB``, with the
nFET series node under a VDD column and the pFET series node under a VSS
column.  ``Q`` collects the pairs on the p via row and leaves on M3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..layout.grid import GATE_PITCH
from ..layout.layers import box, require_gdspy
from ..layout.rules import TRACK_PITCH
from .cli import parse_spec, write_gds
from .finfet import SELECT_X_ENC
from .row import RowBand, RowStack
from .rowcell import (
    CAP,
    HALF,
    ISLAND_OVERHANG,
    PAD,
    RUNSET_RAIL_SPACE,
    draw_frame,
    draw_rails,
    gate_contact,
    island,
    label,
    landing,
    m2_track,
    m3_column,
    sd_contact,
    supply_contact,
    via_y,
)

__all__ = ["OUTPUT_LATCH_PINS", "OutputLatchSpec", "build_output_latch"]

OUTPUT_LATCH_PINS = ("QA", "QAN", "OE", "OEB", "Q", "VDD", "VSS")

_INVERTER_AT = 2  # the inverter's output column, in the row above the latch
_FINGERS_AT = 5  # first column of the tristate
# The bottom row's M2 tie tracks, from its seam: one just below it, two above.
_TIE_OFFSETS = (-19, 17, 53)


@dataclass(frozen=True)
class OutputLatchSpec:
    """A NAND SR latch, an inverter and a tristate output driver.

    Args:
        n_fins: fins of every nFET.
        p_fins: fins of every pFET.
        fingers: parallel fingers of the tristate driver, an even number.
            Each finger is a two-high stack per band, so the driver's
            strength is ``fingers`` times one device of the row.
        band_height: ``(n_band, p_band)`` of one row, in nm.  The default is
            the bitline leaf's row.
        vt: threshold flavor of every device.
    """

    n_fins: int = 3
    p_fins: int = 3
    fingers: int = 2
    band_height: tuple[int, int] = (135, 162)
    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"

    def __post_init__(self) -> None:
        object.__setattr__(self, "band_height", tuple(self.band_height))
        if isinstance(self.fingers, bool) or not isinstance(self.fingers, int):
            raise TypeError(f"fingers must be an integer, got {self.fingers!r}")
        if self.fingers < 2 or self.fingers % 2:
            raise ValueError(f"fingers must be an even number >= 2, got {self.fingers}")
        bands = self.bands
        n_lo, p_lo = bands["n0"], bands["p0"]
        y_n, y_p = via_y(n_lo, RUNSET_RAIL_SPACE), via_y(p_lo, RUNSET_RAIL_SPACE)
        low, _, high = self.tie_levels
        # The lowest tie sits a track pitch above the 80 nm track, on which
        # QA ties from below.
        if low - CAP < y_n + CAP + 18 or low < self.n_level + TRACK_PITCH:
            raise ValueError(
                f"an n band of {n_lo.height} nm leaves no room for the tie tracks"
            )
        if y_p - CAP < high + CAP + 18:
            raise ValueError(
                f"a p band of {p_lo.height} nm leaves no room for the tie tracks"
            )

    # ── Rows ──────────────────────────────────────────────────────────────────
    @property
    def stack(self) -> RowStack:
        """Two rows of the leaf's row, sharing the VDD rail between them."""
        return RowStack(
            rows=((self.n_fins, self.p_fins),) * 2,
            vt=self.vt,
            band_height=self.band_height,
        )

    @property
    def bands(self) -> dict[str, RowBand]:
        """``n0``, ``p0`` (bottom row) and ``p1``, ``n1`` (top row)."""
        n0, p0, p1, n1 = self.stack.bands()
        return {"n0": n0, "p0": p0, "p1": p1, "n1": n1}

    @property
    def height(self) -> int:
        return self.stack.height

    @property
    def rails(self) -> tuple[tuple[int, str], ...]:
        return self.stack.rails

    @property
    def cell_name(self) -> str:
        vt_tag = "" if self.vt == "rvt" else f"_{self.vt}"
        return (
            f"outlatch_{self.n_fins}n{self.p_fins}p_f{self.fingers}"
            f"_h{self.band_height[0]}x{self.band_height[1]}{vt_tag}"
        )

    # ── Columns ───────────────────────────────────────────────────────────────
    @property
    def columns(self) -> int:
        """Diffusion columns: the latch's five under the inverter, then six per finger pair."""
        return _FINGERS_AT + 1 + 3 * self.fingers

    @property
    def width(self) -> int:
        return (self.columns + 2) * GATE_PITCH

    @property
    def column_x(self) -> list[int]:
        return [(i + 1) * GATE_PITCH for i in range(self.columns)]

    def gate_x(self, left: int) -> int:
        """X of the gate stripe between columns `left` and `left + 1`."""
        return self.column_x[left] + GATE_PITCH // 2

    @property
    def n_level(self) -> int:
        """The bottom row's M2 track its n pads reach, a pitch above their via row's landings."""
        return via_y(self.bands["n0"], RUNSET_RAIL_SPACE) + HALF + 18 + HALF + 3

    @property
    def tie_levels(self) -> tuple[int, int, int]:
        """The bottom row's three M2 tie tracks above the 80 nm one: Y2, Y1, QAN.

        A gate contact's M1 bar runs from the seam to its tie, 9 nm beside the
        pads of the neighbouring columns, so the two have to clear each other
        by 18 nm in y (M1.S.6).  ``QA`` ties on the 80 nm track: its stripe
        has no pad beside it.
        """
        seam = self.stack.seam_y(0)
        low, mid, high = (seam + offset for offset in _TIE_OFFSETS)
        return low, mid, high

    @property
    def pairs(self) -> list[int]:
        """First column of each finger pair: its lower ``Q`` pFET drain."""
        return [_FINGERS_AT + 6 * k for k in range(self.fingers // 2)]

    @property
    def track_x(self) -> dict[str, int]:
        """X of the M3 columns that are pins."""
        x = self.column_x
        first = self.pairs[0]
        return {
            "QA": x[1],
            "QAN": x[2],
            "OEB": x[first + 2],
            "Q": x[first + 3],
            "OE": x[first + 4],
        }

    @property
    def pin_positions(self) -> dict[str, tuple[str, tuple[float, float]]]:
        tx = self.track_x
        return {
            "QA": ("M3", (tx["QA"], 40)),
            "QAN": ("M3", (tx["QAN"], 40)),
            "OE": ("M3", (tx["OE"], self.height - 40)),
            "OEB": ("M3", (tx["OEB"], self.height - 40)),
            "Q": ("M3", (tx["Q"], self.height - 40)),
            "VDD": ("M1", (self.width / 2, self.rails[1][0])),
            "VSS": ("M1", (self.width / 2, self.rails[0][0])),
        }

    # ── Netlist ───────────────────────────────────────────────────────────────
    @property
    def devices(self) -> tuple[tuple[str, str, str, str, str, int], ...]:
        """``(name, drain, gate, source, flavor, fins)`` for every transistor."""
        n, p = self.n_fins, self.p_fins
        devices = [
            # Y1 = NAND(Y2, QA); Y2 = NAND(QAN, Y1).
            ("MN1A", "Y1", "Y2", "M1", "n", n),
            ("MN1B", "M1", "QA", "VSS", "n", n),
            ("MP1A", "Y1", "Y2", "VDD", "p", p),
            ("MP1B", "Y1", "QA", "VDD", "p", p),
            ("MN2A", "M2", "QAN", "VSS", "n", n),
            ("MN2B", "Y2", "Y1", "M2", "n", n),
            ("MP2A", "Y2", "QAN", "VDD", "p", p),
            ("MP2B", "Y2", "Y1", "VDD", "p", p),
            ("MNI", "Y2N", "Y2", "VSS", "n", n),
            ("MPI", "Y2N", "Y2", "VDD", "p", p),
        ]
        for f in range(self.fingers):
            devices += [
                (f"MNA{f}", f"N2_{f}", "Y2N", "VSS", "n", n),
                (f"MNE{f}", "Q", "OE", f"N2_{f}", "n", n),
                (f"MPA{f}", f"N1_{f}", "Y2N", "VDD", "p", p),
                (f"MPE{f}", "Q", "OEB", f"N1_{f}", "p", p),
            ]
        return tuple(devices)

    @property
    def series_nodes(self) -> frozenset[str]:
        """Nets that are uncontacted diffusion between two channels.

        The extractor sees one such net per fin, so an LVS reference has to
        name them per fin; the simulation netlist keeps them whole.
        """
        return frozenset(
            {"M1", "M2"} | {f"N{k}_{f}" for k in (1, 2) for f in range(self.fingers)}
        )

    def netlist(self, name: str | None = None) -> str:
        """A BSIM-CMG subcircuit sized by ``nfin``."""
        title = name or self.cell_name
        bands = self.bands
        lines = [
            "* ASAP7 output latch: NAND SR latch, inverter and tristate driver",
            f".SUBCKT {title} {' '.join(OUTPUT_LATCH_PINS)}",
        ]
        for device, drain, gate, source, flavor, fins in self.devices:
            band = bands["n0"] if flavor == "n" else bands["p0"]
            bulk = "VSS" if flavor == "n" else "VDD"
            lines.append(
                f"{device} {drain} {gate} {source} {bulk} {band.spec.model} "
                f"nfin={fins} l={band.spec.gate_length}n nf=1 m=1"
            )
        lines += [f".ENDS {title}", ".END", ""]
        return "\n".join(lines)


# ── Layout ────────────────────────────────────────────────────────────────────
def build_output_latch(
    spec: OutputLatchSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Draw `spec` and return the gdspy Cell (nanometre coordinates)."""
    spec = spec or OutputLatchSpec()
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = (
        lib.new_cell(cell_name)
        if lib is not None
        else gdspy.Cell(cell_name, exclude_from_current=True)
    )
    bands = spec.bands
    n0, p0, p1, n1 = (bands[k] for k in ("n0", "p0", "p1", "n1"))
    width, height = spec.width, spec.height
    c, g = spec.column_x, spec.gate_x
    seam0, seam1 = spec.stack.seam_ys
    y_n0, y_p0 = via_y(n0, RUNSET_RAIL_SPACE), via_y(p0, RUNSET_RAIL_SPACE)
    y_p1, y_n1 = via_y(p1, RUNSET_RAIL_SPACE), via_y(n1, RUNSET_RAIL_SPACE)
    lvl = spec.n_level  # 80: QA's tie, and nothing else on the bottom row
    tie_y2, tie_y1, tie_qan = spec.tie_levels
    # The top row's ties mirror the bottom row's about its own seam: two in the
    # p band for Y2 and Y2N, one in the n band for OEB.
    tie_top_y2, tie_a, tie_oeb = (
        seam1 - _TIE_OFFSETS[1],
        seam1 - _TIE_OFFSETS[2],
        seam1 - _TIE_OFFSETS[0],
    )
    inv, first = _INVERTER_AT, spec.pairs[0]
    tx = spec.track_x

    draw_frame(cell, spec.stack, width, spec.vt)

    # Bottom row: the SR latch.  n: Y1 -Y2- m1 -QA- VSS -QAN- m2 -Y1- Y2, the
    # series nodes uncontacted; p: VDD -Y2- Y1 -QA- VDD -QAN- Y2 -Y1- VDD.
    island(cell, n0, c[0:5], contacted=[c[0], c[2], c[4]])
    island(cell, p0, c[0:5])
    # Top row: the inverter, then the tristate pairs.
    island(cell, n1, [c[inv], c[inv + 1]])
    island(cell, p1, [c[inv], c[inv + 1]])
    for b in spec.pairs:
        island(cell, n1, c[b + 1 : b + 6], contacted=[c[b + 1], c[b + 3], c[b + 5]])
        island(cell, p1, c[b : b + 3], contacted=[c[b], c[b + 2]])
        island(cell, p1, c[b + 4 : b + 7], contacted=[c[b + 4], c[b + 6]])
    assert c[0] - ISLAND_OVERHANG >= SELECT_X_ENC

    draw_rails(cell, spec.stack, width)
    for column in (0, 2, 4):
        supply_contact(cell, p0, c[column])
    supply_contact(cell, n0, c[2])
    supply_contact(cell, n1, c[inv + 1])
    supply_contact(cell, p1, c[inv + 1])
    for b in spec.pairs:
        supply_contact(cell, n1, c[b + 1])
        supply_contact(cell, n1, c[b + 5])
        supply_contact(cell, p1, c[b + 2])
        supply_contact(cell, p1, c[b + 4])

    # Y1: its n drain's column carries M3 up to the p via row, where a short
    # track reaches its p drain, and to its gates' tie.
    sd_contact(cell, n0, c[0], rail_space=RUNSET_RAIL_SPACE)
    landing(cell, n0, c[0], rail_space=RUNSET_RAIL_SPACE)
    sd_contact(cell, p0, c[1], rail_space=RUNSET_RAIL_SPACE)
    m3_column(cell, c[0], y_n0 - CAP, y_p0 + CAP, vias=[tie_y1, y_p0])
    m2_track(cell, y_p0, c[0] - PAD, c[1] + PAD, vias=[c[1]])
    gate_contact(cell, seam0, [g(3)], g(3), tie_y1 + CAP)
    m2_track(cell, tie_y1, c[0] - PAD, g(3) + PAD, vias=[g(3)])
    # Y2: its n drain climbs one column, its p drain another, both to the tie
    # below the seam; the p drain's column carries on up to the inverter.
    sd_contact(cell, n0, c[4], rail_space=RUNSET_RAIL_SPACE)
    landing(cell, n0, c[4], rail_space=RUNSET_RAIL_SPACE)
    m3_column(cell, c[4], y_n0 - CAP, tie_y2 + CAP, vias=[tie_y2])
    sd_contact(cell, p0, c[3], rail_space=RUNSET_RAIL_SPACE)
    landing(cell, p0, c[3], rail_space=RUNSET_RAIL_SPACE)
    m3_column(cell, c[3], tie_y2 - CAP, tie_top_y2 + CAP, vias=[tie_y2, tie_top_y2])
    gate_contact(cell, seam0, [g(0)], g(0), tie_y2 - CAP)
    m2_track(cell, tie_y2, g(0) - PAD, c[4] + PAD, vias=[g(0)])
    # QA on the 80 nm track, QAN above the seam, each to an M3 pin stub.
    gate_contact(cell, seam0, [g(1)], g(1), lvl - CAP)
    m2_track(cell, lvl, c[1] - PAD, g(1) + PAD, vias=[g(1)])
    m3_column(cell, tx["QA"], 0, lvl + CAP, vias=[lvl])
    gate_contact(cell, seam0, [g(2)], g(2), tie_qan + CAP)
    m2_track(cell, tie_qan, c[2] - PAD, g(2) + PAD, vias=[g(2)])
    m3_column(cell, tx["QAN"], 0, tie_qan + CAP, vias=[tie_qan])

    # Top row.  The inverter's gates tie to Y2's column; its drains join on
    # their own column's M3, which also carries Y2N to the tristate's A gates.
    gate_contact(cell, seam1, [g(inv)], g(inv), tie_top_y2 - CAP)
    m2_track(cell, tie_top_y2, g(inv) - PAD, c[inv + 1] + PAD, vias=[g(inv)])
    sd_contact(cell, p1, c[inv], rail_space=RUNSET_RAIL_SPACE)
    landing(cell, p1, c[inv], rail_space=RUNSET_RAIL_SPACE)
    sd_contact(cell, n1, c[inv], rail_space=RUNSET_RAIL_SPACE)
    landing(cell, n1, c[inv], rail_space=RUNSET_RAIL_SPACE)
    a_stripes = [g(b + 1) for b in spec.pairs] + [g(b + 4) for b in spec.pairs]
    m3_column(cell, c[inv], y_p1 - CAP, y_n1 + CAP, vias=[tie_a])
    for s in a_stripes:
        gate_contact(cell, seam1, [s], s, tie_a - CAP)
    m2_track(cell, tie_a, c[inv] - PAD, max(a_stripes) + PAD, vias=a_stripes)
    # OE: the two nFET enable stripes of a pair share one pad, contacted on
    # the Q column between them, below the seam.  OEB: the pFET enable
    # stripes, contacted above the seam; adjacent pairs' meet on one pad.
    q_columns = [b + 3 for b in spec.pairs]
    for b in spec.pairs:
        gate_contact(cell, seam1, [g(b + 2), g(b + 3)], c[b + 3], tie_top_y2 - CAP)
    m2_track(cell, tie_top_y2, c[q_columns[0]] - PAD, max(c[q_columns[-1]], c[first + 4]) + PAD,
             vias=[c[q] for q in q_columns])  # fmt: skip
    m3_column(cell, tx["OE"], tie_top_y2 - CAP, height, vias=[tie_top_y2])
    oeb_stripes = sorted({g(b) for b in spec.pairs} | {g(b + 5) for b in spec.pairs})
    runs: list[list[int]] = []
    for s in oeb_stripes:
        if runs and s - runs[-1][-1] == GATE_PITCH:
            runs[-1].append(s)
        else:
            runs.append([s])
    for run in runs:
        gate_contact(cell, seam1, run, run[0], tie_oeb + CAP)
    m2_track(
        cell,
        tie_oeb,
        oeb_stripes[0] - PAD,
        oeb_stripes[-1] + PAD,
        vias=[run[0] for run in runs],
    )
    m3_column(cell, tx["OEB"], tie_oeb - CAP, height, vias=[tie_oeb])
    # Q: every pFET drain on the p via row's track; each pair's nFET drain
    # climbs its own column to it, and the first pair's column is the pin.
    q_p_columns = sorted({b for b in spec.pairs} | {b + 6 for b in spec.pairs})
    for column in q_p_columns:
        sd_contact(cell, p1, c[column], rail_space=RUNSET_RAIL_SPACE)
    m2_track(
        cell,
        y_p1,
        c[q_p_columns[0]] - PAD,
        c[q_p_columns[-1]] + PAD,
        vias=[c[q] for q in q_p_columns],
    )
    for q in q_columns:
        sd_contact(cell, n1, c[q], rail_space=RUNSET_RAIL_SPACE)
        landing(cell, n1, c[q], rail_space=RUNSET_RAIL_SPACE)
        top = height if c[q] == tx["Q"] else y_n1 + CAP
        m3_column(cell, c[q], y_p1 - CAP, top, vias=[y_p1])

    if draw_pin_labels:
        for pin, (metal, origin) in spec.pin_positions.items():
            if pin not in ("VDD", "VSS"):
                label(cell, pin, metal, origin)
        # Every rail is its own conductor until a power grid joins them.
        for y_rail, net in spec.rails:
            label(cell, net, "M1", (width / 2, y_rail))
    box(cell, "BOUNDARY", 0, 0, width, height)
    return cell


def main(argv: list[str] | None = None) -> None:
    """Write an output-latch GDS: ``asap7-output-latch --fingers 4``."""
    args = parse_spec(
        OutputLatchSpec,
        argv,
        description="Generate an ASAP7 output latch: NAND SR latch, inverter and tristate driver.",
    )
    spec = args.spec
    cell, out = write_gds(lambda lib: build_output_latch(spec, lib=lib), args.out)
    print(f"✓ {cell.name}: {spec.width} x {spec.height} nm -> {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
