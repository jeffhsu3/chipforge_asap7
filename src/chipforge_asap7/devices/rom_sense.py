"""The NOR ROM's read leaves: one row of bitline prechargers and NAND2 sensors.

The center lane ends every bit in two bitlines: ``BLL<b>`` from the left half
and ``BLR<b>`` from the right.  A read fires one wordline, so at most one of
the two discharges; the other stays precharged.  A NAND2 of the pair is
therefore the bit, with no half select at all: both high (nothing
programmed, or the word in the other half) reads 0, either low reads 1.

One leaf per bit, on one dense row (`chipforge_asap7.devices.rowcell`)::

    p band   BLL -PRECHN- VDD -BLL- D -BLR- VDD -PRECHN- BLR
    n band               VSS -BLL- x -BLR- D
    column    c0          c1        c2      c3            c4

The two outer pFETs precharge the bitlines while ``PRECHN`` is low; the inner
four transistors are the NAND2, its series node ``x`` uncontacted.  ``D``
joins the p drain at c2 to the n drain at c3 on an M3 column at c2 and the n
via row's M2.  Each bitline's drain and gate meet on its own M3 column, c0
for ``BLL`` and c4 for ``BLR``, which runs the row's height so that an M4
track at any height in the row can land on it.  ``PRECHN`` collects every
leaf's two outer gates on one M2 track and reaches the M3 column at the
row's first column, which also runs the full height so that stacked rows
share it.

`RomSenseRowSpec` draws ``leaves`` of these, six columns apart, on one row;
`chipforge_asap7.devices.rom_lane` stacks the rows and routes the bitlines.

``D`` falls back to 0 at the next precharge, so ``sample=True`` puts a
sample latch beside each leaf, transparent while ``SAMPLE`` is high and
holding ``Q`` otherwise.  It is the address latch of
`chipforge_asap7.devices.rom_latch` without its pin column: ``D`` reaches the
latch's n drain on one crossing track and its p drain on another, ``DP``,
both from the leaf's c2 column::

    p band   Y -X- VDD -Y- Q -SAMPLE- X -SAMPLEB- D
    n band               D -SAMPLE- X -SAMPLEB- Q -Y- VSS -X- Y
    column   l0    l1     l2        l3          l4    l5     l6

A leaf, a gap and a latch make a 13-column slice.  The row starts with
``PRECHN``'s and ``SAMPLE``'s full-height M3 columns and two inverters on one
island: ``SAMPLE`` makes the row's ``SAMPLEB``, and that makes the row's own
copy of ``SAMPLE``, which is what the latches see.  Long M3 and every latch's
gates on the column would leave a slow edge at the far rows, with the sample
and register latches conducting together while it passes; SPICE at 1024x148
shows ``Q`` changing before ``CLK`` rises.  So the column drives one inverter
per row.  The slice carries eight M2 levels between the
via rows, ``D PRECHN Q SAMPLE DP SAMPLEB Y X`` from the bottom; all but the
latch's own ``Q``, ``Y`` and ``X`` cross the leaf, under its bitline ties.
That needs a 513 nm row of 4-fin bands, the default when ``sample`` is set.
(``D`` cannot take the n via row instead: a V1 on a track that ends beside
the next column's standing landing faces it at 28 nm, M2.S.5.)

``register=True`` makes ``Q`` a rising-edge register output.  A second latch
follows each sample latch, the same circuit with its clocks swapped:
transparent while ``SAMPLE`` is low, holding while it is high.  The sample
latch becomes the master of a flip-flop on ``SAMPLE``'s falling edge, which
the controller makes just after ``CLK`` rises and only once the read has
been sampled, so ``Q`` takes the new word then and holds it for the whole
next cycle.  The sample latch's output becomes ``LQ<k>``.  The register
latch's l0 column carries a jog that takes ``LQ`` from its ``Q`` track down
to ``D`` and up to ``DP``, and the two latches take ``Y`` and ``X`` on
opposite levels.  Each latch's ``Y`` runs its full width, so on one level the
two tracks would end 20 nm apart; on opposite levels every pair of ends on a
level is 47 nm apart.  The slice is 20 columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..layout.grid import FIN_WIDTH, GATE_PITCH
from ..layout.layers import box, require_gdspy
from .cli import parse_spec, write_gds
from .finfet import SELECT_X_ENC
from .row import RowBand, RowStack
from .rowcell import (
    CAP,
    HALF,
    ISLAND_OVERHANG,
    PAD,
    V3_M3_CAP,
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

__all__ = [
    "LEAF_COLUMNS",
    "REGISTER_SLICE_COLUMNS",
    "SAMPLE_SLICE_COLUMNS",
    "RomSenseRowSpec",
    "build_rom_sense_row",
]

#: Columns per leaf: five of diffusion and one empty, between leaves.
LEAF_COLUMNS = 6
#: Columns per sample latch: five p, five n (offset by two).
SAMPLE_LATCH_COLUMNS = 7
#: A leaf, its gap and its sample latch.
SAMPLE_SLICE_COLUMNS = LEAF_COLUMNS + SAMPLE_LATCH_COLUMNS
#: With ``register``, the register latch straight after the sample latch.
REGISTER_SLICE_COLUMNS = SAMPLE_SLICE_COLUMNS + SAMPLE_LATCH_COLUMNS
#: Before the first sample slice: PRECHN's M3, SAMPLE's, the two inverters of
#: the row's clocks on one island (three columns) and a gap.
SAMPLE_LEAD_COLUMNS = 6
#: A standing landing's end faces an M2 track's side across 25 nm (M2.S.2),
#: 51 nm centre to centre from its via.
LANDING_CLEAR = 51
#: The sample slice's M2 levels, bottom up, a track apart from the n via row.
#: D, Q and DP sit two tracks apart: D and Q split the latch's l2 column, Q
#: and DP its l4.
SAMPLE_LEVELS = ("D", "PRECHN", "Q", "CK", "DP", "CKB", "Y", "X")
#: The defaults of each kind of row: ``(n_fins, p_fins, band_height)``.
ROW_DEFAULTS = {False: (2, 2, (162, 189)), True: (4, 4, (243, 270))}
#: How far a full-height M3 column stops short of the row edge.  Mirrored
#: rows stack column on column, so two ends face each other at twice this,
#: past the 31 nm two short edges need.
M3_EDGE = 17
# The M2 tie tracks above the seam, as offsets from the p via row: BLL's gates
# to c0, BLR's to c4.  A gate's M1 bar ends 18 nm under the next column's
# pad (M1.S.6), so the higher tie sits 46 nm under the via row.
_BLL_TIE = -46
_BLR_TIE = -82
# PRECHN's track below the seam, from it.
_PRECHN_TIE = -19


@dataclass(frozen=True)
class RomSenseRowSpec:
    """One row of NOR ROM read leaves.

    Args:
        leaves: read leaves on the row, one per bit.
        n_fins: fins of every nFET.  ``None`` is 2, or 4 with ``sample``.
        p_fins: fins of every pFET, precharger and NAND2 pull-up alike.
            ``None`` is 2, or 4 with ``sample``.
        band_height: ``(n_band, p_band)`` of the row, in nm.  ``None`` is a
            351 nm row, four of which span one strap group of the 2-fin array
            (16 bits and a strap row), or with ``sample`` a 513 nm row.
        vt: threshold flavor of every device.
        sample: put a sample latch beside each leaf, with a ``Q`` pin.
        register: follow each sample latch with a register latch, so that
            ``Q`` changes only as ``SAMPLE`` falls.  Needs ``sample``.
    """

    leaves: int = 1
    n_fins: int | None = None
    p_fins: int | None = None
    band_height: tuple[int, int] | None = None
    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"
    sample: bool = False
    register: bool = False

    def __post_init__(self) -> None:
        n_fins, p_fins, band_height = ROW_DEFAULTS[bool(self.sample)]
        for field, default in (
            ("n_fins", n_fins),
            ("p_fins", p_fins),
            ("band_height", band_height),
        ):
            if getattr(self, field) is None:
                object.__setattr__(self, field, default)
        object.__setattr__(self, "band_height", tuple(self.band_height))
        if isinstance(self.leaves, bool) or not isinstance(self.leaves, int):
            raise TypeError(f"leaves must be an integer, got {self.leaves!r}")
        if self.leaves < 1:
            raise ValueError(f"leaves must be >= 1, got {self.leaves}")
        if self.register and not self.sample:
            raise ValueError("register needs sample: it follows the sample latch")
        n, p = self.bands["n"], self.bands["p"]
        y_n, y_p = via_y(n), via_y(p)
        seam = self.stack.seam_y(0)
        bll, blr, prechn = self.levels["BLL"], self.levels["BLR"], self.levels["PRECHN"]
        # Three M2 tracks a pitch apart: PRECHN's below the seam, BLR's and
        # BLL's above it.
        if blr - prechn < 36 or bll - blr < 36:
            raise ValueError(
                f"a p band of {p.height} nm leaves no room for the two bitline ties"
            )
        # PRECHN's bar on the stripe beside D's n pad clears it by 18 nm.
        if prechn - CAP < y_n + CAP + 18:
            raise ValueError(
                f"an n band of {n.height} nm leaves no room for the PRECHN tie"
            )
        if not (y_n < seam < y_p):
            raise ValueError("the via rows have to lie on either side of the seam")
        if self.sample:
            lv = self.levels
            # The top track's gate bars clear the p pads; the tracks that
            # cross the leaf pass a track under its BLR tie.
            if lv["X"] > y_p - LANDING_CLEAR or lv["CKB"] > blr - 36:
                raise ValueError(
                    f"bands of {self.band_height} nm leave {y_p - y_n} nm between "
                    "the via rows; the sample slice's eight M2 levels need "
                    f"{2 * LANDING_CLEAR + 7 * 36}, and six of them have to "
                    "pass under the BLR tie"
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
    def cell_name(self) -> str:
        vt_tag = "" if self.vt == "rvt" else f"_{self.vt}"
        return (
            f"rom_sense{'_q' if self.sample else ''}{'r' if self.register else ''}"
            f"_x{self.leaves}"
            f"_{self.n_fins}n{self.p_fins}p"
            f"_h{self.band_height[0]}x{self.band_height[1]}{vt_tag}"
        )

    # ── Columns ───────────────────────────────────────────────────────────────
    @property
    def lead(self) -> int:
        """Columns before the first leaf."""
        return SAMPLE_LEAD_COLUMNS if self.sample else 1

    @property
    def pitch(self) -> int:
        """Columns from one leaf to the next."""
        if self.register:
            return REGISTER_SLICE_COLUMNS
        return SAMPLE_SLICE_COLUMNS if self.sample else LEAF_COLUMNS

    @property
    def columns(self) -> int:
        """The lead, then the slices, with no gap after the last."""
        last = self.pitch if self.sample else LEAF_COLUMNS - 1
        return self.lead + self.pitch * (self.leaves - 1) + last

    @property
    def width(self) -> int:
        return (self.columns + 1) * GATE_PITCH

    @property
    def column_x(self) -> list[int]:
        return [(i + 1) * GATE_PITCH for i in range(self.columns)]

    def leaf_x(self, leaf: int, column: int) -> int:
        """X of column `column` (0 to 4) of `leaf`."""
        return self.column_x[self.lead + self.pitch * leaf + column]

    def latch_x(self, leaf: int, column: int) -> int:
        """X of column `column` (0 to 6) of `leaf`'s sample latch."""
        if not self.sample:
            raise ValueError("this row has no sample latches (sample=False)")
        return self.leaf_x(leaf, LEAF_COLUMNS + column)

    def latch_gate_x(self, leaf: int, gate: int) -> int:
        """X of gate `gate` (0 to 5) of `leaf`'s sample latch."""
        return self.latch_x(leaf, gate) + GATE_PITCH // 2

    def register_x(self, leaf: int, column: int) -> int:
        """X of column `column` (0 to 6) of `leaf`'s register latch."""
        if not self.register:
            raise ValueError("this row has no register latches (register=False)")
        return self.leaf_x(leaf, SAMPLE_SLICE_COLUMNS + column)

    def gate_x(self, leaf: int, gate: int) -> int:
        """X of gate `gate` (0 to 3) of `leaf`, between its columns `gate` and `gate + 1`."""
        return self.leaf_x(leaf, gate) + GATE_PITCH // 2

    @property
    def prechn_x(self) -> int:
        return self.column_x[0]

    @property
    def sample_x(self) -> int:
        """X of ``SAMPLE``'s full-height M3 column."""
        if not self.sample:
            raise ValueError("this row has no sample latches (sample=False)")
        return self.column_x[1]

    def q_x(self, leaf: int) -> int:
        """X of `leaf`'s ``Q`` M3 column: the l2 of its last latch."""
        if self.register:
            return self.register_x(leaf, 2)
        return self.latch_x(leaf, 2)

    @property
    def levels(self) -> dict[str, int]:
        """Y of each M2 track: the two via rows, the gate ties and, with
        ``sample``, the latch's tracks (``CK`` is ``SAMPLE``, ``CKB`` is
        ``SAMPLEB``)."""
        y_n, y_p = via_y(self.bands["n"]), via_y(self.bands["p"])
        seam = self.stack.seam_y(0)
        levels = {
            "n": y_n,
            "p": y_p,
            "BLL": y_p + _BLL_TIE,
            "BLR": y_p + _BLR_TIE,
            "PRECHN": seam + _PRECHN_TIE,
        }
        if self.sample:
            first = y_n + LANDING_CLEAR
            levels.update({k: first + 36 * i for i, k in enumerate(SAMPLE_LEVELS)})
        return levels

    @property
    def track_span(self) -> tuple[int, int]:
        """Lowest and highest M4 track centre a bitline column can take a V3 at."""
        return M3_EDGE + V3_M3_CAP, self.height - M3_EDGE - V3_M3_CAP

    @property
    def q_track_span(self) -> tuple[int, int]:
        """Lowest and highest M4 track centre the ``Q`` column can take a V3 at."""
        lv = self.levels
        return lv["Q"] - CAP + V3_M3_CAP, lv["p"] + PAD - V3_M3_CAP

    @property
    def pin_positions(self) -> dict[str, tuple[str, tuple[float, float]]]:
        mid = self.height / 2
        pins: dict[str, tuple[str, tuple[float, float]]] = {}
        for k in range(self.leaves):
            pins[f"BLL<{k}>"] = ("M3", (self.leaf_x(k, 0), mid))
            pins[f"BLR<{k}>"] = ("M3", (self.leaf_x(k, 4), mid))
            pins[f"D<{k}>"] = ("M3", (self.leaf_x(k, 2), self.levels["n"]))
            if self.sample:
                pins[f"Q<{k}>"] = ("M3", (self.q_x(k), self.levels["Q"]))
        pins["PRECHN"] = ("M3", (self.prechn_x, mid))
        if self.sample:
            pins["SAMPLE"] = ("M3", (self.sample_x, mid))
        pins["VDD"] = ("M1", (self.width / 2, self.rails[1][0]))
        pins["VSS"] = ("M1", (self.width / 2, self.rails[0][0]))
        return pins

    @property
    def pins(self) -> tuple[str, ...]:
        return tuple(self.pin_positions)

    # ── Netlist ───────────────────────────────────────────────────────────────
    @property
    def lead_devices(self) -> tuple[tuple[str, str, str, str, str, int], ...]:
        """The row's own transistors: with ``sample``, its two clock inverters."""
        if not self.sample:
            return ()
        return (
            ("MPS", "SAMPLEB", "SAMPLE", "VDD", "p", self.p_fins),
            ("MNS", "SAMPLEB", "SAMPLE", "VSS", "n", self.n_fins),
            ("MPC", "SAMPLEL", "SAMPLEB", "VDD", "p", self.p_fins),
            ("MNC", "SAMPLEL", "SAMPLEB", "VSS", "n", self.n_fins),
        )

    def leaf_devices(self, k: int) -> tuple[tuple[str, str, str, str, str, int], ...]:
        """Leaf `k`'s transistors and its latches': nets ``BLL<k>``,
        ``BLR<k>``, ``D<k>``, ``Q<k>``, the NAND2's series node ``X<k>``, the
        sample latch's nodes ``LX<k>``/``LY<k>`` and, with ``register``, its
        output ``LQ<k>`` and the register latch's ``RX<k>``/``RY<k>``."""
        n, p = self.n_fins, self.p_fins
        bll, blr, d, x = f"BLL<{k}>", f"BLR<{k}>", f"D<{k}>", f"X{k}"
        devices = [
            (f"MPL{k}", bll, "PRECHN", "VDD", "p", p),
            (f"MPR{k}", blr, "PRECHN", "VDD", "p", p),
            (f"MPA{k}", d, bll, "VDD", "p", p),
            (f"MPB{k}", d, blr, "VDD", "p", p),
            (f"MNA{k}", x, bll, "VSS", "n", n),
            (f"MNB{k}", d, blr, x, "n", n),
        ]
        if self.sample:
            q = f"LQ{k}" if self.register else f"Q<{k}>"
            devices += _latch(
                k, "", d, q, f"LX{k}", f"LY{k}", ("SAMPLEL", "SAMPLEB"), n, p
            )
        if self.register:
            devices += _latch(
                k, "R", f"LQ{k}", f"Q<{k}>", f"RX{k}", f"RY{k}",
                ("SAMPLEB", "SAMPLEL"), n, p,
            )  # fmt: skip
        return tuple(devices)

    @property
    def devices(self) -> tuple[tuple[str, str, str, str, str, int], ...]:
        """``(name, drain, gate, source, flavor, fins)`` for every transistor."""
        return self.lead_devices + tuple(
            d for k in range(self.leaves) for d in self.leaf_devices(k)
        )

    @property
    def series_nodes(self) -> frozenset[str]:
        """The NAND2 stacks' uncontacted inner nodes, one net per fin in an extraction."""
        return frozenset(f"X{k}" for k in range(self.leaves))

    def _lines(self, unit_fins: bool) -> list[str]:
        bands = self.bands
        lines = []
        for device, drain, gate, source, flavor, fins in self.devices:
            model = bands[flavor].spec
            bulk = "VSS" if flavor == "n" else "VDD"
            if not unit_fins:
                lines.append(
                    f"{device} {drain} {gate} {source} {bulk} {model.model} "
                    f"nfin={fins} l={model.gate_length}n nf=1 m=1"
                )
                continue
            for fin in range(fins):
                d, s = (
                    f"{net}_f{fin}" if net in self.series_nodes else net
                    for net in (drain, source)
                )
                lines.append(
                    f"{device}_f{fin} {d} {gate} {s} {bulk} {model.model} "
                    f"L={model.gate_length}n W={FIN_WIDTH}n"
                )
        return lines

    def netlist(self, name: str | None = None) -> str:
        """A BSIM-CMG subcircuit sized by ``nfin``."""
        title = name or self.cell_name
        return "\n".join(
            [
                "* ASAP7 NOR ROM read leaves: bitline prechargers and NAND2 sensors",
                f".SUBCKT {title} {' '.join(self.pins)}",
                *self._lines(unit_fins=False),
                f".ENDS {title}",
                ".END",
                "",
            ]
        )

    def lvs_reference(self, name: str | None = None) -> str:
        """The LVS reference: one MOS per FIN x GATE channel."""
        title = name or self.cell_name
        return "\n".join(
            [
                "* ASAP7 NOR ROM read leaves LVS reference: one MOS per FIN x GATE channel",
                f".SUBCKT {title} {' '.join(self.pins)}",
                *self._lines(unit_fins=True),
                f".ENDS {title}",
                ".END",
                "",
            ]
        )


def _latch(
    k: int,
    tag: str,
    d: str,
    q: str,
    x: str,
    y: str,
    clocks: tuple[str, str],
    n: int,
    p: int,
) -> list[tuple[str, str, str, str, str, int]]:
    """A TG latch, transparent while ``clocks[0]`` is high: its pass gates
    take ``d`` to ``x`` and its feedback gates ``q`` back to ``x``."""
    ck, ckb = clocks
    return [
        (f"M{tag}PI{k}", y, x, "VDD", "p", p),
        (f"M{tag}PO{k}", q, y, "VDD", "p", p),
        (f"M{tag}PF{k}", q, ck, x, "p", p),
        (f"M{tag}PT{k}", x, ckb, d, "p", p),
        (f"M{tag}NT{k}", d, ck, x, "n", n),
        (f"M{tag}NF{k}", x, ckb, q, "n", n),
        (f"M{tag}NO{k}", q, y, "VSS", "n", n),
        (f"M{tag}NI{k}", y, x, "VSS", "n", n),
    ]


# ── Layout ────────────────────────────────────────────────────────────────────
def build_rom_sense_row(
    spec: RomSenseRowSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Draw `spec` and return the gdspy Cell (nanometre coordinates)."""
    spec = spec or RomSenseRowSpec()
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = (
        lib.new_cell(cell_name)
        if lib is not None
        else gdspy.Cell(cell_name, exclude_from_current=True)
    )
    n, p = spec.bands["n"], spec.bands["p"]
    width, height = spec.width, spec.height
    seam = spec.stack.seam_y(0)
    lv = spec.levels
    top = height - M3_EDGE

    draw_frame(cell, spec.stack, width, spec.vt)
    draw_rails(cell, spec.stack, width)
    assert spec.leaf_x(0, 0) - ISLAND_OVERHANG >= SELECT_X_ENC

    prechn_gates: list[int] = []
    ck_gates: list[int] = []
    ckb_gates: list[int] = []
    if spec.sample:
        ck_gates, ckb_gates = _draw_sample_lead(cell, spec)
    for k in range(spec.leaves):
        c = [spec.leaf_x(k, i) for i in range(5)]
        g = [spec.gate_x(k, i) for i in range(4)]
        prechn_gates += [g[0], g[3]]
        island(cell, p, c)
        island(cell, n, c[1:4], contacted=[c[1], c[3]])
        supply_contact(cell, p, c[1])
        supply_contact(cell, p, c[3])
        supply_contact(cell, n, c[1])

        # BLL: c0's drain and the stripe at g1 on c0's column.  The tie runs
        # on up into the landing: 20 nm apart, its side would face the
        # landing's end (M2.S.2).
        sd_contact(cell, p, c[0])
        landing(cell, p, c[0])
        gate_contact(cell, seam, [g[1]], g[1], lv["BLL"] + CAP)
        m2_track(cell, lv["BLL"], c[0] - PAD, g[1] + PAD, vias=[g[1]])
        box(cell, "M2", c[0] - HALF, lv["BLL"], c[0] + HALF, lv["p"])
        m3_column(cell, c[0], M3_EDGE, top)
        # BLR: c4's drain and the stripe at g2 on c4's column.
        sd_contact(cell, p, c[4])
        landing(cell, p, c[4])
        gate_contact(cell, seam, [g[2]], g[2], lv["BLR"] + CAP)
        m2_track(cell, lv["BLR"], g[2] - PAD, c[4] + PAD, vias=[g[2]])
        m3_column(cell, c[4], M3_EDGE, top, vias=[lv["BLR"]])
        # D: the p drain at c2 down its own M3 column to the n via row, and
        # along it to the n drain at c3.  (A p via row track to c3 would end
        # 28 nm from c4's landing, tip to side: M2.S.5.)
        sd_contact(cell, p, c[2])
        landing(cell, p, c[2])
        sd_contact(cell, n, c[3])
        m2_track(cell, lv["n"], c[2] - PAD, c[3] + PAD, vias=[c[3]])
        # With a sample latch D and DP take D on to the latch's two drains.
        d_vias = [lv["n"], lv["D"], lv["DP"]] if spec.sample else [lv["n"]]
        m3_column(cell, c[2], lv["n"] - CAP, lv["p"] + CAP, vias=d_vias)
        # PRECHN's two stripes reach down to the shared track.
        for x in (g[0], g[3]):
            gate_contact(cell, seam, [x], x, lv["PRECHN"] - CAP)
        if spec.sample:
            _draw_sample_latch(cell, spec, k, ck_gates, ckb_gates)
        if spec.register:
            _draw_register_latch(cell, spec, k, ck_gates, ckb_gates)

    x_pre = spec.prechn_x
    m2_track(
        cell, lv["PRECHN"], x_pre - PAD, max(prechn_gates) + PAD, vias=prechn_gates
    )
    m3_column(cell, x_pre, 0, height, vias=[lv["PRECHN"]])
    if spec.sample:
        m2_track(cell, lv["CK"], spec.column_x[4] - PAD, max(ck_gates) + PAD,
                 vias=ck_gates)  # fmt: skip
        m2_track(cell, lv["CKB"], spec.column_x[2] - PAD, max(ckb_gates) + PAD,
                 vias=ckb_gates)  # fmt: skip

    if draw_pin_labels:
        for pin, (metal, origin) in spec.pin_positions.items():
            label(cell, pin, metal, origin)
    box(cell, "BOUNDARY", 0, 0, width, height)
    return cell


def _pad(cell: Any, band: RowBand, x: int) -> None:
    """A drain's M1 pad and a standing via stack to M3."""
    y = sd_contact(cell, band, x)
    stack_to_m3(cell, x, y)


def _draw_sample_lead(cell: Any, spec: RomSenseRowSpec) -> tuple[list[int], list[int]]:
    """``SAMPLE``'s column and the row's two inverters; returns the gates of each track.

    ``SAMPLE`` is column 1's full-height M3, which stacked rows share, and its
    own short piece of the ``SAMPLE`` level takes it to the first inverter.
    That inverter's output is column 2, on the ``SAMPLEB`` track; the second's
    gate is on that track and its output is column 4, where the row's copy of
    ``SAMPLE`` starts.  Both share the supply at column 3.
    """
    n, p = spec.bands["n"], spec.bands["p"]
    lv = spec.levels
    seam = spec.stack.seam_y(0)
    cx = spec.column_x
    island(cell, p, cx[2:5])
    island(cell, n, cx[2:5])
    supply_contact(cell, p, cx[3])
    supply_contact(cell, n, cx[3])
    for x in (cx[2], cx[4]):
        _pad(cell, p, x)
        _pad(cell, n, x)
    g_in = cx[2] + GATE_PITCH // 2
    g_out = cx[3] + GATE_PITCH // 2
    gate_contact(cell, seam, [g_in], g_in, _reach(lv["CK"], seam))
    gate_contact(cell, seam, [g_out], g_out, _reach(lv["CKB"], seam))
    m3_column(cell, cx[2], lv["n"] - PAD, lv["p"] + PAD, vias=[lv["CKB"]])
    m3_column(cell, cx[4], lv["n"] - PAD, lv["p"] + PAD, vias=[lv["CK"]])
    m3_column(cell, spec.sample_x, 0, spec.height, vias=[lv["CK"]])
    m2_track(cell, lv["CK"], spec.sample_x - PAD, g_in + PAD, vias=[g_in])
    return [], [g_out]


def _reach(level: int, seam: int) -> int:
    """Where a gate bar from the seam ends to land a V1 on `level`."""
    return level - CAP if level < seam else level + CAP


def _draw_latch(
    cell: Any,
    spec: RomSenseRowSpec,
    c: list[int],
    levels: dict[str, int],
    gates: tuple[list[int], list[int]],
) -> None:
    """One TG latch on columns `c`, its roles on `levels`.

    ``D`` and ``DP`` take the input to the n drain at l2 and the p drain at
    l4 from wherever their tracks start; ``Q``'s track joins the p drain at
    l2 to the n drain at l4.  The stripe at l2-l3 goes to the ``CK`` track
    and the one at l3-l4 to ``CKB``: `gates` collects them.
    """
    n, p = spec.bands["n"], spec.bands["p"]
    lv = {**spec.levels, **levels}
    seam = spec.stack.seam_y(0)
    g = [x + GATE_PITCH // 2 for x in c[:6]]
    island(cell, p, c[0:5])
    island(cell, n, c[2:7])
    supply_contact(cell, p, c[1])
    supply_contact(cell, n, c[5])
    for x in (c[0], c[2], c[3], c[4]):
        _pad(cell, p, x)
    for x in (c[2], c[3], c[4], c[6]):
        _pad(cell, n, x)
    # The input's n drain on l2's lower M3, its p drain on l4's upper.
    m3_column(cell, c[2], lv["n"] - PAD, lv["D"] + CAP, vias=[lv["D"]])
    m3_column(cell, c[4], lv["DP"] - CAP, lv["p"] + PAD, vias=[lv["DP"]])
    # Q: the p drain on l2's M3, the n drain on l4's lower M3, one track.
    m3_column(cell, c[2], lv["Q"] - CAP, lv["p"] + PAD, vias=[lv["Q"]])
    m3_column(cell, c[4], lv["n"] - PAD, lv["Q"] + CAP, vias=[lv["Q"]])
    m2_track(cell, lv["Q"], c[2] - PAD, c[4] + PAD)
    # X: l3's M3 and the two inverter gates on one track.
    m3_column(cell, c[3], lv["n"] - PAD, lv["p"] + PAD, vias=[lv["X"]])
    for x in (g[0], g[5]):
        gate_contact(cell, seam, [x], x, _reach(lv["X"], seam))
    m2_track(cell, lv["X"], g[0] - PAD, g[5] + PAD, vias=[g[0], g[5]])
    # Y: l0's upper M3 and l6's lower, and the output inverter's gates.
    m3_column(cell, c[0], lv["Y"] - CAP, lv["p"] + PAD, vias=[lv["Y"]])
    m3_column(cell, c[6], lv["n"] - PAD, lv["Y"] + CAP, vias=[lv["Y"]])
    for x in (g[1], g[4]):
        gate_contact(cell, seam, [x], x, _reach(lv["Y"], seam))
    m2_track(cell, lv["Y"], c[0] - PAD, c[6] + PAD, vias=[g[1], g[4]])
    # The clocks.
    gate_contact(cell, seam, [g[2]], g[2], _reach(lv["CK"], seam))
    gate_contact(cell, seam, [g[3]], g[3], _reach(lv["CKB"], seam))
    gates[0].append(g[2])
    gates[1].append(g[3])


def _draw_sample_latch(
    cell: Any,
    spec: RomSenseRowSpec,
    k: int,
    ck_gates: list[int],
    ckb_gates: list[int],
) -> None:
    """Leaf `k`'s sample latch, with ``D`` arriving on its two tracks.

    With ``register`` its ``Y`` and ``X`` trade levels (see the module) and
    its ``Q`` track runs on to the register latch's jog.
    """
    lv = spec.levels
    c = [spec.latch_x(k, i) for i in range(SAMPLE_LATCH_COLUMNS)]
    swap = {"Y": lv["X"], "X": lv["Y"]} if spec.register else {}
    _draw_latch(cell, spec, c, swap, (ck_gates, ckb_gates))
    # D from the leaf's c2 to l2 and, as DP, to l4.
    m2_track(cell, lv["D"], spec.leaf_x(k, 2) - PAD, c[2] + PAD)
    m2_track(cell, lv["DP"], spec.leaf_x(k, 2) - PAD, c[4] + PAD)
    if spec.register:
        m2_track(cell, lv["Q"], c[4] - PAD, spec.register_x(k, 0) + PAD)


def _draw_register_latch(
    cell: Any,
    spec: RomSenseRowSpec,
    k: int,
    ck_gates: list[int],
    ckb_gates: list[int],
) -> None:
    """Leaf `k`'s register latch: the sample latch with its clocks swapped.

    ``LQ`` arrives on the ``Q`` track at l0, where a jog under l0's own ``Y``
    column takes it down to ``D`` and up to ``DP``.  Its own ``Q`` keeps the
    ``Q`` track, from l2 on.
    """
    lv = spec.levels
    c = [spec.register_x(k, i) for i in range(SAMPLE_LATCH_COLUMNS)]
    clocks = {"CK": lv["CKB"], "CKB": lv["CK"]}
    _draw_latch(cell, spec, c, clocks, (ckb_gates, ck_gates))
    m3_column(
        cell, c[0], lv["D"] - CAP, lv["DP"] + CAP, vias=[lv["D"], lv["Q"], lv["DP"]]
    )
    m2_track(cell, lv["D"], c[0] - PAD, c[2] + PAD)
    m2_track(cell, lv["DP"], c[0] - PAD, c[4] + PAD)


def main(argv: list[str] | None = None) -> None:
    """Write a read-leaf row GDS: ``asap7-rom-sense --leaves 4``."""
    args = parse_spec(
        RomSenseRowSpec,
        argv,
        description="Generate a row of ASAP7 NOR ROM read leaves: prechargers and NAND2 sensors.",
    )
    spec = args.spec
    cell, out = write_gds(lambda lib: build_rom_sense_row(spec, lib=lib), args.out)
    print(f"✓ {cell.name}: {spec.width} x {spec.height} nm -> {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
