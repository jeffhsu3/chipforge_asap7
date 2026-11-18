"""The NOR ROM's address latches: one dense row of static D latches on PRECHN.

A latch is transparent while ``PRECHN`` is low, which is the precharge, and
holds while it is high, through the evaluation: the address the decoder
sees cannot move under a firing wordline.  Each is a transmission gate in,
two inverters, and a transmission gate back from the output::

    D -[TG in: n CK, p CKB]- X -INV- Y -INV- Q
                             └──[TG back: n CKB, p CK]── Q

with ``CKB = PRECHN`` and ``CK`` its complement, made once at the row's
start.  On one dense row (`chipforge_asap7.devices.rowcell`), eight columns
per latch::

    p band   Y -X- VDD -Y- Q -CK- X -CKB- D
    n band               D -CK- X -CKB- Q -Y- VSS -X- Y
    column   0     1     2      3       4     5      6     7 (D's pin)

The two transmission gates share ``X``, so ``D`` and ``Q`` each sit on
different columns in the two bands.  ``Q`` crosses on M2 between the M3
columns at 2 and 4; ``D`` reaches its pin column at 7 on two more tracks.
``X`` is column 3's M3, and ``Y`` joins columns 0 and 6 on one track.  ``CK``
and ``PRECHN`` run the whole row on their own tracks, seven M2 levels in all
between the via rows, which is what sets the default 486 nm row of 4-fin
bands.  ``Q`` leaves upward on column 2's M3, which runs to the row's top.
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
    ISLAND_OVERHANG,
    PAD,
    draw_frame,
    draw_rails,
    gate_contact,
    island,
    label,
    m2_track,
    m3_column,
    sd_contact,
    stack_to_m3,
    supply_contact,
    via_y,
)

__all__ = ["LATCH_COLUMNS", "RomAddressLatchRowSpec", "build_rom_address_latch_row"]

#: Columns per latch: five p, five n (offset by two), and D's pin column.
LATCH_COLUMNS = 8
#: Columns before the first latch: PRECHN's M3, then CK's inverter, then a gap.
LEAD_COLUMNS = 4
#: A standing landing's end faces an M2 track's side across 25 nm (M2.S.2),
#: 51 nm centre to centre from its via.
LANDING_CLEAR = 51
M3_EDGE = 17


@dataclass(frozen=True)
class RomAddressLatchRowSpec:
    """One row of address latches and the inverter that makes their ``CK``.

    Args:
        latches: latches on the row, one per address bit.
        n_fins: fins of every nFET.
        p_fins: fins of every pFET.
        band_height: ``(n_band, p_band)`` in nm.
        vt: threshold flavor of every device.
    """

    latches: int = 1
    n_fins: int = 4
    p_fins: int = 4
    band_height: tuple[int, int] = (243, 243)
    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"

    def __post_init__(self) -> None:
        object.__setattr__(self, "band_height", tuple(self.band_height))
        if isinstance(self.latches, bool) or not isinstance(self.latches, int):
            raise TypeError(f"latches must be an integer, got {self.latches!r}")
        if self.latches < 1:
            raise ValueError(f"latches must be >= 1, got {self.latches}")
        lv = self.levels
        # The gate bars on the top level clear the p pads by 18 nm (M1.S.6).
        if lv["CKB"] > lv["p"] - LANDING_CLEAR:
            raise ValueError(
                f"bands of {self.band_height} nm leave {lv['p'] - lv['n']} nm between "
                f"the via rows; the seven M2 levels need "
                f"{2 * LANDING_CLEAR + 6 * 36}"
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
            f"rom_alatch_x{self.latches}_{self.n_fins}n{self.p_fins}p"
            f"_h{self.band_height[0]}x{self.band_height[1]}{vt_tag}"
        )

    @property
    def levels(self) -> dict[str, int]:
        """Y of each M2 track, bottom to top, and of the two via rows."""
        y_n, y_p = via_y(self.bands["n"]), via_y(self.bands["p"])
        names = ("D", "CK", "Q", "X", "DP", "Y", "CKB")
        first = y_n + LANDING_CLEAR
        return {"n": y_n, **{k: first + 36 * i for i, k in enumerate(names)}, "p": y_p}

    # ── Columns ───────────────────────────────────────────────────────────────
    @property
    def columns(self) -> int:
        return LEAD_COLUMNS + LATCH_COLUMNS * self.latches

    @property
    def width(self) -> int:
        return (self.columns + 1) * GATE_PITCH

    @property
    def column_x(self) -> list[int]:
        return [(i + 1) * GATE_PITCH for i in range(self.columns)]

    def latch_x(self, latch: int, column: int) -> int:
        return self.column_x[LEAD_COLUMNS + LATCH_COLUMNS * latch + column]

    def gate_x(self, latch: int, gate: int) -> int:
        return self.latch_x(latch, gate) + GATE_PITCH // 2

    @property
    def prechn_x(self) -> int:
        return self.column_x[0]

    def q_x(self, latch: int) -> int:
        return self.latch_x(latch, 2)

    def d_x(self, latch: int) -> int:
        return self.latch_x(latch, 7)

    @property
    def pin_positions(self) -> dict[str, tuple[str, tuple[float, float]]]:
        lv = self.levels
        pins: dict[str, tuple[str, tuple[float, float]]] = {}
        for k in range(self.latches):
            pins[f"D<{k}>"] = ("M3", (self.d_x(k), lv["X"]))
            pins[f"Q<{k}>"] = ("M3", (self.q_x(k), self.height - M3_EDGE - 9))
        pins["PRECHN"] = ("M3", (self.prechn_x, lv["X"]))
        pins["VDD"] = ("M1", (self.width / 2, self.rails[1][0]))
        pins["VSS"] = ("M1", (self.width / 2, self.rails[0][0]))
        return pins

    @property
    def pins(self) -> tuple[str, ...]:
        return tuple(self.pin_positions)

    # ── Netlist ───────────────────────────────────────────────────────────────
    @property
    def devices(self) -> tuple[tuple[str, str, str, str, str, int], ...]:
        """``(name, drain, gate, source, flavor, fins)`` for every transistor."""
        n, p = self.n_fins, self.p_fins
        devices = [
            ("MPC", "CK", "PRECHN", "VDD", "p", p),
            ("MNC", "CK", "PRECHN", "VSS", "n", n),
        ]
        for k in range(self.latches):
            d, q, x, y = f"D<{k}>", f"Q<{k}>", f"X{k}", f"Y{k}"
            devices += [
                (f"MPI{k}", y, x, "VDD", "p", p),
                (f"MPO{k}", q, y, "VDD", "p", p),
                (f"MPF{k}", q, "CK", x, "p", p),
                (f"MPT{k}", x, "PRECHN", d, "p", p),
                (f"MNT{k}", d, "CK", x, "n", n),
                (f"MNF{k}", x, "PRECHN", q, "n", n),
                (f"MNO{k}", q, y, "VSS", "n", n),
                (f"MNI{k}", y, x, "VSS", "n", n),
            ]
        return tuple(devices)

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
                lines.append(
                    f"{device}_f{fin} {drain} {gate} {source} {bulk} {model.model} "
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
            name or self.cell_name, "* ASAP7 NOR ROM address latches", unit_fins=False
        )

    def lvs_reference(self, name: str | None = None) -> str:
        """The LVS reference: one MOS per FIN x GATE channel."""
        return self._subckt(
            name or self.cell_name,
            "* ASAP7 NOR ROM address latches LVS reference: one MOS per FIN x GATE channel",
            unit_fins=True,
        )


# ── Layout ────────────────────────────────────────────────────────────────────
def build_rom_address_latch_row(
    spec: RomAddressLatchRowSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Draw `spec` and return the gdspy Cell (nanometre coordinates)."""
    spec = spec or RomAddressLatchRowSpec()
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
    y_n, y_p = lv["n"], lv["p"]
    top = height - M3_EDGE

    draw_frame(cell, spec.stack, width, spec.vt)
    draw_rails(cell, spec.stack, width)
    cx = spec.column_x
    assert cx[1] - ISLAND_OVERHANG >= SELECT_X_ENC

    def pad(band: RowBand, x: int) -> None:
        """A drain's M1 pad and a standing via stack to M3."""
        y = sd_contact(cell, band, x)
        stack_to_m3(cell, x, y)

    # CK = NOT(PRECHN): supply at column 1, CK at column 2.
    island(cell, p, cx[1:3])
    island(cell, n, cx[1:3])
    supply_contact(cell, p, cx[1])
    supply_contact(cell, n, cx[1])
    pad(p, cx[2])
    pad(n, cx[2])
    g_inv = cx[1] + GATE_PITCH // 2
    gate_contact(cell, seam, [g_inv], g_inv, lv["CKB"] + CAP)
    m3_column(cell, cx[2], y_n - PAD, y_p + PAD, vias=[lv["CK"]])

    ck_gates, ckb_gates = [], [g_inv]
    for k in range(spec.latches):
        c = [spec.latch_x(k, i) for i in range(LATCH_COLUMNS)]
        g = [spec.gate_x(k, i) for i in range(6)]
        ck_gates.append(g[2])
        ckb_gates.append(g[3])
        island(cell, p, c[0:5], contacted=[c[0], c[1], c[2], c[3], c[4]])
        island(cell, n, c[2:7], contacted=[c[2], c[3], c[4], c[5], c[6]])
        supply_contact(cell, p, c[1])
        supply_contact(cell, n, c[5])
        for x in (c[0], c[2], c[3], c[4]):
            pad(p, x)
        for x in (c[2], c[3], c[4], c[6]):
            pad(n, x)
        # X: column 3, and the two inverter gates on one track.
        m3_column(cell, c[3], y_n - PAD, y_p + PAD, vias=[lv["X"]])
        for x in (g[0], g[5]):
            gate_contact(
                cell, seam, [x], x, lv["X"] - CAP if lv["X"] < seam else lv["X"] + CAP
            )
        m2_track(cell, lv["X"], g[0] - PAD, g[5] + PAD, vias=[g[0], g[5]])
        # Y: columns 0 and 6, and the output inverter's gates, on one track.
        m3_column(cell, c[0], lv["Y"] - CAP, y_p + PAD, vias=[lv["Y"]])
        m3_column(cell, c[6], y_n - PAD, lv["Y"] + CAP, vias=[lv["Y"]])
        for x in (g[1], g[4]):
            gate_contact(cell, seam, [x], x, lv["Y"] + CAP)
        m2_track(cell, lv["Y"], c[0] - PAD, c[6] + PAD, vias=[g[1], g[4]])
        # Q: p drain on column 2's upper M3, n drain on column 4's lower, one
        # crossing track; column 2 carries it on up to the row's top.
        m3_column(cell, c[2], lv["Q"] - CAP, top, vias=[lv["Q"]])
        m3_column(cell, c[4], y_n - PAD, lv["Q"] + CAP, vias=[lv["Q"]])
        m2_track(cell, lv["Q"], c[2] - PAD, c[4] + PAD)
        # D: n drain on column 2's lower M3, p drain on column 4's upper, each
        # on a track to the pin column at 7.
        m3_column(cell, c[2], y_n - PAD, lv["D"] + CAP, vias=[lv["D"]])
        m3_column(cell, c[4], lv["DP"] - CAP, y_p + PAD, vias=[lv["DP"]])
        m2_track(cell, lv["D"], c[2] - PAD, c[7] + PAD)
        m2_track(cell, lv["DP"], c[4] - PAD, c[7] + PAD)
        m3_column(cell, c[7], lv["D"] - CAP, lv["DP"] + CAP, vias=[lv["D"], lv["DP"]])
        # The clocks: CK's stripe below the seam, CKB's above.
        gate_contact(cell, seam, [g[2]], g[2], lv["CK"] - CAP)
        gate_contact(cell, seam, [g[3]], g[3], lv["CKB"] + CAP)

    m2_track(cell, lv["CK"], cx[2] - PAD, max(ck_gates) + PAD, vias=ck_gates)
    m2_track(cell, lv["CKB"], cx[0] - PAD, max(ckb_gates) + PAD, vias=ckb_gates)
    m3_column(cell, cx[0], 0, lv["CKB"] + CAP, vias=[lv["CKB"]])

    if draw_pin_labels:
        for pin, (metal, origin) in spec.pin_positions.items():
            label(cell, pin, metal, origin)
    box(cell, "BOUNDARY", 0, 0, width, height)
    return cell


def main(argv: list[str] | None = None) -> None:
    """Write an address-latch row GDS: ``asap7-rom-latch --latches 6``."""
    args = parse_spec(
        RomAddressLatchRowSpec,
        argv,
        description="Generate a row of ASAP7 NOR ROM address latches.",
    )
    spec = args.spec
    cell, out = write_gds(
        lambda lib: build_rom_address_latch_row(spec, lib=lib), args.out
    )
    print(f"✓ {cell.name}: {spec.width} x {spec.height} nm -> {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
