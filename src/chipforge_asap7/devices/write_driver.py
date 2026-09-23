"""Parametric write driver: a data latch driving the sense lines through transmission gates.

The released bank's ``write_driver_sram`` is twelve transistors: an inverter
makes ``DN`` from ``D``; two nFET pass gates, open while write enable is low,
load ``(D, DN)`` into a cross-coupled latch with 1-fin pFET keepers and full
nFET pull-downs; and two transmission gates, open while write enable is high,
put the latch on ``SA``/``SAN``, which the column mux takes to the selected
bitline pair.  The latch's own pull-downs are the write drivers.

`WriteDriverSpec` draws that on two dense rows of the same ``(n, p)`` row the
bitline leaf uses, so the two abut.  The bottom row holds everything gated by
write enable and the inverter; the row above, sharing the VDD rail, holds the
latch, whose weak keepers are their own one-fin island::

    row 1     .    .    .    W    S    WN   .    .    .    .        latch
    row 0     D    W    SA   W'   .    WN'  SAN  WN   DN   S        pass, TG, inverter
    stripe     WRENAN WRENA WRENAN  .    .  WRENAN WRENA WRENAN  D
    band        n     n     p                  p     n     n     n+p

``W``/``WN`` climb from the bottom row to the latch on M3 at the columns their
transmission-gate pFETs sit on.  ``SA`` and ``SAN`` are full-height M3 tracks,
as in the leaf, so a column IO can run them straight through; ``D``,
``WRENA`` and ``WRENAN`` climb from their ties to the top edge on M3, where
a router reaches them clear of anything a block draws below.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from ..layout.grid import GATE_PITCH
from ..layout.layers import box, require_gdspy
from ..layout.rules import TRACK_PITCH
from .finfet import SELECT_X_ENC
from .row import RowBand, RowStack
from .rowcell import (
    CAP,
    HALF,
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
    square,
    supply_contact,
    via_y,
)

__all__ = ["WRITE_DRIVER_PINS", "WriteDriverSpec", "build_write_driver"]

WRITE_DRIVER_PINS = ("D", "WRENA", "WRENAN", "SA", "SAN", "VDD", "VSS")

_COLUMNS = 10  # diffusion columns; one dummy pitch each side
# The bottom row's M2 tie tracks, from its seam: one just below it, two above.
_TIE_OFFSETS = (-19, 17, 53)


@dataclass(frozen=True)
class WriteDriverSpec:
    """A data latch and its transmission gates to the sense lines.

    Args:
        n_fins: fins of every nFET: pass gates, transmission gates, the
            inverter and the latch pull-downs.
        p_fins: fins of the transmission-gate and inverter pFETs.
        keeper_fins: fins of the latch's two pFET keepers.  They have to lose
            to a pass nFET writing a zero, so they are weak: one fin, as in
            the released cell, against three.
        band_height: ``(n_band, p_band)`` of one row, in nm.  The default is
            the bitline leaf's row.
        vt: threshold flavor of every device.
    """

    n_fins: int = 3
    p_fins: int = 3
    keeper_fins: int = 1
    band_height: tuple[int, int] = (135, 162)
    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"

    def __post_init__(self) -> None:
        object.__setattr__(self, "band_height", tuple(self.band_height))
        if self.keeper_fins > self.p_fins:
            raise ValueError(
                f"keeper_fins ({self.keeper_fins}) cannot exceed p_fins ({self.p_fins})"
            )
        # RowStack and FinFETSpec reject what cannot be drawn.
        bands = self.bands
        n_lo, p_lo = bands["n0"], bands["p0"]
        # A gate contact's M1 bar runs from the seam to its tie; on the stripe
        # beside a column it is 9 nm from that column's pad, so the two have
        # to clear each other by 18 nm in y (M1.S.6, corner to corner).  The
        # lowest tie is beside pads that stop on the via row, the next beside
        # pads that reach the 80 nm track, and the highest beside p pads.
        y_n, y_p = via_y(n_lo), via_y(p_lo)
        low, mid, high = self.tie_levels
        if (
            low - CAP < y_n + CAP + 18
            or low < self.n_level + TRACK_PITCH
            or mid - CAP < self.n_level + CAP + 18
        ):
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
        """``n0``, ``p0`` (bottom row), ``p1``, ``n1`` (top row) and ``k1``, the keepers' band."""
        n0, p0, p1, n1 = self.stack.bands()
        keepers = replace(p1, spec=replace(p1.spec, fins=self.keeper_fins))
        return {"n0": n0, "p0": p0, "p1": p1, "n1": n1, "k1": keepers}

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
            f"wrdrv_{self.n_fins}n{self.p_fins}p_k{self.keeper_fins}"
            f"_h{self.band_height[0]}x{self.band_height[1]}{vt_tag}"
        )

    # ── Columns ───────────────────────────────────────────────────────────────
    @property
    def width(self) -> int:
        return (_COLUMNS + 2) * GATE_PITCH

    @property
    def column_x(self) -> list[int]:
        """X of the ten diffusion columns, ``c0`` to ``c9``."""
        return [(i + 1) * GATE_PITCH for i in range(_COLUMNS)]

    def gate_x(self, left: int) -> int:
        """X of the gate stripe between columns `left` and `left + 1`."""
        return self.column_x[left] + GATE_PITCH // 2

    @property
    def n_level(self) -> int:
        """The bottom row's M2 track its n pads reach, a pitch above their via row's landings."""
        return via_y(self.bands["n0"]) + HALF + 18 + HALF + 3

    @property
    def tie_levels(self) -> tuple[int, int, int]:
        """The bottom row's three M2 tie tracks: D, WRENA, WRENAN.

        A gate contact's M1 bar runs from the seam to its tie, 9 nm beside the
        pads of the neighbouring columns, so the two have to clear each other
        by 18 nm in y (M1.S.6).  The lowest tie is beside pads that stop on
        the via row, the middle one beside pads reaching the 80 nm track, and
        the highest beside p pads.
        """
        seam = self.stack.seam_y(0)
        low, mid, high = (seam + offset for offset in _TIE_OFFSETS)
        return low, mid, high

    @property
    def track_x(self) -> dict[str, int]:
        """X of the M3 columns that are pins."""
        x = self.column_x
        # The controls leave at the top edge, on columns that are free above
        # their ties and far enough apart to be reached one by one.
        return {"D": x[0], "WRENAN": x[7], "SA": x[2], "WRENA": x[4], "SAN": x[6]}

    @property
    def pin_positions(self) -> dict[str, tuple[str, tuple[float, float]]]:
        tx = self.track_x
        return {
            "D": ("M3", (tx["D"], self.height - 40)),
            "WRENAN": ("M3", (tx["WRENAN"], self.height - 40)),
            "SA": ("M3", (tx["SA"], self.height / 2)),
            "WRENA": ("M3", (tx["WRENA"], self.height - 40)),
            "SAN": ("M3", (tx["SAN"], self.height / 2)),
            "VDD": ("M1", (self.width / 2, self.rails[1][0])),
            "VSS": ("M1", (self.width / 2, self.rails[0][0])),
        }

    # ── Netlist ───────────────────────────────────────────────────────────────
    @property
    def devices(self) -> tuple[tuple[str, str, str, str, str, int], ...]:
        """``(name, drain, gate, source, flavor, fins)`` for the twelve transistors."""
        n, p, k = self.n_fins, self.p_fins, self.keeper_fins
        return (
            ("MNI", "DN", "D", "VSS", "n", n),
            ("MPI", "DN", "D", "VDD", "p", p),
            ("MNPD", "W", "WRENAN", "D", "n", n),
            ("MNPDN", "WN", "WRENAN", "DN", "n", n),
            ("MNW", "W", "WN", "VSS", "n", n),
            ("MNWN", "WN", "W", "VSS", "n", n),
            ("MPW", "W", "WN", "VDD", "p", k),
            ("MPWN", "WN", "W", "VDD", "p", k),
            ("MNTW", "SA", "WRENA", "W", "n", n),
            ("MPTW", "SA", "WRENAN", "W", "p", p),
            ("MNTWN", "SAN", "WRENA", "WN", "n", n),
            ("MPTWN", "SAN", "WRENAN", "WN", "p", p),
        )

    def netlist(self, name: str | None = None) -> str:
        """A BSIM-CMG subcircuit sized by ``nfin``."""
        title = name or self.cell_name
        bands = self.bands
        lines = [
            "* ASAP7 write driver: data latch and transmission gates to the sense lines",
            f".SUBCKT {title} {' '.join(WRITE_DRIVER_PINS)}",
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
def build_write_driver(
    spec: WriteDriverSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Draw `spec` and return the gdspy Cell (nanometre coordinates)."""
    spec = spec or WriteDriverSpec()
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = (
        lib.new_cell(cell_name)
        if lib is not None
        else gdspy.Cell(cell_name, exclude_from_current=True)
    )
    bands = spec.bands
    n0, p0, n1, k1 = (bands[k] for k in ("n0", "p0", "n1", "k1"))
    width, height = spec.width, spec.height
    c, g = spec.column_x, spec.gate_x
    seam0, seam1 = spec.stack.seam_ys
    y_n0, y_p0, y_n1 = via_y(n0), via_y(p0), via_y(n1)
    lvl = spec.n_level  # 80: the bottom row's n-side track
    tie_d, tie_a, tie_an = spec.tie_levels
    # The latch's ties mirror the upper two about its own seam.
    tie_w, tie_wn = seam1 - _TIE_OFFSETS[1], seam1 - _TIE_OFFSETS[2]

    draw_frame(cell, spec.stack, width, spec.vt)

    # Bottom row.  Two n islands: D -WRENAN- W -WRENA- SA, and
    # SAN -WRENA- WN -WRENAN- DN -D- VSS.  Three p islands: the transmission
    # gate pFETs under their own WRENAN stripes, and the inverter.
    island(cell, n0, [c[0], c[1], c[2]])
    island(cell, n0, [c[6], c[7], c[8], c[9]])
    island(cell, p0, [c[2], c[3]])
    island(cell, p0, [c[5], c[6]])
    island(cell, p0, [c[8], c[9]])
    # Top row: the latch, W -WN- S -W- WN, in both bands; the keepers narrow.
    island(cell, n1, [c[3], c[4], c[5]])
    island(cell, k1, [c[3], c[4], c[5]])
    assert c[0] - ISLAND_OVERHANG >= SELECT_X_ENC

    draw_rails(cell, spec.stack, width)
    supply_contact(cell, n0, c[9])
    supply_contact(cell, p0, c[9])
    supply_contact(cell, n1, c[4])
    supply_contact(cell, k1, c[4])

    # Pins on M3: SA and SAN the full height so a column can run them through;
    # D, WRENA and WRENAN from their ties up to the top edge.
    tx = spec.track_x
    m3_column(cell, tx["SA"], 0, height)
    m3_column(cell, tx["SAN"], 0, height)
    for column, net in ((2, "SA"), (6, "SAN")):
        sd_contact(cell, n0, c[column])
        landing(cell, n0, c[column])
        sd_contact(cell, p0, c[column])
        landing(cell, p0, c[column])

    # D: the pass gate's source, and the inverter's gates on the far stripe.
    sd_contact(cell, n0, c[0])
    landing(cell, n0, c[0])
    m3_column(
        cell, tx["D"], y_n0 - CAP, height, vias=[tie_d]
    )  # over its pad's landing too
    gate_contact(cell, seam0, [g(8)], g(8), tie_d - CAP)
    m2_track(cell, tie_d, c[0] - PAD, g(8) + PAD, vias=[g(8)])
    # DN: the inverter's drains, joined on M3 within the column.
    sd_contact(cell, n0, c[8])
    landing(cell, n0, c[8])
    sd_contact(cell, p0, c[8])
    landing(cell, p0, c[8])
    m3_column(cell, c[8], y_n0 - CAP, y_p0 + CAP)

    # WRENA on the two transmission-gate nFET stripes; WRENAN on the two pass
    # gates and the two transmission-gate pFETs.  Each is one M2 tie with a
    # V2 to its M3 stub.
    for stripes, tie, net in (
        ((1, 6), tie_a, "WRENA"),
        ((0, 2, 5, 7), tie_an, "WRENAN"),
    ):
        for s in stripes:
            gate_contact(cell, seam0, [g(s)], g(s), tie + CAP)
        m2_track(
            cell,
            tie,
            g(stripes[0]) - PAD,
            g(stripes[-1]) + PAD,
            vias=[g(s) for s in stripes],
        )
        m3_column(cell, tx[net], tie - CAP, height, vias=[tie])

    # W and WN.  Each starts on its pass gate's drain in the n band, runs along
    # the 80 nm track to the column of its transmission-gate pFET, and climbs
    # that column's M3 through the pFET's contact to the latch: its keeper
    # drain, its pull-down drain, and the tie of the other half's gates.
    for pad, column, tie, stripe in ((1, 3, tie_w, 4), (7, 5, tie_wn, 3)):
        sd_contact(cell, n0, c[pad], reach=lvl + CAP)
        m2_track(
            cell,
            lvl,
            min(c[pad], c[column]) - PAD,
            max(c[pad], c[column]) + PAD,
            vias=[c[pad]],
        )
        square(cell, "V2", c[column], lvl)
        sd_contact(cell, p0, c[column])
        landing(cell, p0, c[column])
        sd_contact(cell, k1, c[column])
        landing(cell, k1, c[column])
        sd_contact(cell, n1, c[column])
        landing(cell, n1, c[column])
        m3_column(cell, c[column], lvl - CAP, y_n1 + CAP, vias=[tie])
        gate_contact(cell, seam1, [g(stripe)], g(stripe), tie - CAP)
        m2_track(
            cell,
            tie,
            min(c[column], g(stripe)) - PAD,
            max(c[column], g(stripe)) + PAD,
            vias=[g(stripe)],
        )

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
    """Write a write-driver GDS: ``python -m chipforge_asap7.devices.write_driver``."""
    parser = argparse.ArgumentParser(
        description="Generate an ASAP7 write driver: data latch and transmission gates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-fins", type=int, default=3)
    parser.add_argument("--p-fins", type=int, default=3)
    parser.add_argument("--keeper-fins", type=int, default=1)
    parser.add_argument(
        "--band-height", default="135,162", help="n,p band heights in nm."
    )
    parser.add_argument("--vt", choices=("rvt", "lvt", "slvt", "sram"), default="rvt")
    parser.add_argument("--out", type=Path, default=None, help="Output GDS path.")
    args = parser.parse_args(argv)
    spec = WriteDriverSpec(
        n_fins=args.n_fins, p_fins=args.p_fins, keeper_fins=args.keeper_fins,
        band_height=tuple(int(v) for v in args.band_height.split(",")), vt=args.vt,
    )  # fmt: skip
    library = require_gdspy().GdsLibrary(unit=1e-9, precision=1e-10)
    require_gdspy().current_library = library
    cell = build_write_driver(spec, lib=library)
    output = args.out or Path(f"{cell.name}.gds")
    output.parent.mkdir(parents=True, exist_ok=True)
    library.write_gds(str(output))
    print(f"✓ {cell.name}: {spec.width} x {spec.height} nm -> {output}")


if __name__ == "__main__":
    main()
