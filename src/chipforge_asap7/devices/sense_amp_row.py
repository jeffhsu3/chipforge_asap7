"""Parametric sense amplifier on the bitline leaf's row: the released latch, height matched to the IO.

`chipforge_asap7.devices.sense_amp` draws the released ``sense_amp_sram`` as
sixteen tiles on a grid of its own, 7020 x 999 nm, to be read against its
schematic.  This is the same amplifier drawn as an IO cell: two dense rows of
the leaf's ``(n, p)`` row, so it abuts the bitline leaf, the write driver and
the output latch.

The amplifier is a current-latched differential pair: ``SA``/``SAN`` gate the
input nFETs, whose common node is pulled down by the ``SAE`` tail, and whose
drains feed the sources of a cross-coupled nFET pair under a cross-coupled
pFET pair.  ``SAPRECHN`` precharges both outputs high and equalizes them
through a series pFET pair.  (The released cell also carries four off
transistors on the outputs, gated by their own supply; they are matching
dummies and are not drawn.)

The bottom row's n band is one interleaved chain::

    52  57  QAN  57  52  VSS  52  58  QA  58  52          columns (k = 2 fingers, t = 2 tail)
      SA  QA   QA  SA  SAE  SAE  SAN QAN  QAN SAN         stripes

with the input device and the cross-coupled device of each finger sharing
the uncontacted node between them, and every ``52`` column joined on the
80 nm track.  The cross-coupled pFETs need exactly the ``QA``/``QAN``
stripes, so they sit in the same row's p band over them, ``VDD QAN VDD`` and
``VDD QA VDD``, with no stripes of their own.  The row above holds the
precharge and the equalizer as one island, ``VDD QAN 59 QA VDD``, whose four
stripes are all ``SAPRECHN`` and share one pad.

``SA``, ``SAN`` and ``SAE`` are M3 stubs from the bottom edge, ``SAPRECHN``
from the top; ``QA`` and ``QAN`` leave on M3 at the top.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..layout.grid import GATE_PITCH
from ..layout.layers import box, require_gdspy
from .finfet import SELECT_X_ENC
from .row import RowBand, RowStack
from .rowcell import (
    CAP,
    HALF,
    ISLAND_OVERHANG,
    PAD,
    TRACK,
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
from .sense_amp import SENSE_AMP_PINS

__all__ = ["SenseAmpRowSpec", "build_sense_amp_row"]

# The bottom row's M2 tie tracks, from its seam: one just below it, two above.
_TIE_OFFSETS = (-19, 17, 53)


@dataclass(frozen=True)
class SenseAmpRowSpec:
    """The released sense amplifier on two rows of the leaf's row.

    Args:
        n_fins: fins of every nFET, per finger.
        p_fins: fins of every pFET, per finger.
        n_fingers: fingers of each input device and each cross-coupled device,
            n and p alike: the cross-coupled pFETs share the nFETs' stripes.
            An even number, so that each half of the chain starts and ends
            on the common node.
        tail_fingers: fingers of the ``SAE`` tail, an even number, so that the
            tail starts and ends on the common node.
        band_height: ``(n_band, p_band)`` of one row, in nm.  The default is
            the bitline leaf's row.
        vt: threshold flavor of every device.
    """

    n_fins: int = 3
    p_fins: int = 3
    n_fingers: int = 2
    tail_fingers: int = 2
    band_height: tuple[int, int] = (135, 162)
    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"

    def __post_init__(self) -> None:
        object.__setattr__(self, "band_height", tuple(self.band_height))
        for name in ("n_fingers", "tail_fingers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {value!r}")
        if self.n_fingers < 2 or self.n_fingers % 2:
            raise ValueError(
                f"n_fingers must be an even number >= 2, got {self.n_fingers}"
            )
        if self.tail_fingers < 2 or self.tail_fingers % 2:
            raise ValueError(
                f"tail_fingers must be an even number >= 2, got {self.tail_fingers}"
            )
        bands = self.bands
        n_lo, p_lo = bands["n0"], bands["p0"]
        y_n, y_p = via_y(n_lo), via_y(p_lo)
        low, mid, high = self.tie_levels
        # A gate contact's M1 bar runs from the seam to its tie, 9 nm beside the
        # pads of the neighbouring columns, so the two have to clear each other
        # by 18 nm in y (M1.S.6).  The lowest tie is beside pads that stop on
        # the via row, the middle one beside pads reaching the 80 nm track,
        # and the highest beside p pads.
        if (
            low - CAP < y_n + CAP + 18
            or low < self.n_level + TRACK
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
            f"sarow_{self.n_fins}n{self.p_fins}p_f{self.n_fingers}t{self.tail_fingers}"
            f"_h{self.band_height[0]}x{self.band_height[1]}{vt_tag}"
        )

    # ── Columns ───────────────────────────────────────────────────────────────
    @property
    def columns(self) -> int:
        """Diffusion columns of the n chain: two halves of ``2k + 1`` around a tail of ``t`` stripes."""
        return 4 * self.n_fingers + self.tail_fingers + 1

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
    def tail(self) -> tuple[int, int]:
        """First and last column of the tail: the two ``52`` columns the halves start from."""
        return 2 * self.n_fingers, 2 * self.n_fingers + self.tail_fingers

    @property
    def centre(self) -> int:
        """The column the row above is centred on: the middle of the tail."""
        return 2 * self.n_fingers + self.tail_fingers // 2

    def column_nets(self) -> dict[int, str]:
        """Net on every column of the n chain.

        From the tail outward, each half repeats ``52, 57, QAN, 57`` (``58``
        and ``QA`` on the right); the tail alternates ``VSS`` and ``52``.
        """
        k, first, last = self.n_fingers, *self.tail
        nets = {}
        for j in range(2 * k + 1):
            kind = j % 4
            nets[first - j] = (
                "52" if kind == 0 else "QAN" if kind == 2 else f"57_{j // 2}"
            )
            nets[last + j] = (
                "52" if kind == 0 else "QA" if kind == 2 else f"58_{j // 2}"
            )
        for m in range(1, self.tail_fingers):
            nets[first + m] = "VSS" if m % 2 else "52"
        return nets

    def stripe_nets(self) -> dict[int, str]:
        """Net on the gate stripe left of column ``i + 1``, keyed by ``i``."""
        k, first, last = self.n_fingers, *self.tail
        nets = {}
        for j in range(2 * k):
            gate = j % 4 in (1, 2)
            nets[first - j - 1] = "QA" if gate else "SA"
            nets[last + j] = "QAN" if gate else "SAN"
        for m in range(self.tail_fingers):
            nets[first + m] = "SAE"
        return nets

    def columns_of(self, net: str) -> list[int]:
        return sorted(i for i, n in self.column_nets().items() if n == net)

    def stripes_of(self, net: str) -> list[int]:
        return sorted(i for i, n in self.stripe_nets().items() if n == net)

    @property
    def n_level(self) -> int:
        """The bottom row's M2 track its n pads reach, a pitch above their via row's landings."""
        return via_y(self.bands["n0"]) + HALF + 18 + HALF + 3

    @property
    def tie_levels(self) -> tuple[int, int, int]:
        """The bottom row's three M2 tie tracks above the 80 nm one: QA, then SA/SAE/SAN, then QAN."""
        seam = self.stack.seam_y(0)
        low, mid, high = (seam + offset for offset in _TIE_OFFSETS)
        return low, mid, high

    @property
    def risers(self) -> dict[str, int]:
        """The ``QAN`` and ``QA`` columns nearest the tail: they climb to the row above and to the pins."""
        first, last = self.tail
        return {"QAN": first - 2, "QA": last + 2}

    @property
    def track_x(self) -> dict[str, int]:
        """X of the M3 columns that are pins."""
        x = self.column_x
        first, _ = self.tail
        # SA and SAN leave on the outermost column of their half, a 52 column.
        return {
            "SA": x[0],
            "SAN": x[self.columns - 1],
            "SAE": x[first + 1],
            "SAPRECHN": x[self.centre],
            "QAN": x[self.risers["QAN"]],
            "QA": x[self.risers["QA"]],
        }

    @property
    def pin_positions(self) -> dict[str, tuple[str, tuple[float, float]]]:
        tx = self.track_x
        return {
            "SA": ("M3", (tx["SA"], 40)),
            "SAN": ("M3", (tx["SAN"], 40)),
            "SAE": ("M3", (tx["SAE"], 40)),
            "SAPRECHN": ("M3", (tx["SAPRECHN"], self.height - 40)),
            "QA": ("M3", (tx["QA"], self.height - 40)),
            "QAN": ("M3", (tx["QAN"], self.height - 40)),
            "VDD": ("M1", (self.width / 2, self.rails[1][0])),
            "VSS": ("M1", (self.width / 2, self.rails[0][0])),
        }

    # ── Netlist ───────────────────────────────────────────────────────────────
    @property
    def series_nodes(self) -> frozenset[str]:
        """Uncontacted diffusion between two channels: one net per fin in the extraction."""
        return frozenset(
            {"N59"} | {f"N{n}_{i}" for n in (57, 58) for i in range(self.n_fingers)}
        )

    @property
    def devices(self) -> tuple[tuple[str, str, str, str, str, int], ...]:
        """``(name, drain, gate, source, flavor, fins)`` for every transistor."""
        n, p = self.n_fins, self.p_fins
        devices = []
        for i in range(self.n_fingers):
            devices += [
                (f"MN6_{i}", f"N57_{i}", "SA", "TAIL", "n", n),
                (f"MN8_{i}", "QAN", "QA", f"N57_{i}", "n", n),
                (f"MN7_{i}", f"N58_{i}", "SAN", "TAIL", "n", n),
                (f"MN9_{i}", "QA", "QAN", f"N58_{i}", "n", n),
                (f"MP4_{i}", "QAN", "QA", "VDD", "p", p),
                (f"MP5_{i}", "QA", "QAN", "VDD", "p", p),
            ]
        for m in range(self.tail_fingers):
            devices.append((f"MN10_{m}", "TAIL", "SAE", "VSS", "n", n))
        devices += [
            ("MP0", "QAN", "SAPRECHN", "VDD", "p", p),
            ("MP1", "QAN", "SAPRECHN", "N59", "p", p),
            ("MP2", "N59", "SAPRECHN", "QA", "p", p),
            ("MP3", "QA", "SAPRECHN", "VDD", "p", p),
        ]
        return tuple(devices)

    def netlist(self, name: str | None = None) -> str:
        """A BSIM-CMG subcircuit sized by ``nfin``."""
        title = name or self.cell_name
        bands = self.bands
        lines = [
            "* ASAP7 sense amplifier: current-latched differential pair, on the IO row",
            f".SUBCKT {title} {' '.join(SENSE_AMP_PINS)}",
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
def build_sense_amp_row(
    spec: SenseAmpRowSpec | None = None,
    *,
    name: str | None = None,
    lib: Any = None,
    draw_pin_labels: bool = True,
) -> Any:
    """Draw `spec` and return the gdspy Cell (nanometre coordinates)."""
    spec = spec or SenseAmpRowSpec()
    gdspy = require_gdspy()
    cell_name = name or spec.cell_name
    cell = (
        lib.new_cell(cell_name)
        if lib is not None
        else gdspy.Cell(cell_name, exclude_from_current=True)
    )
    bands = spec.bands
    n0, p0, p1 = (bands[k] for k in ("n0", "p0", "p1"))
    width, height = spec.width, spec.height
    c, g = spec.column_x, spec.gate_x
    seam0, seam1 = spec.stack.seam_ys
    y_n0, y_p0, y_p1 = via_y(n0), via_y(p0), via_y(p1)
    lvl = spec.n_level  # 80: the common node of the input pair
    tie_qa, tie_mid, tie_qan = spec.tie_levels
    tie_pre = seam1 - _TIE_OFFSETS[1]  # the row above, below its seam
    tx, risers = spec.track_x, spec.risers
    centre = spec.centre
    nets = spec.column_nets()
    contacted = [c[i] for i, net in nets.items() if net in ("52", "VSS", "QAN", "QA")]

    draw_frame(cell, spec.stack, width, spec.vt)

    # Bottom row: the whole n chain is one island; the cross-coupled pFETs are
    # a three-column island over each half's QA or QAN stripes, for every
    # finger.  Top row: precharge and equalizer, centred on the tail.
    island(cell, n0, c[0 : spec.columns], contacted=contacted)
    for out in ("QAN", "QA"):
        for column in spec.columns_of(out):
            island(cell, p0, [c[column - 1], c[column], c[column + 1]])
    top_columns = list(range(centre - 2, centre + 3))
    island(
        cell,
        p1,
        [c[i] for i in top_columns],
        contacted=[c[i] for i in top_columns if i != centre],
    )
    assert c[0] - ISLAND_OVERHANG >= SELECT_X_ENC

    draw_rails(cell, spec.stack, width)
    for column in spec.columns_of("VSS"):
        supply_contact(cell, n0, c[column])
    for out in ("QAN", "QA"):
        for column in spec.columns_of(out):
            supply_contact(cell, p0, c[column - 1])
            supply_contact(cell, p0, c[column + 1])
    supply_contact(cell, p1, c[centre - 2])
    supply_contact(cell, p1, c[centre + 2])

    # The common node: every 52 column's pad up to the 80 nm track.
    tail_columns = spec.columns_of("52")
    for column in tail_columns:
        sd_contact(cell, n0, c[column], reach=lvl + CAP)
    m2_track(
        cell,
        lvl,
        c[tail_columns[0]] - PAD,
        c[tail_columns[-1]] + PAD,
        vias=[c[i] for i in tail_columns],
    )

    # Inputs and the tail, tied on the middle track.  SA and SAN ride their
    # tie out to a stub on the outermost 52 column; SAE's stripes share one
    # pad, contacted on the tail's first VSS column, and its stub sits there.
    for net in ("SA", "SAN"):
        stripes = spec.stripes_of(net)
        for s in stripes:
            gate_contact(cell, seam0, [g(s)], g(s), tie_mid + CAP)
        reach = [tx[net], *(g(s) for s in stripes)]
        m2_track(
            cell,
            tie_mid,
            min(reach) - PAD,
            max(reach) + PAD,
            vias=[g(s) for s in stripes],
        )
        m3_column(cell, tx[net], 0, tie_mid + CAP, vias=[tie_mid])
    sae = spec.stripes_of("SAE")
    gate_contact(cell, seam0, [g(s) for s in sae], tx["SAE"], tie_mid + CAP)
    m2_track(cell, tie_mid, tx["SAE"] - PAD, tx["SAE"] + PAD, vias=[tx["SAE"]])
    m3_column(cell, tx["SAE"], 0, tie_mid + CAP, vias=[tie_mid])

    # Outputs.  Each QAN column joins its n and p drains on M3 and meets the
    # QAN tie, which runs across to the QAN stripes on the far half; the
    # column nearest the tail carries on into the row above, where a short
    # track on the p via row reaches the precharge island, and to the pin.
    # QA the same way, its tie below the seam and QAN's above.
    for out, tie, gate_net in (("QAN", tie_qan, "QAN"), ("QA", tie_qa, "QA")):
        stripes = spec.stripes_of(gate_net)
        columns = spec.columns_of(out)
        for s in stripes:
            gate_contact(
                cell, seam0, [g(s)], g(s), tie + (CAP if tie > seam0 else -CAP)
            )
        reach = [c[i] for i in columns] + [g(s) for s in stripes]
        m2_track(
            cell, tie, min(reach) - PAD, max(reach) + PAD, vias=[g(s) for s in stripes]
        )
        top_column = centre - 1 if out == "QAN" else centre + 1
        for column in columns:
            sd_contact(cell, n0, c[column])
            landing(cell, n0, c[column])
            sd_contact(cell, p0, c[column])
            landing(cell, p0, c[column])
            top = height if column == risers[out] else y_p0 + CAP
            m3_column(
                cell,
                c[column],
                y_n0 - CAP,
                top,
                vias=[tie] + ([y_p1] if column == risers[out] else []),
            )
        sd_contact(cell, p1, c[top_column])
        m2_track(cell, y_p1, min(c[top_column], tx[out]) - PAD, max(c[top_column], tx[out]) + PAD,
                 vias=[c[top_column]])  # fmt: skip

    # SAPRECHN: one pad over the four stripes, contacted on the equalizer's
    # uncontacted column, to a stub from the top edge.
    gate_contact(
        cell,
        seam1,
        [g(i) for i in range(centre - 2, centre + 2)],
        c[centre],
        tie_pre - CAP,
    )
    m2_track(cell, tie_pre, c[centre] - PAD, c[centre] + PAD, vias=[c[centre]])
    m3_column(cell, tx["SAPRECHN"], tie_pre - CAP, height, vias=[tie_pre])

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
    """Write a sense-amplifier GDS: ``python -m chipforge_asap7.devices.sense_amp_row``."""
    parser = argparse.ArgumentParser(
        description="Generate an ASAP7 sense amplifier on the IO row.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-fins", type=int, default=3)
    parser.add_argument("--p-fins", type=int, default=3)
    parser.add_argument(
        "--n-fingers",
        type=int,
        default=2,
        help="Input and cross-coupled device fingers.",
    )
    parser.add_argument(
        "--tail-fingers", type=int, default=2, help="SAE tail fingers, even."
    )
    parser.add_argument(
        "--band-height", default="135,162", help="n,p band heights in nm."
    )
    parser.add_argument("--vt", choices=("rvt", "lvt", "slvt", "sram"), default="rvt")
    parser.add_argument("--out", type=Path, default=None, help="Output GDS path.")
    args = parser.parse_args(argv)
    spec = SenseAmpRowSpec(
        n_fins=args.n_fins, p_fins=args.p_fins, n_fingers=args.n_fingers, tail_fingers=args.tail_fingers,
        band_height=tuple(int(v) for v in args.band_height.split(",")), vt=args.vt,
    )  # fmt: skip
    library = require_gdspy().GdsLibrary(unit=1e-9, precision=1e-10)
    require_gdspy().current_library = library
    cell = build_sense_amp_row(spec, lib=library)
    output = args.out or Path(f"{cell.name}.gds")
    output.parent.mkdir(parents=True, exist_ok=True)
    library.write_gds(str(output))
    print(f"✓ {cell.name}: {spec.width} x {spec.height} nm -> {output}")


if __name__ == "__main__":
    main()
