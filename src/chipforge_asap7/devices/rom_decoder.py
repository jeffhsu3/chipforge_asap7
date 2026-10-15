"""Transposed wordline decoder for the poly-wordline NOR-ROM.

The ROM's words are poly columns 81 nm apart, so its decoder sits *above* the
array and drives each wordline straight down an M3 track.  One AND gate --
`NandSpec` then `InverterSpec`, 216 + 162 nm on a 270 nm row -- is far wider
than a word, so the gates are spread over `interleave` rows: word ``w`` sits in
row ``w % interleave``, in a slot ``interleave * 81`` nm wide that holds its
NAND, its driver and a filler::

    pair 1  ── vss ── row 3 (flipped)  [NAND INV fill][NAND INV fill] ...
            ── vdd ── row 2            [NAND INV fill][NAND INV fill] ...
            ── vss ──                       M4 predecode rails over each pair
                      (54 nm gap)
    pair 0  ── vss ── row 1 (flipped)  [NAND INV fill][NAND INV fill] ...
            ── vdd ── row 0            [NAND INV fill][NAND INV fill] ...
            ── vss ──
              |  |  |  |  |  |  |  |   WL<w> on M3, down into the array

The interleave is a power of two, so a word's row is its low address bits.
The address splits into two predecode groups: ``PA`` takes the low
`split` bits, ``PB`` the rest, and ``WL<w> = PA<w mod 2^split> . PB<w >>
split>``.  Row ``r`` only ever sees the ``PA`` lines that are ``r`` modulo the
interleave, so a pair of rows needs its own two shares of ``PA`` and all of
``PB`` -- 8 rails for 64 words, 16 for 256 -- not every line.

Each slot routes what its cells leave open:

* NAND ``Y`` to the driver's input on M2, above the seam.
* The driver's output on M2 along its p-band contact row to a V2 on the word's
  wordline track, and M3 from there down to the array.
* ``A`` and ``B`` each on an M3 stub to a V3 on their M4 rail.  The stubs run
  on free tracks, 36 nm from every wordline track, so the wordlines of the
  rows above pass between them.

The two rows of a pair share their VDD rail and stand slot over slot: word
``w`` below, ``w + 1`` above.  That is the one alignment at which neither
row's rail contacts land on the other's source stubs.  The two words share
their PB line, and with it one B stub.  Pairs do not share a rail: the next
pair's slots are a tile further on, so they sit `PAIR_GAP` apart.  Each pair
is as tall as its rails need, and at least its two rows and the gap.  Each row
is padded with filler to one common span, then ends in ``filler tap filler``
and ``tap filler``, and an M3 per PB line beyond the rows' left ends joins the
pairs' copies of it.

The supply grid is an M3 VSS and VDD strap on the outer filler at each end,
with a via stack onto every rail of its net.  The VSS straps run on down to
an M2 bar along the array's top edge, which `build_rom` lands on every tile's
VSS column.  The predecoders that drive the rails are left to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

from ..layout.grid import FIN_PITCH, GATE_PITCH
from ..layout.layers import PIN_LAYERS, box, require_gdspy
from ..layout.rules import (
    CONTACT_SIZE,
    M2_V1_ENCLOSURE,
    M4_PITCH,
    M4_WIDTH,
    M4_X_GRID,
    TRACK_PITCH,
    V3_M4_ENCLOSURE,
)
from .cli import Option, parse_spec, write_gds
from .finfet import DEVICE_GATE_CUT_HEIGHT
from .inverter import InverterSpec, build_inverter
from .nand import NandSpec, build_nand
from .rom_bitcell import ROM_TILE_WORDS, STRAP_WL_XS, RomArraySpec, build_rom_array
from .row_support import RowSupportSpec, build_row_support

__all__ = ["RomDecoderSpec", "build_rom", "build_rom_decoder"]

HALF = CONTACT_SIZE // 2
PAD = HALF + M2_V1_ENCLOSURE  # 17: M2 past a V1 or V2, along the pad
CAP = HALF + 5  # 14: M3 past a V2 (V2.M3.EN.2)
_HALF_M4 = M4_WIDTH // 2
#: M3 past a V3, which is as tall as its 24 nm M4 (V3.M4.AUX.2, V3.M3.EN.1).
V3_CAP = _HALF_M4 + 5
#: M4 tracks are centred on 12 + 48k in absolute Y (M4.AUX.1, M4.AUX.2), so
#: the decoder is drawn where it will sit, on top of the array.
M4_TRACK_OFFSET = _HALF_M4
#: Shortest M2 pad: past 36 nm its ends face a neighbour as long edges.
MIN_PAD = 2 * TRACK_PITCH
#: The array's PSELECT frame takes the first 27 nm above it; at 27 the
#: decoder's bottom rail would also sit under the array's poly ends
#: (GCUT.GATE.S.2) and against its VSS column (M1.S.2).
MIN_GAP = 2 * FIN_PITCH
#: Rails keep this far inside their pair, so the M3 stubs that end on them
#: stay 31 nm from the next pair's (M3.S.4, end to end).
RAIL_MARGIN = V3_CAP + 16
#: An M3 stub at least this long has long edges (M3.S.1, 18 nm) and its area.
MIN_STUB = 2 * TRACK_PITCH
#: The wordline lands on the output pad at least this far right of the
#: driver's output via, so the two vias keep V2.S.1 apart and the pad is long.
OUT_TO_WORDLINE = 2 * TRACK_PITCH
#: Support cells at each row end: ``filler tap filler``, two CPP each.
END_CPP = 2
#: Space between pairs.  Their VSS rails are not shared: at the 162 nm shift
#: between one pair's slots and the next one's, a filler's rail contact would
#: land on a source stub of the other row (V0.M1.AUX.3).  Two fin pitches
#: keep the two rails 36 nm apart.
PAIR_GAP = 2 * FIN_PITCH


@dataclass(frozen=True)
class RomDecoderSpec:
    """The post-decode AND gates and wordline drivers for `array`.

    Args:
        array: the ROM array the decoder drives; its words must be a power of
            two, at least `interleave`.
        rows: ``(n_fins, p_fins)`` of both the NAND and the driver.  The two
            must sit on one row, so they share it; ``(3, 3)`` is the 270 nm
            standard-cell row.
        interleave: decoder rows, a power of two.  Each word gets a slot this
            many words wide, which must hold a NAND, a driver and a filler.
        split: address bits in the ``PA`` group, at least log2(`interleave`).
            ``None`` picks the split with the fewest rails per pair.
        gap: nm between the array's top edge and the decoder's bottom one, a
            fin-pitch multiple of at least `MIN_GAP`.
    """

    array: RomArraySpec = field(default_factory=RomArraySpec)
    rows: tuple[tuple[int, int], ...] = ((3, 3),)
    interleave: int = 8
    split: int | None = None
    gap: int = MIN_GAP

    def __post_init__(self) -> None:
        k = self.interleave
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(f"interleave must be an integer, got {k!r}")
        if k < 2 or k & (k - 1):
            raise ValueError(f"interleave must be a power of two >= 2, got {k}")
        words = self.array.words
        if words & (words - 1) or words < k:
            raise ValueError(
                f"the decoder needs a power-of-two word count >= interleave "
                f"{k}; got {words}"
            )
        if self.nand.stack.rails != self.inverter.stack.rails:
            raise ValueError(
                f"rows {self.rows} give the NAND and the driver different rails"
            )
        if self.filler_cpp < END_CPP:
            raise ValueError(
                f"a {self.slot_width} nm slot cannot hold a {self.nand.width} nm "
                f"NAND, a {self.inverter.width} nm driver and a filler; raise "
                "interleave"
            )
        lo = self.row_bits
        if self.split is not None and not lo <= self.split <= self.address_bits:
            raise ValueError(
                f"split must be {lo}..{self.address_bits} address bits, got {self.split}"
            )
        if self.gap < MIN_GAP or self.gap % FIN_PITCH:
            raise ValueError(
                f"gap must be a multiple of {FIN_PITCH} and at least {MIN_GAP}; "
                f"got {self.gap}"
            )
        for parity in (0, 1):
            self.stubs(parity)  # raises if the slot has no free tracks

    # ── Address ───────────────────────────────────────────────────────────────
    @property
    def address_bits(self) -> int:
        return self.array.words.bit_length() - 1

    @property
    def row_bits(self) -> int:
        return self.interleave.bit_length() - 1

    @property
    def pa_bits(self) -> int:
        """Address bits in the PA group: `split`, or the fewest rails."""
        if self.split is not None:
            return self.split
        return min(
            range(self.row_bits, self.address_bits + 1),
            key=lambda a: (self._pair_rails(a), a),
        )

    def _pair_rails(self, pa_bits: int) -> int:
        share = 2 ** (pa_bits - self.row_bits)
        return 2 * share + 2 ** (self.address_bits - pa_bits)

    @property
    def pair_rails(self) -> int:
        """M4 rails over each pair of rows: two shares of PA, all of PB."""
        return self._pair_rails(self.pa_bits)

    def lines(self, word: int) -> tuple[str, str]:
        """``(PA line, PB line)`` that select `word`."""
        self.array.wordline_x(word)  # range check
        mask = (1 << self.pa_bits) - 1
        return f"PA<{word & mask}>", f"PB<{word >> self.pa_bits}>"

    def rail_groups(self, pair: int) -> tuple[list[str], list[str], list[str]]:
        """``(lower row's PA, PB, upper row's PA)`` lines over `pair`."""
        k = self.interleave
        lower, upper = 2 * pair, 2 * pair + 1
        return (
            [f"PA<{i}>" for i in range(lower, 2**self.pa_bits, k)],
            [f"PB<{j}>" for j in range(2 ** (self.address_bits - self.pa_bits))],
            [f"PA<{i}>" for i in range(upper, 2**self.pa_bits, k)],
        )

    def rails(self, pair: int) -> list[str]:
        """Rails over `pair`, bottom to top."""
        return [line for group in self.rail_groups(pair) for line in group]

    # ── Cells ─────────────────────────────────────────────────────────────────
    @cached_property
    def nand(self) -> NandSpec:
        return NandSpec(rows=self.rows, fingers=2, vt=self.array.bitcell.vt)

    @cached_property
    def inverter(self) -> InverterSpec:
        return InverterSpec(rows=self.rows, fingers=2, vt=self.array.bitcell.vt)

    @property
    def row_height(self) -> int:
        return self.nand.height

    @property
    def slot_width(self) -> int:
        width = self.interleave * self.array.bitcell.word_pitch
        if width % GATE_PITCH:
            raise ValueError(f"a {width} nm slot is off the {GATE_PITCH} nm gate grid")
        return int(width)

    @property
    def filler_cpp(self) -> int:
        return (self.slot_width - self.nand.width - self.inverter.width) // GATE_PITCH

    @property
    def pairs(self) -> int:
        return self.interleave // 2

    @property
    def pair_height(self) -> int:
        """Two rows and `PAIR_GAP`, or the rails' height if more.

        The rails sit on the absolute M4 lattice, so a pair holds its rails
        wherever that lattice falls: one pitch more than they span.
        """
        rails = M4_PITCH * (self.pair_rails + 1) + 2 * RAIL_MARGIN
        rows = 2 * self.row_height + PAIR_GAP
        return max(rows, -(-rails // FIN_PITCH) * FIN_PITCH)

    @property
    def y0(self) -> int:
        """Bottom edge of the decoder, in array coordinates."""
        return self.array.height + self.gap

    @property
    def height(self) -> int:
        return self.pairs * self.pair_height

    def row_of(self, word: int) -> int:
        return word % self.interleave

    def row_y(self, row: int) -> int:
        """Bottom Y of `row`; odd rows are flipped, so their VSS rail is on top."""
        return self.y0 + (row // 2) * self.pair_height + (row % 2) * self.row_height

    def rail_y(self, pair: int, line: str) -> int:
        """Centre Y of `line`'s M4 rail over `pair`, on the absolute M4 lattice.

        The lower row's PA rails take the lowest tracks and the upper row's the
        highest, so each row's A stubs stay on its own side of the pair; PB,
        which both rows' shared B stubs reach, sits midway.
        """
        y_lo = self.y0 + pair * self.pair_height + RAIL_MARGIN
        y_hi = self.y0 + (pair + 1) * self.pair_height - RAIL_MARGIN
        first = y_lo + (M4_TRACK_OFFSET - y_lo) % M4_PITCH
        tracks = (y_hi - first) // M4_PITCH + 1
        lower, pb, upper = self.rail_groups(pair)
        if line in lower:
            index = lower.index(line)
        elif line in upper:
            index = tracks - len(upper) + upper.index(line)
        else:
            free = tracks - len(lower) - len(upper)
            index = len(lower) + (free - len(pb)) // 2 + pb.index(line)
        return first + index * M4_PITCH

    @property
    def slots_span(self) -> tuple[int, int]:
        """X span of every row's slots, once the shorter rows are padded.

        Each pair's slots stand two words, 3 CPP, right of the pair below's,
        so a row is padded with at most ``3 * (pairs - 1)`` CPP of filler on
        either side.
        """
        k, words = self.interleave, self.array.words
        lo = min(self.slot_x(r) for r in range(k))
        hi = max(self.slot_x(words - k + r) for r in range(k)) + self.slot_width
        return lo, hi

    @property
    def rows_span(self) -> tuple[int, int]:
        """X span of every row, with its end cells."""
        lo, hi = self.slots_span
        return lo - 3 * END_CPP * GATE_PITCH, hi + 2 * END_CPP * GATE_PITCH

    @property
    def strap_xs(self) -> dict[str, tuple[int, int]]:
        """X of the left and right M3 strap of each supply.

        They stand on the outer filler at each row end, which has no M2 of its
        own, a track apart from each other and from its edges.  VSS is the
        outer one.
        """
        lo, hi = self.rows_span
        return {
            "VSS": (lo + TRACK_PITCH, hi - TRACK_PITCH),
            "VDD": (lo + 2 * TRACK_PITCH, hi - 2 * TRACK_PITCH),
        }

    @property
    def supply_rails(self) -> list[tuple[int, str]]:
        """``(y, net)`` of every row's M1 rail, bottom to top, shared ones once."""
        rails = set()
        for row in range(self.interleave):
            for y, net in self.nand.rails:
                local = self.row_height - y if row % 2 else y
                rails.add((self.row_y(row) + local, net))
        return sorted(rails)

    # ── Slot geometry, in slot coordinates (row unflipped) ────────────────────
    @property
    def inverter_x(self) -> int:
        return self.nand.width

    @property
    def nand_y_x(self) -> int:
        x0, x1 = self.nand.output_bar_x(self.nand.columns("n", "Y")[0])
        return (x0 + x1) // 2

    @property
    def link_y(self) -> int:
        """NAND Y to driver input, on M2: high on the driver's input strap.

        That keeps it 27 nm above the A pad on the seam (M2.S.2: the pad's ends
        are short) and the input strap's own end 13 nm past the via.
        """
        return self.inverter.input_strap_y(0)[1] - 22

    @property
    def out_via(self) -> tuple[int, int]:
        """V1 on the driver's output, in the p-band contact row's arm.

        5 nm inside the strap's left edge, so the arm gives it the V1.M1.EN.1
        end-cap along the row.
        """
        x0, _ = self.inverter.output_strap_x(self.inverter.drain_xs[0])
        return int(self.inverter_x + x0 + 5 + HALF), int(
            self.inverter.bands[1].contact_y
        )

    def wordline_local(self, parity: int) -> int:
        """Slot-local X of the word's own M3 track, for an even or odd word.

        An even word's track is the first one `OUT_TO_WORDLINE` past the
        output via.  An odd word's is its neighbour's, one tile track further
        on: that lines the two rows of a pair up slot over slot, which is the
        only alignment where neither row's rail contacts land on the other's
        source stubs in the rail they share (V0.M1.AUX.3).
        """
        least = self.out_via[0] + OUT_TO_WORDLINE
        even = least + (STRAP_WL_XS[0] - least) % GATE_PITCH
        return even + STRAP_WL_XS[parity] - STRAP_WL_XS[0]

    def slot_x(self, word: int) -> int:
        """Left edge of `word`'s slot, on the gate grid."""
        return self.array.wordline_x(word) - self.wordline_local(word % 2)

    def wordline_tracks(self, parity: int) -> list[int]:
        """Slot-local X of every wordline track the slot sits under."""
        tile = self.array.bitcell.width
        shift = self.wordline_local(parity) - STRAP_WL_XS[parity]
        return [
            t * tile + x + shift
            for t in range(-2, self.interleave // ROM_TILE_WORDS + 2)
            for x in STRAP_WL_XS
        ]

    def stubs(self, parity: int) -> dict[str, int]:
        """Slot-local X of the A and B stubs: free tracks nearest their pins.

        The two words of a slot column, ``w`` below and ``w + 1`` above, take
        the same PB line, so they share one B stub, from one pin through the
        rail to the other.  Their PA lines differ, so the odd word's A stub
        keeps a track from the even word's.
        """
        taken = list(self.wordline_tracks(parity))
        if parity:
            even = self.stubs(0)
            taken.append(even["A"])
        free = [
            x
            for x in range(0, self.slot_width, HALF)
            if all(abs(x - t) >= TRACK_PITCH for t in taken)
        ]
        a_pin = self.nand.pad_gate_xs("A")[0]
        b_lo, b_hi = self._b_tie_x
        b_mid = (b_lo + b_hi) / 2
        # B lands on its own M2 tie; A needs a pad from its pin.  Take the pair
        # with the least M2, keeping A's V2 clear of the tie's.
        pairs = [
            (abs(x_a - a_pin) + abs(x_b - b_mid), x_a, x_b)
            for x_a in free
            for x_b in free
            if b_lo <= x_b <= b_hi and abs(x_a - x_b) >= TRACK_PITCH + PAD
        ]
        if not pairs:
            raise ValueError(
                f"no two free M3 tracks for the NAND's inputs in a {('even', 'odd')[parity]}-word slot"
            )
        if parity:
            pairs = [p for p in pairs if p[2] == even["B"]]
        if not pairs:
            raise ValueError(
                "no free M3 track for the NAND's A input beside the shared B stub"
            )
        _, x_a, x_b = min(pairs)
        return {"A": x_a, "B": x_b}

    @property
    def _b_tie_x(self) -> tuple[int, int]:
        xs = self.nand.pad_gate_xs("B")
        return xs[0], xs[-1]

    def pin_y(self, row: int, pin: Literal["A", "B"]) -> int:
        """Y of `pin`'s V2 in `row`, in decoder coordinates."""
        local = self.nand.seam_y if pin == "A" else self.nand.tie_ys["B"]
        y0 = self.row_y(row)
        return y0 + local if row % 2 == 0 else y0 + self.row_height - local

    @property
    def cell_name(self) -> str:
        return (
            f"rom_decoder_{self.array.cell_name.removeprefix('rom_array_')}"
            f"_k{self.interleave}_a{self.pa_bits}"
        )


def _new_cell(lib: Any, name: str) -> Any:
    gdspy = require_gdspy()
    if lib is None:
        return gdspy.Cell(name, exclude_from_current=True)
    return lib.new_cell(name)


def _shared(lib: Any, name: str, build: Any) -> Any:
    if lib is not None and name in lib.cells:
        return lib.cells[name]
    return build()


def _square(cell: Any, layer: str, x: float, y: float) -> None:
    box(cell, layer, x - HALF, y - HALF, x + HALF, y + HALF)


def _label(cell: Any, text: str, metal: str, origin: tuple[float, float]) -> None:
    layer, texttype = PIN_LAYERS[metal]
    cell.add(require_gdspy().Label(text, origin, layer=layer, texttype=texttype))


def _slot(spec: RomDecoderSpec, parity: int, lib: Any) -> Any:
    """One word's NAND, driver and filler, routed up to M2 and the V2s."""
    name = f"{spec.cell_name}_slot{parity}"

    def build() -> Any:
        gdspy = require_gdspy()
        nand = _shared(
            lib,
            spec.nand.cell_name,
            lambda: build_nand(spec.nand, lib=lib, draw_pin_labels=False),
        )
        inv = _shared(
            lib,
            spec.inverter.cell_name,
            lambda: build_inverter(spec.inverter, lib=lib, draw_pin_labels=False),
        )
        filler_spec = RowSupportSpec(
            stack=spec.nand.stack, kind="filler", width_cpp=spec.filler_cpp
        )
        filler = _shared(
            lib,
            filler_spec.cell_name,
            lambda: build_row_support(filler_spec, lib=lib, draw_pin_labels=False),
        )
        cell = _new_cell(lib, name)
        x_inv = spec.inverter_x
        cell.add(gdspy.CellReference(nand, (0, 0)))
        cell.add(gdspy.CellReference(inv, (x_inv, 0)))
        cell.add(gdspy.CellReference(filler, (x_inv + spec.inverter.width, 0)))

        # NAND Y -> driver input.
        y = spec.link_y
        x_y, x_in = spec.nand_y_x, x_inv + spec.inverter.input_gate_x
        for x in (x_y, x_in):
            _square(cell, "V1", x, y)
        box(cell, "M2", x_y - PAD, y - HALF, x_in + PAD, y + HALF)

        # Driver output -> the word's M3 track.
        (x_out, y_out), x_wl = spec.out_via, spec.wordline_local(parity)
        _square(cell, "V1", x_out, y_out)
        box(cell, "M2", x_out - PAD, y_out - HALF, x_wl + PAD, y_out + HALF)
        _square(cell, "V2", x_wl, y_out)

        # A: a V1 over the seam V0 of its pad, an M2 pad to its stub's V2.
        stubs = spec.stubs(parity)
        x_a, y_a = spec.nand.pad_gate_xs("A")[0], spec.nand.seam_y
        _square(cell, "V1", x_a, y_a)
        lo, hi = sorted((x_a, stubs["A"]))
        # Grow a short pad away from the link above it, toward the cell edge.
        box(
            cell,
            "M2",
            min(lo - PAD, hi + PAD - MIN_PAD),
            y_a - HALF,
            hi + PAD,
            y_a + HALF,
        )
        _square(cell, "V2", stubs["A"], y_a)
        # B: its V2 lands on the NAND's own M2 tie, which may need lengthening.
        y_b = spec.nand.tie_ys["B"]
        b_lo, b_hi = spec._b_tie_x
        box(
            cell,
            "M2",
            min(b_lo, stubs["B"] - PAD),
            y_b - HALF,
            max(b_hi, stubs["B"] + PAD),
            y_b + HALF,
        )
        _square(cell, "V2", stubs["B"], y_b)
        return cell

    return _shared(lib, name, build)


def _filler(spec: RomDecoderSpec, lib: Any, cpp: int) -> Any:
    support = RowSupportSpec(stack=spec.nand.stack, kind="filler", width_cpp=cpp)
    return _shared(
        lib,
        support.cell_name,
        lambda: build_row_support(support, lib=lib, draw_pin_labels=False),
    )


def _end(spec: RomDecoderSpec, lib: Any, kinds: tuple[str, ...]) -> list[Any]:
    cells = []
    for kind in kinds:
        support = RowSupportSpec(stack=spec.nand.stack, kind=kind, width_cpp=END_CPP)  # type: ignore[arg-type]
        cells.append(
            _shared(
                lib,
                support.cell_name,
                lambda support=support: build_row_support(
                    support, lib=lib, draw_pin_labels=False
                ),
            )
        )
    return cells


def build_rom_decoder(
    spec: RomDecoderSpec, *, name: str | None = None, lib: Any = None
) -> Any:
    """Draw the decoder in array coordinates and return the gdspy Cell.

    It is drawn where it sits, `spec.gap` above `spec.array`: each wordline's
    M3 lands on `RomArraySpec.wordline_x` and runs down across the gap to the
    array's top edge, where the array's own M3 wordline begins.
    """
    gdspy = require_gdspy()
    cell = _new_cell(lib, name or spec.cell_name)
    left = _end(spec, lib, ("filler", "tap", "filler"))
    right = _end(spec, lib, ("tap", "filler"))
    end_w = END_CPP * GATE_PITCH
    h = spec.row_height

    x_left, x_right = spec.slots_span
    for row in range(spec.interleave):
        y0 = spec.row_y(row)
        flip = row % 2 == 1
        y_ref = y0 + h if flip else y0
        words = range(row, spec.array.words, spec.interleave)
        xs = [spec.slot_x(w) for w in words]
        slot = _slot(spec, row % 2, lib)
        for x in xs:
            cell.add(gdspy.CellReference(slot, (x, y_ref), x_reflection=flip))
        # Pad each row out to the common span, so the straps find every row
        # at the same X.
        x_end = xs[-1] + spec.slot_width
        for x0, x1 in ((x_left, xs[0]), (x_end, x_right)):
            if x1 > x0:
                pad = _filler(spec, lib, (x1 - x0) // GATE_PITCH)
                cell.add(gdspy.CellReference(pad, (x0, y_ref), x_reflection=flip))
        for i, support in enumerate(left):
            x = x_left - (len(left) - i) * end_w
            cell.add(gdspy.CellReference(support, (x, y_ref), x_reflection=flip))
        for i, support in enumerate(right):
            x = x_right + i * end_w
            cell.add(gdspy.CellReference(support, (x, y_ref), x_reflection=flip))

        for w, x in zip(words, xs):
            # The wordline: from its V2 down to the decoder's bottom edge.
            y_out = spec.out_via[1]
            y_v2 = y0 + (h - y_out if flip else y_out)
            x_wl = spec.array.wordline_x(w)
            top = spec.array.height
            box(cell, "M3", x_wl - HALF, top, x_wl + HALF, y_v2 + CAP)
            # A and B: M3 stubs up or down to a V3 on their rail.
            stubs = spec.stubs(w % 2)
            for pin, line in zip(("A", "B"), spec.lines(w)):
                x_s = x + stubs[pin]
                y_pin = spec.pin_y(row, pin)
                y_rail = spec.rail_y(row // 2, line)
                lo = min(y_pin - CAP, y_rail - V3_CAP)
                hi = max(y_pin + CAP, y_rail + V3_CAP)
                if hi - lo < MIN_STUB:
                    grow = MIN_STUB - (hi - lo)
                    lo, hi = lo - grow // 2, hi + grow - grow // 2
                box(cell, "M3", x_s - HALF, lo, x_s + HALF, hi)
                box(
                    cell,
                    "V3",
                    x_s - HALF,
                    y_rail - _HALF_M4,
                    x_s + HALF,
                    y_rail + _HALF_M4,
                )

    # Between pairs the two VSS rails' gate cuts would face across 10 nm
    # (GCUT.S.3); one cut spans the gap instead.
    half_cut = DEVICE_GATE_CUT_HEIGHT // 2
    x0, x1 = spec.rows_span
    for pair in range(1, spec.pairs):
        below, above = 2 * pair - 1, 2 * pair
        y_below = spec.row_y(below) + spec.row_height
        box(cell, "GATE_CUT", x0, y_below - half_cut, x1, spec.row_y(above) + half_cut)

    # Rails: M4 across every slot of the pair, on M4's 24 nm X grid.  Every
    # pair has its own copy of each PB line; beyond the rows' left ends one M3
    # per line joins the copies, so PB<j> is one net and one pin.
    pb_lines = spec.rail_groups(0)[1]
    x_rows = spec.rows_span[0]
    joins = {
        line: x_rows - (len(pb_lines) - j) * TRACK_PITCH
        for j, line in enumerate(pb_lines)
    }
    x_min = min(joins.values()) - HALF - V3_M4_ENCLOSURE
    x_max = max(spec.slot_x(w) for w in range(spec.array.words)) + spec.slot_width
    x0 = (x_min // M4_X_GRID) * M4_X_GRID
    x1 = -(-x_max // M4_X_GRID) * M4_X_GRID
    for pair in range(spec.pairs):
        for line in spec.rails(pair):
            y = spec.rail_y(pair, line)
            box(cell, "M4", x0, y - _HALF_M4, x1, y + _HALF_M4)
            if line in joins:
                x = joins[line]
                box(cell, "V3", x - HALF, y - _HALF_M4, x + HALF, y + _HALF_M4)
            if line not in joins or pair == 0:
                _label(cell, line, "M4", (x0 + V3_M4_ENCLOSURE, y))
    for line, x in joins.items():
        ys = [spec.rail_y(pair, line) for pair in range(spec.pairs)]
        y_lo = min(ys) - V3_CAP
        y_hi = max(max(ys) + V3_CAP, y_lo + MIN_STUB)
        box(cell, "M3", x - HALF, y_lo, x + HALF, y_hi)
    for w in range(spec.array.words):
        x = spec.array.wordline_x(w)
        _label(cell, f"WL<{w}>", "M3", (x, spec.y0 + HALF))
    for y, net in spec.supply_rails:
        _label(cell, net, "M1", (spec.rows_span[0] + HALF, y))
    _supply_grid(spec, cell)
    return cell


def _supply_grid(spec: RomDecoderSpec, cell: Any) -> None:
    """M3 VDD and VSS straps at both row ends.

    Each strap takes a V1, an M2 pad and a V2 on every rail of its net.  The
    VSS straps run on down across the gap to the array's top edge, where
    `build_rom` ties them to the array.
    """
    rails = spec.supply_rails
    top = spec.array.height
    for net, xs in spec.strap_xs.items():
        ys = [y for y, n in rails if n == net]
        y_lo = top if net == "VSS" else min(ys)
        for x in xs:
            box(cell, "M3", x - HALF, y_lo - CAP, x + HALF, max(ys) + CAP)
            for y in ys:
                _square(cell, "V1", x, y)
                box(cell, "M2", x - PAD, y - HALF, x + PAD, y + HALF)
                _square(cell, "V2", x, y)
            _label(cell, net, "M3", (x, (y_lo + max(ys)) / 2))


def _tie_array_vss(spec: RomDecoderSpec, cell: Any) -> None:
    """An M2 bar along the array's top edge, from VSS strap to VSS strap.

    A V1 lands it on each tile's VSS column, carried up past the edge for the
    via's end-cap; the decoder's bottom rail stays at least 31 nm clear above.
    """
    top = spec.array.height
    vss = spec.strap_xs["VSS"]
    box(cell, "M2", min(vss) - PAD, top - HALF, max(vss) + PAD, top + HALF)
    for x in vss:
        _square(cell, "V2", x, top)
    bitcell = spec.array.bitcell
    for t in range(spec.array.tiles):
        x_s = t * bitcell.width + bitcell.sd_xs["S"]
        box(cell, "M1", x_s - HALF, top - HALF, x_s + HALF, top + CAP)
        _square(cell, "V1", x_s, top)


def build_rom(
    spec: RomDecoderSpec,
    programmed: Any = (),
    *,
    name: str | None = None,
    lib: Any = None,
) -> Any:
    """The array with its decoder on top, sharing every wordline's M3 and VSS."""
    gdspy = require_gdspy()
    array = build_rom_array(spec.array, programmed, lib=lib)
    decoder = build_rom_decoder(spec, lib=lib)
    cell = _new_cell(lib, name or f"rom_{spec.cell_name.removeprefix('rom_decoder_')}")
    cell.add(gdspy.CellReference(array))
    cell.add(gdspy.CellReference(decoder))
    _tie_array_vss(spec, cell)
    return cell


def main(argv: list[str] | None = None) -> None:
    """Write a ROM with its decoder: ``asap7-rom --array.words 64 --array.bits 8``.

    ``--pattern`` fills it: ``none``, ``all`` or ``checker`` bits set.
    """
    args = parse_spec(
        RomDecoderSpec,
        argv,
        description="Generate an ASAP7 NOR-ROM array with its wordline decoder.",
        options=(
            Option(
                "pattern",
                Literal["none", "all", "checker"],
                "checker",
                "Which bits to program.",
            ),
        ),
    )
    spec: RomDecoderSpec = args.spec
    array = spec.array
    every = [(w, b) for w in range(array.words) for b in range(array.bits)]
    sites = {
        "none": [],
        "all": every,
        "checker": [(w, b) for w, b in every if (w + b) % 2],
    }[args.pattern]
    cell, out = write_gds(lambda lib: build_rom(spec, sites, lib=lib), args.out)
    print(f"✓ {cell.name} -> {out}")
    print(
        f"  {array.words} words x {array.bits} bits; decoder {spec.interleave} rows, "
        f"{spec.pa_bits}+{spec.address_bits - spec.pa_bits} predecode bits, "
        f"{spec.pair_rails} rails per pair, {spec.height} nm tall"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
