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
is as tall as its rails need, and at least its two rows and the gap.

The array is split in two, words below ``words / 2`` on the left and the
rest on the right, with a center lane between the halves.  The same rows
hold the predecoders there (`chipforge_asap7.devices.rom_predecoder`), from
address pins ``A<i>`` to the ``PA`` and ``PB`` lines, in columns of AND2 and
inverter slots, and each half's bitlines end in the lane::

    [ends][left slots][joins][predecoder columns][joins][right slots][ends]
       [ left half array ]== BLL<b>          BLR<b> ==[ right half array ]

The right half stands a whole number of tiles along, so its slots are the
left half's, shifted.  Over each pair a line is an M4 segment, from its
leftmost V3 to its rightmost: the wordline gates' lines keep their fixed
tracks and the predecoders' take whichever track is free over their span.  A
line used in more than one pair -- every ``PB``, and the predecoders' shared
lines -- has an M3 join column beside the predecoders: on the lane's left
side for a ``PB`` line only the left half uses, else on its right.  Each row
is padded with filler to one common span and ends in ``filler tap filler``
and ``tap filler``; a row too long for those to reach gets another tap in the
lane (`RomDecoderSpec.lane_tap_x`).  ``lane=False`` keeps the predecoders at the rows' left
end, ``[ends][predecoder columns][join columns][slots ...][ends]``, and one
array whose bitlines leave its left edge.

The supply grid is an M3 VSS and VDD strap on the outer filler at each end,
with a via stack onto every rail of its net.  The VSS straps run on down to
an M2 bar along the array's top edge, which `build_rom` lands on every tile's
VSS column in both halves.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
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
from .rom_bitcell import (
    LATCHUP_REACH,
    ROM_TILE_WORDS,
    STRAP_WL_XS,
    RomArraySpec,
    build_rom_array,
)
from .rom_control import RomControlSpec, build_rom_control_row
from .rom_lane import M5_WIDTH, TOP_CLEAR, V4_CAP, RomLaneSpec, build_rom_lane
from .rom_latch import M3_EDGE as M3_EDGE_LATCH
from .rom_latch import RomAddressLatchRowSpec, build_rom_address_latch_row
from .rom_predecoder import Gate, Placement, network, validate_placement
from .rom_predecoder import place as place_gates
from .row import RowStack
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
#: How far along a row a PMOS may stand from its well's tap: ACTIVE.LUP.1's reach,
#: less a margin for its round corner across the row and the cells' own widths.
WELL_TAP_REACH = LATCHUP_REACH - 1_000
#: Space between pairs.  Their VSS rails are not shared: at the 162 nm shift
#: between one pair's slots and the next one's, a filler's rail contact would
#: land on a source stub of the other row (V0.M1.AUX.3).  Two fin pitches
#: keep the two rails 36 nm apart.
PAIR_GAP = 2 * FIN_PITCH
#: M4 segments start and end on this grid.  Tips on adjacent tracks are then
#: aligned or 48 nm apart (M4.S.4), and segments that run alongside do so for
#: at least 48 nm (M4.S.5); `_fits` forbids the one case left, two tips
#: touching corner to corner (M4.S.3).
SEGMENT_GRID = 2 * M4_X_GRID
#: M4 past the centre of a V3 at a segment's end, along the track.
SEGMENT_END = HALF + V3_M4_ENCLOSURE
#: M3.S.6: stubs on neighbouring tracks closer than this corner to corner
#: are made to run alongside instead, for `M3_RUN`.
M3_CORNER = 20
M3_RUN = 2 * HALF
#: The lane's VDD strap top under the latch row's lowest M3: M3.S.6's corner
#: and two short edges' 31 nm, with room.
LATCH_STRAP_CLEAR = 40
#: The latch row's bottom rail over the lane's top one: a gate cut (44 nm)
#: and GCUT.S.3's 35 nm, on the fin grid.
LATCH_RAIL_CLEAR = 3 * FIN_PITCH
#: The latch channel's first track centre over the latch row's top edge: the
#: output columns' tops (17 nm under it) and the join columns' ends clear.
LATCH_CHANNEL_CLEAR = 34
#: The decoder's bottom edge over the channel's last track centre.
LATCH_CHANNEL_TOP = 60
#: The last join column stands this far left of the first wordline slot.
JOIN_MARGIN = 27


@dataclass(frozen=True)
class _SlotSpec:
    """Local routing for an independently sized NAND/inverter pair."""

    nand: NandSpec
    inverter: InverterSpec
    stack: RowStack
    interleave: int

    def __post_init__(self) -> None:
        lo, hi = self.inverter.input_strap_y(0)
        if self.link_y < self.nand.seam_y + 45 or not lo + 14 <= self.link_y <= hi - 14:
            raise ValueError(
                "ROM NAND/driver bands leave insufficient M2 link clearance; increase rows or driver fins"
            )

    @cached_property
    def slot_width(self) -> int:
        return self.interleave * 81

    @cached_property
    def inverter_x(self) -> int:
        return self.nand.width

    @cached_property
    def nand_y_x(self) -> int:
        x0, x1 = self.nand.output_bar_x(self.nand.columns("n", "Y")[0])
        return (x0 + x1) // 2

    @cached_property
    def link_y(self) -> int:
        """NAND Y to driver input, on M2: high on the driver's input strap.

        That keeps it 27 nm above the A pad on the seam (M2.S.2: the pad's ends
        are short) and the input strap's own end 13 nm past the via.
        """
        return int(
            min(self.inverter.input_strap_y(0)[1] - 22, self.nand.p_band.contact_y - 5)
        )

    @cached_property
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

    def wordline_tracks(self, parity: int) -> list[int]:
        """Slot-local X of every wordline track the slot sits under."""
        tile = 3 * GATE_PITCH
        shift = (self.wordline_local(parity) - STRAP_WL_XS[parity]) % tile
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

    @cached_property
    def _b_tie_x(self) -> tuple[int, int]:
        xs = self.nand.pad_gate_xs("B")
        return xs[0], xs[-1]

    @cached_property
    def pre_tracks(self) -> tuple[dict[str, int], dict[str, int]]:
        """Slot-local X of each stub of a predecoder slot, lower and upper row.

        The lower row takes an even wordline slot's tracks.  The upper row
        takes an odd one's A and Y; its B, which unlike a wordline gate's is
        not the lower row's line, takes the free track nearest the B tie.
        An inverter's input ``I`` uses the A track.
        """
        lower = {**self.stubs(0), "Y": self.wordline_local(0)}
        upper = {"A": self.stubs(1)["A"], "Y": self.wordline_local(1)}
        taken = [*lower.values(), *upper.values()]
        b_lo, b_hi = self._b_tie_x
        free = [
            x
            for x in range(b_lo, b_hi + 1, HALF)
            if all(abs(x - t) >= TRACK_PITCH for t in taken)
            and abs(x - upper["A"]) >= TRACK_PITCH + PAD
        ]
        if not free:
            raise ValueError("no free M3 track for an upper predecoder gate's B input")
        upper["B"] = min(free, key=lambda x: abs(x - (b_lo + b_hi) / 2))
        return (
            {**lower, "I": lower["A"]},
            {**upper, "I": upper["A"]},
        )

    @cached_property
    def pre_slot_width(self) -> int:
        """NAND, driver and the narrowest filler that keeps the next slot's
        stubs two tracks clear and this one's output pad inside."""
        tracks = [x for row in self.pre_tracks for x in row.values()]
        cpp = END_CPP
        while True:
            width = self.nand.width + self.inverter.width + cpp * GATE_PITCH
            if (
                width - max(tracks) + min(tracks) >= 2 * TRACK_PITCH
                and width >= max(tracks) + PAD + HALF
            ):
                return width
            cpp += 1


@dataclass(frozen=True)
class RomDecoderSpec:
    """The post-decode AND gates and wordline drivers for `array`.

    Args:
        array: the ROM array the decoder drives; its words must be a power of
            two, at least `interleave`.
        rows: fallback fin counts and minimum physical row stack. Each role
            can override its fins; their bands grow to common heights so all
            power rails and wells still align. ``((3, 3),)`` starts at 270 nm.
        interleave: decoder rows, a power of two.  Each word gets a slot this
            many words wide, which must hold a NAND, a driver and a filler.
        split: address bits in the ``PA`` group, at least log2(`interleave`).
            ``None`` picks the split with the fewest rails per pair.
        gap: nm between the array's top edge and the decoder's bottom one, a
            fin-pitch multiple of at least `MIN_GAP`.
        predecode: draw the predecoders, from address pins ``A<i>`` to the
            ``PA`` and ``PB`` rails.  They need at least ``2 * interleave``
            words; without them the rails are pins.
        lane: split the array in two, words below ``words / 2`` on the left
            and the rest on the right, with a lane between the halves for the
            predecoders, the join columns and both halves' bitline ends.  It
            needs at least ``2 * interleave`` words.  Without it the
            predecoders sit at the rows' left end and the bitlines leave the
            array's left edge.
        nand_fins: postdecode NAND (n, p) fins per finger; None uses rows.
        driver_fins: wordline inverter (n, p) fins per finger; None uses rows.
        predecoder_nand_fins: predecoder NAND fins, independent of postdecode.
        predecoder_driver_fins: predecoder output/address-complement inverter
            fins, independent of the wordline drivers.
        nand_fingers: even finger count per postdecode NAND input.
        driver_fingers: even wordline-inverter finger count.
        predecoder_nand_fingers: even finger count per predecoder NAND input.
        predecoder_driver_fingers: even predecoder-inverter finger count.
        predecoder_placement: optional explicit placement from the ROM search.
            PA gates must remain in their wordline pair; every column must
            contain both rows. None uses the original greedy placement.
        enable: gate the wordlines with an ``EVAL`` pin: the predecoders
            AND the top address bit's literals with it, so no wordline rises
            while it is low (`chipforge_asap7.devices.rom_predecoder.network`).
        latch: latch the address on ``PRECHN``: a row of static latches in
            the gap over the lane (`chipforge_asap7.devices.rom_latch`),
            transparent while it is low.  The pins ``A<i>`` move to the
            latches' inputs and the decoder's lines become ``AL<i>``; each
            latch's output reaches its line's join column on M4, in a
            channel over the latches.  Needs the sense lane, whose
            ``PRECHN`` column the latches share; the gap grows to fit.
        sense: fill the lane beside the array with read leaves, one per bit,
            and route both halves' bitlines to them
            (`chipforge_asap7.devices.rom_lane`).  Needs the lane; the right
            half moves out if the leaves need more room than the predecoders.
        sample: put a sample latch beside each read leaf, transparent while
            the ``SAMPLE`` pin is high, with each bit's output on ``Q<b>``
            (`chipforge_asap7.devices.rom_sense`).  Needs the sense lane,
            whose rows grow to fit.
        register: make ``Q`` a rising-edge register output: a register
            latch after each sample latch, open while ``SAMPLE`` is low, so
            that ``Q<b>`` takes the word as ``SAMPLE`` falls and holds it
            for a cycle.  With ``control`` that is just after ``CLK`` rises,
            one cycle after the address was latched.  Needs sample.
        control: time each read itself from ``CLK`` and ``RESETN``: the
            controller (`chipforge_asap7.devices.rom_control`) stands at the
            end of the address-latch row and drives ``PRECHN``, ``EVAL`` and
            ``SAMPLE``, and a replica bit, one more bit row programmed in
            every word (`cells`), tells it when a read is done.  Needs
            enable, latch and sample.
        edge_pins: carry the interface to M5 columns reaching the macro's bottom
            edge, labelled there, as a placed block needs
            (`chipforge_asap7.devices.rom_pins`).  The bus pins become
            ``A[i]`` and ``Q[b]``, the B-ROM's names.  Needs control.
    """

    array: RomArraySpec = field(default_factory=RomArraySpec)
    rows: tuple[tuple[int, int], ...] = ((3, 3),)
    interleave: int = 8
    split: int | None = None
    gap: int = MIN_GAP
    predecode: bool = True
    lane: bool = True
    nand_fins: tuple[int, int] | None = None
    driver_fins: tuple[int, int] | None = None
    predecoder_nand_fins: tuple[int, int] | None = None
    predecoder_driver_fins: tuple[int, int] | None = None
    nand_fingers: int = 2
    driver_fingers: int = 2
    predecoder_nand_fingers: int = 2
    predecoder_driver_fingers: int = 2
    predecoder_placement: tuple[Placement, ...] | None = field(default=None, repr=False)
    enable: bool = False
    latch: bool = False
    sense: bool = False
    sample: bool = False
    register: bool = False
    control: bool = False
    control_fingers: int = 4
    edge_pins: bool = False

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
        if self.predecode and words < 2 * k:
            raise ValueError(
                f"predecoders need at least 2 * interleave = {2 * k} words, so "
                f"that PB has an address bit; got {words}"
            )
        if self.lane and words < 2 * k:
            raise ValueError(
                f"the center lane needs at least 2 * interleave = {2 * k} words, "
                f"so that each half has a slot in every row; got {words} "
                "(lane=False puts every word on one side)"
            )
        if self.enable and not self.predecode:
            raise ValueError(
                "the EVAL gate is in the predecoders: enable needs predecode"
            )
        if self.sense and not self.lane:
            raise ValueError(
                "the sense leaves stand in the center lane: sense needs lane"
            )
        if self.control and not (self.enable and self.latch and self.sample):
            raise ValueError(
                "the controller drives EVAL, the address latches' PRECHN and the "
                "sample latches' SAMPLE: control needs enable, latch and sample"
            )
        if self.edge_pins and not self.control:
            raise ValueError(
                "the edge pins are the self-timed interface's: edge_pins needs control"
            )
        if self.register and not self.sample:
            raise ValueError(
                "the register latches follow the sample latches: register needs sample"
            )
        if self.sample and not self.sense:
            raise ValueError(
                "the sample latches stand beside the read leaves: sample needs sense"
            )
        if self.latch and not (self.sense and self.predecode):
            raise ValueError(
                "the address latches share the sense lane's PRECHN and feed the "
                "predecoders: latch needs sense and predecode"
            )
        if self.latch and self.gap < self._latch_gap:
            object.__setattr__(self, "gap", self._latch_gap)
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
        hi = self._max_pa_bits
        if self.split is not None and not lo <= self.split <= hi:
            raise ValueError(f"split must be {lo}..{hi} address bits, got {self.split}")
        if self.gap < MIN_GAP or self.gap % FIN_PITCH:
            raise ValueError(
                f"gap must be a multiple of {FIN_PITCH} and at least {MIN_GAP}; "
                f"got {self.gap}"
            )
        for parity in (0, 1):
            self.stubs(parity)  # raises if the slot has no free tracks
        if self.predecode:
            self.pre_tracks  # noqa: B018 - raises if the slot has no free tracks
        if self.predecoder_placement is not None:
            if not self.predecode:
                raise ValueError(
                    "an explicit predecoder placement needs predecode=True"
                )
            object.__setattr__(
                self, "predecoder_placement", tuple(self.predecoder_placement)
            )
            validate_placement(
                network(
                    self.address_bits,
                    self.pa_bits,
                    self.row_bits,
                    self.enable_line,
                    address=self.address_line,
                ),
                self.interleave,
                self.predecoder_placement,
            )
        if self.half.width > WELL_TAP_REACH:  # no shorter row can be too long
            longest = max(b - a for a, b in pairwise(self.well_taps))
            if longest > 2 * WELL_TAP_REACH:
                raise ValueError(
                    f"a {self.array.words}-word decoder has {longest} nm of row between "
                    f"N-well taps, which leaves PMOS further than {LATCHUP_REACH} nm "
                    "from a tap (ACTIVE.LUP.1)"
                    + ("" if self.lane else "; the lane would add one")
                )

    # ── Address ───────────────────────────────────────────────────────────────
    @cached_property
    def address_bits(self) -> int:
        return self.array.words.bit_length() - 1

    @cached_property
    def row_bits(self) -> int:
        return self.interleave.bit_length() - 1

    @cached_property
    def pa_bits(self) -> int:
        """Address bits in the PA group: `split`, or the fewest rails."""
        if self.split is not None:
            return self.split
        return min(
            range(self.row_bits, self._max_pa_bits + 1),
            key=lambda a: (self._pair_rails(a), a),
        )

    @cached_property
    def _max_pa_bits(self) -> int:
        """All the address bits, or with predecoders all but one for PB."""
        return self.address_bits - (1 if self.predecode else 0)

    def _pair_rails(self, pa_bits: int) -> int:
        share = 2 ** (pa_bits - self.row_bits)
        return 2 * share + 2 ** (self.address_bits - pa_bits)

    @cached_property
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
    def cell_fins(self) -> tuple[tuple[int, int], ...]:
        if len(self.rows) != 1:
            raise ValueError("the ROM decoder needs one (n_fins, p_fins) row")
        sizes = tuple(
            size if size is not None else self.rows[0]
            for size in (
                self.nand_fins,
                self.driver_fins,
                self.predecoder_nand_fins,
                self.predecoder_driver_fins,
            )
        )
        for size in sizes:
            if len(size) != 2 or any(
                isinstance(f, bool) or not isinstance(f, int) or f < 1 for f in size
            ):
                raise ValueError(
                    "each ROM cell size must be two positive integer fin counts"
                )
        return sizes

    @cached_property
    def stack(self) -> RowStack:
        sizes = self.cell_fins if self.predecode else self.cell_fins[:2]
        return RowStack(
            rows=(tuple(max(self.rows[0][i], *(s[i] for s in sizes)) for i in (0, 1)),),
            vt=self.array.bitcell.vt,
        )

    @property
    def band_height(self) -> tuple[int, int] | None:
        # Preserve the original geometry/names when all roles have one size.
        sizes = self.cell_fins if self.predecode else self.cell_fins[:2]
        if all(size == self.stack.rows[0] for size in sizes):
            return None
        return tuple(b.height for b in self.stack.bands())

    @cached_property
    def nand(self) -> NandSpec:
        return NandSpec(
            rows=(self.cell_fins[0],),
            fingers=self.nand_fingers,
            vt=self.array.bitcell.vt,
            band_height=self.band_height,
            rail_sources=self.band_height is not None,
        )

    @cached_property
    def inverter(self) -> InverterSpec:
        return InverterSpec(
            rows=(self.cell_fins[1],),
            fingers=self.driver_fingers,
            vt=self.array.bitcell.vt,
            band_height=self.band_height,
            rail_sources=self.band_height is not None,
            output_tie_offset=-54 if self.driver_fingers > 2 else 0,
        )

    @cached_property
    def word_slot(self) -> _SlotSpec:
        return _SlotSpec(self.nand, self.inverter, self.stack, self.interleave)

    @cached_property
    def pre_slot(self) -> _SlotSpec:
        return _SlotSpec(
            NandSpec(
                rows=(self.cell_fins[2],),
                fingers=self.predecoder_nand_fingers,
                vt=self.array.bitcell.vt,
                band_height=self.band_height,
                rail_sources=self.band_height is not None,
            ),
            InverterSpec(
                rows=(self.cell_fins[3],),
                fingers=self.predecoder_driver_fingers,
                vt=self.array.bitcell.vt,
                band_height=self.band_height,
                rail_sources=self.band_height is not None,
                output_tie_offset=-54 if self.predecoder_driver_fingers > 2 else 0,
            ),
            self.stack,
            self.interleave,
        )

    @cached_property
    def row_height(self) -> int:
        return self.nand.height

    @cached_property
    def slot_width(self) -> int:
        width = self.interleave * self.array.bitcell.word_pitch
        if width % GATE_PITCH:
            raise ValueError(f"a {width} nm slot is off the {GATE_PITCH} nm gate grid")
        return int(width)

    @property
    def filler_cpp(self) -> int:
        return (self.slot_width - self.nand.width - self.inverter.width) // GATE_PITCH

    @cached_property
    def pairs(self) -> int:
        return self.interleave // 2

    @cached_property
    def _min_pair_height(self) -> int:
        """Two rows and `PAIR_GAP`, or the wordline gates' rails if more.

        The rails sit on the absolute M4 lattice, so a pair holds its rails
        wherever that lattice falls: one pitch more than they span.
        """
        rails = M4_PITCH * (self.pair_rails + 1) + 2 * RAIL_MARGIN
        rows = 2 * self.row_height + PAIR_GAP
        return max(rows, -(-rails // FIN_PITCH) * FIN_PITCH)

    @cached_property
    def pair_height(self) -> int:
        """`_min_pair_height`, grown a fin pitch at a time until every pair's
        M4 segments find tracks, or a bounded routing attempt is rejected."""
        # Adjacent lines within a fixed group stay adjacent at every height.
        # A new placement can make their ends meet corner to corner; growing
        # the pair forever cannot repair that M4.S.3 conflict.
        for pair, spans in enumerate(self._routing_spans):
            for group in self.rail_groups(pair):
                for a, b in pairwise(group):
                    x0, x1 = spans[a]
                    y0, y1 = spans[b]
                    if x0 == y1 or x1 == y0:
                        raise ValueError(
                            f"fixed M4 rails {a} and {b} have conflicting tips in pair {pair}"
                        )
        # Enough extra space for a separate track plus a spacer for every
        # segment, with margins between the fixed groups. Keep a finite bound
        # even if a future routing constraint makes a placement impossible.
        extra = (2 * max(map(len, self._routing_spans), default=0) + 4) * M4_PITCH
        steps = -(-extra // FIN_PITCH)
        for step in range(steps + 1):
            height = self._min_pair_height + step * FIN_PITCH
            if self._pack(height) is not None:
                return height
        raise ValueError("ROM M4 segments do not fit within the row-pair routing bound")

    @cached_property
    def y0(self) -> int:
        """Bottom edge of the decoder, in array coordinates."""
        return self.cells.height + self.gap

    @cached_property
    def height(self) -> int:
        return self.pairs * self.pair_height

    def row_of(self, word: int) -> int:
        return word % self.interleave

    def row_y(self, row: int) -> int:
        """Bottom Y of `row`; odd rows are flipped, so their VSS rail is on top."""
        return self.y0 + (row // 2) * self.pair_height + (row % 2) * self.row_height

    def rail_y(self, pair: int, line: str) -> int:
        """Centre Y of `line`'s M4 segment over `pair`, on the absolute M4 lattice."""
        return self.segments[pair, line][0]

    @cached_property
    def segments(self) -> dict[tuple[int, str], tuple[int, int, int]]:
        """``(y, x0, x1)`` of each line's M4 segment over each pair it is in."""
        plan = self._pack(self.pair_height)
        assert plan is not None
        return plan

    def _tracks(self, pair: int, height: int) -> list[int]:
        """Y of every M4 track over `pair`, `RAIL_MARGIN` inside it."""
        y_lo = self.y0 + pair * height + RAIL_MARGIN
        y_hi = self.y0 + (pair + 1) * height - RAIL_MARGIN
        first = y_lo + (M4_TRACK_OFFSET - y_lo) % M4_PITCH
        return list(range(first, y_hi + 1, M4_PITCH))

    def _wordline_track(self, pair: int, line: str, tracks: int) -> int | None:
        """The fixed track index of a wordline gate's line, if `line` is one.

        The lower row's PA rails take the lowest tracks and the upper row's the
        highest, so each row's A stubs stay on its own side of the pair; PB,
        which both rows' shared B stubs reach, sits midway.
        """
        lower, pb, upper = self.rail_groups(pair)
        if line in lower:
            return lower.index(line)
        if line in upper:
            return tracks - len(upper) + upper.index(line)
        if line in pb:
            free = tracks - len(lower) - len(upper)
            return len(lower) + (free - len(pb)) // 2 + pb.index(line)
        return None

    @cached_property
    def _routing_spans(self) -> tuple[dict[str, tuple[int, int]], ...]:
        """Horizontal spans are independent of the row-pair height being tried."""
        return tuple(
            {line: _span(xs) for line, xs in self.pins(pair).items()}
            for pair in range(self.pairs)
        )

    def _pack(self, height: int) -> dict[tuple[int, str], tuple[int, int, int]] | None:
        """Tracks for every segment at `height`, or None if a pair runs out.

        The wordline gates' lines keep their fixed tracks; the predecoders'
        take the lowest track free over their span, first come first served
        from the left.
        """
        plan = {}
        for pair in range(self.pairs):
            ys = self._tracks(pair, height)
            if len(ys) < len(self.rails(pair)):
                return None
            taken: list[list[tuple[int, int]]] = [[] for _ in ys]
            spans = self._routing_spans[pair]
            fixed = {line: self._wordline_track(pair, line, len(ys)) for line in spans}
            order = sorted(spans, key=lambda line: (fixed[line] is None, spans[line]))
            for line in order:
                x0, x1 = spans[line]
                choices = [fixed[line]] if fixed[line] is not None else range(len(ys))
                track = next((t for t in choices if _fits(taken, t, x0, x1)), None)
                if track is None:
                    return None
                taken[track].append((x0, x1))
                plan[pair, line] = (ys[track], x0, x1)
        return plan

    def pins(self, pair: int) -> dict[str, list[int]]:
        """X of every V3 over `pair`, by line: stubs and join columns."""
        pins: dict[str, list[int]] = {}
        for row in (2 * pair, 2 * pair + 1):
            for w in range(row, self.array.words, self.interleave):
                stubs = self.stubs(w % 2)
                for pin, line in zip(("A", "B"), self.lines(w)):
                    pins.setdefault(line, []).append(self.slot_x(w) + stubs[pin])
        for placed in self.predecoder:
            if placed.pair == pair:
                x = self.pre_x(placed.column)
                for pin, line in self.pre_pins(placed.gate):
                    pins.setdefault(line, []).append(
                        x + self.pre_tracks[placed.row][pin]
                    )
        for line, x in self.joins.items():
            if pair in self.presence[line]:
                pins.setdefault(line, []).append(x)
        return pins

    # ── Predecoders ───────────────────────────────────────────────────────────
    @cached_property
    def predecoder(self) -> list[Placement]:
        """Each predecoder gate's pair, row and column; empty without them."""
        if not self.predecode:
            return []
        if self.predecoder_placement is not None:
            return list(self.predecoder_placement)
        gates = network(
            self.address_bits,
            self.pa_bits,
            self.row_bits,
            self.enable_line,
            address=self.address_line,
        )
        return place_gates(gates, self.interleave)

    @property
    def address_line(self) -> str:
        """The decoder's address lines: ``A`` from the pins, ``AL`` from the latches."""
        return "AL" if self.latch else "A"

    @property
    def enable_line(self) -> str | None:
        """The wordline enable's line, ``EVAL``, or None without one."""
        return "EVAL" if self.enable else None

    @cached_property
    def pre_columns(self) -> int:
        return max((p.column + 1 for p in self.predecoder), default=0)

    @staticmethod
    def pre_pins(gate: Gate) -> list[tuple[str, str]]:
        """``(pin, line)`` of a predecoder gate's stubs: inputs, then ``Y``."""
        names = ("A", "B") if gate.kind == "and" else ("I",)
        return [*zip(names, gate.ins), ("Y", gate.out)]

    @cached_property
    def pre_tracks(self) -> tuple[dict[str, int], dict[str, int]]:
        return self.pre_slot.pre_tracks

    @cached_property
    def pre_slot_width(self) -> int:
        return self.pre_slot.pre_slot_width

    @cached_property
    def presence(self) -> dict[str, set[int]]:
        """The pairs each line has a V3 in: its gate's and its consumers'."""
        presence: dict[str, set[int]] = {}
        for w in range(self.array.words):
            for line in self.lines(w):
                presence.setdefault(line, set()).add(self.row_of(w) // 2)
        for placed in self.predecoder:
            for _, line in self.pre_pins(placed.gate):
                presence.setdefault(line, set()).add(placed.pair)
        return presence

    @cached_property
    def _join_groups(self) -> tuple[dict[str, int], dict[str, int]]:
        """Each joined line's column in the left and in the right group.

        Lines whose pairs do not overlap share a column, bottom to top.  With
        the lane, a line only the left half's words use -- a ``PB`` line whose
        top address bit is 0 -- joins on the lane's left side, beside those
        words; every other line joins right of the predecoders.
        """
        left_only: set[str] = set()
        if self.lane:
            n, words = self.half.words, self.array.words
            right = {line for w in range(n, words) for line in self.lines(w)}
            left_only = {line for w in range(n) for line in self.lines(w)} - right
        spans: tuple[list[tuple[int, int, str]], ...] = ([], [])
        latched = set(self.channel_lines)
        for line, pairs in self.presence.items():
            if line in latched:
                # Its column runs on down to the latch channel, so nothing
                # may share it below.
                spans[1].append((0, max(pairs), line))
            elif len(pairs) > 1:
                spans[line not in left_only].append((min(pairs), max(pairs), line))
        groups = []
        for group in spans:
            columns: list[int] = []  # the top pair each column has reached
            column_of = {}
            for lo, hi, line in sorted(group):
                c = next((c for c, top in enumerate(columns) if top < lo), len(columns))
                if c == len(columns):
                    columns.append(hi)
                columns[c] = hi
                column_of[line] = c
            groups.append(column_of)
        return groups[0], groups[1]

    @cached_property
    def joins(self) -> dict[str, int]:
        """X of the M3 column joining each line that is in more than one pair."""
        left, right = self._join_groups
        n = max(right.values(), default=-1) + 1
        if not self.lane:
            last = self.slots_span[0] - JOIN_MARGIN
            return {line: last - (n - 1 - c) * TRACK_PITCH for line, c in right.items()}
        first = self.pre_x(self.pre_columns) + TRACK_PITCH
        return {
            **{line: self._left_join_x + c * TRACK_PITCH for line, c in left.items()},
            **{line: first + c * TRACK_PITCH for line, c in right.items()},
        }

    @cached_property
    def _left_join_x(self) -> int:
        """X of the lane's first left join column, `JOIN_MARGIN` past the left
        half's slots."""
        return self.side_span(0)[1] + JOIN_MARGIN

    @cached_property
    def join_x0(self) -> int:
        """Left edge of the filler the join columns stand on, on the gate grid."""
        lo = self.slots_span[0]
        first = min(self.joins.values(), default=lo)
        return min(
            lo - END_CPP * GATE_PITCH, (first - TRACK_PITCH) // GATE_PITCH * GATE_PITCH
        )

    def pre_x(self, column: int) -> int:
        """Left edge of predecoder column `column`.

        With the lane the columns start on the gate grid past the left joins
        and at least `END_CPP` past the left half's slots.
        """
        if not self.lane:
            return self.join_x0 - (self.pre_columns - column) * self.pre_slot_width
        lo = self.side_span(0)[1] + END_CPP * GATE_PITCH
        n = max(self._join_groups[0].values(), default=-1) + 1
        if n:
            last = self._left_join_x + (n - 1) * TRACK_PITCH
            lo = max(lo, -(-(last + TRACK_PITCH) // GATE_PITCH) * GATE_PITCH)
        return lo + column * self.pre_slot_width

    # ── Halves and the lane ───────────────────────────────────────────────────
    @cached_property
    def half(self) -> RomArraySpec:
        """The array on each side of the lane, or without it the whole array."""
        if not self.lane:
            return self.cells
        return replace(self.cells, words=self.cells.words // 2)

    @cached_property
    def cells(self) -> RomArraySpec:
        """The array as drawn: `array`, with ``control`` its replica bit on top.

        Its strap rows are as close as `array` says or closer: the lane's rows
        are taller than a few bits, and a narrow array whose lane has too few
        tracks gets the height for another row from more straps.
        """
        cells = (
            replace(self.array, bits=self.array.bits + 1)
            if self.control
            else self.array
        )
        if not self.lane or not self.sense:
            return cells
        every = cells.strap_every
        while every >= 1:
            try:
                candidate = replace(cells, strap_every=every)
                half = replace(candidate, words=candidate.words // 2)
                self._lane_of(half).plan  # noqa: B018
            except ValueError:
                every //= 2
            else:
                return candidate
        return cells

    @property
    def replica_bit(self) -> int:
        """The replica's bit row, over the data bits."""
        if not self.control:
            raise ValueError("this ROM has no replica bit (control=False)")
        return self.array.bits

    @cached_property
    def half_xs(self) -> tuple[int, ...]:
        """X of each half array's origin, left to right."""
        return (0, self.right_x) if self.lane else (0,)

    @cached_property
    def _lane_end(self) -> int:
        """X the lane's predecoders and right join columns need, tap aside."""
        end = self.pre_x(self.pre_columns) + END_CPP * GATE_PITCH
        right = self._join_groups[1]
        if right:
            first = self.pre_x(self.pre_columns) + TRACK_PITCH
            end = max(end, first + max(right.values()) * TRACK_PITCH + JOIN_MARGIN)
        return end

    def _right_x_past(self, end: int) -> int:
        """The first whole tile's x whose slots clear `end` and the lane's rows.

        A whole tile keeps the right half's slots the left half's, shifted.
        """
        tile = self.half.bitcell.width
        right = -(-(end - self.side_span(0)[0]) // tile) * tile
        if self.sense:
            right = max(right, self._lane().min_right_x)
        if self.latch:
            # The right half's wordlines climb through the gap the latches sit in.
            end = self.latch_ends[1] + GATE_PITCH
            right = max(right, -(-end // tile) * tile)
        return right

    @cached_property
    def lane_tap_x(self) -> int | None:
        """X of the well tap in the lane, or ``None`` if the row ends reach it all.

        A row's N-well has its taps at the row's two ends, which hold a PMOS to
        30 um (ACTIVE.LUP.1).  A row too long for that gets a third tap just
        past the predecoders, on the filler every row has there, about halfway
        along.  The p-side needs nothing: the array's tap row runs under the
        whole decoder.
        """
        if not self.lane:
            return None
        lo, hi = self._left_slot_span
        span = self._right_x_past(self._lane_end) + hi - lo + 2 * END_CPP * GATE_PITCH
        if span <= 2 * WELL_TAP_REACH:
            return None
        return self.pre_x(self.pre_columns) + END_CPP * GATE_PITCH

    @cached_property
    def right_x(self) -> int:
        """X of the right half: the first whole tile whose slots clear the lane."""
        if not self.lane:
            raise ValueError("only a ROM with the lane has a right half")
        end = self._lane_end
        if self.lane_tap_x is not None:
            # The tap and the filler after it, before the right half's slots.
            end = max(end, self.lane_tap_x + 2 * END_CPP * GATE_PITCH)
        return self._right_x_past(end)

    @cached_property
    def well_taps(self) -> tuple[int, ...]:
        """X of the N-well taps along every row, left to right."""
        left = self.rows_span[0] + END_CPP * GATE_PITCH
        lane = () if self.lane_tap_x is None else (self.lane_tap_x,)
        return (left, *lane, self.slots_span[1])

    @cached_property
    def vdd_stripe_y(self) -> int:
        """Y of the M4 stripe that joins the sense lane's VDD to the decoder's."""
        top = self.cells.height
        y = top + (M4_TRACK_OFFSET - top) % M4_PITCH
        lowest = min(y for y, _, _ in self.segments.values())
        if lowest - y < M4_PITCH:
            raise ValueError(
                f"the {self.gap} nm gap leaves no M4 track for the lane's VDD stripe"
            )
        return y

    # ── Address latches ──────────────────────────────────────────────────────
    @property
    def latch_row(self) -> RomAddressLatchRowSpec:
        return RomAddressLatchRowSpec(latches=self.address_bits)

    @property
    def latch_y(self) -> int:
        """Y of the latch row: on the array's top edge, a rail clear of the lane's rows.

        Clear means the two rails' gate cuts 35 nm apart (GCUT.S.3).
        """
        lane = self._lane().height
        if self.cells.height - lane >= LATCH_RAIL_CLEAR:
            y = self.cells.height
        else:
            y = lane + LATCH_RAIL_CLEAR
            if 0 < y - self.cells.height < TOP_CLEAR:
                # Just off the VSS bar: its strap pad must clear the bar's M2.
                y = -(-(self.cells.height + TOP_CLEAR) // FIN_PITCH) * FIN_PITCH
        if self.latch_ends[1] > self._lane().strap_xs["VSS"] - GATE_PITCH:
            # The row stands over the lane's VDD strap, which climbs to the
            # VDD stripe: its lowest M3 clears the strap's top.
            top = self.cells.height
            stripe = top + (M4_TRACK_OFFSET - top) % M4_PITCH
            need = (
                stripe + V3_CAP + LATCH_STRAP_CLEAR - self.latch_row.levels["n"] + PAD
            )
            if need > y:
                # Off the VSS bar, its VSS rail's strap pad clears the bar's M2.
                need = max(need, top + TOP_CLEAR)
                y = -(-need // FIN_PITCH) * FIN_PITCH
        return y

    @property
    def channel_lines(self) -> tuple[str, ...]:
        """The decoder lines a channel track brings in from below: each
        latch's ``AL<i>``, and with ``control`` the controller's ``EVAL``."""
        if not self.latch:
            return ()
        lines = tuple(f"AL<{i}>" for i in range(self.address_bits))
        return (*lines, "EVAL") if self.control else lines

    @property
    def channel_nets(self) -> tuple[str, ...]:
        """Every net on the channel: the lines, and with ``control`` the
        controller's ``SAMPLE`` and ``READY``, which stay under them."""
        under = ("SAMPLE", "READY") if self.control else ()
        return (*under, *self.channel_lines)

    def _channel_tracks(self) -> list[int]:
        """Every other M4 track over the latch row, one per channel net."""
        y = self.latch_y + self.latch_row.height + LATCH_CHANNEL_CLEAR
        first = y + (M4_TRACK_OFFSET - y) % M4_PITCH
        return [first + 2 * M4_PITCH * i for i in range(len(self.channel_nets))]

    def control_pin_x(self, net: str) -> int:
        """X of a controller pin or output column, in macro coordinates."""
        cs = self.control_row
        if net in ("CLK", "RESETN", "READY"):
            return self.control_x + cs.pin_x(net)
        gate = next(g for g in cs.gates if g.out == net)
        return self.control_x + cs.out_x(gate)

    @property
    def port_names(self) -> dict[str, str]:
        """Each bus net's name as a pin, where that is not the net's own."""
        if not self.edge_pins:
            return {}
        return {
            **{f"A<{i}>": f"A[{i}]" for i in range(self.address_bits)},
            **{f"Q<{b}>": f"Q[{b}]" for b in range(self.array.bits)},
        }

    @property
    def interface_points(self) -> dict[str, tuple[float, float]]:
        """Where each signal pin meets its M3, in macro coordinates."""
        rs, lane = self.latch_row, self.sense_lane
        top = self.latch_y + rs.height - M3_EDGE_LATCH
        names = self.port_names
        return {
            **{names[f"Q<{b}>"]: lane.q_pin(b) for b in range(self.array.bits)},
            **{
                names[f"A<{i}>"]: (
                    self.latch_x + rs.d_x(i),
                    self.latch_y + rs.levels["X"],
                )
                for i in range(self.address_bits)
            },
            **{pin: (self.control_pin_x(pin), top - 9) for pin in ("CLK", "RESETN")},
        }

    @property
    def vss_strap_points(self) -> list[tuple[int, int]]:
        """A point on each M3 VSS strap: the decoder's two, the lane's and the
        latch row's."""
        rails = [y for y, net in self.supply_rails if net == "VSS"]
        rows = (min(rails) + max(rails)) // 2
        return [
            *((x, rows) for x in self.strap_xs["VSS"]),
            (self.sense_lane.strap_xs["VSS"], self.sense_lane.height // 2),
            (self.latch_strap_xs["VSS"], self.latch_y + self.latch_row.height // 2),
        ]

    @property
    def vdd_strap_xs(self) -> list[int]:
        """Each M3 VDD strap that lands on the M4 stripe."""
        return [
            *self.strap_xs["VDD"],
            self.sense_lane.strap_xs["VDD"],
            self.latch_strap_xs["VDD"],
        ]

    @property
    def _latch_gap(self) -> int:
        """The gap the latch row and its channel need, on the fin grid."""
        need = self._channel_tracks()[-1] + LATCH_CHANNEL_TOP - self.cells.height
        return -(-need // FIN_PITCH) * FIN_PITCH

    @property
    def latch_x(self) -> int:
        """X of the latch row's cell: its PRECHN column on the lane's."""
        return self._lane().prechn_x - self.latch_row.prechn_x

    @property
    def control_row(self) -> RomControlSpec:
        rs = self.latch_row
        return RomControlSpec(
            fingers=self.control_fingers,
            n_fins=rs.n_fins,
            p_fins=rs.p_fins,
            band_height=rs.band_height,
            vt=rs.vt,
        )

    @property
    def control_x(self) -> int:
        """X of the controller's cell: abutting the latch row's right edge."""
        return self.latch_x + self.latch_row.width

    @property
    def latch_ends(self) -> tuple[int, int]:
        """X span of the latch row with its row ends: filler, cell(s), filler, tap, filler."""
        x0 = self.latch_x - END_CPP * GATE_PITCH
        width = self.latch_row.width + (self.control_row.width if self.control else 0)
        return x0, self.latch_x + width + 3 * END_CPP * GATE_PITCH

    @property
    def latch_strap_xs(self) -> dict[str, int]:
        f = self.latch_ends[1] - END_CPP * GATE_PITCH
        return {"VSS": f + TRACK_PITCH, "VDD": f + 2 * TRACK_PITCH}

    def q_x(self, bit: int) -> int:
        return self.latch_x + self.latch_row.q_x(bit)

    def channel_source_x(self, line: str) -> int:
        """X of the column a channel line climbs from: its latch's output, or
        the controller's ``EVAL``."""
        if line == "EVAL":
            return self.control_pin_x("EVAL")
        return self.q_x(int(line[3:-1]))

    def join_bottom(self, line: str) -> int:
        """Bottom of `line`'s join column as the decoder draws it."""
        return min(self.rail_y(pair, line) for pair in self.presence[line]) - V3_CAP

    @cached_property
    def latch_tracks(self) -> dict[str, int]:
        """The channel track of each channel net, by name.

        A latch's output column climbs to its track and the line's join column
        comes down to it.  Where one bit's output column stands within a gate
        pitch of another bit's join column, the output's track has to be the
        lower one, or the two M3 ends would meet.
        """
        bits = self.channel_lines
        joins = {b: self.joins[b] for b in bits}
        source = {b: self.channel_source_x(b) for b in bits}
        below = {b: set() for b in bits}  # b's track is below each of these
        for b in bits:
            for o in bits:
                if o != b and abs(source[b] - joins[o]) < GATE_PITCH:
                    below[b].add(o)
        order, placed = [], set()
        while len(order) < len(bits):
            ready = [
                b
                for b in bits
                if b not in placed
                and not any(b in below[o] for o in bits if o not in placed and o != b)
            ]
            if not ready:
                raise ValueError("the latch outputs and join columns cannot be ordered")
            b = min(ready, key=lambda b: abs(source[b] - joins[b]))
            order.append(b)
            placed.add(b)
        tracks = self._channel_tracks()
        under = len(self.channel_nets) - len(bits)
        result = {b: tracks[i] for i, b in enumerate(self.channel_nets[:under])}
        result.update({b: tracks[under + i] for i, b in enumerate(order)})
        lowest = min(y for y, _, _ in self.segments.values())
        if lowest - max(tracks) < M4_PITCH:
            raise ValueError("the latch channel runs into the decoder's M4")
        return result

    @property
    def channel_track_y(self) -> dict[str, int]:
        """The channel track of every channel net."""
        return self.latch_tracks

    @cached_property
    def sense_lane(self) -> RomLaneSpec:
        """The read leaves and bitline routes in the lane beside the array."""
        if not self.sense:
            raise ValueError("this ROM has no sense lane (sense=False)")
        return self._lane(right_x=self.right_x)

    def _lane(self, right_x: int | None = None) -> RomLaneSpec:
        """The sense lane, at `right_x` or at its narrowest."""
        return self._lane_of(self.half, right_x)

    def _lane_of(self, half: RomArraySpec, right_x: int | None = None) -> RomLaneSpec:
        return RomLaneSpec(
            half,
            right_x=right_x,
            sample=self.sample,
            register=self.register,
            replica=self.control,
            q_labels=not self.edge_pins,
        )

    def wordline_x(self, word: int) -> int:
        """X of `word`'s M3 wordline, in either half."""
        self.array.wordline_x(word)  # range check
        n = self.half.words
        if word >= n:
            return self.right_x + self.half.wordline_x(word - n)
        return self.half.wordline_x(word)

    @cached_property
    def _left_slot_span(self) -> tuple[int, int]:
        k, n = self.interleave, self.half.words
        lo = min(self.slot_x(r) for r in range(k))
        hi = max(self.slot_x(n - k + r) for r in range(k)) + self.slot_width
        return lo, hi

    def side_span(self, side: int) -> tuple[int, int]:
        """X span of one half's slots, 0 left and 1 right, as `slots_span`."""
        lo, hi = self._left_slot_span
        shift = self.right_x if side else 0
        return lo + shift, hi + shift

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

    @cached_property
    def rows_span(self) -> tuple[int, int]:
        """X span of every row: predecoders, joins, slots and end cells."""
        lo, hi = self.slots_span
        if not self.lane:
            lo = self.pre_x(0)
        return lo - 3 * END_CPP * GATE_PITCH, hi + 2 * END_CPP * GATE_PITCH

    def fills(self, row: int) -> list[tuple[int, int]]:
        """X spans of `row` between its slots, predecoders and row ends.

        Rows are padded to `slots_span`; with the lane each half's slots end
        in filler up to the predecoders, which covers the join columns.  The
        filler leaves room for `lane_tap_x`'s tap.
        """
        k, sw = self.interleave, self.slot_width
        xs = [self.slot_x(w) for w in range(row, self.array.words, k)]
        lo, hi = self.slots_span
        if not self.lane:
            spans = [(self.join_x0, xs[0]), (xs[-1] + sw, hi)]
        else:
            m = self.half.words // k
            lane = [(self.pre_x(self.pre_columns), xs[m])]
            if self.lane_tap_x is not None:
                tap = self.lane_tap_x
                lane = [
                    (self.pre_x(self.pre_columns), tap),
                    (tap + END_CPP * GATE_PITCH, xs[m]),
                ]
            spans = [
                (lo, xs[0]),
                (xs[m - 1] + sw, self.pre_x(0)),
                *lane,
                (xs[-1] + sw, hi),
            ]
        return [(x0, x1) for x0, x1 in spans if x1 > x0]

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
        return self.word_slot.inverter_x

    @property
    def nand_y_x(self) -> int:
        return self.word_slot.nand_y_x

    @property
    def link_y(self) -> int:
        return self.word_slot.link_y

    @property
    def out_via(self) -> tuple[int, int]:
        return self.word_slot.out_via

    @property
    def _b_tie_x(self) -> tuple[int, int]:
        return self.word_slot._b_tie_x

    def wordline_local(self, parity: int) -> int:
        return self.word_slot.wordline_local(parity)

    def wordline_tracks(self, parity: int) -> list[int]:
        return self.word_slot.wordline_tracks(parity)

    def stubs(self, parity: int) -> dict[str, int]:
        return self.word_slot.stubs(parity)

    def slot_x(self, word: int) -> int:
        """Left edge of `word`'s slot, on the gate grid."""
        return self.wordline_x(word) - self.wordline_local(word % 2)

    def pin_y(
        self, row: int, pin: Literal["A", "B", "I", "Y"], *, predecode: bool = False
    ) -> int:
        """Y of `pin`'s V2 in `row`, in decoder coordinates.

        ``I`` is an inverter slot's input, ``Y`` a slot's output.
        """
        slot = self.pre_slot if predecode else self.word_slot
        local = {
            "A": slot.nand.seam_y,
            "B": slot.nand.tie_ys["B"],
            "I": slot.link_y,
            "Y": slot.out_via[1],
        }[pin]
        y0 = self.row_y(row)
        return y0 + local if row % 2 == 0 else y0 + self.row_height - local

    @property
    def macro_name(self) -> str:
        """``build_rom``'s default name: this shape's, whatever its contents."""
        return f"rom_{self.cell_name.removeprefix('rom_decoder_')}"

    @property
    def cell_name(self) -> str:
        tag = ""
        identity = (
            self.cell_fins,
            self.rows,
            self.gap,
            self.nand_fingers,
            self.driver_fingers,
            self.predecoder_nand_fingers,
            self.predecoder_driver_fingers,
        )
        if identity != (((3, 3),) * 4, ((3, 3),), MIN_GAP, 2, 2, 2, 2):
            tag = "_" + sha256(repr(identity).encode()).hexdigest()[:12]
        if self.predecoder_placement is not None:
            placement = tuple(
                (p.gate.out, p.pair, p.row, p.column) for p in self.predecoder_placement
            )
            tag += "_p" + sha256(repr(placement).encode()).hexdigest()[:12]
        return (
            f"rom_decoder_{self.array.cell_name.removeprefix('rom_array_')}"
            f"_k{self.interleave}_a{self.pa_bits}{'' if self.predecode else '_np'}"
            f"{'' if self.lane else '_edge'}{'_e' if self.enable else ''}"
            f"{'_l' if self.latch else ''}"
            f"{'_s' if self.sense else ''}{'_q' if self.sample else ''}"
            f"{'r' if self.register else ''}"
            f"{f'_c{self.control_fingers}' if self.control else ''}"
            f"{'_pins' if self.edge_pins else ''}{tag}"
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


def _span(xs: list[int]) -> tuple[int, int]:
    """An M4 segment over V3s at `xs`, its ends out on `SEGMENT_GRID`."""
    x0 = (min(xs) - SEGMENT_END) // SEGMENT_GRID * SEGMENT_GRID
    x1 = -(-(max(xs) + SEGMENT_END) // SEGMENT_GRID) * SEGMENT_GRID
    return x0, x1


def _fits(taken: list[list[tuple[int, int]]], track: int, x0: int, x1: int) -> bool:
    """Whether ``[x0, x1]`` is free on `track` (M4.S.2) and touches no tip of
    a segment on the next track corner to corner (M4.S.3)."""
    if any(x0 < b + SEGMENT_GRID and a < x1 + SEGMENT_GRID for a, b in taken[track]):
        return False
    for t in (track - 1, track + 1):
        if 0 <= t < len(taken) and any(x0 == b or x1 == a for a, b in taken[t]):
            return False
    return True


def _slot(
    spec: _SlotSpec,
    kind: Literal["and", "inv"],
    tracks: dict[str, int],
    width: int,
    name: str,
    lib: Any,
) -> Any:
    """A NAND, driver and filler (or, for ``inv``, filler, driver and filler),
    routed up to M2 and a V2 on each pin's M3 track in `tracks`."""

    def build() -> Any:
        gdspy = require_gdspy()
        inv = _shared(
            lib,
            spec.inverter.cell_name,
            lambda: build_inverter(spec.inverter, lib=lib, draw_pin_labels=False),
        )
        cell = _new_cell(lib, name)
        x_inv = spec.inverter_x
        if kind == "and":
            nand = _shared(
                lib,
                spec.nand.cell_name,
                lambda: build_nand(spec.nand, lib=lib, draw_pin_labels=False),
            )
            cell.add(gdspy.CellReference(nand, (0, 0)))
        else:
            cell.add(gdspy.CellReference(_filler(spec, lib, x_inv // GATE_PITCH)))
        cell.add(gdspy.CellReference(inv, (x_inv, 0)))
        x_fill = x_inv + spec.inverter.width
        cell.add(
            gdspy.CellReference(
                _filler(spec, lib, (width - x_fill) // GATE_PITCH), (x_fill, 0)
            )
        )

        y, x_in = spec.link_y, x_inv + spec.inverter.input_gate_x
        _square(cell, "V1", x_in, y)
        if kind == "and":
            # NAND Y -> driver input.
            x_y = spec.nand_y_x
            _square(cell, "V1", x_y, y)
            box(cell, "M2", x_y - PAD, y - HALF, x_in + PAD, y + HALF)
        else:
            # The driver's input, from its track on the same M2 row.
            x_i = tracks["I"]
            box(cell, "M2", x_i - PAD, y - HALF, x_in + PAD, y + HALF)
            _square(cell, "V2", x_i, y)

        # Driver output -> its M3 track.
        (x_out, y_out), x_wl = spec.out_via, tracks["Y"]
        _square(cell, "V1", x_out, y_out)
        box(cell, "M2", x_out - PAD, y_out - HALF, x_wl + PAD, y_out + HALF)
        _square(cell, "V2", x_wl, y_out)
        if kind == "inv":
            return cell

        # A: a V1 over the seam V0 of its pad, an M2 pad to its stub's V2.
        x_a, y_a = spec.nand.pad_gate_xs("A")[0], spec.nand.seam_y
        _square(cell, "V1", x_a, y_a)
        lo, hi = sorted((x_a, tracks["A"]))
        # Grow a short pad away from the link above it, toward the cell edge.
        box(
            cell,
            "M2",
            min(lo - PAD, hi + PAD - MIN_PAD),
            y_a - HALF,
            hi + PAD,
            y_a + HALF,
        )
        _square(cell, "V2", tracks["A"], y_a)
        # B: its V2 lands on the NAND's own M2 tie, which may need lengthening.
        y_b = spec.nand.tie_ys["B"]
        b_lo, b_hi = spec._b_tie_x
        box(
            cell,
            "M2",
            min(b_lo, tracks["B"] - PAD),
            y_b - HALF,
            max(b_hi, tracks["B"] + PAD),
            y_b + HALF,
        )
        _square(cell, "V2", tracks["B"], y_b)
        return cell

    return _shared(lib, name, build)


def _word_slot(spec: RomDecoderSpec, parity: int, lib: Any) -> Any:
    """One word's AND gate, its output on the word's own wordline track."""
    tracks = {**spec.stubs(parity), "Y": spec.wordline_local(parity)}
    name = f"{spec.cell_name}_slot{parity}"
    return _slot(spec.word_slot, "and", tracks, spec.slot_width, name, lib)


def _pre_slot(spec: RomDecoderSpec, kind: str, row: int, lib: Any) -> Any:
    name = f"{spec.cell_name}_pre_{kind}{row}"
    return _slot(
        spec.pre_slot, kind, spec.pre_tracks[row], spec.pre_slot_width, name, lib
    )  # type: ignore[arg-type]


def _stub_span(y_pin: int, y_rail: int) -> tuple[int, int]:
    """Y span of the M3 from a pin's V2 at `y_pin` to a V3 at `y_rail`."""
    lo = min(y_pin - CAP, y_rail - V3_CAP)
    hi = max(y_pin + CAP, y_rail + V3_CAP)
    if hi - lo < MIN_STUB:
        grow = MIN_STUB - (hi - lo)
        lo, hi = lo - grow // 2, hi + grow - grow // 2
    return lo, hi


def _stub(
    cell: Any, x: int, y_pin: int, y_rail: int, span: tuple[int, int] | None = None
) -> None:
    """M3 from a pin's V2 at `y_pin` to a V3 on the M4 rail at `y_rail`."""
    lo, hi = span or _stub_span(y_pin, y_rail)
    box(cell, "M3", x - HALF, lo, x + HALF, hi)
    box(cell, "V3", x - HALF, y_rail - _HALF_M4, x + HALF, y_rail + _HALF_M4)


def _overlap_corners(stubs: list[list[int]]) -> None:
    """Stretch stubs on neighbouring tracks that nearly meet end to end.

    A predecoder column's two rows put stubs one track apart (the upper B
    beside the lower A), and two such stubs that miss each other in Y face
    corner to corner, closer than M3.S.6's 20 nm.  The lower one is carried
    up to run alongside the upper for `M3_RUN` instead, which M3.S.1 allows.
    Each stub is ``[x, lo, hi, y_pin, y_rail]``.
    """
    for a in stubs:
        for b in stubs:
            if a is b or abs(a[0] - b[0]) >= TRACK_PITCH + HALF or a[2] > b[1]:
                continue
            if b[1] - a[2] < M3_CORNER:
                a[2] = b[1] + M3_RUN


def _filler(spec: RomDecoderSpec | _SlotSpec, lib: Any, cpp: int) -> Any:
    support = RowSupportSpec(stack=spec.stack, kind="filler", width_cpp=cpp)
    return _shared(
        lib,
        support.cell_name,
        lambda: build_row_support(support, lib=lib, draw_pin_labels=False),
    )


def _end(spec: RomDecoderSpec, lib: Any, kinds: tuple[str, ...]) -> list[Any]:
    cells = []
    for kind in kinds:
        support = RowSupportSpec(stack=spec.stack, kind=kind, width_cpp=END_CPP)  # type: ignore[arg-type]
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
    x_left, x_right = spec.rows_span[0], spec.slots_span[1]
    gates = {(p.pair, p.row, p.column): p.gate for p in spec.predecoder}
    pre_stubs: dict[tuple[int, int], list[list[int]]] = {}

    for row in range(spec.interleave):
        y0 = spec.row_y(row)
        flip = row % 2 == 1
        y_ref = y0 + h if flip else y0

        def put(item: Any, x: int, y_ref: int = y_ref, flip: bool = flip) -> None:
            cell.add(gdspy.CellReference(item, (x, y_ref), x_reflection=flip))

        words = range(row, spec.array.words, spec.interleave)
        xs = [spec.slot_x(w) for w in words]
        slot = _word_slot(spec, row % 2, lib)
        for x in xs:
            put(slot, x)
        # Filler under the join columns, and to pad every row to one span.
        for x0, x1 in spec.fills(row):
            put(_filler(spec, lib, (x1 - x0) // GATE_PITCH), x0)
        for i, support in enumerate(left):
            put(support, x_left + i * end_w)
        for i, support in enumerate(right):
            put(support, x_right + i * end_w)
        if spec.lane_tap_x is not None:
            put(left[1], spec.lane_tap_x)

        for w, x in zip(words, xs):
            # The wordline: from its V2 down to the array's top edge.
            y_v2 = spec.pin_y(row, "Y")
            x_wl = spec.wordline_x(w)
            box(cell, "M3", x_wl - HALF, spec.cells.height, x_wl + HALF, y_v2 + CAP)
            # A and B: M3 stubs up or down to a V3 on their rail.
            stubs = spec.stubs(w % 2)
            for pin, line in zip(("A", "B"), spec.lines(w)):
                _stub(
                    cell,
                    x + stubs[pin],
                    spec.pin_y(row, pin),
                    spec.rail_y(row // 2, line),
                )

        # Predecoder gates, each pin on an M3 stub to its line's segment.
        pair, half = divmod(row, 2)
        for column in range(spec.pre_columns):
            x = spec.pre_x(column)
            gate = gates.get((pair, half, column))
            if gate is None:
                put(_filler(spec, lib, spec.pre_slot_width // GATE_PITCH), x)
                continue
            put(_pre_slot(spec, gate.kind, half, lib), x)
            for pin, line in spec.pre_pins(gate):
                y_pin = spec.pin_y(row, pin, predecode=True)  # type: ignore[arg-type]
                y_rail = spec.rail_y(pair, line)
                x_s = x + spec.pre_tracks[half][pin]
                pre_stubs.setdefault((pair, column), []).append(
                    [x_s, *_stub_span(y_pin, y_rail), y_pin, y_rail]
                )
    for stubs in pre_stubs.values():
        _overlap_corners(stubs)
        for x_s, lo, hi, y_pin, y_rail in stubs:
            _stub(cell, x_s, y_pin, y_rail, (lo, hi))

    # Between pairs the two VSS rails' gate cuts would face across 10 nm
    # (GCUT.S.3); one cut spans the gap instead.
    half_cut = DEVICE_GATE_CUT_HEIGHT // 2
    x0, x1 = spec.rows_span
    for pair in range(1, spec.pairs):
        below, above = 2 * pair - 1, 2 * pair
        y_below = spec.row_y(below) + spec.row_height
        box(cell, "GATE_CUT", x0, y_below - half_cut, x1, spec.row_y(above) + half_cut)

    # M4 segments, and the M3 columns joining a line's segments across pairs.
    labelled = set()
    for (pair, line), (y, x0, x1) in sorted(spec.segments.items()):
        box(cell, "M4", x0, y - _HALF_M4, x1, y + _HALF_M4)
        if line in spec.joins:
            x = spec.joins[line]
            box(cell, "V3", x - HALF, y - _HALF_M4, x + HALF, y + _HALF_M4)
        if (line[:2] in ("PA", "PB", "A<") or line == "EVAL") and line not in labelled:
            labelled.add(line)
            _label(cell, line, "M4", (x0 + V3_M4_ENCLOSURE, y))
    for line, x in spec.joins.items():
        ys = [spec.rail_y(pair, line) for pair in spec.presence[line]]
        y_lo = min(ys) - V3_CAP
        y_hi = max(max(ys) + V3_CAP, y_lo + MIN_STUB)
        box(cell, "M3", x - HALF, y_lo, x + HALF, y_hi)
    for w in range(spec.array.words):
        x = spec.wordline_x(w)
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
    top = spec.cells.height
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
    top = spec.cells.height
    vss = spec.strap_xs["VSS"]
    box(cell, "M2", min(vss) - PAD, top - HALF, max(vss) + PAD, top + HALF)
    for x in vss:
        _square(cell, "V2", x, top)
    bitcell = spec.half.bitcell
    for x_half in spec.half_xs:
        for t in range(spec.half.tiles):
            x_s = x_half + t * bitcell.width + bitcell.sd_xs["S"]
            box(cell, "M1", x_s - HALF, top - HALF, x_s + HALF, top + CAP)
            _square(cell, "V1", x_s, top)


def build_rom(
    spec: RomDecoderSpec,
    programmed: Any = (),
    *,
    name: str | None = None,
    lib: Any = None,
) -> Any:
    """The array with its decoder on top, sharing every wordline's M3 and VSS.

    With the lane the array is two halves, each its own cell: the left one's
    bitlines ``BLL<b>`` leave its right edge and the right one's ``BLR<b>``
    its left edge, both into the lane.
    """
    gdspy = require_gdspy()
    sites = list(programmed)
    for word, bit in sites:
        if not (0 <= word < spec.array.words and 0 <= bit < spec.array.bits):
            raise IndexError(
                f"programmed site {(word, bit)} is outside "
                f"{spec.array.words} x {spec.array.bits}"
            )
    if spec.control:
        sites += [(w, spec.replica_bit) for w in range(spec.array.words)]
    decoder = build_rom_decoder(spec, lib=lib)
    cell = _new_cell(lib, name or spec.macro_name)
    if spec.lane:
        n, half = spec.half.words, spec.half
        for side, (x, label, edge) in enumerate(
            ((0, "BLL", "right"), (spec.right_x, "BLR", "left"))
        ):
            array = build_rom_array(
                half,
                [(w - side * n, b) for w, b in sites if (w >= n) == side],
                name=f"{cell.name}_{('left', 'right')[side]}",
                lib=lib,
                bitline=label,
                bitline_side=edge,  # type: ignore[arg-type]
                first_word=side * n,
            )
            cell.add(gdspy.CellReference(array, (x, 0)))
    else:
        cell.add(gdspy.CellReference(build_rom_array(spec.cells, sites, lib=lib)))
    cell.add(gdspy.CellReference(decoder))
    _tie_array_vss(spec, cell)
    if spec.sense:
        lane = build_rom_lane(spec.sense_lane, top=spec.cells.height, lib=lib)
        cell.add(gdspy.CellReference(lane))
        _tie_lane_vdd(spec, cell)
    if spec.latch:
        _draw_address_latches(spec, cell, lib)
    if spec.edge_pins:
        from .rom_pins import draw_macro_pins

        draw_macro_pins(spec, cell)
    return cell


def _draw_address_latches(spec: RomDecoderSpec, cell: Any, lib: Any) -> None:
    """The latch row in the gap, its supplies, PRECHN, and the M4 channel to the joins.

    The row ends in ``filler tap filler`` like every other row, with an M3
    VSS and VDD strap on its outer filler: VSS lands on the array's VSS bar,
    VDD on the lane's VDD stripe.  ``PRECHN`` is the lane's column carried on
    up.  Each latch output climbs on M3 to its channel track, the M4 runs to
    the line's join column, and the join column comes down to meet it.
    """
    gdspy = require_gdspy()
    rs = spec.latch_row
    y = spec.latch_y
    row = _shared(
        lib,
        rs.cell_name,
        lambda: build_rom_address_latch_row(rs, lib=lib, draw_pin_labels=False),
    )

    def support(kind: str) -> Any:
        s = RowSupportSpec(stack=rs.stack, kind=kind, width_cpp=END_CPP)  # type: ignore[arg-type]
        return _shared(lib, s.cell_name, lambda: build_row_support(s, lib=lib))

    x = spec.latch_ends[0]
    for item in ("filler", row, "filler", "tap", "filler"):
        ref = support(item) if isinstance(item, str) else item
        cell.add(gdspy.CellReference(ref, (x, y)))
        x += rs.width if ref is row else END_CPP * GATE_PITCH
        if ref is row and spec.control:
            x += spec.control_row.width  # the controller's cell stands here

    top = spec.cells.height
    rail_vss, rail_vdd = y, y + rs.height
    for net, xs in spec.latch_strap_xs.items():
        rail = rail_vss if net == "VSS" else rail_vdd
        lo = top - CAP if net == "VSS" else spec.vdd_stripe_y - V3_CAP
        box(cell, "M3", xs - HALF, lo, xs + HALF, rail_vdd + CAP)
        _square(cell, "V1", xs, rail)
        box(cell, "M2", xs - PAD, rail - HALF, xs + PAD, rail + HALF)
        _square(cell, "V2", xs, rail)
        if net == "VSS":
            _square(cell, "V2", xs, top)
        else:
            y_s = spec.vdd_stripe_y
            box(cell, "V3", xs - HALF, y_s - _HALF_M4, xs + HALF, y_s + _HALF_M4)

    x_pre = spec.sense_lane.prechn_x
    box(cell, "M3", x_pre - HALF, spec.sense_lane.height, x_pre + HALF, y)

    q_top = y + rs.height - M3_EDGE_LATCH
    if spec.control:
        _draw_controller(spec, cell, lib, q_top)
    for line in spec.channel_lines:
        track = spec.latch_tracks[line]
        xq, xj = spec.channel_source_x(line), spec.joins[line]
        box(cell, "M3", xq - HALF, q_top, xq + HALF, track + V3_CAP)
        box(cell, "M3", xj - HALF, track - V3_CAP, xj + HALF, spec.join_bottom(line))
        x0 = min(xq, xj) - SEGMENT_END
        x1 = max(xq, xj) + SEGMENT_END
        x0 -= x0 % SEGMENT_GRID
        x1 += -x1 % SEGMENT_GRID
        box(cell, "M4", x0, track - _HALF_M4, x1, track + _HALF_M4)
        for xv in (xq, xj):
            box(cell, "V3", xv - HALF, track - _HALF_M4, xv + HALF, track + _HALF_M4)
    if not spec.edge_pins:
        for bit in range(spec.address_bits):
            _label(
                cell,
                f"A<{bit}>",
                "M3",
                (spec.latch_x + rs.d_x(bit), y + rs.levels["X"]),
            )


def _draw_controller(spec: RomDecoderSpec, cell: Any, lib: Any, top: int) -> None:
    """The controller at the latch row's end, and its three routes.

    ``PRECHN`` reaches the latch row's own track on M2.  ``SAMPLE`` climbs to
    the channel's lowest track and comes down to the lane's ``SAMPLE`` column
    through the latch row's column 1, which carries nothing on M3.
    ``READY`` comes up from the replica on M5, over the latch row, to the
    next track, and down to the controller's pin.
    """
    gdspy = require_gdspy()
    rs, cs = spec.latch_row, spec.control_row
    y = spec.latch_y
    row = _shared(
        lib,
        cs.cell_name,
        lambda: build_rom_control_row(cs, lib=lib, draw_pin_labels=False),
    )
    cell.add(gdspy.CellReference(row, (spec.control_x, y)))
    y_pre = y + rs.levels["CKB"]
    x_last = spec.latch_x + rs.gate_x(rs.latches - 1, 3)
    box(cell, "M2", x_last, y_pre - HALF, spec.control_x + PAD, y_pre + HALF)
    lane = spec.sense_lane
    tracks = spec.channel_track_y
    for net in ("SAMPLE", "READY"):
        track = tracks[net]
        xc = spec.control_pin_x(net)
        x_far = lane.sample_x if net == "SAMPLE" else lane.ready_x
        box(cell, "M3", xc - HALF, top, xc + HALF, track + V3_CAP)
        box(cell, "V3", xc - HALF, track - _HALF_M4, xc + HALF, track + _HALF_M4)
        cap = SEGMENT_END if net == "SAMPLE" else V4_CAP
        x0 = min(xc - SEGMENT_END, x_far - cap)
        x1 = max(xc + SEGMENT_END, x_far + cap)
        x0 -= x0 % SEGMENT_GRID
        x1 += -x1 % SEGMENT_GRID
        box(cell, "M4", x0, track - _HALF_M4, x1, track + _HALF_M4)
        if net == "SAMPLE":
            box(cell, "V3", x_far - HALF, track - _HALF_M4, x_far + HALF,
                track + _HALF_M4)  # fmt: skip
            box(cell, "M3", x_far - HALF, lane.height, x_far + HALF, track + V3_CAP)
        else:
            h5 = M5_WIDTH // 2
            box(cell, "V4", x_far - h5, track - h5, x_far + h5, track + h5)
            box(cell, "M5", x_far - h5, lane.height, x_far + h5, track + V4_CAP)
    for pin in ("CLK", "RESETN"):
        _label(cell, pin, "M3", (spec.control_pin_x(pin), top - 9))


def _tie_lane_vdd(spec: RomDecoderSpec, cell: Any) -> None:
    """An M4 VDD stripe in the gap over the array, from VDD strap to VDD strap.

    The decoder's VDD straps come down to it, and so does the sense lane's;
    a V3 joins each.  It is the first M4 track over the array's top edge,
    under the VSS bar's M2, and its ends stand on the decoder's segment grid
    so that its tips meet the bottom pair's rails aligned or a grid apart
    (M4.S.4).
    """
    y = spec.vdd_stripe_y
    xs = [*spec.strap_xs["VDD"], spec.sense_lane.strap_xs["VDD"]]
    x0 = min(xs) - SEGMENT_END
    x1 = max(xs) + SEGMENT_END
    x0 -= x0 % SEGMENT_GRID
    x1 += -x1 % SEGMENT_GRID
    box(cell, "M4", x0, y - _HALF_M4, x1, y + _HALF_M4)
    for x in xs:
        box(cell, "V3", x - HALF, y - _HALF_M4, x + HALF, y + _HALF_M4)
    # The decoder's VDD straps start at its first VDD rail.
    first = min(y_rail for y_rail, net in spec.supply_rails if net == "VDD")
    for x in spec.strap_xs["VDD"]:
        box(cell, "M3", x - HALF, y - V3_CAP, x + HALF, first)
    x = spec.sense_lane.strap_xs["VDD"]
    box(cell, "M3", x - HALF, spec.sense_lane.height, x + HALF, y + V3_CAP)


def main(argv: list[str] | None = None) -> None:
    """Write a ROM with its decoder: ``asap7-rom --array.words 64 --array.bits 8``.

    ``--pattern`` fills it: ``none``, ``all`` or ``checker`` bits set.
    """
    args = parse_spec(
        RomDecoderSpec,
        argv,
        description="Generate an ASAP7 NOR-ROM array with its wordline decoder.",
        hide=("predecoder_placement",),
        options=(
            Option(
                "pattern",
                Literal["none", "all", "checker"],
                "checker",
                "Which bits to program.",
            ),
            Option(
                "auto_size",
                bool,
                False,
                "Size the four ROM cell roles from their loads.",
            ),
            Option(
                "wl_load_ff",
                float | None,
                None,
                "Worst wordline total capacitance in fF; enables sizing and replaces the estimate.",
            ),
            Option(
                "stage_effort",
                float,
                4.0,
                "Target logical effort for automatic sizing.",
            ),
            Option(
                "search",
                bool,
                False,
                "Search decoder splits and shapes before writing the best GDS.",
            ),
            Option("pack", bool, False, "Optimize predecoder placement at this shape."),
            Option(
                "search_interleaves",
                tuple[int, ...] | None,
                None,
                "Decoder interleaves to search; default: current, 8 and 16.",
            ),
            Option(
                "search_straps",
                tuple[int, ...] | None,
                None,
                "Strap intervals to search; default: retain the current interval.",
            ),
            Option(
                "packing_evaluations",
                int,
                64,
                "Maximum placement evaluations per shape, including the baseline.",
            ),
            Option("search_seed", int, 0, "Seed for reproducible placement search."),
            Option(
                "search_report",
                Path | None,
                None,
                "JSON search report; defaults to the GDS path with .search.json.",
            ),
            Option(
                "cdl",
                Path | None,
                None,
                "Also write the macro's flat CDL netlist here.",
            ),
            Option(
                "views",
                Path | None,
                None,
                "Also write the macro's LEF, Verilog and estimated Liberty to "
                "this directory; needs --edge_pins.",
            ),
            Option(
                "words",
                Path | None,
                None,
                "JSON array of unsigned word values to program; replaces --pattern. "
                "A 1 is a programmed bit, and reads back as 1.",
            ),
            Option(
                "package",
                Path | None,
                None,
                "Write the whole macro to this directory: GDS, CDL, views, the "
                "stable-named abstract and manifest.json; needs --edge_pins and "
                "replaces --out, --cdl and --views.",
            ),
            Option("drc", bool, False, "Run gdscheck on the packaged macro."),
            Option("lvs", bool, False, "Run LVS of the packaged macro on its CDL."),
        ),
    )
    spec: RomDecoderSpec = args.spec
    sizing = None
    search = None
    sites: list[tuple[int, int]] | None = None
    size_options = (
        {"wl_load_fF": args.wl_load_ff, "stage_effort": args.stage_effort}
        if args.auto_size or args.wl_load_ff is not None
        else None
    )
    try:
        if args.package is None and (args.drc or args.lvs):
            raise ValueError("--drc and --lvs check a macro written by --package")
        if args.package is not None:
            if args.out is not None or args.cdl is not None or args.views is not None:
                raise ValueError("--package replaces --out, --cdl and --views")
            if not spec.edge_pins:
                raise ValueError("--package writes a placeable macro: add --edge_pins")
        if args.search or args.pack:
            from .rom_search import (
                RomShapeSearch,
                optimize_rom_packing,
                search_rom_shapes,
            )

            if args.search:
                search = search_rom_shapes(
                    spec,
                    interleaves=args.search_interleaves,
                    strap_intervals=args.search_straps,
                    packing_evaluations=args.packing_evaluations,
                    seed=args.search_seed,
                    sizing=size_options,
                )
            else:
                if (
                    args.search_interleaves is not None
                    or args.search_straps is not None
                ):
                    raise ValueError("shape choices require --search")
                packed = optimize_rom_packing(
                    spec,
                    evaluations=args.packing_evaluations,
                    seed=args.search_seed,
                    sizing=size_options,
                )
                search = RomShapeSearch(
                    (packed.selected,), (), args.search_seed, args.packing_evaluations
                )
            spec, sizing = search.best.spec, search.best.sizing
        elif (
            args.search_report is not None
            or args.search_interleaves is not None
            or args.search_straps is not None
        ):
            raise ValueError("search options require --search or --pack")
        elif size_options is not None:
            from .rom_sizing import size_rom_decoder

            sizing = size_rom_decoder(spec, **size_options)
            spec = sizing.spec
        if args.words is not None:
            import json

            from .rom_package import sites_from_words

            sites = sites_from_words(json.loads(args.words.read_text()), spec)
    except (OSError, TypeError, ValueError) as exc:
        import sys

        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    array = spec.array
    every = [(w, b) for w in range(array.words) for b in range(array.bits)]
    if sites is None:
        sites = {
            "none": [],
            "all": every,
            "checker": [(w, b) for w, b in every if (w + b) % 2],
        }[args.pattern]
    failed = False
    if args.package is not None:
        from .rom_package import rom_image_name, write_rom_package

        failed = write_rom_package(
            spec, sites, args.package, drc=args.drc, lvs=args.lvs
        )
        top_name, out = rom_image_name(spec, sites), args.package / "macro.gds"
        print(f"✓ {top_name} -> {args.package / 'manifest.json'}")
    else:
        cell, out = write_gds(lambda lib: build_rom(spec, sites, lib=lib), args.out)
        top_name = cell.name
        print(f"✓ {top_name} -> {out}")
    if args.cdl is not None:
        from .rom_netlist import rom_macro_cdl, rom_macro_ports

        args.cdl.parent.mkdir(parents=True, exist_ok=True)
        args.cdl.write_text(rom_macro_cdl(spec, sites, name=top_name))
        print(f"  CDL -> {args.cdl}: pins {' '.join(rom_macro_ports(spec))}")
    if args.views is not None:
        from .rom_views import rom_macro_views

        args.views.mkdir(parents=True, exist_ok=True)
        for filename, text in rom_macro_views(spec, sites, cell).items():
            (args.views / filename).write_text(text)
            print(f"  view -> {args.views / filename}")
    if search is not None:
        import json

        report_path = args.search_report or out.with_suffix(".search.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = search.report()
        report["gds"] = {"path": str(out), "cell": top_name}
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"  search: {len(search.candidates)} feasible shapes, {len(search.rejected)} rejected; "
            f"array/decoder envelope {search.best.metrics.area_um2:.3f} um² -> {report_path}"
        )
    if sizing is not None:
        print(sizing.summary())
    print(
        f"  {array.words} words x {array.bits} bits; decoder {spec.interleave} rows, "
        f"{spec.pa_bits}+{spec.address_bits - spec.pa_bits} predecode bits, "
        f"{spec.pair_rails} rails per pair, {spec.height} nm tall"
    )
    if spec.predecode:
        print(
            f"  predecoders: {len(spec.predecoder)} gates in {spec.pre_columns} "
            f"columns, {len(spec.joins)} join columns; address pins "
            f"A<0>..A<{spec.address_bits - 1}>"
        )
    if spec.lane:
        lane = spec.right_x - spec.half.width
        print(
            f"  center lane: {lane} nm between the halves, words "
            f"0..{spec.half.words - 1} | {spec.half.words}..{array.words - 1}; "
            f"bitlines BLL<b> | BLR<b>"
        )
    if failed:
        raise SystemExit("ROM physical checks failed; see manifest.json and reports")


if __name__ == "__main__":  # pragma: no cover
    main()
