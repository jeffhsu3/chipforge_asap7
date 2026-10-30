"""Via-programmed NOR-ROM bitcell whose wordline is the gate poly itself.

chipforge's `scripts/rom_compiler.py` gives every bitcell its own gate contact
up to a horizontal M2 wordline.  On ASAP7 that cannot fit the 81 nm row it
was drawn for, or any row near it: the contact's LIG may not sit on the
channel, so it goes above the diffusion, 14 nm clear of the full-height SDT
(LIG.SDT.S.8, SDT.W.2), and the gate cut above it needs 5 nm (LIG.GCUT.S.11)
and 17 nm of its own (GCUT.W.1) -- ``27 * fins + 57`` nm a row, which rounds
to 108 nm for one fin.

This cell has no gate contact at all.  The array is turned on its side: gate
poly runs the full height of the array and *is* the wordline (strapped to
metal by the periphery, every so many rows), each fin row is one bit, and its
bitline runs along the row on M2::

    x:   0    27    54    81   108   135   162
         D    WLa   S    WLb    D    ISO    D ...
         |    ||    |    ||     |    ||     |
         [=== one ACTIVE strip per row, never broken ===]   BL (M2)
              ||   VSS   ||          ||
                  (M1)            tied to VSS

One tile is two words by one bit, three gate pitches wide.  The two word
transistors share their source, which a continuous M1 column carries to VSS;
each has a drain of its own, with a short horizontal M1 stub under the row's
bitline.  A bit is programmed by one V1 on that stub, so a set bit puts its
transistor on the bitline and a clear bit leaves the drain floating -- a
single-mask edit, as in chipforge's ROM, but on the drain side.  The third
gate, ``ISO``, is held at VSS and isolates one tile's last drain from the next
tile's first. Those drains share a bitline only when both bits are programmed;
off-state isolation leakage still needs electrical characterization.

The diffusion is never broken, which is what lets it be this dense: a break
between nets needs 92 nm (ACTIVE.S.2A).  Rows are ``fins + 1`` fin pitches
tall -- the empty pitch is ACTIVE.S.1's 27 nm -- so a bit costs
``81 * 27 * (fins + 1)`` nm²: 81 x 81 nm with two fins.

All units are nanometres.  The tile assumes it is placed on the ASAP7 grids:
x a multiple of `GATE_PITCH` (gates on ``27 + 54k``, GATE.S.1) and y a
multiple of `FIN_PITCH` (fins on ``10 + 27k``, FIN.S.1); `build_rom_array`
places it so.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from ..layout.grid import FIN_PITCH, FIN_WIDTH, GATE_PITCH, GATE_WIDTH
from ..layout.layers import PIN_LAYERS, box, require_gdspy
from ..layout.rules import CONTACT_SIZE, M1_V0_ENCLOSURE, M2_V1_ENCLOSURE
from .cli import Option, parse_spec, write_gds
from .finfet import (
    ACTIVE_ENC,
    SD_BAR_WIDTH,
    SELECT_X_ENC,
    SELECT_Y_ENC,
    TAP_ACTIVE_WIDTH,
    VT_LAYERS,
)
from .inverter import ACTIVE_ABUT_OVERHANG

__all__ = [
    "ROM_TILE_WORDS",
    "STRAP_HEIGHT",
    "TAP_HEIGHT",
    "RomArraySpec",
    "RomBitcellSpec",
    "build_rom_array",
    "build_rom_bitcell",
    "build_rom_strap",
    "build_rom_tap",
]

#: Words per tile: two word transistors share one source column.
ROM_TILE_WORDS = 2
HALF = CONTACT_SIZE // 2
HALF_SD = SD_BAR_WIDTH // 2
HALF_GATE = GATE_WIDTH // 2
#: A drain stub is one via wide and this long, along the row.  M1.A.1 wants
#: 504 nm², which is 18 x 28; lying along the row it is only 18 nm tall, so
#: a one-fin row (54 nm) still leaves 36 nm between the stubs above and below.
STUB_LENGTH = 2 * (HALF + M1_V0_ENCLOSURE)
#: How far the stub runs from its drain toward the source column, away from
#: the next tile.  The stub's 18 nm end then faces the VSS column's side at
#: 26 nm (M1.S.2 asks 25), and the other end faces the neighbouring tile's
#: stub at 36.
STUB_REACH = STUB_LENGTH - HALF
#: Strap row height; see `build_rom_strap` for why it is 108 and not 81.
STRAP_HEIGHT = 4 * FIN_PITCH
#: Strap-row levels, from its bottom: the ISO LIG (and its V0), the M2 VSS
#: track, and the wordline LIGs with their V0/V1/V2 stacks.  The ISO and
#: wordline LIGs are 31 nm apart (LIG.S.4); the M2 landings stand 28 nm above
#: the VSS track (M2.S.2's 25, a short edge facing a long one).
STRAP_ISO_Y = 8
STRAP_VSS_Y = 9
STRAP_WL_Y = 55
#: X of each tile word's strap stack and M3 wordline, from the tile's left edge:
#: as far from the VSS column (x=54) as its LIG may reach (LIG.GATE.S.9B
#: keeps it 17 nm from the next gate).
STRAP_WL_XS = (HALF, 2 * GATE_PITCH - HALF)
#: Tap row height: one fin pitch of n-select clearance above the last bit row
#: (NSELECT.ACTIVE.EN.2) and two of p+ tie.  The tie is 16 nm wide, so it
#: takes two pitches to reach ACTIVE.A.1A's 864 nm².
TAP_TIE_HEIGHT = 2 * FIN_PITCH
TAP_HEIGHT = FIN_PITCH + TAP_TIE_HEIGHT
#: Every n-type ACTIVE must lie within this reach of a substrate tap
#: (ACTIVE.LUP.1).  The array's only tap row is along its top edge.
LATCHUP_REACH = 30_000
HALF_LIG = 8  # a 16 nm LIG (LIG.W.1): 20 x 16 on the gate is LIG.GATE.A.3's 320 nm²
LIG_PAST_GATE = 1  # LIG.GATE.EX.1
CAP = HALF + M1_V0_ENCLOSURE  # 14: M1 past a V0, along the bar
PAD = HALF + M2_V1_ENCLOSURE  # 17: M2 past a V1
#: The edge dummy gates and select stop this far outside the array.
_EDGE = GATE_PITCH
_GDS_UNIT_METERS = 1e-9


@dataclass(frozen=True)
class RomBitcellSpec:
    """A two-word, one-bit NOR-ROM tile, sized in fins.

    Args:
        fins: active fins per row, the only width knob; the row is ``fins +
            1`` fin pitches tall.
        vt: threshold-voltage flavor: ``"rvt"``, ``"lvt"``, ``"slvt"`` or
            ``"sram"``.  Non-RVT choices add the corresponding marker.
    """

    fins: int = 2
    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"

    def __post_init__(self) -> None:
        if isinstance(self.fins, bool) or not isinstance(self.fins, int):
            raise TypeError(f"fins must be an integer, got {self.fins!r}")
        if self.fins < 1:
            raise ValueError(f"fins must be >= 1, got {self.fins}")
        if self.vt not in VT_LAYERS:
            raise ValueError(f"vt must be one of {tuple(VT_LAYERS)}, got {self.vt!r}")

    @property
    def cell_name(self) -> str:
        vt_tag = "" if self.vt == "rvt" else f"_{self.vt}"
        return f"rom_bitcell{vt_tag}_f{self.fins}"

    @property
    def width(self) -> int:
        """Tile width: three gate pitches for two words."""
        return 3 * GATE_PITCH

    @property
    def word_pitch(self) -> float:
        """Width per word, averaged over the tile: 81 nm."""
        return self.width / ROM_TILE_WORDS

    @property
    def height(self) -> int:
        """Row (bit) pitch: the active fins plus one empty fin pitch."""
        return (self.fins + 1) * FIN_PITCH

    @property
    def active_span(self) -> tuple[int, int]:
        """Bottom/top Y of the row's ACTIVE strip: whole fin pitches from y=0."""
        return 0, self.fins * FIN_PITCH

    @property
    def contact_y(self) -> int:
        """Y of every V0, drain stub, programming V1 and the bitline."""
        lo, hi = self.active_span
        return (lo + hi) // 2

    @property
    def gate_xs(self) -> dict[str, int]:
        """Center X of the tile's three gates."""
        return {
            "WLa": GATE_PITCH // 2,
            "WLb": 3 * GATE_PITCH // 2,
            "ISO": 5 * GATE_PITCH // 2,
        }

    @property
    def sd_xs(self) -> dict[str, int]:
        """Center X of the tile's source/drain columns."""
        return {"Da": 0, "S": GATE_PITCH, "Db": 2 * GATE_PITCH}

    @property
    def stubs(self) -> dict[str, tuple[int, int]]:
        """X extent of each drain's M1 stub; each runs toward the source."""
        da, db = self.sd_xs["Da"], self.sd_xs["Db"]
        return {"Da": (da - HALF, da + STUB_REACH), "Db": (db - STUB_REACH, db + HALF)}

    def site(self, word: int) -> tuple[float, float]:
        """Programming V1 centre for `word` (0 or 1) in tile coordinates."""
        if word not in (0, 1):
            raise IndexError(f"a tile holds words 0 and 1, got {word}")
        x0, x1 = self.stubs["Da" if word == 0 else "Db"]
        return (x0 + x1) / 2, self.contact_y

    def netlist(
        self,
        bitline: str = "BL",
        wordlines: tuple[str, str] = ("WLa", "WLb"),
        programmed: tuple[bool, bool] = (True, True),
        vss: str = "VSS",
        prefix: str = "M",
    ) -> str:
        """The tile's transistors; an unprogrammed word's drain is its own floating node."""
        drains = [bitline if programmed[i] else f"{prefix}{i}_float" for i in range(2)]
        lines = [
            f"{prefix}{i} {drains[i]} {wordlines[i]} {vss} {vss} nmos_{self.vt} "
            f"nfin={self.fins} l={GATE_WIDTH}n"
            for i in range(2)
        ]
        return "\n".join(lines)


def _new_cell(name: str, lib: Any) -> Any:
    gdspy = require_gdspy()
    if lib is None:
        return gdspy.Cell(name, exclude_from_current=True)
    if lib.unit != _GDS_UNIT_METERS:
        raise ValueError(
            "the ROM generators emit nanometre coordinates; lib.unit must be "
            f"{_GDS_UNIT_METERS:g} metres, got {lib.unit!r}"
        )
    return lib.new_cell(name)


def _square(cell: Any, layer: str, x: float, y: float) -> None:
    box(cell, layer, x - HALF, y - HALF, x + HALF, y + HALF)


def build_rom_bitcell(
    spec: RomBitcellSpec, *, name: str | None = None, lib: Any = None
) -> Any:
    """Draw the unprogrammed tile and return the gdspy Cell.

    Programming V1s are not part of the tile; `build_rom_array` drops them at
    `RomArraySpec.site`, so one tile serves every bit pattern.  Everything
    that abuts -- fins, gates, ACTIVE, select, the VSS column and the bitline --
    runs to the tile edge; the drain column at x=0 is the tile's own and
    reaches 12 nm left of it.
    """
    cell = _new_cell(name or spec.cell_name, lib)
    width, height = spec.width, spec.height
    act_lo, act_hi = spec.active_span
    y = spec.contact_y

    box(cell, "NSELECT", 0, 0, width, height)
    if vt_layer := VT_LAYERS[spec.vt]:
        box(cell, vt_layer, 0, 0, width, height)
    for k in range(spec.fins + 1):
        y_fin = ACTIVE_ENC + k * FIN_PITCH
        box(cell, "FIN", 0, y_fin, width, y_fin + FIN_WIDTH)
    for x_gate in spec.gate_xs.values():
        box(cell, "GATE", x_gate - HALF_GATE, 0, x_gate + HALF_GATE, height)
    box(cell, "ACTIVE", 0, act_lo, width, act_hi)

    for x_sd in spec.sd_xs.values():
        box(cell, "SDT", x_sd - HALF_SD, act_lo, x_sd + HALF_SD, act_hi)
        box(cell, "LISD", x_sd - HALF_SD, act_lo, x_sd + HALF_SD, act_hi)
        _square(cell, "V0", x_sd, y)
    x_s = spec.sd_xs["S"]
    box(cell, "M1", x_s - HALF, 0, x_s + HALF, height)
    for x0, x1 in spec.stubs.values():
        box(cell, "M1", x0, y - HALF, x1, y + HALF)
    box(cell, "M2", 0, y - HALF, width, y + HALF)

    box(cell, "BOUNDARY", 0, 0, width, height)
    return cell


@dataclass(frozen=True)
class RomArraySpec:
    """`words` x `bits` of `bitcell` tiles, with a strap row under every `strap_every` rows.

    Args:
        words: wordlines, a multiple of `ROM_TILE_WORDS`; word ``w`` is gate
            ``w`` from the left.
        bits: bitlines; bit ``b`` is fin row ``b`` from the bottom.
        bitcell: the tile.
        strap_every: bit rows per strap row.  The array starts with a strap,
            so a wordline's poly runs at most this many rows from its M3.
    """

    words: int = 16
    bits: int = 16
    bitcell: RomBitcellSpec = RomBitcellSpec()
    strap_every: int = 16

    def __post_init__(self) -> None:
        for field in ("words", "bits", "strap_every"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an integer, got {value!r}")
        if self.words < ROM_TILE_WORDS or self.words % ROM_TILE_WORDS:
            raise ValueError(f"words must be a positive multiple of {ROM_TILE_WORDS}")
        if self.bits < 1:
            raise ValueError(f"bits must be >= 1, got {self.bits}")
        if self.strap_every < 1:
            raise ValueError(f"strap_every must be >= 1, got {self.strap_every}")
        if self.height > LATCHUP_REACH:
            raise ValueError(
                f"the array is {self.height} nm tall; its bottom row would be "
                f"further than {LATCHUP_REACH} nm from the tap row (ACTIVE.LUP.1)"
            )

    @property
    def cell_name(self) -> str:
        vt_tag = "" if self.bitcell.vt == "rvt" else f"_{self.bitcell.vt}"
        return (
            f"rom_array{vt_tag}_f{self.bitcell.fins}_{self.words}x{self.bits}"
            f"_s{self.strap_every}"
        )

    @property
    def tiles(self) -> int:
        return self.words // ROM_TILE_WORDS

    @property
    def straps(self) -> int:
        return -(-self.bits // self.strap_every)

    @property
    def width(self) -> int:
        """Width of the tiled core, without the edge dummies and select."""
        return self.tiles * self.bitcell.width

    @property
    def height(self) -> int:
        """Height of the tiled core: every strap row, every bit row and the tap row."""
        return self.tap_y + TAP_HEIGHT

    @property
    def tap_y(self) -> int:
        """Bottom Y of the tap row, which sits on the last bit row."""
        return self.straps * STRAP_HEIGHT + self.bits * self.bitcell.height

    @property
    def strap_ys(self) -> list[int]:
        """Bottom Y of each strap row."""
        group = STRAP_HEIGHT + self.strap_every * self.bitcell.height
        return [g * group for g in range(self.straps)]

    def row_y(self, bit: int) -> int:
        """Bottom Y of bit row `bit`."""
        if not 0 <= bit < self.bits:
            raise IndexError(f"bit {bit} is outside 0..{self.bits - 1}")
        return (bit // self.strap_every + 1) * STRAP_HEIGHT + bit * self.bitcell.height

    def wordline_x(self, word: int) -> int:
        """X of word `word`'s M3 strap."""
        if not 0 <= word < self.words:
            raise IndexError(f"word {word} is outside 0..{self.words - 1}")
        tile, tile_word = divmod(word, ROM_TILE_WORDS)
        return tile * self.bitcell.width + STRAP_WL_XS[tile_word]

    def site(self, word: int, bit: int) -> tuple[float, float]:
        """Programming V1 centre for ``(word, bit)`` in array coordinates."""
        self.wordline_x(word)  # range check
        tile, tile_word = divmod(word, ROM_TILE_WORDS)
        x, y = self.bitcell.site(tile_word)
        return tile * self.bitcell.width + x, self.row_y(bit) + y


def build_rom_strap(
    spec: RomBitcellSpec, *, name: str | None = None, lib: Any = None
) -> Any:
    """Draw the strap row for `spec`'s tile and return the gdspy Cell.

    The row has no ACTIVE.  Each gate gets an LIG, a V0 and an M1 bar:

    - ``WLa`` and ``WLb`` go on up through V1, an M2 landing and V2 to their
      M3 strap.  Their V0s sit at the LIG's far end, flush with it, so the bars
      stand 27 nm off the VSS column (M1.S.2: their 28 nm sides are short).
    - ``ISO``'s bar drops a V1 onto the row's M2 VSS track, which takes a V1
      from the VSS column too.  Its LIG sits on a lower level than the
      wordlines': on one level the 16 nm LIG ends would face across 16 nm,
      and LIG.S.4 wants 31.

    The row is 108 nm tall because of a one-fin tile.  There a wordline bar
    must keep 31 nm from the drain stubs above and below, and its LIG 31 nm
    from the ISO LIG.  81 nm would leave room for neither.
    """
    cell = _new_cell(name or _strap_name(spec), lib)
    width, height = spec.width, STRAP_HEIGHT
    gates = spec.gate_xs

    box(cell, "NSELECT", 0, 0, width, height)
    if vt_layer := VT_LAYERS[spec.vt]:
        box(cell, vt_layer, 0, 0, width, height)
    for k in range(height // FIN_PITCH):
        y_fin = ACTIVE_ENC + k * FIN_PITCH
        box(cell, "FIN", 0, y_fin, width, y_fin + FIN_WIDTH)
    for x_gate in gates.values():
        box(cell, "GATE", x_gate - HALF_GATE, 0, x_gate + HALF_GATE, height)
    x_s = spec.sd_xs["S"]
    box(cell, "M1", x_s - HALF, 0, x_s + HALF, height)

    # ISO: LIG just wide enough for its gate, V0 centred, bar down to VSS.
    x_iso, y = gates["ISO"], STRAP_ISO_Y
    box(cell, "LIG", x_iso - HALF_GATE - LIG_PAST_GATE, y - HALF_LIG,
        x_iso + HALF_GATE + LIG_PAST_GATE, y + HALF_LIG)  # fmt: skip
    _square(cell, "V0", x_iso, y)
    box(cell, "M1", x_iso - HALF, y - CAP - 1, x_iso + HALF, STRAP_VSS_Y + CAP)
    box(cell, "M2", 0, STRAP_VSS_Y - HALF, width, STRAP_VSS_Y + HALF)
    for x in (x_s, x_iso):
        _square(cell, "V1", x, STRAP_VSS_Y)

    # WLa and WLb: LIG from the gate out to the V0, which is flush with its end.
    y = STRAP_WL_Y
    for tile_word, gate in enumerate(("WLa", "WLb")):
        x_gate, x_via = gates[gate], STRAP_WL_XS[tile_word]
        near = x_gate + (HALF_GATE + LIG_PAST_GATE) * (1 if x_via < x_gate else -1)
        far = x_via - HALF if x_via < x_gate else x_via + HALF
        box(cell, "LIG", near, y - HALF_LIG, far, y + HALF_LIG)
        _square(cell, "V0", x_via, y)
        box(cell, "M1", x_via - HALF, y - CAP, x_via + HALF, y + CAP)
        _square(cell, "V1", x_via, y)
        box(cell, "M2", x_via - PAD, y - HALF, x_via + PAD, y + HALF)
        _square(cell, "V2", x_via, y)

    box(cell, "BOUNDARY", 0, 0, width, height)
    return cell


def build_rom_tap(
    spec: RomBitcellSpec, *, name: str | None = None, lib: Any = None
) -> Any:
    """Draw the tap row for `spec`'s tile and return the gdspy Cell.

    The tap is the tile's source column again, but p+ and on the substrate:
    a 16 nm ACTIVE (ACTIVE.W.3) between ``WLa`` and ``WLb``, 9 nm clear of
    each (GATE.ACTIVE.S.4), whose LISD and V0 land on the VSS column that
    already runs through it.  No gate crosses it, so the wordlines pass
    through the row untouched.

    The bottom fin pitch holds no ACTIVE: it is the 27 nm the last bit row's
    NSELECT reaches past its ACTIVE.  The tie fills the two pitches above it,
    16 x 54 nm being ACTIVE.A.1A's 864 nm², and the array's PSELECT frame
    covers the 27 nm above it.
    """
    cell = _new_cell(name or _tap_name(spec), lib)
    width, height = spec.width, TAP_HEIGHT
    box(cell, "PSELECT", 0, 0, width, height)
    if vt_layer := VT_LAYERS[spec.vt]:
        box(cell, vt_layer, 0, 0, width, height)
    for k in range(height // FIN_PITCH):
        y_fin = ACTIVE_ENC + k * FIN_PITCH
        box(cell, "FIN", 0, y_fin, width, y_fin + FIN_WIDTH)
    for x_gate in spec.gate_xs.values():
        box(cell, "GATE", x_gate - HALF_GATE, 0, x_gate + HALF_GATE, height)

    x_s = spec.sd_xs["S"]
    act_lo, act_hi = height - TAP_TIE_HEIGHT, height
    half_tie = TAP_ACTIVE_WIDTH // 2
    box(cell, "ACTIVE", x_s - half_tie, act_lo, x_s + half_tie, act_hi)
    box(cell, "SDT", x_s - HALF_SD, act_lo, x_s + HALF_SD, act_hi)
    box(cell, "LISD", x_s - HALF_SD, act_lo, x_s + HALF_SD, act_hi)
    _square(cell, "V0", x_s, act_lo + TAP_TIE_HEIGHT // 2)
    box(cell, "M1", x_s - HALF, 0, x_s + HALF, height)

    box(cell, "BOUNDARY", 0, 0, width, height)
    return cell


def _tap_name(spec: RomBitcellSpec) -> str:
    return "rom_tap" + ("" if spec.vt == "rvt" else f"_{spec.vt}")


def _strap_name(spec: RomBitcellSpec) -> str:
    return "rom_strap" + ("" if spec.vt == "rvt" else f"_{spec.vt}")


def _cell(lib: Any, name: str, build: Any) -> Any:
    """`lib`'s cell `name`, or a fresh one from `build`; tiles are shared across arrays."""
    if lib is not None and name in lib.cells:
        return lib.cells[name]
    return build()


def build_rom_array(
    spec: RomArraySpec,
    programmed: Iterable[tuple[int, int]] = (),
    *,
    name: str | None = None,
    lib: Any = None,
    bitline: str = "BL",
    bitline_side: Literal["left", "right"] = "left",
    first_word: int = 0,
) -> Any:
    """Tile `spec`, add its straps and close its edges; return the gdspy Cell.

    `programmed` holds ``(word, bit)`` pairs that get a V1.  Bit ``b``'s
    bitline is ``BL<b>`` on M2 (`bitline` names it), word ``w``'s wordline
    ``WL<w>`` on M3, labelled ``WL<first_word + w>``.  The edges are what a
    lone tile cannot provide:

    - a dummy gate at each end of every row, and an 8 nm ACTIVE overhang.
      That is 25 nm past the last gate on it (GATE.ACTIVE.EX.2) and 9 nm short
      of the dummy (GATE.ACTIVE.S.4);
    - the bitlines carried out past the edge on `bitline_side`;
    - select around the lot: NSELECT beside the rows, PSELECT around the
      tap row along the top.
    """
    sites = set(programmed)
    for word, bit in sites:
        if not (0 <= word < spec.words and 0 <= bit < spec.bits):
            raise IndexError(
                f"programmed site {(word, bit)} is outside {spec.words} x {spec.bits}"
            )

    gdspy = require_gdspy()
    bitcell = spec.bitcell
    tile = _cell(lib, bitcell.cell_name, lambda: build_rom_bitcell(bitcell, lib=lib))
    strap = _cell(lib, _strap_name(bitcell), lambda: build_rom_strap(bitcell, lib=lib))
    tap = _cell(lib, _tap_name(bitcell), lambda: build_rom_tap(bitcell, lib=lib))
    cell = _new_cell(name or spec.cell_name, lib)
    width, height = spec.width, spec.height
    for y_strap in spec.strap_ys:
        cell.add(
            gdspy.CellArray(strap, spec.tiles, 1, (bitcell.width, 0), (0, y_strap))
        )
    cell.add(gdspy.CellArray(tap, spec.tiles, 1, (bitcell.width, 0), (0, spec.tap_y)))
    for first in range(0, spec.bits, spec.strap_every):
        rows = min(spec.strap_every, spec.bits - first)
        cell.add(
            gdspy.CellArray(
                tile, spec.tiles, rows, (bitcell.width, bitcell.height),
                (0, spec.row_y(first)),
            )
        )  # fmt: skip

    act_lo, act_hi = bitcell.active_span
    for bit in range(spec.bits):
        y0 = spec.row_y(bit)
        for x0, x1 in (
            (-ACTIVE_ABUT_OVERHANG, 0),
            (width, width + ACTIVE_ABUT_OVERHANG),
        ):
            box(cell, "ACTIVE", x0, y0 + act_lo, x1, y0 + act_hi)
        y_bl = y0 + bitcell.contact_y
        if bitline_side == "left":
            x0, x1, x_pin = -PAD - GATE_PITCH, 0, -GATE_PITCH
        else:
            x0, x1, x_pin = width, width + GATE_PITCH + PAD, width + GATE_PITCH
            # Word 0's programming V1 sits 5 nm inside the left edge and
            # still needs its M2 end-cap (V1.M2.EN.2).
            box(cell, "M2", -PAD, y_bl - HALF, 0, y_bl + HALF)
        box(cell, "M2", x0, y_bl - HALF, x1, y_bl + HALF)
        _label(cell, f"{bitline}<{bit}>", "M2", (x_pin, y_bl))

    for word in range(spec.words):
        x = spec.wordline_x(word)
        box(cell, "M3", x - HALF, 0, x + HALF, height)
        _label(cell, f"WL<{first_word + word}>", "M3", (x, height - HALF))
    for x_gate in (-GATE_PITCH // 2, width + GATE_PITCH // 2):
        box(cell, "GATE", x_gate - HALF_GATE, 0, x_gate + HALF_GATE, height)
    for k in range(height // FIN_PITCH):
        y_fin = ACTIVE_ENC + k * FIN_PITCH
        for x0, x1 in ((-_EDGE, 0), (width, width + _EDGE)):
            box(cell, "FIN", x0, y_fin, x1, y_fin + FIN_WIDTH)

    # Select reaches 46 nm past ACTIVE across the rows (NSELECT.ACTIVE.EN.1)
    # and one fin pitch past the gate ends; the frame covers what the tiles
    # leave.  It is n-type beside the rows and p-type beside and above the
    # tap row.
    sel_x0 = -ACTIVE_ABUT_OVERHANG - SELECT_X_ENC
    sel_x1 = width + ACTIVE_ABUT_OVERHANG + SELECT_X_ENC
    sel_y0, sel_y1 = -SELECT_Y_ENC, height + SELECT_Y_ENC
    for layer, y0, y1 in (
        ("NSELECT", sel_y0, spec.tap_y),
        ("PSELECT", spec.tap_y, sel_y1),
    ):
        box(cell, layer, sel_x0, y0, 0, y1)
        box(cell, layer, width, y0, sel_x1, y1)
    box(cell, "NSELECT", 0, sel_y0, width, 0)
    box(cell, "PSELECT", 0, height, width, sel_y1)
    if vt_layer := VT_LAYERS[bitcell.vt]:
        box(cell, vt_layer, sel_x0, sel_y0, 0, sel_y1)
        box(cell, vt_layer, width, sel_y0, sel_x1, sel_y1)
        box(cell, vt_layer, 0, sel_y0, width, 0)
        box(cell, vt_layer, 0, height, width, sel_y1)

    # The bottom strap's VSS V1 sits on the array edge; carry each VSS column
    # on below it so the via keeps its end-cap (V1.M1.EN.1).
    for t in range(spec.tiles):
        x_s = t * bitcell.width + bitcell.sd_xs["S"]
        box(cell, "M1", x_s - HALF, -HALF, x_s + HALF, 0)
        _label(cell, "VSS", "M1", (x_s, HALF))
    for word, bit in sorted(sites):
        _square(cell, "V1", *spec.site(word, bit))

    box(cell, "BOUNDARY", sel_x0, sel_y0, sel_x1, sel_y1)
    return cell


def _label(cell: Any, text: str, metal: str, origin: tuple[float, float]) -> None:
    layer, texttype = PIN_LAYERS[metal]
    cell.add(require_gdspy().Label(text, origin, layer=layer, texttype=texttype))


def main(argv: list[str] | None = None) -> None:
    """Write a ROM tile, or an array with ``--words``: ``asap7-rom-bitcell --fins 2``."""
    args = parse_spec(
        RomBitcellSpec,
        argv,
        description="Generate the ASAP7 poly-wordline NOR-ROM tile.",
        options=(
            Option(
                "words",
                int,
                0,
                "Words in a test array (a multiple of 2); 0 writes the tile.",
            ),
            Option("bits", int, 1, "Bits (rows) in the test array."),
            Option("strap_every", int, 16, "Bit rows per wordline strap row."),
        ),
    )
    spec = args.spec
    if args.words:
        array = RomArraySpec(args.words, args.bits, spec, args.strap_every)
        build = lambda lib: build_rom_array(array, lib=lib)
    else:
        build = lambda lib: build_rom_bitcell(spec, lib=lib)
    cell, out = write_gds(build, args.out)
    print(f"✓ {cell.name} -> {out}")
    print(f"  bit: {spec.word_pitch:g} x {spec.height} nm ({spec.fins} fins)")


if __name__ == "__main__":  # pragma: no cover
    main()
