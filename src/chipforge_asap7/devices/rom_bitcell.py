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
gate, ``ISO``, isolates one tile's last drain from the next tile's first; both
are the same bitline, so even its leakage is from a net to itself.

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
from .finfet import ACTIVE_ENC, SD_BAR_WIDTH, SELECT_X_ENC, SELECT_Y_ENC, VT_LAYERS
from .inverter import ACTIVE_ABUT_OVERHANG

__all__ = [
    "ROM_TILE_WORDS",
    "RomBitcellSpec",
    "build_rom_array",
    "build_rom_bitcell",
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
    `RomBitcellSpec.site`, so one tile serves every bit pattern.  Everything
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


def build_rom_array(
    spec: RomBitcellSpec,
    words: int,
    bits: int,
    programmed: Iterable[tuple[int, int]] = (),
    *,
    name: str | None = None,
    lib: Any = None,
) -> Any:
    """Tile `words` x `bits` and close its edges; return the gdspy Cell.

    `programmed` holds ``(word, bit)`` pairs that get a V1.  Word ``w`` is
    gate ``w`` from the left; bit ``b`` is row ``b`` from the bottom, with its
    bitline ``BL<b>`` on M2.  The edges are what a lone tile cannot provide:
    a dummy gate and an 8 nm ACTIVE overhang at each end of every row (25 nm
    past the last gate on it, GATE.ACTIVE.EX.2, and 9 nm short of the dummy,
    GATE.ACTIVE.S.4), gates run one fin pitch below the bottom row, the
    bitlines carried out past the first drain's V1, and select around the lot.

    Not here yet: wordline straps, the ISO tie to VSS and a body tap, which
    belong to the periphery.  Until then the gates float and the array reports
    ACTIVE.LUP.1.
    """
    if words < ROM_TILE_WORDS or words % ROM_TILE_WORDS:
        raise ValueError(f"words must be a positive multiple of {ROM_TILE_WORDS}")
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    sites = set(programmed)
    for word, bit in sites:
        if not (0 <= word < words and 0 <= bit < bits):
            raise IndexError(
                f"programmed site {(word, bit)} is outside {words} x {bits}"
            )

    gdspy = require_gdspy()
    tile = build_rom_bitcell(spec, lib=lib)
    cell = _new_cell(name or f"rom_array_f{spec.fins}_{words}x{bits}", lib)
    tiles = words // ROM_TILE_WORDS
    width, height = tiles * spec.width, bits * spec.height
    cell.add(
        gdspy.CellArray(tile, tiles, bits, (spec.width, spec.height), origin=(0, 0))
    )

    act_lo, act_hi = spec.active_span
    y_bot = -FIN_PITCH
    for row in range(bits):
        y0 = row * spec.height
        box(cell, "ACTIVE", -ACTIVE_ABUT_OVERHANG, y0 + act_lo, 0, y0 + act_hi)
        box(
            cell,
            "ACTIVE",
            width,
            y0 + act_lo,
            width + ACTIVE_ABUT_OVERHANG,
            y0 + act_hi,
        )
        y_bl = y0 + spec.contact_y
        box(
            cell,
            "M2",
            -HALF - M2_V1_ENCLOSURE - GATE_PITCH,
            y_bl - HALF,
            0,
            y_bl + HALF,
        )
        _label(cell, f"BL<{row}>", "M2", (-GATE_PITCH, y_bl))

    # The tiles' gates continue down one fin pitch, past the bottom row's
    # ACTIVE; the two dummies span the same height.
    for x_gate in [-GATE_PITCH // 2, width + GATE_PITCH // 2]:
        box(cell, "GATE", x_gate - HALF_GATE, y_bot, x_gate + HALF_GATE, height)
    for t in range(tiles):
        for x_gate in spec.gate_xs.values():
            x = t * spec.width + x_gate
            box(cell, "GATE", x - HALF_GATE, y_bot, x + HALF_GATE, 0)
    y_fin = ACTIVE_ENC - FIN_PITCH
    box(cell, "FIN", -_EDGE, y_fin, width + _EDGE, y_fin + FIN_WIDTH)
    for k in range(bits * (spec.fins + 1)):
        y_fin = ACTIVE_ENC + k * FIN_PITCH
        for x0, x1 in ((-_EDGE, 0), (width, width + _EDGE)):
            box(cell, "FIN", x0, y_fin, x1, y_fin + FIN_WIDTH)

    # Select reaches 46 nm past ACTIVE across the rows and 27 nm past it up and
    # down (NSELECT.ACTIVE.EN.1-2); the frame below covers the margin the
    # tiles leave.
    sel_x0, sel_x1 = (
        -ACTIVE_ABUT_OVERHANG - SELECT_X_ENC,
        width + ACTIVE_ABUT_OVERHANG + SELECT_X_ENC,
    )
    sel_y0, sel_y1 = y_bot - SELECT_Y_ENC, height + SELECT_Y_ENC
    vt_layer = VT_LAYERS[spec.vt]
    for layer in ("NSELECT", vt_layer) if vt_layer else ("NSELECT",):
        box(cell, layer, sel_x0, sel_y0, 0, sel_y1)
        box(cell, layer, width, sel_y0, sel_x1, sel_y1)
        box(cell, layer, 0, sel_y0, width, 0)
        box(cell, layer, 0, height, width, sel_y1)

    for t in range(tiles):
        _label(cell, "VSS", "M1", (t * spec.width + spec.sd_xs["S"], height / 2))
    for word, bit in sorted(sites):
        tile_x, tile_word = divmod(word, ROM_TILE_WORDS)
        x, y = spec.site(tile_word)
        _square(cell, "V1", tile_x * spec.width + x, bit * spec.height + y)

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
        ),
    )
    spec = args.spec
    if args.words:
        build = lambda lib: build_rom_array(spec, args.words, args.bits, lib=lib)
    else:
        build = lambda lib: build_rom_bitcell(spec, lib=lib)
    cell, out = write_gds(build, args.out)
    print(f"✓ {cell.name} -> {out}")
    print(f"  bit: {spec.word_pitch:g} x {spec.height} nm ({spec.fins} fins)")


if __name__ == "__main__":  # pragma: no cover
    main()
