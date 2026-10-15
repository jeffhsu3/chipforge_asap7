# NOR-ROM bitcell: `chipforge_asap7.devices.rom_bitcell`

A via-programmed NOR-ROM tile whose wordline is the gate poly itself. With two
fins a bit is 81 x 81 nm.

```python
from chipforge_asap7.devices import (
    RomArraySpec, RomBitcellSpec, build_rom_array, build_rom_bitcell,
)

cell = RomBitcellSpec(fins=2)       # cell "rom_bitcell_f2"
cell.width, cell.height             # 162, 81: two words by one bit
tile = build_rom_bitcell(cell)      # unprogrammed; V1s are added by the array

spec = RomArraySpec(words=8, bits=6, bitcell=cell, strap_every=4)
spec.row_y(4), spec.site(5, 3)      # 540, (427.0, 378): rows step over straps
spec.wordline_x(5)                  # 423: WL<5>'s M3 strap
array = build_rom_array(spec, programmed={(0, 0), (5, 3)})
```

## Why the wordline is the poly

chipforge's `scripts/rom_compiler.py` gives each bitcell its own gate contact
up to a horizontal M2 wordline, on an 81 nm row. ASAP7 can't fit that contact
into any row near 81 nm:

- The LIG may not sit on the channel (LIG.GATE.S.10), so it goes above the
  diffusion.
- It must stay 14 nm from the SDT (LIG.SDT.S.8), and SDT covers the whole
  ACTIVE height (SDT.W.2, SDT.ACTIVE.OV.1).
- The gate cut above it needs 5 nm of clearance (LIG.GCUT.S.11) and 17 nm of
  width (GCUT.W.1).

That adds up to `27 * fins + 57` nm per row: 108 nm for one fin, 135 nm for
two. That compiler's cell also put the source and drain 36 nm from the gate on
a 54 nm gate pitch. That lays the LISD bars over the next gate stripe, so gdscheck
reports about 13 violations per bit on it.

Here the array is turned on its side. Gate poly runs the full height of the
array and is the wordline. Each fin row is one bit, and its bitline runs along
the row on M2:

```text
x:   0    27    54    81   108   135   162
     D    WLa   S    WLb    D    ISO    D ...
     [== one ACTIVE strip per row, never broken ==]   BL<b> (M2)
               VSS (M1 column)
```

- **Source:** the two word transistors share their source. A continuous M1
  column carries it to VSS.
- **Drains:** each word transistor has its own drain, with a 28 x 18 nm M1
  stub along the row (M1.A.1).
- **Programming:** one V1 on the stub ties the drain to the bitline. A clear
  bit leaves the drain floating. Either way it's a one-mask edit, as in
  chipforge's ROM, but on the drain side.
- **Isolation:** `ISO` separates one tile's last drain from the next tile's
  first. Both are the same bitline, so its leakage runs from a net back to
  itself.
- **Diffusion:** it is never broken. A break between different nets costs
  92 nm (ACTIVE.S.2A).
- **Row height:** each row is `fins + 1` fin pitches. The empty pitch is the
  27 nm that ACTIVE.S.1 requires between rows.

| fins | bit (nm) | area (nm²) |
| ---: | --- | ---: |
| 1 | 81 x 54 | 4374 |
| 2 | 81 x 81 | 6561 |
| 3 | 81 x 108 | 8748 |

## Strap row

The poly wordline is strapped to M3 by a 108 nm strap row. The array starts
with one, then adds another under every `strap_every` bit rows. Every 162 nm
tile of it contacts all three gates:

- **Wordlines:** `WLa` and `WLb` each get an LIG reaching from the gate to a
  V0 at its far end. An M1 bar, V1, an M2 landing and V2 then take it to the
  word's M3 wordline, at x = 9 and 99 in the tile.
- **Isolation tie:** `ISO`'s LIG sits on a lower level, with an M1 bar down to
  an M2 VSS track across the row. A V1 joins that track to the VSS column.

The two LIG levels are 31 nm apart (LIG.S.4). On one level, the ISO LIG and
the wordline LIGs on either side would face each other across 16 nm.

The row would be 81 nm if only two-fin tiles used it. It is 108 nm because a
one-fin tile's drain stubs come within 32 nm of its bottom edge, and each M1
bar has to keep 31 nm from them.
At `strap_every=16` the row adds 8% to a two-fin array's height.

```text
y (strap row)
 55   WLa LIG-V0-M1-V1-M2-V2 -> M3        WLb ... -> M3
  9   ======================= M2 VSS track ========================
  8                                                ISO LIG-V0-M1-V1 -^
x:    9    27        54        81    99         135
```

## Tap row

An 81 nm tap row along the top of the array ties the substrate to VSS
(ACTIVE.LUP.1). Each tile's tie is its source column again, now p+:

- **Tie:** a 16 x 54 nm p+ ACTIVE between `WLa` and `WLb`, 9 nm clear of each
  (ACTIVE.W.3, GATE.ACTIVE.S.4). Its two fin pitches are what ACTIVE.A.1A's
  864 nm² needs at that width.
- **Contact:** LISD and a V0 onto the VSS column, which already runs through
  the row.
- **Gates:** no gate crosses the tie, so the wordline poly runs on through
  untouched.
- **Select:** the bottom fin pitch is empty, because the last bit row's NSELECT
  reaches 27 nm past its ACTIVE. The array's frame is PSELECT beside and above
  the tap row, and NSELECT elsewhere.

One row covers an array up to 30 um tall. `RomArraySpec` refuses a taller one.

## Wordline decoder

```python
from chipforge_asap7.devices import RomDecoderSpec, build_rom

decoder = RomDecoderSpec(array=spec)    # interleave=8, (3, 3)-fin cells
decoder.pa_bits, decoder.pair_rails     # 3, 3: PA takes 3 address bits
decoder.lines(5)                        # ('PA<5>', 'PB<0>')
rom = build_rom(decoder, programmed={(0, 0), (5, 3)})
```

The decoder sits above the array. Each word gets an AND gate: a `NandSpec`
and an `InverterSpec` driver on a 270 nm row. Its output drops down the
word's own M3 track into the array's wordline.

An AND gate is 378 nm wide and a word is 81, so the gates are spread over
`interleave` rows. Word `w` sits in row `w % interleave`, in a slot
`interleave * 81` nm wide: 648 nm, or 12 CPP, at the default of 8. The slot
holds the NAND, the driver and a filler.

```text
pair 1   row 3 (flipped)  [NAND INV fill][NAND INV fill] ...   M4 rails
         row 2            [NAND INV fill][NAND INV fill] ...   over each pair
         -- 54 nm gap --
pair 0   row 1 (flipped)  [NAND INV fill][NAND INV fill] ...
         row 0            [NAND INV fill][NAND INV fill] ...
           |  |  |  |  |  |  |  |    WL<w> on M3, down into the array
```

- **Address:** the interleave is a power of two, so a word's row is its low
  address bits. `PA` takes the low `split` bits and `PB` the rest, and
  `WL<w> = PA<w mod 2^split> . PB<w >> split>`.
- **Rails:** a row only sees the `PA` lines that are its own row modulo the
  interleave. A pair of rows therefore needs two shares of `PA` and all of
  `PB` on M4: 8 rails for 64 words and 16 for 256. With 6 rows, every row
  would need every line, 32 rails at 256 words, which is why the interleave
  is 8 and not 6.
- **Rail placement:** the rails sit on the absolute M4 track lattice
  (M4.AUX.1, M4.AUX.2), so the decoder is drawn in array coordinates, `gap`
  above the array. A pair grows past its two rows when its rails need it.
- **Inputs:** `A` and `B` reach their rails on M3 stubs. The stubs run on free
  tracks, 36 nm from every wordline, so the wordlines of the rows above pass
  between them.
- **Pairs:** the two rows of a pair share a VDD rail and stand slot over slot.
  At any other offset a filler's rail contact lands on the other row's source
  stub (V0.M1.AUX.3). The two words share their `PB` line, so they share one
  B stub. Pairs sit a tile apart in X, so they don't share a rail; a 54 nm gap
  separates them.
- **Joins:** each `PB` line has a copy over every pair. One M3 per line,
  beyond the rows' left ends, joins the copies.
- **Row ends:** each pair's slots stand two words right of the pair below's,
  so the rows are padded with filler to one common span. Each then ends in
  `filler tap filler` on the left and `tap filler` on the right.

### Supply grid

The outer filler at each row end carries an M3 strap for each supply, VSS
outside and VDD a track in. A V1, an M2 pad and a V2 tie each strap to every
rail of its net, so every row's VDD and VSS are one net each.

The VSS straps run on down across the gap to an M2 bar along the array's top
edge. `build_rom` lands that bar on every tile's VSS column with a V1, so the
array's sources and taps share the decoder's VSS. The straps are labelled
`VDD` and `VSS` on M3.

The rails carry current from both ends only. That is enough for a decoder
that fires one wordline at a time, but a long row's IR drop is not checked.

gdscheck reports no violations on the ROM with its decoder:

- 1 to 3 fins;
- 16 to 256 words;
- an interleave of 8 or 16;
- RVT, LVT and SLVT tiles.

`test_each_wordline_is_the_and_of_its_lines` extracts the nets. It checks that:

- each `WL<w>` is driven by one inverter, between `VDD` and `VSS`;
- the pull-ups on that inverter's input are gated by exactly `w`'s `PA` and
  `PB` lines;
- every pin, `VDD` and `VSS` included, is one net, and no two pins short.

```bash
uv run asap7-rom --array.words 64 --array.bits 8 --out build/rom.gds
```

## Status

`build_rom_array` builds the strap rows and the tap row, then closes the
array's edges: dummy gates, ACTIVE overhangs and select. gdscheck reports no
violations on it:

- 1 to 3 fins, with no bits, every bit or a checkerboard programmed;
- LVT and SLVT tiles;
- arrays up to 128 x 33.

`test_straps_wire_every_gate` extracts the nets and checks that:

- each word transistor's gate is on its `WL<w>`;
- every source, every `ISO` gate and every tap tie is on `VSS`;
- a drain is on `BL<b>` exactly when its bit is programmed.

It needs gdstk, which isn't a project dependency, and skips without it.

Still to do, in the periphery:

- The predecoders that drive the `PA` and `PB` rails.
- Sizing the driver to the wordline load with `size_decoder`. The cells are
  fixed at 3 + 3 fins for now.
- Bitline precharge and sensing on the rows' ends.
- KLayout LVS against `RomBitcellSpec.netlist`.
