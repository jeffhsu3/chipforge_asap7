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
  first and is held at VSS. The drains share a bitline only when both bits
  are programmed; off-state leakage still needs electrical characterization.
- **Diffusion:** it is never broken. A break between different nets costs
  92 nm (ACTIVE.S.2A).
- **Row height:** each row is `fins + 1` fin pitches. The empty pitch is the
  27 nm that ACTIVE.S.1 requires between rows.

| fins | bit (nm) | area (nm²) |
| ---: | --- | ---: |
| 1 | 81 x 54 | 4374 |
| 2 | 81 x 81 | 6561 |
| 3 | 81 x 108 | 8748 |

## Density compared with SRAM and BROM

Measured on 2026-11-02 from the generators in this repository. Bitcell areas
are placement-boundary areas per stored bit; they exclude taps, edges and
periphery.

| Bitcell | Tile (nm) | Bits per tile | µm² per bit | Bits per µm² |
| --- | --- | ---: | ---: | ---: |
| NOR-ROM, 1 fin | 162 x 54 | 2 | 0.004374 | 228.6 |
| NOR-ROM, 2 fins (default) | 162 x 81 | 2 | 0.006561 | 152.4 |
| NOR-ROM, 3 fins | 162 x 108 | 2 | 0.008748 | 114.3 |
| BROM selector core ([proof](brom_minimum.md)) | 108 x 432 | 4 | 0.011664 | 85.7 |
| 6T SRAM `sram_cell_6t_122` ([proof](sram_6t_minimum.md)) | 108 x 270 | 1 | 0.029160 | 34.3 |
| 8T SRAM `sram_cell_8t` (dual port) | 108 x 594 | 1 | 0.064152 | 15.6 |

The two-fin NOR-ROM bit is 1.8× denser than the BROM selector core,
4.4× denser than the 6T bitcell and 9.8× denser than the 8T bitcell. With
its strap row every 16 bit rows, the tap row and the closed edges, a
two-fin NOR-ROM array still stores 136.7 bits/µm² (0.007313 µm² per bit at
64 × 32 and 256 × 32).

Larger blocks do not compare on equal terms yet:

| Words × bits | NOR-ROM array + decoder envelope | Bits per µm² | BROM complete macro, mux4 | Bits per µm² |
| --- | ---: | ---: | ---: | ---: |
| 16 × 4 | — | — | 16.416 × 2.339 µm, 38.397 µm² | 1.67 |
| 64 × 16 | — | — | 21.600 × 4.499 µm, 97.178 µm² | 10.54 |
| 64 × 32 | 57.355 µm² | 35.71 | 32.016 × 8.387 µm, 268.518 µm² | 7.63 |
| 256 × 32 | 212.336 µm² | 38.58 | 36.336 × 12.707 µm, 461.722 µm² | 17.74 |
| 1024 × 16 | — | — | 38.496 × 36.899 µm, 1,420.464 µm² | 11.53 |
| 1024 × 148 | — | — | 175.872 × 60.227 µm, 10,592.243 µm² | 14.31 |

The NOR-ROM figures are the selected shapes from
[the shape search](#shape-search-and-routing-aware-packing): array, wordline
decoder and predecoders only. They have no precharge, sensing, column select
or output latch, which the NOR-ROM does not draw yet. The BROM figures are the
bounding box of a complete two-fin RVT macro, including its CGEN, address
latches, precharge, sampling and output latches. On these sizes the
periphery, not the bitcell, sets BROM density. Its 64 × 32 macro is less
dense than 64 × 16 (not yet investigated), and 1024 × 16 uses the
electrical-order fallback placement described in [brom.md](brom.md).
The comparison will be fair only once the NOR-ROM has its read path.

## Read electrically: 1 fin against 2

`asap7-rom-read` (`chipforge_asap7.devices.rom_read_spice`) simulates one
bit row's read in ngspice, with every word a BSIM-CMG 107 transistor. Two
bitlines share one precharge and one wordline pulse:

- **`BLP` reads a programmed bit.** The other `words - 1` words are
  programmed too, so each is an off transistor on the line. That is the most
  drain capacitance the selected word can have to discharge.
- **`BLU` reads an unprogrammed bit beside the same words.** Only their
  off-state leakage pulls it down: the most a line that should stay high can
  droop.

The bitline is a distributed M2 RC ladder, one 81 nm segment per word. It
uses the stack's 3.03 Ω/sq over the 18 nm track (168 Ω/µm) and an assumed
0.2 fF/µm. The selected word is the far one, read through the whole line;
precharge (3-fin PMOS) and sensing are at word 0. Releasing the precharge
kicks both floating lines up to 34 mV above VDD through the PMOS gate
overlap, so the margin is counted from each line's level as the wordline
starts to rise. Times run from the wordline's 50 % point to:

- `t_sense`, when `BLP` has fallen 100 mV (a sense-amplifier read);
- `t_half`, when `BLP` crosses VDD/2 (an inverter read).

`split` is `BLU` minus `BLP` at `t_sense`. A depth fails when leakage takes
more than half of either margin.

```bash
uv run asap7-rom-read --fins 1 2 --words 16 64 256 1024 4096 --temperature-c 125
```

TT model card, RVT, 0.7 V, 40 ps wordline slew; times in ps, levels in mV:

| Fins | Words per bitline | Wire R (kΩ) | `t_sense` 25 °C | `t_half` 25 °C | `t_sense` 125 °C | `t_half` 125 °C | `split` 125 °C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 0.2 | 6.6 | 14.0 | 5.2 | 13.3 | 100 |
| 1 | 64 | 0.9 | 14.0 | 28.3 | 13.2 | 28.4 | 99 |
| 1 | 256 | 3.5 | 33.1 | 84.6 | 33.0 | 86.8 | 98 |
| 1 | 1024 | 14.0 | 150.7 | 375.6 | 150.0 | 375.5 | 94 |
| 1 | 4096 | 55.9 | 1245.5 | 2997.3 | 900.2 | 2278.3 | **38** |
| 2 | 16 | 0.2 | 5.2 | 11.5 | 3.7 | 10.5 | 100 |
| 2 | 64 | 0.9 | 11.7 | 23.0 | 10.8 | 22.8 | 99 |
| 2 | 256 | 3.5 | 29.1 | 66.3 | 28.8 | 67.3 | 98 |
| 2 | 1024 | 14.0 | 150.4 | 349.3 | 148.0 | 340.0 | 91 |
| 2 | 4096 | 55.9 | 1519.9 | 3716.3 | 780.9 | 2208.1 | **11** |

Every depth to 1024 words passes at both temperatures, for both fin counts.
At 25 °C 4096 words pass too; at 125 °C both fail the sense margin, and
2 fins fails the inverter read as well. Findings:

- **Within the checked range, one fin costs little.** The ROM is
  DRC-checked up to 512 words with the center lane, which splits each
  bitline into two halves of 256 words. Nothing in the generator stops a
  longer line. At 256 words, one fin is 14 % slower to sense
  (33.1 against 29.1 ps) and 28 % slower to an inverter read (84.6 against
  66.3 ps), for one-third less core area.
- **Long lines are wire-limited, and there the extra fin hurts.** By 1024
  words the 14 kΩ line, not the cell, sets the sense time; both fin counts
  take 150 ps. At 4096 words one fin is faster (1.25 ns against 1.52 ns at
  25 °C), because a second fin doubles every word's drain capacitance.
- **Leakage is not the limit within the checked range.** At 125 °C,
  1024 words keep 94 mV (1 fin) and 91 mV (2 fins) of the 100 mV split.
  With twice the leaking fins, 2 fins droops more. At 4096 words the
  leakage also helps discharge `BLP`, which is why that read is faster hot.

What the bench does not cover:

- the poly wordline's RC, strapped to M3 every 16 bit rows; the wordline is
  an ideal 40 ps edge;
- the V1/M1 stubs and the grounded isolation gates;
- any process corner beyond TT, and mismatch. A single fin has no averaging,
  so its read current varies more than two fins' does;
- the precharge, sense and output circuits the NOR-ROM does not draw yet.
  Their timing, and a cycle time, are not established.

So one fin is electrically feasible at nominal conditions, at the depths the
ROM is checked to. It is not yet qualified across corners or with the wordline RC.

## Minimum-area proof for this topology

`chipforge_asap7.verification.rom_minimum` proves the minimum periodic core
area **at a fixed fin count, for the existing topology**. It does not claim
optimality over other NOR-ROM architectures or the minimum area of a complete
macro. The two-fin default is not the smallest supported geometry: one fin
uses one-third less core area, subject to the required read current and timing
(see [the read bench](#read-electrically-1-fin-against-2)).

The proof fixes a rectangular repeating tile with two word transistors sharing
a grounded source, two separately V1-programmed drains, and a grounded isolation
gate between tiles. Diffusion is continuous along X; different bit rows have
separate ACTIVE strips. Ordinary ASAP7 rules apply, without an SRAMDRC marker.

The lower bound uses these necessary rules from gdscheck:

| Rule | Constraint used |
| --- | --- |
| `GATE.W.1`, `GATE.S.2` | 20 nm gate width plus 34 nm edge spacing |
| `FIN.W.1`, `FIN.S.1` | 7 nm fin width; distinct parallel fins at least 27 nm apart |
| `ACTIVE.FIN.EX.1` | ACTIVE encloses the outer fins by 10 nm at both ends |
| `ACTIVE.S.1` | Adjacent bit rows have at least 27 nm of ACTIVE separation |

Let the three gate centers be `a`, `b`, `i`, and the repeating tile width be
`W`. The three spacing inequalities, **including the next tile**, are
`b-a >= 54`, `i-b >= 54`, and `a+W-i >= 54`. Adding them gives `W >= 162` nm.
For `f` fins, the ACTIVE height is at least `(f-1)*27 + 7 + 2*10 = 27*f` nm;
the inter-row separation then gives `H >= 27*(f+1)` nm. With two bits per
tile, the core area per bit satisfies `A_bit = W*H/2 >= 2187*(f+1)` nm².

The SMT-LIB model uses real coordinates and minimum fin spacing, relaxing
grid membership and omitting contact and routing constraints (those are
checked separately, below). Every permitted
tile must satisfy these necessary inequalities; allowing more layouts makes
an UNSAT result a valid lower bound. Neither derived dimension bound is
inserted into the base model. Z3 checks that `W < 162`, `H < 27*(f+1)`, and
their disjunction are each UNSAT. The dimension bounds and positive dimensions
give the area bound by multiplication, without nonlinear SMT arithmetic.

### Routing constraints

A second block of assertions models the tile's contacts and wiring, using the
pinned M1, M2, V0, V1 and LISD rules. It extends the topology with these
routing assumptions:

- Each source/drain column has one LISD bar. The bar spans the row's ACTIVE,
  stays clear of both gates, and carries one 18 nm V0.
- The shared source reaches VSS through one straight M1 column. The column is
  full height and continuous across rows.
- Each drain lands on its own M1 stub, which may be horizontal or vertical.
  Its programming V1 sits on that stub.
- Both V1s meet one straight, full-width M2 bitline per row.

| Rule | Constraint used |
| --- | --- |
| `LISD.W.1`, `LISD.S.1`-`S.3` | 24 nm bars; spacing by facing-edge class, across X and to the next row |
| `V0.LISD.EN.2` | 3 nm of LISD on both sides of one opposite pair |
| `V0.M1.EN.1`, `V0.M1.AUX.3` | a 5 nm end-cap on some side; M1 flush with the V0 on one opposite pair |
| `M1.A.1` | 504 nm²; with one flush pair this is linear: an 18 nm stub is at least 28 nm long |
| `M1.S.1`-`S.5` | spacing by the facing edges' lengths: long (≥ 36.25 nm), `mid` (24-36 nm) or `tiny` |
| `V1.M1.EN.1` | 5 nm on one side and 2 nm on the other, on one opposite pair |
| `V1.M2.AUX.2`, `M2.S.1` | the bitline is flush with the V1s across Y; rows of bitlines 18 nm apart |
| `V0.S.1`, `V1.S.1` | 18 nm between vias |

Corner-to-corner spacing (`M1.S.6`) and the conditional V0/V1 spacings are
omitted. Omitting rules only enlarges the modeled space, so the model stays a
relaxation. The X order of the columns is fixed by the topology. Each shape
is checked against the next tile's copy at `+ width` and the next row's copy
at `+ height`. Facing spacings apply wherever the two Y spans overlap.

The area bound is still proved on the FEOL model alone, so the UNSAT results
need none of the routing assumptions. The routing block adds the
`routed_drawn_witness` query: every drawn routing coordinate is fixed, and the
query must be SAT. So the drawn wiring fits the minimum tile under the encoded
rules. In it, the drain stubs stand exactly 26 nm off the VSS column (M1.S.2
asks 25) and 36 nm from the next tile's stub (M1.S.4 asks 31).

The routing rules also hold the row pitch on their own. Without ACTIVE.S.1,
a tile no wider than 162 nm still cannot have `H < 27*(f+1)`. Gates 54 nm
apart leave an LISD bar at most 34 nm wide, so its ends are `mid` edges and
the bars of adjacent rows need LISD.S.3's 27 nm. A wider tile admits a
long-edged bar and an 18 nm gap. The tests keep each routing rule honest:
each one rejects a drawn witness broken in its own way, and accepts the same
geometry once that rule is removed. The broken witnesses are:

- a stub 24 nm from the column;
- stub tips 29 nm apart;
- the old full-ACTIVE-height drain pad;
- a 27 nm stub;
- 2 nm of LISD around the V0.

A separate SAT query fixes all coordinates to the generator's **drawn
polygons**. The runner requires this witness to attain both bounds, writes a
four-word array containing all 16 programming patterns on separate bitlines,
runs the full pinned gdscheck `main` suite, and extracts the GDS. Extraction
checks the transistor inventory and fin counts, grounded sources and isolation
gates, programmed connections to the correct bitline, and independently floating
unprogrammed drains. An ideal-switch connectivity model also exhausts all
16 programming patterns and 16 wordline states of two periodic tiles.

The negative controls matter: removing the gate wraparound, ACTIVE separation,
fin enclosure or fin-spacing constraint lets a smaller model become SAT.
Removing isolation from the switch model produces a programming counterexample:
with only word 0 programmed and only word 3 selected, the merged drains
across the periodic boundary incorrectly discharge the bitline. An extracted
GDS with a deliberately missing programming via must also fail. These controls
guard against proving a contradictory model or assuming the desired minimum
as an input constraint.

To reproduce, install Z3's executable on `PATH` and use a gdscheck checkout
whose ASAP7 sources match revision
`0b383c3b07edd6247586aca812e392928dc3b872`. The Python runner needs `gdspy` and
`gdstk`; the following supplies `gdstk` for the invocation:

```bash
uv run --with gdstk python -m chipforge_asap7.verification.rom_minimum \
  --gdscheck-root ../gdscheck --fins 1 2 3 --out build/rom_minimum
```

`--z3` and `--gdscheck` select explicit executables. The checked-in
`verification/rom_minimum_rules.json` records the audited rule facts, source
revision and SHA-256 hashes of all 35 ASAP7 PDK files. The runner rejects changed
rule sources and passes their explicit `pdk.yml` to gdscheck, so an embedded or
environment-selected PDK cannot silently replace the pinned rules.

Each `fN/` output directory contains the SMT queries, Z3 proof terms and witness
model, the GDS and DRC report, and `report.json`. The report records assumptions,
limits, rule hashes, implementation hashes, executable versions and hashes,
extraction results, and the attained bound. A failed rerun replaces the success
status; missing tools and inconclusive solver results never count as proof.

The trusted boundary includes the rule transcription, topology-to-constraint
mapping, Z3, and the geometry/connectivity tools. Saved Z3 proof terms are not
independently checked by another proof checker. The finite GDS witness does not
certify every array size or programmed image. Leakage, timing, PVT, noise margins,
strap intervals, taps, array edges and periphery remain outside the minimum-area
claim.

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

array = RomArraySpec(words=64, bits=8)
decoder = RomDecoderSpec(array=array)   # interleave=8, (3, 3)-fin cells
decoder.pa_bits, decoder.pair_rails     # 4, 8: PA takes 4 of the 6 address bits
decoder.lines(21)                       # ('PA<5>', 'PB<1>')
rom = build_rom(decoder, programmed={(0, 0), (21, 3)})   # pins A<0>..A<5>
decoder.half.words, decoder.right_x     # 32, 6156: words 32..63 sit right of the lane
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
- **Joins:** each `PB` line has a copy over every pair. One M3 per line, in
  the center lane, joins the copies.
- **Row ends:** each pair's slots stand two words right of the pair below's,
  so the rows are padded with filler to one common span. Each then ends in
  `filler tap filler` on the left and `tap filler` on the right.

### Center lane

The words grow along X, so a deeper ROM makes the bitlines and the decoder
rows longer, not the wordlines. `lane=True`, the default, splits the array
in two with a lane down the middle:

```text
        decoder rows      |  joins  predecoders  joins  |     decoder rows
  [ends][slots, words     |                             | slots, words     ][ends]
          0 .. N/2-1      |                             |   N/2 .. N-1
  [ left half array    ]==BLL<b>                 BLR<b>==[ right half array   ]
```

- **Halves:** `half` is the array of `N/2` words that stands on each side.
  Each half is its own cell, with its own edges, strap rows and tap row. The
  right half stands at `right_x`, a whole number of tiles along, so its slots
  are the left half's slots, shifted.
- **Bitlines:** each half has its own bitlines. The left half's `BLL<b>`
  leave its right edge and the right half's `BLR<b>` its left edge, both
  into the lane. A bitline therefore spans half the words: 41 um instead of
  83 at 1024 words. The top address bit picks the half, so `BLL<b>` carries
  words `0..N/2-1` and `BLR<b>` the rest. The 2:1 select and the sensing
  that will sit in the lane are not drawn yet.
- **Lane:** the lane holds the predecoders and the join columns. The address
  pins `A<i>` are labelled there. A `PB` line used only by the left half
  (top bit 0) joins on the lane's left side, beside its words. Every other
  line joins right of the predecoders.
- **Cost:** pairs get a little shorter, because a `PB` line now spans only
  its own half. Rows get 0.5 to 0.7 um longer, for the extra slot overhangs
  at the lane's edges.

| words | lane (nm) | pair height, lane / edge (nm) | row length, lane / edge (nm) |
| ---: | ---: | ---: | ---: |
| 16 | 2430 | 594 / 594 | 4752 / 4158 |
| 64 | 3564 | 810 / 837 | 9774 / 9288 |
| 256 | 6156 | 1296 / 1350 | 27918 / 27216 |
| 1024 | 10206 | 2322 / 2430 | 94176 / 93582 |

The lane needs at least `2 * interleave` words, so that each half has a slot
in every row. `lane=False` keeps the earlier floorplan: the predecoders at
the rows' left end, the bitlines `BL<b>` leaving the array's left edge, and
`_edge` in the cell name. It is the only floorplan at 8 words.

### Predecoders

The predecoders turn the address pins `A<i>` into the `PA` and `PB` lines.
They stand in the decoder's own rows, in the center lane
(`rom_predecoder.py`):

```text
[ends][left slots][joins][predecoder columns][joins][right slots][ends]
```

- **Gates:** each gate is a slot like a word's: a NAND and its driver for an
  AND2, or a driver alone for an address bit's complement `AN<i>`. The slot
  is 486 nm, 9 CPP, since no wordline passes through it.
- **Logic:** a group of bits is decoded as a tree, its low half and high half
  meeting in one AND2 per line. `PA` splits at the row bits instead,
  `PA<r + k m> = R<r> . M<m>`, because `R<r>` is only needed in row `r`'s
  pair. A one-bit group is a buffer, an AND2 with both inputs on one literal.
- **Placement:** a `PA` gate goes to the pair whose rails carry its line.
  Every other gate goes where most of the gates it drives are, or else to
  the pair with the fewest gates. Gates fill columns of two, one per row.
- **Copies:** across from a filler, a gate's source stubs would take the
  filler's rail contacts (V0.M1.AUX.3). A pair with an odd number of gates
  therefore gets a second copy of its most loaded gate, in parallel.
- **Routing:** each pin reaches its line on an M3 stub, as in a word's slot.
  The two rows of a column use six different tracks. Stubs on neighbouring
  tracks that nearly meet end to end are stretched to run alongside instead
  (M3.S.6).
- **M4 segments:** over each pair, a line is an M4 segment from its leftmost
  V3 to its rightmost. The wordline gates' lines keep their fixed tracks, and
  the predecoders' lines take the lowest track free over their span.
- **Segment ends:** ends sit on a 48 nm grid. Tips on neighbouring tracks
  are then aligned or 48 nm apart, and segments that run alongside do so for
  at least 48 nm (M4.S.2, S.4, S.5). The packer only has to keep two tips
  from touching corner to corner (M4.S.3).
- **Pair height:** pairs grow a fin pitch at a time until every segment fits.
  At 64 words that is 810 nm instead of 594, and at 256 words 1296.
- **Joins:** a line used in more than one pair, such as `PB<j>`, a shared
  `M<m>` or a literal, has an M3 join column on the filler beside the
  predecoders. Lines whose pairs don't overlap share a column.

They need at least `2 * interleave` words, so that `PB` has a bit. At 8
words, `predecode=False` leaves the `PA` and `PB` rails as pins, as before.

| words | gates (copies) | columns | joins | pair height (nm) |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 18 (2) | 3 | 9 | 594 |
| 64 | 38 (2) | 5 | 16 | 810 |
| 256 | 68 (2) | 9 | 31 | 1296 |

### Supply grid

The outer filler at each row end carries an M3 strap for each supply, VSS
outside and VDD a track in. A V1, an M2 pad and a V2 tie each strap to every
rail of its net, so every row's VDD and VSS are one net each.

The VSS straps run on down across the gap to an M2 bar along the array's top
edge, across the lane. `build_rom` lands that bar on every tile's VSS column
in both halves with a V1, so the array's sources and taps share the
decoder's VSS. The straps are labelled
`VDD` and `VSS` on M3.

The rails carry current from both ends only. That is enough for a decoder
that fires one wordline at a time, but a long row's IR drop is not checked.

### Independent cell sizes

The four roles can now be sized independently. Counts are **(n, p) fins per
finger**, and each role also has an even `*_fingers` count:

```python
decoder = RomDecoderSpec(
    array=RomArraySpec(words=64, bits=64),
    nand_fins=(4, 2),                 # postdecode NAND, per input
    driver_fins=(8, 8),               # final wordline inverter
    predecoder_nand_fins=(6, 3),
    predecoder_driver_fins=(12, 12),   # output and address-complement inverters
)
```

An omitted fin override uses `rows`, whose default remains `((3, 3),)`.
`rows` also sets the minimum physical band heights. Every role, filler and
tap shares band heights large enough for the largest devices, keeping their
power rails and wells aligned. Different fin heights use source contacts on
the power rails through extended LISD, avoiding two nearby V0s at an abutted
source column. Folded inverters join their drains on M2 below the seam to
keep that track clear for the NAND's input. Sizes and finger counts are part
of cell identity, so differently sized decoders can share a GDS library.

### Load-based sizing

`size_rom_decoder` sizes this ROM's PA/PB tree. It is separate from
`size_decoder`, which describes the SRAM's PA/PB/PC and WLENA tree.

```python
from chipforge_asap7.devices import size_rom_decoder

base = RomDecoderSpec(array=RomArraySpec(words=64, bits=64))
sizing = size_rom_decoder(base)
print(sizing.summary())
rom = build_rom(sizing.spec, programmed={(0, 0)})

# Supply extracted/modelled loads when available:
sizing = size_rom_decoder(
    base,
    wl_load_fF=8.0,                   # TOTAL gates + wire, worst wordline
    wire_load_fF={"PB<0>": 1.2},     # WIRE ONLY; other nets are estimated
)
```

- The wordline gate load is `bits * bitcell.fins * gate_cap_per_fin`;
  unprogrammed cells still load the gate. Without a supplied total, it adds
  estimated full-height poly capacitance and the longest M3 driver-to-array
  route. The poly coefficient defaults to an assumed 0.2 fF/um and is
  adjustable through `poly_cap_fF_per_um`.
- Named line loads count the placed network's gate inputs and estimated
  M3/M4 wire lengths. Both inputs of a tied-input buffer count, as do the
  extra input loads from parallel placement copies. Each parallel copy is
  conservatively sized to drive the entire output load.
- The postdecode NAND drives the selected wordline inverter. Predecoder
  NANDs and inverters each use one common size for their role's worst load;
  these are independent of the postdecode cells. This is conservative and
  can make low-fanout predecoder gates larger than necessary.
- The sizing loop rounds up to fins, then folds into more fingers if the
  per-band fin limit is reached. It recomputes routing and wire loads until
  the sizes stop growing. It preserves `split` and `interleave`; if the
  resulting cells do not fit, it reports an error instead of clipping drive.
- The default logical-effort calibration is TT/RVT at 0.7 V. Other VT
  flavours require an explicitly supplied `LogicalEffortModel`.

`sizing.stages` reports actual total fins per input, load and effort for each
role. `line_load_fF`, `wire_load_fF`, `fanout`, `supplied_wire_nets` and
`address_input_cap_fF` expose the load accounting. Wire overrides may also
name `post_nand` and `pre_nand` for the private NAND-to-inverter connections.
`supplied_wl_load` distinguishes a supplied total from an estimate.

The estimates exclude coupling, vias and most local interconnect. They are
not PEX. The reported gate delay excludes distributed resistance; an
independently evaluated WL wire contribution can be supplied through
`wl_wire_delay_ps`. Extracted transient simulation, slew and IR-drop checks
are still needed for timing closure.

```bash
uv run asap7-rom --array.words 64 --array.bits 64 --auto-size --out build/rom.gds
uv run asap7-rom --array.words 64 --array.bits 64 --wl-load-ff 8 --out build/rom.gds
```

The original fixed-size behaviour remains the default without either sizing
option. `--stage-effort` changes the automatic sizing target (default 4).

### Shape search and routing-aware packing

`search_rom_shapes` searches the existing NOR-ROM's address split, decoder
interleave and optional strap intervals, at a fixed word count and bit width.
Each shape gets a bounded placement search. It scores the generated routing
plan, including the lane, joins and M4 tracks, rather than counting gates or
rails alone:

```python
from chipforge_asap7.devices import (
    RomArraySpec, RomDecoderSpec, search_rom_shapes, build_rom,
)

base = RomDecoderSpec(RomArraySpec(words=64, bits=32))
search = search_rom_shapes(base, interleaves=(8, 16), packing_evaluations=64)
chosen = search.best
print(chosen.metrics.area_um2, chosen.metrics.wire_length_um)
rom = build_rom(chosen.spec, programmed={(0, 0), (63, 31)})
```

The objective is the array-plus-decoder **envelope area**, with estimated
M3/M4 predecode wire length breaking ties. It includes the reserved row-pair
routing space; it is not the polygon area or the area of a complete read
macro. `search.pareto` retains area/wire-length tradeoffs, and includes the
logical-effort gate-delay estimate when automatic sizing is requested.

The packer starts from the existing placement and keeps an incumbent whose
area/wire-length cost never worsens. It swaps and reorders gates in a pair
and moves non-PA gates between pairs. PA gates remain beside their wordline
rails, and an odd gate count gets a parallel fill copy so both rows of every
column are occupied. All trials use the existing ASAP7 routing clearances
and row alignment. The default seed is zero and the budget is 64 unique
placement evaluations per shape, including the baseline and infeasible
trials. A budget of one preserves the baseline placement.
Fixed-track corner conflicts are rejected before allocating more tracks;
row-pair growth also has a finite routing bound for infeasible placements.

Without explicit choices, search tries the current interleave, 8 and 16,
retains the array's strap interval, and considers every legal address split.
An explicit `base.split` restricts the split. `splits=` and
`strap_intervals=` supply alternative choices. Unsupported shapes appear in
`search.rejected` with their reasons; requesting `search.best` when none fit
raises an error. Banking and column muxing require additional circuitry and
are not part of this search.

Device sizes are preserved by default. Pass `sizing={}` to use the existing
load-based sizer, or pass its options, for example
`sizing={"wl_load_fF": 8.0, "stage_effort": 4.0}`. Every trial is then sized
again: placement changes routed loads and the number of parallel copies.
The gate-delay estimate still excludes distributed wire resistance.

```bash
# Select a shape, pack it, and write the GDS and a JSON search report.
uv run asap7-rom --array.words 64 --array.bits 32 --search \
  --search-interleaves 8 16 --out build/rom_search.gds

# Also compare strap intervals and re-size every placement.
uv run asap7-rom --array.words 64 --array.bits 32 --search --auto-size \
  --search-straps 4 8 16 --out build/rom_sized_search.gds

# Keep the shape fixed and optimize only its predecoder placement.
uv run asap7-rom --array.words 64 --array.bits 32 --pack \
  --packing-evaluations 64 --search-seed 0 --out build/rom_packed.gds
```

The default report is beside the GDS with the suffix `.search.json`;
`--search-report` overrides it. It records the selected shape, complete spec
and placement, candidate metrics, rejected shapes and Pareto front. Generated
candidates need gdscheck and extracted electrical validation before use;
search does not certify timing, and precharge, sensing and output circuitry
are still excluded.

The reusable library boundary is deliberately small:

- `chipforge_asap7.optimization` provides seeded bounded search and Pareto
  selection using caller-supplied states and costs, without process rules.
- `devices.rom_search` supplies the ROM mutations, geometry costs and sizing
  policy; `rom_predecoder` validates placements and `rom_decoder` draws them.
- Immutable device/slot calculations and routing spans are cached so trials
  reuse their geometry while looking for a feasible row-pair height.

Other compilers can reuse the search helpers with their own constraints.
Sharing a GT2N backend would require its own adapter; its dimensions and
timing calibration are not imported into ASAP7.

With default device sizes, strap interval 16, seed zero and 64 placement
evaluations per shape, the default interleave/split search gives:

| Words × bits | Baseline envelope (µm²) | Selected envelope (µm²) | Area reduction | Predecode wire, baseline → selected (µm) | Selected split / interleave |
| --- | ---: | ---: | ---: | ---: | --- |
| 64 × 32 | 60.433 | 57.355 | 5.1% | 324.522 → 289.795 | 3+3 / 8 |
| 256 × 32 | 226.890 | 212.336 | 6.4% | 1586.210 → 1377.755 | 4+4 / 8 |

Both selected checkerboard-programmed layouts pass gdscheck with zero
violations. These are geometry improvements; they do not establish a read
delay improvement. The search tests also check extracted switch-level
decoding of every address and the programmed bitlines for 16- and 64-word
layouts.

gdscheck reports no violations on the ROM with its decoder and predecoders:

- 1 to 3 fins;
- 16 to 512 words with the center lane, and the decoder alone at 512;
- an interleave of 8 or 16;
- RVT, LVT and SLVT tiles;
- `predecode=False`, at 16 and 256 words with the lane and at 8 without;
- `lane=False` at 16 words, and a decoder sized by `size_rom_decoder`.

`test_each_wordline_is_the_and_of_its_lines` extracts the nets. It checks that:

- each `WL<w>` is driven by one inverter, between `VDD` and `VSS`;
- the pull-ups on that inverter's input are gated by exactly `w`'s `PA` and
  `PB` lines;
- every pin, `VDD` and `VSS` included, is one net, and no two pins short.

`test_each_address_raises_its_own_wordline` simulates the extracted
transistors at switch level, from the address pins. For every address,
exactly its own wordline goes high and every other one low. Only that word's
programmed bit pulls a bitline to `VSS`, and only in the word's own half;
every other bitline floats.

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

- Extracted transient validation of the sized decoder and distributed
  wordline RC; the load-based sizing model is an initial estimate.
- Bitline precharge, sensing and the `BLL`/`BLR` select, in the center lane.
- KLayout LVS against `RomBitcellSpec.netlist`.
