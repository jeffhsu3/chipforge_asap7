# 8T 1R1W SRAM

`chipforge_asap7.sram1r1w` is the start of a one-read, one-write SRAM
compiler. It follows the structure of GT2N's rd8t macro
(`chipforge_gt2n/sram/rd8t_*`). Its cell is OpenFinRAM's ASAP7 8T cell
`sram_cell_8t`, a true dual-port cell of 108 x 594 nm. Port A of the cell is
the write port and port B is the read port. This first milestone is one data
bit's column: the cell, its array, a write IO and a read IO. It is DRC
checked, LVS checked and simulated. The second is the macro's periphery:
decoders, wordline driver strips and a one-clock controller, as a netlist
simulated with the columns, and the strips as layout. The macro tile, and
the LEF and Liberty views, come next.

```
write IO | edge cap | array: N wordlines x 4 mux rows, a tap every gcd(N,16) | read IO
(port A, mirrored)                                                             (port B)
```

## The cell, from OpenFinRAM

OpenFinRAM produces the 8T family as build output: the cell, its
mirrored-slot variants, the edge caps, the column-group filler and the
two-slot tap. Git keeps only their geometry digests
(`tech/gds/digests.json`). `load_cell_library()` reads `sram_cell_8t`,
`sram_cell_8t_edges` and `sram_cell_8t_tap` from `$OPENFINRAM_ROOT/tech/gds`,
or from an OpenFinRAM checkout beside this repository. It refuses any library
whose digest is not the recorded one, because a regenerated cell would
silently change what the column is built of.

The cell's netlist (OpenFinRAM `tech/spice/sram_cell_8t.sp`) uses SRAM-Vt
devices: two-fin pull-downs, one-fin pull-ups, and two-fin access
transistors on both ports.

## Array

`build_array(library, cells, wordlines)` tiles the cell as OpenFinRAM's
wordline arrays do. A row is one bitline set (`BLA`/`BLAN`, `BLB`/`BLBN`)
through one cell per wordline, 108 nm apart, with every other cell mirrored
in x. Four rows, every other one mirrored in y, form one bit's 4:1 group,
2376 nm tall. A two-slot tap goes after every `gcd(wordlines, 16)` cells.
Untapped arrays of 2, 32 and 64 wordlines have polygons and labels identical
to OpenFinRAM's.

## IO: one block per port, cut to its role

Each port's IO is `IoColumnSpec` at the cell's bitline heights,
`verification.sram.io_block_specs`, as OpenFinRAM's 2RW column builds it:
- port A uses M2 bitlines and is mirrored at the left;
- port B uses M4 bitlines and sits at the right.

Each block's `role` keeps only what its port needs:

| block | role | keeps | width |
|---|---|---|---|
| write (A) | `write` | 4:1 mux, precharge, write driver | 1404 nm |
| read (B) | `read` | 4:1 mux, precharge, sense amplifier (one phase), output latch | 1620 nm |

`build_column(cells, ColumnSpec(wordlines))` assembles the column in the same
way as OpenFinRAM's `build_colgrp`:
- Port A's M2 bitlines are strapped across the metal-free filler.
- Port B meets the array's last tap.
- Each block's rail LIG stops 14 nm short of its array-facing edge.
- Supplies are tied at both seams.

The column's pins:

| pin | |
|---|---|
| `WWL[i]`, `RWL[i]` | write and read wordline of slot `i` (M3, M5) |
| `wysel[k]`, `wyseln[k]` | write mux select of row `k`, and its complement |
| `wprechn` | write bitline precharge, active low |
| `wrena`, `wrenan` | write driver enable, and its complement |
| `D` | the bit written |
| `rysel[k]`, `ryseln[k]` | read mux select of row `k`, and its complement |
| `rprechn` | read bitline precharge, active low |
| `sae` | sense enable; low resets the amplifier |
| `oe`, `oeb` | output enable for `Q`, and its complement |
| `Q` | the bit read |

`column_netlist` is the column's CDL: the cell, both blocks
(`block_netlist`) and one instance of each.

## Verification

The column with 32 wordlines (7128 x 2376 nm):

- **DRC** (gdscheck v0.2.1). The only findings are the IO cells' known ones:
  - `M1.S.2` (103) and `M2.S.2` (93): tip-to-side spacing that the 297 nm
    rows have no room for until they are redrawn taller.
  - `M4.AUX.1/2` (25): the M4 grid.
  - 8 `GATE.S.3`: tap gates that end alone on the column's open top and
    bottom edges. The dummy end rows of a macro tile close these, as
    OpenFinRAM's `build_leaf` does with its dummy rows and corners.
- **LVS**: each role block matches its flat unit-fin reference.
  - Whole columns match their CDL in hierarchical LVS, with the bitcells,
    taps and edge cells flattened:
    - 4 wordlines, in 12 s; this case is in `tests/test_sram1r1w.py`;
    - 8 wordlines, in 4 min;
    - 8 wordlines with a tap in the middle of the row (`tap_pitch=4`), in
      15 min.
  - KLayout's compare slows steeply as identical cells are added, so a run
    at 16 wordlines takes longer than 5 min, and one at 32 wordlines
    reaches the 1-hour LVS timeout without a result. A macro will need the bitcell
    compared as a unit.
- **Bench** (`python -m chipforge_asap7.sram1r1w.column_bench`). This is an
  ngspice transient on the column's CDL with BSIM-CMG OSDI devices and a
  lumped wire capacitance per bitline.
  - Controls are driven directly, one access per port per cycle on a 2 ns
    cycle: precharge, then both wordlines, `sae` 300 ps later.
  - Each cell starts at `(i + k) % 2`.
  - The sequence:
    - writes both values;
    - reads both values;
    - reads a cell written the cycle before;
    - in one cycle, writes one word and reads another on the same wordline,
      so the read word's cell has both access pairs on;
    - checks every cell that no write targets throughout.

| corner | write, WWL to flip | SAE to Q | bitline swing at SAE | worst hold margin |
|---|---|---|---|---|
| typical (TT, 0.70 V, 25 C) | 47–51 ps | 40–45 ps | 718 mV | 530 mV |
| fast (FF, 0.77 V, −40 C) | 37–39 ps | 32–36 ps | 788 mV | 566 mV |
| slow (SS, 0.63 V, 125 C) | 63–69 ps | 53–59 ps | 640 mV | 484 mV |

The hold margin is `q - qb`, signed by the stored value, and is the smallest
seen at any time in any cell that no write targeted. In every corner the
worst case is the cell read in the cycle that also writes its wordline, with
both of its access pairs on. A 300 ps wordline swings the read bitline
almost fully. With the 4:1 mux, a read and a write may therefore share a
wordline in one cycle without disturbing either word. Only a read of the
word being written returns X.

## The macro's periphery: one clock, two self-timed ports

`MacroSpec(wordlines, bits)` (`sram1r1w/periphery.py`) is a macro of
`4 * wordlines` words by `bits`. Its pins:

| pin | |
|---|---|
| `CLK` | the one clock; each port's access starts at its rise |
| `WEB`, `WADDR<k>`, `DIN<i>` | write enable (active low), address, data |
| `REB`, `RADDR<k>` | read enable (active low), address |
| `DOUT<i>` | the word read; held until the next read |

Address bits 0 and 1 choose the mux row and the rest choose the wordline
slot. A read and a write may share a cycle, on any two addresses. When both
name the same word, the read returns X.

### Floorplan: two mirrored half words

The macro is laid out as GT2N's rd8t macro is. Bits `[0, h)` sit above a
centre band and bits `[h, bits)` below it, with `h = ceil(bits / 2)`. Each
half has its own pair of wordline strips on its own wordlines (`WWLU`/`RWLU`
and `WWLL`/`RWLL`), so each driver sees half the word's load. Everything else
is in the centre band, once for both halves:

```
── upper half: columns of bits [0, h) ──
   WWLU / RWLU strips
   PS rank: one gate per slice, shared by both halves
   RWLL / WWLL strips (mirrored)
── lower half: columns of bits [h, bits) ──
```

The latches, controllers and select rows sit beside the rank in the band.
Their controls (`wprechn`, `rprechn`, `sae`, `ysel`, ...) run on M3 through
both halves.

### Clocking

Per port, the band holds four rows of custom gates (see [Gate rows](#gate-rows)):

| row | write port | read port |
|---|---|---|
| enable latch, on `CLK` | `WEB`; its `Y` node is `WE` | `REB`; its `Y` node is `RE` |
| controller | `control_row("W", write_delay)` | `control_row("R", read_delay)` |
| latches, on the port's `PRECHN` | `WADDR`, `DIN` | `RADDR` |
| select row | column select and `B` lines | the same, plus the replica's `RB` |

The enable latch is open while `CLK` is low. The controller is the NOR ROM's
edge-triggered controller (`chipforge_asap7.devices.rom_control`), with the
port's clock gate as its first gate. It latches a request at the rise, raises
three phases in order, and lowers them in reverse once its `DONE` falls:

| controller output | write port | read port |
|---|---|---|
| `PRECHN` | `wprechn`: bitline precharge off | `rprechn`: bitline precharge off |
| `EVAL` | `wrena`: write driver on | `REN`: read wordline enable |
| `SAMPLE` | `WEN`: write wordline enable | `sae`: sense, once `READY` is high |
| `READY` | none | the replica bitline, inverted |
| `DONE` delay | `write_delay` = 21 inverters: the write pulse | `read_delay` = 7 inverters: how long `sae` stays high |

So a write turns its wordline off before its driver, and its driver off
before the precharge. A read's sense is timed by a replica:
- `replica_cells` = 2 cells hold 0 on one bitline that is loaded like a real
  one;
- a replica wordline is driven like one half's from `REN`;
- the replica bitline therefore discharges twice as fast as a real one, so
  `sae` fires once a real bitline has fallen about `VDD / 4`.

The read IO's output latch takes the word at `sae` and holds it until the
next read.

The address and `DIN` latches are the NOR ROM's
(`chipforge_asap7.devices.rom_latch`). They are open while the port's
`PRECHN` is low, and closed from just after `CLK`'s rise to the end of the
access.

### Decode

From the latched address bit `AL<k>` and its complement `ALN<k>`, each port
makes:
- the column select `ysel[k]` and its complement `yseln[k]`, from bits 0 and
  1: `NAND2 -> INV -> INV`;
- `B[j] = EN . a2 . a3`: a `NAND3`, then an odd chain of inverters that
  tapers up by at most 4 per stage. `EN` is `WEN` for the write port and
  `REN` for the read port;
- `PS[s]`, one AND of the remaining bits per slice:
  - up to three bits are ANDed directly;
  - past three, the rank ANDs groups of two or three bits that the select
    row predecodes;
- in each half's strip, `WL[4s + j] = PS[s] . B[j]`.

The enable is on the four `B` lines, not on each slice's select. So the `PS`
lines are static through an access, and only `B` carries its edge.

Every gate is sized for its load. A wordline's load is 0.181 fF per cell,
wire included (OpenFinRAM's figure), times `4 * h` cells. `size_decoder`
sizes the slices for it, and no slice device has more than 12 active fins.
Each select gate has 12 fins per bit it drives, and a `B` line drives one
slice input per slice in both halves.

### Fused gates

Compared with milestone 2 (one controller per port, behind a separate
gated-clock row), these gates are merged:

1. An address bit's complement is the latch's own `Y` node, so no address
   inverters remain.
2. The column select is `NAND2 -> INV -> INV`: the second inverter's input is
   the complement, so there is no separate complement inverter.
3. The wordline enable is on the four `B` lines, not on every slice's `SEL`.
   This removes one gate per slice per port.
4. `CLKB = NAND(CLK, WE)` is the controller's first gate, replacing the
   separate clock gate and its inverter.
5. The write driver's `wrenan` is the write controller's `EVB` node.
6. The `PS` rank is shared by the two halves.
7. The enable latch sits in the controller's row.

The controller also loses its reset: `REQ = NOR(DONE, REQB)`. The write
controller's `SN` is a `NAND2`, because its `READY` is always high.

At 32 wordlines by 2 bits, the small logic falls from about 223 gates to 151
(−32%). Without the `B` lines' tapered chains it would be −41%. The device
count goes up, from 705 to 807, because the gates are now sized for their
loads. Milestone 2's buffers had a fixed 4–8 fins, and its `B` line was a
4-fin inverter driving 96 fins. The exception is the `PS` inverters: a
slice's pitch leaves room for one finger. A `PS` line follows the address,
not the clock, so a slow one adds to the address setup, not to the access.

`macro_cdl(spec, cells)` is the macro's netlist:
- flat outside the columns: one transistor per device on the macro's nets,
  2449 devices at 32 wordlines by 2 bits, counting both halves' strips;
- one `column_netlist` instance per bit, whose wordlines are its half's.

### Gate rows

`GateRowSpec` (`chipforge_asap7.devices.gate_row`) generalises the NOR ROM's
controller row to any set of gates, input pins and outputs:
- each gate (`inv`, `nand2`, `nor2`, `nand3`, `nor3`) is one diffusion island
  on one dense row, with a gap column after it;
- its inputs are poly stripes, its output an M3 column;
- every net with two or more terminals gets an M2 track, coloured into seven
  levels 36 nm apart.

Given the NOR ROM controller's gates, it draws exactly
`build_rom_control_row`'s polygons, labels and devices.

A track is at least 40 nm long, and a net with only one terminal has no
track. ASAP7's `M2.S.2` needs 25 nm between a short edge (≤ 36.25 nm) and a
long one. A stub's edges are all short, so it fails against the next level's
track, only 36 nm away.

### Wordline strips

`build_strip_pair(cells, column, slice)` (`sram1r1w/strips.py`) lays out
both ports' drivers. It follows OpenFinRAM's `build_wordline_strips`: two
strips on a shared VSS rail, abutted below the column stack.

```
── column stack: WWL[i] on M3, RWL[i] on M5 ──
   port A strip: drivers, then NANDs       WWL on M3 straight up
══ shared VSS rail ══
   port B strip, flipped: NANDs, then drivers
   RWL: V3 + V4 on the driver's output, M5 straight up
```

Each strip has one slice per four wordlines, on the array's 108 nm pitch.
Every other slot under the array holds a filler, including the slot under
the array's last tap, so the tap's gates end on poly.

The strips' placement is constrained by DRC:
- There is no edge margin: the column's fins already sit on a slice's fin
  grid.
- Each strip has one tap, past a filler beyond the array's left end. A tap
  beside a slice breaks the tap's select enclosure, and a tap under the
  array trips `SRAM.LISD.AUX.1`. `ACTIVE.LUP.1` allows 30 um to a tap, about
  256 wordlines. A longer strip is refused until it has a second tap.
- Port B's via row is on an M4 track.

The pair takes its decode from its bottom edge, which faces the centre band:
- each strip's `B[j]` are joined across its slices on an M4 rail, as in
  OpenFinRAM's `_predecode_rails`. The rail runs on to the strip's right end,
  where pins `B_A[j]` and `B_B[j]` are;
- port B's `SEL` (the nearer strip's) is the slice's own M3, carried straight
  down to the bottom edge: pin `SEL_B[k]`;
- port A's `SEL` comes up on M5 over port B's strip and drops onto the
  slice's M3 through a V4 and a V3: pin `SEL_A[k]`. Its M5 is the slice's
  middle free track, or, for a slice past a tap gap, its right-edge track.

Every M4 and M5 shape is on a routing track (`M4.AUX.2`, `M5.AUX.2`): 12 nm
past a multiple of 48 nm, in the column's frame.

### The `PS` rank and the wordline core

`build_wordline_core(spec, cells)` (`sram1r1w/band.py`) is the middle of
the macro:

```
── upper half's columns ──
   upper strip pair (bottom edge: the slice's top rail)
   write rank row: NAND3, gap, INV per slice; a tap at the right end
   read rank row: the same, sharing the other rail with the write row
   lower strip pair, mirrored
── lower half's columns, mirrored ──
```

Each rank row is a `GateRowSpec` with its gates placed (`at`) under their
slices. A slice is 8 gate columns: a NAND3 (4), a gap, a one-finger inverter
(2) and a gap. Each `PS` line reaches its slice in both halves:
- the read row's inverter output column is the slice's `SEL` x. It runs on
  M3 the band's full height, through the write row's gap column, straight
  into both pairs' `SEL_B`;
- the write row's `PS` goes up a V3 to an M4 jog, then a V4 to the slice's
  `SEL_A` M5 track, which runs the band's full height into both pairs.

The address literals come in on each row's M2 tracks from its right edge
(`GateRowSpec.right`). Six literals and the `PSN` nets fill a row's seven
M2 levels, so the rank's layout takes up to three address bits: 32
wordlines, 128 words.

Each row's outer rail meets its pair's bottom rail, which is the slice's top
rail. That is VDD for slices up to `inv9n9p` (32x16) and VSS for the bigger
ones. With VDD outside, the write row stands upright and the read row is
turned over; with VSS it is the other way round.

The band's height and where it sits come from the pair's height P, so that
the lower pair, the upper one mirrored about the band, keeps its M4 on the
same tracks (12 nm past a multiple of 48 nm):

- `rank_bands(P)` takes the smallest bands, whole fin pitches from 243 nm,
  whose band height h is −2P − 24 modulo 48 nm.
- `band_foot(P)` puts the band's foot on the fin grid and at P + 24 modulo
  48 nm.

| slice | pair | bands | band | foot |
|---|---|---|---|---|
| `nand4n2p_inv2n2p`, `nand4n2p_inv4n4p` | 1512 / 1728 nm | 270 nm | 1080 nm | 0 / 216 nm |
| `nand4n2p_inv9n9p` and up | 2268 nm and up | 324 nm | 1296 nm | 324 nm |

The pair drops an M1 stub onto its bottom rail every 108 nm. A rank gate
whose supply contact on that rail lands under a stub fails V0.M1.AUX.3. Each
`PS` inverter is placed clear of the stubs, and each NAND3 takes the first
fit of four options: shifted 0 or 1 columns, mirrored or not.

Each rank row ends in a 2 CPP tap (`GateRowSpec(tap=True)`) that ties both
of its bands. Where the edge is VSS, the rows' n-wells meet only each other
in the band's middle, and without the tap they fail ACTIVE.LUP.1.

### Verification

- **Rows**: each port's controller, select row, rank row and latch rows, as
  `MacroSpec()` sizes them, plus the select rows at 32 bits:
  - each is DRC clean when terminated;
  - each matches its reference netlist in LVS.
- **Wordline core** at 32 wordlines by 2 bits:
  - LVS matches its netlist (both strip pairs and both rank rows);
  - with a column above and a mirrored column below, the core adds no DRC
    finding.
- **Decoders**: a switch-level test
  (`test_each_decoder_selects_one_row_and_one_wordline`) evaluates each
  port's decode at 4 to 128 wordlines. For every sampled address it checks
  that one `ysel` is high and that each half has exactly one wordline on,
  and that no wordline is on without the enable.
- **Strips** at 32 wordlines (8 slices per port, 1.512 um tall):
  - every `WWL[i]` and `RWL[i]` is on the column's pin, layer and x;
  - every `SEL` pin is on the bottom edge, and every `B` rail on a track at
    the right end;
  - LVS matches `strip_pair_netlist`;
  - abutted under a column, the pair adds no DRC finding. It also closes the
    `GATE.S.3` findings that the column's taps had on its bottom edge.
- **Bench** (`python -m chipforge_asap7.sram1r1w.macro_bench`). This is an
  ngspice transient on `macro_cdl` for 128 words by 2 bits, with the same
  models and wire capacitance as the column bench. It drives only the macro's
  pins, over 11 cycles:
  - writes and reads of both values, some in the same cycle;
  - a read and a write on one wordline in one cycle;
  - a read of the word being written, which must return X;
  - an idle cycle.

  Every read is checked against the model, every written cell is checked
  after its write, and every cell is checked at the end. A full run takes
  about 40 min at a 2 ns period, so run the decoder test first.

  Every check passes at each corner, with the fused netlist (milestone 3):

  | corner | period | CLK to `DOUT` | write margin | milestone 2 |
  |---|---|---|---|---|
  | typical (TT, 0.70 V, 25 C) | 0.5 ns | 208–211 ps | 69–70 ps | 200–204 ps, 69–71 ps |
  | fast (FF, 0.77 V, −40 C) | 1 ns | 155–159 ps | 52–53 ps | 150–153 ps, 54–55 ps |
  | slow (SS, 0.63 V, 125 C) | 1 ns | 294–299 ps | 96–97 ps | 283–287 ps, 94–96 ps |

  The write margin is how long the written cell has flipped before its
  wordline falls. Reads are 5–12 ps slower than milestone 2's: the select
  is now NAND2 → INV → INV, and the `PS` rank has one finger. At typical,
  milestone 2's control ran, after `CLK` rises:

  | after CLK rises | write port | read port |
  |---|---|---|
  | precharge off | 86 ps | 86 ps |
  | wordline on (`WEN` / `REN`) | 132 ps | 109 ps |
  | `sae` | | 166 ps |
  | wordline off / `sae` low | 215 ps | 208 ps |
  | precharge on again | 259 ps | 254 ps |

  Both ports are idle again about 260 ps after the rise, so typical passes
  at 0.5 ns with the same timing as at 2 ns. The minimum period has not been
  searched for.

  The bench sets each cell's starting state on `qb` only, with an `.ic`, and
  each output latch's on one node. If both of a cell's nodes are held, the
  fast corner's operating point does not converge in two hours; if both of
  the output latch's are, it fails outright. The bench also runs with
  `chgtol=1e-16 trtol=3`: with ngspice's defaults, the typical corner's
  timestep collapses where a write and a read switch at once.

## Next

- Lay out the rest of the macro tile:
  - the band's control end, under the IO blocks: the latches, controllers
    and select rows (split to fit seven M2 levels), their M3 through-lines
    into both halves' IO, and the risers that join each half's `B` rails;
  - a rank for more than three address bits (more M2 levels, or M4 lines);
  - each half's columns, stacked and mirrored; the tile's dummy end rows
    close the columns' `GATE.S.3` findings;
  - then the macro's DRC and LVS, LEF and Liberty.
- Bank very wide words: at 1024 wordlines by 148 bits a slice's NAND needs
  40 fins.
- Redraw the IO cells on taller rows for gdscheck v0.2.1's `M1.S.2`/`M2.S.2`.
