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

### Clocking

`WEB` and `REB` are latched while `CLK` is low. Each port's clock is
`CLK . !EB`. It drives that port's controller, which is the NOR ROM's
edge-triggered `RomControlSpec(edge=True)`
(`chipforge_asap7.devices.rom_control`). The controller latches a request at
the rise, raises three phases in order, and lowers them in reverse once its
`DONE` falls:

| controller output | write port | read port |
|---|---|---|
| `PRECHN` | `wprechn`: bitline precharge off | `rprechn`: bitline precharge off |
| `EVAL` | `wrena`: write driver on | `REN`: read wordline enable |
| `SAMPLE` | `WEN`: write wordline enable | `sae`: sense, once `READY` is high |
| `READY` | tied high | the replica bitline, inverted |
| `DONE` delay | `write_delay` = 21 inverters: the write pulse | `read_delay` = 7 inverters: how long `sae` stays high |

So a write turns its wordline off before its driver, and its driver off
before the precharge. A read's sense is timed by a replica:
- `replica_cells` = 2 cells hold 0 on one bitline that is loaded like a real
  one;
- a replica wordline is driven like a real one from `REN`;
- the replica bitline therefore discharges twice as fast as a real one, so
  `sae` fires once a real bitline has fallen about `VDD / 4`.

The read IO's output latch takes the word at `sae` and holds it until the
next read.

Each port's address latches, and the write port's `DIN` latches, are the NOR
ROM's (`chipforge_asap7.devices.rom_latch`). They are open while the port's
`PRECHN` is low, and closed from just after `CLK`'s rise to the end of the
access.

### Decode

Under its latches, each port has:
- a column select, `ysel[k]` with complement `yseln[k]`: a buffered one-hot
  decode of address bits 0 and 1;
- a wordline decoder of `DriverSliceSpec` slices, with
  `WL[4s + j] = SEL[s] . B[j]`:
  - `B[j]` decodes wordline-index bits 0 and 1;
  - `SEL[s] = EN . PS[s]`, with `PS[s]` the one-hot decode of the remaining
    bits;
  - `EN` is `WEN` for the write port and `REN` for the read port.

`size_decoder` sizes the slices for the wordline's load. That load is
0.181 fF per cell along the wordline, wire included (OpenFinRAM's figure),
times `4 * bits` cells. No device has more than 12 active fins.

`macro_cdl(spec, cells)` is the macro's netlist:
- flat outside the columns: one transistor per device on the macro's nets,
  1623 devices at 32 wordlines by 2 bits;
- one `column_netlist` instance per bit.

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

Each slice's `SEL` and `B[j]` are its own pins, `SEL_A[k]` and `B_A[k][j]`,
with `_B` for port B. A router joins them to the decoder.

### Verification

- **Rows**:
  - Both controllers and the three latch rows (enables, write address and
    `DIN`, read address), as `MacroSpec()` sizes them, are DRC clean when
    terminated.
  - Each matches its reference netlist in LVS.
- **Strips** at 32 wordlines (8 slices per port, 1.512 um tall):
  - every `WWL[i]` and `RWL[i]` is on the column's pin, layer and x;
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
  about 40 min at a 2 ns period. A switch-level test of both decoders
  (`test_each_decoder_selects_one_row_and_one_wordline`) checks the selects
  and wordlines for every address in a fraction of a second.

  Typical (TT, 0.70 V, 25 C), 2 ns: every check passes.

  | after CLK rises | write port | read port |
  |---|---|---|
  | precharge off | 86 ps | 86 ps |
  | wordline on (`WEN` / `REN`) | 132 ps | 109 ps |
  | `sae` | | 166 ps |
  | wordline off / `sae` low | 215 ps | 208 ps |
  | precharge on again | 259 ps | 254 ps |

  - CLK to `DOUT`: 200–204 ps.
  - Write margin (the cell flips, then its wordline falls): 69–70 ps.
  - Both ports are idle again about 260 ps after the rise, so the cycle is
    far shorter than 2 ns. The minimum period, and the fast and slow
    corners, are still to be measured.

The rest of the periphery is small logic, about 110 gates per macro:
- the enable gating;
- the one-hot decodes and the `SEL` gates;
- the replica.

It has no layout yet.

## Next

- Lay out the small logic and the macro tile:
  - the tile's dummy end rows close the column's top `GATE.S.3` findings;
  - then the floorplan, LEF and Liberty.
- Redraw the IO cells on taller rows for gdscheck v0.2.1's `M1.S.2`/`M2.S.2`.
