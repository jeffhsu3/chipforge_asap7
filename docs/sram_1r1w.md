# 8T 1R1W SRAM: the column

`chipforge_asap7.sram1r1w` is the start of a one-read, one-write SRAM
compiler. It follows the structure of GT2N's rd8t macro
(`chipforge_gt2n/sram/rd8t_*`). Its cell is OpenFinRAM's ASAP7 8T cell
`sram_cell_8t`, a true dual-port cell of 108 x 594 nm. Port A of the cell is
the write port and port B is the read port. This first milestone is one data
bit's column: the cell, its array, a write IO and a read IO. It is DRC
checked, LVS checked and simulated. Decoders, control, the macro tile, and
the LEF and Liberty views come next.

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

## Next

- Decoders, a one-clock controller and the macro tile, as decided on
  2026-11-10:
  - both ports are captured at CLK's rise;
  - the write is timed by an inverter chain;
  - the read self-times from a replica bitline, as the NOR ROM's
    flow-through controller does;
  - the tile has dummy end rows that close the tap `GATE.S.3` findings.
- Redraw the IO cells on taller rows for gdscheck v0.2.1's `M1.S.2`/`M2.S.2`.
