# Column IO and Bitline Periphery: `chipforge_asap7.devices`

The column IO periphery handles bitline precharge, column multiplexing,
differential sense amplification, write driving, and output data latching.
These cells share the `rowcell` substrate: two standard-cell rows pitched for
the bitcell array (594 nm, half an 8T bitcell row per leaf) with shared VDD in
the middle, full-height M3 vertical tracks for data and control lines, and M4
horizontal interconnect across the macro block.

## Parametric bitline leaf: precharge and column mux

`BitlineMuxSpec` is the part of a column IO that repeats at the bitline pitch:
two precharge pFETs and a transmission-gate column select, six transistors per
bitline pair. The released bank draws it (`sram_prech_ymux_6t112`) inside a
core pitched for the 270 nm 6T cell; this one takes its height from the array.
The default row is 297 nm, half a 594 nm 8T bitcell row, which needs bands of
unequal height (`RowStack(band_height=(135, 162))`; a single integer still
means both).

```python
from chipforge_asap7.devices import (
    BitlineMuxSpec, build_bitline_mux, build_bitline_mux_group,
)

spec = BitlineMuxSpec(n_fins=3, p_fins=3, selects=4, select=0)
spec.cell_name            # "blmux_3n3p_h135x162_s0of4"
spec.width, spec.height   # (702, 297) nm
spec.tracks_y             # {"BL": 80, "BLN": 116, "YSEL": 152, "YSELN": 188}  M2
spec.track_x              # SA, SAN, PRECHN, YSEL[0..3], YSELN[0..3]           M3
leaf = build_bitline_mux(spec)
mux = build_bitline_mux_group(spec)   # four leaves, 702 x 1188 nm, a 4:1 mux
```

In the dense row style a poly stripe spans both bands, so an nFET and the pFET
above it share a gate. Here none do: the transmission gate wants `YSEL` below
and `YSELN` above, and precharge has no nFET. Every n-only or p-only device
gets a gate column of its own and the other band has no diffusion under it:

```
column   xa   Ga   x0   G0   x1   G1   x2   G2   x3   G3   x4   Gb   xb
p band         .   SA  YSELN BL  PRECHN VDD PRECHN BLN YSELN SAN   .
n band   BL  YSEL  SA    .    .    .    .    .    .    .   SAN YSEL BLN
```

What a group shares runs straight through every leaf on full-height M3: `SA`,
`SAN` and `PRECHN` over their own columns, and all `selects` select pairs on
36 nm tracks either side, of which a leaf taps its own. So the mux ratio is a
parameter (each select costs 36 nm a side: 486 nm wide at 2:1, 702 at 4:1, 918
at 8:1), and stacking leaves is the whole of the mux routing.
`build_bitline_mux_group` stacks them, every other one flipped so neighbours
share a rail. Bitlines enter on M2 at the left edge.

`tests/test_bitline_mux_drc.py` holds the leaf to the public runset: alone it
reports `ACTIVE.LUP.1` (it has no tap) and nothing else, inside `filler leaf
filler tap filler` it is clean, and stacking adds nothing.
`tests/test_bitline_mux_lvs.py` compares leaves against a unit-fin reference
and a stacked group against `render_bitline_mux_group_lvs_schematic`, flat:
`SA`, `SAN`, `PRECHN` one net each, every select reaching its own leaf only.
Two leaves given the same select fail that comparison, as does a wrong fin
count.

```bash
uv run asap7-bitline-mux --selects 4 --group --out build/blmux/group4.gds
```

### Two rows, and bitlines where the array puts them

Two ports' leaves cannot share a strip: their M3 tracks sit at the same x, and
a leaf shifted a gate pitch runs its tracks through the other's M3 jumpers. So
each port gets a strip at its own end of the bitlines, and with it the whole
594 nm of bitcell row per pair. `rows=2` spends that on the devices: the row
is drawn once as a cell and placed twice, the second time mirrored about the
VDD rail, so every full-height M3 track meets itself and the two copies are in
parallel. Twice the fins for one gate pitch of extra width.

`bitline_entry=(y_BL, y_BLN)` says where the pair crosses the leaf's left edge
and `bitline_layer` on which metal. Two M3 columns outside the select tracks
take each bitline from there to its M2 track in both rows; on M4 the column
ends in an 18 x 24 nm V3 under a 24 nm stub. For OpenFinRAM's `sram_cell_8t`,
whose rows put port A on M2 and port B on M4:

```python
port_a = BitlineMuxSpec(rows=2, bitline_entry=(348.5, 245.5))                     # 756 x 594 nm
port_b = BitlineMuxSpec(rows=2, bitline_entry=(510, 78), bitline_layer="M4")
```

An entry that would put two nets' M2 within a space of each other is rejected
with the offending pair of y's. `build_bitline_mux_group` flips alternate
leaves, which is also how an array flips alternate rows, so a flipped leaf
finds its bitlines where its entry, flipped, says. Port A's strip goes on the
left of the array mirrored in x, port B's on the right as drawn.

The leaf centres a fin *space* on its edge, as ASAP7's standard cells do; an
SRAM bitcell centres a *fin* there, so a leaf beside an array is half a fin
pitch off its grid. `grid_offset=13.5` says the leaf is placed that far up
the array's row, `bitline_entry` staying measured from the row: the fins land
on one grid and the leaf can abut the array. This is how the released ASAP7
bank places its periphery. A group's flipped leaves then sit that far *down*
from their rows, so `for_row(i)` moves their landings the other way, and the
group's `BL[i]` labels are where the leaf's edge meets each row's bar.

Both are in the DRC and LVS tests above, alone, terminated and stacked, with a
leaf whose upper row has lost its V2 as the case that must fail. Against the
real `array_x2x4_sram_8t` (OpenFinRAM `tmp/bitline_io/`), with a strap per
bitline across the gap an edge cell fills, all sixteen bitlines reach six fins
of nFET on their own `YSEL`, six of pFET on their own `YSELN` and six of
precharge, and nothing else.

## Write driver and output latch

The other two per-column IO cells, drawn on two of the leaf's rows (594 nm,
VDD shared in the middle) so the three abut. Both keep the released bank's
topology and take fins, band height and VT as parameters; the helpers they
share with the leaf (frame, rails, contacts, via stacks, tracks) live in
`chipforge_asap7.devices.rowcell`.

```python
from chipforge_asap7.devices import WriteDriverSpec, OutputLatchSpec

WriteDriverSpec(n_fins=3, p_fins=3, keeper_fins=1)   # 648 x 594 nm, "wrdrv_3n3p_k1_h135x162"
OutputLatchSpec(n_fins=3, p_fins=3, fingers=2)        # 756 x 594 nm, "outlatch_3n3p_f2_h135x162"
```

`WriteDriverSpec` is the released `write_driver_sram`: an inverter, two nFET
pass gates that load `(D, DN)` into a cross-coupled latch while write enable
is low, and two transmission gates that put the latch on `SA`/`SAN` while it
is high, so the latch's own pull-downs write the bitline. The bottom row holds
everything gated by write enable and the inverter; the row above holds the
latch, whose pFET keepers (`keeper_fins`, one by default) are their own narrow
island so a pass gate can overwrite them. `W`/`WN` climb between the rows on
the M3 of the columns their transmission-gate pFETs sit on. `SA` and `SAN`
are full-height M3 tracks as in the leaf; `D`, `WRENA`, `WRENAN` are M3 stubs
from the bottom edge.

`OutputLatchSpec` is the released latch and driver: two NAND2s cross-coupled
as an SR latch that the sense amplifier's `QA`/`QAN` set (both high while it
precharges, the latch's hold state), an inverter, and a tristate inverter
onto the data pin enabled by `OE`/`OEB`. The latch is one five-column island
per band with its series nodes uncontacted. The tristate is drawn in pairs of
fingers (`fingers`, even), six stripes a pair, because its nFET and pFET
enable stripes cannot share poly; `Q` collects the pairs on the p via row.
`QA`/`QAN` leave on M3 at the bottom, `OE`/`OEB`/`Q` at the top.

Every variant in `tests/test_write_driver_physical.py` and
`tests/test_output_latch_physical.py` matches a unit-fin LVS reference (the
uncontacted series nodes named per fin, as the NAND's are), reports only
`ACTIVE.LUP.1` alone and nothing inside a filler/tap row, and a stronger
keeper or more fingers than drawn is the case that fails. The three IO cells
abutted on one row pair, terminated, report nothing.

```bash
uv run asap7-write-driver --keeper-fins 1 --out build/io/wrdrv.gds
uv run asap7-output-latch --fingers 4 --out build/io/outlatch.gds
```

## Sense amplifier on the IO row

`SenseAmpRowSpec` is the released amplifier drawn as an IO cell, 702 x 594 nm
at the defaults, so the four column cells abut. A current-latched differential
pair: `SA`/`SAN` gate the input nFETs, `SAE` the tail, the drains feed a
cross-coupled nFET pair under a cross-coupled pFET pair, and `SAPRECHN`
precharges and equalizes the outputs. The four off transistors the released
cell keeps on its outputs as matching dummies are not drawn.

```python
from chipforge_asap7.devices import SenseAmpRowSpec

SenseAmpRowSpec(n_fingers=2, tail_fingers=2)   # "sarow_3n3p_f2t2_h135x162", 702 x 594 nm
SenseAmpRowSpec(n_fingers=4, tail_fingers=4)   # 1242 x 594 nm
```

The bottom row's n band is one interleaved chain, `52 57 QAN 57 52 VSS 52 58
QA 58 52` under `SA QA QA SA SAE SAE SAN QAN QAN SAN`: each finger's input
device and cross-coupled device share the uncontacted node between them and
every `52` column joins on the 80 nm track. The cross-coupled pFETs need
exactly the `QA`/`QAN` stripes, so they sit in that row's p band over them
with no stripes of their own, `n_fingers` each. The row above holds the
precharge and equalizer as one island whose four stripes share one
`SAPRECHN` pad. `SA`, `SAN`, `SAE` leave on M3 at the bottom, `SAPRECHN`,
`QA`, `QAN` at the top.

Every variant in `tests/test_sense_amp_row_physical.py` matches a unit-fin
reference, reports `ACTIVE.LUP.1` alone and nothing in a filler/tap row, a
longer tail than drawn fails, and the four IO cells abutted on one row pair
report nothing. In Xyce (TT, 0.7 V, 20 fF a side) the netlist resolves 10 mV
of either polarity 13 ps after `SAE`, the released amplifier's polarity, 3 ps
behind it.

```bash
uv run asap7-sense-amp-row --n-fingers 4 --tail-fingers 4 --out build/io/sarow.gds
```

## The column IO block

`IoColumnSpec` places the four cells for one port and one data bit and joins
them: the mux group at the left, its bitlines entering at the edge, and beside
it a logic column one row pair per cell, sense amplifier, output latch, a
tap with fillers, write driver. 1620 x 2376 nm for a 4:1 group of two-row
leaves; the logic column is the widest cell, widened if need be so that every
spare beside a narrower cell is at least a support cell.

```python
from chipforge_asap7.devices import IoColumnSpec, BitlineMuxSpec, build_io_column

port_a = IoColumnSpec()   # BitlineMuxSpec(rows=2, bitline_entry=(348.5, 245.5)), defaults elsewhere
port_b = IoColumnSpec(mux=BitlineMuxSpec(rows=2, bitline_entry=(510, 78), bitline_layer="M4"))
cell = build_io_column(port_a)
port_a.routes       # {"SA": [(96, [270, 810]), (1878, [270, 918])], ...}
```

What joins the cells is drawn on M4, the one layer none of them use: a
horizontal wire with an 18 x 24 V3 on each M3 it meets, at a height where
both exist. `SA`/`SAN` run from the group's full-height tracks to the
amplifier's stubs and to the driver's columns; `QA`/`QAN` climb from the
amplifier's risers into the latch's stubs directly above (where riser and
stub share a column, the boundary is bridged on M3 and no M4 is drawn).
Every other pin is the cell's own metal, labelled once: the per-leaf
bitlines and selects, `PRECHN`, `SAE`, `SAPRECHN`, `D`, `WRENA`, `WRENAN`,
`OE`, `OEB`, `Q`, and each rail. The controls leave at the block's top and
bottom edges on M3, clear of the block's own M4, which is where a macro
router can reach them: the write driver's from its top, the latch's `Q`,
`OE`, `OEB` from its top, `QA`/`QAN` internal. `one_sense_phase=True` straps
the amplifier's `SAPRECHN` to `SAE` on M3 and leaves one `SAE` pin, on the
top stub; `block_netlist(spec)` is the whole block as one flat BSIM-CMG
subcircuit, which is how OpenFinRAM's deck carries it.

`tests/test_io_column_physical.py` holds the port-A and port-B blocks and an
8:1 one with a bigger amplifier and driver to the public runset (nothing
fires; without the tap, `ACTIVE.LUP.1` alone) and to a flat unit-fin LVS
reference of all four cells (`render_io_column_lvs_schematic`); the
reference with `QA` and `QAN` crossed between amplifier and latch is the case
that fails.

```bash
uv run asap7-io-column --mux.selects 4 --out build/io/iocol_port_a.gds
uv run asap7-io-column --mux.bitline-entry 510 78 --mux.bitline-layer M4 --out build/io/iocol_port_b.gds
```

OpenFinRAM's `iocol_sram_8t_a/b` are this block (built there at the
bitcell's bitline heights, port A mirrored), and its netlist is what the
compiler's deck instantiates. Two blocks built into one gdspy library share
their logic cells and supports.

`two_sided=True` puts a second group on the logic column's right, mirrored in
x so its bitlines enter from that side, and carries the `SA`/`SAN` M4 lines
on across the logic column to its tracks: one amplifier, write driver and
output latch for two arrays that face each other, such as two banks of one
port, of which a cycle accesses one. Each group keeps its own precharge and
selects (the second's pins are the first's with `_R`: `BL_R[i]`,
`YSEL_R[i]`, `PRECHN_R`). For the four-leaf block that is 2376 nm wide
instead of two blocks' 3240. The sense lines now also carry the second
group's unselected leaves and the wire across the core. The tests hold it to
the runset and to its flat reference, and a reference whose second group has
sense lines of its own is the case that fails.

---

See also:
- [Cell generators index](devices.md)
- [Parametric FinFET](finfet.md)
- [Row decoder and standard-cell row logic](row_decoder.md)
- [Simulation and verification](verification.md)
