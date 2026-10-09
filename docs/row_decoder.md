# Row Decoder and Standard-Cell Row Logic: `chipforge_asap7.devices`

The row decoder and wordline driver hierarchy translates decoded address lines
into driven wordlines on the bitcell array's pitch. These cells share the
standard-cell row style (`RowStack`): shared diffusion, continuous poly,
alternating row orientations, and abutting power rails.

## Stacked-band inverter

`build_finfet` draws one transistor as a self-contained island: its own body
tap, a dummy gate on each side, ACTIVE inset for a full 46 nm select
enclosure, and a routing band above the channel for gate access. That is the
right shape for a device you place and check on its own. It is the wrong shape
for a dense peripheral cell, and the released ASAP7 SRAM says so —
`dec_inv_62f_halved_AND` in `asap7_sram_0p0/gds/srambank_32b.gds` is built the
other way round, and `InverterSpec` reproduces it:

```python
from chipforge_asap7.devices import InverterSpec, build_inverter, build_inverter_row

spec = InverterSpec(rows=((18, 18), (13, 13)), fingers=2)
spec.cell_name            # "inv_fin_18n18p_13n13p_2f"
spec.width, spec.height   # (162, 1890) nm -- the released tile, exactly
spec.row_ys               # (0, 1080, 1890): three rails, two standard-cell rows
spec.pull_down_fins       # 62  (the "62f" the released cell is named after)
spec.drc_verifiable       # False -- 13- and 18-fin bands are past the deck's list

row = build_inverter_row(spec, 4)   # the four wordline drivers, as ASU place them
spec.row_placements(4)    # ((0, False), (216, True), (216, False), (432, True))
```

Four transistor bands stacked on one continuous poly and one output strap:
an 18-fin nFET, an 18-fin pFET, then a flipped row with a 13-fin pFET and a
13-fin nFET. Rows alternate orientation the way standard-cell rows do, so
every rail lies between two bands of one polarity and carries one supply. The
output crosses a rail on M1→M2→M3→M2→M1, because M1 owns that track; folding
past two fingers gives a row two drain columns, which are tied on M2.

Each band is a `FinFETSpec`, and `build_device_band` — factored out of
`build_finfet` — draws the diffusion stack the two have in common: one
fin-quantized ACTIVE rectangle and a 24 nm SDT + LISD bar per column. Taps,
vias, straps, rails, implant and cuts stay with each caller, because that is
exactly where an isolated tile and an abutting array cell disagree.

`abut=True` (the default) is the released array style: ACTIVE and implant
overhang the tile so neighbours merge into one diffusion strip, and each tile
draws one more gate than it owns. `row_placements` mirrors alternate
instances so that extra gate lands on its neighbour's outermost finger and its
outer source column on the neighbour's — `count` tiles become one continuous
54 nm array with `count` independent inputs and outputs. A tile holding half of
every finger pair is what "halved" means in the released cell's name.
`abut=False` draws a self-contained island instead: its own dummy gates, no
overhang, and a full select enclosure.

Neither style carries a body tap — the released cell has none either, because
logic cells rely on periodic tap rows. So the cell cannot satisfy `ACTIVE.LUP.1`
alone, and an abutting one is 37 nm short of `NSELECT/PSELECT/WELL.ACTIVE.EN.1`
at the two *outer* edges of a row. `tests/test_inverter_drc.py` asserts the
exact residual category set for each style rather than waiving them, and pins
the interior down a different way: a four-wide row reports the same violation
*count* as a single tile, which it could not if abutment left any gap.
`tests/test_inverter_lvs.py` extracts rows of one, two and four instances and
matches them against a reference of that many independent inverters — the check
that every shared edge gate really became its neighbour's finger, and that no
two drivers share an output. `tests/test_reference_validation.py` compares the
drawn ACTIVE, SDT, LISD, NWELL and BOUNDARY against the released cell shape for
shape, and asserts the three places the two deliberately differ.

The released cell draws a tall M1 input strap and a gate contact in every row,
which suits a parent that wants to choose where it lands. A parent that lands
once does not need them, and two knobs say so:

```python
InverterSpec(rows=((4, 6), (3, 3)), input_rows=(0,))                         # one landing, in row 0
InverterSpec(rows=((4, 6), (3, 3)), input_rows=(0,), input_reach=(14, 14))   # ... cut to the via pad
```

| `(4, 6), (3, 3)`, isolated | switched cost | A-Y coupling | worst gate R |
| --- | --- | --- | --- |
| default: full strap, every row | 1.214 fF | 0.196 fF | 126 ohm |
| `input_rows=(0,)` | 1.168 (-3.8 %) | 0.186 | 126 |
| ... and `input_reach=(14, 14)` | 1.093 (-10.0 %) | 0.168 (-14 %) | 126 |

`input_reach` is `(below, above)` the seam in nm: `None` is the full strap, and
a parent asks for what its landings need. Both knobs move metal only, so the
netlist and the LVS reference are the same for every setting. Neither can
remove a row's LIG strap: it ties the fingers' gates together at that seam, and
without it the worst gate resistance goes from 126 to 199 ohm. A landing
under 36 nm long answers to M1.S.2 (25 nm to a neighbour, not 18), so it must
stop 2 nm further from the via rows than the strap does; a one-fin band has no
room for that, and the spec says so rather than leaving it to DRC. These
knobs, their limits and the numbers all came from [layout
reduction](verification.md#layout-reduction); `DriverSliceSpec` applies them to its drivers by
default.

```bash
uv run asap7-inverter --rows 18 18 13 13 --fingers 2 --out build/inv.gds
uv run asap7-inverter --rows 4 6 3 3 --input-rows 0 --input-reach 14 14 --no-abut
```

## Row support — tap, filler, decap

An abutting logic row is as clean as a logic cell can be on its own and still
reports eight DRC violations, because two of the rules it has to satisfy are
not properties of a cell at all. Latch-up needs a body tie within 30 µm, and a
logic cell carries no tap — the released `dec_inv_62f_halved_AND` carries none
either. Implant enclosure needs 46 nm past the outermost ACTIVE, and an
abutting cell deliberately stops at 9 nm because the rest comes from its
neighbour. Measured with the public runset on a four-wide inverter row:

| arrangement | violations |
|---|---|
| `row` | 8 — 2× LUP, 2× WELL.EN, 2× NSELECT.EN, 2× PSELECT.EN |
| `filler row filler` | 2 — LUP only |
| `filler row filler tap filler` | **0** |

```python
from chipforge_asap7.devices import InverterSpec, RowSupportSpec, build_row_support

inverter = InverterSpec(rows=((18, 18), (13, 13)), fingers=2)
tap = RowSupportSpec(stack=inverter.stack, kind="tap")   # same stack, by construction
build_row_support(tap)
tap.width, tap.height   # (108, 1890)
```

`RowStack` is what keeps the two in step: it owns where the bands are, how tall
each is, which polarity it carries and which rail it faces, and both
`InverterSpec.stack` and `RowSupportSpec.stack` are the same object. A tap
built from a different stack cannot silently half-fit. By default each band is
as tall as its fins make it (a 2-fin row is 216 nm); `RowStack(rows=((2, 2),),
band_height=135)` pins every band to 135 nm so that a row of any legal fin
count, and its tap and filler, sit on the released 270 nm 7.5-track rails. The
pinned height is carried in the stack's `code` (`2n2p_h135`), so the cells it
names cannot collide with their fin-default counterparts.

**The ordering is not interchangeable.** A filler carries the *same* implant as
the row, which is what extends the enclosure past its last ACTIVE; a tap
carries the opposite. Put a tap straight against a logic cell and the
enclosure violation stays exactly where it was — while latch-up is fixed, which
is what makes the mistake easy to miss.
`test_row_support_drc.py::test_a_tap_next_to_logic_does_not_replace_a_filler`
pins that down.

On a 3-fin/3-fin stack — two 135 nm bands, which is the released 270 nm
standard-cell row — the generated tap and filler are **identical to
`TAPCELL_ASAP7_75t_R` and `FILLER_ASAP7_75t_R` on every layer**, asserted as an
equality in `tests/test_reference_validation.py`. That includes the 16 nm LI
rail under each power rail, which is not decoration: `V0.AUX.1-2` oversizes LIG
by 1 nm before asking whether a contact landed on local interconnect, so 16 nm
is exactly enough for an 18 nm via, and without it every contact sitting on a
rail hangs over the end of its LISD bar.

`decap` is the third of the trio and the only one with devices in it. The two
bands of a row share one gate conductor, which floats, so their capacitors sit
back to back between VDD and VSS and neither oxide sees the full supply — half
the capacitance of a grounded-gate pair, and what fits in a cell whose bands
span the same columns. Stacked rows each get their own plate net, because the
poly is cut at every rail.

The tap is verified electrically, not just geometrically:
`test_row_support_lvs.py` runs LVS with the deck's global body tie switched
*off*, so every body terminal has to reach its rail through drawn geometry. A
terminated row matches; the same row with its tap removed does not. With the
tie declared, both pass — which is the point: the usual LVS setup cannot tell a
working tap from a missing one.

```bash
uv run asap7-row-support --kind tap --stack.rows 18 18 13 13 --out build/tap.gds
```

## Parametric NAND2

`NandSpec` is the post-decode gate of the released row decoder,
`dec_nand_12f_12f_for_and_size_reduced_post_decode_P1N1`, drawn in the same
tapless, abutting, shared-diffusion style as the decoder inverter. In the
bank it is the last NAND of the wordline AND tree, `WL<i> =
INV62(NAND(PA·WLENA, PB·PC<i>))`, and feeds `dec_inv_62f_halved_AND`:

```python
from chipforge_asap7.devices import NandSpec, build_nand, build_nand_row

spec = NandSpec(rows=((14, 7),), fingers=2)   # the released tile
spec.width, spec.height        # (216, 675) nm
spec.gate_roles                # ('B', 'A', 'A', 'B')
spec.column_roles("n")         # ('S', 'x', 'Y', 'x', 'S'): rail, series node, output
spec.column_roles("p")         # ('S', 'Y', 'S', 'Y', 'S')
spec.netlist()                 # one series pair and two pull-ups per finger
```

Gates run `B A A B` so that every A gate is on the output side of its series
stack and every B gate on the rail side; in the decoder A carries the late,
wordline-enable-gated select. The A gates share one LIG pad. The B gates
cannot, and the released cell leaves them for its parent to join on M2;
`build_nand` joins them itself with an M2 bar (one V1 per gate bar, stopping
flush with the outer vias so butted tiles keep 36 nm between ties), so the
cell is one NAND2 that LVS can match on its own. `fingers` is per input;
more than two puts A on M2 as well. `abut=False` draws an island with edge
dummies, and `NandSpec(rows=((3, 3),), fingers=1, abut=False)` is the
NAND2xp33 footprint on the 270 nm row. One row only for now.

`tests/test_nand_drc.py` asserts the same contract as the inverter: an
isolated island reports only the missing body tap, an abutting row alone adds
the row-end implant enclosure, and `filler row filler tap filler` is clean,
for two and four fingers (one and two M2 ties) and for a `band_height=135`
row. `tests/test_nand_lvs.py` matches single tiles and butted rows of one,
two and four against unit-fin references, in which the uncontacted series
node is one net per fin because that is what the extractor sees.
`tests/test_reference_validation.py` compares the diffusion, wells, implant,
gates and contact columns against the released tile.

```bash
uv run asap7-nand --rows 14 7 --fingers 2 --out build/nand.gds
```

## Decoder sizing and the four-wordline driver slice

`size_decoder` sizes every gate between an address bit and the wordline by
logical effort, in fins. It is arithmetic only and needs nothing installed.
The wordline load sizes the driver; the depth only decides how the address is
split into predecode groups and how many gates hang on each predecode line:

```python
from chipforge_asap7.devices import DriverSliceSpec, build_driver_slice, size_decoder

sizing = size_decoder(wl_load_fF=22.8, depth=32, stage_effort=4.43)
sizing.stages["driver"].drive      # 62   -> dec_inv_62f_halved_AND
sizing.nand_rows, sizing.driver_rows   # ((14, 7),)  ((18, 18), (13, 13))
sizing.groups                      # {'PC': 2, 'PB': 2, 'PA': 1}, the released 5-to-32 split
print(sizing.summary())            # every stage: fins, load, effort achieved, delay

spec = DriverSliceSpec.from_sizing(sizing)   # == DriverSliceSpec(), the released slice
spec.width, spec.height            # (432, 3240) nm
spec.wordline_xs                   # [54, 162, 270, 378]: M3 on the 108 nm bitcell pitch
```

The technology numbers in `LogicalEffortModel` were measured with Xyce on the
ASAP7 TT BSIM-CMG card (0.7 V, RVT): tau 1.76 ps, inverter parasitic 0.59,
FO4 8.1 ps, 41.7 aF of gate per fin, and a NAND2 logical effort of 1.44
against the 1.5 that equal n/p strength predicts. With them a 22.8 fF
wordline at the released stage effort reproduces the released driver and NAND
exactly. Holding the stage effort holds the delay, so a deeper decoder costs
address-input capacitance rather than time. The wire capacitance default is
an assumption, and the wordline's own RC is left to the caller.

`DriverSliceSpec` is the upper 3.24 um of the released
`post_Decode_and_size_reduced_x1` slice: two rows of two `NandSpec` tiles, the
upper row mirrored, under four `InverterSpec` drivers interleaved as
`build_inverter_row` places them. `WL<i> = SEL . B<i>`. The slice adds the
routing its leaves cannot have: SEL on an M2 bar along each NAND row's seam
joined by one M3 on the tile boundary; each NAND output up to its driver on
M3, because M1 cannot cross the VSS rail between them, with the lower NAND's
output jogging one gate pitch on M2 and the two landings staggered by a
track; and each wordline from the driver's top drain contact, which the
interleave puts on the bitcell pitch, up M3 to the slice edge.

The drivers are drawn with the input metal that routing lands on and no more
(`spec.driver`: `input_rows=(0,)` and a reach from the two landings up to the
gate contact), which on the released size takes each driver's input M1 from
1.62 um in two straps to 0.49 um in one. `trim_driver_input=False` keeps the
full straps for a custom `router=` that wants the choice; `input_strap_y`
reports what is drawn either way.
`build_driver_slice_support` stacks the filler, tap or decap of both row
stacks into one full-height column. `tests/test_driver_slice_physical.py`
checks the slice alone (tap and row-end enclosure only), one and two butted
slices between their support columns (clean), the released size (only the
deck's height enumeration), and LVS against four ANDs sharing SEL, with a
private-select reference failing. The small gates that make SEL and B<i>
from the predecode lines are standard cells in the released slice and are
not drawn here.

```bash
uv run asap7-wl-slice --wl-load-ff 22.8 --depth 32 --stage-effort 4.43 --out build/slice.gds
```

---

See also:
- [Cell generators index](devices.md)
- [Parametric FinFET](finfet.md)
- [Column IO and bitline periphery](column_io.md)
- [Simulation and verification](verification.md)
