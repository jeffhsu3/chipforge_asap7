# chipforge-asap7

ASAP7 PDK layout collateral for the
[chipforge](https://github.com/jeffhsu3/chipforge) hardware ML compiler.

ASAP7 is chipforge core's **primary** PDK target, so — unlike `chipforge_gf180`
— this package is *not* a set of compiler hook-point subclasses. Core keeps its
ASAP7 SRAM wrapper, `sram_backend` strings and chip-top flows. What lives here
is the ASAP7 *geometry* that core's hand-built array compilers keep re-deriving
independently.

## Install

The package is a standalone, importable library — it does **not** depend on
chipforge, so any project can consume it.

```bash
git clone https://github.com/jeffhsu3/chipforge_asap7.git
cd chipforge_asap7
python -m pip install .          # constants and grid math
python -m pip install ".[gds]"  # also install gdspy for GDS drawing
```

The repository currently requires GitHub access. To develop against a local
checkout, run `python -m pip install -e ".[gds]"` from the repository root.
The package is licensed under [BSD 3-Clause](LICENSE); see
[third-party notices](THIRD_PARTY_NOTICES.md) for the ASAP7 attribution.

**`chipforge_asap7` has no required dependencies.** The layer map, the grid
constants and every pure-geometry helper (`column_x`, `column_gate_track`,
`fin_ys`, `gate_track_xs`, `centered_fin_ys`, `row_boundary_ys`,
`spans_excluding`, `snap_to_*`) import and run with nothing else installed — so
LEF / Liberty / sizing / DRC scripts that only need pitches and alignment math
can depend on it cheaply. gdspy is the optional `gds` extra, imported lazily and
only by the `draw_*` / `box` helpers; call `have_gdspy()` to check, or just call
a drawing helper and get an actionable `ImportError`. The package ships
`py.typed`, so its annotations are visible to consumers.

## `chipforge_asap7.layout`

The fin / gate substrate grid that every abutted-array macro is built on, lifted
verbatim (same constants, same drawn rectangles) from the NOR-ROM flow in
chipforge's `scripts/brom_compiler.py`, `scripts/rom_compiler.py`,
`scripts/pseudo_nmos_compiler.py` and `scripts/gen_rom_bitcell.py`:

- **`LAYERS`** — the ASAP7 GDS layer map, as `{"layer", "datatype"}` mappings
  ready to splat into gdspy constructors, plus a `box()` helper.
- **Grid constants** — `FIN_PITCH` 27, `GATE_PITCH` (CPP) 54, `COLUMN_PITCH`
  108, `WORDLINE_PITCH` 81, `CANDIDATE_PITCH` 108, `STD_CELL_HEIGHT` 270,
  `COLUMN_X_SHIFT` 46. Short aliases (`FIN_P`, `GATE_P`, `PITCH_X`, `WL_PITCH`,
  `X_SHIFT`, `STD_CELL_H`) match the names the scripts already use.
- **`draw_fin_grid` / `draw_gate_grid` / `draw_gate_cuts`** — the three loops
  each compiler wrote by hand.
- **`centered_fin_ys`** — active-fin placement inside one cell, with the
  isolation-pitch check that `gen_rom_bitcell` does.
- **`column_x` / `column_gate_track`** — the column-to-gate-grid alignment that
  `COLUMN_X_SHIFT = 46` exists to satisfy: with the 2-CPP bitcell (gate at
  `column_x - 36`), column *b*'s gate lands on gate track `2b + 2`.
  `column_gate_track` raises if a pitch/shift combination breaks that, instead
  of letting it surface as a post-hardening DRC/LVS surprise.
- **`row_boundary_ys` / `spans_excluding`** — row boundaries carry the gate
  cuts; `spans_excluding` splits a full-width feature (gate cuts, wordline
  rails) around mid-array buffer strips.

All coordinates are integers in nanometres, matching the gdspy libraries these
flows emit (`unit=1e-9, precision=1e-10`).

```python
import gdspy
from chipforge_asap7.layout import (
    column_x,
    draw_fin_grid,
    draw_gate_grid,
    draw_gate_cuts,
    row_boundary_ys,
    spans_excluding,
    WORDLINE_PITCH,
)

lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
cell = lib.new_cell("ROM_FULL")

height = (depth + 1) * WORDLINE_PITCH
draw_fin_grid(cell, left_margin, total_width, 0, height)
draw_gate_grid(cell, left_margin, total_width, -1200, height)
draw_gate_cuts(
    cell,
    spans_excluding(left_margin - 17, total_width - 17, buffer_strips),
    row_boundary_ys(depth, WORDLINE_PITCH),
)
```

## `chipforge_asap7.devices` — parametric FinFET

A four-terminal, multi-finger ASAP7 FinFET built on the grid primitives above.
It drops the continuous `width` knob used by planar generators: a FinFET's drive
strength comes in whole fins, so the device takes three **discrete counts** and
is named after them —

```python
from chipforge_asap7.devices import FinFETSpec, nmos_fin, pmos_fin

cell = nmos_fin(fins=1, fingers=2, multipliers=2)  # cell.name == "nmos_fin_122"
spec = FinFETSpec(flavor="p", fins=3, fingers=4)  # "pmos_fin_341"
spec.width, spec.height  # 432, 216  (nm), including tap/routing geometry
spec.gate_xs, spec.sd_xs  # [189, 243, 297, 351], [162, 216, 270, 324, 378]
spec.total_fins  # 12
spec.netlist()  # M0 D G S B pmos_rvt nfin=3 l=20n nf=4 m=1
lvt = FinFETSpec(vt="lvt")  # model nmos_lvt, cell nmos_lvt_fin_111
```

`nmos_fin_122` reads *1 fin, 2 fingers, 2 multipliers* — the same fin-count
nomenclature as OpenFinRAM's `sram_cell_6t_122` (`nfin_pu=1, nfin_pd=2,
nfin_pg=2`), rather than the planar `W=3u` style. Counts of 10 or more switch
to `_`-separated fields (`12_2_1`) so `1,12,1` and `11,2,1` can't collide.

Fins sit on the 27 nm fin pitch, gates on the 54 nm contacted-poly pitch, and
source/drain columns midway between gates — fingers share the column between
them. Multipliers are folded into additional physical fingers, preserving
`fins * fingers * multipliers` channel intersections in one interdigitated row.
LIG joins every active gate, and standard-cell-style M1 conductors physically
tie all source columns and all drain columns. A separate opposite-polarity tap
column exposes the real B terminal, so the generated cell has four independent
M1 pins: D, G, S and B.

The geometry is based on the released `INVxp33_ASAP7_75t_R` and
`TAPCELL_ASAP7_75t_R`: full FIN/GATE manufacturing grids, edge dummy gates,
fin-quantized ACTIVE, legal select/well enclosure, split gate cuts, 24 nm LISD,
matching 24 nm SDT source/drain markers, 18 nm V0, and minimum-area M1
landings. RVT is the default; `vt="lvt"`,
`"slvt"` and `"sram"` select the matching model and marker.

The generator only accepts the DRC-legal device grid: 20 nm gate length, one
54 nm CPP, one through `MAX_FINS` (18) fins, and no arbitrary `sd_dx`.
`row_height` may be enlarged only in whole fin pitches and must leave enough
select enclosure for the body tap. Invalid requests raise instead of silently
producing dirty GDS.

Two ceilings, for two different reasons. `MAX_VERIFIABLE_FINS` (12) is where
the *public KLayout runset* stops: it spells `ACTIVE.W.2` / `SDT.W.3`
("height is an integer multiple of 27 nm") as an enumerated list of twelve
heights, because KLayout has no integer-multiple predicate, so it reports a
false violation on anything taller. That is a property of the deck, not of
ASAP7 — the released SRAM banks ship 13-, 14- and 18-fin devices
(`dec_inv_62f_halved_AND` in `asap7_sram_0p0` has both an 18-fin and a 13-fin
pair), and running the same runset on ASU's own cell reports those two rules.
`MAX_FINS` (18) is therefore the device limit, set from the tallest device in
the released collateral. `FinFETSpec.drc_verifiable` reports which side of the
line a device falls on.

`tests/test_finfet_drc.py` runs a representative NMOS/PMOS, VT, fin, finger,
multiplier and row-height matrix through the public ASAP7 KLayout runset,
restricted to `drc_verifiable` specs, and asserts it is clean. A companion test
builds the 13- and 18-fin devices and asserts the runset's violation set is
*exactly* `{ACTIVE.W.2, SDT.W.3}` — so a real violation on a tall device shows
up as a new category, and a future deck that fixes the enumeration fails the
test and lets `MAX_VERIFIABLE_FINS` be raised. Both auto-discover this
workspace's tools or accept `KLAYOUT_BIN` and `ASAP7_DRC_DECK`; set
`ASAP7_REQUIRE_TOOLS=1` to turn "tool not installed" skips into failures. `tests/test_finfet_connectivity.py` independently subtracts
GCUT, traces LI/V0/M1 connectivity, proves D/G/S/B are separate, and checks the
physical channel count. The public educational runset is the DRC authority for
this project; it is not a claim of commercial foundry sign-off.

### Stacked-band inverter

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
reduction](#layout-reduction); `DriverSliceSpec` applies them to its drivers by
default.

```bash
uv run asap7-inverter --rows 18:18,13:13 --fingers 2 --out build/inv.gds
uv run asap7-inverter --rows 4:6,3:3 --input-rows 0 --input-reach 14,14 --isolated
```

### Row support — tap, filler, decap

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
uv run asap7-row-support --kind tap --rows 18:18,13:13 --out build/tap.gds
```

### Parametric NAND2

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
uv run asap7-nand --row 14:7 --fingers 2 --out build/nand.gds
```

### Decoder sizing and the four-wordline driver slice

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

### Parametric bitline leaf: precharge and column mux

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

#### Two rows, and bitlines where the array puts them

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

Write driver, output latch and the per-group block around `SenseAmpSpec` are
still to come.

### Write driver and output latch

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

### Sense amplifier on the IO row

`SenseAmpRowSpec` is the released amplifier drawn as an IO cell, 702 x 594 nm
at the defaults, so the four column cells abut (`SenseAmpSpec` above stays
the spacious tile-per-transistor version). A current-latched differential
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

### The column IO block

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
uv run asap7-io-column --selects 4 --out build/io/iocol_port_a.gds
uv run asap7-io-column --bitline-entry 510,78 --bitline-layer M4 --out build/io/iocol_port_b.gds
```

OpenFinRAM's `iocol_sram_8t_a/b` are this block (built there at the
bitcell's bitline heights, port A mirrored), and its netlist is what the
compiler's deck instantiates. Two blocks built into one gdspy library share
their logic cells and supports.

### Parametric sense amplifier

`SenseAmpSpec` builds the same 16-transistor differential latch used by the
OpenFinRAM SPICE fixture. `SA`/`SAN` are the differential inputs, `SAE` enables
regeneration, active-low `SAPRECHN` precharges the latch, and `QA`/`QAN` are
complementary outputs. Both polarity banks are sized in integer fins and can
select any supported ASAP7 threshold flavor:

```python
from chipforge_asap7.devices import SenseAmpSpec, build_sense_amp

spec = SenseAmpSpec(n_fins=12, p_fins=4, vt="rvt")
cell = build_sense_amp(spec)
spec.cell_name       # "sense_amp_sram_n12_p4"
spec.width, spec.height  # (7020, 999) nm
spec.pin_positions   # SA, SAN, SAE, SAPRECHN, QA, QAN, VDD, VSS on M2
```

The first physical implementation is intentionally spacious and
verification-oriented. Sixteen reusable FinFET tiles are arranged in mirrored
pairs, a continuous FIN grid covers the macro, and explicit V1/M2/V2/M3 routes
connect all 64 device terminals. This preserves a readable one-to-one mapping
between GDS and schematic before a later diffusion-sharing compaction pass.

Generate a viewable default GDS with:

```bash
uv run asap7-sense-amp \
  --n-fins 12 --p-fins 4 \
  --out build/sense_amp/sense_amp_sram_n12_p4.gds
```

`tests/test_sense_amp_physical.py` checks representative sizes and VT flavors
with the public KLayout DRC deck and compares the routed layout against a
unit-fin-expanded 16-transistor LVS reference.

### Single-device SPICE testbench

A transistor still needs a small netlist for simulation: the instance line
names D/G/S/B and selects the compact model, while voltage sources define the
biases to sweep. GDS contains polygons, not semiconductor equations or input
stimuli, so SPICE cannot consume it directly.

The DC bench below uses one FinFET for Id-Vg and Id-Vd characterization, then a
second one-transistor fixture with a resistor load to confirm that the output
node switches. It translates the ASAP7 BSIM-CMG 107 HSPICE model declaration to
ngspice's OSDI syntax without changing the model parameters. On this workspace,
the model card and matching Verilog-A source are discovered automatically:

```bash
uv run python -m chipforge_asap7.devices.spice
```

For a standalone checkout, provide their locations explicitly:

```bash
uv run python -m chipforge_asap7.devices.spice \
  --model-card /path/to/7nm_TT_160803.pm \
  --bsimcmg-source /path/to/bsimcmg_107.0.0/code \
  --out build/finfet_spice
```

The output directory contains the two runnable `.sp` decks, Id-Vg/Id-Vd/switch
data, simulator logs, and `results.json`. This is a **compact-model test**, not
a post-layout proof. The generated GDS is covered by the KLayout DRC and
layout-connectivity regressions described above. A formal post-layout proof
is provided by the open flow below; it remains a calibrated research flow
rather than a replacement for the unpublished sign-off decks.

### Open LVS and PEX

The package implements an end-to-end open-source alternative for the public
ASAP7 release:

1. KLayout extracts unit-fin MOS4 devices and proves D/G/S/B topology against
   CDL. Released standard-cell `NFIN` devices are normalized to the same
   one-device-per-fin representation, and tap-less cell wells are handled
   explicitly.
2. FasterCap solves a 3-D Maxwell capacitance matrix for diffusion, gate, LI,
   vias, and BEOL conductors.
3. KPEX/2.5D extracts distributed sheet and via resistance. The merger fixes a
   gap in KPEX 0.3.x's stock expander by reconnecting each MOS terminal to its
   extracted resistance-port node.
4. The two results are combined with one BSIM-CMG `NFIN=1` instance per
   physical channel. Exact zero-ohm port aliases become 1 mOhm in the runnable
   SPICE view to avoid an ill-conditioned ngspice matrix; raw values remain in
   the CSV/protobuf artifacts.
5. `INVxp33_ASAP7_75t_R` is correlated against the released xACT3D netlist,
   TT NLDM Liberty table, and a BSIM-CMG 107/OpenVAF transient.

KPEX 0.3.12 requires Python 3.12 or newer:

```bash
uv sync --extra gds --extra pex

# FasterCap is LGPL-2.1 and built from its three pinned upstream repositories.
# System prerequisites: cmake, g++, git, wxWidgets development headers, Eigen.
scripts/build_fastercap.sh build/fastercap
```

`find_fastercap` picks up that build location on its own, along with
`FasterCap` on `PATH` and `/opt/FasterCap/FasterCap`; set `FASTERCAP_EXE` only
to point at a solver somewhere else. `tests/test_pex.py` runs the real
field-solve regression whenever it finds one, and `ASAP7_REQUIRE_TOOLS=1`
turns "not built" into a failure rather than a skip.

Run a custom cell:

```bash
uv run python -m chipforge_asap7.verification pex \
  --gds build/nmos_fin_111.gds \
  --schematic build/nmos_fin_111.cdl \
  --cell nmos_fin_111 --substrate-net B \
  --model-card /path/to/7nm_TT_160803.pm \
  --out build/pex/nmos_fin_111
```

Or run the released inverter extraction and all reference checks in one command:

```bash
uv run python -m chipforge_asap7.verification reference-inverter \
  --asap7-root /path/to/asap7 \
  --model-card /path/to/7nm_TT_160803.pm \
  --out build/pex/reference_inv
```

Each run retains the LVS database/log, generated KPEX technology JSON,
FasterCap geometry/log/raw and symmetrized matrices, distributed-R protobuf and
CSV, raw KLayout netlist, runnable post-layout SPICE, calibration manifest, and
JSON report. `--smoke` selects a coarse field mesh for CI.

The stack is intentionally labeled `research-grade_not_signoff`. Released
xACT views supply Gate/LIG/M1/M2 sheet and V0/V1 resistance calibration;
upper-metal geometry is extrapolated from public dimensions, while dielectric
constants and unpublished vertical gaps carry explicit uncertainty. On the
included reference run, topology matched; BSIM rise/fall delays were about
0.87x the nearest Liberty entries, while open A-Y field coupling was about
0.41x xACT. Those numbers are useful guardrails and also show why this should
not be presented as foundry-accurate extraction.

The same BSIM-CMG/OpenVAF path also has a transient bench for OpenFinRAM's real
16-transistor `sense_amp_sram` topology:

```bash
uv run python -m chipforge_asap7.devices.sense_amp_spice
```

It precharges `QA/QAN`, releases active-low `SAPRECHN`, asserts `SAE` 20 ps
later, and runs both `SA > SAN` and `SAN > SA`. A case passes only if the
expected output ends above 80% of VDD, its complement falls below 20%, and the
output differential reaches 80% of VDD. The default 100 mV input differential,
12-fin nFETs, 4-fin pFETs, and 1 fF output loads can all be overridden from the
CLI. Generated decks, waveforms, logs, resolution delays, peak currents, energy,
and `results.json` are written under `build/sense_amp_spice/`.

```bash
python -m chipforge_asap7.devices --fins 2 --fingers 2 --multipliers 2
```

### LVS of an assembled macro

`run_lvs` proves a generated cell against a reference written for it.
`run_hierarchical_lvs` is for a layout this package did not draw: a macro of
released standard cells, hard IP and an SRAM array, against the designer's own
netlist.

```python
from chipforge_asap7.verification import run_hierarchical_lvs

result = run_hierarchical_lvs("sram.gds", "sram.sp", "build/lvs", cell_name="sram_x4x2x1",
                              flatten_circuits=("sram_cell_*",))
result.matched, result.failing, result.series_order_cells
print(result.describe())
```

Three things had to be true before such a comparison meant anything, and each
is an option of `run_lvs` and of the deck:

- **`merge_fin_diffusion`.** On the fins, the uncontacted node inside a series
  stack is a net per fin, and the generated cells' references are written that
  way. The ASAP7 Calibre deck, and so every released CDL, takes source/drain as
  the whole ACTIVE outside the gate, where it is one net. Without it no
  standard cell with a stack matches its CDL; with it, all 24 in OpenFinRAM's
  controller do except the two below.
- **`flatten_circuits`.** A bitcell's transistors are completed by its
  neighbours: the layout's `sram_cell_8t` holds 3 devices and its schematic 14.
- **Series order.** Released `AOI211xp5` stacks its pull-up C, B, (A1‖A2) in
  the GDS and the other way round in the CDL; `AO21x1` has A1 and A2 swapped in
  its pull-down. Calibre's gate recognition passes both. KLayout fails the
  first and *matches* the second by pairing A1 with A2, which then breaks the
  parent. `summarize_lvsdb` reports such pairings (`renamed_pins`) and decides
  `series_order_only`: both sides reduced to series/parallel trees with
  unordered series, internal nets named from the pins inward, equal trees and
  equal fins per input. A library cell proven that way is compared by pin name
  (`blank_circuits`) in a second pass and named in the result. A NOR offered
  for a NAND, or a cell one fin short, is not excused.

A failing circuit also gets `supply_shorts`: instance pins on a supply net in
the layout and not in the reference. A short to a rail swallows a net whole,
and the compare can then only say that nothing matches; this says which pins
went where. `double_implant_is_tap=False` is for seeing past one specific
error, ACTIVE under both implants acting as a well tap, not for passing with
it. The report reader runs inside KLayout's own Python, so it needs the binary
and nothing else.

On OpenFinRAM's two-port macro this matched every block below the top at
transistor level (the controller, both 514-device column IO blocks) and found
two shorts to VSS at the top that the compiler's metal-graph check cannot see;
see `docs/macro_verification.md` there.

### Layout reduction

Generators inherit geometry. `InverterSpec` draws a tall M1 input strap and a
gate contact in every row because the released decoder inverter does, and
nothing in a generator says which shapes are load-bearing. `reduce_layout`
finds out the way test-case reducers do: remove a piece, keep the removal if
the cell is still right.

"Still right" is three checks, cheapest first:

1. **LVS matches and every reference pin is still extracted.** The pin check is
   separate because KLayout pairs nets by topology, so a label that has lost
   its metal still "matches". The shape under each top-level pin label is
   pinned: never deleted, only shrunk to a landing around the label.
2. **No DRC category is worse** than in the unedited layout
   (`verification.drc.run_drc`, the public KLayout runset).
3. **Extraction agrees.** No net's worst resistance rises more than
   `r_tolerance` (5 %): from the pin to each device terminal, or, on a net with
   no pin, from each driving diffusion to each gate. And the
   switched-capacitance cost, in which signal-to-signal coupling counts as
   Miller capacitance, does not go up.

Deletion runs over vias, then metals, then LIG, to a fixed point; probes run in
parallel. Then each surviving wire has its ends pulled in by a parallel
k-section search. Shapes touching their cell's `BOUNDARY` (rails, abutment
stubs, a wordline leaving the top edge) are never edited, and an end that lies
under other metal is a junction, not a stub, and is left alone.

```bash
asap7-reduce --gds inv.gds --schematic inv.sp --cell INV --out build/reduce
asap7-reduce ... --no-capacitance   # resistance floors only, no field solver
```

```python
from chipforge_asap7.verification import ReductionConfig, reduce_layout

result = reduce_layout("inv.gds", "inv.sp", "build/reduce", cell_name="INV")
print(result.summary())           # edits per layer, what extraction refused, before/after
result.gds, result.report         # INV_reduced.gds, reduction.json
```

On `InverterSpec(rows=((4, 6), (3, 3)), fingers=2, abut=False)`, with its one
pin on row 0, three edits survive (234 probes and three field solves, 5 minutes
on 8 cores):

| Edit | Shape | Why it could go |
| --- | --- | --- |
| delete | row-1 M1 input strap, 18 x 134 | a second landing nothing uses |
| delete | the V0 under it | only fed the strap |
| shrink | row-0 M1 input strap, 242 nm to 36 nm | the landing at the contact is enough |

Cost falls 1.214 to 1.096 fF (-9.7 %) and A-Y coupling 0.196 to 0.169 fF, with
every pin-to-device resistance unchanged: what went was dead-end metal. The
row-1 **LIG pad stays**. LVS and DRC let it go, since the gates run through
both rows, but it ties the two fingers' gates together at the far end, and
without it the worst gate resistance goes 126 to 199 ohm. That is the case for
the third check.

Two things learned the hard way, both now defaults:

- **The field solver must be converged to compare layouts.** FasterCap refines
  its mesh until two passes agree to a tolerance. At 0.08-0.15 the same three
  edits read as anything from -56 % to +6 %; at 0.02 they track a hundredfold
  finer input mesh to within a point. A converged solve is minutes on anything
  larger than a leaf, so it runs three times (before, after deleting, after
  shrinking), while `run_open_pex(capacitance=False)`, which is exact and takes
  seconds, gates every batch and assigns blame shape by shape.
- **Reduce a cell in the context that uses it.** A leaf reduced alone gives up
  landings its parent needs. Reduce the parent and let edits fall in the
  leaves; the report names the cell each edit belongs to. Reducing a whole
  `DriverSliceSpec` (the same inverter, a 4/2-fin NAND, `--no-capacitance`,
  1223 probes, 9 minutes on 12 cores) deletes the same row-1 strap and V0 in
  the inverter leaf, but shortens the row-0 strap only to 68-171, where the
  slice router lands, rather than to the landing at the label; trims three NAND
  input bars; and keeps the inverter's row-1 LIG because the NAND-to-inverter
  net, `int[B0,SEL,WL0]`, would go 384 to 457 ohm. That net has no pin: nets
  inside the cell are measured from driving diffusion to driven gate, and
  named by the pins around them because the extractor's numbering does not
  survive an edit.

The output is a list of findings, not a generator: promote what it finds into
spec knobs. These became `InverterSpec.input_rows` and `input_reach`, and
`DriverSliceSpec.trim_driver_input`, which asks each driver for one row-0
landing reaching from the router's two landings to the gate contact. Built
that way the inverter extracts to the reducer's numbers, and
`tests/test_reduce.py` offers it back to the reducer, which takes nothing more
from the input track and still refuses the gate tie.

## Geometry parity

These helpers were *copied* out of chipforge's compilers, which still carry
their own inlined copies — so the two can drift. `tests/test_grid_parity.py`
builds a real `ROM_FULL`, `BROM_ARRAY_SLAVE`, `PNMOS_FULL` and `ROM_BitCell`
with those scripts, asserts identical array FIN / GATE / GATE_CUT polygon
sets, and checks bitcell fin centering. It finds the chipforge checkout
automatically (or set `CHIPFORGE_ROOT`) and skips cleanly when there isn't one,
so the suite still passes standalone. The standalone FinFET intentionally uses
the released standard-cell device/tap topology instead of the compact,
macro-specific ROM bitcell topology.

## Development

```bash
uv sync --all-extras   # dev group + the gds extra
uv run pytest
uv run ruff check src tests && uv run ruff format src tests
uv build               # wheel + sdist into dist/
```
