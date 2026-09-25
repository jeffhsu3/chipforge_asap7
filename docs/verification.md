# Simulation and verification

## Single-device SPICE testbench

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
data (as `.dat` columns and full `.raw` files), simulator logs, and
`results.json`. This is a **compact-model test**, not
a post-layout proof. The generated GDS is covered by the KLayout DRC and
layout-connectivity regressions described in
[the cell generators](devices.md). A formal post-layout proof
is provided by the open flow below; it remains a calibrated research flow
rather than a replacement for the unpublished sign-off decks.

## Open LVS and PEX

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
16-transistor `sense_amp_sram` topology (`sense_amp_transistors()`, the
circuit `SenseAmpRowSpec` draws plus its four output dummies):

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

### Looking at the results

Both benches take `--plot` (with the `plot` extra installed) and draw the run
to a PNG beside `results.json`, with the bench's own pass criteria drawn on it.
They plot whether the run passed or not, since a failing run is the one worth
looking at:

- `dc.png` for a FinFET: Id-Vg on a log axis with the off and on currents the
  verdict is read from, Id-Vd, and the switch's output against the levels it
  must reach. Both flavours are drawn against |Vgs| and |Vds|, so an nFET
  and a pFET read the same way.
- `transient.png` for the sense amplifier: one column per input polarity, from
  just before precharge releases to after the decision, with the controls,
  the inputs, the outputs against the 20 % / 80 % thresholds and the measured
  resolution point, and the supply current.

```bash
uv run python -m chipforge_asap7.devices.sense_amp_spice --plot
uv run python -m chipforge_asap7.devices.spice_plots build/sense_amp_spice build/finfet_spice/nmos_fin_111
```

The second form re-plots finished runs without simulating again.

Each deck also saves every node of its analysis to a `.raw` file
(`transfer.raw`, `sa_high.raw`, ...), including the internal nodes the `.dat`
columns leave out, such as the latch's `N52`/`N57`. Open it in an analog
waveform viewer (xschem's, gaw) or read it in Python with `spicelib`, when a
case fails for a reason the fixed plots do not show. For a quick look without
either, `ngspice` interactive mode can load it: `load sa_high.raw` then
`plot v(qa) v(qan)`.

## LVS of an assembled macro

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

## Layout reduction

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
