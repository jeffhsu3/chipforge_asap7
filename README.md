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
# from another uv project, tracking the checkout live
uv add --editable /home/jeff/iv4/repos/chipforge_asap7

# or pin it in that project's pyproject.toml
#   [project]
#   dependencies = ["chipforge-asap7[gds]"]
#   [tool.uv.sources]
#   chipforge-asap7 = { path = "/home/jeff/iv4/repos/chipforge_asap7", editable = true }

# plain pip works too
pip install /home/jeff/iv4/repos/chipforge_asap7           # constants + grid math
pip install "/home/jeff/iv4/repos/chipforge_asap7[gds]"    # + the draw_* helpers
```

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

```bash
uv run asap7-inverter --rows 18:18,13:13 --fingers 2 --out build/inv.gds
```

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
