# ASAP7 blocked ROM

`chipforge_asap7.brom` implements the four-word BROM architecture used in
`chipforge_gt2n`, separately from the poly-wordline NOR-ROM compiler. It
generates transistor CDL, unit-fin LVS references and routed GDS for the
macro, CGEN, address decoder and precharge/output leaf.

```python
from chipforge_asap7.brom import BromSpec, macro_circuit, build_macro

spec = BromSpec(depth=16, bits=8, ways=1, fins=2, vt="rvt")
circuit = macro_circuit(spec, list(range(16)))
cdl = circuit.netlist()
cell = build_macro(circuit)  # nanometre coordinates
```

Compile a JSON array of unsigned integer words:

```bash
python -m chipforge_asap7.brom words.json --bits 8 --ways 1 \
  --out build/my_brom --drc --lvs
```

The installed command is `asap7-brom`. Each output directory contains
`macro`, `cgen`, `decoder` and `output` GDS/CDL/LVS files, plus
`manifest.json`; the macro also gets a LEF abstract and a Verilog model
(see [Integration views](#integration-views)). `--drc` runs the full ASAP7 gdscheck suite; `--lvs` runs
the packaged KLayout unit-fin extractor, including real body connections.
Either failed check makes the command fail. Checks omitted from the command
are recorded as `not_run`, and an old successful manifest is invalidated
before a new build starts.

Dense shared-diffusion CMOS is the default. Install `chipforge-asap7[gds,optimize]`
for the layout compiler. `--compact-periphery` is a deprecated no-op. Dense
layouts require their own timing characterization; earlier channel-layout
results are historical.

## Integration views

`asap7-brom` writes `macro.lef` and `macro.v` beside the macro's GDS; the
same text comes from `macro_lef(circuit, top)` and
`macro_verilog(circuit, spec, words)` in `chipforge_asap7.brom.views`.

- **LEF** (5.8, `CLASS BLOCK`) is read off the built GDS. Its origin is the
  boundary's lower-left corner, and `FOREIGN` records where the GDS origin
  then sits, since the macro's routing reaches below it. Every pin is an M5
  column that runs from its cell's row down to the bottom edge, where a
  router can reach it; `build_macro` draws that column for exactly this.
  VDD and VSS are one such column each, so a PDN must strap onto them.
  M1-M3 are blocked over the whole footprint and M4-M6 only where drawn,
  which leaves M6 and up free over the macro.
- **Verilog** is a zero-delay behavioral model with the ROM contents inline.
  It follows the transistors rather than an idealized read: address latches
  open while PRECHN is low, bitlines are dynamic and keeperless (they stay
  discharged until the next precharge), unused address codes read zero, and
  Q latches are transparent while SAMPLE is high, with the rising-edge
  output register after them when the macro has one. A bitline keeper only
  holds a level the model already holds, so it adds nothing to the model
  (and is the one option the switch-level comparison below cannot cover,
  since it cannot settle the keeper's deliberate fight with a selector). The clocked macro carries
  the controller's equations with the four acknowledgments as ports; the
  self-timed one ties them to their meanings with zero delay. Under
  `SYNTHESIS` the module body drops out, leaving a black box.

The macro's bus pins are named `A[3]` and `Q[3]` everywhere (GDS labels,
CDL, LEF, Verilog), as a gate netlist and DEF name them, so chip-level LVS
needs no pin map; nets inside the macro keep `<>`. Neither view carries timing: there is no Liberty file yet,
since it needs characterized, extracted timing (`timing_qualified` is still
false). Tests check the LEF pins against the GDS labels and read it with
KLayout, and run the Verilog under Icarus against the transistor switch-level
model on random reads, comparing Q and, where the macro generates them,
PRECHN, EVAL and SAMPLE at every step.

Every cell a macro carries is named by its contents, so macros merged into one
chip GDS cannot clash: the macro and selector array by a hash of their
programming, and each peripheral transistor cell (`brom_<device>_<hash>`) by
a hash of its drawn shapes, so a device drawn differently by another compiler
version gets a different name rather than silently replacing this one.

## Circuit and programming

Each block encodes four words. For output column `c`, its selector index is
`k = sum(bit(word[offset], c) << offset)`, for offsets 0 through 3.
The decoder generates `G[k] = (k >> offset) & 1`, and each block's CGEN
generates `C[block,k] = SEL[block] & G[k]`. One NMOS per block and physical
column discharges the bitline when its programmed candidate is high.

The sixteen G functions follow GT2N's literal/NOR/NAND/AND/OR/XOR mapping.
The CGEN contains sixteen six-transistor AND2s, with SEL at the ground side
of each NAND stack. Block decoding uses a shared binary prefix tree.
`SEL` is gated by both EVAL and PRECHN; precharge suppresses selection even
if EVAL is asserted. This Boolean interlock does not replace timing margins.

The output leaf contains a PMOS precharger and a CMOS inverter. At 4:1,
four physical columns feed each output through a static one-hot mux.
An eight-transistor static transmission-gate latch holds each output while
the next read precharges. Address latches are transparent during precharge
and hold the sampled address throughout evaluation. The default is keeperless; an optional [feedback keeper](brom_keeper.md)
can be enabled with `--keeper-fins 1`.

Depth is a positive multiple of `4 * ways`; ways is 1 or 4. Non-power-of-two
depths are allowed. Unused address codes select no block. With mux4, the
lowest two address bits select the column, the next two select the word
offset, and the remaining bits select the block. Physical column
`bit * ways + way` belongs to output `bit`.

## Read interface

The macro exposes `A[i]`, `PRECHN`, `EVAL`, `SAMPLE`, `Q[i]`, VDD and VSS.
PRECHN is active low; EVAL and SAMPLE are active high.

1. Keep SAMPLE low. Deassert EVAL and let all candidates fall before
   asserting precharge (`PRECHN=0`). Apply the next address.
2. Let the bitlines restore and the address decoder settle. Deassert
   precharge (`PRECHN=1`) to close the address latches before asserting EVAL.
   Respect address setup/hold around that transition.
3. Let the selected bitline discharge and the readout/mux settle.
   Assert SAMPLE to open the output latches.
4. Deassert SAMPLE to hold the result, then deassert EVAL before the next
   precharge. Q is unspecified until the first completed sample.

SAMPLE is a level-sensitive latch enable, not an edge-triggered clock.
An optional [clocked controller](brom_controller.md) now generates these phases
from CLK, RESETN and four external completion acknowledgments (`--clocked`).
The experimental [selector replica](brom_replica.md) supplies internal feedback
with `--self-timed`. Either controller mode supports an optional
[rising-edge output register](brom_registered_output.md) with `--registered-output`.
Extracted interconnect timing is not yet qualified. GT2N's
clock delays, replica sizing and read-cycle numbers are not applicable to these
ASAP7 devices and wires.

Optional [sized control distribution](brom_distribution.md) adds tapered
drivers based on connected gate loads with `--sized-controls`. Wire-RC timing
and receiver-specific feedback remain unqualified. The alternative
[wire-RC-aware search](brom_wire_rc.md), `--wire-rc-controls`, includes routed
length estimates and moves buffer groups toward their drivers and loads.

## Physical implementation and migration

The macro now exclusively uses the compact periodic selector array from the
[OR-Tools CP-SAT proof](brom_minimum.md): **108 × 432 nm per four-bit selector**.
Each selector has one live NMOS and one grounded isolation NMOS. Candidate
conductors occupy eight M2 and eight M4 tracks; bitlines use M3. Programming
changes only V2/V3 cuts. All other geometry is independent of ROM contents.

The packaged `selector_plans.json` supplies the qualified selector track plans
for one, two and three fins. The dense peripheral generator requires the
`optimize` extra (OR-Tools) as well as `gds`. Unsupported fin counts and arrays
exceeding the current source-tap reach are rejected. The current limit is 69
four-word blocks (`depth / (4 * ways)`), including extra replica blocks.

Periphery now uses paired NMOS/PMOS rows on a 54 nm gate grid. CP-SAT places
transistors with matching diffusion endpoints, compatible fin counts, legal
92 nm diffusion breaks and shared gate columns. A second constrained solve
routes actual M1/M2 access boxes and M3 trunks. Uncontacted series diffusion
nodes stay inside the cell. Rows share supply rails and taps; there are no
individually tapped transistor tiles. Placement certificates distinguish a
proved gate-slot bound from routing width and overall macro area: this is
**not a global minimum-area proof**.

The two selector halves flank a central CGEN spine. Each 432 nm block has
sixteen dense AND2 cells per half. Complete mux groups stay on one side,
including odd output widths. Independent left/right candidate drivers are
present in both the schematic and layout. Their output inverters can be folded
into parallel fingers during sizing. The decoder, address latches, controller,
readout and register cells occupy the bottom strip. Local SAMPLE receivers,
latches and closure observers follow the circuit's physical ordering.

Peripheral rows now abut on a 432 nm pitch, with alternate rows reflected
vertically to share VDD/VDD and VSS/VSS rails. A cell edge whose outer
diffusion in every band ends on that band's supply is *ready*. Its supply
contacts sit on the cell boundary, so two ready edges abut on one shared
contact and their islands merge. The device placement first proves the
minimum gate-slot count, then makes as many edges ready as that count allows.
Every other edge keeps its outer contacts at least 54 nm inside. Neighbouring
edges then need a 0, 54 or 108 nm spacer: either a merged contact or a 92 nm
diffusion break. Cell widths are 54 nm multiples. Near an edge, M2 keeps to
the lower half of the cell on the left and the upper half on the right, so
any two cells abut without a metal conflict. Spacers, fillers and taps close
every gap, so each row is one continuous frame. Most periphery cells end on a
signal net (an output or a series node) on at least one side. In the measured
banks only a few percent of neighbouring pairs actually merge diffusion. Most
of the width saved comes from dropping the 108 nm cell rounding and from M3
trunks usable at the cell edges.

In a macro, the bottom bank is placed against the array above it. A quadratic
(clique-model) placement pulls each cell toward the cells it shares nets with
and toward the bitline, CGEN `G` and `SEL` drop points. Abacus-style
legalization then packs the rows within the body-tap reach, keeping each
pair of edges' spacer. Each SAMPLE receiver, latch and closure observer stays
one contiguous run. Cells with bank ports prefer upper rows, since their header
columns rise through every row above them; cells with macro pins prefer lower
rows, since their pin columns descend to the bottom edge. The bank routes several row counts
and keeps the one that puts the arrays lowest: fins and M4 tracks lie on
absolute 27 and 48 nm grids, so the arrays start on a multiple of 432 nm, one
track pitch above the highest header track and the join stubs below the
array. A row count that does not route is retried with its cells spread
toward uniform density. If nothing routes inside the array
footprint, the bank falls back to electrical-order rows: at each width it
tries rows filled to 100%, 75% and 50%, with the cells spread across the
width, before widening by 10%. Standalone banks always use electrical order.

Local M4 escapes stay over the cells. Same-net pins within 864 nm share one M5
column, which spans only the rows it serves; columns in disjoint row ranges
share an x position. One M6 bus per net joins its groups, in the row needing
the least column extension that has room. Rows above the cells take only
overflow buses. Most bank ports are *anchored*: the port's column is asked to
sit where the parent can join it directly, so the port needs no header
track, and the cell's M4 row escape absorbs the offset between the cell and
the column. Each bitline's column sits on the lattice slot nearest its
array drop and rises to the join stub below the array; each select's column
is its own M3 riser slot in the centre strip; fourteen of the sixteen `G<k>`
columns rise into the block-0 lane row, where the net runs inward on a lane
M4 or M6 track (below). An anchor that finds no free slot, or whose escape
would overfill a row, falls back to a header track. Macro pins are the M5
columns of their cells, run down to the bottom edge; they take no header
track either. What remains in the header are the supplies, the two
outermost `G<k>`, and any fallbacks, each spanning only its columns and
drops; disjoint spans share tracks. Buses that overflow above the rows are
packed from the lowest free M6 track upward and balanced against M4 header
tracks so the two layers end level; a
port whose own bus moves onto M4 has every column rise to its one track.
M5 and M6 line ends lie on a 48 nm lattice, which satisfies ASAP7's tip
stagger, parallel-run and same-track rules by construction. The packers check
the remaining corner rule directly. When a row needs more than its eight M4
escape tracks, an anchored port crossing the crowded x moves its run to the
row with the most room: a short escape to a column near its cell, which
rises to that row, an M4 jog there to the anchored column, and the anchored
column on up. Failing a row with room, the widest other group crossing is
split in two, each half with its own column; failing that, the widest
anchored port is released to a header track. The bank is then routed
again. Columns of one net never take adjacent tracks where their spans
overlap, which would put two of their vias 24 nm apart; nor do anchored
columns rising to M4 and to M6 lane tracks, whose ends can differ by less
than a tip stagger. Array connections use the clear M3 space above the cell
rows. Wide banks gain periodic body taps.

This removes the initial dense implementation's 1,944 nm and 1,512 nm gaps
between its three 4×4 peripheral rows. That intermediate macro occupied
145.660 µm²; the first abutting-row version occupied 84.734 µm², anchored
placement with the row-ranged router 75.367 µm², edge-ended pin tracks
with the tighter array margin 68.130 µm², anchored ports with bottom
pins and lane-track `G<k>` trunks 53.503 µm², and tightened array join stubs
with replica-wide anchoring 45.489 µm². Routing still contributes to
the macro area; this is not a proof of minimum macro area. The regression
checks actual shared-rail connectivity, continuous row frames, zero row gaps and
the drawn area.

The original selector lattice and its eight M2/eight M4 candidate tracks are
unchanged. Over the CGEN, six candidates use M6 and descend to M2 outside the
array; two use direct M2 and eight use M4. This avoids shorts between global
candidates and the dense cells' local wiring. Every candidate trunk runs
outward from its cell, so each lane M4 and M6 track is free from the
centre up to the cell it serves; the `G<k>` inputs run inward there, from
one lane's G rail to the mirrored lane's, meeting the bank's riser between
them. Placing the two M2-direct cells first frees the inner stretch of the
first M4 and M6 tracks as well: outer cells take the tightest M4 track that
reaches them, inner ones an M6 track reached by a jog on the first M4 track
and an M5 stub. The two outermost cells no track reaches (one M6 and one M4)
keep their header tracks. Supply connections cross the array on free M4
boundary tracks. The independent replica occupies the far edge and retains
its own candidate loading, bitlines, readouts and guards.

The four-way output mux is now two AOI22 gates and a NAND2 (20T versus 42T).
Its complete 256-entry truth table is tested, including non-one-hot selects.
Readout cells combine precharge, inverter and optional series keeper devices.
The 8T sampling latches and 16T master/slave output registers retain their
original electrical topology. Controller feedback, all-latch SAMPLE closure
and the register's both-rail clock interlock are preserved.

The former discrete-device peripheral layout has been removed. Dense CMOS is
the default for `build_macro` and `build_periphery`; `--compact-periphery` and
`compact_periphery()` remain deprecated no-ops. `build_periphery` rejects a
macro circuit. `macro_plan` returns the actual dense physical-plan dictionary;
`periphery_plan` returns the dense bank and its ports, units, routes and metal
checkpoints. Callers should no longer use old device-row pitches or bus offsets.

GDS, LEF, wire estimates and timing provenance use the same physical plan.
LEF covers M1–M3 and emits exact M4–M6 obstructions. The manifest includes
placement certificates, device inventories, the physical plan and geometry
fingerprints. Timing attachments must match both the circuit and physical
fingerprints; regenerate old reports. Actual hierarchical metal is checked for
opens and shorts, including programmed gate stems and supplies. CLI LVS also
requires exact public pin names and inventory. Shared diffusion requires
`merge_fin_diffusion=True` in the unit-fin LVS extractor.

Measured two-fin RVT examples, compared with the archived channel-compacted
layout in `build/brom_compact/area.json`:

| Layout | Previous area | Dense area | Reduction factor |
|---|---:|---:|---:|
| Standalone 96T CGEN | 107.845 µm² | 3.149 µm² | 34.2× |
| Registered self-timed 4×4 | 2,594.74 µm² | 45.489 µm² | 57.0× |

The external-phase 16×3 mux4 example is 16.416 × 1.907 µm (31.305 µm²).

Compared with the intermediate dense channel layout in
`build/brom_dense/area.json`, external-phase mux4 macros shrink as follows:

| Macro | Channel layout | Dense | Size |
|---|---:|---:|---|
| 16×4 | 125.107 µm² | 38.397 µm² | 16.416 × 2.339 µm |
| 64×16 | 496.303 µm² | 97.178 µm² | 21.600 × 4.499 µm |
| 1024×16 | 2,512.173 µm² | 1,420.464 µm² | 38.496 × 36.899 µm |

Shared-diffusion cells make the 16×4 bank's cells 30% narrower in total
(24.3 → 17.0 µm) but did not shrink the 16×4 or 64×16 macro by themselves:
the 16×4 height is set by its header, and rows of narrower cells crowd their
M4 escapes and M5 columns, so the 64×16 placement spreads the cells back out
to route. Ending each external pin's track at the nearer edge and tightening
the array margin then cut the header band: 16×4 59.847 → 52.888 µm² and
64×16 144.154 → 125.728 µm². Anchoring the bitline, select and `G<k>`
ports, taking the pins out through the bottom edge, tightening the array join
stubs and balancing the
overflow buses between M6 and M4 cut it again: 16×4 52.888 → 38.397 µm²
and 64×16
125.728 → 97.178 µm², and 1024×16 1,655.349 → 1,420.464 µm², where the
bottom pins and the buses moved onto M4 shorten the fallback's band. The
1024×16 fallback trades whitespace rows for width: it was 2,035.234 µm²
(48.864 × 41.651 µm) with padded cells.

The 64×16 bank now fits the array width. The 1024×16 bank does not route with
anchored placement inside the 24.6 µm array footprint: its M6 buses saturate
the rows and their columns then fill the upper rows. It uses the
electrical-order fallback, 38.496 µm wide, with its ports released to
header tracks. In small
macros the band above the rows still sets the bank height: at 16×4 it holds
four port tracks (the supplies and the two `G<k>` no lane track reaches)
plus the overflow buses, 0.86 µm of the 2.34 µm macro height. At
64×16 a readout slice is wider than four column pitches, so readouts drift
up to 2.3 µm from their drops; the jogs spread those runs over all four
rows' M4 channels and anchor 63 of the 64 bitlines, the last one taking a
header track. The band there is set by the overflow buses, 16 of them from
the DOUT nets, rather than by port tracks. Area remains an optimization
limitation even when DRC/LVS pass.

These are drawn bounding boxes, including routing and taps. The macro comparison
includes duplicated half-array drivers. It is not a promise of 100× reduction
or a comparison normalized across process technologies. Regenerate measurements
with `scripts/brom_periphery_area.py --matrix --reference
build/brom_compact/area.json --out build/brom_dense/area.json`.

The physical shape matrix covers 4×1, 4×4, 4×16 and 16×9 mux1, plus 16×4,
64×16 and 1024×16 mux4. A separate option matrix covers all fin counts,
keepers, registered outputs, external/clocked/self-timed interfaces and both
sizing passes. Negative tests remove programming and distribution vias and
bridge independent nets. These checks do not establish setup/hold, leakage,
PVT limits or extracted-RC timing. `timing_qualified` remains `false`.
