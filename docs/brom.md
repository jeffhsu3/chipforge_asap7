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

## Integration views

`asap7-brom` writes `macro.lef` and `macro.v` beside the macro's GDS; the
same text comes from `macro_lef(circuit, top)` and
`macro_verilog(circuit, spec, words)` in `chipforge_asap7.brom.views`.

- **LEF** (5.8, `CLASS BLOCK`) is read off the built GDS. Its origin is the
  boundary's lower-left corner, and `FOREIGN` records where the GDS origin
  then sits, since the macro's routing reaches below and left of it. Every
  pin is an M4 bar that runs from its label out to the left edge, where a
  router can reach it; `build_macro` draws that bar for exactly this.
  VDD and VSS are one such bar each, so a PDN must strap onto them.
  M1-M3 are blocked over the whole footprint and M4-M5 only where drawn,
  which leaves M6 and up free over the macro.
- **Verilog** is a zero-delay behavioral model with the ROM contents inline.
  It follows the transistors rather than an idealized read: address latches
  open while PRECHN is low, bitlines are dynamic and keeperless (they stay
  discharged until the next precharge), unused address codes read zero, and
  Q latches are transparent while SAMPLE is high. The clocked macro carries
  the controller's equations with the four acknowledgments as ports; the
  self-timed one ties them to their meanings with zero delay. Under
  `SYNTHESIS` the module body drops out, leaving a black box.

Both views name bus bits `A[3]`, as a gate netlist does; the GDS labels and
CDL keep `A<3>`. Neither view carries timing: there is no Liberty file yet,
since it needs characterized, extracted timing (`timing_qualified` is still
false). Tests check the LEF pins against the GDS labels and read it with
KLayout, and run the Verilog under Icarus against the transistor switch-level
model on random reads, comparing Q and, where the macro generates them,
PRECHN, EVAL and SAMPLE at every step.

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
and hold the sampled address throughout evaluation. There is no bitline keeper.

Depth is a positive multiple of `4 * ways`; ways is 1 or 4. Non-power-of-two
depths are allowed. Unused address codes select no block. With mux4, the
lowest two address bits select the column, the next two select the word
offset, and the remaining bits select the block. Physical column
`bit * ways + way` belongs to output `bit`.

## Read interface

The macro exposes `A<i>`, `PRECHN`, `EVAL`, `SAMPLE`, `Q<i>`, VDD and VSS.
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
with `--self-timed`. Extracted interconnect timing is not yet qualified. GT2N's
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

The packaged `selector_plans.json` supplies the qualified track plans for
one, two and three fins; compilation does not require OR-Tools. Unsupported
fin counts and arrays exceeding the current source-tap reach are rejected.
The current limit is 69 four-word blocks (depth divided by `4 * ways`).

CGEN, decoder, precharge, mux and latch devices occupy a peripheral bank
above the core. Private M3 accesses and M5 trunks connect the array candidates,
bitlines and VSS to that bank. The transistor references include the array's
isolation devices. Internal array labels are removed from the macro interface.
The peripheral bank uses folded rows of individually tapped devices.
Functional CGEN, address-latch, readout and output-latch groups stay together.
Device placement uses multiples of the 108 nm selector pitch, and each row
starts on the 432 nm block lattice. Array width sets row capacity, with an
eight-device minimum for a complete latch and a 32-device upper limit.

Each row has local M4 net segments; M3 terminal accesses stop at their own
net. Only nets crossing rows receive M5 distribution spines. Candidate,
bitline and public pins escape directly from their owning row, avoiding a
second bank-wide bus. The macro's M2/M4 candidate and M3 bitline interfaces
remain connected through the explicit array-to-bank routes. Peripheral rows
span multiple selector periods; this is not a one-cell-per-bitline layout.

The 16-word × 3-bit mux4 example decreased from 328.140 × 26.898 µm to
20.220 × 100.428 µm, or from 8,826.31 to 2,030.65 µm² (77.0% less bounding-box
area). These are generated layout dimensions, not a minimum-area proof or
a timing result. The selector proof does not establish minimum macro area.

The previous discrete-device **macro array layout has been removed**, with
no legacy layout option or fallback. Use `build_macro(macro_circuit(...))`
for macros and `build_periphery(...)` for standalone peripheral circuits.
`build_circuit` is a deprecated compatibility shim dispatching to those
builders; it cannot generate the old macro layout. `build_periphery` rejects
macro circuits to prevent accidentally rebuilding a discrete-device array.

The manifest records the selected plan, core dimensions, peripheral origin,
array ports and device counts, plus peripheral row capacity, device pitch,
row count, dimensions, shared-spine count and pin positions. Every peripheral
build audits all terminal vias and pins for metal opens and shorts, including
both supply nets, without substituting well/substrate continuity. Every macro
build also checks the actual metal connectivity of each array-to-bank route,
including VSS, and rejects opens or shorts. This supplements LVS: substrate continuity can otherwise mask a
missing VSS metal bridge. CLI LVS checks also require the exact public pin
inventory and names.

Tests cover all 16 patterns, mux1/mux4 addressing, multiple blocks, all three
supported fin counts, address/output holding, base-geometry invariance,
full DRC, transistor LVS and deliberately disconnected array interfaces.
Switch-level checks do not establish setup/hold, leakage, PVT limits,
access time or extracted-RC behavior. The separate [transistor timing bench](brom_timing.md)
measures loaded BSIM-CMG transients; `timing_qualified` remains `false` pending
extracted-interconnect and broader timing qualification.
