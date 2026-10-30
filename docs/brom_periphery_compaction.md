# CP-SAT peripheral routing compaction

`--compact-periphery` reduces the area of the CGEN, decoder, output leaf,
controller, output register and integrated BROM peripheral bank. It applies
the GT2N library's constrained-routing approach to ASAP7 geometry. It preserves
every transistor, fin count, functional placement group and logical connection,
including the local SAMPLE receivers and all-latch closure return.

Install `chipforge-asap7[gds,optimize]` to use it. The default folded layout
remains available; importing the library does not require OR-Tools.

```sh
asap7-brom words.json --bits 4 --self-timed --registered-output \
  --compact-periphery --wire-rc-controls --drc --lvs --out build/compact_rom
```

For Python callers, select the layout before sizing:

```python
c = compact_periphery(self_timed_macro_circuit(spec, words))
size_wire_rc_distribution(c)
top = build_macro(c)
```

## What transfers from GT2N

The sibling `chipforge_gt2n` uses CPCell/CP-SAT for fixed-core CGEN routing
(`docs/brom_cgen_search.md`, `periphery/brom_cgen_lane.py`) and constrained
functional-unit placement (`periphery/dense_io.py`). Its diffusion chains,
buried rails, device dimensions and metal rules are technology-specific.
This implementation transfers the routing optimization method, using the
existing independently verifiable ASAP7 device cells. It does not port GT2N
polygons or introduce shared-diffusion transistor cores.

The old bank allocated one 96 nm-pitch M4 track per net, a 96 nm-pitch M5
spine per cross-row net, and space for a full device row even when partly empty.
The compact bank instead:

1. Reserves horizontal space for the widest occupied row.
2. Routes M4 on a 48 nm pitch, with endpoints aligned to 48 nm. Alignment
   avoids short tip staggers and sub-44 nm parallel runs.
3. Uses CP-SAT to share tracks between nonoverlapping routes, including the
   40 nm M4 end-spacing keepout. Real tips that meet in x must be separated
   by two tracks to meet corner clearance.
4. Places M5 spines on their legal 48 nm grid, adding an empty track where
   adjacent spine tips meet or have insufficient overlap.

External pins reserve their entire escape to x=0, so track sharing cannot
short macro pin escapes. Route verification still checks every transistor
terminal and macro interface against the intended net. The wire-RC estimator
uses the same bus spans, spine locations and compact plan as the GDS builder.

## Scope of the optimality certificate

Each row's CP-SAT model minimizes its number of routing tracks for fixed
device positions, row packing and spine order. A greedy legal assignment
provides an upper bound; a simultaneous interval clique supplies an independent
lower bound. Corner-spacing constraints can require more tracks than the
clique bound. Only solver status `OPTIMAL` is accepted; a timeout is an error,
not a silently accepted minimum. Search uses one worker and a fixed seed.

The manifest records each row's intervals, track assignment, solver status,
objective/bound and clique witness, for the macro and every standalone unit.
These are channel-width certificates, **not proofs of globally minimum
transistor layout or macro area**. Device ordering, diffusion sharing and
alternative floorplans remain outside this search.

## Measured geometry

The following comparison uses the same two-fin circuits, with no control
sizing, keeper or netlist simplification. Areas are actual emitted bounding
boxes, including their boundary margins, in µm².

| Circuit | Folded area | Compact area | Reduction |
|---|---:|---:|---:|
| CGEN | 185.46 | 107.85 | 41.85% |
| 16×4 decoder | 305.36 | 185.88 | 39.13% |
| Precharge/inverter output leaf | 23.09 | 4.08 | 82.32% |
| Controller | 122.22 | 84.68 | 30.71% |
| Four-bit output register | 191.85 | 128.00 | 33.28% |
| Registered self-timed 4×4 macro | 4201.08 | 2594.74 | 38.24% |

The 4×4 macro uses `[10,12,0,0]`. Its peripheral bank shrinks by 47.82%; the
whole macro shrinks less because selector geometry and array escape routing
are unchanged. The output leaf's large reduction includes removal of unused
row capacity. These percentages do not predict savings for every macro shape
or a differently sized control network.

Reproduce the comparison, including geometry hashes and row certificates:

```sh
PYTHONPATH=src python scripts/brom_periphery_area.py \
  --out build/brom_compact/area.json
```

## Verification and timing boundary

The compact physical regression covers standalone units and external-phase,
clocked and self-timed macros; one-, two- and three-fin devices; mux1/mux4;
RVT/LVT; keepers, registered outputs and control-buffer variants. It runs full
ASAP7 gdscheck and transistor LVS, including exact pin names and metal route
connectivity. Targeted tests check the interval bound, touching-tip constraint,
certificate witnesses, reduced area, preserved circuit inventory and RC layout
selection. Compiler artifacts are under `build/brom_compact/registered4x4` and
`build/brom_compact/registered4x2_wirerc`.

Compact tops use a distinct `_cp` identity. Old folded-layout timing reports
do not match their circuit hash. The replica timing CLI also accepts
`--compact-periphery`, selecting it before wire-RC sizing and netlist creation.
The earlier SAMPLE-closure margins were measured on the folded layout and must
not be attributed to the compact layout. New RC timing runs, especially
fast-return versus slow-receiver closure checks, are needed before qualifying
a compact registered schedule. Physical and estimated timing qualification
flags remain false.
