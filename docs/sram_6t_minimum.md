# Conditional minimum-area proof for the ASAP7 6T bitcell

`chipforge_asap7.verification.sram_6t_minimum` proves that the retained
`sram_cell_6t_122` template needs at least **108 × 270 nm = 0.02916 µm²/bit**.
The measured reference cell attains that bound. This is the placement-boundary
area of one bit; it excludes taps, dummies, array edges and periphery.

The workflow uses the same pinned gdscheck authority and shared SMT/reporting
helpers as the [NOR-ROM proof](rom.md#minimum-area-proof-for-this-topology).
It checks the reference GDS from
`../OpenFinRAM/tech/gds/srambank_32b_boundary_2.gds`, including the leaf and its
`array_x2x4` assembly.

## Scope: retained template versus SRAM rules

The 6T cell has an `SRAMDRC` marker. In the pinned gdscheck PDK, ordinary
gate/fin lattice rules and ACTIVE fin enclosure checks exclude this region.
Consequently, the proof explicitly treats the following as **template
assumptions**, rather than requirements imposed on this SRAM by gdscheck:

- Two distinct gate columns on a 54 nm lattice, with 20 nm gate length.
- Four separate horizontal ACTIVE bands in N/P/P/N order. PD and PG share
  diffusion within each N band.
- Band fin counts of 2/1/1/2, corresponding to PU/PD/PG sizing of 1/2/2.
- A 27 nm fin lattice, 7 nm fins, and 10 nm ACTIVE enclosure of the outer fins.
- Rectangular periodic tiling, with separate ACTIVE bands across the vertical
  boundary. Merging or staggering bands is outside the template.

The model also includes the applicable 13.5 nm SRAM well enclosure/spacing
rules and a necessary 17 nm ACTIVE height for the SRAM SDT overlap rule.
The [rule manifest](../src/chipforge_asap7/verification/sram_6t_minimum_rules.json)
separates applicable rule facts from retained template policies and identifies
their sources. All remaining DRC checks are applied to the concrete witnesses.

Two gate columns need at least two gate pitches, giving 108 nm width. The
fin width plus two ACTIVE enclosures is exactly one fin pitch:
`7 + 2 × 10 = 27 nm`. Strict separation between bands therefore needs one
unused fin track per band, including the periodic boundary. Six occupied
tracks plus four unused tracks give `10 × 27 = 270 nm` height. The SMT model
uses integer lattice indices and real ACTIVE/well edges; it allows asymmetric
gaps and does not assert either desired dimension as a lower bound.

This does **not** establish optimality over all SRAM-rule layouts. The runner
saves a fifth query with the 10 nm enclosure policy removed. That query is
SAT: the current Z3 run finds a **108 × 216 nm** abstract model. It is a scope
counterexample, not a routed or DRC-validated smaller bitcell.

## What is checked

Three UNSAT queries exclude a smaller width, a smaller height, and either
smaller dimension. A fourth SAT query binds every model coordinate to the
actual GDS, establishing that the bounds are attained rather than vacuous.
Z3 proof terms and SAT models are saved for replay.

The GDS checks measure BOUNDARY, SRAMDRC, GATE, FIN, ACTIVE and NWELL polygons.
Independent connectivity extraction requires exactly the six transistors,
their 1/2/2 fin sizing, cross-coupled storage nodes and WL/BL/BLN/supply pins.
Ideal-switch checks cover both stored values in hold and read-access modes.
The source leaf has two separate VSS landing roots that must be connected
externally; repeated supply labels are treated as external aliases.

Full pinned gdscheck `main` runs must pass on both the leaf and the supplied
array. The array must actually instantiate the measured leaf. These checks
cover the finite supplied witnesses, not every possible array construction.

Tests also remove essential model constraints, shrink the witness, remove a
fin, move a gate off-grid, break a feedback connection and change a pinned
rule. These changes must either admit a smaller abstract model or fail the
corresponding validation.

## Run and inspect

Install Z3 and build gdscheck at revision
`0b383c3b07edd6247586aca812e392928dc3b872`. The runner needs `gdstk` in addition
to the project dependencies:

```bash
uv run --with gdstk python -m chipforge_asap7.verification.sram_6t_minimum \
  --gds ../OpenFinRAM/tech/gds/srambank_32b_boundary_2.gds \
  --gdscheck-root ../gdscheck --out build/sram_6t_minimum
```

`--z3` and `--gdscheck` select executables explicitly. `--cell` and
`--array-cell` select named witnesses in another GDS containing the same
template. The runner rejects changes to any of the 35 pinned ASAP7 PDK files
and passes the pinned `pdk.yml` explicitly to gdscheck.

The output directory contains `report.json`, the SMT queries and solver
outputs, `witness.gds`, and DRC reports. The JSON report records assumptions,
limits, extracted topology, source/implementation/tool hashes and the attained
minimum. Failures overwrite any previous success status. A successful report
has status `proved_for_retained_6t_122_template`.

The trusted boundary includes rule transcription, mapping the template to
constraints, Z3, and the geometry/connectivity tools. Saved proof terms are
not independently checked. Read stability, write margin, analog bistability,
timing, leakage and PVT behavior remain outside this geometric proof.
