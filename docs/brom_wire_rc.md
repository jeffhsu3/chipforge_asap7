# Wire-RC-aware BROM control sizing and placement

`--wire-rc-controls` searches driver sizes and placement using the routing
coordinates of the generated macro. It is an alternative to the gate-only
`--sized-controls` option. Both remain experimental.

```bash
python -m chipforge_asap7.brom words.json --bits 2 --self-timed \
  --wire-rc-controls --out build/brom_wire_rc/compiled --drc --lvs
```

The Python entry point is `size_wire_rc_distribution(circuit, WireRCSpec())`.
Call it on an unsized macro after adding its controller and replica. It searches
on copies and changes the supplied circuit only after selecting a result.
An insufficient resource limit raises an error without changing that circuit.

## Model and provenance

The estimator follows peripheral M2 terminal escapes, M3 terminal columns,
M4 row buses, M5 distribution spines, array-to-bank escapes, candidate tracks
and selector gate access. Lengths include the router's metal end extensions.
The shared macro/bank landing overlap is conservatively counted twice.

Sheet and via resistances come from `verification.stack.STACK_CONDUCTORS`.
The M3–M5 entries are public-geometry extrapolations; their provenance appears
in the manifest. Capacitance uses the existing explicit 0.2 fF/µm wire
assumption. Gate load uses 0.0417 fF/fin, from the repository's TT/RVT logical
effort model. These coefficients are configurable through `WireRCSpec` and
are not corner-independent constants.

Each net reports lengths by layer, wire and gate capacitance, summed segment
resistance, and maximum estimated driver-to-sink path resistance. The latter
walks terminal accesses, row buses and spines, including the candidate escape
and array access where applicable. It chooses the nearest same-polarity driver
copy for each sink and retains the slower of the n/p geometric paths. It does
not assume both transistor polarities conduct simultaneously.

This is a routed-length estimate, **not KPEX/FasterCap extraction**. It excludes
device-local/poly resistance and capacitance, coupling, via capacitance, power
integrity and core bitline RC. Device capacitance remains in BSIM-CMG during
simulation. The wire estimate must eventually be correlated with the existing
open extraction flow; even that flow uses an incomplete public ASAP7 stack.

## Search and placement

The gate-sized circuit is the baseline. The search evaluates relocating buffer
stages next to the original driver and, for later stages, near the median sink
functional group. Complete latch and logic groups stay intact; selector array
geometry and programming remain unchanged.

Further rounds try bounded increases on the four largest estimated wire-RC
cost nets, considering partial and full wire-load compensation. Every candidate
is repacked and rerouted analytically before scoring. This accounts for the
extra metal caused by larger drivers. Blindly compensating all wire loads can
make the bank much larger and slower under the model.

The objective sums estimated inverter drive costs and `Rpath × (Cwire + Cgate)`
along the selected control buffer chains. It is a heuristic comparison, not a
propagation-delay measurement or a global optimum. Defaults permit three sizing
rounds, at most 32 parallel copies per stage and 4096 added devices. The best
candidate, including the unchanged baseline, is retained.

`control_distribution` in the manifest records the complete search, selected
placement, model assumptions, route estimates and final stage-load audit.
`wire_rc_included: true` means estimates participated in the search;
`extracted: false` and `timing_qualified: false` remain explicit. A failed
`stage_effort_target_met` means the target of four was not reached after wire
loading. The search does not silently declare load or timing closure.

## Transient stress check

```bash
PYTHONPATH=src python -m chipforge_asap7.brom.replica_timing \
  --depth 4 --bits 2 --wire-rc-controls --klu \
  --osdi build/brom_timing/model/BSIMCMG107.osdi \
  --out build/brom_wire_rc/tt_path_stress
```

The timing option also inserts a two-capacitor pi approximation on the original
driver output and each added buffer output: estimated path R between driver
and receivers, with half the estimated wire capacitance at each end. It applies
only to those selected control nets. All their receivers share the lumped sink
node. This is a pessimistic stress approximation, not a distributed extracted
network or proof of remote-receiver skew. Unbuffered nets and bitlines retain
ideal wiring, so a passing run would still not qualify the whole macro.

`--phase-scale` scales the clock schedule and initial precharge interval; the
reported margins stay in physical ps. The final request is canceled 60 ps after
its falling clock edge at every scale. Scaling the schedule does not change
wire/device parameters or the transistor timestep.

## Initial results and outstanding work

For the 4×2 two-fin self-timed example programmed with `[0,1,2,3]`, the aggregate
estimated cost falls from 6221.7 to 4523.3 (27.3%). These numbers have ps units
but represent a sum across control paths, not macro access time. PRECHN changes
from stage multiplicities [1,2] to [2,5], and buffer placement moves nearer the
drivers and loads. The selected circuit has 568 transistors. DRC, LVS, exact
pin matching and routing checks pass for this CLI build and for separate
one-/three-fin mux1/mux4 cases.

The TT/0.70 V/25 °C stress run at the old 1 ns cycle **fails**: sampling and
return to precharge miss the schedule, and phase/hold checks also report
failures. The report contains no passing margins. Final estimated stage efforts
still exceed four. Neither the cost reduction nor physical correctness implies
that the old ideal-wire operating schedule survives these wire loads.

Sink-clustered routing, shorter shared spines, calibrated extraction and
receiver-specific acknowledgments remain necessary to establish useful timing.
Artifacts, including failed stress experiments, are retained under
`build/brom_wire_rc`.

The same 4×2 TT stress bench passes with `--phase-scale 10` (a 10 ns cycle):
minimum read margin 249.6 ps, latch-closure margin 43.3 ps, and candidate-off
margin 269.7 ps. It uses the timing bench's default words `[2,0,0,0]`, while
the layout-cost example above uses `[0,1,2,3]`. This checks the modeled control
sequence at a slower schedule; it does not measure maximum frequency or qualify
all wiring. Both runs retain the same wire coefficients and 0.5 ps timestep.

The combined BROM/controller/replica/distribution/wire-RC/timing/minimum-proof/
packaging suite passed 159 tests. Final estimator, budget, stimulus and checker
unit checks passed after report metadata updates.

The wider 8×16 two-fin self-timed CLI build also passes DRC/LVS and pin/route
checks for the macro and standalone peripheral cells. Its physical artifacts
are in `build/brom_wire_rc/compiled_8x16`; no RC transient qualification is
claimed for that wider macro.

## Registered output clock stress

For registered macros the transient bench also models CLK and register phase
nets. The [timing-estimate results](brom_timing_estimates.md) identify phase
overlap at a 10 ns period; the unregistered passing schedule above does not
qualify the registered mode.
