# BROM selector replica

`self_timed_macro_circuit(spec, words, replica=ReplicaSpec())` adds a physical
selector replica and closes the controller's four completion feedback paths.
The public interface is address, CLK, RESETN, Q, VDD and VSS. CLK high requests
precharge; CLK low requests a read. Q is level-sensitive during sampling and
holds afterward. RESETN cancels an operation through the shutdown sequence;
it does not reset Q.

```bash
python -m chipforge_asap7.brom words.json --bits 2 --self-timed \
  --out build/brom_replica/compiled --drc --lvs
```

The manifest records `controller: replica_feedback`, `replica_integrated: true`,
`timing_qualified: false` and `replica.feedback_timing_qualified: false`.
`--clocked` retains the external completion interface for independent controller
experiments. Omitting both options retains the external three-phase interface.

## Circuit and geometry

The replica uses the same compact 108 × 432 nm selector geometry, fin count,
threshold flavor and isolation devices as the data array. Its full column count
matches the data array's candidate fanout. It has one extra four-word block by
default, adding intrinsic bitline loading. All columns discharge through one
live selector in the final block, driven by candidate 15. Other candidate gates
are grounded. Every replica column has its own precharger and inverter.

The replicated control path includes evaluation, block selection and all sixteen
candidate AND gates. The last column feeds a matching mux path for mux4 macros,
a real output-latch input load, and four CMOS buffer pairs before READ_READY.
This is transistor-generated completion, with no behavioral delay source.
CANDIDATE_BUSY includes pending intermediate control stages and a guarded
candidate return path so cancellation cannot omit an in-flight transition.

PRECHN_FAR is derived through buffers placed at the end of the folded bank.
SAMPLE_CLOSED observes SAMPLE and the actual complementary latch control,
then buffers the result. These bank-end observers are not proof that every
remote receiver has completed: extracted RC must establish that relationship.

The replica core sits above and to the right of the data core, with separate
candidate and bitline escapes to the shared peripheral bank. Both arrays are
included in transistor extraction, top-level routing checks and LVS. Grounded
replica candidates intentionally share a supply net. Standalone selector labels
remain separate from the unlabeled embedded cores.

`--replica-extra-blocks` and `--replica-guard-pairs` select positive integer
padding and delay counts (defaults 1 and 4). Changing either requires renewed
timing checks. The current single-tap-row reach limits the combined replica
depth to 69 blocks, leaving at most 68 data blocks with the default padding.
The full-width replica has substantial area overhead; it is a conservative
starting implementation, not an area-optimized replica column.

## Electrical validation

```bash
PYTHONPATH=src python -m chipforge_asap7.brom.replica_timing \
  --depth 8 --bits 16 --klu \
  --osdi build/brom_timing/model/BSIMCMG107.osdi \
  --out build/brom_replica/tt_patterns_klu
```

`--klu` requires a KLU-enabled ngspice with compatible OSDI support. Omit it to
use the default sparse solver. `--model-card`, `--vdd`, `--temperature-c`,
`--step-ps`, `--guard-pairs` and `--extra-blocks` select the experiment.

The bench drives only address, CLK and RESETN. Every completion acknowledgment
comes from the integrated transistor circuit. It reads each address, changes
the external address during evaluation to exercise address holding, checks
output retention, and cancels a final read before sampling. The 1 ns clock
schedule has a 550 ps low interval. It is a verification schedule, not a measured
maximum operating frequency. Outputs have 1 fF loads; wires are ideal.

Checks require restored bitlines before each read, valid data before SAMPLE
opens, correct retained Q, closed output latches before evaluation ends, and
inactive actual data candidates before precharge. They also reject missing
completion and incomplete/nonfinite transient output. Synthetic faults test
these checks independently. `--bitline-load-ff` adds load to the data bitlines
only, allowing deliberate mismatch tests; it does not represent extracted RC.
Each run retains the deck, waveform, signal map, log and JSON report with model,
OSDI and deck hashes.

## Recorded results

RVT, two-fin, 1 fF output-load runs at 0.5 ps maximum timestep:

| Macro / corner | Read margin | Latch closure margin | Candidate-off margin |
|---|---:|---:|---:|
| 4×2 mux1, TT / 0.70 V / 25 °C | 53.9 ps | 42.3 ps | 54.8 ps |
| 8×16 mux1, TT / 0.70 V / 25 °C | 61.9 ps | 30.3 ps | 64.7 ps |
| 8×16 mux1, SS / 0.63 V / 125 °C | 88.3 ps | 42.5 ps | 90.6 ps |
| 16×4 mux4, TT / 0.70 V / 25 °C | 61.6 ps | 40.0 ps | 66.3 ps |

These are minimum margins across the tested reads, with ideal wires. The
16-column pattern benches cover all sixteen programmed selector patterns.
KLU and the default sparse solver agreed within 0.1 ps on minimum margins for
the 4×2 TT case. An adversarial 10 fF load on only the data bitlines correctly
failed the data-before-SAMPLE check.

FF / 0.77 V / −40 °C attempts failed to converge at the initial operating point
with both solvers; an additional reset-idle NODESET experiment also failed.
No FF transient qualification is claimed. Original failed and interrupted runs
are retained alongside the successful reports under `build/brom_replica`.
The SS report was reanalyzed after correcting the checker to allow precharge
return through the next cycle's idle checkpoint (+1180 ps), before its request
at +1200 ps; the original +1000 ps cutoff was earlier than that deadline.

Physical tests pass DRC, LVS, exact top-level pin matching and metal connectivity
for 4×2 mux1/two fins, 8×1 mux1/one fin and 16×1 mux4/three fins. The CLI example
also passes DRC/LVS for the macro and all four standalone peripheral artifacts.
The BROM/controller/replica/timing/minimum-proof/packaging suite passed 130 tests;
the additional precharge-deadline regression passed with the checker tests.

The optional [sized control distribution](brom_distribution.md) pass applies
to both the real array and replica with `--sized-controls`. The results above
are for the original drivers; the sized circuit requires separate timing checks.

## Remaining qualification

This implementation is experimental. Device-level timing on selected shapes
and corners does not establish margins for every width, depth, ROM content,
threshold flavor, fin count or load. Interconnect extraction, phase distribution
sizing and receiver-specific acknowledgment margins remain necessary before
physical timing signoff. Guard defaults are not guaranteed for arbitrary macros.

CLK high must allow shutdown and full precharge; CLK low must allow read
completion and the sampling aperture. The controller does not make arbitrarily
short pulses safe. Release RESETN with CLK high and complete precharge before
reading. Keeperless dynamic bitlines also prevent an indefinitely stopped-low
clock. See the [controller contract](brom_controller.md) for reset and completion
semantics.
