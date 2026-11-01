# Controller SAMPLE closure

The global SAMPLE controls could acknowledge closure before the sampling
transmission gates met the checker's 10%/90% thresholds. The register was not
the cause. Self-timed macros now give every data latch and the dummy replica
latch a local receiver and combine all closure indications before releasing
evaluation. The controller's Boolean equations and register clock circuit are
unchanged.

## Failure mechanism

The old observer combined inverted SAMPLE with the actual SAMPLEB rail and
passed that result through two buffer pairs. Its logic switched near the
middle of a slow edge; the buffers did not guarantee that the loaded global
rails had finished settling.

At FF 4×2, relative to the rising clock edge, SAMPLE reached 10% at 480.24 ps,
while SAMPLE_CLOSED reached 90% at 293.82 ps and EVAL began falling through
90% at 429.49 ps. The receiving latch was still open for about 50.75 ps after
evaluation started shutting down. At TT 4×4, SAMPLEB did not reach 90% until
1203.32 ps, versus EVAL at 1100.41 ps. The earlier register circuit also failed
that 4×4 controller check. Increasing the clock period did not fix this local
shutdown ordering.

Baseline decks and reports remain under `build/brom_clock_closure`, including
`rc_ff_10_sparse`, `rc_tt_4x4` and `baseline_rc_tt_4x4`.

## Receiving controls and completion

For each data/replica sampling latch:

```
mid        = ~SAMPLE
enable     = ~mid
complement = ~enable
off        = ~enable
quiet      = mid & off
closed     = quiet & complement
```

Only that latch uses its enable/complement pair. Receiver, latch and observer
groups are adjacent in the folded peripheral bank. Checking `mid` prevents an
in-flight command from being ignored by a receiver that has not yet opened.
A balanced AND tree combines every `closed`, including the dummy latch; the
existing two buffer pairs generate SAMPLE_CLOSED from this join. EVAL waits
for SAMPLE low and this acknowledgment.

The transistor bench saves each actual receiving pair. It checks earliest
opening and latest closure across all pairs, and rejects SAMPLE_CLOSED reaching
90% before every enable is below 10% and every complement above 90%. It also
retains the independent EVAL shutdown check. No threshold was relaxed.

With `--wire-rc-controls`, the bench includes Rpath pi sections for the local
receiver/observer nets, closure tree and return buffers, in addition to sized
control distribution and register clock nets. Reports record this RC scope
and compiler manifests expose `sample_distribution`.

## Selected transistor results

All cases below pass the full read/cancel and registered-output checks. Loads
are 1 fF per Q, input slew 5 ps, two-fin devices, no keeper, mux1. Closure margin
is receiving-gate closure to EVAL shutdown; acknowledgment margin is receiving
closure to SAMPLE_CLOSED. Values are minimum margins across checked reads.

| Case | Period | ROM words | Closure margin | Acknowledgment margin |
|---|---|---|---|---|
| FF 4×2, control/clock RC | 10 ns | `[2,0,0,0]` | 514.85 ps | 541.66 ps |
| TT 4×4, control/clock RC | 10 ns | `[10,12,0,0]` | 916.31 ps | 948.14 ps |
| SS 4×2, control/clock RC | 10 ns | `[2,0,0,0]` | 1049.15 ps | 1047.03 ps |
| TT 4×2, ideal wires | 1 ns | `[2,0,0,0]` | 67.23 ps | 41.90 ps |

Reports and saved decks are under `build/brom_sample_closure`, respectively
`selected_ff_4x2`, `qualified_tt_4x4`, `qualified_ss_4x2`, and
`selected_ideal_tt`. Directory names are experiment labels, not signoff claims;
`timing_qualified` remains false. TT uses 0.7 V/25 C, SS 0.63 V/125 C and FF
0.77 V/−40 C. RC runs use KLU, 2 ps timesteps at TT/SS and 1 ps at FF. The
ideal-wire run uses SPARSE with 0.5 ps steps.

The bench now supplies consistent logical idle NODESET guesses for both sides
of RC sections. These help DC convergence; they do not force transient values
with `.ic` or `uic`. FF uses a 25 ps supply ramp after fixed-supply DC attempts
failed numerically. Failed attempts are retained. Power-up behavior is not
qualified. `--timeout-s` allows these large runs to exceed the default 600 s.

For example, reproduce the TT 4×4 case with:

```sh
PYTHONPATH=src .venv/bin/python -m chipforge_asap7.brom.replica_timing \
  --depth 4 --bits 4 --registered-output --wire-rc-controls --klu \
  --phase-scale 10 --step-ps 2 --timeout-s 1800 \
  --model-card ../asap7/asap7_pdk_r1p7/models/hspice/7nm_TT_160803.pm \
  --osdi build/brom_timing/model/BSIMCMG107.osdi \
  --out build/brom_sample_closure/reproduce_tt_4x4
```

## Limits and regression coverage

These are selected schedules, not a minimum-period or worst-content result.
The images do not cover all sixteen selector patterns. Address setup/hold,
duty cycle, arbitrary dimensions, loads and slew still need characterization.
The pi model shares one receiving node per net: sensor-versus-latch skew needs
receiver-specific extraction. Closure-return RC contributes substantial delay
in the passing RC cases; a fastest-return/slowest-receiver stress remains to
be characterized. Bitline/data-path RC and coupling are not included.

Tests cover unique local gate controls, all-latch completion, individual stalled
receivers (including pending first stages), release after stalls, dummy-latch
timing failures and consistent DC hints. The BROM and packaging regression
passes 209 tests, including existing physical-view checks.

Old full-macro timing reports have a different circuit hash and must be
regenerated before attachment. They remain historical evidence; changing their
hashes to reuse old measurements would invalidate that check.
