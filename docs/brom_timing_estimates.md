# BROM timing estimates and sampled operating bounds

The timing tools now characterize the rising-edge output register, sweep
complete self-timed read schedules, and attach matching results to compiler
manifests. Results remain estimates: `timing_qualified`, `extracted` and
`setup_hold_qualified` stay false. These regenerated results use the
[both-control closure interlock](brom_clock_closure.md).

## Standalone register

```bash
PYTHONPATH=src python -m chipforge_asap7.brom.register_characterization \
  --osdi build/brom_timing/model/BSIMCMG107.osdi \
  --out build/brom_estimates/register_tt
```

Each deck contains 62 trials, with both data polarities and a conditioning
capture before each trial. Sweeps cover data setup and hold relative to the
rising CLK 50% crossing, high pulse width and low pulse width. All input ramps
are centered on the specified times. The low-width test launches data one
input-ramp duration after the falling edge; its bound includes that stimulus
constraint. These are not independently optimized setup/hold surfaces.

A trial must retain the old word before the edge, capture and hold the new
word, satisfy the conservative master/slave nonoverlap check, and keep
clock-to-10%/90%-valid delay within 10% of the same-polarity reference. The
report accepts only a passing suffix of the sampled offsets and identifies
nonmonotonic results and the largest failing sample. It does not treat an
isolated passing trial as a timing constraint. Setup and hold are swept
separately with the opposite constraint relaxed, not tested simultaneously
at their reported bounds. This is not metastability characterization.

RVT/two-fin, ideal-wire results (ps):

| Corner, load, input ramp | CLK→Q fall / rise | Setup passing bound | Hold passing bound | High / low pulse passing bounds |
|---|---:|---:|---:|---:|
| TT, 0.70 V, 25 °C, 1 fF, 5 ps | 56.5 / 63.0 | 0 | 10 | 40 / 40 |
| TT, 0.70 V, 25 °C, 1 fF, 2 ps | 55.8 / 62.4 | 0 | 10 | 40 / 40 |
| TT, 0.70 V, 25 °C, 4 fF, 5 ps | 89.1 / 104.9 | 0 | 10 | 80 / 40 |
| TT, 0.70 V, 25 °C, 4 fF, 2 ps | 88.5 / 104.2 | 0 | 10 | 80 / 40 |
| SS, 0.63 V, 125 °C, 1 fF, 5 ps | 80.6 / 92.6 | 0 | 20 | 80 / 80 |
| FF, 0.77 V, −40 °C, 1 fF, 5 ps | Numerical failure | — | — | — |

For the completed TT/SS cases, setup samples at −10 ps fail the delay/capture
criterion; 0 ps and larger pass. TT/1 fF/5 ps uses KLU; the other passing
standalone cases use SPARSE. The updated FF standalone sweep aborts numerically
near a late stimulus event, including KLU and a finer 0.25 ps retry, so it
supplies no setup/hold or pulse-width bounds. Its earlier-register results are
archived and must not be reused for this topology. A 0 ps sampled bound does not mean the register has no setup requirement:
internal clock delay permits data capture after the pin's 50% crossing, and
the actual boundary lies between samples. SS hold is bracketed by 10 ps
failing and 20 ps passing; other hold cases by 0 and 10 ps. These constraints
refer to the standalone register's D pin, **not the ROM address pins**.

## Complete read cycle and compiler report

```bash
PYTHONPATH=src python -m chipforge_asap7.brom.timing_estimate \
  --depth 4 --bits 2 --phase-scales .3 .4 .5 .6 --slew-ps 2 \
  --osdi build/brom_timing/model/BSIMCMG107.osdi \
  --register-report build/brom_estimates/register_tt_slew2/report.json \
  --out build/brom_estimates/period_tt
```

Generate the matching standalone report first, using `--slew-ps 2`. The
optional attachment checks the register-cell netlist hash, model/OSDI hashes,
fin count, VT, supply,
temperature, output load, input slew and timestep. Each macro case preserves
its deck, waveform, report, model hashes and circuit hash. Numerical errors
remain errors and cannot establish a failing timing boundary. Use `--words`
to characterize a particular JSON ROM image; otherwise the existing selector
pattern generator supplies the words.

The sweep **jointly scales clock and address timing** at a fixed 45% high /
55% low duty cycle, including the initial precharge interval. Input slew and
the final request’s 60 ps reset-cancellation delay remain fixed. It cannot isolate a minimum read-path period or
identify the limiting constraint. Address setup/hold is not independently swept.

For the 4×2 mux1 TT macro programmed `[2,0,0,0]`, 1 fF loads and 2 ps ramps:

| Tested period | Scheduled address lead before falling CLK (50%→50%) | Result and observed failed checks |
|---|---:|---|
| 300 ps | 48 ps | No first sample; subsequent idle/precharge and bitline-restoration failures; invalid sampled data/capture; canceled-read sampling. |
| 400 ps | 64 ps | Incomplete return to precharge; PRECHN not idle on later requests; one bitline-restoration failure. |
| 500 ps | 80 ps | Passes the exercised protocol checks. |
| 600 ps | 96 ps | Passes the exercised protocol checks. |

These address leads describe scheduled transitions, not measured setup margins;
not every address bit changes on every event. With the fixed 2 ps input ramps,
the address ramp finishes only 46 ps before the falling CLK ramp starts at the
300 ps schedule. A passing result does not establish an address setup/hold limit.

For this single image, 500 ps is the fastest tested passing complete schedule, with 225 ps high,
275 ps low and the corresponding address timing. The 400/500 ps observations
are a failing/passing schedule pair, not a minimum-period bracket. The failure
lists above describe symptoms and do not distinguish address setup failure from
read-path, recovery, or interacting constraints. In particular, the 400 ps run
reports recovery/precharge failures; that is not proof of an isolated read limit.

The JSON exposes `observed_schedule_transition_ps`, the endpoint's exact
`failed_checks_at_schedule_transition`, and per-sample address lead times.
`estimated_minimum_period_bracket_ps` is retained as null for v1 consumers;
`limiting_constraint` is null and `address_setup_hold_swept` is false.
Numerical errors cannot supply a failing endpoint, and a nonmonotonic sweep
suppresses the transition pair. Each sample retains its full failure list.

At the passing 500 ps schedule, worst falling-CLK→DATA is 113.8 ps,
falling-CLK→sampled word is 191.8 ps, rising-CLK→idle plus restored bitlines
is 196.2 ps, and rising-CLK→valid Q is 59.3 ps. These measured paths do not
establish independent minimum high/low widths or a universal frequency bound.
The failing/passing samples are 100 ps apart; the sweep does not resolve the
transition inside that interval. No midpoint or tighter bound is inferred.
The JSON records that spacing and the selector patterns actually present.

This 4×2 image is the default generator's output for that shape, but its two
selectors cover only patterns 0 and 1. It is an illustrative case, not a
worst-case headline. A default 4×16 mux1 or 16×4 mux4 image covers all sixteen
selector patterns; those larger macros need their own schedule sweeps and
finer sampling near a transition. They also change geometry and loading, so
same-shape image sweeps are needed for a direct 4×2 comparison. All-pattern
coverage alone does not prove worst-case timing. Those broader, finer sweeps
have not been performed here.

To attribute a constraint, hold the other schedule components at validated
margins and sweep address setup/hold, evaluation time and recovery time
independently; this flow does not yet perform those sweeps.

Additional ideal-wire checks at a 1 ns period and 5 ps input ramps:

| Macro / condition | Falling CLK→sampled word | Rising CLK→idle and precharged | Rising CLK→Q |
|---|---:|---:|---:|
| 16×2 mux4, TT | 240.2 ps | 218.0 ps | 67.5 ps |
| 4×2 mux1, one-fin keeper, SS | 277.4 ps | 273.8 ps | 98.7 ps |
| 4×2 mux1, FF, ramped startup | 148.0 ps | 149.5 ps | 48.9 ps |

The full-ROM FF DC solve failed with SPARSE, KLU and additional legal clock
phase NODESET hints. `replica_timing --startup-ramp-ps 25` converged and passed:
supply and initially high input sources ramp together from zero over 25 ps,
before the first address stimulus. The deck still solves a DC operating point;
it does not skip initialization or force transient storage states. The report
records the ramp. This resolves the tested transient's initialization problem,
not arbitrary power-up behavior. Default startup remains an ideal fixed supply.

Attach a matching completed estimate during compilation:

```bash
PYTHONPATH=src python -m chipforge_asap7.brom words.json --bits 2 \
  --self-timed --registered-output \
  --timing-report build/brom_estimates/period_tt/timing_estimate.json \
  --out build/brom_estimates/compiled
```

The compiler verifies the exact spec, ROM words and transistor-netlist hash
before embedding the report. Changing keeper, sizing, register mode or ROM
contents invalidates the attachment. It never promotes estimates to signoff.

## Routed clock/control RC

For a registered macro, `replica_timing --wire-rc-controls` now adds routed
Rpath pi sections to CLK and all nine named phase, complement and closure nets,
in addition to the existing sized control chains. The external CLK source is
on the driver side; its receiver waveform is also saved. Register latency is
measured from the external pin. Feedback and latch gates use the receiving
phase node. No gate capacitance is added twice: BSIM supplies device load;
the explicit pi capacitors contain estimated wire capacitance only.

The same route estimator and wire coefficients used by the compiler are used
here; the report includes the clock-net estimates. Each net still has one
lumped receiver node. It does not resolve skew between individual receiving
gates, and excludes bitline/data-path RC, coupling and local device wiring.

The earlier complement-only registered 4×2 TT stress case at 10 ns failed
non-overlap, with up to 13 ps of observed overlap. The
[both-control closure interlock](brom_clock_closure.md) now passes full 4×2 TT
and SS RC reads, with minimum gaps of 32.5 ps and 48.7 ps. TT also passes for
both `[2,0,0,0]` and `[0,1,2,3]` ROM images. The checker retains its 20%/80%
thresholds and additionally rejects interpolated negative gaps between samples.

Before the local SAMPLE receiver fix, the register phases passed at FF with
a 21.1 ps minimum gap, but that full RC
macro fails the controller SAMPLE-closure check. A wider 4×4 TT case likewise
passes register non-overlap but fails controller closure; the same controller
failure occurs on the pre-fix 4×4 circuit. The subsequent
[local SAMPLE receiver fix](brom_sample_closure.md) passes selected FF 4×2 and
TT 4×4 full-macro checks at 10 ns. Neither a universal RC-safe period nor
extracted timing signoff is claimed.

Generated evidence, including failed attempts, is under `build/brom_estimates`.
Full-macro reports in that directory predate the local SAMPLE receiver change
and do not attach to its circuit hash; regenerate them before reuse. Current
selected-case reports are under `build/brom_sample_closure`. The next physical
timing step is to correlate receiving-gate and closure-return skew against
extraction, then broaden the shape,
content, load/slew, PVT and duty-cycle coverage.
