# BROM timing estimates and sampled operating bounds

The timing tools now characterize the rising-edge output register, sweep
complete self-timed read schedules, and attach matching results to compiler
manifests. Results remain estimates: `timing_qualified`, `extracted` and
`setup_hold_qualified` stay false.

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
| TT, 0.70 V, 25 °C, 1 fF, 5 ps | 44.8 / 50.9 | 0 | 10 | 40 / 20 |
| TT, 0.70 V, 25 °C, 1 fF, 2 ps | 44.1 / 50.3 | 0 | 10 | 40 / 20 |
| TT, 0.70 V, 25 °C, 4 fF, 5 ps | 77.4 / 92.7 | 0 | 10 | 40 / 20 |
| TT, 0.70 V, 25 °C, 4 fF, 2 ps | 76.8 / 92.1 | 0 | 10 | 40 / 20 |
| SS, 0.63 V, 125 °C, 1 fF, 5 ps | 64.0 / 75.5 | 0 | 20 | 40 / 40 |
| FF, 0.77 V, −40 °C, 1 fF, 5 ps | 33.2 / 36.6 | 0 | 10 | 20 / 20 |

All setup samples at −10 ps fail the delay/capture criterion; 0 ps and larger
pass. A 0 ps sampled bound does not mean the register has no setup requirement:
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
optional attachment checks model/OSDI hashes, fin count, VT, supply,
temperature, output load, input slew and timestep. Each macro case preserves
its deck, waveform, report, model hashes and circuit hash. Numerical errors
remain errors and cannot establish a failing timing boundary. Use `--words`
to characterize a particular JSON ROM image; otherwise the existing selector
pattern generator supplies the words.

The clock sweep uses a fixed 45% high / 55% low duty cycle. Its minimum-period
bracket is inferred between a failing and a passing sampled schedule, assuming
monotonic behavior between them. It is suppressed if sampled results are
nonmonotonic. The fastest tested passing schedule is reported separately.
High and low times are a tested pair, not independently minimized limits.

For the 4×2 mux1 TT macro programmed `[2,0,0,0]`, 1 fF loads and 2 ps ramps:

- 300 and 400 ps schedules fail; 500 and 600 ps pass.
- Sampled minimum-period bracket: **400–500 ps**, at the tested duty cycle.
- Fastest passing schedule: **225 ps high / 275 ps low**.
- At 500 ps: worst falling-CLK→DATA is 189.3 ps, falling-CLK→sampled
  word is 199.3 ps, and rising-CLK→idle controller plus restored bitlines is
  196.2 ps. Worst rising-CLK→valid Q is 47.8 ps.

Those path metrics include controller/decoder/readout behavior; they cannot
be summed into a universal frequency bound. Address setup/hold is not swept.
All complete-cycle cases also check replica margin, latch closure,
candidate/precharge exclusion, output retention and a reset-canceled request.

Additional ideal-wire checks at a 1 ns period and 5 ps input ramps:

| Macro / condition | Falling CLK→sampled word | Rising CLK→idle and precharged | Rising CLK→Q |
|---|---:|---:|---:|
| 16×2 mux4, TT | 240.2 ps | 218.0 ps | 55.4 ps |
| 4×2 mux1, one-fin keeper, SS | 288.9 ps | 273.8 ps | 81.7 ps |
| 4×2 mux1, FF, ramped startup | 148.0 ps | 149.5 ps | 39.9 ps |

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
Rpath pi sections to CLK and all five named register phase/complement nets,
in addition to the existing sized control chains. The external CLK source is
on the driver side; its receiver waveform is also saved. Register latency is
measured from the external pin. Feedback and latch gates use the receiving
phase node. No gate capacitance is added twice: BSIM supplies device load;
the explicit pi capacitors contain estimated wire capacitance only.

The same route estimator and wire coefficients used by the compiler are used
here; the report includes the clock-net estimates. Each net still has one
lumped receiver node. It does not resolve skew between individual receiving
gates, and excludes bitline/data-path RC, coupling and local device wiring.

The registered 4×2 TT stress case at 10 ns **fails the conservative phase
nonoverlap criterion**, even though it captures the expected words. Observed
overlap samples span up to 13 ps near falling transitions and 2.5 ps near
rising transitions. The checker uses 20%/80% transmission-gate control
thresholds; this is a warning about the modeled phase relationship, not proof
of silicon race-through. The failed run produces no passing timing margins
or qualified frequency. Slower periods do not remove this transition-local
problem. Clock-phase distribution and receiver-specific feedback need work
before claiming an RC-safe operating schedule.

Generated evidence, including failed attempts, is under `build/brom_estimates`.
The next physical timing step is to address that phase-distribution finding,
then correlate the lumped model against extraction and broaden the shape,
content, load/slew, PVT and duty-cycle coverage.
