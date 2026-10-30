# BROM loaded transistor timing

The BROM compiler has a repeatable ngspice/BSIM-CMG 107 transient bench:
`python -m chipforge_asap7.brom.timing` (installed as `asap7-brom-timing`).
It simulates every transistor in the generated circuit, including the grounded
isolation devices, address latches, decoder, CGEN, precharge/readout, optional
mux4 and output latches.

**Scope: transistor-level timing with ideal wires.** Intrinsic model parasitics
and explicit loads are included. The folded layout's wire resistance,
capacitance and coupling are not extracted into these decks. The macro's
`timing_qualified` flag remains false. These results do not establish a
post-layout access time, maximum frequency, setup/hold table or Liberty model.

## Measured cases

Runs on 2026-10-28 used RVT, two fins, 20 nm gate length, 5 ps input ramps,
1 fF on every Q output and no additional lumped bitline capacitance. The
bitline still includes its transistor drain/gate capacitances. Supplies and
input drivers are ideal. Numbers below are rounded to the nearest picosecond.

| Macro | Model / VDD / temperature | BL precharge to 90% | EVAL → internal data valid | SAMPLE → Q valid |
|---|---|---:|---:|---:|
| 8 × 16, mux1 | TT / 0.70 V / 25 °C | 7 ps | 54 ps | 36 ps |
| 8 × 16, mux1 | SS / 0.63 V / 125 °C | 10 ps | 78 ps | 60 ps |
| 8 × 16, mux1 | FF / 0.77 V / −40 °C | 6 ps | 40 ps | 30 ps |
| 16 × 4, mux4 | TT / 0.70 V / 25 °C | 6 ps | 79 ps | 30 ps |

The mux1 cases have two four-word blocks; the mux4 case has one. Each block
covers all sixteen programming patterns across its sixteen physical columns.
Every address is read forwards and backwards. All four selected runs passed
read correctness and output holding, including through the following
precharge/evaluation: 80 reads and 896 output-bit checks across the four
selected runs. External address pins are deliberately changed during
evaluation, while the address latches should remain closed.

The table reports maxima across the exercised addresses, bitlines and outputs.
The EVAL metric ends at the input of the output latch, after mux selection.
The Q metric starts at SAMPLE; these are separate phase delays. With the
actual bench's scheduling, nominal mux1 EVAL-to-captured-Q is about 176 ps,
including the deliberately inserted 140 ps EVAL-to-SAMPLE interval.

TT and FF use a 400 ps cycle; SS uses 800 ps. The original 60 ps SAMPLE-high
window was insufficient at SS (including its input ramp and 90% settling
criterion), so the passing SS run uses 120 ps. These are tested schedules,
not minimum periods. No full PVT envelope or arbitrary ROM size is certified.

## Measurement and numerical checks

Times are measured from the ideal control's 50% crossing to the **last**
10%/90% crossing that remains valid through the measurement window. A signal
already valid contributes zero delay. A failed read, hold violation, missing
output or truncated transient prevents a successful timing summary. A
separate negative check deliberately changes the expected word and confirms
that the saved mux4 waveform is rejected.

The solver uses Gear integration, `reltol=1e-3`, `abstol=1e-12`,
`vntol=1e-6`, a 1e15 ohm convergence shunt and one worker thread. A DC operating
point precedes the transient. NODESET hints select a legal zero state for
closed latches; they do not force those nodes during simulation. Power-up
and metastability behavior are outside this characterization.

Selected TT/FF mux1 results use a 0.5 ps maximum timestep. SS and mux4 use
2 ps. Refining nominal mux1 from 2 ps to 0.5 ps changed worst EVAL-to-data
by about 1.25 ps and SAMPLE-to-Q by less than 0.04 ps. A targeted
0.25 ps run of addresses 3, 4, 5, 4 also passed (53.5 ps EVAL-to-data,
36.1 ps SAMPLE-to-Q). The coarse FF run
aborted numerically and is retained as a failure; the 0.5 ps retry completed
and passed. Do not interpret the rounded table as sub-picosecond accuracy.

## Reproduce

Install the `timing` extra; `--plot` additionally needs the `plot` extra.
The bench requires ngspice with OSDI support and a BSIM-CMG 107 library built
with L, NFIN and NF exposed as instance parameters. Older cached libraries
may not support these parameters. The existing helper builds a compatible
library from the sibling Xyce source:

```python
from pathlib import Path
from chipforge_asap7.devices.spice import (
    compile_bsimcmg107, find_default_bsimcmg107_source,
)
compile_bsimcmg107(
    find_default_bsimcmg107_source(),
    Path("build/brom_timing/model/BSIMCMG107.osdi"),
)
```

Nominal mux1:

```bash
python -m chipforge_asap7.brom.timing \
  --depth 8 --bits 16 --fins 2 --step-ps 0.5 \
  --osdi build/brom_timing/model/BSIMCMG107.osdi \
  --out build/brom_timing/final/mux1_tt_fine --plot
```

Use the same shape for SS, with `--model-card` pointing to
`7nm_SS_160803.pm`, `--vdd 0.63 --temperature-c 125 --phase-scale 2 --step-ps 2`.
For FF use `7nm_FF_160803.pm`, `--vdd 0.77 --temperature-c -40 --step-ps 0.5`.
Use distinct output directories for each case.

For mux4 use `--depth 16 --bits 4 --ways 4 --step-ps 2` at default TT conditions.
`--words image.json` characterizes a specific ROM image (depth comes from the
image); `--addresses 3 4 5 4` limits the read sequence for a targeted experiment.
`--output-load-ff`, `--bitline-load-ff` and `--slew-ps` support explicit load
and slew experiments. Added bitline capacitance is an assumed load, not PEX.

Each case retains `timing.sp`, translated `models.pm`, `timing.log`,
`waveform.dat`, signal/node maps, `report.json`, and optionally `waveform.png`.
The report records conditions, per-read measurements, model/OSDI/deck hashes
and simulator version. It is invalidated before a new run. The CLI returns
failure on failed observations or simulation errors, with a ten-minute
simulator timeout.

The selected run artifacts and consolidated report are under
`build/brom_timing/final/`. They are generated outputs; this document preserves
the measured headline results in the repository.

## Registered and closed-loop timing estimates

The [registered timing-estimate flow](brom_timing_estimates.md) adds standalone
setup/hold and pulse-width sweeps, complete-schedule pass/fail transitions, routed clock
RC stress, and compiler-manifest attachment with circuit/content checks.
