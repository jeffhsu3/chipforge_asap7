# Optional BROM bitline feedback keeper

`--keeper-fins 1` enables a weak feedback keeper on every data and replica
bitline. The default remains `--keeper-fins 0` (disabled). The standalone
output/readout artifact receives the same keeper configuration.

```bash
python -m chipforge_asap7.brom words.json --bits 2 --self-timed \
  --keeper-fins 1 --keeper-stack 2 --out build/brom_keeper/compiled --drc --lvs
```

The Python configuration is `BromSpec(..., keeper_fins=1, keeper_stack=2)`.
Keeper fin counts 1–3 and series counts 1–4 are supported geometrically;
these are not all electrically qualified. `--sized-controls` and
`--wire-rc-controls` can also be used. Public pins and programmed array geometry
are unchanged. Enabled macros and readout cells have a `_k<fin>s<stack>` name
suffix. The manifest records the strength and replica matching explicitly.

## Circuit

A PMOS path connects VDD to BL, with every keeper gate driven by DOUT, the
existing bitline inverter output. BL high makes DOUT low and enables the
keeper. A selected NMOS must initially overcome that pull-up; once BL falls
and DOUT rises, the keeper switches off. All keeper bodies connect to VDD.

The default enabled configuration uses **two series one-fin PMOS devices**.
Series devices provide weaker drive than a single minimum-fin PMOS. Increasing
the fin count strengthens the keeper; increasing the series count generally
weakens it, but the actual delay/current must be measured. Every replica
bitline receives the same topology, fin count and threshold flavor as the data
bitlines. Its existing extra selector block remains part of the replica load.
The keeper does not change selector sizing or the CP-SAT geometry proof.

This circuit adds positive feedback. Ideal switch-level simulation cannot
establish whether a selector overcomes the keeper: that is an analog sizing
question. An intentionally overstrong three-fin, single-device keeper prevents
a one-fin selector from completing the tested TT read. Do not interpret legal
layout parameters as guaranteed functional choices.

## Characterization

```bash
PYTHONPATH=src python -m chipforge_asap7.brom.keeper_timing \
  --blocks 16 --fins 1 --keeper-fins 1 --keeper-stack 2 \
  --model-card /path/to/7nm_SS_160803.pm --vdd 0.63 --temperature-c 125 \
  --osdi build/brom_timing/model/BSIMCMG107.osdi \
  --out build/brom_keeper/ss_comparison
```

Run the same command with `--keeper-fins 0` for the baseline. The isolated
bitline bench includes two NMOS drains per block (live selector plus isolation),
a precharger, inverter, optional keeper and a 1 fF readout-output load. Wire
capacitance is zero unless explicitly supplied with `--bitline-load-ff`.
`--leakage-na` adds labeled imposed discharge current; the recorded comparisons
below use only intrinsic BSIM-CMG model leakage, with no imposed current.

Precharge is released at 200 ps. The read experiment enables one selector at
400 ps and measures BL discharge to 10%, DOUT validity at 90%, peak positive
keeper supply-branch current and its integrated charge/energy during discharge.
That measured current includes transient displacement current. Read success
requires a restored bitline before evaluation and a stable low BL/high DOUT
afterward. An already-decayed bitline cannot count as a successful read.

The retention experiment keeps every selector off. It reports the first BL
crossings below 90% and 50% of VDD, minimum BL voltage and late keeper current.
A null crossing means the threshold was not crossed within the observation
window, not infinite retention. The default 100 ns experiment observes
99.995 ns after the precharge ramp ends. Report status `passed` requires the
read test to pass; retention is a measured interval, not a pass/fail target.

The bench retains complete waveforms, decks, logs and model/OSDI/deck hashes.
Incomplete or nonfinite transients are errors. The default solver is SPARSE;
`--klu` is available, but several KLU cases had operating-point convergence or
truncated-transient failures. NODESET supplies only a precharged-state initial
guess, not a forced voltage during the measured experiments.

## Measured results

RVT, two series one-fin keeper PMOS, ideal wires, 1 fF readout-output load:

| Corner / selector / blocks | Keeperless 90% retention | Keeper retention | DOUT valid delay, off → on |
|---|---:|---:|---:|
| TT 0.70 V / 25 °C, two fins, 16 blocks | 59.0 ns | ≥99.995 ns | 27.5 → 30.6 ps |
| SS 0.63 V / 125 °C, one fin, 16 blocks | 2.30 ns | ≥99.995 ns | 64.1 → 76.2 ps |
| SS 0.63 V / 125 °C, one fin, 68 blocks | 1.52 ns | ≥99.995 ns | 91.3 → 110.9 ps |
| SS 0.63 V / 125 °C, two fins, 68 blocks | 1.52 ns | ≥99.995 ns | 66.8 → 74.2 ps |

The 68-block, one-fin SS keeper draws about 7.38 µA peak during discharge and
0.225 fJ from its supply branch during that interval. Its late retention current
is about 71 nA per bitline. The 69-block replica-load checks also pass for one-
and two-fin selectors at SS; both hold above 90% for the full window. These are
per-bitline quantities, not whole-macro power measurements.

The SPARSE sweep covers TT/SS/FF, selector fins 1/2, blocks 1/16/68 and keeper
on/off: 33 of 36 runs completed with successful reads. Every completed enabled
keeper case held above 90% throughout the window. Three cases failed numerically:
TT one-fin/one-block without keeper, TT one-fin/68-block with keeper, and FF
two-fin/68-block with keeper. A separate KLU run completed the first of those;
the two deep enabled-keeper cases remain unverified. Failed runs are retained.

Full controller/replica transients pass for 4×2 two-fin TT, 16×2 mux4 two-fin
TT, and 4×2 one-fin SS with enabled keepers and a 1 ns clock. These use ideal wires and are separate
from the earlier wire-RC stress results. DRC/LVS and exact pin/route checks pass
for one-, two- and three-fin examples, mux1/mux4, and the CLI macro with
wire-RC-aware control sizing enabled.

The combined regression suite passed 174 tests; the subsequent pre-read-decay
checker test also passed. Reports and comparisons are under `build/brom_keeper`.

## Limits

The keeper improves high-bitline retention at the cost of read delay, added
feedback capacitance, contention energy and static leakage replacement current.
Matching it in the replica is necessary but does not qualify all replica timing
margins. Extracted interconnect, remaining convergence cases, other threshold
flavors, stronger keeper settings and arbitrary stopped-clock durations remain
unqualified. The compiler continues to emit `timing_qualified: false`.
