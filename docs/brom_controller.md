# BROM clock-request controller

`controller_circuit()` implements the controller as static CMOS transistors.
`clocked_macro_circuit(spec, words)` connects it to the existing compact array,
decoder, CGEN, prechargers, mux and output latches. The controller is also
available as a standalone routed cell with transistor CDL and LVS reference.

```bash
python -m chipforge_asap7.brom words.json --bits 2 --clocked \
  --out build/clocked_brom --drc --lvs
```

The macro exposes address and Q pins, CLK, RESETN, the four completion inputs
below, VDD and VSS. PRECHN, EVAL and SAMPLE become internal nets. The CLI also
emits a standalone `controller` artifact. Without either controller mode, the existing
externally sequenced phase interface remains available.

`--clocked` retains external completion feedback. The experimental
[selector replica](brom_replica.md) connects internal acknowledgments with
`--self-timed`; extracted feedback timing and load-sized control distribution
remain unqualified. For `--clocked`, the manifest
records `controller: external_completion_handshake`, `replica_integrated: false`
and `timing_qualified: false`. Q still uses level-sensitive output latches:
it can update during CLK low once READ_READY arrives. An edge-triggered output
register is a separate remaining block. Do not tie the feedback inputs to constants to
simulate completion.

## Clock and feedback contract

CLK high requests precharge; CLK low requests evaluation. RESETN is active low.
The controller uses GT2N's high-precharge/low-evaluate convention but does not
copy GT2N's technology-dependent delay values.

| Feedback input | Required meaning |
|---|---|
| PRECHN_FAR | High only when precharge is off at the receiving precharge gates; follows the physical far end of PRECHN. |
| READ_READY | High after the selected read data, including the mux, is valid. Low during precharge; retain it through sampling. |
| SAMPLE_CLOSED | High only when all output latch input transmission gates have closed. It must acknowledge the receiving gates, not merely invert the controller's SAMPLE output. |
| CANDIDATE_BUSY | High while any candidate can still activate a selector; a qualified replica may conservatively approximate this. |

The acknowledgments belong to the same read operation. A new read requires a
completed idle/precharge phase: PRECHN_FAR=0, READ_READY=0, SAMPLE_CLOSED=1 and
CANDIDATE_BUSY=0, plus sufficient time for bitline restoration and address
setup. CLK low must last long enough for read completion and the required
sampling aperture. Release RESETN with CLK high and complete a precharge
interval before lowering CLK. Do not stop CLK low indefinitely: the default keeperless
bitlines have a bounded dynamic retention interval. The controller does not stretch an arbitrarily short clock
pulse into a successful read or synchronize unrelated clock domains.

The sequence is:

1. CLK falls with RESETN high. PRECHN rises, closing address latches and
   commanding precharge off. EVAL waits for PRECHN_FAR.
2. EVAL rises. SAMPLE waits for READ_READY; Q retains its prior value meanwhile.
3. SAMPLE rises and the output latches capture the valid data.
4. CLK rises, or RESETN falls. SAMPLE falls first. EVAL remains high until
   SAMPLE is low and SAMPLE_CLOSED acknowledges closure.
5. EVAL falls. PRECHN remains high until CANDIDATE_BUSY falls, then precharge
   resumes and the address latches reopen.

RESETN cancels a request through the same shutdown sequence; it does not clear
Q, force precharge on top of an active read, or bypass stuck feedback. If reset
arrives after sampling has already occurred, Q may hold that newly sampled word.
If a request is canceled before READ_READY, Q holds the previous word. Q is
unspecified until a successful sample. Reset release and completion indications
must meet the eventual characterized timing constraints; this is not an
arbitration circuit for coincident asynchronous transitions.

For review, the steady-state equations are:

```
request = RESETN & ~CLK
sampling = SAMPLE | ~SAMPLE_CLOSED
EVAL = PRECHN_FAR & (request | sampling)
SAMPLE = request & EVAL & READ_READY
PRECHN = request | EVAL | sampling | CANDIDATE_BUSY
```

These feedback paths are intentional. A stalled acknowledgment holds the
preceding phase rather than advancing based on an assumed inverter delay.

## Verification

Tests cover stalled acknowledgments, normal completion, reset during sampling,
cancellation before readiness, mux1/mux4 read-and-hold behavior, standalone
controller DRC/LVS at one and three fins, and integrated macro DRC/LVS.
The switch-level tests assume instantaneous complementary latch controls;
actual complement delays are exercised separately in transistor simulation.

The transient bench uses the full controller-integrated 4×2 RVT/two-fin macro,
1 fF output loads, ideal wires, 5 ps ramps and externally scheduled feedback.
It reads four addresses, asserts reset during sampling, changes address pins
while they should be held, and cancels a fifth request before readiness.
The bench passed TT/0.70 V/25 °C, SS/0.63 V/125 °C and FF/0.77 V/−40 °C
with a 0.5 ps maximum timestep. A preliminary 2 ps TT run aborted numerically
and is retained as a failed run.

Checks reject premature precharge, sampling without evaluation/readiness, evaluation
ending before closure, incorrect reads, hold failures and a canceled read
changing the output. Synthetic fault injections test the checker itself.

```bash
python -m chipforge_asap7.brom.controller_timing \
  --osdi build/brom_timing/model/BSIMCMG107.osdi \
  --out build/brom_controller/tt_fine
```

`--model-card`, `--vdd`, `--temperature-c` and `--step-ps` select model and solver
conditions. A complete transient and all checks must pass; truncated runs fail.
Decks, waveforms, logs, signal maps and reports are retained in each run directory.
The benchmark's generous 1.2 ns cycle and externally scheduled acknowledgments
are test stimuli, not a measured maximum clock frequency or a physical replica.

An optional [feedback keeper](brom_keeper.md) improves measured bitline
retention, but does not qualify indefinite stopped-clock operation.
