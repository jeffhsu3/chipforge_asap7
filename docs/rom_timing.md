# NOR ROM read timing

`asap7-rom --characterize REPORT` simulates a macro's read in ngspice with
BSIM-CMG and writes what it measured. `--timing REPORT` then builds the macro's
Liberty from those numbers:

```sh
asap7-rom --array.words 1024 --array.bits 148 \
    --sense --enable --latch --sample --control --register --edge_pins \
    --characterize out/timing.json --jobs 6 --resolution 20

asap7-rom --array.words 1024 --array.bits 148 \
    --sense --enable --latch --sample --control --register --edge_pins \
    --words words.json --package out/macro --timing out/timing.json
```

A report belongs to one shape and one circuit: `--timing` refuses a report whose
macro name differs, or whose `circuit_sha256` no longer matches the macro's
transistors. A shape's name stays the same when its circuit changes, so the name
alone wouldn't catch a stale report. Working files go in `out/timing.work`.

**This is simulation, not sign-off.** Wire resistance and capacitance are
estimates from drawn length, local wiring has none, the supplies are ideal, and
nothing is extracted. `timing_qualified` stays `false`.

## What it simulates

A 1024x148 macro has a quarter of a million transistors, nearly all idle in one
read. `read_path` keeps what a read of one bit uses and replaces the rest with
what the kept nets feel of them: gate load, and off devices that leak as the
originals do. That leaves about 1,900 devices at 1024x148. At 16x6, the only
size compared so far, the reduced netlist agrees with the flat one within about
4%, on the slow side.

It measures bit 0, the far end of its wordline and of the `PRECHN` and `SAMPLE`
columns, reading the two words furthest from the lane, with every other word of
that bit programmed. Each read takes `Q[0]` the other way: words 0, W-1, 0.

## What it measures

| Field | Meaning |
|---|---|
| `clk_q_rise`, `clk_q_fall` | `CLK` rising to `Q` at 50%, per load in `loads_ff` |
| `q_rise_slew`, `q_fall_slew` | `Q`'s 20% to 80% time, per load |
| `setup`, `hold` | How close to `CLK` falling the address can change; setup can be negative |
| `min_low`, `min_high` | The shortest `CLK` pulses that still read correctly |
| `droop` | The worst bitline's fall in mV/ns while `CLK` is low, on a line nothing discharges |
| `max_low` | How long `CLK` can stay low before that line reaches 2/3 of VDD, in ps |
| `aborted` | Searched simulations that did not finish; they count as failures |

A read is correct if `Q` holds its word when the next `CLK` rise arrives, so a
word that gets through the register early counts as wrong.

Each limit is searched with the others relaxed. A failure above a pass moves the
search above that failure, so a limit that isn't monotonic comes out safe. A
separate run put all of them together, both pulses at their minimums, setup at
5 ps and hold at the limit plus 5 ps, and also at the Liberty's numbers. All three
reads were correct at every corner at 16x6 and at 1024x148. That was one bit at
1 fF, reading words 0, W-1, 0.

| Corner | VDD | Temperature |
|---|---|---|
| `typical` | 0.70 V | 25 °C |
| `fast` | 0.77 V | −40 °C |
| `slow` | 0.63 V | 125 °C |

Solver settings matter here: `rshunt=1e10` is what lets the fast corner find an
operating point, and a time step above 1 ps aborts at the fast and typical
corners. Where larger steps finished at 64x16, they moved results under 1 ps.

## Measured results

Bare measurements at a 5 ps input slew, before the Liberty's margin, all in ps.
`clk_q` is at 1 fF. Both shapes finished with no aborted simulation.

| | 16x6 tt | 16x6 ff | 16x6 ss | 1024x148 tt | 1024x148 ff | 1024x148 ss |
|---|---:|---:|---:|---:|---:|---:|
| `clk_q_rise` | 80.8 | 60.0 | 116.0 | 119.5 | 95.8 | 158.9 |
| `clk_q_fall` | 58.6 | 44.0 | 83.6 | 99.2 | 80.5 | 128.4 |
| `setup` | −20.4 | −12.2 | −28.6 | −22.4 | −16.0 | −56.0 |
| `hold` | 28.6 | 20.4 | 36.7 | 49.7 | 31.0 | 95.4 |
| `min_low` | 150 | 110 | 207 | 745 | 608 | 995 |
| `min_high` | 61 | 45 | 78 | 303 | 231 | 411 |
| `max_low` (ns) | 230 | 262 | 45 | 264 | 334 | 17 |

`min_low` is what limits the clock. With a 50% duty cycle the period is twice
`min_low`, not the Liberty's `min_period`, which is the sum of the two pulses.
After the margin, 1024x148 at 50% duty needs 1863 ps at typical and 2488 ps at
slow.

The logical-effort estimate is not a safe stand-in for these. At typical, with
its own 1.5 margin:

| | 16x6 estimate | 16x6 measured | 1024x148 estimate | 1024x148 measured |
|---|---:|---:|---:|---:|
| `clk_q` | 152 | 80.8 | 305 | 119.5 |
| `setup` | 10 | −20.4 | 18 | −22.4 |
| `hold` | 59 | 28.6 | 263 | 49.7 |
| `min_low` | **117** | 150 | 811 | 745 |
| `min_high` | 75 | 61 | 399 | 303 |

It is pessimistic everywhere except `min_low`, where it is below the measured
number at 16x6 and only 9% above it at 1024x148, less than the 1.25 margin the
measured Liberty applies.

## Liberty

A measured corner's arcs are the measurement times 1.25, because of what the
simulation leaves out. Setup and hold have a 10 ps floor. `Q`'s arcs are tables
over load. Pin capacitances are still estimates. A corner the report doesn't
have keeps the estimate, and each file's header says which it is. The manifest
records the report's hash and each corner's source and `max_low_ps`.

## Limits to design around

**Don't stop `CLK` low for long.** The read leaf has no keeper. While `CLK` is
low, a bitline reading 0 floats and every programmed bit on it leaks, so it
droops until the leaf reads 1. `max_low` is a straight-line extrapolation, which
is on the safe side because the rate falls with voltage. It is shortest at the
slow corner, which is the hot one. A direct 20 ns run at 1024x148 there left the
bitline at 0.39 V of 0.63 V and `D` at 7% of VDD.

**The first read's `Q` is unknown.** Reset doesn't initialize the register. `Q`
is X in the Verilog model, and either rail in SPICE, until the first `CLK` rise
that follows a `CLK` low with `RESETN` high.
