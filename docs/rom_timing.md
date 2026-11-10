# NOR ROM read timing

`asap7-rom --characterize REPORT` simulates a macro's read in ngspice with
BSIM-CMG and writes what it measured. `--timing REPORT` then builds the macro's
Liberty from those numbers:

```sh
asap7-rom --array.words 1024 --array.bits 148 --flow-through \
    --characterize out/timing.json --jobs 6 --resolution 20

asap7-rom --array.words 1024 --array.bits 148 --flow-through \
    --words words.json --package out/macro --timing out/timing.json
```

`--flow-through` turns on the read path it stands on (`--control`, and with
it `--enable`, `--latch`, `--sample` and `--sense`), and `--characterize`,
`--package` and `--views` turn on `--edge_pins`. Spelling them out still
works.

The macro reads one of two ways, and core's bank script builds the first:

- **`--flow-through`** (`edge_flow_through` in core's
  `docs/rom_macro_contract.md`): `CLK` rising latches `A` and starts a
  self-timed read. When the replica says the read is done, the sample latches
  open for `done_delay` inverters, then close, and the bitlines precharge.
  `Q` holds until the next read changes it. There is no register and no keeper.
- **`--register`** (`two_phase`): `CLK` high precharges, `CLK` falling latches
  `A`, the read runs while `CLK` is low, and a register shows the word from the
  next rise. `--keeper` holds its bitlines while `CLK` stays low.

Both have a latency of one cycle. Most of this page measures the two-phase
read, which came first; [the flow-through read](#the-flow-through-read) has
its own numbers.

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
| `setup`, `hold` | How close to `CLK` falling (rising, flow-through) the address can change; setup can be negative |
| `min_low`, `min_high` | The shortest `CLK` pulses that still read correctly |
| `min_period` | Flow-through only: the shortest period at 50% duty that still reads correctly |
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
With keepers, 1024x148 still found no operating point at the fast corner
without help. When ngspice says so, `simulate` runs the deck again with a first
guess for each `Q` pin, VDD and then 0. `Q` may start at either rail, and the
first read overwrites it, so the guess doesn't change what is measured. `rshunt=1e9` also
worked, but it leaks about 0.8 nA from every node.

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

**Without `--keeper`, don't stop `CLK` low for long.** While `CLK` is low, a
bitline reading 0 floats and every programmed bit on it leaks, so it droops
until the leaf reads 1. `max_low` is a straight-line extrapolation, which is on
the safe side because the rate falls with voltage. It is shortest at the slow
corner, which is the hot one. A direct 20 ns run at 1024x148 there left the
bitline at 0.39 V of 0.63 V and `D` at 7% of VDD.

`--keeper` adds two pFETs in series from VDD to each bitline, one gated by the
leaf's output `D` and one by `SAMPLEB`. They are off while the line evaluates,
so they don't fight a discharge, and they hold a line that read 0 from `SAMPLE`
rising until `CLK` rises. A line that holds reports a `droop` under
0.01 mV/ns and no `max_low`. A keeper sized only to be weak doesn't work here:
in the 4-fin band, two in series held the line at 0.49 V against the 2-fin
cell, and the read never finished.

At 1024x148, with the timing measured without keepers, a read with keepers was
correct at all three corners. It was also correct with a card made from the
slow NMOS and fast PMOS models, at both temperatures, where a keeper fights
hardest. The PDK has no such corner. `clk_q` rose
about 10% later, from the extra load on `SAMPLEB`: 175.4 ps against 158.9 ps at
slow, 105.9 against 95.8 at fast. Characterizing macros with keepers gives their
own numbers:

| | 16x6 tt | 16x6 ff | 16x6 ss | 1024x148 tt | 1024x148 ff | 1024x148 ss |
|---|---:|---:|---:|---:|---:|---:|
| `clk_q_rise` | 94.8 | 70.3 | 136.6 | 131.0 | 102.2 | 175.4 |
| `clk_q_fall` | 71.3 | 54.0 | 101.3 | 109.6 | 82.5 | 142.4 |
| `setup` | −8.0 | −8.0 | −24.0 | −29.0 | −19.0 | −58.8 |
| `hold` | 32.0 | 24.0 | 40.0 | 49.1 | 36.8 | 84.8 |
| `min_low` | 163 | 131 | 226 | 766 | 614 | 1018 |
| `min_high` | 68 | 52 | 84 | 307 | 226 | 409 |

These are bare measurements at 1 fF, in ps. The bitline holds at every corner:
`droop` is 0 and there is no `max_low`. Against the keeperless macro, `clk_q`
is 17–23% later at 16x6 and 3–11% later at 1024x148, from the extra load on
`SAMPLEB`. `min_low` and `min_high` grow 8–20% at 16x6, but at 1024x148,
where they limit the clock, they change by under 3%. No simulation aborted.
At 1024x148 typical, one simulation stalled past the default one-hour timeout
(the 409 ps hold check). Rerun with a longer timeout, it finished in 22
minutes. The report records that rerun.

**The first read's `Q` is unknown.** Reset doesn't initialize the register or
the sample latches. `Q` is X in the Verilog model, and either rail in SPICE,
until the first read with `RESETN` high: for the two-phase read, the first
`CLK` rise that follows a `CLK` low; for the flow-through read, the first `CLK`
rise.

## The flow-through read

`CLK` rising sets a request latch, `REQ`, which turns the precharge off
(`PRECHN` rises) and then the wordlines on (`EVAL`). The address latches close
as `PRECHN` rises, just after the rise, so `A` is held to the rise. The
replica bitline's fall raises `SAMPLE`, which opens one latch per bit. A chain
of `done_delay` inverters after `SAMPLE` raises `DONE`, which resets `REQ`:
`SAMPLE` falls, the latches close on the word, and the bitlines precharge.
`CLK` must fall between reads to arm the next one, but only for a few gates.

**`done_delay` is 15.** It sets how long the latches stay open, and at 3,
about 20 ps, the latches did not flip `Q` at 256x148. At 9 they did at
typical, and 15, about 60 ps, read at every corner up to 1024x148.
`RomControlSpec.done_delay` and `RomDecoderSpec.done_delay` default to 15.

The shortest period at 50% duty against the two-phase read's, in ps, bare, at
148 bits. A two-phase period is twice `min_low`:

| Words | two-phase tt | flow-through tt | flow-through ff | flow-through ss | flow-through `clk_q` tt |
|---:|---:|---:|---:|---:|---:|
| 256 | 745.6 | 569.6 | 417.3 | 804.9 | 446.6 |
| 512 | 956.8 | 710.4 | 535.0 | 990.1 | 550.5 |
| 1024 | 1496.0 | 1078.4 | 800.0 | 1449.9 | 832.4 |

The flow-through read is 24–28% shorter at typical. These searched only
`min_period`, with the address set up and held well clear of the rise, so
setup and hold for this read are still the estimate's.

**`clk_q` is most of the period.** `Q` changes only once the read is done, so
whatever takes `Q` gets the period less `clk_q`: 246 ps at 1024x148 typical,
at the shortest period, against nearly the whole cycle behind the two-phase
read's register. A path from `Q` longer than that sets the period instead.

**The estimate is safe for this read.** At typical, with its 1.5 margin, the
Liberty's estimate is at or above the measurement at each size: `clk_q` 448
against 446.6 at 256x148 and 942 against 832.4 at 1024x148, `min_period` 650
against 569.6 and 1319 against 1078.4. Without the margin it is 25–35% short.

The flow-through read has no register, so it has no keeper either: the
bitlines are precharged except while a read runs.
