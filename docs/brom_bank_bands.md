# Function bands and shared SAMPLE receivers

The bottom bank of a dense self-timed, registered B-ROM holds each output
bit's readouts, mux, SAMPLE receiver and output latch, and register latches,
plus the decoder, controller and replica logic. The earlier placements left
that bank mostly empty. At 1024x148, 4-way, its cells added up to 726 µm of
row length, enough for 9 rows across the 83 µm array, but it used 24 rows
(36% fill). Bit tiles stack each bit's whole chain in one row of a tile at
the 0.56 µm bit pitch, so the row count follows the bits per tile, not the
cell area. Flat placement runs out of M5 columns: each multi-row net needs
its own column, and a 148-bit datapath spread over many rows needs more
than the bank has.

## Function bands

`dense_slices.banded_rows` gives each function its own band of rows. From
the array down, the bands are:

1. readouts, aligned to their bitlines;
2. mux cells;
3. SAMPLE receiver runs, each with its output latches;
4. register latches.

Abacus legalization (`_legalize`) puts each bit's block in its band, as
close under the bit's bitlines as the band's rows allow. A band has the
fewest rows its cells fill at the chosen fill. The decoder, controller,
replica and closure tree go into the bands' free space, around the fixed
data pins in both dimensions (`_place_controls`). Spare rows go below the
bands only when that free space runs out.

A bit's nets then join cells in neighbouring rows at nearly the same x. Each
column spans a few rows, and columns in disjoint row ranges share an x.

`dense_bank._anchored` tries every distinct band placement, fewest rows
first, over band fills of 100%, 85% and 70% and up to five spare rows. It
keeps the first that routes without joint repair. Joint repair is skipped
because a band placement that fails takes minutes to repair, while the same
bands with one more row route in seconds. Wide words, with 64 or more bits,
keep the cheaper of bands and bit tiles. Smaller banks keep the cheaper of
bands and flat rows. Bands leave rows part-empty when a word's bits are
sparse across the array: each bit's cells stay under its own bitlines. For
example, 16x9 4-way bands take 10 rows for cells that fill 4.

A trunk on the lanes' second M4 track can now end left of its rail, where
a band places its riser. Its end moves away from line ends one track over:
the lanes' candidate escapes and the bank's header tracks. Before this, the
12x1 and 8x1 physical tests failed M4.S.4.

## Shared SAMPLE receivers

Each SAMPLE receiver had six cells per output bit, plus a share of the
closure AND tree:

- input inverter
- output inverter
- complement inverter
- off inverter
- `quiet` AND
- `closed` AND

At 1024x148 these cells were 27% of the bank's cell area.
`ReplicaSpec(share=N)` (`asap7-brom --replica-share N`) has one receiver
drive N adjacent output latches. Its output and complement inverters are
built at N/2 times the base fin count, rounded up. This keeps each driver
at two latch loads per base-strength inverter. The receiver's closure
observer watches the shared enable and complement, and those nets reach
only the N latches in the receiver's own run.

The replica latch keeps its own receiver. The default is `share=1`, which
gives the same circuit as before. A macro built with sharing carries `_sN`
in its name, and its manifest records `latches_per_sample_receiver`.

## Results

These are dense self-timed registered macros with RVT, two fins, no keeper
and random words. "Before" is the build before this change; "Bands" uses
`share=1`; "+ share 4" also shares receivers.

| Shape | Before | Bands | + share 4 |
|---|---|---|---|
| 256x30, 4-way | 29.81 × 14.00 = 417 µm² | 29.81 × 11.41 = 340 µm² | 29.81 × 10.98 = 327 µm² |
| 1024x148, 4-way | 83.38 × 40.79 = 3401 µm² | 83.38 × 40.79 = 3401 µm² (bit tiles) | 83.38 × 42.52 = 3545 µm² (bit tiles) |

The 256x30 band build passes DRC and LVS (`--drc --lvs`). At 1024x148 no
band placement routes. The bank's M4 escapes crowd into the 15 µm gap
between the two selector arrays (x ≈ 35–41 µm), where every SEL and G drop
lands, and the row there runs out of tracks. That holds with the gap kept
free of cells, with a spread decoder and with joint repair, so the macro
keeps its bit tiles. Sharing receivers makes the bit-tile bank taller
there, not shorter.

Narrow words gain most from 4 ways. A 256x30 1-way macro is 861 µm², because
its CGEN lanes, not its bank, set its size. Core's bank script
(`scripts/asap7_brom_bank.py --brom-ways auto`) compiles each legal way and
keeps the smaller macro.

## Limits

Shared receivers change the SAMPLE delivery and closure paths. Each driver
sees four latches' gate loads and a longer local wire. The replica transient
bench (`replica_timing --registered-output --wire-rc-controls --klu
--phase-scale 10`, TT, 0.70 V, 25 °C) passes with sharing, at smaller
margins. These are the minimum margins over the bench's reads:

| Macro, step | Share | Read | Closure | Candidate-off | Sample-closed |
|---|---:|---:|---:|---:|---:|
| 4x4, 1 ps | 1 | 71.7 ps | 125.0 ps | 44.8 ps | 86.4 ps |
| 4x4, 1 ps | 4 | 55.6 ps | 95.4 ps | 41.4 ps | 54.3 ps |
| 16x8, 2 ps | 1 | 67.4 ps | 145.5 ps | 36.8 ps | 102.3 ps |
| 16x8, 2 ps | 4 | 43.8 ps | 124.0 ps | 26.9 ps | 80.7 ps |

At a 2 ps step the 4x4 bench stalls ("Timestep too small") at 2.14 ns, as
PRECHN rises through its wire RC, for share 4 with Gear and share 1 with
trapezoidal integration. A 1 ps step runs both. Trapezoidal share 4 agrees
with the 1 ps Gear run within 0.3 ps. The analyzer now counts the SAMPLE
receivers it traces, not one per bit. Sharing stays off by default until
slow and fast corners and extracted RC are checked; the qualification noted
in [SAMPLE closure](brom_sample_closure.md) still applies.

The register is unchanged. Reusing the output latch as the register's master,
as the NOR ROM does, would save one latch per bit: about 56 µm of row length
at 1024x148, under one bank row. That gain is small, and the register is
being reworked separately, so it is not done here.
