# NOR ROM macro package

`asap7-rom --package DIR` writes everything a flow needs to place one programmed
NOR ROM, and can check it:

```sh
asap7-rom --array.words 1024 --array.bits 148 \
    --sense --enable --latch --sample --control --register --keeper --edge_pins \
    --words words.json --package out/ --drc --lvs
```

`--keeper` keeps each bitline that read 0 from drooping while `CLK` stays low.
It costs three columns per bit slice. See [the read's limits](rom_timing.md#limits-to-design-around).

`words.json` is a JSON array with one unsigned integer per word. Bit `b` of word
`w` set means a V1 at that bit's site, and that bit reads back as 1. Without
`--words`, `--pattern` fills the array as usual.

`--package` needs `--edge_pins` and replaces `--out`, `--cdl` and `--views`.
The command exits non-zero if a check it ran failed.

## What it writes

| File | Contents |
|---|---|
| `macro.gds` | The programmed macro, named by its contents |
| `macro.cdl` | Its flat netlist, sized by `nfin` |
| `macro.lef`, `macro.v` | Its abstract and behavioral model |
| `macro_{tt,ff,ss}.lib` | Timing, with `--register` only: estimated, or measured with `--timing` |
| `abstract.gds`, `abstract.lef` | The shape, under a stable name |
| `manifest.json` | The record of all of the above |

The same input gives the same files byte for byte, in any process.

## Which shapes it can build

A registered macro builds at every width from 2 to 200 bits at 16 to 1024 words.
A latched macro does up to 167 bits at those depths, and refuses some widths above
that with a clear error. The array height limit (ACTIVE.LUP.1) ends both near
320 bits.

A narrow array adds strap
rows to make room for another lane row, which is the only way the lane gets more
tracks: 8 bits draws a strap every 8 rows, 3 bits every 2, and 2 bits every row.
The manifest's `strap_every` records what was drawn. A 1-bit array is refused
because even a strap on every row leaves no room for one lane row.

Building is not the same as checking. DRC and LVS have run on a sample of these
shapes, so use `--drc --lvs` on any shape you ship.

## Sharing one abstract

Programming only adds V1s at the array's bit sites, so every image of one shape
has the same footprint, pins and other shapes. A flow can harden a block once
against the abstract and swap each unit's image in afterwards.

- **Names.** `abstract.*` uses `spec.macro_name`, which depends only on the
  shape. An image is `<macro_name>_p<digest>`, and its two array halves are
  `<image>_left` and `<image>_right`. Those three are the only cells whose names
  depend on the contents. Every other cell is the same in every image of a
  shape, so units merged into one chip GDS cannot clash.
- **Abstract GDS.** It is the macro with no data bits set, so it is the same
  file for every image. A block built on it reads all zeros until its image is
  swapped in.
- **Fingerprints.** The manifest's `abstract` record holds `cell`, `lef_sha256`,
  `frozen_geometry_sha256` and `programming_layers`. A flow should refuse an
  image whose record differs from the one it hardened against.

V1 is also the periphery's via, so `frozen_geometry_sha256` does not skip the
whole layer as the B-ROM's does for V2 and V3. It leaves out only a V1 centred on
a data bit's site. A V1 anywhere else is frozen, including the replica row's,
which every image has. `programming_layers` still names V1, so a layer-based XOR
of two images passes on any V1 change. The fingerprint is what catches a
periphery V1 changing.

## The read it keeps

With `--register`, the manifest's `read` record names the read in the terms of
chipforge's `ROMMacroInterface` (its `docs/rom_macro_contract.md`):

```json
"read": {"contract": "two_phase", "latency": 1, "addr_hold_edge": "fall",
         "reset": {"pin": "RESETN", "active": "low"}}
```

`CLK` high precharges, `CLK` falling latches `A`, and `Q` takes the word just
after the next rise. Core's bank script refuses a macro whose record differs
from the interface it declares. Without `--register`, `Q` follows the sample
latches into the next cycle, which is no read core supports, and `read` is
`null`.

## Checks

| Flag | Runs | Recorded in `artifacts.macro` |
|---|---|---|
| `--drc` | gdscheck on `macro.gds` | `drc`: count per rule, `{}` when clean |
| `--lvs` | KLayout on `macro.gds` against `macro.cdl` | `lvs`: `passed`, `failed` or `timed_out`, with `renamed_pins`, `missing_pins`, `extra_pins` once it finishes |

Both start as `not_run`. Any older `manifest.json` is deleted before the build,
so a run that dies leaves no earlier result behind.

LVS runs on a copy in `lvs/labelled.gds` with its unprogrammed drains labelled,
which makes the compare about 10x faster. Those labels are not in `macro.gds`.
KLayout compares pins by topology, so a pass also requires every expected label
to be present under its own name.

LVS reads a label only as a name, so it can't tell a VDD column labelled VSS.
That is covered by the connectivity tests in `tests/test_rom_views.py`, not by
`--lvs`.

`timing_qualified` is `false`. Without `--timing` the Liberty is a logical-effort
estimate, which SPICE shows can be optimistic: at 16x6 it gives a shorter `CLK`
low than the read needs. Use a report from [`--characterize`](rom_timing.md). That
is simulation with estimated wires, so it stays `false` too.
