# chipforge-asap7

ASAP7 PDK layout collateral for the
[chipforge](https://github.com/jeffhsu3/chipforge) hardware ML compiler.

ASAP7 is a chipforge core PDK target and critical for deteremining the size and types of models that are hardcodable and is loosely based around [gLayout](https://github.com/ReaLLMASIC/gLayout). Core keeps its
ASAP7 SRAM wrapper, `sram_backend` strings and chip-top flows. What lives here
is the ASAP7 *geometry* that core's hand-built array compilers use, the ROM and
SRAM compilers, and, like chipforge_gt2n, the ROM macro backends core's
`rom_macro_backend` names (`asap7_brom`, `asap7_nor_rom`).

chipforge_asap7 is also a critical dependency for opensource FinFET SRAM compilers (primarily 8T versions, see https://github.com/shao-chien-lu/OpenFinRAM)

## Install

The package is a standalone, importable library — it does **not** depend on
chipforge, so any project can consume it. Only the ROM macro backends need
chipforge, through the optional `chipforge` extra.

```bash
git clone https://github.com/jeffhsu3/chipforge_asap7.git
cd chipforge_asap7
python -m pip install .          # constants and grid math
python -m pip install ".[gds]"  # also gdspy for drawing, tyro for the commands
python -m pip install ".[plot]"  # matplotlib, to plot the SPICE benches
```

To develop against a local
checkout, run `python -m pip install -e ".[gds]"` from the repository root.
The package is licensed under [BSD 3-Clause](LICENSE); see
[third-party notices](THIRD_PARTY_NOTICES.md) for the ASAP7 attribution.

**`chipforge_asap7` has no required dependencies.** The layer map, the grid
constants and every pure-geometry helper (`column_x`, `column_gate_track`,
`fin_ys`, `gate_track_xs`, `centered_fin_ys`, `row_boundary_ys`,
`spans_excluding`, `snap_to_*`) import and run with nothing else installed — so
LEF / Liberty / sizing / DRC scripts that only need pitches and alignment math
can depend on it cheaply. gdspy is the optional `gds` extra, imported lazily and
only by the `draw_*` / `box` helpers (the extra also brings tyro, which the
`asap7-*` commands parse their options with); call `have_gdspy()` to check, or just call
a drawing helper and get an actionable `ImportError`. The package ships
`py.typed`, so its annotations are visible to consumers.

## Command line

One `asap7` command, laid out like chipforge_gt2n's `gt2n`: a group, then a
generator, then that generator's own options (`asap7 rom brom -h`).

```bash
asap7 cell finfet --fins 2 --fingers 2 --output finfet.gds
asap7 rom brom --words-json words.json --word-bits 8 --column-mux 4 \
    --self-timed --registered-output --output-dir build/brom
asap7 rom nor --array.words 64 --array.bits 8
asap7 sram macro-bench --wordlines 32 --word-bits 2
```

Groups: `cell`, `lane`, `sram` (`6t-read`, `column-bench`, `macro-bench`), `rom`
(`nor`, `brom`, `brom-timing`, `nor-read`, `nor-address-search`), `verify` and
`reduce`. Shape options use gt2n's names (`--word-bits`, `--column-mux`,
`--words-json`, `--output-dir`, and `--output` for a single GDS); the old
names (`--bits`, `--ways`, `--out`, a positional words file) still work, and so
do the single-purpose `asap7-*` scripts. The B-ROM manifest carries gt2n's
shape keys (`architecture`, `platform`, `depth`, `word_bits`,
`physical_word_bits`, `column_mux`, `column_mux_in_macro`, `address_bits`)
beside its own `spec`.

## ChipForge ROM backends

With the `chipforge` extra (`uv sync --extra chipforge`), importing
`chipforge_asap7` registers `rom_macro_backend="asap7_brom"` (two-phase read)
and `"asap7_nor_rom"` (edge flow-through) in chipforge's ROM interface
registry, with a 256-row bank; `register_asap7_brom_backend(rows, root,
lanes)` re-registers another. `chipforge_asap7.chipforge_backend` holds the
port contracts, collateral resolvers, wrappers and estimated Liberty that used
to live in core; core's `chipforge.memory.asap7_brom` and `asap7_nor_rom`
re-export them. The built banks stay in core's `macros/` (or
`$CHIPFORGE_ASAP7_MACROS`). Without chipforge the package imports as before.

## `chipforge_asap7.layout`

All coordinates are integers in nanometres, matching the gdspy libraries these
flows emit (`unit=1e-9, precision=1e-10`).

```python
import gdspy
from chipforge_asap7.layout import (
    column_x,
    draw_fin_grid,
    draw_gate_grid,
    draw_gate_cuts,
    row_boundary_ys,
    spans_excluding,
    WORDLINE_PITCH,
)

lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
cell = lib.new_cell("ROM_FULL")

height = (depth + 1) * WORDLINE_PITCH
draw_fin_grid(cell, left_margin, total_width, 0, height)
draw_gate_grid(cell, left_margin, total_width, -1200, height)
draw_gate_cuts(
    cell,
    spans_excluding(left_margin - 17, total_width - 17, buffer_strips),
    row_boundary_ys(depth, WORDLINE_PITCH),
)
```

## `chipforge_asap7.devices`

Parametric ASAP7 cells, each a frozen `*Spec` dataclass (sizes in whole fins
and fingers, names derived from them) and a `build_*` function that draws it
with gdspy. Every one is held to the public KLayout DRC runset and matched
against a unit-fin LVS reference in the tests. The full write-up of each is in
[docs/devices.md](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/devices.md).

| Spec | What it draws |
| --- | --- |
| [`FinFETSpec`](docs/finfet.md) | One four-terminal multi-finger FinFET with its own body tap |
| [`InverterSpec`](docs/row_decoder.md#stacked-band-inverter) | Stacked-band inverter, abutting or isolated; the released `dec_inv_62f_halved_AND` |
| [`RowSupportSpec`](docs/row_decoder.md#row-support--tap-filler-decap) | Tap, filler and decap cells on the same `RowStack` as the logic |
| [`NandSpec`](docs/row_decoder.md#parametric-nand2) | NAND2, the released post-decode gate |
| [`DriverSliceSpec`](docs/row_decoder.md#decoder-sizing-and-the-four-wordline-driver-slice) | Four-wordline decoder slice, sized by `size_decoder` |
| [`BitlineMuxSpec`](docs/column_io.md#parametric-bitline-leaf-precharge-and-column-mux) | Precharge and column-mux leaf, stacked into an N:1 mux |
| [`WriteDriverSpec`, `OutputLatchSpec`](docs/column_io.md#write-driver-and-output-latch) | Write driver and output latch on the IO row |
| [`SenseAmpRowSpec`](docs/column_io.md#sense-amplifier-on-the-io-row) | Current-latched sense amplifier on the IO row |
| [`IoColumnSpec`](docs/column_io.md#the-column-io-block) | One port's column IO: mux group, sense amp, latch, write driver |
| [`RomBitcellSpec`](docs/rom.md) | Via-programmed NOR-ROM tile with a poly wordline, 81 x 81 nm per bit at two fins |
| [`RomDecoderSpec`](docs/rom.md#wordline-decoder) | The ROM's wordline decoder and predecoders: address pins to an AND gate per word, interleaved over rows above the array, with the predecoders and bitline ends in a center lane between two half arrays |
| [`BromSpec`](docs/brom.md) | Four-word BROM compact-array compiler with CGEN, block decoder, precharge/readout, mux and output latches; routed GDS, transistor CDL, LEF abstract and Verilog model |

## Simulation and verification

[docs/verification.md](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/verification.md) covers the open flows the generators are checked
with:

- [SPICE testbenches](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/verification.md#single-device-spice-testbench) for one FinFET
  and for OpenFinRAM's sense amplifier, plus a 6T SRAM read path from wordline
  to Q, on BSIM-CMG 107 through OpenVAF.
- [Open LVS and PEX](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/verification.md#open-lvs-and-pex): KLayout LVS, FasterCap
  capacitance and KPEX resistance, correlated against the released inverter.
  Research grade, not sign-off.
- [Hierarchical LVS](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/verification.md#lvs-of-an-assembled-macro) of a macro built
  from released standard cells, hard IP and an SRAM array.
- [Layout reduction](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/verification.md#layout-reduction): delete and shrink what LVS,
  DRC and extraction do not need.
- Minimum-area proofs for the [NOR-ROM topology](docs/rom.md#minimum-area-proof-for-this-topology)
  and [retained 6T SRAM template](docs/sram_6t_minimum.md), with SMT queries,
  measured GDS witnesses, connectivity extraction and pinned gdscheck validation.

## Development

```bash
uv sync --all-extras   # dev group + the gds extra
uv run pytest           # add --runslow for the sizing searches and the 1024x16 shape
uv run ruff check src tests && uv run ruff format src tests
uv build               # wheel + sdist into dist/
```
