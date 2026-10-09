# chipforge-asap7

ASAP7 PDK layout collateral for the
[chipforge](https://github.com/jeffhsu3/chipforge) hardware ML compiler.

ASAP7 is a chipforge core PDK target and critical for deteremining the size and types of models that are hardcodable and is loosely based around [gLayout](https://github.com/ReaLLMASIC/gLayout). Core keeps its
ASAP7 SRAM wrapper, `sram_backend` strings and chip-top flows. What lives here
is the ASAP7 *geometry* that core's hand-built array compilers.

chipforge_asap7 is also a critical dependency for opensource FinFET compilers (primarily 8T versions, see OpenFinRAM)

## Install

The package is a standalone, importable library — it does **not** depend on
chipforge, so any project can consume it.

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

## Simulation and verification

[docs/verification.md](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/verification.md) covers the open flows the generators are checked
with:

- [SPICE testbenches](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/verification.md#single-device-spice-testbench) for one FinFET
  and for OpenFinRAM's sense amplifier, on BSIM-CMG 107 through OpenVAF.
- [Open LVS and PEX](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/verification.md#open-lvs-and-pex): KLayout LVS, FasterCap
  capacitance and KPEX resistance, correlated against the released inverter.
  Research grade, not sign-off.
- [Hierarchical LVS](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/verification.md#lvs-of-an-assembled-macro) of a macro built
  from released standard cells, hard IP and an SRAM array.
- [Layout reduction](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/verification.md#layout-reduction): delete and shrink what LVS,
  DRC and extraction do not need.

## Development

```bash
uv sync --all-extras   # dev group + the gds extra
uv run pytest
uv run ruff check src tests && uv run ruff format src tests
uv build               # wheel + sdist into dist/
```
