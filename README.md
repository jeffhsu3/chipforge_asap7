# chipforge-asap7

ASAP7 PDK layout collateral for the
[chipforge](https://github.com/jeffhsu3/chipforge) hardware ML compiler.

ASAP7 is chipforge core's **primary** PDK target. Core keeps its
ASAP7 SRAM wrapper, `sram_backend` strings and chip-top flows. What lives here
is the ASAP7 *geometry* that core's hand-built array compilers.

## Install

The package is a standalone, importable library — it does **not** depend on
chipforge, so any project can consume it.

```bash
git clone https://github.com/jeffhsu3/chipforge_asap7.git
cd chipforge_asap7
python -m pip install .          # constants and grid math
python -m pip install ".[gds]"  # also install gdspy for GDS drawing
```

The repository currently requires GitHub access. To develop against a local
checkout, run `python -m pip install -e ".[gds]"` from the repository root.
The package is licensed under [BSD 3-Clause](LICENSE); see
[third-party notices](THIRD_PARTY_NOTICES.md) for the ASAP7 attribution.

**`chipforge_asap7` has no required dependencies.** The layer map, the grid
constants and every pure-geometry helper (`column_x`, `column_gate_track`,
`fin_ys`, `gate_track_xs`, `centered_fin_ys`, `row_boundary_ys`,
`spans_excluding`, `snap_to_*`) import and run with nothing else installed — so
LEF / Liberty / sizing / DRC scripts that only need pitches and alignment math
can depend on it cheaply. gdspy is the optional `gds` extra, imported lazily and
only by the `draw_*` / `box` helpers; call `have_gdspy()` to check, or just call
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
| [`FinFETSpec`](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/devices.md#parametric-finfet) | One four-terminal multi-finger FinFET with its own body tap |
| [`InverterSpec`](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/devices.md#stacked-band-inverter) | Stacked-band inverter, abutting or isolated; the released `dec_inv_62f_halved_AND` |
| [`RowSupportSpec`](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/devices.md#row-support--tap-filler-decap) | Tap, filler and decap cells on the same `RowStack` as the logic |
| [`NandSpec`](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/devices.md#parametric-nand2) | NAND2, the released post-decode gate |
| [`DriverSliceSpec`](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/devices.md#decoder-sizing-and-the-four-wordline-driver-slice) | Four-wordline decoder slice, sized by `size_decoder` |
| [`BitlineMuxSpec`](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/devices.md#parametric-bitline-leaf-precharge-and-column-mux) | Precharge and column-mux leaf, stacked into an N:1 mux |
| [`WriteDriverSpec`, `OutputLatchSpec`](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/devices.md#write-driver-and-output-latch) | Write driver and output latch on the IO row |
| [`SenseAmpRowSpec`](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/devices.md#sense-amplifier-on-the-io-row) | Current-latched sense amplifier on the IO row |
| [`IoColumnSpec`](https://github.com/jeffhsu3/chipforge_asap7/blob/main/docs/devices.md#the-column-io-block) | One port's column IO: mux group, sense amp, latch, write driver |

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
