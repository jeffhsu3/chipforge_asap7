# Cell generators: `chipforge_asap7.devices`

`chipforge_asap7.devices` implements parametric ASAP7 FinFET cell generators for
logic and memory compiler periphery. Each generator is defined by a frozen `*Spec`
dataclass (whose parameters specify discrete fin counts, finger counts, and
architectural options) and a `build_*` function that generates gdspy layout.

Every generator has a command-line interface (`asap7-finfet`, `asap7-inverter`, ...)
whose options are derived automatically by [tyro](https://github.com/brentyi/tyro)
from the spec's types, defaults, and docstrings:
- Tuples are space-separated (`--rows 18 18 13 13`).
- Nested spec fields are prefixed with their field name (`--mux.selects 8`, `--stack.rows 4 6`).
- Boolean flags that default to `True` are inverted with `--no-<field>` (`--no-abut`).
- `--out` specifies the output GDS path (defaults to `<cell_name>.gds`).
- `--help` displays all available flags and defaults for any command.

---

## Architectural Guides

The cell library is decomposed into three architectural tiers:

### 1. [Parametric FinFET](finfet.md)
The fundamental active transistor primitive:
- Discrete fin, finger, and multiplier counting (`nmos_fin_122` nomenclature).
- Four-terminal device with dedicated opposite-polarity substrate tap (D, G, S, B).
- Standard-cell alignment (27 nm fin pitch, 54 nm CPP) and VT markers (RVT, LVT, SLVT, SRAMVT).
- DRC ceilings: public KLayout runset limits (`MAX_VERIFIABLE_FINS=12`) vs physical maximum (`MAX_FINS=18`).
- **[FinFET showcase](finfet.md#finfet-showcase)**: Visual layout comparison of six inspection geometries.

### 2. [Row Decoder and Standard-Cell Row Logic](row_decoder.md)
Horizontal row-based logic and wordline driving periphery:
- **`RowStack` substrate**: Pinned band heights (e.g. 135 nm bands on 270 nm standard-cell rails).
- **Stacked-band inverter** (`InverterSpec`): Continuous poly, shared diffusion, abutting/halved rows, input reach trimming.
- **Row support trio** (`RowSupportSpec`): Tap cell, filler cell, and decap cell; ordered placement rules (`filler row filler tap filler`) for clean latch-up and well enclosure.
- **Parametric NAND2** (`NandSpec`): Post-decode gate with M2 series node strapping.
- **Decoder sizing & driver slice** (`DriverSliceSpec`, `size_decoder`): Logical effort sizing and routing of 4-wordline driver slices on the bitcell pitch.

### 3. [Column IO and Bitline Periphery](column_io.md)
Vertical data-path periphery for SRAM arrays:
- **`rowcell` substrate**: 594 nm 2-row pairs with shared VDD, full-height M3 vertical tracks, and M4 horizontal interconnect.
- **Bitline leaf** (`BitlineMuxSpec`): Precharge pFETs and transmission-gate column multiplexer for 8T array bitline pitches, with support for M2/M4 bitline entries.
- **Write driver** (`WriteDriverSpec`): Cross-coupled keeper latch, write-enable pass gates, and transmission gates.
- **Output latch** (`OutputLatchSpec`): Sense amp SR hold latch with tri-state output driver.
- **Sense amplifier** (`SenseAmpRowSpec`): Current-latched differential pair on the IO row with precharge/equalizer.
- **Column IO block** (`IoColumnSpec`): Macro block assembly integrating mux group, sense amp, latch, and write driver with M4 cross-row routing (single-sided and two-sided).

---

## Generator Reference

| Spec Dataclass | Command | What it draws | Documentation |
| :--- | :--- | :--- | :--- |
| [`FinFETSpec`](finfet.md) | `asap7-finfet` | One four-terminal multi-finger FinFET with body tap | [Parametric FinFET](finfet.md) |
| [`InverterSpec`](row_decoder.md#stacked-band-inverter) | `asap7-inverter` | Stacked-band inverter, abutting or isolated (`dec_inv_62f_halved_AND`) | [Row Decoder](row_decoder.md#stacked-band-inverter) |
| [`RowSupportSpec`](row_decoder.md#row-support--tap-filler-decap) | `asap7-row-support` | Tap, filler, and decap cells sharing the logic's `RowStack` | [Row Support](row_decoder.md#row-support--tap-filler-decap) |
| [`NandSpec`](row_decoder.md#parametric-nand2) | `asap7-nand` | Parametric NAND2, the released post-decode gate | [Parametric NAND2](row_decoder.md#parametric-nand2) |
| [`DriverSliceSpec`](row_decoder.md#decoder-sizing-and-the-four-wordline-driver-slice) | `asap7-wl-slice` | Four-wordline decoder slice, sized by `size_decoder` | [Driver Slice](row_decoder.md#decoder-sizing-and-the-four-wordline-driver-slice) |
| [`BitlineMuxSpec`](column_io.md#parametric-bitline-leaf-precharge-and-column-mux) | `asap7-bitline-mux` | Precharge and column-mux leaf, stacked into an N:1 mux | [Bitline Mux](column_io.md#parametric-bitline-leaf-precharge-and-column-mux) |
| [`WriteDriverSpec`](column_io.md#write-driver-and-output-latch) | `asap7-write-driver` | Write driver on the IO row with cross-coupled latch | [Write Driver](column_io.md#write-driver-and-output-latch) |
| [`OutputLatchSpec`](column_io.md#write-driver-and-output-latch) | `asap7-output-latch` | SR latch and tri-state data output driver | [Output Latch](column_io.md#write-driver-and-output-latch) |
| [`SenseAmpRowSpec`](column_io.md#sense-amplifier-on-the-io-row) | `asap7-sense-amp-row` | Current-latched differential sense amplifier on IO row | [Sense Amplifier](column_io.md#sense-amplifier-on-the-io-row) |
| [`IoColumnSpec`](column_io.md#the-column-io-block) | `asap7-io-column` | Complete column IO block (mux group, SA, latch, write driver) | [Column IO Block](column_io.md#the-column-io-block) |

---

## Verification

All cell generators are verified using public open-source verification flows:
- **Design Rule Checking (DRC)**: Validated against the public educational ASAP7 KLayout runset (`drc_ASAP7.lydrc`).
- **Layout Versus Schematic (LVS)**: Extracted unit-fin transistor topologies matched against CDL netlist references.
- **SPICE Characterization**: Simulated with ngspice and Xyce using BSIM-CMG compact models.

See [Simulation and verification](verification.md) for full details on DRC, LVS, PEX, and SPICE testbenches.
