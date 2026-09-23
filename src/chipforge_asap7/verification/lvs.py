"""KLayout FinFET LVS runner for the public ASAP7 layer map."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..devices.bitline_mux import (
    BITLINE_MUX_PINS,
    BitlineMuxSpec,
    bitline_mux_group_pins,
)
from ..devices.driver_slice import DRIVER_SLICE_PINS, DriverSliceSpec
from ..devices.finfet import FinFETSpec
from ..devices.inverter import INVERTER_PINS, InverterSpec
from ..devices.io_column import (
    IoColumnSpec,
    block_series_nodes,
    io_column_pins,
)
from ..devices.io_column import (
    _block_devices as block_devices,
)
from ..devices.nand import NAND_PINS, NandSpec
from ..devices.output_latch import OUTPUT_LATCH_PINS, OutputLatchSpec
from ..devices.row_support import RowSupportSpec
from ..devices.sense_amp import SENSE_AMP_PINS
from ..devices.sense_amp_row import SenseAmpRowSpec
from ..devices.sizing import WORDLINES_PER_SLICE
from ..devices.write_driver import WRITE_DRIVER_PINS, WriteDriverSpec
from ..layout.grid import FIN_WIDTH

__all__ = [
    "LVSResult",
    "find_klayout",
    "lvs_deck_path",
    "normalize_asap7_cdl_reference",
    "render_driver_slice_lvs_schematic",
    "render_finfet_lvs_schematic",
    "render_inverter_lvs_schematic",
    "render_inverter_row_lvs_schematic",
    "render_nand_lvs_schematic",
    "render_nand_row_lvs_schematic",
    "render_row_support_lvs_schematic",
    "run_lvs",
]

_MOS_INSTANCE_RE = re.compile(
    r"^\s*(M\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)(.*)$", re.IGNORECASE
)
_PARAM_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*=\s*([^\s]+)")


@dataclass(frozen=True)
class LVSResult:
    """Files and status produced by one KLayout LVS invocation."""

    matched: bool
    returncode: int
    report: Path
    extracted_netlist: Path
    log: Path
    command: tuple[str, ...]


def lvs_deck_path() -> Path:
    """Return the packaged ASAP7 KLayout LVS/PEX preparation deck."""

    return Path(__file__).with_name("asap7.lvs")


def find_klayout(executable: str | Path | None = None) -> Path:
    """Resolve a KLayout executable with batch LVS support."""

    candidates: list[str | Path | None] = [
        executable,
        os.environ.get("KLAYOUT_BIN"),
        shutil.which("klayout"),
        Path.home() / "iv4/repos/OpenRAM/miniconda/bin/klayout",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "KLayout was not found; install KLayout >= 0.30 or set KLAYOUT_BIN"
    )


def render_finfet_lvs_schematic(
    spec: FinFETSpec,
    *,
    cell_name: str | None = None,
    pins: tuple[str, str, str, str] = ("D", "G", "S", "B"),
) -> str:
    """Render the unit-channel reference used by the FinFET LVS deck.

    KLayout's generic MOS extractor measures planar ``W`` and ``L``.  A
    FinFET is instead represented as one 7 nm-wide unit MOS for every physical
    FIN x GATE intersection.  Each unit is later emitted as BSIM-CMG
    ``NFIN=1``; parallel unit devices exactly preserve fins, fingers, and
    multipliers without treating planar width as a sizing knob.
    """

    name = cell_name or spec.cell_name
    d, g, s, b = pins
    instances = []
    for finger in range(spec.layout_fingers):
        # Source and drain orientation alternates with the shared S/D columns.
        # MOS LVS treats S/D as swappable, but preserving the physical order
        # makes extracted/reference netlists easier to inspect.
        left, right = (s, d) if finger % 2 == 0 else (d, s)
        for fin in range(spec.fins):
            instances.append(
                f"Mch_f{finger}_n{fin} {right} {g} {left} {b} {spec.model} "
                f"L={spec.gate_length}n W={FIN_WIDTH}n"
            )
    body = "\n".join(instances)
    return (
        "* ASAP7 FinFET LVS reference: one MOS per FIN x GATE channel\n"
        f".SUBCKT {name} {d} {g} {s} {b}\n"
        f"{body}\n"
        f".ENDS {name}\n"
        ".END\n"
    )


def render_inverter_lvs_schematic(
    spec: InverterSpec,
    *,
    cell_name: str | None = None,
) -> str:
    """Render a stacked-band inverter as one LVS MOS per physical fin.

    Same unit-fin representation as :func:`render_finfet_lvs_schematic`: the
    deck extracts every ``FIN x GATE`` intersection as its own device, so each
    band contributes ``fins x fingers`` parallel units between the output and
    its rail.  The cell carries no body tap, so run it with ``tie_bodies``
    enabled -- the substrate and well reach their supplies through the tap
    rows of the design that places it, exactly as for a released logic cell.
    """

    name = cell_name or spec.cell_name
    body = "\n".join(_inverter_units(spec))
    pins = " ".join(INVERTER_PINS)
    return (
        "* ASAP7 inverter LVS reference: one MOS per FIN x GATE channel\n"
        f".SUBCKT {name} {pins}\n"
        f"{body}\n"
        f".ENDS {name}\n"
        ".END\n"
    )


def render_inverter_row_lvs_schematic(
    spec: InverterSpec,
    count: int,
    *,
    cell_name: str,
) -> str:
    """Render `count` abutting inverters as one flat unit-fin reference.

    Abutting instances share diffusion, source columns and a gate track, so
    the extractor cannot keep them apart as separate subcircuits -- their
    shapes merge across the tile boundary.  The reference is therefore flat
    too, with pins ``A0/Y0 .. A<n-1>/Y<n-1>`` matching
    `chipforge_asap7.devices.inverter.build_inverter_row`.

    A row this reference matches is the real proof that the interleave works:
    every tile's shared edge gate has to have become its neighbour's finger,
    and no two drivers may share an output.

    An even `count` terminates itself -- the last instance is mirrored, so the
    gate it does not own faces its neighbour and its outer column is a
    contacted source.  An odd `count` ends on an unmirrored tile whose extra
    gate has no neighbour to claim it, leaving one finger driven by floating
    poly against an uncontacted diffusion stub per fin.  Those devices are
    real and are modelled here rather than quietly dropped; a placed design
    terminates the row with a filler instead.
    """

    instances: list[str] = []
    for index in range(count):
        instances += _inverter_units(
            spec, gate=f"A{index}", drain=f"Y{index}", prefix=f"M{index}_"
        )
    if count % 2:
        instances += [
            f"Medge_b{index}_n{fin} EDGE_D{index}_{fin} EDGE_G "
            f"{band.rail_net} {band.rail_net} {band.spec.model} "
            f"L={band.spec.gate_length}n W={FIN_WIDTH}n"
            for index, band in enumerate(spec.bands)
            for fin in range(band.fins)
        ]
    body = "\n".join(instances)
    header = " ".join([f"A{i}" for i in range(count)])
    header += " " + " ".join([f"Y{i}" for i in range(count)]) + " VDD VSS"
    return (
        f"* ASAP7 inverter row LVS reference: {count} abutting drivers\n"
        f".SUBCKT {cell_name} {header}\n"
        f"{body}\n"
        f".ENDS {cell_name}\n"
        ".END\n"
    )


def render_row_support_lvs_schematic(
    spec: RowSupportSpec,
    *,
    cell_name: str | None = None,
) -> str:
    """Render a row-support cell's devices, one MOS per FIN x GATE channel.

    A filler has none, and a tap has none either: its ties are diffusion, not
    devices, and what they connect is the well and substrate.  The way to see
    a tap work is to run LVS over a row that contains one with body ties
    *disabled* -- the bodies then have to reach their rails through the tap
    rather than by declaration.

    A decap does have devices: each plate's source and drain both land on its
    band's rail, and the two bands of a row share one floating gate conductor,
    so they sit back to back between VDD and VSS.  The plate net is per *row*,
    not per cell -- the poly is cut at every rail, which is what stops two
    stacked rows from becoming one capacitor with the wrong terminals.
    """

    name = cell_name or spec.cell_name
    instances = [
        f"M{index}_g{gate}_n{fin} {band.rail_net} PLATE{band.row} "
        f"{band.rail_net} {band.rail_net} {band.spec.model} "
        f"L={band.spec.gate_length}n W={FIN_WIDTH}n"
        for index, band in enumerate(spec.bands)
        for gate in range(len(spec.gate_xs))
        for fin in range(band.fins)
    ]
    body = "\n".join(instances)
    return (
        "* ASAP7 row-support LVS reference: one MOS per FIN x GATE channel\n"
        f".SUBCKT {name} VDD VSS\n"
        f"{body}\n"
        f".ENDS {name}\n"
        ".END\n"
    )


def _nand_units(
    spec: NandSpec,
    *,
    a: str = "A",
    b: str = "B",
    y: str = "Y",
    tag: str = "",
    prefix: str = "M",
) -> list[str]:
    """One unit-fin MOS line per FIN x GATE channel of a NAND2.

    The node between the two series nFETs of a stack is uncontacted diffusion,
    so the extractor sees one such net per fin; the reference names them the
    same way (``n<stack>_f<fin>``) rather than pretending they are one node.
    `tag` keeps instance and series-node names apart when several NANDs share
    one reference.
    """

    n_spec, p_spec = spec.n_band.spec, spec.p_band.spec
    lines: list[str] = []
    for stack in range(spec.fingers):
        for fin in range(n_spec.fins):
            node = f"n{stack}{tag}_f{fin}"
            lines.append(
                f"{prefix}Na{stack}_n{fin}{tag} {y} {a} {node} VSS {n_spec.model} "
                f"L={n_spec.gate_length}n W={FIN_WIDTH}n"
            )
            lines.append(
                f"{prefix}Nb{stack}_n{fin}{tag} {node} {b} VSS VSS {n_spec.model} "
                f"L={n_spec.gate_length}n W={FIN_WIDTH}n"
            )
        for fin in range(p_spec.fins):
            for which, gate in (("a", a), ("b", b)):
                lines.append(
                    f"{prefix}P{which}{stack}_n{fin}{tag} {y} {gate} VDD VDD "
                    f"{p_spec.model} L={p_spec.gate_length}n W={FIN_WIDTH}n"
                )
    return lines


def render_driver_slice_lvs_schematic(
    spec: DriverSliceSpec, *, cell_name: str | None = None
) -> str:
    """Four NAND2 + driver pairs sharing SEL, flat, one MOS per physical fin.

    Flat because the drivers are: interleaved inverter tiles share gate tracks
    and diffusion, so the extractor cannot keep them apart as subcircuits.
    Four is an even count, so the row terminates itself and there is no
    unpaired edge finger to model.
    """

    name = cell_name or spec.cell_name
    body: list[str] = []
    for i in range(WORDLINES_PER_SLICE):
        body += _nand_units(spec.nand, a="SEL", b=f"B{i}", y=f"N{i}", tag=f"_{i}")
        body += _inverter_units(
            spec.inverter, gate=f"N{i}", drain=f"WL{i}", prefix=f"MD{i}_"
        )
    return (
        "* ASAP7 driver slice LVS reference: one MOS per FIN x GATE channel\n"
        f".SUBCKT {name} {' '.join(DRIVER_SLICE_PINS)}\n"
        + "\n".join(body)
        + f"\n.ENDS {name}\n.END\n"
    )


def render_nand_lvs_schematic(spec: NandSpec, *, cell_name: str | None = None) -> str:
    """Render a NAND2 as one LVS MOS per physical fin (see `_nand_units`)."""

    name = cell_name or spec.cell_name
    body = "\n".join(_nand_units(spec))
    return (
        "* ASAP7 NAND2 LVS reference: one MOS per FIN x GATE channel\n"
        f".SUBCKT {name} {' '.join(NAND_PINS)}\n"
        f"{body}\n"
        f".ENDS {name}\n"
        ".END\n"
    )


def render_bitline_mux_lvs_schematic(
    spec: BitlineMuxSpec, *, cell_name: str | None = None
) -> str:
    """Render a bitline leaf as one LVS MOS per physical fin.

    Every source/drain column of the leaf is contacted, so unlike the NAND it
    has no per-fin internal node: each logical transistor is simply `fins`
    unit devices in parallel.
    """

    name = cell_name or spec.cell_name
    n_band, p_band = spec.bands
    lines = []
    for device, drain, gate, source, flavor, fins in spec.devices:
        band = n_band if flavor == "n" else p_band
        bulk = "VSS" if flavor == "n" else "VDD"
        for fin in range(fins):
            lines.append(
                f"{device}_f{fin} {drain} {gate} {source} {bulk} {band.spec.model} "
                f"L={band.spec.gate_length}n W={FIN_WIDTH}n"
            )
    body = "\n".join(lines)
    return (
        "* ASAP7 bitline leaf LVS reference: one MOS per FIN x GATE channel\n"
        f".SUBCKT {name} {' '.join(BITLINE_MUX_PINS)}\n"
        f"{body}\n"
        f".ENDS {name}\n"
        ".END\n"
    )


def render_write_driver_lvs_schematic(
    spec: WriteDriverSpec, *, cell_name: str | None = None
) -> str:
    """Render the write driver as one LVS MOS per physical fin."""
    bands = spec.bands
    lines = []
    for device, drain, gate, source, flavor, fins in spec.devices:
        band = bands["n0"] if flavor == "n" else bands["p0"]
        bulk = "VSS" if flavor == "n" else "VDD"
        for fin in range(fins):
            lines.append(
                f"{device}_f{fin} {drain} {gate} {source} {bulk} {band.spec.model} "
                f"L={band.spec.gate_length}n W={FIN_WIDTH}n"
            )
    name = cell_name or spec.cell_name
    return (
        "* ASAP7 write driver LVS reference: one MOS per FIN x GATE channel\n"
        f".SUBCKT {name} {' '.join(WRITE_DRIVER_PINS)}\n"
        + "\n".join(lines)
        + f"\n.ENDS {name}\n.END\n"
    )


def render_output_latch_lvs_schematic(
    spec: OutputLatchSpec, *, cell_name: str | None = None
) -> str:
    """Render the output latch as one LVS MOS per physical fin.

    The series nodes of the NAND stacks and the tristate are uncontacted
    diffusion, one net per fin in the extraction, so they are named per fin
    here too, as `_nand_units` does.
    """
    bands = spec.bands
    series = spec.series_nodes
    lines = []
    for device, drain, gate, source, flavor, fins in spec.devices:
        band = bands["n0"] if flavor == "n" else bands["p0"]
        bulk = "VSS" if flavor == "n" else "VDD"
        for fin in range(fins):
            d, s_ = (
                f"{net}_f{fin}" if net in series else net for net in (drain, source)
            )
            lines.append(
                f"{device}_f{fin} {d} {gate} {s_} {bulk} {band.spec.model} "
                f"L={band.spec.gate_length}n W={FIN_WIDTH}n"
            )
    name = cell_name or spec.cell_name
    return (
        "* ASAP7 output latch LVS reference: one MOS per FIN x GATE channel\n"
        f".SUBCKT {name} {' '.join(OUTPUT_LATCH_PINS)}\n"
        + "\n".join(lines)
        + f"\n.ENDS {name}\n.END\n"
    )


def render_sense_amp_row_lvs_schematic(
    spec: SenseAmpRowSpec, *, cell_name: str | None = None
) -> str:
    """Render the row sense amplifier as one LVS MOS per physical fin, series nodes per fin."""
    bands = spec.bands
    series = spec.series_nodes
    lines = []
    for device, drain, gate, source, flavor, fins in spec.devices:
        band = bands["n0"] if flavor == "n" else bands["p0"]
        bulk = "VSS" if flavor == "n" else "VDD"
        for fin in range(fins):
            d, s_ = (
                f"{net}_f{fin}" if net in series else net for net in (drain, source)
            )
            lines.append(
                f"{device}_f{fin} {d} {gate} {s_} {bulk} {band.spec.model} "
                f"L={band.spec.gate_length}n W={FIN_WIDTH}n"
            )
    name = cell_name or spec.cell_name
    return (
        "* ASAP7 row sense amplifier LVS reference: one MOS per FIN x GATE channel\n"
        f".SUBCKT {name} {' '.join(SENSE_AMP_PINS)}\n"
        + "\n".join(lines)
        + f"\n.ENDS {name}\n.END\n"
    )


def render_io_column_lvs_schematic(spec: IoColumnSpec, *, cell_name: str) -> str:
    """Render one port's column IO flat, one MOS per fin, series nodes per fin."""
    n_band, p_band = spec.mux.bands
    series = block_series_nodes(spec)
    lines = []
    for device, d, g, s, flavor, fins, _ in block_devices(spec):
        band = n_band if flavor == "n" else p_band
        bulk = "VSS" if flavor == "n" else "VDD"
        for fin in range(fins):
            d_, s_ = (f"{net}_f{fin}" if net in series else net for net in (d, s))
            lines.append(
                f"{device}_f{fin} {d_} {g} {s_} {bulk} {band.spec.model} "
                f"L={band.spec.gate_length}n W={FIN_WIDTH}n"
            )
    return (
        "* ASAP7 column IO LVS reference: one MOS per FIN x GATE channel\n"
        f".SUBCKT {cell_name} {' '.join(io_column_pins(spec))}\n"
        + "\n".join(lines)
        + f"\n.ENDS {cell_name}\n.END\n"
    )


def render_bitline_mux_group_lvs_schematic(
    spec: BitlineMuxSpec, *, cell_name: str
) -> str:
    """Render a stacked `spec.selects`-to-one column mux, flat, one MOS per fin.

    Leaf ``i`` has its own ``BL[i]``, ``BLN[i]``, ``YSEL[i]`` and ``YSELN[i]``;
    ``SA``, ``SAN`` and ``PRECHN`` are one net each across the group.  That is
    the whole claim of `build_bitline_mux_group` -- that the leaves' M3 tracks
    meet and nothing else does -- so a select wired to the wrong leaf, or two
    leaves sharing a bitline, fails here.
    """

    n_band, p_band = spec.bands
    local = {"BL", "BLN", "YSEL", "YSELN"}
    lines = []
    for i in range(spec.selects):
        for device, drain, gate, source, flavor, fins in spec.devices:
            band = n_band if flavor == "n" else p_band
            bulk = "VSS" if flavor == "n" else "VDD"
            nets = [
                f"{net}[{i}]" if net in local else net for net in (drain, gate, source)
            ]
            for fin in range(fins):
                lines.append(
                    f"{device}{i}_f{fin} {nets[0]} {nets[1]} {nets[2]} {bulk} {band.spec.model} "
                    f"L={band.spec.gate_length}n W={FIN_WIDTH}n"
                )
    body = "\n".join(lines)
    return (
        "* ASAP7 column-mux group LVS reference: one MOS per FIN x GATE channel\n"
        f".SUBCKT {cell_name} {' '.join(bitline_mux_group_pins(spec))}\n"
        f"{body}\n"
        f".ENDS {cell_name}\n"
        ".END\n"
    )


def render_nand_row_lvs_schematic(
    spec: NandSpec, count: int, *, cell_name: str | None = None
) -> str:
    """`count` independent NAND2s, as `build_nand_row` labels them (``A<k>/B<k>/Y<k>``).

    Butted tiles share only source columns, so unlike the inverter row there is
    no edge finger to account for: the row is exactly `count` NANDs.
    """

    name = cell_name or f"{spec.cell_name}_x{count}"
    pins = " ".join(f"{pin}{k}" for k in range(count) for pin in ("A", "B", "Y"))
    body = "\n".join(
        line
        for k in range(count)
        for line in _nand_units(spec, a=f"A{k}", b=f"B{k}", y=f"Y{k}", tag=f"_{k}")
    )
    return (
        "* ASAP7 NAND2 row LVS reference: one MOS per FIN x GATE channel\n"
        f".SUBCKT {name} {pins} VDD VSS\n"
        f"{body}\n"
        f".ENDS {name}\n"
        ".END\n"
    )


def _inverter_units(
    spec: InverterSpec,
    *,
    gate: str = "A",
    drain: str = "Y",
    prefix: str = "M",
) -> list[str]:
    """One unit-fin MOS line per FIN x GATE channel in `spec`."""

    return [
        f"{prefix}b{index}_f{finger}_n{fin} {drain} {gate} "
        f"{band.rail_net} {band.rail_net} {band.spec.model} "
        f"L={band.spec.gate_length}n W={FIN_WIDTH}n"
        for index, band in enumerate(spec.bands)
        for finger in range(spec.fingers)
        for fin in range(band.fins)
    ]


def normalize_asap7_cdl_reference(source: str | Path, output: str | Path) -> Path:
    """Convert released ASAP7 ``NFIN`` MOS lines to unit-fin LVS devices.

    The public CDL uses the process fin pitch (27 nm) as ``W`` and one device
    with ``NFIN=N``.  KLayout's generic MOS extractor instead measures the
    drawn 7 nm FIN stripe and emits one device per stripe.  Expanding the CDL
    to the same unit-fin representation makes the comparison structural while
    preserving every D/G/S/B connection.  This is reference normalization,
    not a compact-model rewrite.
    """

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"CDL reference does not exist: {source_path}")

    # Fold SPICE continuation lines before inspecting instance parameters.
    statements: list[str] = []
    for raw_line in source_path.read_text().splitlines():
        if raw_line.lstrip().startswith("+") and statements:
            statements[-1] += " " + raw_line.lstrip()[1:].strip()
        else:
            statements.append(raw_line)

    normalized: list[str] = []
    for statement in statements:
        match = _MOS_INSTANCE_RE.match(statement)
        if not match or statement.lstrip().startswith("*"):
            normalized.append(statement)
            continue

        instance, drain, gate, source_net, bulk, model, tail = match.groups()
        parameters = {key.lower(): value for key, value in _PARAM_RE.findall(tail)}
        if "nfin" not in parameters:
            normalized.append(statement)
            continue
        try:
            fins = int(float(parameters["nfin"]))
            fingers = int(float(parameters.get("nf", "1")))
            multiplier = int(float(parameters.get("m", "1")))
        except ValueError as error:
            raise ValueError(f"invalid ASAP7 MOS sizing in: {statement}") from error
        if fins < 1 or fingers < 1 or multiplier < 1:
            raise ValueError(f"NFIN, NF and M must be positive in: {statement}")

        # A finger is a parallel device of NFIN fins, drawn as such.
        length = parameters.get("l", "20n")
        for copy_index in range(fins * fingers * multiplier):
            normalized.append(
                f"{instance}__uf{copy_index + 1} {drain} {gate} {source_net} "
                f"{bulk} {model} L={length} W={FIN_WIDTH}n"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(normalized).rstrip() + "\n")
    return output_path


def _report_matched(report: Path, log_text: str) -> bool:
    """Read KLayout's explicit comparison result from its log/report."""

    lowered = log_text.lower()
    if "circuits match" in lowered or "lvs completed successfully" in lowered:
        return True
    if "circuits do not match" in lowered or "mismatch" in lowered:
        return False

    # LayoutVsSchematic is available from the optional KLayout Python wheel.
    # Keep the CLI runner dependency-free and use it only when installed.
    try:
        import klayout.db as kdb

        database = kdb.LayoutVsSchematic()
        database.read(str(report))
        return bool(database.is_clean())
    except (ImportError, RuntimeError, AttributeError):
        return False


def run_lvs(
    gds: str | Path,
    schematic: str | Path,
    output_dir: str | Path,
    *,
    cell_name: str | None = None,
    klayout: str | Path | None = None,
    deck: str | Path | None = None,
    compare: bool = True,
    asap7_standard_cell: bool = False,
    tie_bodies: bool | None = None,
    merge_fin_diffusion: bool | None = None,
    flatten_circuits: Sequence[str] = (),
    blank_circuits: Sequence[str] = (),
    double_implant_is_tap: bool = True,
    flat: bool = False,
    timeout: float = 300,
) -> LVSResult:
    """Run ASAP7 KLayout LVS and retain the LVSDB needed by KPEX.

    ``compare=False`` prepares connectivity and an LVSDB even when no trusted
    schematic is available.  Normal verification should keep comparison on.

    ``asap7_standard_cell`` does two independent things: it expands a released
    ``NFIN`` CDL to unit-fin devices, and it ties the well and substrate to
    VDD/VSS.  A cell this package generates without a body tap -- the stacked
    inverter, for instance -- needs the second without the first, and says so
    with ``tie_bodies=True``.  It defaults to following
    ``asap7_standard_cell``, so existing callers are unaffected.

    The rest are for layouts this package did not draw -- an assembled macro
    holding released standard cells and an SRAM array:

    * ``merge_fin_diffusion`` takes source/drain as the whole ACTIVE outside
      the gate, as the ASAP7 Calibre deck and so every released CDL does; the
      uncontacted node of a series stack is then one net instead of one per
      fin.  Follows ``asap7_standard_cell`` unless given.
    * ``flatten_circuits`` (glob patterns) dissolves cells on both sides.  A
      bitcell's transistors are completed by its neighbours, so its layout cell
      holds a fraction of the devices its schematic does.
    * ``blank_circuits`` compares cells by their pins' names alone.  Use it for
      a released cell whose GDS and CDL order a series stack differently, once
      `series_order_is_the_only_difference` has said that is all it is.
    * ``double_implant_is_tap=False`` stops ACTIVE under both implants acting
      as a well tap.  It is a way to see past that error, not around it.
    * ``flat`` extracts without hierarchy.  KLayout 0.30's hierarchical
      extractor has been seen to stop on an internal assertion
      (``dbHierNetworkProcessor.cc ... id_new != 0``) on some arrays; flat
      extraction is slower and cannot compare a cell by pin name.
    """

    gds_path = Path(gds).expanduser().resolve()
    schematic_path = Path(schematic).expanduser().resolve()
    if not gds_path.is_file():
        raise FileNotFoundError(f"layout does not exist: {gds_path}")
    if not schematic_path.is_file():
        raise FileNotFoundError(f"schematic does not exist: {schematic_path}")

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    tie = asap7_standard_cell if tie_bodies is None else tie_bodies
    merge = asap7_standard_cell if merge_fin_diffusion is None else merge_fin_diffusion
    if asap7_standard_cell:
        schematic_path = normalize_asap7_cdl_reference(
            schematic_path, out / "reference_unit_fins.cdl"
        )
    report = out / "layout.lvsdb.gz"
    extracted = out / "layout_extracted.spice"
    log = out / "klayout_lvs.log"
    command = [
        str(find_klayout(klayout)),
        "-b",
        "-r",
        str(Path(deck).resolve() if deck else lvs_deck_path()),
        "-rd",
        f"input={gds_path}",
        "-rd",
        f"schematic={schematic_path}",
        "-rd",
        f"report={report}",
        "-rd",
        f"target_netlist={extracted}",
        "-rd",
        f"compare={'true' if compare else 'false'}",
        "-rd",
        f"tie_standard_cell_bodies={'true' if tie else 'false'}",
        "-rd",
        f"merge_fin_diffusion={'true' if merge else 'false'}",
        "-rd",
        f"double_implant_is_tap={'true' if double_implant_is_tap else 'false'}",
        "-rd",
        f"flatten_circuits={','.join(flatten_circuits)}",
        "-rd",
        f"blank_circuits={','.join(blank_circuits)}",
        "-rd",
        f"run_mode={'flat' if flat else 'deep'}",
        "-rd",
        "top_lvl_pins=true",
    ]
    if cell_name:
        command.extend(["-rd", f"topcell={cell_name}"])

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    log_text = completed.stdout + completed.stderr
    log.write_text(log_text)
    if completed.returncode != 0:
        raise RuntimeError(
            f"KLayout LVS failed with exit {completed.returncode}; see {log}"
        )
    if not report.is_file() or not extracted.is_file():
        raise RuntimeError(f"KLayout did not create the expected outputs; see {log}")

    matched = _report_matched(report, log_text) if compare else True
    return LVSResult(
        matched=matched,
        returncode=completed.returncode,
        report=report,
        extracted_netlist=extracted,
        log=log,
        command=tuple(command),
    )
