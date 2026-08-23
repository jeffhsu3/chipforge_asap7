"""Open ASAP7 resistance/capacitance extraction and SPICE assembly.

The flow deliberately separates the public evidence from its assumptions:

* KLayout LVS establishes devices, terminals, pins, and connected polygons.
* KPEX's open 2.5-D resistance engine distributes sheet/via resistance.
* FasterCap solves the 3-D Maxwell capacitance matrix.
* This module reconnects each MOS terminal to its distributed-R port and
  emits one BSIM-CMG unit-fin instance per extracted physical channel.

ASAP7 does not publish the sign-off LVS/xACT decks or a complete vertical
material stack.  The generated report therefore remains research-grade and
records the calibrated/extrapolated values from :mod:`.stack`.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..devices.spice import translate_asap7_model
from .lvs import LVSResult, find_klayout, run_lvs
from .stack import calibration_manifest, write_kpex_technology

__all__ = [
    "PEXResult",
    "add_distributed_resistance",
    "add_maxwell_capacitance",
    "find_fastercap",
    "normalize_unit_finfet_netlist",
    "reference_subcircuit_pins",
    "reorder_subcircuit_pins",
    "run_open_pex",
]

_FASTER_CAP_AXIS_RE = re.compile(r"^g\d+_(.*)$")
_R_PORT_RE = re.compile(r"^P(\d+)(?:\.|$)")
_MODEL_INSTANCE_RE = re.compile(
    r"^\s*M\S*\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)", re.IGNORECASE
)


@dataclass(frozen=True)
class PEXResult:
    """Stable output paths from one complete open PEX run."""

    cell_name: str
    lvs: LVSResult
    post_layout_netlist: Path
    raw_klayout_netlist: Path
    report: Path
    technology: Path
    calibration: Path
    fastercap_input: Path
    fastercap_log: Path
    capacitance_raw_csv: Path
    capacitance_csv: Path
    resistance_csv: Path
    resistance_json: Path
    translated_model_card: Path | None


def _require_kpex() -> None:
    try:
        import klayout_pex  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "open PEX needs KPEX; install chipforge-asap7[pex] or "
            "`pip install klayout-pex`"
        ) from error


def find_fastercap(executable: str | Path | None = None) -> Path:
    """Resolve the official FasterCap console executable."""

    candidates: list[str | Path | None] = [
        executable,
        os.environ.get("KPEX_FASTERCAP_EXE"),
        os.environ.get("FASTERCAP_EXE"),
        shutil.which("FasterCap"),
        shutil.which("fastercap"),
        Path("/opt/FasterCap/FasterCap"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    raise FileNotFoundError(
        "FasterCap was not found; pass fastercap=..., set FASTERCAP_EXE, "
        "or build the official ediloren/FasterCap source"
    )


def _solver_environment(
    *, library_dir: str | Path | None, threads: int | None
) -> dict[str, str]:
    environment = os.environ.copy()
    if library_dir:
        library = str(Path(library_dir).expanduser().resolve())
        previous = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = (
            f"{library}{os.pathsep}{previous}" if previous else library
        )
    if threads is not None:
        if threads < 1:
            raise ValueError("FasterCap threads must be positive")
        environment["OMP_NUM_THREADS"] = str(threads)
    return environment


def _run_fastercap(
    executable: Path,
    input_file: Path,
    log: Path,
    *,
    tolerance: float,
    d_coeff: float,
    refinement: float,
    library_dir: str | Path | None,
    threads: int | None,
    timeout: float,
) -> tuple[str, ...]:
    if tolerance <= 0 or d_coeff <= 0 or refinement <= 0:
        raise ValueError(
            "FasterCap tolerance, d_coeff, and refinement must be positive"
        )
    command = (
        str(executable),
        "-b",
        "-i",
        "-v",
        f"-a{tolerance:g}",
        f"-d{d_coeff:g}",
        f"-m{refinement:g}",
        "-ap",
        str(input_file),
    )
    completed = subprocess.run(
        command,
        cwd=input_file.parent,
        env=_solver_environment(library_dir=library_dir, threads=threads),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    log.write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(
            f"FasterCap failed with exit {completed.returncode}; see {log}"
        )
    return command


def _find_or_create_net(circuit: Any, nets: dict[str, Any], name: str) -> Any:
    net = nets.get(name)
    if net is None:
        net = circuit.create_net(name)
        nets[name] = net
    return net


def _request_terminal_candidates(request: Any, node: Any) -> list[Any]:
    """Rebuild KPEX's polygon-port order for one resistance node."""

    candidates: list[Any] = []
    for terminal in request.device_terminals:
        for layer_region in terminal.region_by_layer:
            if layer_region.layer.canonical_layer_name != node.layer_name:
                continue
            # RExtractor creates one polygon port for every protobuf shape.
            candidates.extend([terminal] * len(layer_region.region.shapes))
    return candidates


def add_distributed_resistance(
    netlist: Any,
    top_cell_name: str,
    request: Any,
    result: Any,
) -> tuple[list[dict[str, Any]], int]:
    """Insert KPEX R elements and reconnect MOS terminals to their R ports.

    KPEX 0.3.x emits device-terminal nodes in its protobuf result but its stock
    netlist expander leaves the original MOS attached directly to the top net.
    This function uses the corresponding request's device/terminal IDs to make
    the electrically essential reconnection before inserting resistors.
    """

    import klayout.db as kdb

    circuit = netlist.circuit_by_name(top_cell_name)
    if circuit is None:
        raise ValueError(f"extracted netlist has no circuit {top_cell_name!r}")
    nets = {net.expanded_name(): net for net in circuit.each_net()}
    requests = {
        network.net_name: network for network in request.net_extraction_requests
    }

    resistor_class = kdb.DeviceClassResistor()
    resistor_class.name = "PEX_R"
    resistor_class.description = "Distributed resistance extracted by KPEX/2.5D"
    netlist.add(resistor_class)

    records: list[dict[str, Any]] = []
    reconnect_count = 0
    resistor_index = 0
    for network in result.networks:
        nodes = {node.node_id: node for node in network.nodes}
        network_request = requests.get(network.net_name)
        if network_request is None:
            raise ValueError(f"missing R request for net {network.net_name!r}")

        for node in network.nodes:
            # KIND_DEVICE_TERMINAL == 2 in kpex/r/r_network.proto.
            if int(node.node_kind) != 2:
                continue
            port_match = _R_PORT_RE.match(node.node_name)
            if port_match is None:
                raise ValueError(f"cannot decode KPEX device port {node.node_name!r}")
            candidates = _request_terminal_candidates(network_request, node)
            port_index = int(port_match.group(1))
            if port_index >= len(candidates):
                raise ValueError(
                    f"KPEX port {node.node_name!r} has no matching terminal on "
                    f"{node.layer_name}; candidates={len(candidates)}"
                )
            terminal = candidates[port_index]
            device = circuit.device_by_id(terminal.device_id)
            if device is None:
                raise ValueError(
                    f"KPEX referenced missing device id {terminal.device_id}"
                )
            port_net = _find_or_create_net(circuit, nets, node.net_name)
            device.connect_terminal(terminal.terminal_id, port_net)
            reconnect_count += 1

        for element in network.elements:
            node_a = nodes[element.node_a.node_id]
            node_b = nodes[element.node_b.node_id]
            net_a = _find_or_create_net(circuit, nets, node_a.net_name)
            net_b = _find_or_create_net(circuit, nets, node_b.net_name)
            if net_a == net_b:
                continue
            resistor_index += 1
            device = circuit.create_device(resistor_class, f"ext_{resistor_index}")
            device.connect_terminal("A", net_a)
            device.connect_terminal("B", net_b)
            device.set_parameter("R", element.resistance)
            records.append(
                {
                    "name": f"Rext_{resistor_index}",
                    "parent_net": network.net_name,
                    "net_a": node_a.net_name,
                    "net_b": node_b.net_name,
                    "layer_a": node_a.layer_name,
                    "layer_b": node_b.layer_name,
                    "resistance_ohm": element.resistance,
                }
            )
    return records, reconnect_count


def _axis_name(name: str) -> str:
    match = _FASTER_CAP_AXIS_RE.match(name)
    return match.group(1) if match else name


def _collapse_capacitance_matrix(
    conductor_names: list[str],
    rows: list[list[float]],
    *,
    available_nets: set[str],
    substrate_net: str,
) -> tuple[list[str], list[list[float]]]:
    if len(rows) != len(conductor_names) or any(
        len(row) != len(conductor_names) for row in rows
    ):
        raise ValueError("FasterCap returned a non-square capacitance matrix")

    targets: list[str] = []
    for raw_name in conductor_names:
        name = _axis_name(raw_name)
        target = substrate_net if name == "VSUBS" else name
        if target not in available_nets:
            raise ValueError(
                f"FasterCap conductor {name!r} has no extracted circuit net; "
                f"available nets: {sorted(available_nets)}"
            )
        targets.append(target)

    # Shorted conductors (notably VSUBS and a physical VSS/B body route) must
    # be combined as C' = P^T C P, rather than simply dropping their mutual C.
    unique_targets = list(dict.fromkeys(targets))
    target_index = {name: index for index, name in enumerate(unique_targets)}
    collapsed = [[0.0 for _ in unique_targets] for _ in unique_targets]
    for i, row in enumerate(rows):
        ci = target_index[targets[i]]
        for j, value in enumerate(row):
            cj = target_index[targets[j]]
            collapsed[ci][cj] += value
    return unique_targets, collapsed


def add_maxwell_capacitance(
    netlist: Any,
    top_cell_name: str,
    conductor_names: list[str],
    rows: list[list[float]],
    *,
    substrate_net: str,
    threshold_farad: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert a Maxwell matrix to mutual/ground capacitors in ``netlist``."""

    import klayout.db as kdb

    if threshold_farad < 0:
        raise ValueError("capacitance threshold cannot be negative")
    circuit = netlist.circuit_by_name(top_cell_name)
    if circuit is None:
        raise ValueError(f"extracted netlist has no circuit {top_cell_name!r}")
    nets = {net.expanded_name(): net for net in circuit.each_net()}
    axes, matrix = _collapse_capacitance_matrix(
        conductor_names,
        rows,
        available_nets=set(nets),
        substrate_net=substrate_net,
    )
    if axes[0] != substrate_net:
        raise ValueError(
            f"first FasterCap conductor must resolve to substrate {substrate_net!r}, "
            f"got {axes[0]!r}"
        )

    capacitor_class = kdb.DeviceClassCapacitor()
    capacitor_class.name = "PEX_C"
    capacitor_class.description = "Maxwell capacitance extracted by FasterCap"
    netlist.add(capacitor_class)

    records: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def add_cap(net_a_name: str, net_b_name: str, value: float, kind: str) -> None:
        if net_a_name == net_b_name:
            return
        if value <= threshold_farad:
            warnings.append(
                {
                    "kind": "nonpositive_or_below_threshold_capacitance",
                    "net_a": net_a_name,
                    "net_b": net_b_name,
                    "capacitance_farad": value,
                }
            )
            return
        index = len(records) + 1
        device = circuit.create_device(capacitor_class, f"ext_{index}")
        device.connect_terminal("A", nets[net_a_name])
        device.connect_terminal("B", nets[net_b_name])
        device.set_parameter("C", value)
        records.append(
            {
                "name": f"Cext_{index}",
                "net_a": net_a_name,
                "net_b": net_b_name,
                "kind": kind,
                "capacitance_farad": value,
            }
        )

    for i, row in enumerate(matrix):
        residual = row[i]
        for j, matrix_value in enumerate(row):
            if i == j:
                continue
            mutual = -matrix_value
            residual -= mutual
            if j > i:
                add_cap(axes[i], axes[j], mutual, "mutual")
        if i > 0:
            add_cap(axes[i], axes[0], residual, "substrate_or_infinity")
    return records, warnings


def _fold_spice_continuations(text: str) -> list[str]:
    statements: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("+") and statements:
            statements[-1] += " " + line.lstrip()[1:].strip()
        else:
            statements.append(line)
    return statements


def reference_subcircuit_pins(schematic: str | Path, cell_name: str) -> tuple[str, ...]:
    """Read a subcircuit's positional pin contract from SPICE/CDL."""

    path = Path(schematic).expanduser().resolve()
    for statement in _fold_spice_continuations(path.read_text()):
        fields = statement.split()
        if (
            len(fields) >= 2
            and fields[0].lower() == ".subckt"
            and fields[1].lower() == cell_name.lower()
        ):
            pins: list[str] = []
            for field in fields[2:]:
                if "=" in field or field.lower().startswith("params:"):
                    break
                pins.append(field)
            if not pins:
                raise ValueError(f"subcircuit {cell_name!r} has no pins in {path}")
            return tuple(pins)
    raise ValueError(f"subcircuit {cell_name!r} was not found in {path}")


def reorder_subcircuit_pins(netlist: str, cell_name: str, pins: tuple[str, ...]) -> str:
    """Restore reference pin order while retaining extracted net names."""

    lines = netlist.splitlines()
    for index, line in enumerate(lines):
        fields = line.split()
        if (
            len(fields) >= 2
            and fields[0].lower() == ".subckt"
            and fields[1].lower() == cell_name.lower()
        ):
            extracted_pins = fields[2:]
            if {pin.lower() for pin in extracted_pins} != {pin.lower() for pin in pins}:
                raise ValueError(
                    f"reference/extracted pin sets differ for {cell_name}: "
                    f"{pins} vs {tuple(extracted_pins)}"
                )
            lines[index] = f".SUBCKT {cell_name} {' '.join(pins)}"
            return "\n".join(lines).rstrip() + "\n"
    raise ValueError(f"post-layout netlist has no subcircuit {cell_name!r}")


def normalize_unit_finfet_netlist(
    text: str,
    *,
    dialect: str = "ngspice_osdi",
    zero_resistance_floor_ohm: float = 1e-3,
) -> str:
    """Rewrite KLayout planar-MOS syntax as one-fin BSIM-CMG instances."""

    if dialect not in {"ngspice_osdi", "hspice"}:
        raise ValueError("dialect must be 'ngspice_osdi' or 'hspice'")
    if zero_resistance_floor_ohm <= 0:
        raise ValueError("zero-resistance floor must be positive")
    output = [
        "* ASAP7 post-layout netlist: one BSIM-CMG instance per physical FIN x GATE",
        "* Extracted RC is external; BSIM-CMG supplies device-internal parasitics.",
        f"* Exact KPEX port shorts are emitted as {zero_resistance_floor_ohm:g} ohm for numerical conditioning.",
        "* Use .options rshunt=1e15 in fixtures that include floating dummy gates.",
    ]
    for statement in _fold_spice_continuations(text):
        stripped = statement.strip()
        if stripped[:1].upper() == "R" and stripped.split()[-1:] == ["PEX_R"]:
            resistor_fields = stripped.split()[:-1]
            if float(resistor_fields[3]) == 0:
                resistor_fields[3] = f"{zero_resistance_floor_ohm:g}"
            output.append(" ".join(resistor_fields))
            continue
        if not stripped or stripped.startswith("*") or stripped[:1].upper() != "M":
            output.append(statement)
            continue
        fields = stripped.split()
        if len(fields) < 6:
            output.append(statement)
            continue
        name, drain, gate, source, bulk, model = fields[:6]
        parameters = {
            match.group(1).upper(): match.group(2)
            for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]*)=([^\s]+)", stripped)
        }
        length = parameters.get("L", "20n")
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", name)
        if dialect == "ngspice_osdi":
            output.append(
                f"NPEX_{safe_name} {drain} {gate} {source} {bulk} {model} "
                f"L={length} NFIN=1 NF=1"
            )
        else:
            output.append(
                f"MPEX_{safe_name} {drain} {gate} {source} {bulk} {model} "
                f"L={length} W=27n NFIN=1 NF=1"
            )
    return "\n".join(output).rstrip() + "\n"


def _referenced_models(raw_netlist: str) -> list[str]:
    return list(
        dict.fromkeys(
            match.group(1)
            for line in _fold_spice_continuations(raw_netlist)
            if (match := _MODEL_INSTANCE_RE.match(line))
        )
    )


def _write_model_card(
    source: Path,
    output: Path,
    *,
    raw_netlist: str,
    dialect: str,
) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"BSIM-CMG model card does not exist: {source}")
    if dialect == "hspice":
        shutil.copyfile(source, output)
        return output
    card = source.read_text()
    models = _referenced_models(raw_netlist)
    translated = [translate_asap7_model(card, model) for model in models]
    output.write_text("\n".join(translated))
    return output


def _write_resistance_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "name",
        "parent_net",
        "net_a",
        "net_b",
        "layer_a",
        "layer_b",
        "resistance_ohm",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _tool_version(
    command: list[str], *, environment: dict[str, str] | None = None
) -> str:
    try:
        completed = subprocess.run(
            command,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    lines = (completed.stdout + completed.stderr).strip().splitlines()
    return lines[0] if lines else "unknown"


def run_open_pex(
    gds: str | Path,
    schematic: str | Path,
    output_dir: str | Path,
    *,
    cell_name: str,
    klayout: str | Path | None = None,
    fastercap: str | Path | None = None,
    fastercap_library_dir: str | Path | None = None,
    fastercap_threads: int | None = None,
    asap7_standard_cell: bool = False,
    substrate_net: str | None = None,
    model_card: str | Path | None = None,
    dialect: str = "ngspice_osdi",
    capacitance_tolerance: float = 0.08,
    capacitance_mesh_area_um2: float = 0.1,
    capacitance_mesh_quality: float = 1.0,
    capacitance_refinement: float = 0.5,
    capacitance_threshold_farad: float = 0.0,
    resistance_mesh_area_um2: float = 0.0,
    zero_resistance_floor_ohm: float = 1e-3,
    timeout: float = 900.0,
) -> PEXResult:
    """Run stages 1-4 of the open flow and emit a post-layout subcircuit."""

    _require_kpex()
    if capacitance_mesh_area_um2 < 0 or resistance_mesh_area_um2 < 0:
        raise ValueError("mesh areas cannot be negative")

    import klayout.db as kdb
    from google.protobuf.json_format import MessageToJson
    from klayout_pex.common.capacitance_matrix import CapacitanceMatrix
    from klayout_pex.extraction_engine import ExtractionEngine
    from klayout_pex.fastercap.fastercap_input_builder import FasterCapInputBuilder
    from klayout_pex.fastercap.fastercap_runner import (
        fastercap_parse_capacitance_matrix,
    )
    from klayout_pex.klayout.lvsdb_extractor import KLayoutExtractionContext
    from klayout_pex.klayout.netlist_printer import NetlistPrinter
    from klayout_pex.pdk_config import PDKConfig
    from klayout_pex.rcx25.r.r_extractor import RExtractor
    from klayout_pex.tech_info import TechInfo
    from klayout_pex.version import __version__ as kpex_version
    from klayout_pex_protobuf.kpex.klayout.r_extractor_tech_pb2 import (
        RExtractorTech,
    )

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_pins = reference_subcircuit_pins(schematic, cell_name)
    lvs = run_lvs(
        gds,
        schematic,
        output / "lvs",
        cell_name=cell_name,
        klayout=klayout,
        asap7_standard_cell=asap7_standard_cell,
        timeout=timeout,
    )
    if not lvs.matched:
        raise RuntimeError(
            f"LVS mismatch; PEX stopped before parasitic merge: {lvs.log}"
        )

    technology_path = write_kpex_technology(output / "asap7_open_tech.pb.json")
    calibration_path = output / "calibration.json"
    calibration_path.write_text(json.dumps(calibration_manifest(), indent=2) + "\n")
    tech = TechInfo.from_json(str(technology_path), dielectric_filter=None)

    lvsdb = kdb.LayoutVsSchematic()
    lvsdb.read(str(lvs.report))
    context = KLayoutExtractionContext.prepare_extraction(
        lvsdb=lvsdb,
        top_cell=cell_name,
        tech=tech,
        blackbox_devices=False,
    )
    if context.unnamed_layers:
        names = [layer.lvs_layer_name for layer in context.unnamed_layers]
        raise RuntimeError(f"KPEX cannot map extracted LVS layers: {names}")

    cap_dir = output / "fastercap"
    cap_dir.mkdir(exist_ok=True)
    cap_builder = FasterCapInputBuilder(
        pex_context=context,
        tech_info=tech,
        k_void=3.5,
        delaunay_amax=capacitance_mesh_area_um2,
        delaunay_b=capacitance_mesh_quality,
    )
    cap_generator = cap_builder.build()
    cap_input = Path(
        cap_generator.write_fastcap(
            output_dir_path=str(cap_dir), prefix=f"{cell_name}_"
        )
    ).resolve()
    cap_log = cap_dir / "fastercap.log"
    fastercap_executable = find_fastercap(fastercap)
    fastercap_command = _run_fastercap(
        fastercap_executable,
        cap_input,
        cap_log,
        tolerance=capacitance_tolerance,
        d_coeff=1.0,
        refinement=capacitance_refinement,
        library_dir=fastercap_library_dir,
        threads=fastercap_threads,
        timeout=timeout,
    )
    raw_matrix: CapacitanceMatrix = fastercap_parse_capacitance_matrix(str(cap_log))
    raw_cap_csv = cap_dir / "capacitance_raw.csv"
    raw_matrix.write_csv(str(raw_cap_csv))
    cap_matrix = raw_matrix.averaged_off_diagonals()
    cap_csv = cap_dir / "capacitance_averaged.csv"
    cap_matrix.write_csv(str(cap_csv))

    square_counting = RExtractorTech.Algorithm.ALGORITHM_SQUARE_COUNTING
    resistance_extractor = RExtractor(
        pex_context=context,
        substrate_algorithm=square_counting,
        wire_algorithm=square_counting,
        delaunay_b=0.5,
        delaunay_amax=resistance_mesh_area_um2,
        via_merge_distance=0.0,
        skip_simplify=True,
    )
    resistance_request = resistance_extractor.prepare_request()
    resistance_result = resistance_extractor.extract(resistance_request)
    resistance_json = output / "resistance_networks.json"
    resistance_json.write_text(
        MessageToJson(resistance_result, preserving_proto_field_name=True) + "\n"
    )

    expanded = context.lvsdb.netlist().dup()
    resistance_records, reconnect_count = add_distributed_resistance(
        expanded,
        cell_name,
        resistance_request,
        resistance_result,
    )
    resistance_csv = output / "resistance.csv"
    _write_resistance_csv(resistance_csv, resistance_records)

    top_circuit = expanded.circuit_by_name(cell_name)
    top_net_names = {net.expanded_name() for net in top_circuit.each_net()}
    if substrate_net is None:
        for candidate in ("VSS", "B", "VSUBS"):
            if candidate in top_net_names:
                substrate_net = candidate
                break
    if substrate_net is None:
        raise ValueError(
            "cannot infer substrate net; pass substrate_net (normally VSS or B)"
        )
    if substrate_net not in top_net_names:
        raise ValueError(f"substrate net {substrate_net!r} is not a circuit net")

    cap_records, cap_warnings = add_maxwell_capacitance(
        expanded,
        cell_name,
        list(cap_matrix.conductor_names),
        [list(row) for row in cap_matrix.rows],
        substrate_net=substrate_net,
        threshold_farad=capacitance_threshold_farad,
    )

    raw_netlist = output / f"{cell_name}.klayout_pex.spice"
    pdk = PDKConfig(
        name="ASAP7-open-calibrated",
        pex_lvs_script_path=str(lvs.command[3]),
        tech_pb_json_path=str(technology_path),
    )
    NetlistPrinter(ExtractionEngine.K25D, pdk).write(expanded, raw_netlist)
    raw_text = raw_netlist.read_text()

    translated_model: Path | None = None
    model_include = ""
    if model_card is not None:
        translated_model = output / (
            "asap7_bsimcmg_osdi.pm" if dialect == "ngspice_osdi" else "asap7_bsimcmg.pm"
        )
        _write_model_card(
            Path(model_card).expanduser().resolve(),
            translated_model,
            raw_netlist=raw_text,
            dialect=dialect,
        )
        model_include = f'.include "{translated_model.name}"\n'

    post_layout = output / f"{cell_name}.post_layout.spice"
    normalized_netlist = normalize_unit_finfet_netlist(
        raw_text,
        dialect=dialect,
        zero_resistance_floor_ohm=zero_resistance_floor_ohm,
    )
    normalized_netlist = reorder_subcircuit_pins(
        normalized_netlist, cell_name, reference_pins
    )
    post_layout.write_text(model_include + normalized_netlist)

    solver_environment = _solver_environment(
        library_dir=fastercap_library_dir,
        threads=fastercap_threads,
    )
    capacitance_by_pair: dict[str, float] = {}
    for item in cap_records:
        pair = "|".join(sorted((item["net_a"], item["net_b"])))
        capacitance_by_pair[pair] = (
            capacitance_by_pair.get(pair, 0.0) + item["capacitance_farad"]
        )
    resistance_by_net: dict[str, float] = {}
    for item in resistance_records:
        parent_net = item["parent_net"]
        resistance_by_net[parent_net] = (
            resistance_by_net.get(parent_net, 0.0) + item["resistance_ohm"]
        )
    report_path = output / "pex_report.json"
    report = {
        "status": "research-grade_not_signoff",
        "cell_name": cell_name,
        "lvs_matched": lvs.matched,
        "dialect": dialect,
        "substrate_net": substrate_net,
        "counts": {
            "resistors": len(resistance_records),
            "capacitors": len(cap_records),
            "reconnected_device_terminals": reconnect_count,
            "fastercap_conductors": cap_matrix.dimension,
        },
        "resistance": {
            "total_ohm": sum(item["resistance_ohm"] for item in resistance_records),
            "maximum_element_ohm": max(
                (item["resistance_ohm"] for item in resistance_records), default=0.0
            ),
            "element_sum_by_parent_net_ohm": resistance_by_net,
        },
        "capacitance": {
            "total_lumped_farad": sum(
                item["capacitance_farad"] for item in cap_records
            ),
            "pair_totals_farad": capacitance_by_pair,
            "matrix_conductors": list(cap_matrix.conductor_names),
            "warnings": cap_warnings,
        },
        "settings": {
            "capacitance_tolerance": capacitance_tolerance,
            "capacitance_mesh_area_um2": capacitance_mesh_area_um2,
            "capacitance_mesh_quality": capacitance_mesh_quality,
            "capacitance_refinement": capacitance_refinement,
            "resistance_mesh_area_um2": resistance_mesh_area_um2,
            "zero_resistance_floor_ohm": zero_resistance_floor_ohm,
        },
        "tools": {
            "klayout": _tool_version([str(find_klayout(klayout)), "-v"]),
            "kpex": kpex_version,
            "fastercap": _tool_version(
                [str(fastercap_executable), "-bv"], environment=solver_environment
            ),
            "fastercap_command": list(fastercap_command),
        },
        "outputs": {
            "post_layout_netlist": str(post_layout),
            "raw_klayout_netlist": str(raw_netlist),
            "capacitance_csv": str(cap_csv),
            "resistance_csv": str(resistance_csv),
            "technology": str(technology_path),
            "calibration": str(calibration_path),
        },
        "calibration": calibration_manifest(),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    return PEXResult(
        cell_name=cell_name,
        lvs=lvs,
        post_layout_netlist=post_layout,
        raw_klayout_netlist=raw_netlist,
        report=report_path,
        technology=technology_path,
        calibration=calibration_path,
        fastercap_input=cap_input,
        fastercap_log=cap_log,
        capacitance_raw_csv=raw_cap_csv,
        capacitance_csv=cap_csv,
        resistance_csv=resistance_csv,
        resistance_json=resistance_json,
        translated_model_card=translated_model,
    )


def result_as_json(result: PEXResult) -> dict[str, Any]:
    """Return a JSON-safe view useful to the CLI."""

    values = asdict(result)
    values["lvs"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(result.lvs).items()
    }
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in values.items()
    }
