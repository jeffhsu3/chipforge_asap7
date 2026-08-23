"""Correlation against the released ASAP7 INVxp33 collateral.

This is a bounded research correlation, not a sign-off equivalence claim.  It
compares independently published artifacts: the xACT3D extracted SPICE view,
the TT NLDM Liberty view, and a BSIM-CMG/OpenVAF transient of our open PEX
subcircuit.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from ..devices.spice import (
    compile_bsimcmg107,
    find_default_bsimcmg107_source,
)
from .pex import reference_subcircuit_pins

__all__ = [
    "BSIMTiming",
    "ValidationResult",
    "find_asap7_release",
    "parse_liberty_reference",
    "parse_xact_reference",
    "run_bsim_inverter_timing",
    "validate_released_inverter",
]

DEFAULT_REFERENCE_CELL = "INVxp33_ASAP7_75t_R"
DEFAULT_INPUT_SLEW_PS = 5.0
DEFAULT_OUTPUT_LOAD_FF = 0.36


@dataclass(frozen=True)
class BSIMTiming:
    """Measured 50%-to-50% propagation delays from ngspice."""

    rise_delay_ps: float
    fall_delay_ps: float
    input_slew_ps: float
    output_load_ff: float
    deck: Path
    log: Path


@dataclass(frozen=True)
class ValidationResult:
    """Result and report path for released-inverter correlation."""

    passed_research_bounds: bool
    checks: dict[str, bool]
    report: Path
    bsim_timing: BSIMTiming | None


def find_asap7_release(root: str | Path | None = None) -> Path:
    """Resolve a public ASAP7 release checkout."""

    configured = os.environ.get("ASAP7_ROOT")
    candidates: list[str | Path | None] = [
        root,
        configured,
        Path(__file__).resolve().parents[4] / "asap7",
        Path.home() / "iv4/repos/asap7",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / "asap7sc7p5t_28").is_dir():
            return path
    raise FileNotFoundError(
        "ASAP7 release not found; pass asap7_root or set ASAP7_ROOT"
    )


def _spice_number(value: str) -> float:
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        r"([a-zA-Z]+)?",
        value.strip(),
    )
    if match is None:
        raise ValueError(f"invalid SPICE number: {value!r}")
    base = float(match.group(1))
    suffix = (match.group(2) or "").lower()
    scales = {
        "": 1.0,
        "a": 1e-18,
        "f": 1e-15,
        "p": 1e-12,
        "n": 1e-9,
        "u": 1e-6,
        "m": 1e-3,
        "k": 1e3,
        "meg": 1e6,
        "g": 1e9,
        "t": 1e12,
    }
    try:
        return base * scales[suffix]
    except KeyError as error:
        raise ValueError(f"unsupported SPICE suffix in {value!r}") from error


def _fold_continuations(text: str) -> list[str]:
    statements: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("+") and statements:
            statements[-1] += " " + line.lstrip()[1:].strip()
        else:
            statements.append(line)
    return statements


def _xact_design_block(text: str, cell_name: str) -> str:
    marker = re.compile(rf"^\*\s*Design:\s*{re.escape(cell_name)}\s*$", re.MULTILINE)
    match = marker.search(text)
    if match is None:
        raise ValueError(f"xACT reference has no design {cell_name!r}")
    next_design = re.search(r"^\*\s*Design:\s*", text[match.end() :], re.MULTILINE)
    end = match.end() + next_design.start() if next_design else len(text)
    return text[match.start() : end]


def parse_xact_reference(path: str | Path, cell_name: str) -> dict[str, Any]:
    """Extract resistance and capacitance metrics from one xACT design block."""

    source = Path(path).expanduser().resolve()
    block = _xact_design_block(source.read_text(errors="replace"), cell_name)
    ground_caps: list[float] = []
    coupling_caps: list[float] = []
    resistors: list[float] = []
    resistors_by_layer: dict[str, list[float]] = {}
    for statement in _fold_continuations(block):
        fields = statement.split()
        if len(fields) < 4 or fields[0].startswith("*"):
            continue
        element = fields[0].lower()
        if element.startswith("cc_"):
            coupling_caps.append(_spice_number(fields[3]))
        elif element.startswith("c"):
            ground_caps.append(_spice_number(fields[3]))
        elif element.startswith("r"):
            value = _spice_number(fields[3])
            resistors.append(value)
            layer_match = re.search(r"\$layer=([^\s]+)", statement, re.IGNORECASE)
            layer = layer_match.group(1) if layer_match else "unknown"
            resistors_by_layer.setdefault(layer, []).append(value)

    return {
        "path": str(source),
        "cell_name": cell_name,
        "ground_capacitance_farad": sum(ground_caps),
        "coupling_capacitance_farad": sum(coupling_caps),
        "coupling_capacitor_count": len(coupling_caps),
        "resistor_count": len(resistors),
        "resistance_element_sum_ohm": sum(resistors),
        "maximum_resistor_ohm": max(resistors, default=0.0),
        "median_resistor_by_layer_ohm": {
            layer: median(values) for layer, values in resistors_by_layer.items()
        },
    }


def _extract_archive_text(archive: Path, seven_zip: str = "7z") -> str:
    executable = shutil.which(seven_zip) if not Path(seven_zip).is_file() else seven_zip
    if executable is None:
        raise FileNotFoundError("7z is required to read the released Liberty archive")
    completed = subprocess.run(
        [str(executable), "e", "-so", str(archive)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode:
        raise RuntimeError(f"7z could not read {archive}: {completed.stderr.strip()}")
    return completed.stdout


def _named_group(text: str, kind: str, name: str | None = None) -> str:
    name_pattern = r"[^)]*" if name is None else re.escape(name)
    pattern = re.compile(
        rf"\b{re.escape(kind)}\s*\(\s*{name_pattern}\s*\)\s*\{{",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None:
        label = f" {name}" if name else ""
        raise ValueError(f"Liberty group {kind}{label} was not found")
    depth = 1
    index = match.end()
    quoted = False
    escaped = False
    while index < len(text) and depth:
        character = text[index]
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            quoted = not quoted
        elif not quoted and character == "{":
            depth += 1
        elif not quoted and character == "}":
            depth -= 1
        index += 1
    if depth:
        raise ValueError(f"unterminated Liberty group {kind} {name or ''}")
    return text[match.start() : index]


def _quoted_numbers(value: str) -> list[float]:
    quoted = re.search(r'"([^"]+)"', value)
    if quoted is None:
        raise ValueError(f"Liberty numeric list is not quoted: {value!r}")
    return [float(cell.strip()) for cell in quoted.group(1).split(",")]


def _liberty_table(group: str, table_name: str) -> dict[str, Any]:
    table = _named_group(group, table_name)
    index_1 = re.search(r"index_1\s*\(([^;]+)\)\s*;", table, re.IGNORECASE)
    index_2 = re.search(r"index_2\s*\(([^;]+)\)\s*;", table, re.IGNORECASE)
    values = re.search(r"values\s*\((.*?)\)\s*;", table, re.IGNORECASE | re.DOTALL)
    if index_1 is None or index_2 is None or values is None:
        raise ValueError(f"incomplete Liberty {table_name} table")
    rows = [
        [float(cell.strip()) for cell in row.split(",")]
        for row in re.findall(r'"([^"]+)"', values.group(1))
    ]
    return {
        "index_1": _quoted_numbers(index_1.group(1)),
        "index_2": _quoted_numbers(index_2.group(1)),
        "values": rows,
    }


def _nearest_table_value(
    table: dict[str, Any], x: float, y: float
) -> tuple[float, float, float]:
    x_index = min(
        range(len(table["index_1"])), key=lambda i: abs(table["index_1"][i] - x)
    )
    y_index = min(
        range(len(table["index_2"])), key=lambda i: abs(table["index_2"][i] - y)
    )
    return (
        table["values"][x_index][y_index],
        table["index_1"][x_index],
        table["index_2"][y_index],
    )


def parse_liberty_reference(
    path: str | Path,
    cell_name: str,
    *,
    input_slew_ps: float = DEFAULT_INPUT_SLEW_PS,
    output_load_ff: float = DEFAULT_OUTPUT_LOAD_FF,
) -> dict[str, Any]:
    """Read INV input capacitance and nearest NLDM rise/fall delays."""

    source = Path(path).expanduser().resolve()
    text = (
        _extract_archive_text(source) if source.suffix == ".7z" else source.read_text()
    )
    cell = _named_group(text, "cell", cell_name)
    input_pin = _named_group(cell, "pin", "A")
    cap_match = re.search(r"\bcapacitance\s*:\s*([^;]+);", input_pin, re.IGNORECASE)
    if cap_match is None:
        raise ValueError(f"Liberty input pin A has no capacitance in {cell_name}")

    output_pin = _named_group(cell, "pin", "Y")
    timing = _named_group(output_pin, "timing")
    rise = _nearest_table_value(
        _liberty_table(timing, "cell_rise"), input_slew_ps, output_load_ff
    )
    fall = _nearest_table_value(
        _liberty_table(timing, "cell_fall"), input_slew_ps, output_load_ff
    )
    return {
        "path": str(source),
        "cell_name": cell_name,
        "input_capacitance_ff": float(cap_match.group(1)),
        "rise_delay_ps": rise[0],
        "fall_delay_ps": fall[0],
        "selected_input_slew_ps": rise[1],
        "selected_output_load_ff": rise[2],
    }


def _absolute_nested_includes(text: str, parent: Path) -> str:
    pattern = re.compile(r'^(\s*\.include\s+)["\']?([^"\'\s]+)["\']?', re.IGNORECASE)
    output: list[str] = []
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            include_path = Path(match.group(2))
            if not include_path.is_absolute():
                include_path = (parent / include_path).resolve()
            line = f'{match.group(1)}"{include_path}"'
        output.append(line)
    return "\n".join(output).rstrip() + "\n"


def _find_osdi(
    osdi: str | Path | None,
    output_dir: Path,
    bsimcmg_source: str | Path | None,
) -> Path:
    configured = os.environ.get("BSIMCMG107_OSDI")
    candidates: list[str | Path | None] = [
        osdi,
        configured,
        Path.cwd() / "build/sense_amp_spice/BSIMCMG107.osdi",
        Path.cwd() / "build/finfet_spice/nmos_fin_111/BSIMCMG107.osdi",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().is_file():
            return Path(candidate).expanduser().resolve()
    source = (
        Path(bsimcmg_source).expanduser().resolve()
        if bsimcmg_source
        else find_default_bsimcmg107_source()
    )
    if source is None:
        raise FileNotFoundError(
            "BSIM-CMG OSDI library not found; pass osdi or bsimcmg_source"
        )
    return compile_bsimcmg107(source, output_dir / "BSIMCMG107.osdi")


def run_bsim_inverter_timing(
    post_layout_netlist: str | Path,
    output_dir: str | Path,
    *,
    cell_name: str = DEFAULT_REFERENCE_CELL,
    osdi: str | Path | None = None,
    bsimcmg_source: str | Path | None = None,
    ngspice: str = "ngspice",
    input_slew_ps: float = DEFAULT_INPUT_SLEW_PS,
    output_load_ff: float = DEFAULT_OUTPUT_LOAD_FF,
    vdd: float = 0.7,
) -> BSIMTiming:
    """Measure post-layout INV rise/fall delays with BSIM-CMG 107."""

    if input_slew_ps <= 0 or output_load_ff <= 0 or vdd <= 0:
        raise ValueError("slew, load, and VDD must be positive")
    netlist = Path(post_layout_netlist).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    osdi_path = _find_osdi(osdi, output, bsimcmg_source)
    executable = shutil.which(ngspice) if not Path(ngspice).is_file() else ngspice
    if executable is None:
        raise FileNotFoundError(f"ngspice was not found: {ngspice!r}")

    pins = reference_subcircuit_pins(netlist, cell_name)
    node_by_pin = {"A": "a", "VDD": "vdd", "VSS": "0", "Y": "y"}
    try:
        instance_nodes = " ".join(node_by_pin[pin.upper()] for pin in pins)
    except KeyError as error:
        raise ValueError(f"unexpected inverter pin in {pins}") from error

    simulation_dut = output / "simulation_dut.spice"
    simulation_dut.write_text(
        _absolute_nested_includes(netlist.read_text(), netlist.parent)
    )
    # Liberty transition is 10%-90%; a linear PULSE reaches that span in 80%
    # of its full edge time.
    pulse_edge_ps = input_slew_ps / 0.8
    deck = output / "bsim_pex_timing.spice"
    deck.write_text(
        f"""* ASAP7 open-PEX INV timing correlation
.include "{simulation_dut}"
Vsup vdd 0 {vdd:g}
Vin a 0 PULSE(0 {vdd:g} 20p {pulse_edge_ps:g}p {pulse_edge_ps:g}p 50p 130p)
Cload y 0 {output_load_ff:g}f
Xdut {instance_nodes} {cell_name}
.options rshunt=1e15 reltol=1e-4
.temp 25
.control
pre_osdi {osdi_path}
tran 0.02p 150p
meas tran t_fall TRIG v(a) VAL={vdd / 2:g} RISE=1 TARG v(y) VAL={vdd / 2:g} FALL=1
meas tran t_rise TRIG v(a) VAL={vdd / 2:g} FALL=1 TARG v(y) VAL={vdd / 2:g} RISE=1
wrdata bsim_pex_waveform.dat v(a) v(y)
quit
.endc
.end
"""
    )
    completed = subprocess.run(
        [str(executable), "-b", deck.name],
        cwd=output,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    log = output / "bsim_pex_timing.log"
    log_text = completed.stdout + completed.stderr
    log.write_text(log_text)
    fall_match = re.search(r"t_fall\s*=\s*([+\-0-9.eE]+)", log_text, re.IGNORECASE)
    rise_match = re.search(r"t_rise\s*=\s*([+\-0-9.eE]+)", log_text, re.IGNORECASE)
    if completed.returncode or fall_match is None or rise_match is None:
        raise RuntimeError(f"ngspice BSIM-CMG timing failed; see {log}")
    return BSIMTiming(
        rise_delay_ps=float(rise_match.group(1)) * 1e12,
        fall_delay_ps=float(fall_match.group(1)) * 1e12,
        input_slew_ps=input_slew_ps,
        output_load_ff=output_load_ff,
        deck=deck,
        log=log,
    )


def _in_bound(value: float, reference: float, lower: float, upper: float) -> bool:
    return reference > 0 and lower <= value / reference <= upper


def validate_released_inverter(
    pex_report: str | Path,
    post_layout_netlist: str | Path,
    output_dir: str | Path,
    *,
    asap7_root: str | Path | None = None,
    cell_name: str = DEFAULT_REFERENCE_CELL,
    xact_netlist: str | Path | None = None,
    liberty: str | Path | None = None,
    osdi: str | Path | None = None,
    bsimcmg_source: str | Path | None = None,
    ngspice: str = "ngspice",
    run_bsim: bool = True,
    input_slew_ps: float = DEFAULT_INPUT_SLEW_PS,
    output_load_ff: float = DEFAULT_OUTPUT_LOAD_FF,
) -> ValidationResult:
    """Run stage 5 and write explicit correlation ratios/bounds."""

    release = find_asap7_release(asap7_root)
    library_root = release / "asap7sc7p5t_28"
    xact_path = (
        Path(xact_netlist).resolve()
        if xact_netlist
        else (library_root / "CDL/xAct3D_extracted/asap7sc7p5t_28_R.sp")
    )
    liberty_path = (
        Path(liberty).resolve()
        if liberty
        else (library_root / "LIB/NLDM/asap7sc7p5t_INVBUF_RVT_TT_nldm_220122.lib.7z")
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    open_pex = json.loads(Path(pex_report).read_text())
    xact = parse_xact_reference(xact_path, cell_name)
    liberty_metrics = parse_liberty_reference(
        liberty_path,
        cell_name,
        input_slew_ps=input_slew_ps,
        output_load_ff=output_load_ff,
    )
    timing = (
        run_bsim_inverter_timing(
            post_layout_netlist,
            output / "bsim",
            cell_name=cell_name,
            osdi=osdi,
            bsimcmg_source=bsimcmg_source,
            ngspice=ngspice,
            input_slew_ps=input_slew_ps,
            output_load_ff=output_load_ff,
        )
        if run_bsim
        else None
    )

    pair_caps = open_pex["capacitance"].get("pair_totals_farad", {})
    open_ay = pair_caps.get("A|Y", 0.0)
    open_input_external = sum(
        value for pair, value in pair_caps.items() if "A" in pair.split("|")
    )
    open_total_cap = open_pex["capacitance"]["total_lumped_farad"]
    xact_total_cap = (
        xact["ground_capacitance_farad"] + xact["coupling_capacitance_farad"]
    )
    open_max_r = open_pex["resistance"]["maximum_element_ohm"]
    liberty_input_f = liberty_metrics["input_capacitance_ff"] * 1e-15

    checks = {
        "lvs_matched": bool(open_pex.get("lvs_matched")),
        # Full public vertical/dielectric data is unavailable.  These broad
        # bounds are diagnostic guardrails, not process qualification limits.
        "a_y_coupling_within_0p33x_3x_xact": _in_bound(
            open_ay, xact["coupling_capacitance_farad"], 1 / 3, 3.0
        ),
        "external_input_cap_below_1p25x_liberty_effective": (
            0 < open_input_external <= 1.25 * liberty_input_f
        ),
        "total_external_cap_within_0p5x_2x_xact": _in_bound(
            open_total_cap, xact_total_cap, 0.5, 2.0
        ),
        "maximum_r_within_0p33x_3x_xact": _in_bound(
            open_max_r, xact["maximum_resistor_ohm"], 1 / 3, 3.0
        ),
    }
    if timing is not None:
        checks.update(
            {
                "bsim_rise_delay_within_0p5x_2x_liberty": _in_bound(
                    timing.rise_delay_ps,
                    liberty_metrics["rise_delay_ps"],
                    0.5,
                    2.0,
                ),
                "bsim_fall_delay_within_0p5x_2x_liberty": _in_bound(
                    timing.fall_delay_ps,
                    liberty_metrics["fall_delay_ps"],
                    0.5,
                    2.0,
                ),
            }
        )

    report_path = output / "reference_validation.json"
    report = {
        "status": "research-grade_not_signoff",
        "cell_name": cell_name,
        "passed_research_bounds": all(checks.values()),
        "checks": checks,
        "open_pex": {
            "report": str(Path(pex_report).resolve()),
            "a_y_coupling_farad": open_ay,
            "external_input_capacitance_farad": open_input_external,
            "total_lumped_capacitance_farad": open_total_cap,
            "maximum_resistor_ohm": open_max_r,
        },
        "released_xact": xact,
        "released_liberty": liberty_metrics,
        "bsim_cmg_openvaf": (
            {
                **asdict(timing),
                "deck": str(timing.deck),
                "log": str(timing.log),
            }
            if timing is not None
            else None
        ),
        "ratios": {
            "open_to_xact_a_y_coupling": (open_ay / xact["coupling_capacitance_farad"]),
            "open_external_to_liberty_input_capacitance": (
                open_input_external / liberty_input_f
            ),
            "open_to_xact_total_external_capacitance": (
                open_total_cap / xact_total_cap
            ),
            "open_to_xact_maximum_resistor": (
                open_max_r / xact["maximum_resistor_ohm"]
            ),
            "bsim_to_liberty_rise_delay": (
                timing.rise_delay_ps / liberty_metrics["rise_delay_ps"]
                if timing
                else None
            ),
            "bsim_to_liberty_fall_delay": (
                timing.fall_delay_ps / liberty_metrics["fall_delay_ps"]
                if timing
                else None
            ),
        },
        "limitations": [
            "The public release omits the sign-off LVS/xACT rule decks and complete material stack.",
            "Liberty input capacitance includes nonlinear BSIM-CMG device capacitance; FasterCap reports only external field capacitance.",
            "The timing comparison uses nearest-grid NLDM values and a linear 10%-90% input edge.",
            "Passing these bounds detects major topology/unit/calibration errors; it does not establish foundry accuracy.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return ValidationResult(
        passed_research_bounds=all(checks.values()),
        checks=checks,
        report=report_path,
        bsim_timing=timing,
    )
