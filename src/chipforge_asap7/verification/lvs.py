"""KLayout FinFET LVS runner for the public ASAP7 layer map."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..devices.finfet import FinFETSpec
from ..devices.sense_amp import SENSE_AMP_PINS, SenseAmpSpec, sense_amp_transistors
from ..layout.grid import FIN_WIDTH

__all__ = [
    "LVSResult",
    "find_klayout",
    "lvs_deck_path",
    "normalize_asap7_cdl_reference",
    "render_finfet_lvs_schematic",
    "render_sense_amp_lvs_schematic",
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


def render_sense_amp_lvs_schematic(
    spec: SenseAmpSpec,
    *,
    cell_name: str | None = None,
) -> str:
    """Render the sense amplifier as one LVS MOS per physical fin.

    The layout deck extracts each ``FIN x GATE`` intersection independently,
    so every logical transistor is expanded to its requested integer fin
    count.  This is the same unit-fin representation used by
    :func:`render_finfet_lvs_schematic`.
    """

    name = cell_name or spec.cell_name
    instances: list[str] = []
    for device in sense_amp_transistors():
        device_spec = spec.device_spec(device.flavor)
        for fin in range(device_spec.fins):
            instances.append(
                f"M{device.name}_n{fin} {device.drain} {device.gate} "
                f"{device.source} {device.bulk} {device_spec.model} "
                f"L={device_spec.gate_length}n W={FIN_WIDTH}n"
            )
    body = "\n".join(instances)
    pins = " ".join(SENSE_AMP_PINS)
    return (
        "* ASAP7 sense amplifier LVS reference: one MOS per physical fin\n"
        f".SUBCKT {name} {pins}\n"
        f"{body}\n"
        f".ENDS {name}\n"
        ".END\n"
    )


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
            multiplier = int(float(parameters.get("m", "1")))
        except ValueError as error:
            raise ValueError(f"invalid ASAP7 MOS sizing in: {statement}") from error
        if fins < 1 or multiplier < 1:
            raise ValueError(f"NFIN and M must be positive in: {statement}")

        length = parameters.get("l", "20n")
        for copy_index in range(fins * multiplier):
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
    timeout: float = 300,
) -> LVSResult:
    """Run ASAP7 KLayout LVS and retain the LVSDB needed by KPEX.

    ``compare=False`` prepares connectivity and an LVSDB even when no trusted
    schematic is available.  Normal verification should keep comparison on.
    """

    gds_path = Path(gds).expanduser().resolve()
    schematic_path = Path(schematic).expanduser().resolve()
    if not gds_path.is_file():
        raise FileNotFoundError(f"layout does not exist: {gds_path}")
    if not schematic_path.is_file():
        raise FileNotFoundError(f"schematic does not exist: {schematic_path}")

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
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
        f"tie_standard_cell_bodies={'true' if asap7_standard_cell else 'false'}",
        "-rd",
        "run_mode=deep",
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
