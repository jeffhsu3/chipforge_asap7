"""DC characterization bench for one parametric ASAP7 FinFET.

SPICE cannot simulate GDS polygons directly.  This module creates the smallest
electrical fixture around :class:`~chipforge_asap7.devices.FinFETSpec`: one DUT,
bias sources, and (in a second deck) one load resistor.  It supports the ASAP7
BSIM-CMG 107 HSPICE card through ngspice's OSDI interface.

The resulting curves validate the compact model and the generated device
parameters.  They do *not* validate that a GDS extractor recognizes the same
D/G/S/B connectivity; that requires LVS/parasitic extraction first, after which
the extracted DUT can replace the generated instance in these same fixtures.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .finfet import FinFETSpec

__all__ = [
    "DCTestResult",
    "compile_bsimcmg107",
    "find_default_bsimcmg107_source",
    "find_default_model_card",
    "render_characterization_deck",
    "render_switch_deck",
    "run_dc_testbench",
    "translate_asap7_model",
]

DEFAULT_VDD = 0.7
DEFAULT_STEP = 0.01
DEFAULT_LOAD_OHMS = 100_000.0

_MODEL_RE = re.compile(r"^\s*\.model\s+(\S+)\s+(\S+)", re.IGNORECASE)
_VERSION_RE = re.compile(r"\bversion\s*=\s*\S+", re.IGNORECASE)
_MODEL_LENGTH_RE = re.compile(r"\bl\s*=\s*\S+", re.IGNORECASE)
_INSTANCE_PARAMETER_RE = re.compile(
    r"^(\s*)(parameter\s+(?:real|integer)\s+(?:NFIN|NF|L)\b)",
    re.MULTILINE,
)

# The Xyce copy of BSIM-CMG 107 carries declaration attributes that OpenVAF
# does not accept.  The equations stay in their original include files; this
# wrapper only omits those simulator-specific annotations.
_OPENVAF_WRAPPER = """\
// OpenVAF wrapper for BSIM-CMG 107.0.0 (equations loaded from source_dir).
`define __RDSMOD__
`define attr(txt)
`include "constants.vams"
`include "disciplines.vams"
`define __OPINFO__
`include "common_defs.include"
`include "bsimcmg_cfringe.include"

module bsimcmg(d, g, s, e);
    inout d, g, s, e;
    electrical d, g, s, e;
    electrical di, si;
    `include "bsimcmg107_openvaf_body.include"
endmodule
"""


@dataclass(frozen=True)
class DCTestResult:
    """Headline measurements from the characterization and switch decks."""

    model: str
    cell_name: str
    id_off_a: float
    id_on_a: float
    on_off_ratio: float
    output_off_v: float
    output_on_v: float
    current_increases: bool
    switch_moves_correctly: bool
    passed: bool


def translate_asap7_model(card: str, model_name: str) -> str:
    """Return one ASAP7 model block in ngspice/OpenVAF form.

    ASAP7's card declares a built-in HSPICE ``level=72`` MOS model.  ngspice
    loads the same BSIM-CMG 107 equations as an OSDI device named ``bsimcmg``;
    only that declaration and the HSPICE-only ``version`` selector differ.
    """

    lines = card.splitlines()
    start: int | None = None
    flavor: str | None = None
    for index, line in enumerate(lines):
        match = _MODEL_RE.match(line)
        if match and match.group(1).lower() == model_name.lower():
            start = index
            flavor = match.group(2).lower()
            break
    if start is None or flavor not in {"nmos", "pmos"}:
        raise ValueError(f"model {model_name!r} was not found as an NMOS/PMOS")

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if _MODEL_RE.match(lines[index]):
            end = index
            break

    devtype = 1 if flavor == "nmos" else 0
    output = [
        f"* {model_name}: ASAP7 BSIM-CMG 107 card adapted for ngspice OSDI",
        f".model {model_name} bsimcmg DEVTYPE={devtype}",
    ]
    for line in lines[start + 1 : end]:
        converted = _VERSION_RE.sub("", line)
        # L is an instance parameter in the generated DUT, so the model-level
        # default from the HSPICE card must not be repeated on .model.
        converted = _MODEL_LENGTH_RE.sub("", converted).rstrip()
        if converted.strip() == "+":
            continue
        output.append(converted)
    return "\n".join(output).rstrip() + "\n"


def compile_bsimcmg107(
    source_dir: Path,
    output: Path,
    *,
    openvaf: str = "openvaf",
) -> Path:
    """Compile the BSIM-CMG 107 Verilog-A includes into an OSDI library."""

    source_dir = source_dir.resolve()
    required = (
        "common_defs.include",
        "bsimcmg_cfringe.include",
        "bsimcmg_body.include",
    )
    missing = [name for name in required if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{source_dir} is not a BSIM-CMG 107 source directory; missing {missing}"
        )

    compiler = shutil.which(openvaf) if not Path(openvaf).is_file() else openvaf
    if compiler is None:
        raise FileNotFoundError(
            "openvaf was not found; pass --openvaf or provide a precompiled --osdi"
        )

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = output.with_name("bsimcmg107_openvaf.va")
    wrapper.write_text(_OPENVAF_WRAPPER)
    body = (source_dir / "bsimcmg_body.include").read_text()
    body = _INSTANCE_PARAMETER_RE.sub(r'\1(* type="instance" *) \2', body)
    # The ASAP7 card puts EOTACC exactly on BSIM-CMG's inclusive 0.1 nm
    # boundary.  OpenVAF/OSDI's floating-point bound check rejects that equal
    # value, so retain the card value and relax only the declaration bound.
    body = body.replace(
        "parameter real EOTACC    =  EOT from [0.1n:inf)",
        "parameter real EOTACC    =  EOT from (0:inf)",
    )
    output.with_name("bsimcmg107_openvaf_body.include").write_text(body)
    completed = subprocess.run(
        [str(compiler), "-I", str(source_dir), str(wrapper), "-o", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    (output.parent / "openvaf.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode or not output.is_file():
        raise RuntimeError(
            f"OpenVAF failed (exit {completed.returncode}); see "
            f"{output.parent / 'openvaf.log'}"
        )
    return output


def _spice_path(path: Path) -> str:
    value = str(path.resolve())
    if '"' in value:
        raise ValueError(f"SPICE paths cannot contain a double quote: {value}")
    return f'"{value}"'


def _osdi_command_path(path: Path) -> str:
    value = str(path.resolve())
    if any(character.isspace() for character in value):
        raise ValueError(f"ngspice pre_osdi paths cannot contain whitespace: {value}")
    return value


def _osdi_instances(spec: FinFETSpec, source: str, bulk: str) -> str:
    # BSIM-CMG 107 has NF and NFIN but no portable multiplicity parameter.
    # Separate parallel instances exactly represent multiplier rows.
    return "\n".join(
        f"Ndut{index} d g {source} {bulk} {spec.model} "
        f"NFIN={spec.fins} L={spec.gate_length}n NF={spec.fingers}"
        for index in range(spec.multipliers)
    )


def render_characterization_deck(
    spec: FinFETSpec,
    *,
    model_card: Path,
    osdi: Path,
    vdd: float = DEFAULT_VDD,
    step: float = DEFAULT_STEP,
) -> str:
    """Render Id-Vg and Id-Vd sweeps for a single FinFET DUT."""

    if vdd <= 0 or step <= 0 or step > vdd:
        raise ValueError("vdd and step must be positive, with step <= vdd")
    is_nmos = spec.flavor == "n"
    source_bias = 0.0 if is_nmos else vdd
    gate_on = vdd if is_nmos else 0.0
    drain_on = vdd if is_nmos else 0.0
    gate_sweep = f"0 {vdd:g} {step:g}" if is_nmos else f"{vdd:g} 0 {-step:g}"
    drain_sweep = f"0 {vdd:g} {step:g}" if is_nmos else f"{vdd:g} 0 {-step:g}"
    drain_current = "-i(Vdrain)" if is_nmos else "i(Vdrain)"
    instances = _osdi_instances(spec, "s", "b")

    return f"""\
* ASAP7 {spec.cell_name}: compact-model DC characterization
* Scope: generated BSIM-CMG parameters, not GDS extraction/LVS.
.include {_spice_path(model_card)}

Vdrain d 0 {drain_on:g}
Vgate g 0 {gate_on:g}
Vsource s 0 {source_bias:g}
Vbulk b 0 {source_bias:g}
{instances}

.temp 25
.control
pre_osdi {_osdi_command_path(osdi)}
set wr_singlescale
set wr_vecnames
dc Vgate {gate_sweep}
let drain_current_a = {drain_current}
wrdata transfer.dat drain_current_a
dc Vdrain {drain_sweep}
let drain_current_a = {drain_current}
wrdata output.dat drain_current_a
quit
.endc
.end
"""


def render_switch_deck(
    spec: FinFETSpec,
    *,
    model_card: Path,
    osdi: Path,
    vdd: float = DEFAULT_VDD,
    step: float = DEFAULT_STEP,
    load_ohms: float = DEFAULT_LOAD_OHMS,
) -> str:
    """Render a resistor-loaded DC switch sweep using one transistor."""

    if vdd <= 0 or step <= 0 or step > vdd or load_ohms <= 0:
        raise ValueError("vdd, step, and load_ohms must be positive; step <= vdd")
    is_nmos = spec.flavor == "n"
    gate_off = 0.0 if is_nmos else vdd
    gate_sweep = f"0 {vdd:g} {step:g}" if is_nmos else f"{vdd:g} 0 {-step:g}"
    source = "0" if is_nmos else "vdd"
    bulk = source
    load = f"Rload vdd d {load_ohms:g}" if is_nmos else f"Rload d 0 {load_ohms:g}"
    instances = _osdi_instances(spec, source, bulk)

    return f"""\
* ASAP7 {spec.cell_name}: one-transistor resistor-loaded switch
* NMOS pulls down; PMOS pulls up.  V(output) must move when the gate turns on.
.include {_spice_path(model_card)}

Vsupply vdd 0 {vdd:g}
Vgate g 0 {gate_off:g}
{load}
{instances}

.temp 25
.control
pre_osdi {_osdi_command_path(osdi)}
set wr_singlescale
set wr_vecnames
dc Vgate {gate_sweep}
let supply_current_a = -i(Vsupply)
wrdata switch.dat v(d) supply_current_a
quit
.endc
.end
"""


def _read_wrdata(path: Path) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    for line in path.read_text().splitlines():
        fields = line.split()
        try:
            values = tuple(float(field) for field in fields)
        except ValueError:
            continue
        if values:
            rows.append(values)
    if not rows:
        raise RuntimeError(f"ngspice produced no numeric data in {path}")
    return rows


def _run_ngspice(deck: Path, executable: str, expected_outputs: Sequence[Path]) -> None:
    simulator = (
        shutil.which(executable) if not Path(executable).is_file() else executable
    )
    if simulator is None:
        raise FileNotFoundError(f"ngspice was not found: {executable!r}")
    # Never accept curves left by an older successful run if this run aborts.
    for output in expected_outputs:
        output.unlink(missing_ok=True)
    completed = subprocess.run(
        [str(simulator), "-b", deck.name],
        cwd=deck.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    log = deck.with_suffix(".log")
    log.write_text(completed.stdout + completed.stderr)
    missing = [output.name for output in expected_outputs if not output.is_file()]
    if completed.returncode or missing:
        raise RuntimeError(
            f"ngspice failed for {deck.name} (exit {completed.returncode}, "
            f"missing outputs: {missing}); see {log}"
        )


def run_dc_testbench(
    spec: FinFETSpec,
    *,
    model_card: Path,
    osdi: Path,
    output_dir: Path,
    ngspice: str = "ngspice",
    vdd: float = DEFAULT_VDD,
    step: float = DEFAULT_STEP,
    load_ohms: float = DEFAULT_LOAD_OHMS,
) -> DCTestResult:
    """Generate, run, and check the two single-device DC fixtures."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    translated = output_dir / f"{spec.model}.pm"
    translated.write_text(translate_asap7_model(model_card.read_text(), spec.model))

    characterization = output_dir / "characterization.sp"
    switch = output_dir / "switch.sp"
    characterization.write_text(
        render_characterization_deck(
            spec, model_card=translated, osdi=osdi, vdd=vdd, step=step
        )
    )
    switch.write_text(
        render_switch_deck(
            spec,
            model_card=translated,
            osdi=osdi,
            vdd=vdd,
            step=step,
            load_ohms=load_ohms,
        )
    )

    _run_ngspice(
        characterization,
        ngspice,
        (output_dir / "transfer.dat", output_dir / "output.dat"),
    )
    _run_ngspice(switch, ngspice, (output_dir / "switch.dat",))

    transfer = _read_wrdata(output_dir / "transfer.dat")
    switch_data = _read_wrdata(output_dir / "switch.dat")
    id_off = abs(transfer[0][-1])
    id_on = abs(transfer[-1][-1])
    output_off = switch_data[0][-2]
    output_on = switch_data[-1][-2]
    ratio = id_on / max(id_off, 1e-30)
    current_increases = id_on > id_off and ratio >= 100.0
    if spec.flavor == "n":
        switch_moves = output_off >= 0.9 * vdd and output_on <= 0.5 * vdd
    else:
        switch_moves = output_off <= 0.1 * vdd and output_on >= 0.5 * vdd
    result = DCTestResult(
        model=spec.model,
        cell_name=spec.cell_name,
        id_off_a=id_off,
        id_on_a=id_on,
        on_off_ratio=ratio,
        output_off_v=output_off,
        output_on_v=output_on,
        current_increases=current_increases,
        switch_moves_correctly=switch_moves,
        passed=current_increases and switch_moves,
    )
    (output_dir / "results.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n"
    )
    return result


def _repo_sibling(relative: str) -> Path:
    return Path(__file__).resolve().parents[3].parent / relative


def find_default_model_card() -> Path | None:
    """Find the ASAP7 TT model card from the environment or sibling checkout."""

    configured = os.environ.get("ASAP7_MODEL_CARD")
    candidates = [
        Path(configured).expanduser() if configured else None,
        _repo_sibling("asap7/asap7_pdk_r1p7/models/hspice/7nm_TT_160803.pm"),
    ]
    return next((path for path in candidates if path and path.is_file()), None)


def find_default_bsimcmg107_source() -> Path | None:
    """Find BSIM-CMG 107 Verilog-A from the environment or sibling checkout."""

    configured = os.environ.get("BSIMCMG107_SOURCE")
    candidates = [
        Path(configured).expanduser() if configured else None,
        _repo_sibling("Xyce/utils/ADMS/examples/bsimcmg_107.0.0/code"),
    ]
    return next((path for path in candidates if path and path.is_dir()), None)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: generate and run a single-FinFET DC testbench."""

    parser = argparse.ArgumentParser(
        description="Characterize one ASAP7 BSIM-CMG FinFET with ngspice."
    )
    parser.add_argument("--flavor", choices=("n", "p"), default="n")
    parser.add_argument("--vt", choices=("rvt", "lvt", "slvt", "sram"), default="rvt")
    parser.add_argument("--fins", type=int, default=1)
    parser.add_argument("--fingers", type=int, default=1)
    parser.add_argument("--multipliers", type=int, default=1)
    parser.add_argument("--vdd", type=float, default=DEFAULT_VDD)
    parser.add_argument("--step", type=float, default=DEFAULT_STEP)
    parser.add_argument("--load-ohms", type=float, default=DEFAULT_LOAD_OHMS)
    parser.add_argument("--model-card", type=Path)
    parser.add_argument("--osdi", type=Path)
    parser.add_argument("--bsimcmg-source", type=Path)
    parser.add_argument("--openvaf", default="openvaf")
    parser.add_argument("--ngspice", default="ngspice")
    parser.add_argument("--out", type=Path, default=Path("build/finfet_spice"))
    args = parser.parse_args(argv)

    spec = FinFETSpec(
        flavor=args.flavor,
        vt=args.vt,
        fins=args.fins,
        fingers=args.fingers,
        multipliers=args.multipliers,
    )
    model_card = args.model_card or find_default_model_card()
    if model_card is None or not model_card.is_file():
        parser.error(
            "ASAP7 model card not found; pass --model-card or ASAP7_MODEL_CARD"
        )

    output_dir = args.out / spec.cell_name
    osdi = args.osdi
    if osdi is None:
        source = args.bsimcmg_source or find_default_bsimcmg107_source()
        if source is None:
            parser.error(
                "BSIM-CMG 107 source not found; pass --bsimcmg-source or --osdi"
            )
        osdi = compile_bsimcmg107(
            source, output_dir / "BSIMCMG107.osdi", openvaf=args.openvaf
        )
    elif not osdi.is_file():
        parser.error(f"OSDI library does not exist: {osdi}")

    result = run_dc_testbench(
        spec,
        model_card=model_card,
        osdi=osdi,
        output_dir=output_dir,
        ngspice=args.ngspice,
        vdd=args.vdd,
        step=args.step,
        load_ohms=args.load_ohms,
    )
    print(json.dumps(asdict(result), indent=2))
    print(f"artifacts: {output_dir.resolve()}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
