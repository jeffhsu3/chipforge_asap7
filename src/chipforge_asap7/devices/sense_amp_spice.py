"""Transient BSIM-CMG testbench for OpenFinRAM's ASAP7 sense amplifier.

This is the transistor-level ``sense_amp_sram`` used by OpenFinRAM, not a
behavioral approximation.  Its inputs ``SA``/``SAN`` receive the differential
bitline voltage, active-low ``SAPRECHN`` precharges both outputs, and ``SAE``
starts regenerative evaluation.  The bench runs both input polarities so a
hard-wired or one-sided latch cannot pass.

As with the single-device bench in :mod:`chipforge_asap7.devices.spice`, this
validates the compact-model circuit.  Testing a physical sense-amplifier GDS
requires extraction/LVS and then replacing the DUT subcircuit in these decks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .finfet import FinFETSpec
from .sense_amp import (
    DEFAULT_N_FINS,
    DEFAULT_P_FINS,
    sense_amp_transistors,
)
from .spice import (
    compile_bsimcmg107,
    find_default_bsimcmg107_source,
    find_default_model_card,
    translate_asap7_model,
)

__all__ = [
    "SenseAmpCaseResult",
    "SenseAmpTestResult",
    "render_openfinram_sense_amp",
    "render_sense_amp_deck",
    "run_sense_amp_testbench",
]

DEFAULT_VDD = 0.7
DEFAULT_DIFFERENTIAL_V = 0.1
DEFAULT_OUTPUT_LOAD_FF = 1.0
DEFAULT_PRECHARGE_RELEASE_NS = 0.5
DEFAULT_ENABLE_NS = 0.52
DEFAULT_STOP_NS = 1.5
DEFAULT_STEP_PS = 0.5
CONTROL_RISE_NS = 0.005


@dataclass(frozen=True)
class SenseAmpCaseResult:
    """Measured result for one polarity of differential input."""

    case: str
    sa_v: float
    san_v: float
    expected_high_output: str
    qa_precharge_v: float
    qan_precharge_v: float
    qa_final_v: float
    qan_final_v: float
    output_differential_v: float
    resolution_delay_s: float | None
    peak_supply_current_a: float
    evaluation_energy_j: float
    passed: bool


@dataclass(frozen=True)
class SenseAmpTestResult:
    """Results for the two complementary sense-amplifier decisions."""

    topology: str
    nmos_model: str
    pmos_model: str
    vdd_v: float
    input_differential_v: float
    n_fins: int
    p_fins: int
    sa_high: SenseAmpCaseResult
    san_high: SenseAmpCaseResult
    passed: bool


def _validate_parameters(
    *,
    vdd: float,
    differential_v: float,
    n_fins: int,
    p_fins: int,
    output_load_ff: float,
    precharge_release_ns: float,
    enable_ns: float,
    stop_ns: float,
    step_ps: float,
) -> None:
    if vdd <= 0:
        raise ValueError("vdd must be positive")
    if differential_v <= 0 or differential_v >= vdd:
        raise ValueError("differential_v must be positive and less than vdd")
    for name, value in (("n_fins", n_fins), ("p_fins", p_fins)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if output_load_ff < 0:
        raise ValueError("output_load_ff must be non-negative")
    if precharge_release_ns <= CONTROL_RISE_NS:
        raise ValueError("precharge_release_ns must allow the 5 ps control rise")
    if enable_ns - CONTROL_RISE_NS < precharge_release_ns:
        raise ValueError("SAE must begin rising after SAPRECHN finishes rising")
    if stop_ns <= enable_ns:
        raise ValueError("stop_ns must be later than enable_ns")
    if step_ps <= 0 or step_ps * 1e-3 >= stop_ns:
        raise ValueError("step_ps must be positive and shorter than stop_ns")


def render_openfinram_sense_amp(
    *,
    nmos_model: str = "nmos_rvt",
    pmos_model: str = "pmos_rvt",
    n_fins: int = DEFAULT_N_FINS,
    p_fins: int = DEFAULT_P_FINS,
) -> str:
    """Render OpenFinRAM's 16-transistor ``sense_amp_sram`` subcircuit.

    Node order and device sizing match ``scripts/characterize_read.py`` in the
    OpenFinRAM checkout.  ``N`` is ngspice's OSDI device prefix; widths are
    quantized with ``NFIN`` and therefore have no planar ``W`` parameter.
    """

    if not nmos_model or not pmos_model:
        raise ValueError("nmos_model and pmos_model must be non-empty")
    # FinFETSpec owns the legal ASAP7 fin range.
    FinFETSpec(flavor="n", fins=n_fins)
    FinFETSpec(flavor="p", fins=p_fins)
    lines = [".subckt sense_amp_sram sa san SAE SAPRECHN qa qan vdd vss"]
    for device in sense_amp_transistors():
        model = nmos_model if device.flavor == "n" else pmos_model
        fins = n_fins if device.flavor == "n" else p_fins
        lines.append(
            f"N{device.name} {device.drain} {device.gate} {device.source} "
            f"{device.bulk} {model} L=20n NFIN={fins}"
        )
    lines.append(".ends sense_amp_sram")
    return "\n".join(lines) + "\n"


def _spice_include_path(path: Path) -> str:
    value = str(path.resolve())
    if '"' in value:
        raise ValueError(f"SPICE include paths cannot contain quotes: {value}")
    return f'"{value}"'


def _osdi_path(path: Path) -> str:
    value = str(path.resolve())
    if any(character.isspace() for character in value):
        raise ValueError(f"ngspice pre_osdi paths cannot contain whitespace: {value}")
    return value


def render_sense_amp_deck(
    *,
    sa_high: bool,
    model_card: Path,
    osdi: Path,
    vdd: float = DEFAULT_VDD,
    differential_v: float = DEFAULT_DIFFERENTIAL_V,
    n_fins: int = DEFAULT_N_FINS,
    p_fins: int = DEFAULT_P_FINS,
    output_load_ff: float = DEFAULT_OUTPUT_LOAD_FF,
    precharge_release_ns: float = DEFAULT_PRECHARGE_RELEASE_NS,
    enable_ns: float = DEFAULT_ENABLE_NS,
    stop_ns: float = DEFAULT_STOP_NS,
    step_ps: float = DEFAULT_STEP_PS,
) -> str:
    """Render one transient decision with either ``SA`` or ``SAN`` higher."""

    _validate_parameters(
        vdd=vdd,
        differential_v=differential_v,
        n_fins=n_fins,
        p_fins=p_fins,
        output_load_ff=output_load_ff,
        precharge_release_ns=precharge_release_ns,
        enable_ns=enable_ns,
        stop_ns=stop_ns,
        step_ps=step_ps,
    )
    high, low = vdd, vdd - differential_v
    sa_v, san_v = (high, low) if sa_high else (low, high)
    case = "sa_high" if sa_high else "san_high"
    rise_ns = CONTROL_RISE_NS
    precharge_before_rise = precharge_release_ns - rise_ns
    enable_before_rise = enable_ns - rise_ns
    subckt = render_openfinram_sense_amp(
        n_fins=n_fins,
        p_fins=p_fins,
    )

    return f"""\
* OpenFinRAM sense_amp_sram: {case}, ASAP7 BSIM-CMG 107
* Expected polarity: SA > SAN -> QA high; SAN > SA -> QAN high.
.include {_spice_include_path(model_card)}

{subckt}
Vsupply vdd 0 {vdd:g}
Vsa sa 0 {sa_v:g}
Vsan san 0 {san_v:g}
Vsaprechn SAPRECHN 0 PWL(0 0 {precharge_before_rise:g}n 0 {precharge_release_ns:g}n {vdd:g} {stop_ns:g}n {vdd:g})
Vsae SAE 0 PWL(0 0 {enable_before_rise:g}n 0 {enable_ns:g}n {vdd:g} {stop_ns:g}n {vdd:g})
XSA sa san SAE SAPRECHN qa qan vdd 0 sense_amp_sram
Cqa qa 0 {output_load_ff:g}f
Cqan qan 0 {output_load_ff:g}f

.options abstol=1e-15 reltol=1e-4
.temp 25
.control
pre_osdi {_osdi_path(osdi)}
set wr_singlescale
set wr_vecnames
tran {step_ps:g}p {stop_ns:g}n uic
let supply_current_a = -i(Vsupply)
wrdata {case}.dat v(sa) v(san) v(SAPRECHN) v(SAE) v(qa) v(qan) supply_current_a
write {case}.raw
quit
.endc
.end
"""


def _read_data(path: Path) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    for line in path.read_text().splitlines():
        try:
            values = tuple(float(field) for field in line.split())
        except ValueError:
            continue
        if values:
            rows.append(values)
    if not rows:
        raise RuntimeError(f"ngspice produced no numeric data in {path}")
    return rows


def _crossing_time(
    samples: Sequence[tuple[float, float]],
    threshold: float,
    *,
    start_s: float = 0.0,
) -> float | None:
    previous: tuple[float, float] | None = None
    for current in samples:
        if current[0] < start_s:
            previous = current
            continue
        if previous is not None and previous[1] < threshold <= current[1]:
            t0, value0 = previous
            t1, value1 = current
            if value1 == value0:
                return t1
            fraction = (threshold - value0) / (value1 - value0)
            return t0 + fraction * (t1 - t0)
        previous = current
    return None


def _integrate_energy(rows: Sequence[tuple[float, ...]], start_s: float) -> float:
    energy = 0.0
    previous: tuple[float, float] | None = None
    for row in rows:
        time_s, current_a = row[0], max(row[7], 0.0)
        if time_s < start_s:
            continue
        if previous is not None:
            energy += 0.5 * (previous[1] + current_a) * (time_s - previous[0])
        previous = (time_s, current_a)
    return energy


def _analyze_case(
    rows: Sequence[tuple[float, ...]],
    *,
    sa_high: bool,
    vdd: float,
    precharge_release_ns: float,
) -> SenseAmpCaseResult:
    if any(len(row) < 8 for row in rows):
        raise RuntimeError("sense-amplifier data has fewer than eight columns")
    final = rows[-1]
    sa_v, san_v, qa, qan = final[1], final[2], final[5], final[6]
    precharge_probe_s = (precharge_release_ns - CONTROL_RISE_NS) * 1e-9
    precharge_row = max(
        (row for row in rows if row[0] <= precharge_probe_s),
        key=lambda row: row[0],
        default=rows[0],
    )
    qa_precharge, qan_precharge = precharge_row[5], precharge_row[6]
    expected_high_output = "qa" if sa_high else "qan"
    high_output, low_output = (qa, qan) if sa_high else (qan, qa)
    signed_differential = qa - qan if sa_high else qan - qa
    enable_samples = [(row[0], row[4]) for row in rows]
    decision_samples = [
        (row[0], row[5] - row[6] if sa_high else row[6] - row[5]) for row in rows
    ]
    enable_time = _crossing_time(enable_samples, 0.5 * vdd)
    resolved_time = _crossing_time(
        decision_samples,
        0.8 * vdd,
        start_s=enable_time or 0.0,
    )
    resolution_delay = (
        resolved_time - enable_time
        if enable_time is not None and resolved_time is not None
        else None
    )
    evaluation_start = precharge_release_ns * 1e-9
    peak_current = max(
        (max(row[7], 0.0) for row in rows if row[0] >= evaluation_start),
        default=0.0,
    )
    passed = (
        qa_precharge >= 0.8 * vdd
        and qan_precharge >= 0.8 * vdd
        and high_output >= 0.8 * vdd
        and low_output <= 0.2 * vdd
        and signed_differential >= 0.8 * vdd
        and resolution_delay is not None
    )
    return SenseAmpCaseResult(
        case="sa_high" if sa_high else "san_high",
        sa_v=sa_v,
        san_v=san_v,
        expected_high_output=expected_high_output,
        qa_precharge_v=qa_precharge,
        qan_precharge_v=qan_precharge,
        qa_final_v=qa,
        qan_final_v=qan,
        output_differential_v=qa - qan,
        resolution_delay_s=resolution_delay,
        peak_supply_current_a=peak_current,
        evaluation_energy_j=vdd * _integrate_energy(rows, evaluation_start),
        passed=passed,
    )


def _run_ngspice(
    deck: Path,
    data: Path,
    executable: str,
    *,
    expected_stop_s: float,
) -> None:
    simulator = (
        shutil.which(executable) if not Path(executable).is_file() else executable
    )
    if simulator is None:
        raise FileNotFoundError(f"ngspice was not found: {executable!r}")
    data.unlink(missing_ok=True)
    completed = subprocess.run(
        [str(simulator), "-b", deck.name],
        cwd=deck.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    log = deck.with_suffix(".log")
    log_text = completed.stdout + completed.stderr
    log.write_text(log_text)
    reached_stop = False
    if data.is_file():
        rows = _read_data(data)
        reached_stop = rows[-1][0] >= 0.99 * expected_stop_s
    if completed.returncode or "simulation(s) aborted" in log_text or not reached_stop:
        raise RuntimeError(
            f"ngspice failed for {deck.name} (exit {completed.returncode}); see {log}"
        )


def run_sense_amp_testbench(
    *,
    model_card: Path,
    osdi: Path,
    output_dir: Path,
    ngspice: str = "ngspice",
    vdd: float = DEFAULT_VDD,
    differential_v: float = DEFAULT_DIFFERENTIAL_V,
    n_fins: int = DEFAULT_N_FINS,
    p_fins: int = DEFAULT_P_FINS,
    output_load_ff: float = DEFAULT_OUTPUT_LOAD_FF,
    precharge_release_ns: float = DEFAULT_PRECHARGE_RELEASE_NS,
    enable_ns: float = DEFAULT_ENABLE_NS,
    stop_ns: float = DEFAULT_STOP_NS,
    step_ps: float = DEFAULT_STEP_PS,
) -> SenseAmpTestResult:
    """Generate, run, and check both differential input decisions."""

    _validate_parameters(
        vdd=vdd,
        differential_v=differential_v,
        n_fins=n_fins,
        p_fins=p_fins,
        output_load_ff=output_load_ff,
        precharge_release_ns=precharge_release_ns,
        enable_ns=enable_ns,
        stop_ns=stop_ns,
        step_ps=step_ps,
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    translated = output_dir / "asap7_rvt_bsimcmg107.pm"
    source_card = model_card.read_text()
    translated.write_text(
        translate_asap7_model(source_card, "nmos_rvt")
        + "\n"
        + translate_asap7_model(source_card, "pmos_rvt")
    )

    case_results: dict[bool, SenseAmpCaseResult] = {}
    for sa_high in (True, False):
        case = "sa_high" if sa_high else "san_high"
        deck = output_dir / f"{case}.sp"
        data = output_dir / f"{case}.dat"
        deck.write_text(
            render_sense_amp_deck(
                sa_high=sa_high,
                model_card=translated,
                osdi=osdi,
                vdd=vdd,
                differential_v=differential_v,
                n_fins=n_fins,
                p_fins=p_fins,
                output_load_ff=output_load_ff,
                precharge_release_ns=precharge_release_ns,
                enable_ns=enable_ns,
                stop_ns=stop_ns,
                step_ps=step_ps,
            )
        )
        _run_ngspice(
            deck,
            data,
            ngspice,
            expected_stop_s=stop_ns * 1e-9,
        )
        case_results[sa_high] = _analyze_case(
            _read_data(data),
            sa_high=sa_high,
            vdd=vdd,
            precharge_release_ns=precharge_release_ns,
        )

    result = SenseAmpTestResult(
        topology="OpenFinRAM sense_amp_sram (16 transistor)",
        nmos_model="nmos_rvt",
        pmos_model="pmos_rvt",
        vdd_v=vdd,
        input_differential_v=differential_v,
        n_fins=n_fins,
        p_fins=p_fins,
        sa_high=case_results[True],
        san_high=case_results[False],
        passed=case_results[True].passed and case_results[False].passed,
    )
    (output_dir / "results.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the BSIM-CMG sense-amplifier testbench."""

    parser = argparse.ArgumentParser(
        description="Run OpenFinRAM's ASAP7 sense amplifier in ngspice."
    )
    parser.add_argument("--vdd", type=float, default=DEFAULT_VDD)
    parser.add_argument("--differential-v", type=float, default=DEFAULT_DIFFERENTIAL_V)
    parser.add_argument("--n-fins", type=int, default=DEFAULT_N_FINS)
    parser.add_argument("--p-fins", type=int, default=DEFAULT_P_FINS)
    parser.add_argument("--output-load-ff", type=float, default=DEFAULT_OUTPUT_LOAD_FF)
    parser.add_argument(
        "--precharge-release-ns", type=float, default=DEFAULT_PRECHARGE_RELEASE_NS
    )
    parser.add_argument("--enable-ns", type=float, default=DEFAULT_ENABLE_NS)
    parser.add_argument("--stop-ns", type=float, default=DEFAULT_STOP_NS)
    parser.add_argument("--step-ps", type=float, default=DEFAULT_STEP_PS)
    parser.add_argument("--model-card", type=Path)
    parser.add_argument("--osdi", type=Path)
    parser.add_argument("--bsimcmg-source", type=Path)
    parser.add_argument("--openvaf", default="openvaf")
    parser.add_argument("--ngspice", default="ngspice")
    parser.add_argument("--out", type=Path, default=Path("build/sense_amp_spice"))
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also draw the run to a PNG beside results.json (needs the plot extra).",
    )
    args = parser.parse_args(argv)

    model_card = args.model_card or find_default_model_card()
    if model_card is None or not model_card.is_file():
        parser.error(
            "ASAP7 model card not found; pass --model-card or ASAP7_MODEL_CARD"
        )
    output_dir = args.out.resolve()
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

    result = run_sense_amp_testbench(
        model_card=model_card,
        osdi=osdi,
        output_dir=output_dir,
        ngspice=args.ngspice,
        vdd=args.vdd,
        differential_v=args.differential_v,
        n_fins=args.n_fins,
        p_fins=args.p_fins,
        output_load_ff=args.output_load_ff,
        precharge_release_ns=args.precharge_release_ns,
        enable_ns=args.enable_ns,
        stop_ns=args.stop_ns,
        step_ps=args.step_ps,
    )
    print(json.dumps(asdict(result), indent=2))
    if args.plot:
        # A failing run is the one most worth looking at, so plot either way.
        from .spice_plots import plot_sense_amp

        print(f"plot: {plot_sense_amp(output_dir)}")
    print(f"artifacts: {output_dir}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
