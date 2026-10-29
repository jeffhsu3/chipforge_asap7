"""Transient check of an integrated controller with externally supplied acknowledgments.

This tests sequencing, not a physical replica or extracted distribution delay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..devices.spice import (
    _osdi_command_path,
    _spice_path,
    find_default_model_card,
    translate_asap7_model,
)
from .compiler import BromSpec
from .controller import CONTROLLER_INPUTS, CONTROLLER_OUTPUTS, clocked_macro_circuit
from .timing import TimingConditions, _run_ngspice


def render_controller_bench(model_card, osdi, conditions):
    spec = BromSpec(4, 2)
    c = clocked_macro_circuit(spec, [0, 1, 2, 3])
    nets = sorted({net for d in c.devices for net in d.nodes})
    nodes = {net: f"n{i}" for i, net in enumerate(nets)}
    nodes.update(VDD="vdd", VSS="0")
    vdd, slew = conditions.vdd, conditions.slew_ps
    initial = {
        "CLK": 1,
        "RESETN": 0,
        "PRECHN_FAR": 0,
        "READ_READY": 0,
        "SAMPLE_CLOSED": 1,
        "CANDIDATE_BUSY": 0,
        "A<0>": 0,
        "A<1>": 0,
    }
    events = {net: [] for net in initial}
    events["RESETN"].append((100, 1))
    # Four successful reads; read 2 is reset after sampling. The fifth read
    # is canceled before READ_READY and must leave the previous output intact.
    addresses = [0, 1, 2, 3, 1]
    for cycle, address in enumerate(addresses):
        base = cycle * 1200
        for bit in range(2):
            events[f"A<{bit}>"].append((base + 40, address >> bit & 1))
            events[f"A<{bit}>"].append((base + 740, 1 - (address >> bit & 1)))
        edges = {
            "CLK": [(200, 0), (850, 1)],
            "PRECHN_FAR": [(400, 1), (1180, 0)],
            "READ_READY": [(650, 1), (1120, 0)],
            "CANDIDATE_BUSY": [(470, 1), (1100, 0)],
        }
        if cycle != 4:
            edges["SAMPLE_CLOSED"] = [(700, 0), (1000, 1)]
        for net, changes in edges.items():
            events[net].extend((base + t, value) for t, value in changes)
    events["RESETN"].extend(
        [(2 * 1200 + 760, 0), (3 * 1200 + 50, 1), (4 * 1200 + 550, 0)]
    )
    lines = [
        "* Integrated BROM clock controller, externally scheduled completion acknowledgments",
        f".include {_spice_path(model_card)}",
        f".temp {conditions.temperature_c:g}",
        ".options method=gear reltol=1e-3 abstol=1e-12 rshunt=1e15",
        f"Vdd vdd 0 {vdd:g}",
    ]
    for i, (net, changes) in enumerate(events.items()):
        old = initial[net]
        points = [f"0 {old * vdd:g}"]
        for t, value in sorted(changes):
            if old != value:
                points += [f"{t}p {old * vdd:g}", f"{t + slew:g}p {value * vdd:g}"]
            old = value
        lines.append(f"V{i} {nodes[net]} 0 PWL({' '.join(points)})")
    for d in c.devices:
        lines.append(
            f"N{d.name} {' '.join(nodes[n] for n in d.nodes)} {d.spec.model} L=20n NFIN=2 NF=1"
        )
    for bit in range(2):
        lines.append(f"CQ{bit} {nodes[f'Q<{bit}>']} 0 {conditions.output_load_ff:g}f")
    by_name = {d.name: d for d in c.devices}
    for group, members in c.placement_groups:
        if group.startswith(("address_", "output_latch_")):
            for index, voltage in ((0, 0), (2, vdd), (4, 0)):
                lines.append(
                    f".nodeset V({nodes[by_name[members[index]].nodes[0]]})={voltage:g}"
                )
    signals = [*CONTROLLER_INPUTS, *CONTROLLER_OUTPUTS, "Q<0>", "Q<1>"]
    vectors = [f"v({nodes[n]})" for n in signals]
    lines += [
        ".control",
        "set num_threads=1",
        f"pre_osdi {_osdi_command_path(osdi)}",
        "set wr_singlescale",
        "set wr_vecnames",
        "set numdgt=12",
        "save " + " ".join(vectors),
        f"tran {conditions.step_ps:g}p 6200p 0 {conditions.step_ps:g}p",
        "wrdata waveform.dat " + " ".join(vectors),
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines), signals


def check_controller_waveform(time, traces, vdd):
    failures = []
    t = time * 1e12
    high = lambda net: traces[net] >= 0.8 * vdd
    low = lambda net: traces[net] <= 0.2 * vdd
    valid = t >= 150  # allow initialization while RESETN is asserted
    for label, violation in (
        ("evaluation before precharge release", high("EVAL") & ~high("PRECHN")),
        ("sampling outside evaluation", high("SAMPLE") & ~high("EVAL")),
        ("sampling before read completion", high("SAMPLE") & ~high("READ_READY")),
        (
            "evaluation ended before sample closure",
            low("SAMPLE_CLOSED") & high("PRECHN_FAR") & ~high("EVAL"),
        ),
        (
            "precharge while sampling/evaluating/candidates active",
            low("PRECHN")
            & (
                ~low("EVAL")
                | ~low("SAMPLE")
                | ~low("CANDIDATE_BUSY")
                | ~high("SAMPLE_CLOSED")
            ),
        ),
    ):
        if np.any(violation & valid):
            failures.append(label)
    expected_phases = [
        (300, (1, 0, 0)),
        (550, (1, 1, 0)),
        (800, (1, 1, 1)),
        (950, (1, 1, 0)),
        (1080, (1, 0, 0)),
        (1190, (0, 0, 0)),
    ]
    for when, levels in expected_phases:
        for net, value in zip(CONTROLLER_OUTPUTS, levels, strict=True):
            voltage = np.interp(when, t, traces[net])
            if voltage < 0.8 * vdd if value else voltage > 0.2 * vdd:
                failures.append(f"{net} wrong at {when} ps")
    for cycle, word in enumerate([0, 1, 2, 3, 3]):
        window = (t >= cycle * 1200 + 950) & (t <= (cycle + 1) * 1200 + 640)
        for bit in range(2):
            good = high(f"Q<{bit}>") if word >> bit & 1 else low(f"Q<{bit}>")
            if not np.all(good[window]):
                failures.append(f"read/hold failure: cycle {cycle}, Q<{bit}>")
    reset_drain = (t >= 2 * 1200 + 900) & (t <= 3 * 1200 + 200)
    if not np.all(low("SAMPLE")[reset_drain]):
        failures.append("reset did not close sampling")
    canceled = (t >= 4 * 1200 + 650) & (t <= 5 * 1200)
    if not np.all(low("SAMPLE")[canceled]):
        failures.append("canceled read sampled data")
    return {"passed": not failures, "failures": failures}


def run_controller_bench(*, out, model_card, osdi, conditions=None):
    conditions = conditions or TimingConditions(step_ps=0.5)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    report = {
        "status": "running",
        "timing_qualified": False,
        "scope": "ideal wires; externally scheduled completion acknowledgments",
        "vdd": conditions.vdd,
        "temperature_c": conditions.temperature_c,
        "output_load_ff": conditions.output_load_ff,
        "step_ps": conditions.step_ps,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        card = Path(model_card).read_text()
        translated = out / "models.pm"
        translated.write_text(
            "\n".join(translate_asap7_model(card, m) for m in ("nmos_rvt", "pmos_rvt"))
        )
        deck, signals = render_controller_bench(translated, osdi, conditions)
        path = out / "controller.sp"
        path.write_text(deck)
        _run_ngspice(path, "ngspice", [out / "waveform.dat"])
        values = np.loadtxt(out / "waveform.dat", skiprows=1)
        if (
            values.ndim != 2
            or values.shape[1] != len(signals) + 1
            or not np.isfinite(values).all()
            or values[-1, 0] < 6.2e-9 * (1 - 1e-6)
            or np.any(np.diff(values[:, 0]) <= 0)
        ):
            raise RuntimeError("invalid or incomplete controller transient")
        report.update(
            check_controller_waveform(
                values[:, 0],
                dict(zip(signals, values[:, 1:].T, strict=True)),
                conditions.vdd,
            )
        )
        report.update(
            status="passed" if report["passed"] else "failed",
            model_sha256=hashlib.sha256(Path(model_card).read_bytes()).hexdigest(),
            osdi_sha256=hashlib.sha256(Path(osdi).read_bytes()).hexdigest(),
            deck_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        (out / "signals.json").write_text(json.dumps(signals, indent=2) + "\n")
    except Exception as exc:
        report.update(status="error", passed=False, error=str(exc))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--osdi", type=Path, required=True)
    parser.add_argument("--model-card", type=Path, default=find_default_model_card())
    parser.add_argument("--step-ps", type=float, default=0.5)
    parser.add_argument("--vdd", type=float, default=0.7)
    parser.add_argument("--temperature-c", type=float, default=25)
    args = parser.parse_args()
    if args.model_card is None:
        parser.error("model card required")
    result = run_controller_bench(
        out=args.out,
        model_card=args.model_card,
        osdi=args.osdi,
        conditions=TimingConditions(
            vdd=args.vdd, temperature_c=args.temperature_c, step_ps=args.step_ps
        ),
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
