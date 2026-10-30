"""Sweep complete BROM stimulus schedules without attributing a limiting path."""

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

from ..devices.spice import find_default_model_card
from .compiler import BromSpec, selector_map
from .replica_timing import characterize_replica
from .timing import TimingConditions


def summarize_sweep(reports):
    if not reports:
        raise ValueError("at least one timing report is required")
    first = reports[0]
    identity = (
        "spec",
        "words",
        "replica",
        "wire_rc_controls",
        "sized_controls",
        "circuit_sha256",
        "model_sha256",
        "osdi_sha256",
        "register_clock_rc_included",
        "startup_ramp_ps",
    )
    conditions = {k: v for k, v in first["conditions"].items() if k != "phase_scale"}
    for r in reports:
        if any(r.get(k) != first.get(k) for k in identity):
            raise ValueError("timing reports describe different circuits/models")
        if {
            k: v for k, v in r["conditions"].items() if k != "phase_scale"
        } != conditions:
            raise ValueError("timing reports describe different conditions")
    runs = sorted(reports, key=lambda r: r["conditions"]["phase_scale"])
    samples = []
    for r in runs:
        period = 1000 * r["conditions"]["phase_scale"]
        if not math.isfinite(period) or period <= 0:
            raise ValueError("invalid clock period")
        if r["passed"] and r["status"] != "passed":
            raise ValueError("inconsistent timing status")
        samples.append(
            {
                "period_ps": period,
                "low_ps": 0.55 * period,
                "high_ps": 0.45 * period,
                # Address and clock ramp starts both scale; slew does not.
                # These are stimulus opportunities, not measured setup limits.
                "scheduled_address_lead_to_falling_clk_ps": 0.16 * period,
                "scheduled_address_settled_before_falling_ramp_ps": 0.16 * period
                - r["conditions"]["slew_ps"],
                "status": r["status"],
                "passed": r["passed"],
                "failures": r.get("failures", []),
                "error": r.get("error"),
                "report": r.get("report_path"),
                "deck_sha256": r.get("deck_sha256"),
                "registered_output": r.get("registered_output"),
                "minimum_margins_ps": r.get("minimum_margins_ps", {}),
                "cycle_paths": r.get("cycle_paths"),
            }
        )
    passing = [r for r in samples if r["passed"]]
    best = passing[0] if passing else None
    nonmonotonic = any(
        a["passed"] and b["status"] == "failed"
        for a in samples
        for b in samples
        if a["period_ps"] < b["period_ps"]
    )
    below = [
        r["period_ps"]
        for r in samples
        if r["status"] == "failed" and best and r["period_ps"] < best["period_ps"]
    ]
    transition = [max(below), best["period_ps"]] if below and not nonmonotonic else None
    transition_failures = sorted(
        {
            failure
            for r in samples
            if transition
            and r["period_ps"] == transition[0]
            and r["status"] == "failed"
            for failure in r["failures"]
        }
    )
    mapping = selector_map(BromSpec(**first["spec"]), first["words"])
    patterns = sorted({pattern for row in mapping for pattern in row})
    return {
        "schema": "brom_timing_estimate_v1",
        "image_coverage": {
            "selector_patterns": patterns,
            "all_16_selector_patterns": len(patterns) == 16,
            "worst_case_qualified": False,
        },
        "timing_qualified": False,
        "extracted": False,
        **{k: first.get(k) for k in identity},
        "conditions": conditions,
        "scope": first["scope"],
        "duty_cycle_high": 0.45,
        "fastest_tested_passing_period_ps": best["period_ps"] if best else None,
        # Retained as null for readers of v1; this sweep cannot identify a
        # minimum period independent of the jointly scaled address schedule.
        "estimated_minimum_period_bracket_ps": None,
        "observed_schedule_transition_ps": transition,
        "observed_schedule_spacing_ps": transition[1] - transition[0]
        if transition
        else None,
        "failed_checks_at_schedule_transition": transition_failures,
        "limiting_constraint": None,
        "address_setup_hold_swept": False,
        "schedule_scaling": "clock phases, address events and initial precharge scale together; slew is fixed",
        "nonmonotonic": nonmonotonic,
        "passing_clock_constraints": {
            "high_ps": best["high_ps"],
            "low_ps": best["low_ps"],
            "independently_minimized": False,
        }
        if best
        else None,
        "clk_to_valid_ps_at_fastest_pass": best["registered_output"].get(
            "maximum_clk_to_valid_ps"
        )
        if best and best["registered_output"]
        else None,
        "setup_hold_qualified": False,
        "samples": samples,
        "limitations": [
            "Sampled schedules at fixed duty cycle; numerical errors are not failing timing bounds.",
            "The observed transition is between complete tested schedules, not a read-path or minimum-period bound.",
            "Failed checks are symptoms, not root-cause attribution; address setup/hold may be limiting.",
            "Wire model excludes core bitlines, coupling, data paths and receiver-specific skew.",
            "No macro address setup/hold or independent minimum high/low pulse qualification.",
        ],
    }


def sweep_timing(
    spec,
    *,
    scales,
    out,
    model_card,
    osdi,
    conditions=None,
    words=None,
    wire_rc_controls=False,
    klu=False,
    startup_ramp_ps=0,
):
    scales = sorted(set(scales))
    if not scales or any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("phase scales must be finite and positive")
    conditions = conditions or TimingConditions()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "timing_estimate.json"
    summary_path.write_text(
        json.dumps({"status": "running", "timing_qualified": False}) + "\n"
    )
    reports = []
    for index, scale in enumerate(scales):
        run = out / f"case_{index:03d}"
        # A report left by an earlier sweep must not stand in for this run.
        (run / "report.json").unlink(missing_ok=True)
        try:
            r = characterize_replica(
                spec,
                out=run,
                model_card=model_card,
                osdi=osdi,
                conditions=replace(conditions, phase_scale=scale),
                words=words,
                wire_rc_controls=wire_rc_controls,
                startup_ramp_ps=startup_ramp_ps,
                klu=klu,
            )
        except Exception:
            # characterize_replica records any failure it meets while running
            # a case (solver, timeout, analysis) as an error sample; keep it
            # and go on. A failure that never reached the report is a bug.
            recorded = run / "report.json"
            r = json.loads(recorded.read_text()) if recorded.exists() else {}
            if r.get("status") != "error":
                raise
        # Hash inputs even when ngspice cannot find an operating point.
        r.update(
            model_sha256=hashlib.sha256(Path(model_card).read_bytes()).hexdigest(),
            osdi_sha256=hashlib.sha256(Path(osdi).read_bytes()).hexdigest(),
            report_path=str(run / "report.json"),
        )
        reports.append(r)
    summary = summarize_sweep(reports)
    summary["status"] = "complete"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def add_register_estimate(summary, path):
    """Attach standalone bounds only at the same model, load and slew."""
    from .output_register import output_register_circuit

    expected_register = hashlib.sha256(
        output_register_circuit(fins=summary["spec"]["fins"], vt=summary["spec"]["vt"])
        .netlist()
        .encode()
    ).hexdigest()
    r = json.loads(Path(path).read_text())
    fields = ("vdd", "temperature_c", "slew_ps", "output_load_ff", "step_ps")
    if (
        r.get("register_cell_sha256") != expected_register
        or r.get("status") != "passed"
        or not r.get("passed")
        or r.get("fins") != summary["spec"]["fins"]
        or r.get("vt") != summary["spec"]["vt"]
        or any(r.get(k) != summary.get(k) for k in ("model_sha256", "osdi_sha256"))
        or any(r["conditions"].get(k) != summary["conditions"].get(k) for k in fields)
    ):
        raise ValueError(
            "register characterization does not match timing conditions/model or register cell"
        )
    summary["standalone_register"] = {
        "report": str(Path(path).resolve()),
        "report_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "bounds": r["bounds"],
        "reference_clk_to_valid_ps": r["reference_clk_to_valid_ps"],
        "register_cell_sha256": r["register_cell_sha256"],
        "scope": r["scope"],
        "delay_degradation_limit": r["delay_degradation_limit"],
        "macro_address_setup_hold": False,
    }
    return summary


def attach_estimate(path, circuit, spec, words):
    """Reject stale/foreign estimates before including them in a compiler manifest."""
    from dataclasses import asdict

    report = json.loads(Path(path).read_text())
    if (
        report.get("schema") != "brom_timing_estimate_v1"
        or report.get("status") != "complete"
        or report.get("spec") != asdict(spec)
        or report.get("words") != list(words)
        or report.get("circuit_sha256")
        != hashlib.sha256(circuit.netlist().encode()).hexdigest()
    ):
        raise ValueError(
            "timing estimate does not match this completed circuit and ROM image"
        )
    if (
        report.get("timing_qualified") is not False
        or report.get("extracted") is not False
    ):
        raise ValueError("unsupported timing qualification claim")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--bits", type=int, default=2)
    p.add_argument("--ways", type=int, choices=(1, 4), default=1)
    p.add_argument("--keeper-fins", type=int, choices=(0, 1, 2, 3), default=0)
    p.add_argument("--words", type=Path)
    p.add_argument("--register-report", type=Path)
    p.add_argument("--phase-scales", type=float, nargs="+", default=[0.6, 0.8, 1])
    p.add_argument("--wire-rc-controls", action="store_true")
    p.add_argument("--klu", action="store_true")
    p.add_argument("--startup-ramp-ps", type=float, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--osdi", type=Path, required=True)
    p.add_argument("--model-card", type=Path, default=find_default_model_card())
    p.add_argument("--vdd", type=float, default=0.7)
    p.add_argument("--temperature-c", type=float, default=25)
    p.add_argument("--step-ps", type=float, default=0.5)
    p.add_argument("--slew-ps", type=float, default=5)
    p.add_argument("--output-load-ff", type=float, default=1)
    args = p.parse_args()
    words = json.loads(args.words.read_text()) if args.words else None
    spec = BromSpec(
        len(words) if words is not None else args.depth,
        args.bits,
        ways=args.ways,
        keeper_fins=args.keeper_fins,
        registered_output=True,
    )
    r = sweep_timing(
        spec,
        scales=args.phase_scales,
        out=args.out,
        model_card=args.model_card,
        osdi=args.osdi,
        words=words,
        wire_rc_controls=args.wire_rc_controls,
        startup_ramp_ps=args.startup_ramp_ps,
        klu=args.klu,
        conditions=TimingConditions(
            vdd=args.vdd,
            temperature_c=args.temperature_c,
            slew_ps=args.slew_ps,
            step_ps=args.step_ps,
            output_load_ff=args.output_load_ff,
        ),
    )
    if args.register_report:
        add_register_estimate(r, args.register_report)
        (args.out / "timing_estimate.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2))
    return 0 if r["fastest_tested_passing_period_ps"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
