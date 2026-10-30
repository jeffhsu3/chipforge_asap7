"""Reject false timing bounds, stale reports and invalid waveform evidence."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from chipforge_asap7.brom import BromSpec, ReplicaSpec, self_timed_macro_circuit
from chipforge_asap7.brom.register_characterization import (
    SIGNALS,
    analyze_register_trials,
    render_register_bench,
    sampled_bound,
)
from chipforge_asap7.brom.replica_timing import render_replica_bench
from chipforge_asap7.brom.timing import TimingConditions
from chipforge_asap7.brom.timing_estimate import attach_estimate, summarize_sweep


def report(scale=1, status="passed"):
    spec = BromSpec(4, 1, registered_output=True)
    words = [0, 1, 0, 1]
    c = self_timed_macro_circuit(spec, words)
    return {
        "status": status,
        "passed": status == "passed",
        "spec": asdict(spec),
        "words": words,
        "conditions": asdict(TimingConditions(phase_scale=scale)),
        "circuit_sha256": hashlib.sha256(c.netlist().encode()).hexdigest(),
        "scope": "ideal wires",
        "model_sha256": "model",
        "osdi_sha256": "osdi",
    }


def test_schedule_transition_excludes_numerical_errors_and_reports_duty():
    a = summarize_sweep([report(0.7, "error"), report()])
    assert a["estimated_minimum_period_bracket_ps"] is None
    assert a["observed_schedule_transition_ps"] is None
    b = summarize_sweep([report(0.6, "failed"), report(0.7, "error"), report()])
    assert b["estimated_minimum_period_bracket_ps"] is None
    assert b["observed_schedule_transition_ps"] == [600, 1000]
    assert b["passing_clock_constraints"] == {
        "high_ps": 450,
        "low_ps": 550,
        "independently_minimized": False,
    }
    assert not b["timing_qualified"]
    assert (
        summarize_sweep([report(1, "failed")])["fastest_tested_passing_period_ps"]
        is None
    )


def test_nonmonotonic_schedule_sweep_reports_no_transition_pair():
    r = summarize_sweep([report(0.6, "failed"), report(0.8), report(1, "failed")])
    assert r["nonmonotonic"] and r["observed_schedule_transition_ps"] is None


@pytest.mark.parametrize(
    "key,value",
    [("words", [1, 1, 1, 1]), ("model_sha256", "other"), ("circuit_sha256", "other")],
)
def test_mismatched_sweeps_rejected(key, value):
    b = report()
    b[key] = value
    with pytest.raises(ValueError, match="different"):
        summarize_sweep([report(), b])


def test_manifest_rejects_stale_circuit_and_unqualified_claims(tmp_path):
    spec = BromSpec(4, 1, registered_output=True)
    words = [0, 1, 0, 1]
    c = self_timed_macro_circuit(spec, words)
    r = summarize_sweep([report()])
    r["status"] = "complete"
    path = tmp_path / "estimate.json"
    path.write_text(json.dumps(r))
    assert attach_estimate(path, c, spec, words) == r
    with pytest.raises(ValueError, match="match"):
        attach_estimate(path, c, spec, [1, 1, 0, 1])
    c.inv("CLK", "extra")
    with pytest.raises(ValueError, match="match"):
        attach_estimate(path, c, spec, words)
    r["timing_qualified"] = True
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="qualification"):
        attach_estimate(path, self_timed_macro_circuit(spec, words), spec, words)


def test_sampled_bound_requires_both_polarities_and_passing_suffix():
    records = [
        {"kind": "hold", "value_ps": v, "passed": good, "high": bool(i % 2)}
        for i, (v, good) in enumerate(
            [
                (0, True),
                (0, True),
                (10, False),
                (10, True),
                (20, True),
                (20, True),
            ]
        )
    ]
    r = sampled_bound(records, "hold")
    assert r["passing_bound_ps"] == 20
    assert r["largest_failing_sample_ps"] == 10
    assert r["nonmonotonic"]


def synthetic_register():
    t = np.arange(2801, dtype=float)
    traces = {n: np.zeros_like(t) for n in SIGNALS}
    traces["CLK"][400:800] = 1
    traces["CLK"][1500:1900] = 1
    traces["Q<0>"][1550:] = 1
    traces["D<0>"][900:] = 1
    # Nonoverlapping input transmission gates.
    traces["REG_MB"][:] = 1
    traces["REG_SB"][:] = 1
    return t * 1e-12, traces, [{"kind": "reference", "high": True, "value_ps": 200}]


def test_register_waveform_faults_and_equivalent_duplicate_timestamps():
    t, traces, records = synthetic_register()
    a = analyze_register_trials(t, traces, records, 1)
    assert a["trials"][0]["passed"]
    assert a["reference_clk_to_valid_ps"]["rise"] == pytest.approx(50.4)
    dup = 1200
    td = np.insert(t, dup, t[dup])
    vd = {n: np.insert(v, dup, v[dup]) for n, v in traces.items()}
    assert analyze_register_trials(td, vd, records, 1)["trials"] == a["trials"]
    vd["Q<0>"][dup] = 1
    with pytest.raises(ValueError, match="duplicate"):
        analyze_register_trials(td, vd, records, 1)
    with pytest.raises(ValueError, match="incomplete"):
        analyze_register_trials(
            t[:-10], {n: v[:-10] for n, v in traces.items()}, records, 1
        )
    traces["Q<0>"][1800] = 0
    assert not analyze_register_trials(t, traces, records, 1)["trials"][0]["passed"]


def test_register_pulse_trials_preserve_ramp_widths_and_both_polarities(tmp_path):
    deck, records = render_register_bench(
        TimingConditions(), tmp_path / "models.pm", tmp_path / "model.osdi"
    )
    assert len(records) == 62
    for kind in ("reference", "setup", "hold", "high_width", "low_width"):
        assert {r["high"] for r in records if r["kind"] == kind} == {False, True}
    assert "1497.5p" in deck and "1502.5p" in deck
    assert "173600p" in deck
    assert "166542.5p" in deck and "166547.5p" in deck


def test_registered_rc_stress_includes_pin_clock_and_phase_feedback(tmp_path):
    deck, signals = render_replica_bench(
        BromSpec(4, 1, registered_output=True),
        [0, 1, 0, 1],
        ReplicaSpec(),
        TimingConditions(phase_scale=10),
        tmp_path / "models.pm",
        tmp_path / "model.osdi",
        wire_rc_controls=True,
    )
    for net in (
        "CLK",
        "REG_CLKB",
        "REG_M",
        "REG_MB",
        "REG_S",
        "REG_SB",
        "REG_M_OFF",
        "REG_S_OFF",
        "REG_M_CLOSED",
        "REG_S_CLOSED",
    ):
        assert f"* estimated control wire: {net}\n" in deck
    assert "CLK_RECEIVER" in signals
    assert "Vstim0 rcdrv_" in deck
    assert "* estimated control wire: SAMPLED" not in deck


def test_cycle_metrics_have_physical_units():
    from test_brom_replica import synthetic_replica_waveform

    from chipforge_asap7.brom.cycle_timing import measure_cycle_paths

    t, traces, spec, words = synthetic_replica_waveform()
    # The protocol fixture omits CLK; add its external stimulus.
    traces = deepcopy(traces)
    traces["CLK"] = np.ones_like(t)
    for i in range(spec.depth + 1):
        traces["CLK"][(t * 1e12 >= i * 1000 + 200) & (t * 1e12 < i * 1000 + 750)] = 0
    a = measure_cycle_paths(t, traces, spec, words, 1)
    b = measure_cycle_paths(t * 10, traces, spec, words, 1, phase_scale=10)
    for key, value in a["maxima_ps"].items():
        assert value is not None
        assert b["maxima_ps"][key] == pytest.approx(value * 10)


def test_startup_ramp_is_explicit_and_finishes_before_address(tmp_path):
    args = (
        BromSpec(4, 1, registered_output=True),
        [0, 1, 0, 1],
        ReplicaSpec(),
        TimingConditions(),
        tmp_path / "models.pm",
        tmp_path / "model.osdi",
    )
    deck, _ = render_replica_bench(*args, startup_ramp_ps=25)
    assert "Vsupply vdd 0 PWL(0 0 25p 0.7)" in deck
    assert "PWL(0 0 25p 0.7" in deck
    with pytest.raises(ValueError, match="startup ramp"):
        render_replica_bench(*args, startup_ramp_ps=50)


def test_register_report_requires_matching_load_and_model(tmp_path):
    from chipforge_asap7.brom.output_register import output_register_circuit
    from chipforge_asap7.brom.timing_estimate import add_register_estimate

    summary = summarize_sweep([report()])
    r = {
        "status": "passed",
        "passed": True,
        "fins": 2,
        "vt": "rvt",
        "model_sha256": "model",
        "osdi_sha256": "osdi",
        "conditions": summary["conditions"],
        "bounds": {},
        "reference_clk_to_valid_ps": {},
        "scope": "standalone ideal wires",
        "delay_degradation_limit": 0.1,
        "register_cell_sha256": hashlib.sha256(
            output_register_circuit().netlist().encode()
        ).hexdigest(),
    }
    path = tmp_path / "register.json"
    path.write_text(json.dumps(r))
    assert not add_register_estimate(summary, path)["standalone_register"][
        "macro_address_setup_hold"
    ]
    current_hash = r.pop("register_cell_sha256")
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="conditions/model"):
        add_register_estimate(summary, path)
    r["register_cell_sha256"] = current_hash

    r["conditions"] = {**r["conditions"], "output_load_ff": 4}
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="conditions"):
        add_register_estimate(summary, path)


def test_sweep_retains_solver_errors_and_propagates_startup(tmp_path, monkeypatch):
    import chipforge_asap7.brom.timing_estimate as module

    card, osdi = tmp_path / "card.pm", tmp_path / "model.osdi"
    card.write_text("model")
    osdi.write_text("osdi")
    calls = []

    def characterize(spec, **kwargs):
        calls.append(kwargs)
        r = report(kwargs["conditions"].phase_scale)
        r["startup_ramp_ps"] = kwargs["startup_ramp_ps"]
        if len(calls) == 1:
            r.update(status="error", passed=False, error="no operating point")
            kwargs["out"].mkdir(parents=True)
            (kwargs["out"] / "report.json").write_text(json.dumps(r))
            raise RuntimeError("no operating point")
        return r

    monkeypatch.setattr(module, "characterize_replica", characterize)
    r = module.sweep_timing(
        BromSpec(4, 1, registered_output=True),
        scales=[0.7, 1],
        out=tmp_path / "sweep",
        model_card=card,
        osdi=osdi,
        startup_ramp_ps=25,
    )
    assert len(calls) == 2 and all(c["startup_ramp_ps"] == 25 for c in calls)
    assert r["fastest_tested_passing_period_ps"] == 1000
    assert r["estimated_minimum_period_bracket_ps"] is None
    assert r["samples"][0]["status"] == "error"


def test_register_startup_ramp_preserves_conditioning_interval(tmp_path):
    args = (TimingConditions(), tmp_path / "models.pm", tmp_path / "model.osdi")
    deck, _ = render_register_bench(*args, startup_ramp_ps=25)
    assert "Vsupply vdd 0 PWL(0 0 25p 0.7)" in deck
    assert "uic" not in deck
    with pytest.raises(ValueError, match="first register data edge"):
        render_register_bench(*args, startup_ramp_ps=50)


def test_scaled_schedule_does_not_attribute_failures_to_a_read_path():
    slow = report(0.6, "failed")
    slow["conditions"]["phase_scale"] = 0.3
    slow["failures"] = ["address 0: registered Q[0] did not capture"]
    slow["conditions"]["slew_ps"] = 2
    fast = report()
    fast["conditions"]["phase_scale"] = 0.5
    fast["conditions"]["slew_ps"] = 2
    r = summarize_sweep([slow, fast])
    assert r["observed_schedule_transition_ps"] == [300, 500]
    assert r["failed_checks_at_schedule_transition"] == slow["failures"]
    assert r["estimated_minimum_period_bracket_ps"] is None
    assert r["limiting_constraint"] is None
    assert not r["address_setup_hold_swept"]
    assert r["samples"][0]["scheduled_address_lead_to_falling_clk_ps"] == 48
    assert r["samples"][0]["scheduled_address_settled_before_falling_ramp_ps"] == 46
    assert r["samples"][1]["scheduled_address_lead_to_falling_clk_ps"] == 80


def test_schedule_report_exposes_sparse_pattern_coverage_and_spacing():
    from chipforge_asap7.brom.timing import pattern_words

    sparse = report(0.6, "failed")
    sparse["spec"] = asdict(BromSpec(4, 2, registered_output=True))
    sparse["words"] = pattern_words(BromSpec(**sparse["spec"]))
    passing = deepcopy(sparse)
    passing.update(status="passed", passed=True)
    passing["conditions"]["phase_scale"] = 0.7
    r = summarize_sweep([sparse, passing])
    assert r["image_coverage"]["selector_patterns"] == [0, 1]
    assert not r["image_coverage"]["all_16_selector_patterns"]
    assert r["observed_schedule_spacing_ps"] == pytest.approx(100)
    full = report()
    full["spec"] = asdict(BromSpec(4, 16, registered_output=True))
    full["words"] = pattern_words(BromSpec(**full["spec"]))
    coverage = summarize_sweep([full])["image_coverage"]
    assert coverage["all_16_selector_patterns"]
    assert not coverage["worst_case_qualified"]
