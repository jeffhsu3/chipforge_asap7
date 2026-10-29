import numpy as np
import pytest

from chipforge_asap7.brom import BromSpec, macro_circuit, selector_map
from chipforge_asap7.brom.timing import (
    TimingConditions,
    analyze,
    characterize,
    pattern_words,
    render_deck,
    settling_delay,
)


@pytest.mark.parametrize("falling", [False, True])
def test_settling_uses_last_crossing_after_glitch(falling):
    t = np.arange(6, dtype=float)
    v = np.array([0, 1, 1, 0, 1, 1], dtype=float)
    if falling:
        v = 1 - v
    assert settling_delay(t, v, 0, 5, not falling, 1) == pytest.approx(3.9)
    assert settling_delay(t, v, 4, 5, not falling, 1) == 0
    assert settling_delay(t, v, 0, 3, not falling, 1) is None


@pytest.mark.parametrize("spec", [BromSpec(8, 16), BromSpec(32, 4, ways=4)])
def test_patterns_cover_all_selectors_in_each_block(spec):
    assert all(
        set(row) == set(range(16)) for row in selector_map(spec, pattern_words(spec))
    )


def test_deck_contains_full_macro_and_physical_loads(tmp_path):
    spec = BromSpec(8, 2, fins=3)
    words = pattern_words(spec)
    c = macro_circuit(spec, words)
    deck, signals, addresses, nodes = render_deck(
        spec,
        words,
        TimingConditions(output_load_ff=2, bitline_load_ff=3),
        tmp_path / "model.pm",
        tmp_path / "model.osdi",
    )
    assert len([line for line in deck.splitlines() if line.startswith("NM")]) == len(
        c.devices
    )
    assert deck.count("NFIN=3") == len(c.devices)
    assert addresses == [*range(8), *reversed(range(8))]
    assert "C" + "Q0 " + nodes["Q<0>"] + " 0 2f" in deck
    assert "CBL0 " + nodes["BL<0>"] + " 0 3f" in deck
    assert "DATA<0>" in signals


def test_failed_read_has_no_successful_delay_summary():
    spec = BromSpec(4, 1)
    t = np.linspace(0, 700e-12, 1001)
    traces = {f"{name}<0>": np.zeros_like(t) for name in ("BL", "DATA", "Q")}
    result = analyze(t, traces, spec, [1, 0, 0, 0], TimingConditions(), [0])
    assert not result["passed"]
    assert result["worst_ps"] == {}
    assert any("hold failure" in f for f in result["failures"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vdd": 0},
        {"step_ps": 5},
        {"phase_scale": 0},
        {"output_load_ff": -1},
        {"temperature_c": float("nan")},
    ],
)
def test_invalid_conditions_rejected(kwargs):
    with pytest.raises(ValueError):
        TimingConditions(**kwargs)


def test_simulator_failure_invalidates_old_report(tmp_path, monkeypatch):
    import json

    from chipforge_asap7.brom import timing

    out = tmp_path / "out"
    out.mkdir()
    report = out / "report.json"
    report.write_text('{"status": "passed"}')
    model, osdi = tmp_path / "models.pm", tmp_path / "model.osdi"
    model.write_text(
        ".model nmos_rvt nmos\n+ version=107\n.model pmos_rvt pmos\n+ version=107\n"
    )
    osdi.write_bytes(b"fixture")

    def fail(*args, **kwargs):
        raise RuntimeError("simulation failed")

    from types import SimpleNamespace

    monkeypatch.setattr(
        timing.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="fixture")
    )
    monkeypatch.setattr(timing, "_run_ngspice", fail)
    with pytest.raises(RuntimeError, match="simulation failed"):
        characterize(
            BromSpec(4, 1),
            [0, 1, 0, 1],
            TimingConditions(),
            model_card=model,
            osdi=osdi,
            out=out,
        )
    assert json.loads(report.read_text())["status"] == "error"


@pytest.mark.parametrize("sequence", [[], [-1], [4], [True]])
def test_invalid_read_sequence_rejected(tmp_path, sequence):
    with pytest.raises(ValueError, match="valid addresses"):
        render_deck(
            BromSpec(4, 1),
            [0, 1, 0, 1],
            TimingConditions(),
            tmp_path / "model.pm",
            tmp_path / "model.osdi",
            sequence,
        )


def test_missing_model_cannot_leave_stale_success(tmp_path):
    import json

    report = tmp_path / "report.json"
    report.write_text('{"status": "passed"}')
    with pytest.raises(FileNotFoundError):
        characterize(
            BromSpec(4, 1),
            [0, 1, 0, 1],
            TimingConditions(),
            model_card=tmp_path / "missing.pm",
            osdi=tmp_path / "missing.osdi",
            out=tmp_path,
        )
    assert json.loads(report.read_text())["status"] != "passed"
