"""Geometry-based RC estimates and bounded physical driver optimization."""

from copy import deepcopy

import pytest

from chipforge_asap7.brom import (
    BromSpec,
    WireRCSpec,
    routed_control_rc,
    self_timed_macro_circuit,
    size_control_distribution,
    size_wire_rc_distribution,
)


def circuit():
    return self_timed_macro_circuit(BromSpec(4, 2), [0, 1, 2, 3])


def test_rc_assumptions_are_explicit_and_scale_capacitance():
    c = circuit()
    a = routed_control_rc(c)
    b = routed_control_rc(c, WireRCSpec(wire_cap_ff_per_um=0.4))
    assert "VDD" not in a and "VSS" not in a
    for net in ("PRECHN", "SAMPLE", "C<0,10>"):
        assert a[net]["wire_cap_ff"] > 0
        assert a[net]["resistance_sum_ohm"] > 0
        assert b[net]["resistance_sum_ohm"] == a[net]["resistance_sum_ohm"]
        assert b[net]["wire_cap_ff"] == pytest.approx(2 * a[net]["wire_cap_ff"])
        assert b[net]["gate_cap_ff"] == a[net]["gate_cap_ff"]
    assert "M5" in a["PRECHN"]["length_um_by_layer"]


def test_search_preserves_array_and_reduces_estimated_cost():
    c = circuit()
    original = {
        d.name: d
        for d in c.devices
        if d.name in c.array.device_names | c.replica.selector.device_names
    }
    size_wire_rc_distribution(c)
    report = c.distribution
    assert report["selected_score_ps"] < report["initial_score_ps"]
    assert report["wire_rc_included"] and not report["extracted"]
    assert not report["timing_qualified"]
    assert original == {d.name: d for d in c.devices if d.name in original}
    assert len({d.name for d in c.devices}) == len(c.devices)
    assert (
        len(c.devices) <= len(circuit().devices) + report["model"]["max_added_devices"]
    )
    other = size_wire_rc_distribution(circuit())
    assert c.devices == other.devices
    assert c.distribution == other.distribution
    with pytest.raises(ValueError, match="unsized"):
        size_wire_rc_distribution(c)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wire_cap_ff_per_um": 0},
        {"gate_cap_ff_per_fin": float("nan")},
        {"max_rounds": True},
        {"max_parallel": 0},
    ],
)
def test_invalid_model(kwargs):
    with pytest.raises(ValueError):
        WireRCSpec(**kwargs)


def test_wire_loads_can_change_sizing_and_invalid_values_are_rejected():
    c = circuit()
    a = size_control_distribution(deepcopy(c))
    b = size_control_distribution(deepcopy(c), wire_gate_fins={"PRECHN": 100})

    def count(x):
        return next(
            r["stages"][-1]["parallel_inverters"]
            for r in x.distribution["nets"]
            if r["net"] == "PRECHN"
        )

    assert count(b) > count(a)
    with pytest.raises(ValueError):
        size_control_distribution(c, wire_gate_fins={"PRECHN": -1})


def test_estimated_spine_length_matches_router_coordinates():
    from chipforge_asap7.brom.layout import macro_periphery
    from chipforge_asap7.brom.periphery import periphery_plan

    c = circuit()
    plan = periphery_plan(macro_periphery(c))
    ys = [r.buses["PRECHN"] for r in plan.rows if "PRECHN" in r.buses]
    rc = routed_control_rc(c)["PRECHN"]
    assert rc["length_um_by_layer"]["M5"] == pytest.approx(
        (max(ys) - min(ys) + 48) / 1000
    )
    assert 0 < rc["max_driver_sink_path_ohm"] <= rc["resistance_sum_ohm"]


def test_too_small_budget_leaves_circuit_unchanged():
    c = circuit()
    original = deepcopy(c)
    with pytest.raises(ValueError, match="resource limit"):
        size_wire_rc_distribution(c, WireRCSpec(max_parallel=1))
    assert c.devices == original.devices and c.name == original.name
    assert c.distribution is None


def test_rc_stress_deck_has_real_resistors_and_scaled_cancel(tmp_path):
    from chipforge_asap7.brom import ReplicaSpec
    from chipforge_asap7.brom.replica_timing import render_replica_bench
    from chipforge_asap7.brom.timing import TimingConditions

    deck, _ = render_replica_bench(
        BromSpec(4, 2),
        [0, 1, 2, 3],
        ReplicaSpec(),
        TimingConditions(phase_scale=10, step_ps=0.5),
        tmp_path / "model.pm",
        tmp_path / "model.osdi",
        wire_rc_controls=True,
    )
    resistors = [s for s in deck.splitlines() if s.startswith("Rwire")]
    capacitors = [s for s in deck.splitlines() if s.startswith("Cwire")]
    assert resistors and len(capacitors) == 2 * len(resistors)
    assert all(float(s.split()[-1]) > 0 for s in resistors)
    assert "42060" in deck and "52000" in deck
    assert len([s for s in deck.splitlines() if s.startswith("Vstim")]) == 4


def test_scaled_checker_keeps_physical_margin_units():
    from test_brom_replica import synthetic_replica_waveform

    from chipforge_asap7.brom.replica_timing import analyze_replica

    t, traces, spec, words = synthetic_replica_waveform()
    a = analyze_replica(t, traces, spec, words, 1)
    b = analyze_replica(t * 10, traces, spec, words, 1, phase_scale=10)
    assert b["passed"]
    for key, value in a["minimum_margins_ps"].items():
        assert b["minimum_margins_ps"][key] == pytest.approx(value * 10)
