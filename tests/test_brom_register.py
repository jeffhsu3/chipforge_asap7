"""Rising-edge output contract, physical register integration and timing checks."""

from dataclasses import replace

import numpy as np
import pytest
from test_brom import settle

from chipforge_asap7.brom import (
    BromSpec,
    macro_circuit,
    output_register_circuit,
    self_timed_macro_circuit,
)


@pytest.mark.parametrize("fins", [1, 2, 3])
def test_register_captures_only_on_rising_edge(fins):
    c = output_register_circuit(fins=fins)
    state = None
    for clk, data, expected in [
        (0, 0, None),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (1, 1, 1),
        (1, 0, 1),
        (0, 0, 1),
        (0, 1, 1),
        (0, 0, 1),
        (1, 0, 0),
    ]:
        state = settle(
            c, {"CLK": clk, "D<0>": data}, state, ideal_controls=("REG_M", "REG_S")
        )
        assert not (state["REG_M"] and state["REG_S"])
        if expected is not None:
            assert state["Q<0>"] == expected


def test_registered_output_requires_clocked_macro_and_has_distinct_sample_node():
    spec = BromSpec(4, 2, registered_output=True)
    with pytest.raises(ValueError, match="clocked"):
        macro_circuit(spec, [0, 1, 2, 3])
    with pytest.raises(TypeError):
        BromSpec(4, 1, registered_output=1)
    c = self_timed_macro_circuit(spec, [0, 1, 2, 3])
    assert c.output_register["edge"] == "rising"
    assert (
        c.pins
        == self_timed_macro_circuit(
            replace(spec, registered_output=False), [0, 1, 2, 3]
        ).pins
    )
    by_name = {d.name: d for d in c.devices}
    for bit in range(2):
        names = dict(c.placement_groups)[f"output_latch_{bit}"]
        assert by_name[names[4]].nodes[0] == f"SAMPLED<{bit}>"
    assert max(len(m) for _, m in c.placement_groups) <= 8


def waveform():
    t = np.arange(5201, dtype=float)
    spec = BromSpec(4, 1, registered_output=True)
    words = [0, 1, 0, 1]
    traces = {
        n: np.zeros_like(t)
        for n in ("CLK", "REG_M", "REG_MB", "REG_S", "REG_SB", "SAMPLED<0>", "Q[0]")
    }
    traces["CLK"][:] = 1
    traces["REG_S"][:] = 1
    traces["REG_MB"][:] = 1
    for i, word in enumerate(words):
        base = i * 1000
        traces["CLK"][base + 200 : base + 750] = 0
        traces["REG_M"][base + 220 : base + 760] = 1
        traces["REG_MB"][base + 225 : base + 765] = 0
        traces["REG_S"][base + 205 : base + 780] = 0
        traces["REG_SB"][base + 210 : base + 785] = 1
        traces["SAMPLED<0>"][base + 450 :] = word
        traces["Q[0]"][base + 810 :] = word
    return t * 1e-12, traces, spec, words


def test_registered_checker_accepts_and_rejects_contract_violations():
    from chipforge_asap7.brom.register_timing import analyze_registered_output

    t, traces, spec, words = waveform()
    r = analyze_registered_output(t, traces, spec, words, 1)
    assert r["passed"], r["failures"]
    assert r["minimum_data_before_edge_ps"] > 0
    assert 50 < r["maximum_clk_to_valid_ps"] < 100
    traces["Q[0]"][1400] = 1
    assert not analyze_registered_output(t, traces, spec, words, 1)["passed"]


@pytest.mark.parametrize(
    "net,index,value", [("REG_S", 300, 1), ("SAMPLED<0>", 1750, 0), ("Q[0]", 1900, 0)]
)
def test_registered_checker_rejects_overlap_late_data_and_wrong_capture(
    net, index, value
):
    from chipforge_asap7.brom.register_timing import analyze_registered_output

    t, traces, spec, words = waveform()
    traces[net][index] = value
    assert not analyze_registered_output(t, traces, spec, words, 1)["passed"]


@pytest.mark.parametrize("clk", [0, 1])
@pytest.mark.parametrize("enable,complement", [(0, 0), (1, 0), (1, 1), (0, 1)])
def test_register_waits_for_both_receiving_gate_controls(clk, enable, complement):
    # Split only the acknowledgment inputs to represent independently delayed
    # receiving rails. Driving a local complement is not an acknowledgment.
    c = output_register_circuit()
    sense_groups = {
        "register_master_off",
        "register_master_closed",
        "register_slave_off",
        "register_slave_closed",
    }
    sense = {
        member
        for name, members in c.placement_groups
        if name in sense_groups
        for member in members
    }
    rails = {"REG_M", "REG_MB", "REG_S", "REG_SB"}
    c.devices = [
        replace(d, nodes=(d.nodes[0], "RETURN_" + d.nodes[1], *d.nodes[2:]))
        if d.name in sense and d.nodes[1] in rails
        else d
        for d in c.devices
    ]
    inputs = {
        "CLK": clk,
        "D<0>": 0,
        "RETURN_REG_M": 0,
        "RETURN_REG_MB": 1,
        "RETURN_REG_S": 0,
        "RETURN_REG_SB": 1,
    }
    opposite = "REG_M" if clk else "REG_S"
    inputs["RETURN_" + opposite] = enable
    inputs["RETURN_" + opposite + "B"] = complement
    state = settle(c, inputs, ideal_controls=("REG_M", "REG_S"))
    assert state["REG_S" if clk else "REG_M"] == int(enable == 0 and complement == 1)
    assert state["REG_M" if clk else "REG_S"] == 0


def test_interpolated_phase_handoff_catches_between_sample_overlap():
    from chipforge_asap7.brom.register_timing import analyze_registered_output

    t, traces, spec, words = waveform()
    traces["REG_S"][:220] = 1
    traces["REG_SB"][:220] = 0
    traces["REG_MB"][220:225] = 0
    r = analyze_registered_output(t, traces, spec, words, 1)
    # Adjacent samples show opposite open latches, but the interpolated rails
    # overlap between samples. A pointwise Boolean test alone would miss it.
    assert not r["phase_overlap_intervals"]
    assert not r["passed"]
    assert r["minimum_nonoverlap_ps"] == pytest.approx(-0.6)
    assert any("negative falling" in f for f in r["failures"])
