"""Controller phase ordering, integration and physical connectivity."""

import pytest
from test_brom import settle

from chipforge_asap7.brom import (
    BromSpec,
    clocked_macro_circuit,
    controller_circuit,
)


def inputs(**changes):
    return {
        "CLK": 1,
        "RESETN": 1,
        "PRECHN_FAR": 0,
        "READ_READY": 0,
        "SAMPLE_CLOSED": 1,
        "CANDIDATE_BUSY": 0,
        **changes,
    }


def phases(state):
    return tuple(state[n] for n in ("PRECHN", "EVAL", "SAMPLE"))


@pytest.mark.parametrize("cancel_with_reset", [False, True])
def test_controller_waits_for_each_acknowledgment(cancel_with_reset):
    c = controller_circuit()
    pins = inputs(RESETN=0)
    state = settle(c, pins)
    assert phases(state) == (0, 0, 0)
    steps = [
        ({"RESETN": 1}, (0, 0, 0)),
        ({"CLK": 0}, (1, 0, 0)),
        ({"PRECHN_FAR": 1}, (1, 1, 0)),
        ({"CANDIDATE_BUSY": 1}, (1, 1, 0)),
        ({"READ_READY": 1, "SAMPLE_CLOSED": 0}, (1, 1, 1)),
        ({"RESETN": 0} if cancel_with_reset else {"CLK": 1}, (1, 1, 0)),
        ({"SAMPLE_CLOSED": 1}, (1, 0, 0)),
        ({"CANDIDATE_BUSY": 0}, (0, 0, 0)),
        ({"PRECHN_FAR": 0, "READ_READY": 0}, (0, 0, 0)),
    ]
    for change, expected in steps:
        pins.update(change)
        state = settle(c, pins, state)
        assert phases(state) == expected


def test_controller_can_abort_before_read_ready():
    c = controller_circuit()
    state = settle(c, inputs(CLK=0, PRECHN_FAR=1, CANDIDATE_BUSY=1))
    assert phases(state) == (1, 1, 0)
    state = settle(c, inputs(PRECHN_FAR=1, CANDIDATE_BUSY=1), state)
    assert phases(state) == (1, 0, 0)
    state = settle(c, inputs(), state)
    assert phases(state) == (0, 0, 0)


@pytest.mark.parametrize("ways,depth", [(1, 8), (4, 16)])
def test_clocked_macro_reads_and_holds(ways, depth):
    spec = BromSpec(depth, 2, ways=ways)
    words = [i % 4 for i in range(depth)]
    c = clocked_macro_circuit(spec, words)
    assert not {"PRECHN", "EVAL", "SAMPLE"} & set(c.pins)
    state = None
    for address, word in enumerate(words):
        pins = {
            **inputs(),
            **{f"A[{i}]": address >> i & 1 for i in range(spec.address_bits)},
        }
        state = settle(c, pins, state, ideal_controls=("PRECHN", "SAMPLE"))
        pins["CLK"] = 0
        state = settle(c, pins, state, ideal_controls=("PRECHN", "SAMPLE"))
        pins.update(PRECHN_FAR=1, CANDIDATE_BUSY=1)
        state = settle(c, pins, state, ideal_controls=("PRECHN", "SAMPLE"))
        pins.update(READ_READY=1, SAMPLE_CLOSED=0)
        state = settle(c, pins, state, ideal_controls=("PRECHN", "SAMPLE"))
        assert sum(state[f"Q[{i}]"] << i for i in range(spec.bits)) == word
        pins["CLK"] = 1
        state = settle(c, pins, state, ideal_controls=("PRECHN", "SAMPLE"))
        assert phases(state) == (1, 1, 0)
        pins["SAMPLE_CLOSED"] = 1
        state = settle(c, pins, state, ideal_controls=("PRECHN", "SAMPLE"))
        pins["CANDIDATE_BUSY"] = 0
        state = settle(c, pins, state, ideal_controls=("PRECHN", "SAMPLE"))
        assert sum(state[f"Q[{i}]"] << i for i in range(spec.bits)) == word


def controller_waveform():
    import numpy as np

    from chipforge_asap7.brom.controller import CONTROLLER_INPUTS, CONTROLLER_OUTPUTS

    t = np.arange(6201, dtype=float)
    traces = {
        net: np.zeros_like(t)
        for net in (*CONTROLLER_INPUTS, *CONTROLLER_OUTPUTS, "Q[0]", "Q[1]")
    }
    traces["SAMPLE_CLOSED"][:] = 1
    for cycle in range(5):
        base = 1200 * cycle
        intervals = {
            "PRECHN": (250, 1120),
            "EVAL": (450, 1030 if cycle < 4 else 600),
            "CANDIDATE_BUSY": (470, 1100),
            "PRECHN_FAR": (400, 1180),
            "READ_READY": (650, 1120),
        }
        if cycle < 4:
            intervals["SAMPLE"] = (680, 810 if cycle == 2 else 890)
            traces["SAMPLE_CLOSED"][(t >= base + 700) & (t < base + 1000)] = 0
        for net, (lo, hi) in intervals.items():
            traces[net][(t >= base + lo) & (t < base + hi)] = 1
        if cycle < 4:
            for bit in range(2):
                traces[f"Q[{bit}]"][t >= base + 800] = cycle >> bit & 1
    return t * 1e-12, traces


def test_analog_checker_accepts_ordered_waveform():
    from chipforge_asap7.brom.controller_timing import check_controller_waveform

    t, traces = controller_waveform()
    assert check_controller_waveform(t, traces, 1)["passed"]


@pytest.mark.parametrize(
    "net,at,value",
    [
        ("PRECHN", 500, 0),
        ("EVAL", 960, 0),
        ("SAMPLE", 200, 1),
        ("READ_READY", 800, 0),
        ("Q[0]", 5800, 0),
    ],
)
def test_analog_checker_rejects_phase_or_data_fault(net, at, value):
    from chipforge_asap7.brom.controller_timing import check_controller_waveform

    t, traces = controller_waveform()
    traces[net][at] = value
    assert not check_controller_waveform(t, traces, 1)["passed"]
