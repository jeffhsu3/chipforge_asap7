"""Independent structural, routing and electrical checks for replica integration."""

import gdspy
import pytest

from chipforge_asap7.brom import (
    BromSpec,
    ReplicaSpec,
    build_macro,
    self_timed_macro_circuit,
)
from chipforge_asap7.brom.dense_units import dense_units
from chipforge_asap7.brom.dense_macro import (
    bank_circuit,
    dense_macro_layout,
    verify_dense_routes,
)


@pytest.mark.parametrize("ways,depth,bits", [(1, 8, 2), (4, 32, 2), (1, 8, 32)])
def test_replica_loads_and_public_interface(ways, depth, bits):
    spec = BromSpec(depth, bits, ways=ways)
    c = self_timed_macro_circuit(spec, [i % 4 for i in range(depth)])
    assert set(c.pins) == {
        *(f"A[{i}]" for i in range(spec.address_bits)),
        "CLK",
        "RESETN",
        *(f"Q[{i}]" for i in range(bits)),
        "VDD",
        "VSS",
    }
    replica = c.replica
    rows = spec.blocks + 1
    # One full-height column, the observed bitline. Its candidate is live
    # only in the last block; every block keeps a selector and isolation gate.
    assert replica.selector.mapping == ((0,),) * (rows - 1) + ((15,),)
    assert replica.port_nets["BL<0>"] == "RBL"
    candidate = f"RC<{rows - 1},15>"
    assert replica.port_nets[f"C<{rows - 1},15>"] == candidate
    strip = [d for d in c.devices if d.name in replica.selector.device_names]
    assert len(strip) == 2 * rows
    assert all(d.nodes[0] == "RBL" and d.nodes[2] == "VSS" for d in strip)
    assert sorted(d.nodes[1] for d in strip) == [candidate] + ["VSS"] * (2 * rows - 1)
    # The candidate carries as many gates as one real half's candidate:
    # the live one, and dummy gates in bank cells with drain and source on
    # VSS, each the selector's size.
    assert replica.dummy_loads == spec.half_columns - 1
    dummies = [
        d
        for d in c.devices
        if d.nodes[1] == candidate and d.name not in replica.selector.device_names
        and d.nodes[0] == d.nodes[2] == "VSS"
    ]
    assert len(dummies) == replica.dummy_loads
    assert all(d.spec.fins == c.array.plan.fins for d in dummies)
    # No device has every terminal on VSS: they add only solver nodes.
    assert not [d for d in c.devices if set(d.nodes) == {"VSS"}]
    groups = dict(c.placement_groups)
    loads = [n for n in groups if n.startswith("replica_candidate_load_")]
    assert sum(len(groups[n]) for n in loads) == replica.dummy_loads
    units = {u.name: u for u in dense_units(bank_circuit(c))}
    assert all(len(units[n].members) == len(groups[n]) for n in loads)
    readouts = [n for n, _ in c.placement_groups if n.startswith("replica_readout")]
    assert readouts == ["replica_readout"]
    assert {d.name for d in bank_circuit(c).devices}.isdisjoint(
        replica.selector.device_names | c.array.device_names
    )
    _, plan = dense_macro_layout(c)
    assert plan["arrays"] and plan["array_y_nm"] > 0
    build_macro(c)  # checks every route


@pytest.mark.parametrize(
    "kwargs", [{"guard_pairs": 0}, {"guard_pairs": True}, {"extra_blocks": 0}]
)
def test_bad_replica_parameters(kwargs):
    with pytest.raises(ValueError):
        ReplicaSpec(**kwargs)


def test_replica_tap_limit_is_enforced():
    spec = BromSpec(276, 1)
    with pytest.raises(ValueError, match="tap row reach"):
        self_timed_macro_circuit(spec, [0] * spec.depth)


def test_disconnected_replica_bitline_is_rejected():
    from chipforge_asap7.layout.layers import LAYERS

    c = self_timed_macro_circuit(BromSpec(4, 1), [0, 1, 0, 1])
    # The bank is a subcell; flatten so its vias can be removed here.
    top = build_macro(c).flatten()
    _, plan = dense_macro_layout(c)
    _, _, (x, y), _ = next(r for r in plan["routes"] if r[0] == "RBL" and r[1] == "V4")
    count = []

    def remove(points, layer, datatype):
        match = (
            layer == LAYERS["V4"]["layer"]
            and datatype == 0
            and points[:, 0].mean() == x
            and points[:, 1].mean() == y
        )
        if match:
            count.append(1)
        return match

    top.remove_polygons(remove)
    assert len(count) == 1
    with pytest.raises(ValueError, match="disconnected dense route"):
        verify_dense_routes(top, plan["checkpoints"])


def test_existing_standalone_array_labels_are_preserved():
    from chipforge_asap7.brom import build_selector_array

    c = self_timed_macro_circuit(BromSpec(4, 1), [0, 1, 0, 1])
    lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    leaf = build_selector_array(c.array.plan, c.array.mapping, lib=lib)
    labels = [p.text for p in leaf.labels]
    top = build_macro(c, lib=lib)
    assert [p.text for p in leaf.labels] == labels
    assert labels
    assert all(not r.ref_cell.labels for r in top.references)


def synthetic_replica_waveform():
    import numpy as np

    spec = BromSpec(4, 1)
    words = [0, 1, 0, 1]
    t = np.arange(5201, dtype=float)
    names = [
        "PRECHN",
        "EVAL",
        "READ_READY",
        "SAMPLE",
        "SAMPLEB",
        "CANDIDATE_BUSY",
        "DATA<0>",
        "BL<0>",
        "Q[0]",
    ]
    names += [f"C<0,{k}>" for k in range(16)]
    names += [f"LC<0,{k}>" for k in range(16)]
    traces = {n: np.zeros_like(t) for n in names}
    traces["SAMPLEB"][:] = 1
    traces["BL<0>"][:] = 1
    for address, word in enumerate(words):
        base = 1000 * address
        for net, lo, hi in [
            ("PRECHN", 230, 930),
            ("EVAL", 270, 870),
            ("READ_READY", 410, 880),
            ("SAMPLE", 450, 790),
            ("CANDIDATE_BUSY", 290, 900),
            ("C<0,15>", 300, 850),
            ("LC<0,15>", 300, 850),
        ]:
            traces[net][(t >= base + lo) & (t < base + hi)] = 1
        traces["SAMPLEB"][(t >= base + 460) & (t < base + 810)] = 0
        if word:
            traces["DATA<0>"][(t >= base + 350) & (t < base + 950)] = 1
            traces["BL<0>"][(t >= base + 330) & (t < base + 960)] = 0
        traces["Q[0]"][t >= base + 500] = word
    return t * 1e-12, traces, spec, words


def test_replica_checker_accepts_valid_sequence():
    from chipforge_asap7.brom.replica_timing import analyze_replica

    t, traces, spec, words = synthetic_replica_waveform()
    result = analyze_replica(t, traces, spec, words, 1)
    assert result["passed"], result["failures"]
    assert min(result["minimum_margins_ps"].values()) > 0


@pytest.mark.parametrize(
    "net,index,value",
    [
        ("SAMPLE", 250, 1),
        ("DATA<0>", 1449, 0),
        ("PRECHN", 500, 0),
        ("Q[0]", 4700, 0),
        ("SAMPLEB", 950, 0),
        ("SAMPLE", 4500, 1),
        ("C<0,15>", 950, 1),
        ("LC<0,15>", 950, 1),
    ],
)
def test_replica_checker_rejects_races_and_bad_reads(net, index, value):
    from chipforge_asap7.brom.replica_timing import analyze_replica

    t, traces, spec, words = synthetic_replica_waveform()
    traces[net][index] = value
    result = analyze_replica(t, traces, spec, words, 1)
    assert not result["passed"]
    assert result["minimum_margins_ps"] == {}


@pytest.mark.parametrize("sized_controls", [False, True])
def test_bench_never_drives_feedback_inputs(tmp_path, sized_controls):
    from chipforge_asap7.brom.replica_timing import render_replica_bench
    from chipforge_asap7.brom.timing import TimingConditions

    spec = BromSpec(4, 2)
    deck, signals = render_replica_bench(
        spec,
        [0, 1, 2, 3],
        ReplicaSpec(),
        TimingConditions(),
        tmp_path / "model.pm",
        tmp_path / "model.osdi",
        sized_controls=sized_controls,
    )
    sources = [line for line in deck.splitlines() if line.startswith("Vstim")]
    assert len(sources) == spec.address_bits + 2
    assert "READ_READY" in signals and "CANDIDATE_BUSY" in signals
    assert "C<0,0>" in signals and "LC<0,0>" in signals
    assert not any(line.startswith("B") for line in deck.splitlines())


def test_replica_precharge_deadline_is_before_next_request():
    from chipforge_asap7.brom.replica_timing import analyze_replica

    t, traces, spec, words = synthetic_replica_waveform()
    traces["PRECHN"][930:1060] = 1
    result = analyze_replica(t, traces, spec, words, 1)
    assert result["passed"], result["failures"]
    traces["PRECHN"][1060:1190] = 1
    result = analyze_replica(t, traces, spec, words, 1)
    assert not result["passed"]
