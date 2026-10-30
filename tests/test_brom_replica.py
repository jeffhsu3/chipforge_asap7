"""Independent structural, routing and electrical checks for replica integration."""

import json

import gdspy
import pytest

from chipforge_asap7.brom import (
    BromSpec,
    ReplicaSpec,
    build_macro,
    self_timed_macro_circuit,
)
from chipforge_asap7.brom.layout import (
    macro_periphery,
    macro_plan,
    selector_placements,
    verify_macro_routes,
)


@pytest.mark.parametrize("ways,depth", [(1, 8), (4, 32)])
def test_replica_loads_and_public_interface(ways, depth):
    spec = BromSpec(depth, 2, ways=ways)
    c = self_timed_macro_circuit(spec, [i % 4 for i in range(depth)])
    assert set(c.pins) == {
        *(f"A[{i}]" for i in range(spec.address_bits)),
        "CLK",
        "RESETN",
        "Q[0]",
        "Q[1]",
        "VDD",
        "VSS",
    }
    replica = c.replica
    assert len(replica.selector.mapping) == spec.blocks + 1
    assert all(len(row) == spec.columns for row in replica.selector.mapping)
    assert set(replica.selector.mapping[-1]) == {15}
    assert all(set(row) == {0} for row in replica.selector.mapping[:-1])
    devs = [d for d in c.devices if d.name in replica.selector.device_names]
    for col in range(spec.columns):
        column = [d for d in devs if d.nodes[0] == f"RBL<{col}>"]
        assert len(column) == 2 * (spec.blocks + 1)
        assert sum(d.nodes[1] != "VSS" for d in column) == 1
    assert {d.name for d in macro_periphery(c).devices}.isdisjoint(
        replica.selector.device_names | c.array.device_names
    )
    placements = list(selector_placements(c))
    assert len(placements) == 2
    assert placements[1][2] > spec.columns * 108 + 108
    assert placements[1][3] > spec.blocks * 432 + 108


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


@pytest.mark.parametrize(
    "spec", [BromSpec(4, 2), BromSpec(8, 1, fins=1), BromSpec(16, 1, ways=4, fins=3)]
)
def test_replica_macro_drc_lvs(spec, tmp_path, asap7_drc, require_klayout):
    from chipforge_asap7.verification.lvs import run_lvs
    from chipforge_asap7.verification.lvs_report import summarize_lvsdb

    c = self_timed_macro_circuit(
        spec, [i % (1 << spec.bits) for i in range(spec.depth)]
    )
    lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    top = build_macro(c, lib=lib)
    assert len(top.references) == 3
    assert asap7_drc(lib, top, tag=top.name) == []
    gds, ref = tmp_path / "macro.gds", tmp_path / "macro.spice"
    lib.write_gds(str(gds))
    ref.write_text(c.lvs_reference())
    result = run_lvs(gds, ref, tmp_path / "lvs", cell_name=top.name)
    assert result.matched, result.log.read_text()
    assert not any(
        r["renamed_pins"] for r in summarize_lvsdb(result.report)["circuits"]
    )
    pins = {
        line.removeprefix("* pin ").strip()
        for line in result.extracted_netlist.read_text().splitlines()
        if line.startswith("* pin ")
    }
    assert pins == set(c.pins)


def test_disconnected_replica_bitline_is_rejected():
    from chipforge_asap7.layout.layers import LAYERS

    c = self_timed_macro_circuit(BromSpec(4, 1), [0, 1, 0, 1])
    top = build_macro(c)
    plan = macro_plan(c)
    i = next(i for i, p in enumerate(plan.ports) if p.net == "RBL<0>")
    x, y = plan.trunk_x[i], plan.route_y[i]
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
    with pytest.raises(ValueError, match="disconnected macro route"):
        verify_macro_routes(c, top)


def test_replica_cli(tmp_path):
    from chipforge_asap7.brom.__main__ import main

    words = tmp_path / "words.json"
    words.write_text("[0,1,2,3]")
    out = tmp_path / "out"
    main([str(words), "--bits", "2", "--self-timed", "--out", str(out)])
    r = json.loads((out / "manifest.json").read_text())
    assert r["replica_integrated"] and not r["timing_qualified"]
    assert not r["replica"]["feedback_timing_qualified"]
    assert r["replica"]["columns"] == 2 and r["replica"]["blocks"] == 2
    assert "READ_READY" not in r["pins"]
    assert "controller" in r["artifacts"]


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
    assert not any(line.startswith("B") for line in deck.splitlines())


def test_replica_cli_options_require_self_timed(tmp_path):
    from chipforge_asap7.brom.__main__ import main

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "unused.json",
                "--bits",
                "1",
                "--out",
                str(tmp_path),
                "--replica-guard-pairs",
                "4",
            ]
        )


def test_replica_precharge_deadline_is_before_next_request():
    from chipforge_asap7.brom.replica_timing import analyze_replica

    t, traces, spec, words = synthetic_replica_waveform()
    traces["PRECHN"][930:1060] = 1
    result = analyze_replica(t, traces, spec, words, 1)
    assert result["passed"], result["failures"]
    traces["PRECHN"][1060:1190] = 1
    result = analyze_replica(t, traces, spec, words, 1)
    assert not result["passed"]
