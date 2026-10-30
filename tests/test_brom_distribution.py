"""Electrical fanout, logic polarity, and physical sized-driver integration."""

import json
from collections import defaultdict

import gdspy
import pytest
from test_brom import settle

from chipforge_asap7.brom import (
    BromSpec,
    DistributionSpec,
    build_macro,
    macro_circuit,
    self_timed_macro_circuit,
    size_control_distribution,
)
from chipforge_asap7.brom.circuit import Circuit


def gate_loads(c):
    loads = defaultdict(int)
    for d in c.devices:
        loads[d.nodes[1]] += d.spec.total_fins
    return loads


@pytest.mark.parametrize("count", [1, 9, 33, 129])
def test_taper_bounds_load_and_preserves_polarity(count):
    c = Circuit("fanout", ("A", "VDD", "VSS"))
    c.inv("A", "control")
    for i in range(count):
        c.inv("control", f"y{i}")
    size_control_distribution(c)
    for value in (0, 1):
        state = settle(c, {"A": value})
        assert state["control"] == 1 - value
        assert all(state[f"y{i}"] == value for i in range(count))
    loads = gate_loads(c)
    for record in c.distribution["nets"]:
        stages = record["stages"]
        assert len(stages) % 2 == 0
        assert loads[stages[0]["input"]] <= 4 * record["original_driver_fins"]
        for stage in stages:
            assert stage["output_load_gate_fins"] == loads[stage["output"]]
            assert stage["stage_effort"] <= 4
            assert loads[stage["output"]] <= 4 * 4 * stage["parallel_inverters"]
    assert bool(c.distribution["nets"]) == (count > 4)
    with pytest.raises(ValueError, match="already"):
        size_control_distribution(c)


def test_external_phase_input_is_buffered_without_driving_pin():
    c = macro_circuit(BromSpec(8, 16), [0] * 8)
    old_array = [d for d in c.devices if d.name in c.array.device_names]
    size_control_distribution(c)
    assert old_array == [d for d in c.devices if d.name in c.array.device_names]
    records = {r["net"]: r for r in c.distribution["nets"]}
    for net in ("PRECHN", "SAMPLE"):
        assert records[net]["external_input"]
        assert records[net]["receiver_net"] != net
        assert all(net not in (d.nodes[0], d.nodes[2]) for d in c.devices)
    assert max(len(m) for _, m in c.placement_groups) <= 8


def test_all_generated_stage_loads_match_actual_netlist():
    c = self_timed_macro_circuit(BromSpec(8, 16), [0] * 8)
    old = gate_loads(c)
    size_control_distribution(c)
    records = {r["net"]: r for r in c.distribution["nets"]}
    assert {
        "PRECHN",
        "EVAL",
        "SAMPLE",
        "SEL<0>",
        "C<0,0>",
        "RC<2,15>",
    } <= records.keys()
    for net, record in records.items():
        assert record["load_gate_fins"] == old[net]
        assert all(d.spec.fins == 2 and d.spec.fingers == 1 for d in c.devices)


@pytest.mark.parametrize(
    "spec", [BromSpec(4, 2), BromSpec(8, 1, fins=1), BromSpec(16, 1, ways=4, fins=3)]
)
def test_sized_macro_drc_lvs(spec, tmp_path, asap7_drc, require_klayout):
    from chipforge_asap7.verification.lvs import run_lvs
    from chipforge_asap7.verification.lvs_report import summarize_lvsdb

    c = size_control_distribution(
        self_timed_macro_circuit(
            spec, [i % (1 << spec.bits) for i in range(spec.depth)]
        )
    )
    assert c.distribution["nets"]
    lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    top = build_macro(c, lib=lib)
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


def test_cli_reports_sizing(tmp_path):
    from chipforge_asap7.brom.__main__ import main

    words = tmp_path / "words.json"
    words.write_text("[0,1,2,3]")
    main(
        [
            str(words),
            "--bits",
            "2",
            "--self-timed",
            "--sized-controls",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    report = json.loads((tmp_path / "out/manifest.json").read_text())
    assert report["control_distribution"]["nets"]
    assert not report["control_distribution"]["wire_rc_included"]
    assert not report["timing_qualified"]


@pytest.mark.parametrize("value", [1, True, 2.5])
def test_bad_stage_effort(value):
    with pytest.raises(ValueError):
        DistributionSpec(value)
