"""Rising-edge output contract, physical register integration and timing checks."""

import json
from dataclasses import replace

import gdspy
import numpy as np
import pytest
from test_brom import settle

from chipforge_asap7.brom import (
    BromSpec,
    build_macro,
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


@pytest.mark.parametrize(
    "spec",
    [
        BromSpec(4, 2),
        BromSpec(8, 1, fins=1),
        BromSpec(16, 1, ways=4, fins=3, keeper_fins=1),
    ],
)
def test_registered_macro_drc_lvs(spec, tmp_path, asap7_drc, require_klayout):
    from chipforge_asap7.verification.lvs import run_lvs
    from chipforge_asap7.verification.lvs_report import summarize_lvsdb

    spec = replace(spec, registered_output=True)
    c = self_timed_macro_circuit(
        spec, [i % (1 << spec.bits) for i in range(spec.depth)]
    )
    lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    top = build_macro(c, lib=lib)
    assert asap7_drc(lib, top, tag=c.name) == []
    gds, ref = tmp_path / "macro.gds", tmp_path / "ref.spice"
    lib.write_gds(str(gds))
    ref.write_text(c.lvs_reference())
    r = run_lvs(gds, ref, tmp_path / "lvs", cell_name=c.name)
    assert r.matched, r.log.read_text()
    assert not any(d["renamed_pins"] for d in summarize_lvsdb(r.report)["circuits"])
    pins = {
        line.removeprefix("* pin ").strip()
        for line in r.extracted_netlist.read_text().splitlines()
        if line.startswith("* pin ")
    }
    assert pins == set(c.pins)


def test_cli_exposes_register_mode_and_rejects_external_phases(tmp_path):
    from chipforge_asap7.brom.__main__ import main

    words = tmp_path / "words.json"
    words.write_text("[0,1,2,3]")
    args = [
        str(words),
        "--bits",
        "2",
        "--registered-output",
        "--out",
        str(tmp_path / "out"),
    ]
    with pytest.raises(SystemExit):
        main(args)
    main([*args, "--self-timed"])
    r = json.loads((tmp_path / "out/manifest.json").read_text())
    assert r["registered_output"] and not r["output_register"]["setup_hold_qualified"]
    assert "output_register" in r["artifacts"]


def waveform():
    t = np.arange(5201, dtype=float)
    spec = BromSpec(4, 1, registered_output=True)
    words = [0, 1, 0, 1]
    traces = {
        n: np.zeros_like(t)
        for n in ("CLK", "REG_M", "REG_MB", "REG_S", "REG_SB", "SAMPLED<0>", "Q<0>")
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
        traces["Q<0>"][base + 810 :] = word
    return t * 1e-12, traces, spec, words


def test_registered_checker_accepts_and_rejects_contract_violations():
    from chipforge_asap7.brom.register_timing import analyze_registered_output

    t, traces, spec, words = waveform()
    r = analyze_registered_output(t, traces, spec, words, 1)
    assert r["passed"], r["failures"]
    assert r["minimum_data_before_edge_ps"] > 0
    assert 50 < r["maximum_clk_to_valid_ps"] < 100
    traces["Q<0>"][1400] = 1
    assert not analyze_registered_output(t, traces, spec, words, 1)["passed"]


@pytest.mark.parametrize(
    "net,index,value", [("REG_S", 300, 1), ("SAMPLED<0>", 1750, 0), ("Q<0>", 1900, 0)]
)
def test_registered_checker_rejects_overlap_late_data_and_wrong_capture(
    net, index, value
):
    from chipforge_asap7.brom.register_timing import analyze_registered_output

    t, traces, spec, words = waveform()
    traces[net][index] = value
    assert not analyze_registered_output(t, traces, spec, words, 1)["passed"]
