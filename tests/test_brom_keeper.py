"""Matched feedback keepers: topology, quantized sizing, physical verification."""

import json
from dataclasses import replace

import gdspy
import numpy as np
import pytest

from chipforge_asap7.brom import (
    BromSpec,
    build_macro,
    macro_circuit,
    self_timed_macro_circuit,
)
from chipforge_asap7.brom.compiler import output_circuit


@pytest.mark.parametrize("fins,stack", [(1, 1), (1, 2), (2, 3), (3, 4)])
def test_data_and_replica_use_matching_keeper_chains(fins, stack):
    spec = BromSpec(8, 2, keeper_fins=fins, keeper_stack=stack)
    c = self_timed_macro_circuit(spec, [i % 4 for i in range(8)])
    by_name = {d.name: d for d in c.devices}
    groups = dict(c.placement_groups)
    for prefix, bl, out in [
        ("readout", "BL", "DOUT"),
        ("replica_readout", "RBL", "RDOUT"),
    ]:
        for col in range(spec.columns):
            devices = [by_name[n] for n in groups[f"{prefix}_{col}"]]
            assert len(devices) == 3 + stack
            source = "VDD"
            for d in devices[3:]:
                assert d.spec.flavor == "p" and d.spec.fins == fins
                assert d.nodes[1:] == (f"{out}<{col}>", source, "VDD")
                source = d.nodes[0]
            assert source == f"{bl}<{col}>"
    assert (
        c.array.mapping
        == macro_circuit(
            replace(spec, keeper_fins=0), [i % 4 for i in range(8)]
        ).array.mapping
    )
    assert f"_k{fins}s{stack}" in c.name


def test_disabled_default_and_standalone_configuration():
    c = output_circuit()
    assert len(c.devices) == 3 and c.name == "brom_output_f2_rvt"
    c = output_circuit(4, "slvt", keeper_fins=1, keeper_stack=2)
    assert len(c.devices) == 5
    assert [d.spec.fins for d in c.devices] == [4, 4, 4, 1, 1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"keeper_fins": True},
        {"keeper_fins": 4},
        {"keeper_stack": 0},
        {"keeper_stack": 5},
    ],
)
def test_invalid_keeper_config(kwargs):
    with pytest.raises((ValueError, TypeError)):
        BromSpec(4, 1, **kwargs)
    with pytest.raises((ValueError, TypeError)):
        output_circuit(**kwargs)


@pytest.mark.parametrize(
    "spec",
    [
        BromSpec(4, 2, keeper_fins=1),
        BromSpec(8, 1, fins=1, keeper_fins=1, keeper_stack=4),
        BromSpec(16, 1, ways=4, fins=3, keeper_fins=2, keeper_stack=1),
    ],
)
def test_keeper_macro_drc_lvs(spec, tmp_path, asap7_drc, require_klayout):
    from chipforge_asap7.verification.lvs import run_lvs
    from chipforge_asap7.verification.lvs_report import summarize_lvsdb

    c = self_timed_macro_circuit(
        spec, [i % (1 << spec.bits) for i in range(spec.depth)]
    )
    lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    top = build_macro(c, lib=lib)
    assert asap7_drc(lib, top, tag=c.name) == []
    gds, ref = tmp_path / "macro.gds", tmp_path / "ref.spice"
    lib.write_gds(str(gds))
    ref.write_text(c.lvs_reference())
    result = run_lvs(gds, ref, tmp_path / "lvs", cell_name=c.name)
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


def test_keeper_cli_metadata(tmp_path):
    from chipforge_asap7.brom.__main__ import main

    words = tmp_path / "words.json"
    words.write_text("[0,1,2,3]")
    main(
        [
            str(words),
            "--bits",
            "2",
            "--self-timed",
            "--keeper-fins",
            "1",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    report = json.loads((tmp_path / "out/manifest.json").read_text())
    assert report["keeper"] == {
        "fins": 1,
        "series_devices": 2,
        "replica_matched": True,
        "timing_qualified": False,
    }
    assert report["artifacts"]["output"]["transistors"] == 5


def synthetic(read=True):
    t = np.arange(2001, dtype=float)
    bl = np.full_like(t, 0.7)
    dout = np.zeros_like(t)
    cur = np.zeros_like(t)
    sel = np.zeros_like(t)
    if read:
        sel[401:] = 0.7
        bl[420:] = 0
        dout[430:] = 0.7
        cur[401:430] = 1e-6
    return np.column_stack([t * 1e-12, bl, dout, cur, sel])


def test_checker_rejects_incomplete_and_detects_read_failure():
    from chipforge_asap7.brom.keeper_timing import analyze_keeper

    a = synthetic()
    r = analyze_keeper(a, read=True, vdd=0.7, stop_ps=2000)
    assert r["read_passed"] and r["peak_keeper_contention_ua"] == pytest.approx(1)
    assert r["keeper_contention_charge_fc"] > 0
    a[500:, 1] = 0.7
    assert not analyze_keeper(a, read=True, vdd=0.7, stop_ps=2000)["read_passed"]
    with pytest.raises(ValueError, match="incomplete"):
        analyze_keeper(a[:100], read=True, vdd=0.7, stop_ps=2000)


def test_retention_censoring_and_threshold_loss():
    from chipforge_asap7.brom.keeper_timing import analyze_keeper

    a = synthetic(False)
    r = analyze_keeper(a, read=False, vdd=0.7, stop_ps=2000)
    assert r["retention_90_ns"] is None
    a[1000:, 1] = 0.2
    r = analyze_keeper(a, read=False, vdd=0.7, stop_ps=2000)
    assert r["retention_90_ns"] is not None and r["retention_50_ns"] is not None


def test_checker_does_not_count_pre_read_decay_as_a_successful_read():
    from chipforge_asap7.brom.keeper_timing import analyze_keeper

    a = synthetic()
    a[350:400, 1] = 0
    a[350:400, 2] = 0.7
    r = analyze_keeper(a, read=True, vdd=0.7, stop_ps=2000)
    assert not r["read_passed"] and not r["restored_before_read"]
