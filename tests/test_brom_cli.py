"""`asap7-brom` end to end: what each mode writes and records, and what it refuses.

Each run builds every top's GDS, so the options share runs: the self-timed
one also carries sizing, a keeper and the registered output.
"""

import json

import pytest

from chipforge_asap7.brom.__main__ import main


def compile_words(tmp_path, *flags):
    words = tmp_path / "words.json"
    words.write_text("[0, 1, 2, 3]")
    out = tmp_path / "out"
    main([str(words), "--bits", "2", *flags, "--out", str(out)])
    return out, json.loads((out / "manifest.json").read_text())


def test_external_phases(tmp_path):
    out, r = compile_words(tmp_path)
    assert r["timing_qualified"] is False
    assert r["controller"] == "external_phases"
    assert r["layout"] == "compact_selector_array"
    assert r["periphery_layout"] == "folded_local_rows"
    assert r["periphery_plan"]["row_count"] > 1
    assert r["array_core_nm"] == [216, 432]
    assert r["isolation_devices"] == 2
    assert r["selector_map"] == [[10, 12]]
    assert set(r["artifacts"]) == {"macro", "cgen", "decoder", "output"}
    for role, artifact in r["artifacts"].items():
        assert artifact["drc"] == artifact["lvs"] == "not_run"
        assert (out / artifact["gds"]).is_file()
        assert (out / artifact["cdl"]).is_file()
        assert (out / f"{role}.lvs.spice").is_file()
    macro = r["artifacts"]["macro"]
    assert (out / macro["lef"]).read_text().startswith("VERSION 5.8")
    assert f"module {macro['cell']} " in (out / macro["verilog"]).read_text()


def test_clocked_keeps_external_feedback(tmp_path):
    _, r = compile_words(tmp_path, "--clocked")
    assert "controller" in r["artifacts"]
    assert r["controller"] == "external_completion_handshake"
    assert not r["replica_integrated"] and not r["timing_qualified"]
    assert "CLK" in r["pins"] and "READ_READY" in r["pins"]
    assert "PRECHN" not in r["pins"]


def test_self_timed_with_sizing_keeper_and_register(tmp_path):
    _, r = compile_words(
        tmp_path,
        "--self-timed",
        "--sized-controls",
        "--keeper-fins",
        "1",
        "--registered-output",
    )
    assert r["controller"] == "replica_feedback"
    assert r["replica_integrated"] and not r["timing_qualified"]
    assert not r["replica"]["feedback_timing_qualified"]
    assert r["replica"]["columns"] == 2 and r["replica"]["blocks"] == 2
    assert "READ_READY" not in r["pins"]
    assert r["control_distribution"]["nets"]
    assert not r["control_distribution"]["wire_rc_included"]
    assert r["keeper"] == {
        "fins": 1,
        "series_devices": 2,
        "replica_matched": True,
        "timing_qualified": False,
    }
    assert r["artifacts"]["output"]["transistors"] == 5
    assert r["registered_output"]
    assert not r["output_register"]["setup_hold_qualified"]
    assert {"controller", "output_register"} <= set(r["artifacts"])


@pytest.mark.parametrize(
    "flags",
    [
        ["--replica-guard-pairs", "4"],  # replica options need --self-timed
        ["--registered-output"],  # the register needs a clock
        ["--sized-controls", "--wire-rc-controls"],
    ],
)
def test_rejected_option_combinations(tmp_path, flags):
    words = tmp_path / "words.json"
    words.write_text("[0, 1, 2, 3]")
    with pytest.raises(SystemExit, match="2"):
        main([str(words), "--bits", "2", *flags, "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_bad_timing_report_is_a_usage_error(tmp_path):
    words = tmp_path / "words.json"
    words.write_text("[0, 1, 2, 3]")
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({"schema": "brom_timing_estimate_v1"}))
    for report in (tmp_path / "missing.json", stale):
        with pytest.raises(SystemExit, match="2"):
            main(
                [str(words), "--bits", "2", "--self-timed"]
                + ["--timing-report", str(report), "--out", str(tmp_path / "out")]
            )
        assert not (tmp_path / "out").exists()
