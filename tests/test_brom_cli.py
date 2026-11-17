"""`asap7-brom` end to end: what each mode writes and records, and what it refuses.

Each run builds every top's GDS, so the options share runs: the self-timed
one also carries sizing, a keeper and the registered output.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from chipforge_asap7.brom.__main__ import main


def compile_words(tmp_path, *flags):
    words = tmp_path / "words.json"
    words.write_text("[0, 1, 2, 3]")
    out = tmp_path / "out"
    main([str(words), "--bits", "2", *flags, "--out", str(out)])
    return out, json.loads((out / "manifest.json").read_text())


BLAS_PROBE = """
import os, chipforge_asap7.brom
print(os.environ["OPENBLAS_NUM_THREADS"], len(os.listdir("/proc/self/task")))
"""


def blas_probe(**overrides):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    }
    done = subprocess.run(
        [sys.executable, "-c", BLAS_PROBE],
        env=env | overrides,
        capture_output=True,
        text=True,
        check=True,
    )
    setting, threads = done.stdout.split()
    return setting, int(threads)


@pytest.mark.skipif(not Path("/proc/self/task").exists(), reason="needs Linux /proc")
def test_import_pins_blas_to_one_thread_unless_overridden():
    # A thread count, not just the variable: it proves the pin beat NumPy's load.
    assert blas_probe() == ("1", 1)
    setting, threads = blas_probe(OPENBLAS_NUM_THREADS="4")
    assert setting == "4" and threads >= 4


def test_external_phases(tmp_path):
    out, r = compile_words(tmp_path)
    assert r["timing_qualified"] is False
    assert r["controller"] == "external_phases"
    assert r["layout"] == "dense_cmos_split_selector"
    assert r["periphery_layout"] == "shared_diffusion_cmos"
    assert len(r["physical_plan"]["bottom_units"]) > 1
    assert r["array_core_nm"] == [216, 432]
    assert r["isolation_devices"] == 2
    assert r["selector_map"] == [[10, 12]]
    assert set(r["artifacts"]) == {"macro", "cgen", "decoder", "output"}
    for role, artifact in r["artifacts"].items():
        assert artifact["drc"] == artifact["lvs"] == "not_run"
        assert (out / artifact["gds"]).is_file()
        assert (out / artifact["cdl"]).is_file()
        assert (out / f"{role}.lvs.spice").is_file()
        certificates = artifact["routing_certificates"]
        if role == "cgen":
            assert not certificates  # Fixed lane routing has no channel solve.
        else:
            assert certificates
        for cert in certificates:
            assert cert["track_count"] == max(cert["tracks"]) + 1
            assert (
                len(cert["clique_witness"])
                <= cert["lower_bound"]
                <= cert["track_count"]
            )
            assert cert["optimal"] == (cert["lower_bound"] == cert["track_count"])
            assert cert["method"] in ("greedy", "sweep", "cp_sat")
            assert cert["solver_status"] in (None, "OPTIMAL", "FEASIBLE", "UNKNOWN")
    macro = r["artifacts"]["macro"]
    physical = r["physical_plan"]
    assert macro["routing_certificates"] == physical["routing_certificates"]
    assert {c["channel"] for c in macro["routing_certificates"]} == {"row", "header"}
    # Every certificate describes wires in the emitted plan, with the right
    # keepout removed and each track translated into its actual M4 height.
    routes = {(n, layer, tuple(a), tuple(b)) for n, layer, a, b in physical["routes"]}
    for cert in macro["routing_certificates"]:
        for net, track, (a, b) in zip(
            cert["nets"], cert["tracks"], cert["intervals_nm"]
        ):
            y = cert["track_y_nm"] + cert["pitch_nm"] * track
            assert (net, cert["layer"], (a, y), (b - cert["end_space_nm"], y)) in routes
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
    # Each half of a 2-bit macro has one column: no dummy loads are needed.
    assert r["replica"]["columns"] == 1 and r["replica"]["blocks"] == 2
    assert r["replica"]["candidate_gates"] == 1 and r["replica"]["dummy_loads"] == 0
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
    assert r["replica"]["latches_per_sample_receiver"] == 1


def test_one_sample_receiver_can_drive_several_latches(tmp_path):
    _, r = compile_words(
        tmp_path, "--self-timed", "--registered-output", "--replica-share", "2"
    )
    assert r["replica"]["latches_per_sample_receiver"] == 2
    assert r["top_cell"].endswith("_s2")


@pytest.mark.parametrize(
    "flags",
    [
        ["--replica-guard-pairs", "4"],  # replica options need --self-timed
        ["--replica-share", "2"],
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



def test_gt2n_flag_names_through_the_asap7_command(tmp_path):
    """`asap7 rom brom` takes chipforge_gt2n's names, builds what the old
    `asap7-brom` names build, and records the shape keys gt2n's manifests carry."""
    from chipforge_asap7.cli import main as asap7

    words = tmp_path / "words.json"
    words.write_text(json.dumps(list(range(4)) * 4))
    old, new = tmp_path / "old", tmp_path / "new"
    main([str(words), "--bits", "2", "--ways", "4", "--out", str(old)])
    status = asap7(
        ["rom", "brom", "--words-json", str(words), "--word-bits", "2"]
        + ["--column-mux", "4", "--output-dir", str(new)]
    )
    before = json.loads((old / "manifest.json").read_text())
    after = json.loads((new / "manifest.json").read_text())
    assert status == 0
    assert after["physical_sha256"] == before["physical_sha256"]
    assert {
        key: after[key]
        for key in (
            "architecture",
            "platform",
            "depth",
            "num_words",
            "word_bits",
            "physical_word_bits",
            "column_mux",
            "column_mux_in_macro",
            "address_bits",
        )
    } == {
        "architecture": "brom",
        "platform": "asap7",
        "depth": 16,
        "num_words": 16,
        "word_bits": 2,
        "physical_word_bits": 8,
        "column_mux": 4,
        "column_mux_in_macro": True,
        "address_bits": 4,
    }


def test_the_words_file_is_given_once(tmp_path, capsys):
    words = tmp_path / "words.json"
    words.write_text("[0, 1, 2, 3]")
    for argv in (
        ["--bits", "2", "--out", str(tmp_path / "a")],
        [str(words), "--words-json", str(words), "--bits", "2", "--out", str(tmp_path / "b")],
    ):
        with pytest.raises(SystemExit):
            main(argv)
    assert "--words-json" in capsys.readouterr().err
