"""Switch-level, physical DRC/LVS, view, and CLI checks for pruned BROM macros."""

from __future__ import annotations

import json

import gdspy
import pytest

from chipforge_asap7.brom import (
    BromSpec,
    build_macro,
    build_pruned_macro,
    macro_circuit,
    pruned_macro_circuit,
)
from chipforge_asap7.cli import main as asap7
from chipforge_asap7.brom.views import macro_lef, macro_verilog
from chipforge_asap7.verification.drc import run_gdscheck
from chipforge_asap7.verification.lvs import run_lvs
from chipforge_asap7.verification.lvs_report import summarize_lvsdb

from test_brom import settle


@pytest.mark.parametrize("ways", [1, 4])
def test_pruned_macro_reads_and_holds_every_four_bit_pattern(ways):
    spec = BromSpec(8 * ways, 16, ways, fins=2, vt="lvt")
    words = [
        sum((((k + (a % ways)) % 16) >> ((a // ways) % 4) & 1) << k for k in range(16))
        for a in range(spec.depth)
    ]
    c = pruned_macro_circuit(spec, words)
    state = None
    held = None
    for address, expected in enumerate(words):
        pins = {f"A[{i}]": address >> i & 1 for i in range(spec.address_bits)}
        state = settle(c, {**pins, "PRECHN": 0, "EVAL": 0, "SAMPLE": 0}, state)
        if held is not None:
            assert sum(state[f"Q[{i}]"] << i for i in range(spec.bits)) == held
        assert all(state[f"BL<{i}>"] == 1 for i in range(spec.columns))
        state = settle(c, {**pins, "PRECHN": 1, "EVAL": 1, "SAMPLE": 0}, state)
        noise = {key: 1 - value for key, value in pins.items()}
        state = settle(c, {**noise, "PRECHN": 1, "EVAL": 1, "SAMPLE": 1}, state)
        held = sum(state[f"Q[{i}]"] << i for i in range(spec.bits))
        assert held == expected
        state = settle(c, {**pins, "PRECHN": 1, "EVAL": 1, "SAMPLE": 0}, state)
        state = settle(c, {**pins, "PRECHN": 1, "EVAL": 0, "SAMPLE": 0}, state)


def test_pruned_macro_reduces_area_and_transistors_vs_unpruned():
    spec = BromSpec(16, 8, 1, fins=2, vt="lvt")
    words = [((i * 0x9E3779B1) >> 16) & 0xFF for i in range(16)]
    base_c = macro_circuit(spec, words)
    pruned_c = pruned_macro_circuit(spec, words)
    assert len(pruned_c.devices) < len(base_c.devices)

    base_top = build_macro(base_c)
    pruned_top = build_pruned_macro(pruned_c)
    b_box = base_top.get_bounding_box()
    p_box = pruned_top.get_bounding_box()
    base_area = (b_box[1, 0] - b_box[0, 0]) * (b_box[1, 1] - b_box[0, 1])
    pruned_area = (p_box[1, 0] - p_box[0, 0]) * (p_box[1, 1] - p_box[0, 1])
    assert pruned_area < 0.65 * base_area

    lef = macro_lef(pruned_c, pruned_top)
    verilog = macro_verilog(pruned_c, spec, words)
    assert f"MACRO {pruned_c.name}" in lef
    assert f"module {pruned_c.name} " in verilog


@pytest.mark.parametrize(
    "spec,words",
    [
        (
            BromSpec(16, 8, 1, fins=2, vt="lvt"),
            [0] * 16,
        ),
        (
            BromSpec(24, 6, 1, fins=2, vt="rvt"),
            [((i * 37 + 13) & 0x3F) for i in range(24)],
        ),
        (
            BromSpec(32, 8, 4, fins=2, vt="lvt"),
            [((i * 73 + 5) & 0xFF) for i in range(32)],
        ),
        (
            # Adversarial block-symmetric all-16-candidates-per-block stress test
            BromSpec(16, 16, 1, fins=2, vt="lvt"),
            [
                sum(((bit >> off) & 1) << bit for bit in range(16))
                for _b in range(4)
                for off in range(4)
            ],
        ),
        (
            # Three fins move M2 track 0 next to the k = 0 VSS strap; all five
            # M6-lifted tracks are in use, none of which may be track 0.
            BromSpec(16, 16, 1, fins=3, vt="lvt"),
            [
                sum(((bit >> off) & 1) << bit for bit in range(16))
                for _b in range(4)
                for off in range(4)
            ],
        ),
    ],
)
def test_pruned_macro_drc_and_unit_fin_lvs(tmp_path, spec, words):
    c = pruned_macro_circuit(spec, words)
    lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    top = build_macro(c, lib=lib)
    gds = tmp_path / "macro.gds"
    ref = tmp_path / "macro.lvs.spice"
    lib.write_gds(str(gds))
    ref.write_text(c.lvs_reference())

    violations = run_gdscheck(gds, tmp_path / "drc", cell_name=top.name)
    assert not violations

    lvs = run_lvs(
        gds,
        ref,
        tmp_path / "lvs",
        cell_name=top.name,
        merge_fin_diffusion=True,
        timeout=30,
    )
    assert lvs.matched
    summary = summarize_lvsdb(lvs.report)
    renamed = [p for cir in summary["circuits"] for p in cir["renamed_pins"]]
    assert not renamed


def test_cli_pruned_mode_drc_and_lvs(tmp_path):
    words_file = tmp_path / "words.json"
    words_file.write_text(json.dumps([0, 3, 5, 12, 9, 6, 15, 0]))
    out = tmp_path / "out"
    asap7(
        [
            "rom",
            "brom",
            "--words-json",
            str(words_file),
            "--word-bits",
            "4",
            "--vt",
            "lvt",
            "--pruned",
            "--drc",
            "--lvs",
            "--output-dir",
            str(out),
        ]
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["layout"] == "pruned_dense_cmos"
    assert (manifest["architecture"], manifest["word_bits"]) == ("brom", 4)
    assert manifest["abstract"] is None
    assert set(manifest["artifacts"]) == {"macro"}
    macro = manifest["artifacts"]["macro"]
    assert macro["drc"] == {}
    assert macro["lvs"] == "passed"
    assert (out / "macro.lef").is_file()
    assert (out / "macro.v").is_file()


@pytest.mark.parametrize("ways", [1, 4])
def test_pruned_wide_word_bank_takes_the_bit_tile_path(ways):
    """The bank's wide-word placers find bits and columns by unit name.

    Under other names a 148-bit pruned bank fell back to flat rows and came
    out 1.7-1.85x the unpruned macro's area.
    """
    from chipforge_asap7.brom.dense_bank import SLICE_FIRST_BITS
    from chipforge_asap7.brom.dense_slices import _datapaths
    from chipforge_asap7.brom.dense_units import dense_units

    spec = BromSpec(4 * ways, SLICE_FIRST_BITS, ways, fins=2, vt="lvt")
    words = [(0x9E3779B97F4A7C15 * (i + 1)) & ((1 << spec.bits) - 1) for i in range(spec.depth)]
    c = pruned_macro_circuit(spec, words)
    excluded = set(c.array.device_names) | set(c.pruned_cgen_devices)
    units = dense_units(c, exclude=excluded)
    assert sum(u.name.startswith("output_latch_") for u in units) == spec.bits
    anchors = {f"BL<{col}>": [108 * col] for col in range(spec.columns)}
    found = _datapaths(units, anchors)
    assert found is not None
    assert found[0] == ways
    assert set(found[1]) == set(range(spec.bits))
