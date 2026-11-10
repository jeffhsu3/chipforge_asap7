"""`asap7-rom --package`: what it writes, what it checks and what images share.

Two images of one shape must differ only in the V1s on their data bit sites,
since a flow hardens against the abstract and swaps each unit's image in.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices.rom_bitcell import RomArraySpec
from chipforge_asap7.devices.rom_decoder import RomDecoderSpec, build_rom, main
from chipforge_asap7.devices.rom_netlist import rom_macro_ports
from chipforge_asap7.devices.rom_package import (
    frozen_geometry_sha256,
    programming_digest,
    programming_sites,
    rom_image_name,
    sites_from_words,
    write_rom_package,
)
from chipforge_asap7.devices.rom_timing import REPORT_FORMAT, circuit_digest
from chipforge_asap7.layout.layers import LAYERS

WORDS, BITS = 16, 6
V1 = (LAYERS["V1"]["layer"], LAYERS["V1"]["datatype"])
SHAPE = (
    "--sense", "--enable", "--latch", "--sample", "--control", "--register",
    "--edge_pins",
)  # fmt: skip
IMAGE = [(i * 5 + 3) % 2**BITS for i in range(WORDS)]


def _spec(words: int = WORDS, bits: int = BITS) -> RomDecoderSpec:
    return RomDecoderSpec(
        array=RomArraySpec(words, bits),
        sense=True,
        enable=True,
        latch=True,
        sample=True,
        control=True,
        register=True,
        edge_pins=True,
    )


def _compile(
    tmp_path: Path, name: str, words: list[int], *flags: str, bits: int = BITS
):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(words))
    out = tmp_path / name
    main(
        [
            "--array.words", str(len(words)), "--array.bits", str(bits), *SHAPE,
            "--words", str(path), "--package", str(out), *flags,
        ]
    )  # fmt: skip
    return out, json.loads((out / "manifest.json").read_text())


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("rom_package")
    return (
        _compile(tmp_path, "zero", [0] * WORDS),
        _compile(tmp_path, "image", IMAGE),
        tmp_path,
    )


def _library(path: Path) -> gdspy.GdsLibrary:
    return gdspy.GdsLibrary(infile=str(path))


def _flat(path: Path, cell: str) -> dict[tuple[int, int], set[bytes]]:
    """One written cell's flattened shapes, independent of the fingerprint."""
    polygons = _library(path).cells[cell].get_polygons(by_spec=True)
    return {
        layer: {(p.round(1) + 0.0).tobytes() for p in shapes}
        for layer, shapes in polygons.items()
    }


def _built(sites=()):
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return build_rom(_spec(), list(sites), name="top", lib=library)


def _own(cell) -> tuple:
    """A cell's own shapes, labels and instances, children compared by name."""
    shapes = sorted(
        (layer, datatype, (points.round(1) + 0.0).tobytes())
        for polygon in cell.polygons
        for layer, datatype, points in zip(
            polygon.layers, polygon.datatypes, polygon.polygons
        )
    )
    labels = sorted((x.text, x.layer, tuple(x.position)) for x in cell.labels)
    references = sorted(
        (r.ref_cell.name, tuple(r.origin), r.rotation or 0, bool(r.x_reflection),
         getattr(r, "columns", 1), getattr(r, "rows", 1))
        for r in cell.references
    )  # fmt: skip
    return shapes, labels, references


# ── Words and names ──────────────────────────────────────────────────────────
def test_words_set_their_sites():
    spec = _spec(words=16)
    words = [0] * 16
    words[3], words[15] = 0b101, 0b100000
    assert sites_from_words(words, spec) == [(3, 0), (3, 2), (15, 5)]


@pytest.mark.parametrize(
    "words, error",
    [
        ({"words": []}, "JSON array"),
        ([0] * 15, "15 words for a 16-word"),
        ([0] * 15 + [1 << BITS], "does not fit"),
        ([0] * 15 + [-1], "does not fit"),
        ([0] * 15 + ["1"], "not an integer"),
        ([0] * 15 + [True], "not an integer"),
    ],
)
def test_words_that_are_not_an_image_are_refused(words, error):
    with pytest.raises((TypeError, ValueError), match=error):
        sites_from_words(words, _spec())


def test_the_digest_names_the_contents_alone():
    assert programming_digest([(1, 2), (0, 3)]) == programming_digest(
        [(0, 3), (1, 2), (0, 3)]
    )
    assert programming_digest([(1, 2)]) != programming_digest([(2, 1)])
    spec = _spec()
    assert rom_image_name(spec, [(1, 2)]).startswith(f"{spec.macro_name}_p")


# ── The fingerprint ──────────────────────────────────────────────────────────
def test_the_fingerprint_ignores_programming():
    spec = _spec()
    assert frozen_geometry_sha256(spec, _built()) == frozen_geometry_sha256(
        spec, _built([(0, 0), (7, 5), (8, 2), (15, 5)])
    )


def test_no_frozen_v1_sits_on_a_data_site():
    """Otherwise the fingerprint would miss that V1 going away."""
    polygons = _built().get_polygons(by_spec=True)[V1]
    centres = {
        tuple(((p.min(axis=0) + p.max(axis=0)) / 2).round(1).tolist()) for p in polygons
    }
    assert len(centres) > 16  # The replica row and the periphery.
    assert not centres & programming_sites(_spec())


@pytest.mark.parametrize(
    "layer, box",
    [
        ("V1", (0, 0, 18, 18)),  # A V1 off every site is not programming.
        ("M2", (0, 0, 18, 18)),
    ],
)
def test_the_fingerprint_sees_anything_else(layer, box):
    spec = _spec()
    top = _built()
    before = frozen_geometry_sha256(spec, top)
    x0, y0, x1, y1 = box
    top.add(
        gdspy.Rectangle(
            (x0, y0), (x1, y1),
            layer=LAYERS[layer]["layer"], datatype=LAYERS[layer]["datatype"],
        )
    )  # fmt: skip
    assert frozen_geometry_sha256(spec, top) != before


def test_the_fingerprint_sees_a_label_move():
    spec = _spec()
    top = _built()
    before = frozen_geometry_sha256(spec, top)
    top.labels[0].position = top.labels[0].position + (1, 0)
    assert frozen_geometry_sha256(spec, top) != before


def test_the_fingerprint_sees_the_replica_row():
    """A replica V1 sits on a bit site of its own, and no image may change it."""
    spec = _spec()
    top = _built()
    before = frozen_geometry_sha256(spec, top)
    half = next(r.ref_cell for r in top.references if r.ref_cell.name == "top_left")
    x, y = spec.half.site(0, spec.replica_bit)

    def on_the_site(polygon) -> bool:
        p = polygon.polygons[0]
        centre = (p.min(axis=0) + p.max(axis=0)) / 2
        return polygon.layers[0] == V1[0] and tuple(centre.round(1)) == (
            round(x, 1),
            round(y, 1),
        )

    kept = [polygon for polygon in half.polygons if not on_the_site(polygon)]
    assert len(kept) == len(half.polygons) - 1
    half.polygons = kept
    assert frozen_geometry_sha256(spec, top) != before


# ── Two images, one abstract ─────────────────────────────────────────────────
def test_images_share_an_abstract(images):
    (zero_dir, zero), (image_dir, image), _ = images
    assert zero["top_cell"] != image["top_cell"]
    assert zero["abstract"] == image["abstract"]
    assert zero["abstract"]["programming_layers"] == {"V1": list(V1)}
    for name in ("abstract.lef", "abstract.gds"):
        assert (zero_dir / name).read_bytes() == (image_dir / name).read_bytes()


def test_the_abstract_carries_the_fields_a_swap_compares(images):
    """``cell``, the fingerprints and the layers: core refuses without them."""
    (_, zero), _, _ = images
    assert {
        "cell", "lef", "gds", "lef_sha256", "frozen_geometry_sha256", "programming_layers",
    } == set(zero["abstract"])  # fmt: skip


def test_abstract_views_carry_the_stable_name(images):
    (zero_dir, zero), _, _ = images
    cell = zero["abstract"]["cell"]
    assert cell == _spec().macro_name
    lef = (zero_dir / "abstract.lef").read_text()
    assert f"MACRO {cell}\n" in lef and f"FOREIGN {cell} " in lef
    assert cell in {c.name for c in _library(zero_dir / "abstract.gds").top_level()}
    # The content-named views keep their own name.
    top = zero["top_cell"]
    assert f"MACRO {top}\n" in (zero_dir / "macro.lef").read_text()
    assert top in {c.name for c in _library(zero_dir / "macro.gds").top_level()}
    assert f"module {top} " in (zero_dir / "macro.v").read_text()
    assert f".subckt {top} " in (zero_dir / "macro.cdl").read_text()


def test_images_differ_by_exactly_their_programmed_vias(images):
    (zero_dir, zero), (image_dir, image), _ = images
    empty = _flat(zero_dir / "macro.gds", zero["top_cell"])
    programmed = _flat(image_dir / "macro.gds", image["top_cell"])
    assert empty.keys() == programmed.keys()
    assert [k for k in empty if empty[k] != programmed[k]] == [V1]
    assert not empty[V1] - programmed[V1]
    assert len(programmed[V1] - empty[V1]) == sum(w.bit_count() for w in IMAGE)


def test_the_abstract_is_the_macro_with_no_bits_set(images):
    (zero_dir, zero), _, _ = images
    assert _flat(zero_dir / "abstract.gds", zero["abstract"]["cell"]) == _flat(
        zero_dir / "macro.gds", zero["top_cell"]
    )


def test_images_only_name_apart_what_programming_changes(images):
    """A chip merges every unit's image, so a shared name must be one cell."""
    (zero_dir, _), (image_dir, image), _ = images
    a = _library(zero_dir / "macro.gds").cells
    b = _library(image_dir / "macro.gds").cells
    top = image["top_cell"]
    assert set(b) - set(a) == {top, f"{top}_left", f"{top}_right"}
    for name in a.keys() & b.keys():
        assert _own(a[name]) == _own(b[name]), name


def test_a_different_shape_has_a_different_abstract(images):
    (_, zero), _, tmp_path = images
    _, other = _compile(tmp_path, "narrow", [0] * WORDS, bits=BITS - 2)
    assert other["abstract"]["cell"] != zero["abstract"]["cell"]
    assert (
        other["abstract"]["frozen_geometry_sha256"]
        != zero["abstract"]["frozen_geometry_sha256"]
    )


def test_one_input_gives_one_package_in_any_process(tmp_path):
    """Images are compiled apart, so nothing may follow Python's hash seed."""
    words = tmp_path / "words.json"
    words.write_text(json.dumps(IMAGE))
    outs = []
    for seed in ("1", "2"):
        out = tmp_path / seed
        subprocess.run(
            [
                sys.executable, "-c",
                "from chipforge_asap7.devices.rom_decoder import main; main()",
                "--array.words", str(WORDS), "--array.bits", str(BITS), *SHAPE,
                "--words", str(words), "--package", str(out),
            ],
            env=os.environ | {"PYTHONHASHSEED": seed},
            check=True,
            capture_output=True,
        )  # fmt: skip
        outs.append(out)
    names = sorted(p.name for p in outs[0].iterdir())
    assert names == sorted(p.name for p in outs[1].iterdir())
    for name in names:
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes(), name


# ── The manifest ─────────────────────────────────────────────────────────────
def test_the_manifest_describes_the_macro(images):
    _, (out, r), _ = images
    spec = _spec()
    assert r["pins"] == list(rom_macro_ports(spec))
    assert r["timing_qualified"] is False
    assert r["registered_output"] is True
    assert r["read"] == {
        "contract": "two_phase",
        "latency": 1,
        "addr_hold_edge": "fall",
        "reset": {"pin": "RESETN", "active": "low"},
    }
    assert r["spec"]["array"]["words"] == WORDS
    assert r["programming"] == {
        "digest": programming_digest(sites_from_words(IMAGE, spec)),
        "bits_set": sum(w.bit_count() for w in IMAGE),
        "bits": WORDS * BITS,
    }
    macro = r["artifacts"]["macro"]
    assert macro["drc"] == macro["lvs"] == "not_run"
    assert macro["liberty"] == {
        "tt": "macro_tt.lib", "ff": "macro_ff.lib", "ss": "macro_ss.lib",
    }  # fmt: skip
    for key in ("gds", "cdl", "lef", "verilog"):
        assert (out / macro[key]).is_file()
    for filename in macro["liberty"].values():
        assert (out / filename).is_file()
    assert f"{macro['transistors']} transistors" in (out / "macro.cdl").read_text()
    assert macro["width_nm"] > 0 and macro["height_nm"] > 0
    assert (macro["width_nm"], macro["height_nm"]) == pytest.approx(
        tuple(
            float(v) * 1000
            for v in next(
                line.split()[1::2]
                for line in (out / "macro.lef").read_text().splitlines()
                if line.strip().startswith("SIZE")
            )
        )
    )


def test_a_flow_through_macro_reads_on_the_rising_edge(tmp_path):
    path = tmp_path / "words.json"
    path.write_text(json.dumps(IMAGE))
    flags = [f if f != "--register" else "--flow-through" for f in SHAPE]
    main(
        [
            "--array.words", str(WORDS), "--array.bits", str(BITS), *flags,
            "--words", str(path), "--package", str(tmp_path / "out"),
        ]
    )  # fmt: skip
    r = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert r["registered_output"] is True
    assert r["read"] == {
        "contract": "edge_flow_through",
        "latency": 1,
        "addr_hold_edge": "rise",
        "reset": {"pin": "RESETN", "active": "low"},
    }
    assert r["interface"].startswith("CLK rising closes the address latches")
    assert set(r["artifacts"]["macro"]["liberty"]) == {"tt", "ff", "ss"}
    lib = (tmp_path / "out" / "macro_tt.lib").read_text()
    assert "setup_rising" in lib and "hold_rising" in lib
    assert "setup_falling" not in lib


def test_a_latched_macro_has_no_liberty(tmp_path):
    path = tmp_path / "words.json"
    path.write_text(json.dumps(IMAGE))
    flags = [f for f in SHAPE if f != "--register"]
    main(
        [
            "--array.words", str(WORDS), "--array.bits", str(BITS), *flags,
            "--words", str(path), "--package", str(tmp_path / "out"),
        ]
    )  # fmt: skip
    r = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert r["registered_output"] is False
    assert r["read"] is None
    assert r["artifacts"]["macro"]["liberty"] == {}
    assert not list((tmp_path / "out").glob("*.lib"))


def test_a_failed_build_leaves_no_old_manifest(tmp_path, monkeypatch):
    out, _ = _compile(tmp_path, "image", IMAGE)

    def broken(*_args, **_kwargs):
        raise RuntimeError("no netlist")

    monkeypatch.setattr("chipforge_asap7.devices.rom_netlist.rom_macro_cdl", broken)
    with pytest.raises(RuntimeError, match="no netlist"):
        _compile(tmp_path, "image", IMAGE)
    assert not (out / "manifest.json").exists()


@pytest.mark.parametrize(
    "flags, error",
    [
        (["--drc"], "--package"),
        (["--lvs"], "--package"),
        (["--package", "x", "--out", "y"], "replaces --out"),
        (["--package", "x", "--cdl", "y"], "replaces --out"),
        (["--package", "x", "--views", "y"], "replaces --out"),
    ],
)
def test_the_command_line_refuses_what_it_cannot_do(flags, error, capsys, tmp_path):
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "--array.words", str(WORDS), "--array.bits", str(BITS), *SHAPE,
                *[str(tmp_path / f) if f in "xy" else f for f in flags],
            ]
        )  # fmt: skip
    assert caught.value.code == 2
    assert error in capsys.readouterr().err


def test_a_words_file_that_cannot_be_read_is_a_usage_error(capsys, tmp_path):
    for path, content in ((tmp_path / "missing.json", None), (tmp_path / "bad", "[0,")):
        if content is not None:
            path.write_text(content)
        with pytest.raises(SystemExit) as caught:
            main(
                [
                    "--array.words", str(WORDS), "--array.bits", str(BITS), *SHAPE,
                    "--words", str(path), "--package", str(tmp_path / "out"),
                ]
            )  # fmt: skip
        assert caught.value.code == 2
    assert not (tmp_path / "out").exists()


def test_a_package_needs_edge_pins(capsys, tmp_path):
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "--array.words", str(WORDS), "--array.bits", str(BITS),
                "--package", str(tmp_path),
            ]
        )  # fmt: skip
    assert caught.value.code == 2
    assert "--edge_pins" in capsys.readouterr().err


def test_bad_words_are_a_usage_error(capsys, tmp_path):
    path = tmp_path / "words.json"
    path.write_text("[0, 0]")
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "--array.words", str(WORDS), "--array.bits", str(BITS), *SHAPE,
                "--words", str(path), "--package", str(tmp_path / "out"),
            ]
        )  # fmt: skip
    assert caught.value.code == 2
    assert "2 words for a 16-word ROM" in capsys.readouterr().err


# ── The checks ───────────────────────────────────────────────────────────────
def test_the_checks_pass_and_are_recorded(tmp_path, require_klayout, external_tool):
    from chipforge_asap7.verification.drc import find_gdscheck, run_gdscheck

    try:
        find_gdscheck()
        found = True
    except FileNotFoundError:
        found = False
    external_tool(found, "gdscheck is not installed (set GDSCHECK)")
    out, r = _compile(tmp_path, "checked", IMAGE, "--drc", "--lvs")
    macro = r["artifacts"]["macro"]
    assert macro["drc"] == {}
    assert macro["lvs"] == "passed"
    assert macro["renamed_pins"] == macro["missing_pins"] == macro["extra_pins"] == []
    # The labelled copy is only the compare's.
    assert not any(
        label.text.startswith(("AL_F_", "AR_F_"))
        for label in _library(out / "macro.gds").cells[r["top_cell"]].get_labels()
    )
    # A hardened block holds the abstract until its image is swapped in.
    assert not run_gdscheck(
        out / "abstract.gds", tmp_path / "abstract_drc", cell_name=r["abstract"]["cell"]
    )


@pytest.mark.parametrize(
    "bits, strap_every",
    [
        (8, 8),  # One lane row is too few tracks, so more straps add a row.
        (12, 16),  # The bitlines leave a track for the Q columns.
        (15, 8),  # More straps, and the latch row's rail just off the VSS bar.
    ],
)
def test_narrow_widths_pass_their_checks(
    bits, strap_every, tmp_path, require_klayout, external_tool
):
    from chipforge_asap7.verification.drc import find_gdscheck

    try:
        find_gdscheck()
        found = True
    except FileNotFoundError:
        found = False
    external_tool(found, "gdscheck is not installed (set GDSCHECK)")
    words = [(i * 5 + 3) % 2**bits for i in range(WORDS)]
    _, r = _compile(tmp_path, "narrow", words, "--drc", "--lvs", bits=bits)
    assert r["strap_every"] == strap_every
    assert r["artifacts"]["macro"]["drc"] == {}
    assert r["artifacts"]["macro"]["lvs"] == "passed"


def test_a_netlist_that_disagrees_fails_the_run(tmp_path, monkeypatch, require_klayout):
    from chipforge_asap7.devices import rom_netlist

    real = rom_netlist.rom_macro_cdl

    def missing_a_bit(spec, programmed=(), *, name):
        return real(spec, sorted(programmed)[1:], name=name)

    monkeypatch.setattr(rom_netlist, "rom_macro_cdl", missing_a_bit)
    with pytest.raises(SystemExit) as caught:
        _compile(tmp_path, "bad", IMAGE, "--lvs")
    assert "physical checks failed" in str(caught.value.code)
    r = json.loads((tmp_path / "bad" / "manifest.json").read_text())
    assert r["artifacts"]["macro"]["lvs"] == "failed"


def test_an_lvs_that_times_out_fails_the_run(tmp_path, monkeypatch):
    def out_of_time(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("klayout", 1)

    monkeypatch.setattr("chipforge_asap7.verification.lvs.run_lvs", out_of_time)
    with pytest.raises(SystemExit) as caught:
        _compile(tmp_path, "slow", IMAGE, "--lvs")
    assert "physical checks failed" in str(caught.value.code)
    r = json.loads((tmp_path / "slow" / "manifest.json").read_text())
    assert r["artifacts"]["macro"]["lvs"] == "timed_out"


def test_a_drc_violation_fails_the_run(tmp_path, monkeypatch):
    from chipforge_asap7.verification import drc

    monkeypatch.setattr(
        drc,
        "run_gdscheck",
        lambda *_a, **_k: [drc.DRCViolation("M1.W.1", (0.0, 0.0, 1.0, 1.0))],
    )
    with pytest.raises(SystemExit) as caught:
        _compile(tmp_path, "bad", IMAGE, "--drc")
    assert "physical checks failed" in str(caught.value.code)
    r = json.loads((tmp_path / "bad" / "manifest.json").read_text())
    assert r["artifacts"]["macro"]["drc"] == {"M1.W.1": 1}


@pytest.mark.parametrize("words, bits, register", [(16, 3, True), (128, 7, False)])
def test_latch_row_vss_strap_clears_the_lanes_vdd_strap(words, bits, register):
    """The two climb past the array's top edge together; 18 nm apart they touched,
    which shorted VDD to VSS."""
    spec = RomDecoderSpec(
        array=RomArraySpec(words, bits),
        sense=True,
        enable=True,
        latch=True,
        sample=True,
        control=True,
        register=register,
        edge_pins=True,
    )
    assert spec.latch_tap_shift == 1
    gap = spec.latch_strap_xs["VSS"] - spec.sense_lane.strap_xs["VDD"]
    assert abs(gap) >= 54


def test_macro_with_a_moved_latch_tap_is_clean(tmp_path: Path, require_klayout):
    spec = RomDecoderSpec(
        array=RomArraySpec(16, 3),
        sense=True,
        enable=True,
        latch=True,
        sample=True,
        control=True,
        register=True,
        edge_pins=True,
    )
    sites = [(w, b) for w in range(16) for b in range(3) if (w + b) % 2]
    assert not write_rom_package(spec, sites, tmp_path, drc=True, lvs=True)


def _timing_report(path: Path, spec: RomDecoderSpec) -> Path:
    corner = {
        "clk_q_rise": [90.0, 100.0, 130.0],
        "clk_q_fall": [80.0, 90.0, 120.0],
        "q_rise_slew": [10.0, 20.0, 50.0],
        "q_fall_slew": [9.0, 19.0, 49.0],
        "setup": -20.0,
        "hold": 30.0,
        "min_low": 150.0,
        "min_high": 60.0,
        "droop": 1.0,
        "max_low": 200000,
        "aborted": 0,
    }
    path.write_text(
        json.dumps(
            {
                "format": REPORT_FORMAT,
                "macro": spec.macro_name,
                "circuit_sha256": circuit_digest(spec),
                "corners": {"typical": corner},
                "failed": {"fast": "did not converge"},
            }
        )
    )
    return path


def _small(words: int = 16, bits: int = 6) -> RomDecoderSpec:
    return RomDecoderSpec(
        array=RomArraySpec(words, bits),
        sense=True,
        enable=True,
        latch=True,
        sample=True,
        control=True,
        register=True,
        edge_pins=True,
    )


def test_package_takes_measured_corners_and_says_which(tmp_path: Path):
    spec = _small()
    report = _timing_report(tmp_path / "timing.json", spec)
    out = tmp_path / "pkg"
    assert not write_rom_package(spec, [], out, timing=report)
    assert "SIMULATED, NOT SIGN-OFF" in (out / "macro_tt.lib").read_text()
    assert "ESTIMATED" in (out / "macro_ff.lib").read_text()
    manifest = json.loads((out / "manifest.json").read_text())
    timing = manifest["timing"]
    assert manifest["timing_qualified"] is False
    assert timing["corners"]["typical"] == {"source": "spice", "max_low_ps": 200000}
    assert timing["corners"]["fast"] == {"source": "estimate"}
    assert len(timing["report_sha256"]) == 64


def test_package_without_a_report_records_no_measurement(tmp_path: Path):
    assert not write_rom_package(_small(), [], tmp_path)
    timing = json.loads((tmp_path / "manifest.json").read_text())["timing"]
    assert timing["report_sha256"] is None
    assert {c["source"] for c in timing["corners"].values()} == {"estimate"}


def test_package_refuses_another_shapes_report(tmp_path: Path):
    report = _timing_report(tmp_path / "timing.json", _small(32, 6))
    with pytest.raises(ValueError, match="measured"):
        write_rom_package(_small(), [], tmp_path / "pkg", timing=report)
    assert not (tmp_path / "pkg" / "manifest.json").exists()


def _argv(*extra: str) -> list[str]:
    return [
        "--array.words", "16", "--array.bits", "6", "--sense", "--enable", "--latch",
        "--sample", "--control", "--edge_pins", *extra,
    ]  # fmt: skip


def test_characterize_needs_the_register(tmp_path: Path, capsys):
    with pytest.raises(SystemExit):
        main(_argv("--characterize", str(tmp_path / "t.json")))
    assert "--register" in capsys.readouterr().err


def test_timing_needs_something_to_apply_to(tmp_path: Path, capsys):
    with pytest.raises(SystemExit):
        main(_argv("--register", "--timing", str(tmp_path / "t.json")))
    assert "--timing is for" in capsys.readouterr().err
