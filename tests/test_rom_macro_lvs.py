"""The whole NOR ROM macro against its flat unit-fin reference.

The comparison is flat: the reference is, and KLayout 0.30's hierarchical
extractor was seen to run past 20 minutes on a 64x32 macro that flat
extraction matches in 11 s.  The unprogrammed drains are labelled
(`label_floating_drains`), which cuts the compare about tenfold.  The
compare is topological, so a match must also pair each pin with its own
name.  LVS skips when KLayout is missing; ``ASAP7_REQUIRE_TOOLS=1`` makes
that a failure.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import gdspy
import pytest

from chipforge_asap7.devices import RomDecoderSpec, build_rom
from chipforge_asap7.devices.rom_bitcell import RomArraySpec
from chipforge_asap7.verification.lvs import run_lvs
from chipforge_asap7.verification.lvs_report import summarize_lvsdb
from chipforge_asap7.verification.rom_macro_lvs import (
    label_floating_drains,
    rom_macro_reference,
)

CONFIGS: dict[str, dict[str, Any]] = {
    "16x8": {"words": 16, "bits": 8},
    "16x8_no_lane": {"words": 16, "bits": 8, "lane": False},
    "16x8_sensed": {"words": 16, "bits": 8, "sense": True},
    "64x32_latched": {
        "words": 64,
        "bits": 32,
        "sense": True,
        "enable": True,
        "latch": True,
    },
    "64x32_sampled": {
        "words": 64,
        "bits": 32,
        "sense": True,
        "enable": True,
        "latch": True,
        "sample": True,
    },
    "16x6_self_timed": {"words": 16, "bits": 6, "control": True},
    "64x32_self_timed": {"words": 64, "bits": 32, "control": True},
    "16x6_registered": {"words": 16, "bits": 6, "control": True, "register": True},
    "64x32_registered": {"words": 64, "bits": 32, "control": True, "register": True},
}


def _spec(words: int, bits: int, control: bool = False, **flags: Any) -> RomDecoderSpec:
    if control:
        flags = {
            "register": flags.get("register", False),
            "sense": True,
            "enable": True,
            "latch": True,
            "sample": True,
            "control": True,
        }
    return RomDecoderSpec(array=RomArraySpec(words, bits), **flags)


def _sites(spec: RomDecoderSpec) -> set[tuple[int, int]]:
    rng = random.Random(spec.array.words * spec.array.bits)
    words, bits = spec.array.words, spec.array.bits
    return {(w, b) for w in range(words) for b in range(bits) if rng.random() < 0.5}


def _build(
    spec: RomDecoderSpec, sites: set, tmp_path: Path, labelled: set | None = None
) -> tuple[Path, str]:
    """The macro's GDS, its unprogrammed drains labelled as `labelled` has them."""
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    top = build_rom(spec, sites, lib=library)
    if labelled is not None:
        label_floating_drains(spec, labelled, top)
    gds = tmp_path / "rom.gds"
    library.write_gds(str(gds))
    return gds, top.name


def _labels(spec: RomDecoderSpec, sites: set, named: bool) -> set[str]:
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    top = build_rom(spec, sites, lib=library)
    if named:
        label_floating_drains(spec, sites, top)
    return {label.text for label in top.get_labels()}


@pytest.mark.parametrize("named", [False, True], ids=["plain", "named"])
@pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS.keys())
def test_the_pins_are_the_macros_labels(config, named):
    spec = _spec(**config)
    sites = _sites(spec)
    reference = rom_macro_reference(spec, sites, name_floating=named)
    assert set(reference.pins) == _labels(spec, sites, named)


def _lvs(gds: Path, text: str, tmp_path: Path):
    schematic = tmp_path / "reference.spice"
    schematic.write_text(text)
    name = text.split("\n", 2)[1].split()[1]
    return run_lvs(gds, schematic, tmp_path / "lvs", cell_name=name, flat=True)


@pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS.keys())
def test_macro_matches_its_unit_fin_reference(config, tmp_path: Path, require_klayout):
    spec = _spec(**config)
    sites = _sites(spec)
    gds, name = _build(spec, sites, tmp_path, labelled=sites)
    reference = rom_macro_reference(spec, sites, name_floating=True)
    result = _lvs(gds, reference.render(name), tmp_path)
    assert result.matched
    circuits = summarize_lvsdb(result.report)["circuits"]
    assert [c.get("renamed_pins", []) for c in circuits] == [[]] * len(circuits)
    assert result.extracted_netlist.read_text().count("M$") == reference.fins


def test_the_labels_only_speed_the_compare(tmp_path: Path, require_klayout):
    """Without the drain labels the macro matches all the same."""
    spec = _spec(**CONFIGS["16x6_self_timed"])
    sites = _sites(spec)
    gds, name = _build(spec, sites, tmp_path)
    assert _lvs(gds, rom_macro_reference(spec, sites).render(name), tmp_path).matched


def test_a_bit_the_reference_does_not_program_fails(tmp_path: Path, require_klayout):
    """The layout holds one programmed site more than the reference."""
    spec = _spec(**CONFIGS["16x6_self_timed"])
    sites = _sites(spec)
    site = min(sites)
    gds, name = _build(spec, sites, tmp_path, labelled=sites)
    text = rom_macro_reference(spec, sites - {site}, name_floating=True).render(name)
    assert not _lvs(gds, text, tmp_path).matched


def test_a_label_cannot_hide_a_programmed_bit(tmp_path: Path, require_klayout):
    """Labels and reference both miss one programmed site: the label lands on
    a drain its V1 joined to the bitline, and the compare must still fail."""
    spec = _spec(**CONFIGS["16x6_self_timed"])
    sites = _sites(spec)
    site = min(sites)
    gds, name = _build(spec, sites, tmp_path, labelled=sites - {site})
    text = rom_macro_reference(spec, sites - {site}, name_floating=True).render(name)
    assert not _lvs(gds, text, tmp_path).matched


def test_two_words_drivers_crossed_fails(tmp_path: Path, require_klayout):
    """Word 3's driver on word 4's wordline and back must not match."""
    spec = _spec(**CONFIGS["16x6_self_timed"])
    sites = _sites(spec)
    gds, name = _build(spec, sites, tmp_path, labelled=sites)
    reference = rom_macro_reference(spec, sites, name_floating=True)
    lines = reference.render(name).split("\n")
    swap = {"WL<3>": "WL<4>", "WL<4>": "WL<3>"}
    crossed = [
        " ".join(swap.get(t, t) for t in line.split())
        if line.startswith(("MDW3_", "MDW4_"))
        else line
        for line in lines
    ]
    assert crossed != lines
    assert not _lvs(gds, "\n".join(crossed), tmp_path).matched
