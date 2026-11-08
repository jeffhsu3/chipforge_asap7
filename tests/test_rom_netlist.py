"""The NOR ROM macro's CDL: its interface, its sizing, and LVS against it.

The LVS runs on the CDL itself: `run_lvs` expands each ``nfin`` instance to
unit fins, and the extractor merges fin diffusion as the CDL assumes.  A
match must also pair every pin with the net of its own name -- KLayout's
compare is topological and would otherwise accept two outputs swapped.
LVS skips when KLayout is missing; ``ASAP7_REQUIRE_TOOLS=1`` makes that a
failure.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import gdspy
import pytest

from chipforge_asap7.devices import RomDecoderSpec, build_rom
from chipforge_asap7.devices.rom_bitcell import RomArraySpec
from chipforge_asap7.devices.rom_netlist import (
    rom_macro_cdl,
    rom_macro_devices,
    rom_macro_ports,
)
from chipforge_asap7.verification.lvs import run_lvs
from chipforge_asap7.verification.lvs_report import summarize_lvsdb
from chipforge_asap7.verification.rom_macro_lvs import (
    label_floating_drains,
    rom_macro_reference,
)

SELF_TIMED = {"sense": True, "enable": True, "latch": True, "sample": True}
CONFIGS: dict[str, dict[str, Any]] = {
    "16x8": {"words": 16, "bits": 8},
    "16x8_no_lane": {"words": 16, "bits": 8, "lane": False},
    "16x8_sensed": {"words": 16, "bits": 8, "sense": True, "enable": True},
    "64x32_sampled": {"words": 64, "bits": 32, **SELF_TIMED},
    "16x6_self_timed": {"words": 16, "bits": 6, **SELF_TIMED, "control": True},
    "16x6_registered": {
        "words": 16,
        "bits": 6,
        **SELF_TIMED,
        "control": True,
        "register": True,
    },
    "64x32_registered": {
        "words": 64,
        "bits": 32,
        **SELF_TIMED,
        "control": True,
        "register": True,
    },
}


def _spec(words: int, bits: int, **flags: Any) -> RomDecoderSpec:
    return RomDecoderSpec(array=RomArraySpec(words, bits), **flags)


def _sites(spec: RomDecoderSpec) -> set[tuple[int, int]]:
    rng = random.Random(spec.array.words * spec.array.bits)
    words, bits = spec.array.words, spec.array.bits
    return {(w, b) for w in range(words) for b in range(bits) if rng.random() < 0.5}


def _build(spec: RomDecoderSpec, sites: set, tmp_path: Path) -> tuple[Path, str]:
    """The macro's GDS with its unprogrammed drains labelled for the compare."""
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    top = build_rom(spec, sites, lib=library)
    label_floating_drains(spec, sites, top)
    gds = tmp_path / "rom.gds"
    library.write_gds(str(gds))
    return gds, top.name


def _lvs(gds: Path, cdl: str, tmp_path: Path, name: str):
    schematic = tmp_path / "rom.cdl"
    schematic.write_text(cdl)
    return run_lvs(
        gds,
        schematic,
        tmp_path / "lvs",
        cell_name=name,
        asap7_standard_cell=True,
        tie_bodies=False,
        merge_fin_diffusion=True,
        flat=True,
    )


def _renamed(result) -> list:
    return [
        pair
        for circuit in summarize_lvsdb(result.report)["circuits"]
        for pair in circuit.get("renamed_pins", [])
    ]


def test_the_ports_are_the_interface():
    registered = _spec(16, 6, **SELF_TIMED, control=True, register=True)
    assert rom_macro_ports(registered) == (
        *(f"A<{i}>" for i in range(4)),
        "CLK",
        "RESETN",
        *(f"Q<{b}>" for b in range(6)),
        "VDD",
        "VSS",
    )
    sampled = _spec(16, 6, **SELF_TIMED)
    assert rom_macro_ports(sampled)[4:7] == ("PRECHN", "EVAL", "SAMPLE")
    assert rom_macro_ports(_spec(16, 6, sense=True))[4:6] == ("PRECHN", "D<0>")
    assert rom_macro_ports(_spec(16, 6))[4] == "BLL<0>"
    assert rom_macro_ports(_spec(16, 6, lane=False))[4] == "BL<0>"


@pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS.keys())
def test_the_cdl_sizes_the_unit_fin_reference(config):
    """Expanded by ``nfin``, the CDL is the LVS reference's devices."""
    spec = _spec(**config)
    sites = _sites(spec)
    cdl = rom_macro_cdl(spec, sites, name="rom")
    lines = cdl.splitlines()
    assert lines[1] == f".subckt rom {' '.join(rom_macro_ports(spec))}"
    assert lines[-1] == ".ends rom"
    instances = [line for line in lines if line.startswith("M")]
    devices = rom_macro_devices(spec, sites)
    assert len(instances) == len(devices) == len({d.name for d in devices})
    assert all(line.endswith(" l=20n nf=1 m=1") for line in instances)
    fins = sum(int(line.split("nfin=")[1].split()[0]) for line in instances)
    assert fins == rom_macro_reference(spec, sites).fins


@pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS.keys())
def test_macro_matches_its_cdl(config, tmp_path: Path, require_klayout):
    spec = _spec(**config)
    sites = _sites(spec)
    gds, name = _build(spec, sites, tmp_path)
    result = _lvs(gds, rom_macro_cdl(spec, sites, name=name), tmp_path, name)
    assert result.matched
    assert _renamed(result) == []
    fins = sum(d.fins for d in rom_macro_devices(spec, sites))
    assert result.extracted_netlist.read_text().count("M$") == fins


def test_two_outputs_swapped_fail_by_name(tmp_path: Path, require_klayout):
    """``Q<0>`` and ``Q<1>`` swapped inside the CDL still match topologically;
    only the pin pairing tells."""
    spec = _spec(**CONFIGS["16x6_registered"])
    sites = _sites(spec)
    gds, name = _build(spec, sites, tmp_path)
    swap = {"Q<0>": "Q<1>", "Q<1>": "Q<0>"}
    lines = [
        " ".join(swap.get(t, t) for t in line.split()) if line.startswith("M") else line
        for line in rom_macro_cdl(spec, sites, name=name).splitlines()
    ]
    result = _lvs(gds, "\n".join(lines) + "\n", tmp_path, name)
    assert sorted(_renamed(result)) == [["Q<0>", "Q<1>"], ["Q<1>", "Q<0>"]]


def test_a_bit_the_cdl_does_not_program_fails(tmp_path: Path, require_klayout):
    spec = _spec(**CONFIGS["16x6_registered"])
    sites = _sites(spec)
    gds, name = _build(spec, sites, tmp_path)
    cdl = rom_macro_cdl(spec, sites - {min(sites)}, name=name)
    assert not _lvs(gds, cdl, tmp_path, name).matched
