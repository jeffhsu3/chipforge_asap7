"""The poly-wordline NOR-ROM tile: grid alignment, programming sites and DRC.

The DRC half skips when gdscheck is missing; set ``ASAP7_REQUIRE_TOOLS=1`` to
make that a failure instead.
"""

from __future__ import annotations

import gdspy
import pytest

from chipforge_asap7.devices import RomBitcellSpec, build_rom_array, build_rom_bitcell
from chipforge_asap7.devices.rom_bitcell import main
from chipforge_asap7.layout import FIN_PITCH, GATE_PITCH, GATE_WIDTH

#: The array has no body tap yet; a tap within 30 um comes with the periphery.
NO_TAP_IN_ARRAY = {"ACTIVE.LUP.1"}


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


@pytest.mark.parametrize("fins", [1, 2, 3])
def test_pitches(fins):
    spec = RomBitcellSpec(fins=fins)
    assert spec.width == 3 * GATE_PITCH
    assert spec.word_pitch == 81
    assert spec.height == (fins + 1) * FIN_PITCH
    assert spec.active_span == (0, fins * FIN_PITCH)


def test_two_fins_is_an_81_nm_square_bit():
    spec = RomBitcellSpec()
    assert (spec.word_pitch, spec.height) == (81, 81)


def test_tile_sits_on_the_asap7_grids(boxes_on):
    """Gates on 27 + 54k (GATE.S.1), fins on 10 + 27k (FIN.S.1), S/D between gates."""
    spec = RomBitcellSpec(fins=2)
    cell = build_rom_bitcell(spec)
    for x0, _, x1, _ in boxes_on(cell, "GATE"):
        assert x1 - x0 == GATE_WIDTH
        assert ((x0 + x1) / 2 - GATE_PITCH / 2) % GATE_PITCH == 0
    for _, y0, _, _ in boxes_on(cell, "FIN"):
        assert (y0 - 10) % FIN_PITCH == 0
    for name, x in spec.sd_xs.items():
        assert x % GATE_PITCH == 0, name


def test_tile_has_no_gate_contact(boxes_on):
    """The wordline is the poly: no LIG, no gate cut, no V1 until programmed."""
    cell = build_rom_bitcell(RomBitcellSpec())
    for layer in ("LIG", "GATE_CUT", "V1"):
        assert boxes_on(cell, layer) == []


def test_sites_land_on_the_drain_stubs():
    spec = RomBitcellSpec()
    for word, drain in ((0, "Da"), (1, "Db")):
        x, y = spec.site(word)
        x0, x1 = spec.stubs[drain]
        assert (x - 9 - x0, x1 - x - 9) == (5, 5)  # V1.M1.EN.1 on both ends
        assert y == spec.contact_y
    with pytest.raises(IndexError):
        spec.site(2)


def test_array_places_one_v1_per_programmed_site(boxes_on):
    spec = RomBitcellSpec()
    _library()
    sites = {(0, 0), (3, 1), (5, 2)}
    cell = build_rom_array(spec, 6, 3, sites)
    vias = boxes_on(cell, "V1")
    assert len(vias) == len(sites)
    centres = {((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in vias}
    for word, bit in sites:
        tile, tile_word = divmod(word, 2)
        x, y = spec.site(tile_word)
        assert (tile * spec.width + x, bit * spec.height + y) in centres


@pytest.mark.parametrize(
    ("words", "bits", "sites", "error"),
    [
        (3, 1, (), ValueError),
        (2, 0, (), ValueError),
        (2, 1, [(2, 0)], IndexError),
    ],
)
def test_array_rejects_bad_shapes(words, bits, sites, error):
    _library()
    with pytest.raises(error):
        build_rom_array(RomBitcellSpec(), words, bits, sites)


@pytest.mark.parametrize("fins", [0, True, 1.5])
def test_spec_rejects_bad_fins(fins):
    with pytest.raises((TypeError, ValueError)):
        RomBitcellSpec(fins=fins)


def test_netlist_floats_unprogrammed_drains():
    lines = RomBitcellSpec().netlist(programmed=(True, False)).splitlines()
    assert lines[0].startswith("M0 BL WLa VSS VSS nmos_rvt nfin=2")
    assert lines[1].startswith("M1 M1_float WLb VSS VSS")


def test_cli_writes_tile_and_array(tmp_path):
    main(["--fins", "1", "--out", str(tmp_path / "tile.gds")])
    main(["--words", "4", "--bits", "2", "--out", str(tmp_path / "array.gds")])
    assert (tmp_path / "tile.gds").stat().st_size > 0
    assert (tmp_path / "array.gds").stat().st_size > 0


@pytest.mark.parametrize("fins", [1, 2, 3])
@pytest.mark.parametrize("pattern", ["none", "all", "checker"])
def test_array_drc(asap7_drc, fins, pattern):
    words, bits = 8, 6
    every = [(w, b) for w in range(words) for b in range(bits)]
    sites = {
        "none": [],
        "all": every,
        "checker": [(w, b) for w, b in every if (w + b) % 2],
    }[pattern]
    library = _library()
    top = build_rom_array(
        RomBitcellSpec(fins=fins), words, bits, sites, lib=library,
        name=f"rom_f{fins}_{pattern}",
    )  # fmt: skip
    assert set(asap7_drc(library, top, tag=top.name)) == NO_TAP_IN_ARRAY
