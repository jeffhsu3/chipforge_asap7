"""DRC and LVS of the NOR ROM's address-latch row.

Alone the row reports only the latch-up rule; inside ``filler row filler tap
filler`` it reports nothing.  LVS compares it with a unit-fin reference under
``tie_bodies=True``.  Skips when KLayout or gdscheck is missing;
``ASAP7_REQUIRE_TOOLS=1`` makes that a failure.
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import RowSupportSpec, build_row_support
from chipforge_asap7.devices.rom_latch import (
    RomAddressLatchRowSpec,
    build_rom_address_latch_row,
)
from chipforge_asap7.verification.lvs import run_lvs

NO_TAP_IN_CELL = {"ACTIVE.LUP.1"}

SPECS = [
    RomAddressLatchRowSpec(),
    RomAddressLatchRowSpec(latches=4),
    RomAddressLatchRowSpec(latches=2, vt="lvt"),
    RomAddressLatchRowSpec(latches=2, n_fins=3, p_fins=3, band_height=(270, 270)),
]


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def test_isolated_rows_only_trip_the_missing_body_tap(asap7_drc):
    library = _library()
    top = library.new_cell("rom_latch_isolated")
    cursor = 0
    for spec in SPECS:
        cell = build_rom_address_latch_row(spec, lib=library)
        top.add(gdspy.CellReference(cell, (cursor, 0)))
        cursor += spec.width + 1080
    assert set(asap7_drc(library, top, tag="isolated")) == NO_TAP_IN_CELL


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.cell_name)
def test_terminated_row_is_clean(spec, asap7_drc):
    library = _library()
    row = build_rom_address_latch_row(spec, lib=library)
    support = {
        kind: build_row_support(
            RowSupportSpec(stack=spec.stack, kind=kind), lib=library
        )
        for kind in ("filler", "tap")
    }
    top = library.new_cell("rom_latch_terminated")
    cursor = 0
    for item in ("filler", "row", "filler", "tap", "filler"):
        cell = row if item == "row" else support[item]
        top.add(gdspy.CellReference(cell, origin=(cursor, 0)))
        cursor += spec.width if item == "row" else cell.get_bounding_box()[1][0]
    assert asap7_drc(library, top, tag="terminated") == []


def test_too_short_a_row_is_refused():
    with pytest.raises(ValueError, match="seven M2 levels"):
        RomAddressLatchRowSpec(n_fins=2, p_fins=2, band_height=(216, 216))


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.cell_name)
def test_row_matches_its_unit_fin_reference(spec, tmp_path: Path, require_klayout):
    library = _library()
    build_rom_address_latch_row(spec, lib=library)
    gds = tmp_path / "row.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(spec.lvs_reference())
    result = run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    )
    assert result.matched
    fins = sum(device[5] for device in spec.devices)
    assert result.extracted_netlist.read_text().count("M$") == fins


def test_feedback_from_the_wrong_side_fails(tmp_path: Path, require_klayout):
    """A reference whose feedback gate closes from D instead of Q must not match."""
    spec = RomAddressLatchRowSpec(latches=2)
    library = _library()
    build_rom_address_latch_row(spec, lib=library)
    gds = tmp_path / "row.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(
        spec.lvs_reference().replace("MNF0_f0 X0 PRECHN Q<0>", "MNF0_f0 X0 PRECHN D<0>")
    )
    assert not run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    ).matched
