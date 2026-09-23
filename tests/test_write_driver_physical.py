"""Public-runset DRC and KLayout LVS of the parametric write driver.

The same contract as the bitline leaf: alone it reports the missing body tap
and nothing else, in a filler/tap row nothing at all, and every variant
matches a unit-fin reference.  Skips when the tools are missing;
``ASAP7_REQUIRE_TOOLS=1`` makes that a failure.
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import (
    RowSupportSpec,
    WriteDriverSpec,
    build_row_support,
    build_write_driver,
)
from chipforge_asap7.verification.lvs import render_write_driver_lvs_schematic, run_lvs

NO_TAP_IN_CELL = {"ACTIVE.LUP.1"}
SPECS = [
    WriteDriverSpec(),
    WriteDriverSpec(n_fins=4, p_fins=4, keeper_fins=2, band_height=(162, 189)),
    WriteDriverSpec(n_fins=2, p_fins=2, vt="sram"),
]


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def test_isolated_drivers_only_trip_the_missing_body_tap(asap7_drc):
    library = _library()
    top = library.new_cell("wrdrv_isolated")
    cursor = 0
    for spec in SPECS:
        top.add(
            gdspy.CellReference(
                build_write_driver(spec, lib=library), origin=(cursor, 0)
            )
        )
        cursor += spec.width + 1000
    assert set(asap7_drc(library, top, tag="isolated")) == NO_TAP_IN_CELL


@pytest.mark.parametrize("spec", SPECS[:2], ids=lambda spec: spec.cell_name)
def test_terminated_driver_is_clean(spec, asap7_drc):
    library = _library()
    cell = build_write_driver(spec, lib=library)
    support = {
        kind: build_row_support(
            RowSupportSpec(stack=spec.stack, kind=kind), lib=library
        )
        for kind in ("filler", "tap")
    }
    top = library.new_cell("wrdrv_terminated")
    cursor = 0
    for item in ("filler", "cell", "filler", "tap", "filler"):
        placed = cell if item == "cell" else support[item]
        top.add(gdspy.CellReference(placed, origin=(cursor, 0)))
        cursor += spec.width if item == "cell" else placed.get_bounding_box()[1][0]
    assert asap7_drc(library, top, tag="terminated") == []


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.cell_name)
def test_driver_matches_its_unit_fin_reference(spec, tmp_path: Path, require_klayout):
    library = _library()
    build_write_driver(spec, lib=library)
    gds = tmp_path / "cell.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(render_write_driver_lvs_schematic(spec))
    result = run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    )
    assert result.matched
    fins = sum(count for *_, count in spec.devices)
    assert result.extracted_netlist.read_text().count("M$") == fins


def test_stronger_keepers_than_drawn_do_not_match(tmp_path: Path, require_klayout):
    spec = WriteDriverSpec()
    library = _library()
    build_write_driver(spec, lib=library)
    gds = tmp_path / "cell.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(
        render_write_driver_lvs_schematic(
            WriteDriverSpec(keeper_fins=3), cell_name=spec.cell_name
        )
    )
    assert not run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    ).matched
