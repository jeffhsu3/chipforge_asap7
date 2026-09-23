"""Public-runset DRC and KLayout LVS of the parametric output latch.

Skips when the tools are missing; ``ASAP7_REQUIRE_TOOLS=1`` makes that a
failure.
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import (
    OutputLatchSpec,
    RowSupportSpec,
    build_output_latch,
    build_row_support,
)
from chipforge_asap7.verification.lvs import render_output_latch_lvs_schematic, run_lvs

NO_TAP_IN_CELL = {"ACTIVE.LUP.1"}
SPECS = [
    OutputLatchSpec(),
    OutputLatchSpec(fingers=4),
    OutputLatchSpec(n_fins=4, p_fins=4, band_height=(162, 189)),
    OutputLatchSpec(fingers=6, vt="sram"),
]


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def _terminated(library, cells, stack, tag):
    support = {
        kind: build_row_support(RowSupportSpec(stack=stack, kind=kind), lib=library)
        for kind in ("filler", "tap")
    }
    top = library.new_cell(tag)
    cursor = 0
    for placed in (
        support["filler"],
        *cells,
        support["filler"],
        support["tap"],
        support["filler"],
    ):
        top.add(gdspy.CellReference(placed, origin=(cursor, 0)))
        (x0, _), (x1, _) = placed.get_bounding_box()
        boundary = [p for p in placed.polygons if p.layers[0] == 100]
        cursor += boundary[0].polygons[0][:, 0].max() if boundary else x1 - min(0, x0)
    return top


def test_isolated_latches_only_trip_the_missing_body_tap(asap7_drc):
    library = _library()
    top = library.new_cell("outlatch_isolated")
    cursor = 0
    for spec in SPECS:
        top.add(
            gdspy.CellReference(
                build_output_latch(spec, lib=library), origin=(cursor, 0)
            )
        )
        cursor += spec.width + 1000
    assert set(asap7_drc(library, top, tag="isolated")) == NO_TAP_IN_CELL


def test_terminated_latch_is_clean(asap7_drc):
    spec = OutputLatchSpec()
    library = _library()
    top = _terminated(
        library,
        [build_output_latch(spec, lib=library)],
        spec.stack,
        "outlatch_terminated",
    )
    assert asap7_drc(library, top, tag="terminated") == []


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.cell_name)
def test_latch_matches_its_unit_fin_reference(spec, tmp_path: Path, require_klayout):
    library = _library()
    build_output_latch(spec, lib=library)
    gds = tmp_path / "cell.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(render_output_latch_lvs_schematic(spec))
    result = run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    )
    assert result.matched
    fins = sum(count for *_, count in spec.devices)
    assert result.extracted_netlist.read_text().count("M$") == fins


def test_more_fingers_than_drawn_do_not_match(tmp_path: Path, require_klayout):
    spec = OutputLatchSpec(fingers=2)
    library = _library()
    build_output_latch(spec, lib=library)
    gds = tmp_path / "cell.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(
        render_output_latch_lvs_schematic(
            OutputLatchSpec(fingers=4), cell_name=spec.cell_name
        )
    )
    assert not run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    ).matched
