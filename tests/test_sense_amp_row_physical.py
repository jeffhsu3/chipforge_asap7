"""Public-runset DRC and KLayout LVS of the row sense amplifier, and of all four IO cells abutted.

Skips when the tools are missing; ``ASAP7_REQUIRE_TOOLS=1`` makes that a
failure.
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import (
    BitlineMuxSpec,
    OutputLatchSpec,
    RowSupportSpec,
    SenseAmpRowSpec,
    WriteDriverSpec,
    build_bitline_mux,
    build_output_latch,
    build_row_support,
    build_sense_amp_row,
    build_write_driver,
)
from chipforge_asap7.verification.lvs import render_sense_amp_row_lvs_schematic, run_lvs

NO_TAP_IN_CELL = {"ACTIVE.LUP.1"}
SPECS = [
    SenseAmpRowSpec(),
    SenseAmpRowSpec(n_fingers=4),
    SenseAmpRowSpec(tail_fingers=4),
    SenseAmpRowSpec(
        n_fingers=4, tail_fingers=6, n_fins=4, p_fins=4, band_height=(162, 189)
    ),
    SenseAmpRowSpec(vt="sram"),
    # The compact 6T column's: a 162 nm n band, room for M1.S.2's 25 nm.
    SenseAmpRowSpec(band_height=(162, 162)),
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


def test_isolated_amplifiers_only_trip_the_missing_body_tap(asap7_drc):
    library = _library()
    top = library.new_cell("sarow_isolated")
    cursor = 0
    for spec in SPECS:
        top.add(
            gdspy.CellReference(
                build_sense_amp_row(spec, lib=library), origin=(cursor, 0)
            )
        )
        cursor += spec.width + 1000
    assert set(asap7_drc(library, top, tag="isolated")) == NO_TAP_IN_CELL


@pytest.mark.parametrize(
    "spec", [SenseAmpRowSpec(), SenseAmpRowSpec(band_height=(162, 162))], ids=lambda spec: spec.cell_name
)
def test_terminated_amplifier_is_clean(asap7_drc, spec):
    library = _library()
    top = _terminated(
        library,
        [build_sense_amp_row(spec, lib=library)],
        spec.stack,
        "sarow_terminated",
    )
    assert asap7_drc(library, top, tag="terminated") == []


def test_the_four_io_cells_abut_on_one_row_pair(asap7_drc):
    """Bitline leaf, sense amplifier, write driver and output latch side by side, terminated: nothing fires."""
    library = _library()
    cells = [
        build_bitline_mux(
            BitlineMuxSpec(rows=2, bitline_entry=(348.5, 245.5)), lib=library
        ),
        build_sense_amp_row(SenseAmpRowSpec(), lib=library),
        build_write_driver(WriteDriverSpec(), lib=library),
        build_output_latch(OutputLatchSpec(), lib=library),
    ]
    top = _terminated(library, cells, WriteDriverSpec().stack, "io_row")
    assert asap7_drc(library, top, tag="io_row") == []


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.cell_name)
def test_amplifier_matches_its_unit_fin_reference(
    spec, tmp_path: Path, require_klayout
):
    library = _library()
    build_sense_amp_row(spec, lib=library)
    gds = tmp_path / "cell.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(render_sense_amp_row_lvs_schematic(spec))
    result = run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    )
    assert result.matched
    fins = sum(count for *_, count in spec.devices)
    assert result.extracted_netlist.read_text().count("M$") == fins


def test_a_longer_tail_than_drawn_does_not_match(tmp_path: Path, require_klayout):
    spec = SenseAmpRowSpec()
    library = _library()
    build_sense_amp_row(spec, lib=library)
    gds = tmp_path / "cell.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(
        render_sense_amp_row_lvs_schematic(
            SenseAmpRowSpec(tail_fingers=4), cell_name=spec.cell_name
        )
    )
    assert not run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    ).matched
