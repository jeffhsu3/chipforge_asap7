"""KLayout LVS regressions for the parametric bitline leaf and the stacked mux.

The leaf carries no body tap, so these run with ``tie_bodies=True`` like the
other tapless cells.  Skips when KLayout is missing; ``ASAP7_REQUIRE_TOOLS=1``
makes that a failure instead.
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import (
    BitlineMuxSpec,
    build_bitline_mux,
    build_bitline_mux_group,
)
from chipforge_asap7.verification.lvs import (
    render_bitline_mux_group_lvs_schematic,
    render_bitline_mux_lvs_schematic,
    run_lvs,
)

# A whole 8T bitcell row per pair, bitlines where sram_cell_8t puts them.
PORT_A = BitlineMuxSpec(rows=2, bitline_entry=(348.5, 245.5))
PORT_B = BitlineMuxSpec(rows=2, bitline_entry=(510, 78), bitline_layer="M4")


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


@pytest.mark.parametrize(
    "spec",
    [
        BitlineMuxSpec(),
        BitlineMuxSpec(selects=1),
        BitlineMuxSpec(selects=8, select=5),
        BitlineMuxSpec(n_fins=2, p_fins=4, selects=2, select=1, vt="sram"),
        BitlineMuxSpec(n_fins=4, p_fins=4, band_height=(162, 162)),
        BitlineMuxSpec(rows=2),
        PORT_A,
        PORT_B,
        BitlineMuxSpec(bitline_entry=(224, 260)),
    ],
    ids=lambda spec: spec.cell_name,
)
def test_leaf_matches_its_unit_fin_reference(spec, tmp_path: Path, require_klayout):
    library = _library()
    build_bitline_mux(spec, lib=library)
    gds = tmp_path / "leaf.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(render_bitline_mux_lvs_schematic(spec))
    result = run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    )
    assert result.matched
    fins = spec.rows * (2 * spec.n_fins + 4 * spec.p_fins)
    assert result.extracted_netlist.read_text().count("M$") == fins


def test_wrong_sizing_fails(tmp_path: Path, require_klayout):
    spec = BitlineMuxSpec()
    library = _library()
    build_bitline_mux(spec, lib=library)
    gds = tmp_path / "leaf.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(
        render_bitline_mux_lvs_schematic(
            BitlineMuxSpec(n_fins=2), cell_name=spec.cell_name
        )
    )
    assert not run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    ).matched


@pytest.mark.parametrize(
    "spec",
    [BitlineMuxSpec(selects=2), BitlineMuxSpec(), PORT_A, PORT_B],
    ids=lambda spec: spec.cell_name,
)
def test_stacked_leaves_are_one_mux(spec, tmp_path: Path, require_klayout):
    """``SA``/``SAN``/``PRECHN`` one net each, every select on its own leaf only."""
    library = _library()
    group = build_bitline_mux_group(spec, lib=library)
    gds = tmp_path / "group.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(
        render_bitline_mux_group_lvs_schematic(spec, cell_name=group.name)
    )
    result = run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=group.name, tie_bodies=True
    )
    assert result.matched
    assert (
        result.extracted_netlist.read_text().count("M$")
        == spec.selects * spec.rows * 18
    )


def test_two_leaves_on_one_select_are_not_a_mux(tmp_path: Path, require_klayout):
    """The check above can fail: give both leaves select 0 and it does."""
    spec = BitlineMuxSpec(selects=2)
    library = _library()
    group = build_bitline_mux_group(spec, lib=library)
    good, bad = (
        next(
            ref.ref_cell
            for ref in group.references
            if ref.ref_cell.name.endswith(f"leaf{i}")
        )
        for i in (0, 1)
    )
    for ref in group.references:
        if ref.ref_cell is bad:
            ref.ref_cell = good
    gds = tmp_path / "group.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(
        render_bitline_mux_group_lvs_schematic(spec, cell_name=group.name)
    )
    assert not run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=group.name, tie_bodies=True
    ).matched


def test_an_upper_row_cut_off_from_its_bitline_fails(tmp_path: Path, require_klayout):
    """Two rows only match if the entry column reaches both: drop the upper V2 and they do not."""
    spec = PORT_A
    library = _library()
    cell = build_bitline_mux(spec, lib=library)
    x_col, y_top = spec.entry_x["BL"], spec.track_ys("BL")[1]
    via = (25, 0)
    doomed = [
        polygon
        for polygon in cell.polygons
        if (polygon.layers[0], polygon.datatypes[0]) == via
        and tuple(polygon.polygons[0].mean(axis=0)) == (x_col, y_top)
    ]
    assert len(doomed) == 1
    cell.remove_polygons(
        lambda points, layer, datatype: (
            (layer, datatype) == via and tuple(points.mean(axis=0)) == (x_col, y_top)
        )
    )
    gds = tmp_path / "leaf.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(render_bitline_mux_lvs_schematic(spec))
    assert not run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    ).matched
