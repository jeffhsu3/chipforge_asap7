"""KLayout LVS of the 270 nm (6T row) bitline leaf and its stacked group.

The leaf is `BitlineMuxSpec`'s six devices, so `BitlineMuxSpec`'s unit-fin
references apply to it unchanged.  No body tap: ``tie_bodies=True``.
Skips when KLayout is missing; ``ASAP7_REQUIRE_TOOLS=1`` makes that a failure.
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import SidewaysMuxSpec, build_sideways_mux, build_sideways_mux_group
from chipforge_asap7.verification.lvs import (
    render_bitline_mux_group_lvs_schematic,
    render_bitline_mux_lvs_schematic,
    run_lvs,
)


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


@pytest.mark.parametrize(
    "spec",
    [SidewaysMuxSpec(selects=1), SidewaysMuxSpec(), SidewaysMuxSpec(swapped=True),
     SidewaysMuxSpec(selects=16, select=11)],
    ids=lambda spec: spec.cell_name,
)
def test_leaf_matches_its_unit_fin_reference(spec, tmp_path: Path, require_klayout):
    library = _library()
    build_sideways_mux(spec, lib=library)
    gds = tmp_path / "leaf.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(render_bitline_mux_lvs_schematic(spec))
    result = run_lvs(gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True)
    assert result.matched
    assert result.extracted_netlist.read_text().count("M$") == 6 * spec.fins


@pytest.mark.parametrize("selects", [4, 8, 16])
def test_stacked_leaves_are_one_mux(selects, tmp_path: Path, require_klayout):
    """``SA``/``SAN``/``PRECHN`` one net each, every select on its own leaf only."""
    spec = SidewaysMuxSpec(selects=selects)
    library = _library()
    group = build_sideways_mux_group(spec, lib=library)
    gds = tmp_path / "group.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(render_bitline_mux_group_lvs_schematic(spec, cell_name=group.name))
    result = run_lvs(gds, reference, tmp_path / "lvs", cell_name=group.name, tie_bodies=True)
    assert result.matched
    assert result.extracted_netlist.read_text().count("M$") == selects * 6 * spec.fins


def test_two_leaves_on_one_select_are_not_a_mux(tmp_path: Path, require_klayout):
    """The check above can fail: give leaf 1 leaf 0's selects and it does."""
    spec = SidewaysMuxSpec(selects=2)
    library = _library()
    group = build_sideways_mux_group(spec, lib=library)
    good, bad = (next(ref.ref_cell for ref in group.references if ref.ref_cell.name.endswith(f"leaf{i}"))
                 for i in (0, 1))  # fmt: skip
    for ref in group.references:
        if ref.ref_cell is bad:
            ref.ref_cell = good
    gds = tmp_path / "group.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(render_bitline_mux_group_lvs_schematic(spec, cell_name=group.name))
    assert not run_lvs(gds, reference, tmp_path / "lvs", cell_name=group.name, tie_bodies=True).matched
