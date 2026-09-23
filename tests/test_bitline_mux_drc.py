"""Run the public ASAP7 KLayout runset on the parametric bitline leaf.

Same contract as the inverter and the NAND: the leaf carries no tap, so alone
it reports the latch-up rule and nothing else, and inside ``filler leaf filler
tap filler`` it reports nothing.  Skips when KLayout or the runset is missing;
``ASAP7_REQUIRE_TOOLS=1`` makes that a failure.
"""

from __future__ import annotations

from dataclasses import replace

import gdspy
import pytest

from chipforge_asap7.devices import (
    BitlineMuxSpec,
    RowSupportSpec,
    build_bitline_mux,
    build_bitline_mux_group,
    build_row_support,
)

NO_TAP_IN_CELL = {"ACTIVE.LUP.1"}

SPECS = [
    BitlineMuxSpec(),  # half an 8T bitcell row, four to one
    BitlineMuxSpec(selects=1),  # no mux at all: precharge and an isolation gate
    BitlineMuxSpec(n_fins=2, p_fins=4, selects=8, select=5, vt="sram"),
    BitlineMuxSpec(n_fins=4, p_fins=4, band_height=(162, 162)),
]
# A whole 8T bitcell row per pair, bitlines where sram_cell_8t puts them.
PORT_A = BitlineMuxSpec(rows=2, bitline_entry=(348.5, 245.5))
PORT_B = BitlineMuxSpec(rows=2, bitline_entry=(510, 78), bitline_layer="M4")


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def test_isolated_leaves_only_trip_the_missing_body_tap(asap7_drc):
    library = _library()
    top = library.new_cell("blmux_isolated")
    cursor = 0
    for spec in SPECS:
        top.add(
            gdspy.CellReference(
                build_bitline_mux(spec, lib=library), origin=(cursor, 0)
            )
        )
        cursor += spec.width + 1000
    assert set(asap7_drc(library, top, tag="isolated")) == NO_TAP_IN_CELL


@pytest.mark.parametrize(
    "spec", [*SPECS[:2], *SPECS[3:], PORT_A, PORT_B], ids=lambda spec: spec.cell_name
)
def test_terminated_leaf_is_clean(spec, asap7_drc):
    library = _library()
    leaf = build_bitline_mux(spec, lib=library)
    support = {
        kind: build_row_support(
            RowSupportSpec(stack=spec.stack, kind=kind), lib=library
        )
        for kind in ("filler", "tap")
    }
    top = library.new_cell("blmux_terminated")
    cursor = 0
    for item in ("filler", "leaf", "filler", "tap", "filler"):
        cell = leaf if item == "leaf" else support[item]
        top.add(gdspy.CellReference(cell, origin=(cursor, 0)))
        cursor += spec.width if item == "leaf" else cell.get_bounding_box()[1][0]
    assert asap7_drc(library, top, tag="terminated") == []


@pytest.mark.parametrize("selects", [2, 4])
def test_stacking_leaves_adds_no_violation(selects, asap7_drc):
    """Rails merge, M3 tracks meet end to end, and no seam rule fires."""
    library = _library()
    group = build_bitline_mux_group(BitlineMuxSpec(selects=selects), lib=library)
    assert set(asap7_drc(library, group, tag=f"group{selects}")) == NO_TAP_IN_CELL


@pytest.mark.parametrize(
    "spec",
    [PORT_A, PORT_B, replace(PORT_A, grid_offset=13.5), replace(PORT_B, grid_offset=13.5)],
    ids=["port_a_m2", "port_b_m4", "port_a_m2_on_the_array_grid", "port_b_m4_on_the_array_grid"],
)
def test_two_row_leaves_and_their_groups_add_no_violation(spec, asap7_drc):
    library = _library()
    top = library.new_cell("blmux_two_row")
    top.add(gdspy.CellReference(build_bitline_mux(spec, lib=library)))
    group = build_bitline_mux_group(spec, lib=library)
    top.add(gdspy.CellReference(group, origin=(spec.width + 1000, 0)))
    assert set(asap7_drc(library, top, tag="two_row")) == NO_TAP_IN_CELL
