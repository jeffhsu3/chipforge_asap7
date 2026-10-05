"""Run the public ASAP7 KLayout runset on the 270 nm (6T row) bitline leaf.

The bitline leaf's contract: no tap in the cell, so alone it reports the
latch-up rule and nothing else, and inside ``filler leaf filler tap filler``
on a 270 nm row it reports nothing.  Skips when KLayout or the runset is
missing; ``ASAP7_REQUIRE_TOOLS=1`` makes that a failure.
"""

from __future__ import annotations

import gdspy
import pytest

from chipforge_asap7.devices import (
    RowSupportSpec,
    SidewaysMuxSpec,
    build_row_support,
    build_sideways_mux,
    build_sideways_mux_group,
)
from chipforge_asap7.devices.row import RowStack

NO_TAP_IN_CELL = {"ACTIVE.LUP.1"}
#: The 7.5-track standard-cell row the leaf sits in.
ROW_270 = RowStack(rows=((3, 3),), vt="rvt", band_height=135)


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def test_isolated_leaves_only_trip_the_missing_body_tap(asap7_drc):
    library = _library()
    top = library.new_cell("blmux270_isolated")
    cursor = 0
    for spec in (SidewaysMuxSpec(selects=1), SidewaysMuxSpec(), SidewaysMuxSpec(selects=16, select=9)):
        top.add(gdspy.CellReference(build_sideways_mux(spec, lib=library), origin=(cursor, 0)))
        cursor += spec.width + 1000
    assert set(asap7_drc(library, top, tag="isolated")) == NO_TAP_IN_CELL


@pytest.mark.parametrize(
    "selects, swapped, local_ysel",
    [(1, False, False), (2, False, False), (4, False, False), (4, True, False), (8, True, False),
     (16, False, False), (8, False, True), (8, True, True), (16, True, True)],
)
def test_terminated_leaf_is_clean(selects, swapped, local_ysel, asap7_drc):
    spec = SidewaysMuxSpec(selects=selects, select=selects - 1, swapped=swapped, local_ysel=local_ysel)
    library = _library()
    leaf = build_sideways_mux(spec, lib=library)
    support = {kind: build_row_support(RowSupportSpec(stack=ROW_270, kind=kind), lib=library)
               for kind in ("filler", "tap")}  # fmt: skip
    top = library.new_cell("blmux270_terminated")
    cursor = 0
    for item in ("filler", "leaf", "filler", "tap", "filler"):
        cell = leaf if item == "leaf" else support[item]
        top.add(gdspy.CellReference(cell, origin=(cursor, 0)))
        cursor += spec.width if item == "leaf" else cell.get_bounding_box()[1][0]
    assert asap7_drc(library, top, tag="terminated") == []


@pytest.mark.parametrize("selects, local_ysel", [(4, False), (8, False), (16, False), (8, True), (16, True)])
def test_stacking_leaves_adds_no_violation(selects, local_ysel, asap7_drc):
    """Mirrored leaves share their rails (and the inverters' wells), the M3 tracks meet, and no seam rule fires."""
    library = _library()
    group = build_sideways_mux_group(SidewaysMuxSpec(selects=selects, local_ysel=local_ysel), lib=library)
    assert set(asap7_drc(library, group, tag=f"group{selects}")) == NO_TAP_IN_CELL
