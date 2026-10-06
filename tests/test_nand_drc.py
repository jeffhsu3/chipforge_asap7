"""Run the public ASAP7 KLayout runset on the parametric NAND2.

Same contract as the inverter: an abutting logic cell carries no tap and
stops its implant 9 nm past its own ACTIVE, so on its own it reports exactly
the categories a placed row closes, and inside ``filler row filler tap
filler`` it reports none.  Skips when KLayout or the runset is missing;
``ASAP7_REQUIRE_TOOLS=1`` makes that a failure.
"""

from __future__ import annotations

import gdspy

from chipforge_asap7.devices import (
    NandSpec,
    RowSupportSpec,
    build_nand,
    build_nand_row,
    build_row_support,
)
from chipforge_asap7.verification.drc import runset_height_enumeration

NO_TAP_IN_CELL = {"ACTIVE.LUP.1"}
ROW_END_ENCLOSURE = {"ACTIVE.WELL.EN.1", "NSELECT.ACTIVE.EN.1", "PSELECT.ACTIVE.EN.1"}
#: What a tall but on-grid ACTIVE/SDT trips: the KLayout runset's 1-12 fin lists; nothing in gdscheck.
DECK_HEIGHT_ENUMERATION = runset_height_enumeration()


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def _terminated(spec: NandSpec, count: int, tag: str):
    """``filler row filler tap filler`` around `count` abutting NANDs."""
    library = _library()
    row = build_nand_row(spec, count, lib=library)
    support = {
        kind: build_row_support(
            RowSupportSpec(stack=spec.stack, kind=kind), lib=library
        )
        for kind in ("filler", "tap")
    }
    top = library.new_cell(f"nand_{tag}")
    cursor = 0
    for item in ("filler", "row", "filler", "tap", "filler"):
        cell = row if item == "row" else support[item]
        top.add(gdspy.CellReference(cell, origin=(cursor, 0)))
        cursor += count * spec.width if item == "row" else cell.get_bounding_box()[1][0]
    return library, top


def test_isolated_nand2_only_trips_the_missing_body_tap(asap7_drc):
    """The NAND2xp33 footprint and a fingered island, checked alone."""
    specs = [
        NandSpec(rows=((3, 3),), fingers=1, abut=False),
        NandSpec(rows=((6, 4),), fingers=2, abut=False, vt="lvt"),
        NandSpec(rows=((6, 4),), fingers=3, abut=False),
        NandSpec(rows=((2, 2),), fingers=2, abut=False, band_height=135),
    ]
    assert all(spec.drc_verifiable for spec in specs)
    library = _library()
    top = library.new_cell("nand_isolated")
    cursor = 0
    for spec in specs:
        top.add(gdspy.CellReference(build_nand(spec, lib=library), origin=(cursor, 0)))
        cursor += spec.width + 1000
    assert set(asap7_drc(library, top, tag="isolated")) == NO_TAP_IN_CELL


def test_abutting_row_alone_reports_only_the_row_end_residual(asap7_drc):
    spec = NandSpec(rows=((6, 4),), fingers=2)
    library = _library()
    row = build_nand_row(spec, 3, lib=library)
    assert set(asap7_drc(library, row, tag="row")) == NO_TAP_IN_CELL | ROW_END_ENCLOSURE


def test_terminated_rows_are_clean(asap7_drc):
    """Two fingers (one M2 tie), four fingers (two ties), and a pinned 270 nm row."""
    for spec, tag in (
        (NandSpec(rows=((6, 4),), fingers=2), "f2"),
        (NandSpec(rows=((6, 4),), fingers=4, vt="sram"), "f4"),
        (NandSpec(rows=((3, 3),), fingers=2, band_height=135), "row270"),
    ):
        assert spec.drc_verifiable
        assert asap7_drc(*_terminated(spec, 3, tag), tag=tag) == [], tag


def test_released_size_trips_only_the_decks_height_enumeration(asap7_drc):
    """A 14-fin band is past the runset's enumerated height list; nothing else fires."""
    spec = NandSpec(rows=((14, 7),), fingers=2)
    assert not spec.drc_verifiable
    categories = set(asap7_drc(*_terminated(spec, 2, "released"), tag="released"))
    assert categories == DECK_HEIGHT_ENUMERATION
