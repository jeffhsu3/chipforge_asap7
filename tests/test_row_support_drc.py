"""What a tap and a filler are actually for, measured with the public runset.

An abutting logic row is as clean as a logic cell can be on its own and still
reports eight violations, because two of the rules are not properties of a
cell.  These tests are the evidence for that claim and for the fix.

Skips when KLayout or the runset is missing; set ``ASAP7_REQUIRE_TOOLS=1`` to
make that a failure instead.
"""

from __future__ import annotations

import gdspy

from chipforge_asap7.devices import (
    InverterSpec,
    RowStack,
    RowSupportSpec,
    build_inverter_row,
    build_row_support,
)
from chipforge_asap7.devices.finfet import MAX_VERIFIABLE_FINS
from chipforge_asap7.devices.row_support import ROW_SUPPORT_KINDS

#: No logic cell satisfies this alone -- it needs a body tie within 30 um.
NO_TAP_IN_CELL = {"ACTIVE.LUP.1"}
#: An abutting cell's implant stops 9 nm past its own ACTIVE instead of 46 nm.
ROW_END_ENCLOSURE = {
    "ACTIVE.WELL.EN.1",
    "NSELECT.ACTIVE.EN.1",
    "PSELECT.ACTIVE.EN.1",
}
#: The deck enumerates ACTIVE/SDT heights 1x27..12x27 rather than testing for a
#: multiple of 27; a tap fills its band, so a tall stack lands past that list.
DECK_HEIGHT_ENUMERATION = {"ACTIVE.W.2", "SDT.W.3"}

INVERTER = InverterSpec(rows=((4, 6),))
_KIND_OF = {"f": "filler", "t": "tap", "d": "decap", "r": None}


def _plan(plan: str, count: int = 4):
    """Abut `plan` left to right; ``r`` is the logic row, the rest are support."""
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    row = build_inverter_row(INVERTER, count, lib=library)
    top = library.new_cell(f"plan_{plan}")
    built: dict[str, tuple] = {}
    cursor = 0
    for item in plan:
        if item == "r":
            top.add(gdspy.CellReference(row, origin=(cursor, 0)))
            cursor += INVERTER.row_width(count)
            continue
        kind = _KIND_OF[item]
        if kind not in built:
            spec = RowSupportSpec(stack=INVERTER.stack, kind=kind)
            built[kind] = (spec, build_row_support(spec, lib=library))
        spec, cell = built[kind]
        top.add(gdspy.CellReference(cell, origin=(cursor, 0)))
        cursor += spec.width
    return library, top


def _matrix(specs, tag):
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    top = library.new_cell(f"support_{tag}")
    cursor = 0
    for spec in specs:
        top.add(
            gdspy.CellReference(
                build_row_support(spec, lib=library), origin=(cursor, 0)
            )
        )
        cursor += spec.width + 800
    return library, top


def test_support_cells_are_drc_clean_on_their_own(asap7_drc):
    """Unlike a logic cell, these three have no structural residual at all."""
    specs = [
        RowSupportSpec(stack=stack, kind=kind, width_cpp=width)
        for stack in (
            RowStack(rows=((4, 6),)),
            RowStack(rows=((1, 1),)),
            RowStack(rows=((2, 3), (5, 5), (1, 1)), vt="lvt"),
            RowStack(rows=((10, 10), (8, 8)), vt="sram"),
        )
        for kind in ROW_SUPPORT_KINDS
        for width in ({"filler": (2, 4), "tap": (2, 3), "decap": (3, 6)})[kind]
    ]
    assert all(spec.drc_verifiable for spec in specs)
    assert len({spec.cell_name for spec in specs}) == len(specs)
    assert asap7_drc(*_matrix(specs, "clean"), tag="clean") == []


def test_a_tall_stacks_tie_only_trips_the_decks_height_enumeration(asap7_drc):
    """A tap has no fins, but its tie still has a height the deck cannot decide."""
    tall = RowStack(rows=((18, 18), (13, 13)))
    specs = [RowSupportSpec(stack=tall, kind=kind) for kind in ("tap", "decap")]
    assert all(not spec.drc_verifiable for spec in specs)
    assert max(spec.tie_fins(band) for spec in specs for band in spec.bands) > (
        MAX_VERIFIABLE_FINS
    )
    categories = set(asap7_drc(*_matrix(specs, "tall"), tag="tall"))
    assert categories == DECK_HEIGHT_ENUMERATION


def test_a_terminated_row_is_finally_clean(asap7_drc):
    """The whole point of the trio, in three measurements.

    A filler at each end supplies the implant enclosure the row's outermost
    ACTIVE is short of; a tap anywhere in reach supplies the body tie.  Neither
    can be drawn inside a logic cell, and together they take the row to zero.
    """
    alone = set(asap7_drc(*_plan("r"), tag="alone"))
    assert alone == NO_TAP_IN_CELL | ROW_END_ENCLOSURE

    capped = set(asap7_drc(*_plan("frf"), tag="capped"))
    assert capped == NO_TAP_IN_CELL

    assert asap7_drc(*_plan("frftf"), tag="terminated") == []
    assert asap7_drc(*_plan("frftfdf"), tag="with_decap") == []


def test_a_tap_next_to_logic_does_not_replace_a_filler(asap7_drc):
    """The ordering is not interchangeable, and this is why.

    A filler carries the *same* implant as the row, which is what extends the
    enclosure past its last ACTIVE.  A tap carries the opposite, so putting one
    straight against a logic cell leaves the enclosure exactly as short as it
    was -- while still fixing latch-up, which is what makes the mistake easy to
    miss.
    """
    categories = set(asap7_drc(*_plan("ftrtf"), tag="tap_adjacent"))
    assert "ACTIVE.LUP.1" not in categories
    assert categories & ROW_END_ENCLOSURE
