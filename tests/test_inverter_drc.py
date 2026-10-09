"""Run the public ASAP7 KLayout runset on the stacked-band inverter.

The inverter is a *logic* cell.  It carries no body tap, so latch-up is the
placed design's problem, and in `abut` style its implant deliberately overhangs
so that neighbours merge.  Neither can be drawn away inside one cell, so these
tests assert the exact residual category set rather than silence: a new
category is a real regression, and a category disappearing means a rule or a
deck changed and the reasoning below needs revisiting.

Skips when KLayout or the runset is missing; set ``ASAP7_REQUIRE_TOOLS=1`` to
make that a failure instead.
"""

from __future__ import annotations

import gdspy

from chipforge_asap7.devices import InverterSpec, build_inverter, build_inverter_row
from chipforge_asap7.devices.finfet import MAX_VERIFIABLE_FINS
from chipforge_asap7.devices.inverter import MIN_INPUT_REACH
from chipforge_asap7.verification.drc import runset_height_enumeration

#: No logic cell satisfies this alone -- it needs a tap row within 30 um.  The
#: released `dec_inv_62f_halved_AND` has no tap either.
NO_TAP_IN_CELL = {"ACTIVE.LUP.1"}
#: An abutting cell's implant and well stop 9 nm past its ACTIVE instead of
#: 46 nm; the rest of the enclosure comes from the neighbour it merges with.
#: Only the two *outer* edges of a placed row are ever short, which is what
#: `test_abutted_row_residual_does_not_grow_with_the_row` pins down.
ROW_END_ENCLOSURE = {
    "ACTIVE.WELL.EN.1",
    "NSELECT.ACTIVE.EN.1",
    "PSELECT.ACTIVE.EN.1",
}
#: The deck enumerates ACTIVE/SDT heights 1x27..12x27 instead of testing for a
#: multiple of 27, so a taller band reads as a violation.  See
#: `chipforge_asap7.devices.finfet.MAX_VERIFIABLE_FINS`.
#: What a tall but on-grid ACTIVE/SDT trips: the KLayout runset's 1-12 fin lists; nothing in gdscheck.
DECK_HEIGHT_ENUMERATION = runset_height_enumeration()

RELEASED = InverterSpec(rows=((18, 18), (13, 13)), fingers=2)


def _library(cells) -> tuple:
    """Build `cells` into a fresh library beside each other; return it and a top."""
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    top = library.new_cell("inverter_drc_top")
    cursor = 0
    for make, width in cells:
        top.add(gdspy.CellReference(make(library), origin=(cursor, 0)))
        cursor += width + 1000
    return library, top


def test_isolated_inverters_only_trip_the_missing_body_tap(asap7_drc):
    """Everything a self-contained cell can be responsible for is clean.

    The matrix covers one and two rows, both finger counts that exercise the
    output topology (one drain column, and two tied on M2), a VT marker, and
    the tallest band the runset can decide.
    """
    specs = [
        InverterSpec(rows=((4, 6),), abut=False),
        InverterSpec(rows=((1, 1),), abut=False),
        InverterSpec(rows=((2, 3),), fingers=4, abut=False, vt="lvt"),
        InverterSpec(rows=((12, 12), (8, 8)), abut=False),
        InverterSpec(rows=((3, 5), (5, 3), (2, 2)), abut=False, vt="sram"),
        InverterSpec(
            rows=((MAX_VERIFIABLE_FINS, MAX_VERIFIABLE_FINS),), fingers=4, abut=False
        ),
    ]
    assert all(spec.drc_verifiable for spec in specs)
    assert len({spec.cell_name for spec in specs}) == len(specs)

    library, top = _library(
        [
            ((lambda lib, s=spec: build_inverter(s, lib=lib)), spec.width)
            for spec in specs
        ]
    )
    assert set(asap7_drc(library, top, tag="isolated")) == NO_TAP_IN_CELL


def test_trimmed_input_is_as_clean_as_the_full_strap(asap7_drc):
    """One landing, down to the bare via pad, which is M1 minimum area exactly.

    Also a landing in the upper row only, and one that reaches a long way down
    and barely up -- the shape a driver slice asks for.
    """
    bare = (MIN_INPUT_REACH, MIN_INPUT_REACH)
    specs = [
        InverterSpec(
            rows=((4, 6), (3, 3)), abut=False, input_rows=(0,), input_reach=bare
        ),
        InverterSpec(rows=((4, 6), (3, 3)), abut=False, input_rows=(1,)),
        # One p fin leaves 13 nm above the seam, hard against the via row's
        # clearance; only a landing 36 nm long may sit that close (M1.S.2).
        InverterSpec(rows=((1, 1),), abut=False, input_reach=(23, 13)),
        InverterSpec(rows=((2, 3),), fingers=4, abut=False, vt="lvt", input_reach=bare),
        InverterSpec(
            rows=((12, 12), (8, 8)), abut=False, input_rows=(0,), input_reach=(310, 14)
        ),
    ]
    assert len({spec.cell_name for spec in specs}) == len(specs)
    library, top = _library(
        [
            ((lambda lib, s=spec: build_inverter(s, lib=lib)), spec.width)
            for spec in specs
        ]
    )
    assert set(asap7_drc(library, top, tag="trimmed")) == NO_TAP_IN_CELL


def test_abutting_style_adds_only_the_enclosure_its_neighbour_supplies(asap7_drc):
    specs = [
        InverterSpec(rows=((4, 6),)),
        InverterSpec(rows=((2, 3),), fingers=4, vt="lvt"),
        InverterSpec(rows=((12, 12), (8, 8))),
    ]
    library, top = _library(
        [
            ((lambda lib, s=spec: build_inverter(s, lib=lib)), spec.width)
            for spec in specs
        ]
    )
    categories = set(asap7_drc(library, top, tag="abutting"))
    assert categories == NO_TAP_IN_CELL | ROW_END_ENCLOSURE | {"LVT.ACTIVE.EN.1"}


def test_abutted_row_residual_does_not_grow_with_the_row(asap7_drc):
    """The proof that the interleave is complete.

    If abutment left any gap -- diffusion, implant, well, a missing gate -- the
    violation count would scale with the number of instances.  It does not: a
    four-wide row reports exactly what one tile does, because every extra
    violation would have to be interior and there are none.
    """
    spec = InverterSpec(rows=((4, 6),))
    counts = {}
    for instances in (1, 2, 4):
        library, top = _library(
            [
                (
                    (lambda lib, n=instances: build_inverter_row(spec, n, lib=lib)),
                    spec.row_width(instances),
                )
            ]
        )
        found = asap7_drc(library, top, tag=f"row{instances}")
        counts[instances] = len(found)
        assert set(found) == NO_TAP_IN_CELL | ROW_END_ENCLOSURE

    assert counts[1] == counts[2] == counts[4]


def test_the_released_decoder_inverter_only_adds_the_decks_height_enumeration(
    asap7_drc,
):
    """13- and 18-fin bands are legal ASAP7 and beyond what the deck decides."""
    assert not RELEASED.drc_verifiable
    library, top = _library(
        [
            (
                (lambda lib: build_inverter_row(RELEASED, 4, lib=lib)),
                RELEASED.row_width(4),
            )
        ]
    )
    categories = set(asap7_drc(library, top, tag="released"))
    assert categories == NO_TAP_IN_CELL | ROW_END_ENCLOSURE | DECK_HEIGHT_ENUMERATION
