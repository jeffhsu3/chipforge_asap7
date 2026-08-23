"""Run the public ASAP7 KLayout rule deck when it is available locally.

This is the regression behind the "DRC clean" claim in
`chipforge_asap7.devices.finfet`.  It skips when KLayout or the runset is
missing; set ``ASAP7_REQUIRE_TOOLS=1`` to make that a failure instead, so CI
cannot report green on a machine where the claim was never checked.
"""

from __future__ import annotations

import gdspy

from chipforge_asap7.devices import FinFETSpec, build_finfet
from chipforge_asap7.devices.finfet import MAX_FINS, MAX_VERIFIABLE_FINS


def _tile(specs: list[FinFETSpec], tag: str):
    """Tile `specs` side by side in a fresh library; return it and its top cell."""
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    top = library.new_cell(f"finfet_{tag}")
    cursor = 0
    for spec in specs:
        device = build_finfet(spec, lib=library)
        top.add(gdspy.CellReference(device, origin=(cursor, 0)))
        cursor += spec.width + 500
    return library, top


def test_representative_finfet_matrix_is_asap7_drc_clean(asap7_drc):
    # Exercise every fin height the runset can actually decide, for both
    # polarities, then add nontrivial fold counts, VT markers and a custom band.
    # Anything above MAX_VERIFIABLE_FINS is covered by the companion test below,
    # which pins down the deck's false positive rather than asserting silence.
    specs = [
        FinFETSpec(flavor=flavor, fins=fins)
        for flavor in ("n", "p")
        for fins in range(1, MAX_VERIFIABLE_FINS + 1)
    ] + [
        FinFETSpec(fins=2, fingers=3, multipliers=2),
        FinFETSpec(flavor="p", fins=4, fingers=3, vt="slvt"),
        FinFETSpec(fins=MAX_VERIFIABLE_FINS, fingers=2, vt="lvt"),
        FinFETSpec(fins=1, fingers=4),
        FinFETSpec(fins=1, row_height=135, vt="sram"),
        FinFETSpec(fins=1, fingers=5, row_height=378),
    ]
    # Distinct cell names are what keep `new_cell` from rejecting the matrix;
    # `cell_name` has to separate the custom-band specs from the default ones.
    assert len({spec.cell_name for spec in specs}) == len(specs)

    assert all(spec.drc_verifiable for spec in specs)
    assert asap7_drc(*_tile(specs, "drc_matrix"), tag="drc_matrix") == []


def test_tall_devices_only_trip_the_runsets_enumerated_height_rules(asap7_drc):
    """Above MAX_VERIFIABLE_FINS the deck is wrong, and only in two ways.

    ACTIVE.W.2 and SDT.W.3 both read "vertical height increment is an integer
    multiple of 27nm", but KLayout has no integer-multiple predicate so the
    deck enumerates 1x27..12x27.  A 13- or 18-fin device *is* a multiple of 27
    and still falls out of that list.  ASU's own SRAM hits this: running the
    same runset on `dec_inv_62f_halved_AND` reports these two rules too.

    Asserting the exact violation set keeps this honest in both directions --
    if a future deck fixes the enumeration this test fails and
    MAX_VERIFIABLE_FINS can be raised; if a tall device grows a *real*
    violation, that shows up here as a new category.
    """
    # The two fin heights the released decoder inverter is built from.
    specs = [
        FinFETSpec(flavor=flavor, fins=fins, fingers=3)
        for flavor in ("n", "p")
        for fins in (13, 18)
    ]
    assert all(not spec.drc_verifiable for spec in specs)
    assert max(spec.fins for spec in specs) == MAX_FINS

    categories = set(asap7_drc(*_tile(specs, "tall_devices"), tag="tall_devices"))
    assert categories == {"ACTIVE.W.2", "SDT.W.3"}
