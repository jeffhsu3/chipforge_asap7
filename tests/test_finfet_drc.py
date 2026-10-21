"""Run ASAP7 DRC with gdscheck when it is available locally.

This is the regression behind the "DRC clean" claim in
`chipforge_asap7.devices.finfet`.  It skips when gdscheck is
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
    # Exercise both polarities, fold counts, VT markers and a custom band.
    # The companion test covers the released decoder's taller devices.
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


def test_tall_devices_are_drc_clean(asap7_drc):
    """Check legal 13- and 18-fin devices with gdscheck.

    The KLayout runset's historical height enumeration limited the
    `drc_verifiable` flag to 12 fins; it does not limit gdscheck.
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
    # Tall, on-grid ACTIVE heights need no exemption from gdscheck.
    assert categories == set()
