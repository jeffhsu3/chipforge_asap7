"""DRC and LVS of the four-wordline driver slice.

Same contract as its leaf cells: tapless and abutting, so alone it reports
the missing body tap and the row-end implant enclosure, and between a filler
and a tap column of its own height it reports nothing.  LVS runs with
``tie_bodies=True``.  Skips without KLayout or the public runset;
``ASAP7_REQUIRE_TOOLS=1`` makes that a failure.
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import (
    DriverSliceSpec,
    InverterSpec,
    NandSpec,
    build_driver_slice,
    build_driver_slice_support,
    size_decoder,
)
from chipforge_asap7.devices.driver_slice import RUNSET_ACTIVE_FINS
from chipforge_asap7.verification.lvs import render_driver_slice_lvs_schematic, run_lvs

NO_TAP_IN_CELL = {"ACTIVE.LUP.1"}
ROW_END_ENCLOSURE = {"ACTIVE.WELL.EN.1", "NSELECT.ACTIVE.EN.1", "PSELECT.ACTIVE.EN.1"}
DECK_HEIGHT_ENUMERATION = {"ACTIVE.W.2", "SDT.W.3"}

SMALL = DriverSliceSpec(
    nand=NandSpec(rows=((6, 4),)), inverter=InverterSpec(rows=((8, 8), (4, 4)))
)
ONE_ROW = DriverSliceSpec(
    nand=NandSpec(rows=((4, 2),), vt="lvt"),
    inverter=InverterSpec(rows=((6, 6),), vt="lvt"),
)
RELEASED = DriverSliceSpec()
#: OpenFinRAM's 64-cell slice: size_decoder's 17-fin driver row split to stay
#: within the runset's ACTIVE heights.
C64 = DriverSliceSpec.from_sizing(size_decoder(0.181 * 64, 32), max_active_fins=RUNSET_ACTIVE_FINS)


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def _terminated(spec: DriverSliceSpec, count: int, tag: str):
    """``filler slices filler tap filler``, every column the slice's full height."""
    library = _library()
    slice_cell = build_driver_slice(spec, lib=library)
    support = {
        kind: build_driver_slice_support(spec, kind, lib=library)
        for kind in ("filler", "tap")
    }
    top = library.new_cell(f"slice_{tag}")
    cursor = 0
    for item in ("filler", *["slice"] * count, "filler", "tap", "filler"):
        top.add(
            gdspy.CellReference(
                slice_cell if item == "slice" else support[item], origin=(cursor, 0)
            )
        )
        cursor += spec.width if item == "slice" else 108
    return library, top


def test_slice_alone_reports_only_what_a_placed_row_closes(asap7_drc):
    library = _library()
    cell = build_driver_slice(SMALL, lib=library)
    assert (
        set(asap7_drc(library, cell, tag="alone")) == NO_TAP_IN_CELL | ROW_END_ENCLOSURE
    )


@pytest.mark.parametrize(("spec", "tag"), [(SMALL, "small"), (ONE_ROW, "one_row"), (C64, "c64")])
def test_terminated_slices_are_clean(spec, tag, asap7_drc):
    """One slice and two butted slices: the M2 ties and M3 tracks have to clear
    their neighbours' across the slice boundary too."""
    assert spec.drc_verifiable
    assert asap7_drc(*_terminated(spec, 1, tag), tag=tag) == []
    assert asap7_drc(*_terminated(spec, 2, f"{tag}_x2"), tag=f"{tag}_x2") == []


def test_released_size_trips_only_the_decks_height_enumeration(asap7_drc):
    assert not RELEASED.drc_verifiable
    categories = set(asap7_drc(*_terminated(RELEASED, 1, "released"), tag="released"))
    assert categories == DECK_HEIGHT_ENUMERATION


@pytest.mark.parametrize("spec", [SMALL, ONE_ROW, RELEASED, C64])
def test_slice_matches_four_ands_sharing_sel(spec, tmp_path: Path, require_klayout):
    library = _library()
    build_driver_slice(spec, lib=library)
    gds = tmp_path / "slice.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(render_driver_slice_lvs_schematic(spec))
    result = run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    )
    assert result.matched
    units = 4 * (spec.nand.total_fins + spec.inverter.total_fins)
    assert result.extracted_netlist.read_text().count("M$") == units


def test_a_slice_with_private_selects_does_not_match(tmp_path: Path, require_klayout):
    """SEL really is one net: a reference with four separate selects must fail."""
    library = _library()
    build_driver_slice(SMALL, lib=library)
    gds = tmp_path / "slice.gds"
    library.write_gds(str(gds))
    text = render_driver_slice_lvs_schematic(SMALL)
    lines = []
    for line in text.splitlines():
        fields = line.split()
        if line.startswith("M") and "SEL" in fields:
            index = fields[0].split("_")[-1]  # the NAND's tag: ..._<i>
            fields = [f"SEL{index}" if f == "SEL" else f for f in fields]
            line = " ".join(fields)
        lines.append(line)
    split = "\n".join(lines).replace(" SEL B0", " SEL0 SEL1 SEL2 SEL3 B0") + "\n"
    reference = tmp_path / "split.spice"
    reference.write_text(split)
    assert not run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=SMALL.cell_name, tie_bodies=True
    ).matched
