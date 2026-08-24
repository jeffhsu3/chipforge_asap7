"""Electrical proof that a tap ties bodies and that a decap is two capacitors.

Skips when KLayout is missing; set ``ASAP7_REQUIRE_TOOLS=1`` to make that a
failure instead.
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import (
    InverterSpec,
    RowStack,
    RowSupportSpec,
    build_inverter_row,
    build_row_support,
)
from chipforge_asap7.layout import LAYERS
from chipforge_asap7.verification.lvs import (
    render_inverter_row_lvs_schematic,
    render_row_support_lvs_schematic,
    run_lvs,
)

INVERTER = InverterSpec(rows=((4, 6),))
_PIN = LAYERS["M1_PIN"]


@pytest.fixture(autouse=True)
def _isolated_library():
    gdspy.current_library = gdspy.GdsLibrary()


def _row(plan: str, path: Path, count: int = 4):
    """Abut `plan` and label the rails; ``r`` is the logic row."""
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    row = build_inverter_row(INVERTER, count, lib=library)
    top = library.new_cell("terminated_row")
    built: dict[str, tuple] = {}
    cursor = 0
    for item in plan:
        if item == "r":
            top.add(gdspy.CellReference(row, origin=(cursor, 0)))
            cursor += INVERTER.row_width(count)
            continue
        kind = {"f": "filler", "t": "tap", "d": "decap"}[item]
        if kind not in built:
            spec = RowSupportSpec(stack=INVERTER.stack, kind=kind)
            built[kind] = (
                spec,
                build_row_support(spec, lib=library, draw_pin_labels=False),
            )
        spec, cell = built[kind]
        top.add(gdspy.CellReference(cell, origin=(cursor, 0)))
        cursor += spec.width
    for y_rail, net in INVERTER.rails:
        top.add(
            gdspy.Label(
                net,
                (cursor / 2, y_rail),
                layer=_PIN["layer"],
                texttype=_PIN["datatype"],
            )
        )
    library.write_gds(str(path))
    return top


def test_the_tap_is_what_ties_the_bodies(tmp_path: Path, require_klayout):
    """Run LVS with the deck's global body tie switched off.

    Normally `tie_bodies` declares the well to be VDD and the substrate VSS,
    which means a row with no tap in it still passes -- the tie is asserted,
    not extracted.  With it off, every body terminal has to reach its rail
    through drawn geometry.  A terminated row does; the same row without its
    tap does not.  That is the difference between a tap that works and a tap
    that merely looks right.
    """
    reference = tmp_path / "row.spice"
    reference.write_text(
        render_inverter_row_lvs_schematic(INVERTER, 4, cell_name="terminated_row")
    )

    with_tap = tmp_path / "with_tap.gds"
    _row("frftf", with_tap)
    assert run_lvs(
        with_tap,
        reference,
        tmp_path / "with_tap",
        cell_name="terminated_row",
        tie_bodies=False,
    ).matched

    without_tap = tmp_path / "without_tap.gds"
    _row("frf", without_tap)
    assert not run_lvs(
        without_tap,
        reference,
        tmp_path / "without_tap",
        cell_name="terminated_row",
        tie_bodies=False,
    ).matched

    # ... and the control: declaring the ties hides the difference entirely.
    assert run_lvs(
        without_tap,
        reference,
        tmp_path / "declared",
        cell_name="terminated_row",
        tie_bodies=True,
    ).matched


@pytest.mark.parametrize(
    "spec",
    [
        RowSupportSpec(stack=RowStack(rows=((4, 6),)), kind="decap", width_cpp=6),
        RowSupportSpec(stack=RowStack(rows=((4, 6),)), kind="decap", width_cpp=3),
        RowSupportSpec(stack=RowStack(rows=((2, 3), (5, 5))), kind="decap"),
    ],
    ids=["6cpp", "3cpp", "two_rows"],
)
def test_a_decap_extracts_as_back_to_back_capacitors(
    spec, tmp_path: Path, require_klayout
):
    """Both bands of a row on one floating gate, each plate's S and D on a rail.

    The two-row case is the one that pins down the plate net being per *row*:
    the poly is cut at every rail, so stacked rows cannot share a gate.
    """
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    build_row_support(spec, lib=library)
    gds = tmp_path / "decap.gds"
    library.write_gds(str(gds))

    reference = tmp_path / "decap.spice"
    reference.write_text(render_row_support_lvs_schematic(spec))
    result = run_lvs(
        gds, reference, tmp_path / "match", cell_name=spec.cell_name, tie_bodies=True
    )
    assert result.matched
    expected = sum(count for _band, count in spec.devices)
    assert result.extracted_netlist.read_text().count("M$") == expected


def test_a_decap_reference_will_not_match_the_wrong_plate_count(
    tmp_path: Path, require_klayout
):
    stack = RowStack(rows=((4, 6),))
    spec = RowSupportSpec(stack=stack, kind="decap", width_cpp=6)
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    build_row_support(spec, lib=library)
    gds = tmp_path / "decap.gds"
    library.write_gds(str(gds))

    reference = tmp_path / "wrong.spice"
    reference.write_text(
        render_row_support_lvs_schematic(
            RowSupportSpec(stack=stack, kind="decap", width_cpp=5),
            cell_name=spec.cell_name,
        )
    )
    assert not run_lvs(
        gds, reference, tmp_path / "wrong", cell_name=spec.cell_name, tie_bodies=True
    ).matched


def test_a_tap_and_a_filler_have_no_devices_to_render():
    for kind in ("tap", "filler"):
        spec = RowSupportSpec(stack=RowStack(rows=((4, 6),)), kind=kind)
        assert spec.devices == ()
        text = render_row_support_lvs_schematic(spec)
        assert ".SUBCKT" in text and "nmos" not in text
