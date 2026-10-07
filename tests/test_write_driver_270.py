"""The write driver on one 297 nm row: the two-row cell's devices, then DRC and flat LVS.

The physical tests skip when KLayout or the public runset is missing;
``ASAP7_REQUIRE_TOOLS=1`` makes that a failure.
"""

from __future__ import annotations

from pathlib import Path

import gdspy

from chipforge_asap7.devices import (
    WRITE_DRIVER_PINS,
    RowSupportSpec,
    SenseAmpRowSpec,
    WriteDriver270Spec,
    WriteDriverSpec,
    build_row_support,
    build_sense_amp_row,
    build_write_driver_270,
    render_write_driver_270_lvs_schematic,
)
from chipforge_asap7.verification.lvs import run_lvs

SPEC = WriteDriver270Spec()
#: The compact column's amplifier: a 162 nm n band, as the driver's.
SENSE_AMP = SenseAmpRowSpec(band_height=(162, 162))


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def _lvs(tmp_path: Path, reference: str) -> bool:
    library = _library()
    cell = build_write_driver_270(SPEC, lib=library)
    gds = tmp_path / "cell.gds"
    library.write_gds(str(gds))
    path = tmp_path / "reference.spice"
    path.write_text(reference)
    return run_lvs(gds, path, tmp_path / "lvs", cell_name=cell.name, tie_bodies=True, flat=True, timeout=600).matched


def test_one_row_under_the_sense_amplifier_in_a_4_to_1_group():
    sense_amp = SENSE_AMP
    assert SPEC.height == 297
    assert SPEC.height + sense_amp.height <= 4 * 270
    assert SPEC.width <= sense_amp.width


def test_devices_and_pins_are_the_two_row_cells():
    reference = WriteDriverSpec(n_fins=3, p_fins=3, keeper_fins=1)
    assert SPEC.devices == reference.devices
    assert set(SPEC.pin_positions) == set(WRITE_DRIVER_PINS)
    assert SPEC.track_x["SA"] == SENSE_AMP.track_x["SA"]  # straight up into the amplifier's


def test_in_a_tapped_row_is_drc_clean(asap7_drc):
    library = _library()
    cell = build_write_driver_270(SPEC, lib=library)
    filler, tap = (build_row_support(RowSupportSpec(stack=SPEC.stack, kind=k), lib=library) for k in ("filler", "tap"))
    top = library.new_cell("row")
    x = 0
    for placed, width in ((filler, 108), (cell, SPEC.width), (tap, 108), (filler, 108)):
        top.add(gdspy.CellReference(placed, origin=(x, 0)))
        x += width
    assert asap7_drc(library, top, tag="row") == []


def test_mirrored_under_the_sense_amplifier_is_drc_clean(asap7_drc):
    """Its VSS rail is the amplifier's bottom one; each row has its own tap."""
    library = _library()
    sense_amp = SENSE_AMP
    driver = build_write_driver_270(SPEC, lib=library, draw_pin_labels=False)
    amplifier = build_sense_amp_row(sense_amp, lib=library, draw_pin_labels=False)
    tap = build_row_support(RowSupportSpec(stack=SPEC.stack, kind="tap"), lib=library)
    tap_amplifier = build_row_support(RowSupportSpec(stack=sense_amp.stack, kind="tap", width_cpp=2), lib=library)
    top = library.new_cell("column")
    h = SPEC.height
    top.add(gdspy.CellReference(driver, origin=(0, h), x_reflection=True))
    top.add(gdspy.CellReference(tap, origin=(SPEC.width, h), x_reflection=True))
    top.add(gdspy.CellReference(amplifier, origin=(0, h)))
    top.add(gdspy.CellReference(tap_amplifier, origin=(sense_amp.width, h)))
    assert asap7_drc(library, top, tag="column") == []


def test_matches_its_unit_fin_reference(tmp_path: Path, require_klayout):
    assert _lvs(tmp_path, render_write_driver_270_lvs_schematic(SPEC))


def test_a_keeper_on_the_wrong_gate_does_not_match(tmp_path: Path, require_klayout):
    """MPW gated by W instead of WN: the cross-couple is checked device by device."""
    lines = []
    for line in render_write_driver_270_lvs_schematic(SPEC).splitlines():
        if line.startswith("MPW_"):
            line = line.replace(" WN ", " W ", 1)
        lines.append(line)
    assert not _lvs(tmp_path, "\n".join(lines) + "\n")
