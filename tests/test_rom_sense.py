"""DRC and LVS of the NOR ROM's read-leaf row.

Same contract as the other dense-row cells: the row carries no body tap, so
alone it reports the latch-up rule and nothing else, and inside ``filler row
filler tap filler`` it reports nothing.  LVS compares it with a unit-fin
reference under ``tie_bodies=True``.  Skips when KLayout or gdscheck is
missing; ``ASAP7_REQUIRE_TOOLS=1`` makes that a failure.
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import RowSupportSpec, build_row_support
from chipforge_asap7.devices.rom_sense import RomSenseRowSpec, build_rom_sense_row
from chipforge_asap7.verification.lvs import run_lvs

NO_TAP_IN_CELL = {"ACTIVE.LUP.1"}

SPECS = [
    RomSenseRowSpec(),
    RomSenseRowSpec(leaves=4),
    RomSenseRowSpec(leaves=2, n_fins=3, p_fins=3),
    RomSenseRowSpec(leaves=2, vt="lvt"),
    RomSenseRowSpec(sample=True),
    RomSenseRowSpec(leaves=3, sample=True),
    RomSenseRowSpec(leaves=2, sample=True, vt="lvt"),
    RomSenseRowSpec(register=True, sample=True),
    RomSenseRowSpec(leaves=3, sample=True, register=True),
]


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def test_isolated_rows_only_trip_the_missing_body_tap(asap7_drc):
    library = _library()
    top = library.new_cell("rom_sense_isolated")
    cursor = 0
    for spec in SPECS:
        top.add(
            gdspy.CellReference(build_rom_sense_row(spec, lib=library), (cursor, 0))
        )
        cursor += spec.width + 1080
    assert set(asap7_drc(library, top, tag="isolated")) == NO_TAP_IN_CELL


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.cell_name)
def test_terminated_row_is_clean(spec, asap7_drc):
    library = _library()
    row = build_rom_sense_row(spec, lib=library)
    support = {
        kind: build_row_support(
            RowSupportSpec(stack=spec.stack, kind=kind), lib=library
        )
        for kind in ("filler", "tap")
    }
    top = library.new_cell("rom_sense_terminated")
    cursor = 0
    for item in ("filler", "row", "filler", "tap", "filler"):
        cell = row if item == "row" else support[item]
        top.add(gdspy.CellReference(cell, origin=(cursor, 0)))
        cursor += spec.width if item == "row" else cell.get_bounding_box()[1][0]
    assert asap7_drc(library, top, tag="terminated") == []


@pytest.mark.parametrize(
    "sample", [False, True, "register"], ids=["leaves", "sampled", "registered"]
)
def test_mirrored_rows_stack_cleanly(asap7_drc, sample):
    """Rows flipped about their shared rail, as the lane stacks them."""
    library = _library()
    spec = RomSenseRowSpec(leaves=3, sample=bool(sample), register=sample == "register")
    row = build_rom_sense_row(spec, lib=library)
    top = library.new_cell("rom_sense_stacked")
    for r in range(4):
        y = r * spec.height + (spec.height if r % 2 else 0)
        top.add(gdspy.CellReference(row, origin=(0, y), x_reflection=bool(r % 2)))
    assert set(asap7_drc(library, top, tag="stacked")) == NO_TAP_IN_CELL


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.cell_name)
def test_row_matches_its_unit_fin_reference(spec, tmp_path: Path, require_klayout):
    library = _library()
    build_rom_sense_row(spec, lib=library)
    gds = tmp_path / "row.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(spec.lvs_reference())
    result = run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    )
    assert result.matched
    fins = sum(device[5] for device in spec.devices)
    assert result.extracted_netlist.read_text().count("M$") == fins


def test_swapped_bitlines_fail(tmp_path: Path, require_klayout):
    """A reference with BLL and BLR swapped on one NAND input must not match."""
    spec = RomSenseRowSpec(leaves=2)
    library = _library()
    build_rom_sense_row(spec, lib=library)
    gds = tmp_path / "row.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(
        spec.lvs_reference().replace("MPA0_f0 D<0> BLL<0>", "MPA0_f0 D<0> BLR<0>")
    )
    assert not run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    ).matched


def test_too_short_a_sample_row_is_refused():
    with pytest.raises(ValueError, match="eight M2 levels"):
        RomSenseRowSpec(sample=True, band_height=(243, 243))


def test_sample_feedback_from_the_wrong_side_fails(tmp_path: Path, require_klayout):
    """A reference whose hold gate closes from D instead of Q must not match."""
    spec = RomSenseRowSpec(leaves=2, sample=True)
    library = _library()
    build_rom_sense_row(spec, lib=library)
    gds = tmp_path / "row.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(
        spec.lvs_reference().replace(
            "MNF0_f0 LX0 SAMPLEB Q<0>", "MNF0_f0 LX0 SAMPLEB D<0>"
        )
    )
    assert "MNF0_f0 LX0 SAMPLEB D<0>" in reference.read_text()
    assert not run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    ).matched


def test_register_needs_the_sample_latch():
    with pytest.raises(ValueError, match="register needs sample"):
        RomSenseRowSpec(register=True)


def test_register_on_the_sample_latchs_clock_fails(tmp_path: Path, require_klayout):
    """A reference whose register latch passes on SAMPLE high, like the
    sample latch, must not match: the two would be transparent together."""
    spec = RomSenseRowSpec(leaves=2, sample=True, register=True)
    library = _library()
    build_rom_sense_row(spec, lib=library)
    gds = tmp_path / "row.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    text = spec.lvs_reference()
    wrong = text.replace("MRNT0_f0 LQ0 SAMPLEB RX0", "MRNT0_f0 LQ0 SAMPLE RX0")
    assert wrong != text
    reference.write_text(wrong)
    assert not run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    ).matched
