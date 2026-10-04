"""The 270 nm tristate inverter and the output latch built from 270 nm rows: geometry, then DRC and flat LVS.

The physical tests skip when KLayout or the public runset is missing;
``ASAP7_REQUIRE_TOOLS=1`` makes that a failure.
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import (
    OUTPUT_LATCH_PINS,
    OUTPUT_LATCH_270_PINS,
    OutputLatch270Spec,
    RowSupportSpec,
    TristateSpec,
    build_output_latch_270,
    build_row_support,
    build_tristate,
    render_output_latch_270_lvs_schematic,
    render_tristate_lvs_schematic,
)
from chipforge_asap7.verification.lvs import run_lvs

TRISTATES = [TristateSpec(), TristateSpec(fins=2)]


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def _gds(library, tmp_path: Path) -> Path:
    gds = tmp_path / "cell.gds"
    library.write_gds(str(gds))
    return gds


def test_tristate_is_six_gate_pitches_on_the_270_row():
    spec = TristateSpec()
    assert (spec.width, spec.height) == (324, 270)
    assert {d[2] for d in spec.devices} == {"A", "EN", "ENB"}
    with pytest.raises(ValueError, match="1 to 3 fins"):
        TristateSpec(fins=4)


def test_latch_is_the_released_srlatchs_footprint_with_the_same_pins():
    spec = OutputLatch270Spec()
    assert (spec.width, spec.height) == (432, 4 * 270)
    assert OUTPUT_LATCH_270_PINS == OUTPUT_LATCH_PINS
    # The amplifier's outputs arrive at the bottom, the bus side is the top.
    pins = spec.pin_positions
    assert all(pins[p][1][1] < 270 for p in ("QA", "QAN"))
    assert all(pins[p][1][1] > 3 * 270 for p in ("OE", "OEB", "Q"))
    # Same function as the parametric latch: 16 devices on the same gates.
    gates = sorted(d[2] for d in spec.devices)
    assert len(gates) == 16 and {"QA", "QAN", "OE", "OEB"} <= set(gates)


@pytest.mark.parametrize("spec", TRISTATES, ids=lambda s: s.cell_name)
def test_tristate_in_a_tapped_row_is_drc_clean(spec, asap7_drc):
    library = _library()
    tri = build_tristate(spec, lib=library)
    filler, tap = (build_row_support(RowSupportSpec(stack=spec.stack, kind=k), lib=library) for k in ("filler", "tap"))
    top = library.new_cell("row")
    x = 0
    for cell, width in ((filler, 108), (tri, spec.width), (filler, 108), (tap, 108), (filler, 108)):
        top.add(gdspy.CellReference(cell, origin=(x, 0)))
        x += width
    assert asap7_drc(library, top, tag="row") == []


@pytest.mark.parametrize("spec", TRISTATES, ids=lambda s: s.cell_name)
def test_tristate_matches_its_unit_fin_reference(spec, tmp_path: Path, require_klayout):
    library = _library()
    cell = build_tristate(spec, lib=library)
    reference = tmp_path / "reference.spice"
    reference.write_text(render_tristate_lvs_schematic(spec))
    result = run_lvs(_gds(library, tmp_path), reference, tmp_path / "lvs", cell_name=cell.name,
                     tie_bodies=True, timeout=600)  # fmt: skip
    assert result.matched


def test_latch_is_drc_clean(asap7_drc):
    library = _library()
    cell = build_output_latch_270(lib=library)
    assert asap7_drc(library, cell, tag="latch") == []


def test_latch_matches_its_unit_fin_reference(tmp_path: Path, require_klayout):
    spec = OutputLatch270Spec()
    library = _library()
    cell = build_output_latch_270(spec, lib=library)
    reference = tmp_path / "reference.spice"
    reference.write_text(render_output_latch_270_lvs_schematic(spec))
    result = run_lvs(_gds(library, tmp_path), reference, tmp_path / "lvs", cell_name=cell.name,
                     tie_bodies=True, flat=True, timeout=600)  # fmt: skip
    assert result.matched


def test_latch_with_a_nand_stacked_the_other_way_does_not_match(tmp_path: Path, require_klayout):
    """NAND2's series pair flipped (QAN on the rail side) fails.

    (QA and QAN exchanged would not: the cross-coupled pair is symmetric, and
    top-level pins match by topology, so that is only a relabelling.)
    """
    spec = OutputLatch270Spec()
    library = _library()
    cell = build_output_latch_270(spec, lib=library)
    flipped = []
    for line in render_output_latch_270_lvs_schematic(spec).splitlines():
        if line.startswith(("MMN2A_", "MMN2B_")):
            line = line.replace(" QAN ", " QX ").replace(" Y1 ", " QAN ").replace(" QX ", " Y1 ")
        flipped.append(line)
    reference = tmp_path / "reference.spice"
    reference.write_text("\n".join(flipped) + "\n")
    assert not run_lvs(_gds(library, tmp_path), reference, tmp_path / "lvs", cell_name=cell.name,
                       tie_bodies=True, flat=True, timeout=600).matched  # fmt: skip
