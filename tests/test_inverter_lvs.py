"""KLayout LVS regressions for the stacked-band inverter and abutted rows.

Skips when KLayout is missing; set ``ASAP7_REQUIRE_TOOLS=1`` to make that a
failure instead.

The inverter carries no body tap, so these run with ``tie_bodies=True``: the
well and substrate reach their supplies through the tap rows of the design
that places the cell, exactly as for a released logic cell.
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import InverterSpec, build_inverter, build_inverter_row
from chipforge_asap7.verification.lvs import (
    render_inverter_lvs_schematic,
    render_inverter_row_lvs_schematic,
    run_lvs,
)

RELEASED = InverterSpec(rows=((18, 18), (13, 13)), fingers=2)


@pytest.fixture(autouse=True)
def _isolated_library():
    gdspy.current_library = gdspy.GdsLibrary()


def _write(cell_maker, path: Path):
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    cell = cell_maker(library)
    library.write_gds(str(path))
    return cell


def test_isolated_inverter_lvs_matches_and_detects_wrong_sizing(
    tmp_path: Path, require_klayout
):
    spec = InverterSpec(rows=((12, 12), (8, 8)), abut=False)
    gds = tmp_path / "inverter.gds"
    _write(lambda lib: build_inverter(spec, lib=lib), gds)

    reference = tmp_path / "reference.spice"
    reference.write_text(render_inverter_lvs_schematic(spec))
    matched = run_lvs(
        gds, reference, tmp_path / "match", cell_name=spec.cell_name, tie_bodies=True
    )
    assert matched.matched
    # Two rows means two VSS rails; they are one net only because both are
    # labelled, and this count is what breaks if a rail label is dropped.
    assert matched.extracted_netlist.read_text().count("M$") == spec.total_fins

    wrong = tmp_path / "wrong.spice"
    wrong.write_text(
        render_inverter_lvs_schematic(
            InverterSpec(rows=((12, 12), (8, 7)), abut=False),
            cell_name=spec.cell_name,
        )
    )
    assert not run_lvs(
        gds, wrong, tmp_path / "mismatch", cell_name=spec.cell_name, tie_bodies=True
    ).matched

    # A topology error has to fail independently of sizing: short the input to
    # the output in an otherwise correctly sized reference.
    shorted = tmp_path / "shorted.spice"
    shorted.write_text(
        "\n".join(
            " ".join("Y" if field == "A" else field for field in line.split())
            if line.startswith("Mb")
            else line
            for line in reference.read_text().splitlines()
        )
        + "\n"
    )
    assert not run_lvs(
        gds, shorted, tmp_path / "short", cell_name=spec.cell_name, tie_bodies=True
    ).matched


def test_four_finger_output_is_tied_on_m2_not_left_split(
    tmp_path: Path, require_klayout
):
    """Folding past two fingers gives a row two drain columns.

    Without the M2 tie they would be two separate outputs, which LVS sees as
    two half-strength inverters sharing an input rather than one device.
    """
    spec = InverterSpec(rows=((2, 3),), fingers=4, abut=False)
    assert len(spec.drain_xs) == 2
    gds = tmp_path / "wide.gds"
    _write(lambda lib: build_inverter(spec, lib=lib), gds)
    reference = tmp_path / "wide.spice"
    reference.write_text(render_inverter_lvs_schematic(spec))
    assert run_lvs(
        gds, reference, tmp_path / "wide", cell_name=spec.cell_name, tie_bodies=True
    ).matched


@pytest.mark.parametrize("count", [1, 2, 4])
def test_abutted_row_lvs_proves_the_interleave(count, tmp_path: Path, require_klayout):
    """Every tile's shared edge gate has to become its neighbour's finger.

    If it did not, the row would extract with the wrong device count, a
    shorted pair of outputs, or an undriven finger -- all three are LVS
    failures against a reference of `count` independent inverters.  An odd
    `count` additionally leaves one unpaired edge gate on floating poly, which
    the reference models rather than ignores.
    """
    spec = InverterSpec(rows=((4, 6),), vt="lvt")
    gds = tmp_path / f"row{count}.gds"
    row = _write(lambda lib: build_inverter_row(spec, count, lib=lib), gds)

    reference = tmp_path / f"row{count}.spice"
    reference.write_text(
        render_inverter_row_lvs_schematic(spec, count, cell_name=row.name)
    )
    result = run_lvs(
        gds, reference, tmp_path / f"row{count}", cell_name=row.name, tie_bodies=True
    )
    assert result.matched

    units = result.extracted_netlist.read_text().count("M$")
    dangling = 0 if count % 2 == 0 else sum(band.fins for band in spec.bands)
    assert units == count * spec.total_fins + dangling


def test_a_row_reference_will_not_match_a_row_of_the_wrong_length(
    tmp_path: Path, require_klayout
):
    spec = InverterSpec(rows=((4, 6),))
    gds = tmp_path / "row2.gds"
    row = _write(lambda lib: build_inverter_row(spec, 2, lib=lib), gds)
    reference = tmp_path / "row4.spice"
    reference.write_text(render_inverter_row_lvs_schematic(spec, 4, cell_name=row.name))
    assert not run_lvs(
        gds, reference, tmp_path / "wrong_len", cell_name=row.name, tie_bodies=True
    ).matched
