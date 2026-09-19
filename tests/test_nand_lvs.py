"""KLayout LVS regressions for the parametric NAND2 and abutted rows.

The cell carries no body tap, so these run with ``tie_bodies=True`` like the
inverter.  Skips when KLayout is missing; ``ASAP7_REQUIRE_TOOLS=1`` makes that
a failure instead.
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import NandSpec, build_nand, build_nand_row
from chipforge_asap7.verification.lvs import (
    render_nand_lvs_schematic,
    render_nand_row_lvs_schematic,
    run_lvs,
)


@pytest.fixture(autouse=True)
def _isolated_library():
    gdspy.current_library = gdspy.GdsLibrary()


def _write(cell_maker, path: Path):
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    cell = cell_maker(library)
    library.write_gds(str(path))
    return cell


@pytest.mark.parametrize(
    "spec",
    [
        NandSpec(rows=((14, 7),), fingers=2),  # the released post-decode tile
        NandSpec(rows=((3, 3),), fingers=1, abut=False),  # NAND2xp33 footprint
        NandSpec(rows=((6, 4),), fingers=4, vt="lvt"),  # both inputs tied on M2
        NandSpec(rows=((6, 4),), fingers=3, abut=False),
    ],
)
def test_nand_matches_its_unit_fin_reference(spec, tmp_path: Path, require_klayout):
    gds = tmp_path / "nand.gds"
    _write(lambda lib: build_nand(spec, lib=lib), gds)
    reference = tmp_path / "reference.spice"
    reference.write_text(render_nand_lvs_schematic(spec))
    result = run_lvs(
        gds, reference, tmp_path / "match", cell_name=spec.cell_name, tie_bodies=True
    )
    assert result.matched
    assert result.extracted_netlist.read_text().count("M$") == spec.total_fins


def test_wrong_sizing_and_a_shorted_input_both_fail(tmp_path: Path, require_klayout):
    spec = NandSpec(rows=((6, 4),), fingers=2)
    gds = tmp_path / "nand.gds"
    _write(lambda lib: build_nand(spec, lib=lib), gds)

    wrong = tmp_path / "wrong.spice"
    wrong.write_text(
        render_nand_lvs_schematic(
            NandSpec(rows=((6, 5),), fingers=2), cell_name=spec.cell_name
        )
    )
    assert not run_lvs(
        gds, wrong, tmp_path / "wrong", cell_name=spec.cell_name, tie_bodies=True
    ).matched

    # A topology error has to fail independently of sizing: short input A to
    # the output in an otherwise correctly sized reference.  (Swapping the A
    # and B pin names is not one -- KLayout pairs pins by topology, and the
    # renamed reference is still a NAND2 with the same series order.)
    reference = render_nand_lvs_schematic(spec)
    shorted = tmp_path / "shorted.spice"
    shorted.write_text(
        "\n".join(
            " ".join("Y" if field == "A" else field for field in line.split())
            if line.startswith(("MN", "MP"))
            else line
            for line in reference.splitlines()
        )
        + "\n"
    )
    assert not run_lvs(
        gds, shorted, tmp_path / "shorted", cell_name=spec.cell_name, tie_bodies=True
    ).matched


@pytest.mark.parametrize("count", [1, 2, 4])
def test_abutted_row_is_count_independent_nands(count, tmp_path: Path, require_klayout):
    """Shared edge columns are sources, so butting tiles cannot short outputs."""
    spec = NandSpec(rows=((6, 4),), fingers=2)
    gds = tmp_path / f"row{count}.gds"
    row = _write(lambda lib: build_nand_row(spec, count, lib=lib), gds)
    reference = tmp_path / f"row{count}.spice"
    reference.write_text(render_nand_row_lvs_schematic(spec, count, cell_name=row.name))
    result = run_lvs(
        gds, reference, tmp_path / f"row{count}", cell_name=row.name, tie_bodies=True
    )
    assert result.matched
    assert result.extracted_netlist.read_text().count("M$") == count * spec.total_fins
