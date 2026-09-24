"""Public-runset DRC and flat KLayout LVS of the column IO block.

The block carries its own tap, so alone it is DRC clean; without one it
reports the latch-up rule and nothing else.  Skips when the tools are
missing; ``ASAP7_REQUIRE_TOOLS=1`` makes that a failure.
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import (
    BitlineMuxSpec,
    IoColumnSpec,
    OutputLatchSpec,
    SenseAmpRowSpec,
    build_io_column,
)
from chipforge_asap7.verification.lvs import render_io_column_lvs_schematic, run_lvs

PORT_A = IoColumnSpec()
PORT_B = IoColumnSpec(
    mux=BitlineMuxSpec(rows=2, bitline_entry=(510, 78), bitline_layer="M4")
)
EIGHT = IoColumnSpec(
    mux=BitlineMuxSpec(rows=2, selects=8, bitline_entry=(348.5, 245.5)),
    sense_amp=SenseAmpRowSpec(n_fingers=4, tail_fingers=4),
    output_latch=OutputLatchSpec(fingers=4),
)
# Port B of two banks facing each other: one amplifier, driver and latch for
# a group on each side (the OpenFinRAM 8T bitcell's M4 bitlines, on its fin grid).
TWO_SIDED = IoColumnSpec(
    mux=BitlineMuxSpec(rows=2, bitline_entry=(510, 78), bitline_layer="M4", grid_offset=13.5),
    one_sense_phase=True,
    two_sided=True,
)


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


@pytest.mark.parametrize(
    "spec", [PORT_A, PORT_B, EIGHT, TWO_SIDED], ids=lambda spec: spec.cell_name
)
def test_block_is_drc_clean_with_its_tap(spec, asap7_drc):
    library = _library()
    cell = build_io_column(spec, lib=library)
    assert asap7_drc(library, cell, tag="block") == []


def test_without_a_tap_only_the_latch_up_rule_fires(asap7_drc):
    library = _library()
    cell = build_io_column(IoColumnSpec(tap=False), lib=library)
    assert set(asap7_drc(library, cell, tag="notap")) == {"ACTIVE.LUP.1"}


@pytest.mark.parametrize(
    "spec", [PORT_A, PORT_B, EIGHT, TWO_SIDED], ids=lambda spec: spec.cell_name
)
def test_block_matches_its_flat_unit_fin_reference(
    spec, tmp_path: Path, require_klayout
):
    library = _library()
    cell = build_io_column(spec, lib=library)
    gds = tmp_path / "block.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(render_io_column_lvs_schematic(spec, cell_name=cell.name))
    result = run_lvs(
        gds,
        reference,
        tmp_path / "lvs",
        cell_name=cell.name,
        tie_bodies=True,
        flat=True,
        timeout=1800,
    )
    assert result.matched
    groups = 2 if spec.two_sided else 1
    fins = groups * spec.mux.selects * sum(f for *_, f in spec.mux.devices) + sum(
        sum(f for *_, f in part.devices)
        for part in (spec.sense_amp, spec.write_driver, spec.output_latch)
    )
    assert result.extracted_netlist.read_text().count("M$") == fins


def test_the_outputs_crossed_do_not_match(tmp_path: Path, require_klayout):
    """The reference with QA and QAN swapped between amplifier and latch fails: the routes are checked."""
    spec = PORT_A
    library = _library()
    cell = build_io_column(spec, lib=library)
    gds = tmp_path / "block.gds"
    library.write_gds(str(gds))
    text = render_io_column_lvs_schematic(spec, cell_name=cell.name)
    crossed = []
    for line in text.splitlines():
        if line.startswith("Mol_"):
            line = (
                line.replace(" QA ", " QAX ")
                .replace(" QAN ", " QA ")
                .replace(" QAX ", " QAN ")
            )
        crossed.append(line)
    reference = tmp_path / "reference.spice"
    reference.write_text("\n".join(crossed) + "\n")
    assert not run_lvs(
        gds,
        reference,
        tmp_path / "lvs",
        cell_name=cell.name,
        tie_bodies=True,
        flat=True,
        timeout=1800,
    ).matched


def test_a_second_group_on_its_own_sense_lines_does_not_match(tmp_path: Path, require_klayout):
    """The shared amplifier is checked: a reference whose second group has sense lines of its own fails."""
    spec = TWO_SIDED
    library = _library()
    cell = build_io_column(spec, lib=library)
    gds = tmp_path / "block.gds"
    library.write_gds(str(gds))
    text = render_io_column_lvs_schematic(spec, cell_name=cell.name)
    split = [
        line.replace(" SA ", " SA_OWN ").replace(" SAN ", " SAN_OWN ")
        if line.split(" ", 1)[0].split("_")[0].endswith("R") else line
        for line in text.splitlines()
    ]
    assert split != text.splitlines()
    reference = tmp_path / "reference.spice"
    reference.write_text("\n".join(split) + "\n")
    assert not run_lvs(
        gds,
        reference,
        tmp_path / "lvs",
        cell_name=cell.name,
        tie_bodies=True,
        flat=True,
        timeout=1800,
    ).matched
