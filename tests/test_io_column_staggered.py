"""The staggered column IO for 270 nm (6T) rows: placement and pins, then DRC and flat LVS.

The physical tests skip when KLayout or the public runset is missing;
``ASAP7_REQUIRE_TOOLS=1`` makes that a failure.
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import (
    BitlineMuxSpec,
    StaggeredIoColumnSpec,
    build_staggered_io_column,
    io_column_pins,
    staggered_block_netlist,
)
from chipforge_asap7.devices.rowcell import label
from chipforge_asap7.verification.lvs import render_io_column_lvs_schematic, run_lvs

FOUR = StaggeredIoColumnSpec()
EIGHT = StaggeredIoColumnSpec(selects=8)
SIXTEEN = StaggeredIoColumnSpec(selects=16)


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def test_leaves_alternate_columns_on_the_rows_they_serve():
    spec = FOUR
    assert (spec.width, spec.height) == (3834, 1080)
    assert spec.column_x == (108, 864) and spec.logic_x == 1620
    # Even rows in column E, raised 40.5 nm; odd rows in column O, mirrored.
    assert spec.leaf_origin(0) == (108, 40.5, False)
    assert spec.leaf_origin(1) == (864, 540 - 40.5, True)
    assert spec.leaf_span(1) == (202.5, 499.5)
    assert spec.leaf_origin(2) == (108, 580.5, False)
    # Every leaf covers its own row's bitlines.
    for row in range(spec.selects):
        y0, y1 = spec.leaf_span(row)
        assert all(y0 < y < y1 for y in spec.bitline_ys(row).values())
    # Odd rows are the array's rows mirrored.
    assert spec.bitline_ys(1) == {"BL": 353.5, "BLN": 456.5}
    # Column E answers YSEL[0], [2]; column O YSEL[1], [3].
    assert [spec.global_select("E", k) for k in range(2)] == [0, 2]
    assert [spec.global_select("O", k) for k in range(2)] == [1, 3]


def test_joins_keep_clear_of_the_bitlines_crossing_column_e():
    for spec in (FOUR, EIGHT, SIXTEEN):
        crossing = [y for row in spec.rows_of("O") for y in spec.bitline_ys(row).values()]
        for y in spec.joins.values():
            assert all(abs(y - b) >= 48 for b in crossing)
            assert all(abs(y - s) >= 48 for s in spec.sense_line_ys)


def test_pins_are_io_column_pins():
    spec = FOUR
    positions = spec.pin_positions
    assert set(positions) == set(io_column_pins(spec))
    assert positions["BL[1]"] == ("M2", (17, 353.5))
    assert positions["YSEL[1]"][1][0] == spec.track_x("O", "YSEL[0]")
    assert positions["YSEL[2]"][1][0] == spec.track_x("E", "YSEL[1]")
    netlist = staggered_block_netlist(spec)
    assert netlist.count("\nM") == 4 * 6 + sum(
        len(cell.devices) for cell in (spec.sense_amp, spec.write_driver, spec.output_latch)
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"selects": 3}, "even"),
        ({"leaf_offset": 27}, "odd multiple"),
        ({"leaf": BitlineMuxSpec(rows=2)}, "one row"),
        ({"row_pitch": 135}, "does not fit"),
    ],
)
def test_impossible_blocks_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        StaggeredIoColumnSpec(**kwargs)



@pytest.mark.parametrize("spec", [FOUR, EIGHT, SIXTEEN], ids=lambda spec: spec.cell_name)
def test_block_is_drc_clean(spec, asap7_drc):
    library = _library()
    cell = build_staggered_io_column(spec, lib=library)
    assert asap7_drc(library, cell, tag="block") == []


def test_stacked_blocks_are_drc_clean(asap7_drc):
    library = _library()
    block = build_staggered_io_column(FOUR, lib=library, draw_pin_labels=False)
    top = library.new_cell("stack")
    for b in range(3):
        top.add(gdspy.CellReference(block, origin=(0, b * FOUR.height)))
    assert asap7_drc(library, top, tag="stack") == []


@pytest.mark.parametrize("spec", [FOUR, EIGHT, SIXTEEN], ids=lambda spec: spec.cell_name)
def test_block_matches_its_flat_unit_fin_reference(spec, tmp_path: Path, require_klayout):
    library = _library()
    cell = build_staggered_io_column(spec, lib=library)
    gds = tmp_path / "block.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(render_io_column_lvs_schematic(spec, cell_name=cell.name))
    result = run_lvs(gds, reference, tmp_path / "lvs", cell_name=cell.name,
                     tie_bodies=True, flat=True, timeout=1800)  # fmt: skip
    assert result.matched


def test_swapped_column_selects_do_not_match(tmp_path: Path, require_klayout):
    """YSEL[1] and YSEL[2] exchanged in the reference fails: each column's selects are checked."""
    spec = FOUR
    library = _library()
    cell = build_staggered_io_column(spec, lib=library)
    gds = tmp_path / "block.gds"
    library.write_gds(str(gds))
    text = render_io_column_lvs_schematic(spec, cell_name=cell.name)
    swapped = []
    for line in text.splitlines():
        if line.startswith("M"):
            line = line.replace(" YSEL[1] ", " YSELX ").replace(" YSEL[2] ", " YSEL[1] ")
            line = line.replace(" YSELX ", " YSEL[2] ")
        swapped.append(line)
    reference = tmp_path / "reference.spice"
    reference.write_text("\n".join(swapped) + "\n")
    assert not run_lvs(gds, reference, tmp_path / "lvs", cell_name=cell.name,
                       tie_bodies=True, flat=True, timeout=1800).matched  # fmt: skip


def test_stacked_blocks_keep_their_sense_lines_apart(tmp_path: Path, require_klayout):
    """Three blocks abutted: selects, precharge and supplies join; every bit's own nets stay its own."""
    spec, blocks = FOUR, 3
    library = _library()
    block = build_staggered_io_column(spec, lib=library, draw_pin_labels=False)
    top = library.new_cell("stack")
    pins = io_column_pins(spec)
    shared = {"VDD", "VSS", "PRECHN"} | {f"{p}[{i}]" for p in ("YSEL", "YSELN") for i in range(4)}
    for b in range(blocks):
        top.add(gdspy.CellReference(block, origin=(0, b * spec.height)))
        for pin, (metal, (x, y)) in spec.pin_positions.items():
            if pin in shared and b:
                continue
            label(top, pin if pin in shared else f"B{b}_{pin}", metal, (x, y + b * spec.height))
    gds = tmp_path / "stack.gds"
    library.write_gds(str(gds))

    def rename(token: str, b: int) -> str:
        return token if token in shared else f"B{b}_{token}"

    one = render_io_column_lvs_schematic(spec, cell_name="one").splitlines()
    devices = [line.split() for line in one if line.startswith("M")]
    lines = [
        " ".join([f"MB{b}_{t[0][1:]}", *(rename(n, b) for n in t[1:4]), t[4], *t[5:]])
        for b in range(blocks)
        for t in devices
    ]
    ports = sorted(shared) + [f"B{b}_{p}" for b in range(blocks) for p in pins if p not in shared]
    reference = tmp_path / "reference.spice"
    reference.write_text(f".SUBCKT stack {' '.join(ports)}\n" + "\n".join(lines) + "\n.ENDS stack\n.END\n")
    assert run_lvs(gds, reference, tmp_path / "lvs", cell_name="stack",
                   tie_bodies=True, flat=True, timeout=3000).matched  # fmt: skip
