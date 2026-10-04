"""The single-column column IO for 270 nm (6T) rows: placement and pins, then DRC and flat LVS.

The physical tests skip when KLayout or the public runset is missing;
``ASAP7_REQUIRE_TOOLS=1`` makes that a failure.
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import (
    SidewaysIoColumnSpec,
    StaggeredIoColumnSpec,
    build_sideways_io_column,
    io_column_pins,
    sideways_block_netlist,
    staggered_block_netlist,
)
from chipforge_asap7.devices.rowcell import label
from chipforge_asap7.verification.lvs import render_io_column_lvs_schematic, run_lvs

FOUR = SidewaysIoColumnSpec()
EIGHT = SidewaysIoColumnSpec(selects=8)
SIXTEEN = SidewaysIoColumnSpec(selects=16)


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


@pytest.mark.parametrize("spec", [FOUR, EIGHT, SIXTEEN], ids=lambda spec: spec.cell_name)
def test_one_column_is_narrower_than_the_staggered_two(spec):
    staggered = StaggeredIoColumnSpec(selects=spec.selects)
    assert spec.height == staggered.height == spec.selects * 270
    assert spec.width < staggered.width - 900


def test_each_row_enters_its_leaf_at_the_arrays_own_heights():
    spec = FOUR
    assert spec.bitline_ys(0) == {"BL": 186.5, "BLN": 83.5}
    assert spec.bitline_ys(1) == {"BL": 540 - 186.5, "BLN": 540 - 83.5}
    # The leaves sit half a fin pitch up: each row's bitlines are its leaf's.
    group = spec.leaves.group_pin_positions()
    for row in range(spec.selects):
        for net, y in spec.bitline_ys(row).items():
            assert group[f"{net}[{row}]"][1][1] + spec.grid_offset == y


def test_the_logic_sits_on_the_leaves_fin_grid_clear_of_their_rails():
    spec = FOUR
    assert (spec.logic_y - spec.grid_offset) % 27 == 0
    for y, _ in spec.logic_rails:
        assert all(abs(y - r) >= 3 * 27 for r, _ in spec.leaf_rails)


def test_pins_and_netlist_are_the_staggered_blocks():
    """A drop-in for `StaggeredIoColumnSpec`: same pins, same devices."""
    for spec in (FOUR, SIXTEEN):
        staggered = StaggeredIoColumnSpec(selects=spec.selects)
        assert set(spec.pin_positions) == set(io_column_pins(spec)) == set(io_column_pins(staggered))
        body = [line for line in sideways_block_netlist(spec, name="b").splitlines() if line[:1] == "M"]
        reference = [line for line in staggered_block_netlist(staggered, name="b").splitlines() if line[:1] == "M"]
        assert body == reference


def test_rejected_specs():
    with pytest.raises(ValueError, match="even"):
        SidewaysIoColumnSpec(selects=3)
    with pytest.raises(ValueError, match="land at"):
        SidewaysIoColumnSpec(bitline_entry=(190, 83.5))


@pytest.mark.parametrize("spec", [FOUR, EIGHT, SIXTEEN], ids=lambda spec: spec.cell_name)
def test_block_is_drc_clean(spec, asap7_drc):
    library = _library()
    cell = build_sideways_io_column(spec, lib=library)
    assert asap7_drc(library, cell, tag="block") == []


def test_stacked_blocks_are_drc_clean(asap7_drc):
    library = _library()
    block = build_sideways_io_column(FOUR, lib=library, draw_pin_labels=False)
    top = library.new_cell("stack")
    for b in range(3):
        top.add(gdspy.CellReference(block, origin=(0, b * FOUR.height)))
    assert asap7_drc(library, top, tag="stack") == []


@pytest.mark.parametrize("spec", [FOUR, EIGHT, SIXTEEN], ids=lambda spec: spec.cell_name)
def test_block_matches_its_flat_unit_fin_reference(spec, tmp_path: Path, require_klayout):
    library = _library()
    cell = build_sideways_io_column(spec, lib=library)
    gds = tmp_path / "block.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(render_io_column_lvs_schematic(spec, cell_name=cell.name))  # pyright: ignore[reportArgumentType]
    result = run_lvs(gds, reference, tmp_path / "lvs", cell_name=cell.name,
                     tie_bodies=True, flat=True, timeout=1800)  # fmt: skip
    assert result.matched


def test_swapped_selects_do_not_match(tmp_path: Path, require_klayout):
    """YSEL[1] and YSEL[2] exchanged in the reference fails: each leaf's select is checked."""
    spec = FOUR
    library = _library()
    cell = build_sideways_io_column(spec, lib=library)
    gds = tmp_path / "block.gds"
    library.write_gds(str(gds))
    swapped = []
    for line in render_io_column_lvs_schematic(spec, cell_name=cell.name).splitlines():  # pyright: ignore[reportArgumentType]
        if line.startswith("M"):
            line = line.replace(" YSEL[1] ", " YSELX ").replace(" YSEL[2] ", " YSEL[1] ").replace(" YSELX ", " YSEL[2] ")
        swapped.append(line)
    reference = tmp_path / "reference.spice"
    reference.write_text("\n".join(swapped) + "\n")
    assert not run_lvs(gds, reference, tmp_path / "lvs", cell_name=cell.name,
                       tie_bodies=True, flat=True, timeout=1800).matched  # fmt: skip


def test_stacked_blocks_keep_their_sense_lines_apart(tmp_path: Path, require_klayout):
    """Three blocks abutted: selects, precharge and supplies join; every bit's own nets stay its own."""
    spec, blocks = FOUR, 3
    library = _library()
    block = build_sideways_io_column(spec, lib=library, draw_pin_labels=False)
    top = library.new_cell("stack")
    pins = io_column_pins(spec)  # pyright: ignore[reportArgumentType]
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

    one = render_io_column_lvs_schematic(spec, cell_name="one").splitlines()  # pyright: ignore[reportArgumentType]
    devices = [line.split() for line in one if line.startswith("M")]
    lines = [" ".join([f"MB{b}_{t[0][1:]}", *(rename(n, b) for n in t[1:4]), t[4], *t[5:]])
             for b in range(blocks) for t in devices]  # fmt: skip
    ports = sorted(shared) + [f"B{b}_{p}" for b in range(blocks) for p in pins if p not in shared]
    reference = tmp_path / "reference.spice"
    reference.write_text(f".SUBCKT stack {' '.join(ports)}\n" + "\n".join(lines) + "\n.ENDS stack\n.END\n")
    assert run_lvs(gds, reference, tmp_path / "lvs", cell_name="stack",
                   tie_bodies=True, flat=True, timeout=3000).matched  # fmt: skip
