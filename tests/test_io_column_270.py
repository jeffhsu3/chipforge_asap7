"""The single-column column IO for 270 nm (6T) rows: placement and pins, then DRC and flat LVS.

The physical tests skip when KLayout or the public runset is missing;
``ASAP7_REQUIRE_TOOLS=1`` makes that a failure.
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import (
    OutputLatch270Spec,
    SidewaysIoColumnSpec,
    StaggeredIoColumnSpec,
    build_sideways_io_column,
    io_column_pins,
    sideways_block_netlist,
    staggered_block_netlist,
)
from chipforge_asap7.devices.rowcell import label
from chipforge_asap7.verification.lvs import render_io_column_lvs_schematic, run_lvs

FOUR = SidewaysIoColumnSpec(compact=True)
FOUR_WIDE = SidewaysIoColumnSpec()  # the two-row cells side by side
EIGHT_LY = SidewaysIoColumnSpec(selects=8, local_ysel=True)
SIXTEEN_LY = SidewaysIoColumnSpec(selects=16, local_ysel=True)
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
    assert spec.width <= staggered.width - 800


def test_each_row_enters_its_leaf_at_the_arrays_own_heights():
    spec = FOUR
    assert spec.bitline_ys(0) == {"BL": 186.5, "BLN": 83.5}
    assert spec.bitline_ys(1) == {"BL": 540 - 186.5, "BLN": 540 - 83.5}
    # The leaves sit half a fin pitch up: each row's bitlines are its leaf's.
    group = spec.leaves.group_pin_positions()
    for row in range(spec.selects):
        for net, y in spec.bitline_ys(row).items():
            assert group[f"{net}[{row}]"][1][1] + spec.grid_offset == y


@pytest.mark.parametrize("spec", [FOUR, FOUR_WIDE], ids=("compact", "wide"))
def test_the_logic_sits_on_the_leaves_fin_grid_clear_of_their_rails(spec):
    assert (spec.logic_y - spec.grid_offset) % 27 == 0
    clear = (2 if spec.is_compact else 3) * 27
    for y, _ in spec.logic_rails:
        assert all(abs(y - r) >= clear for r, _ in spec.leaf_rails)


def _devices(netlist: str) -> list[str]:
    return [line for line in netlist.splitlines() if line[:1] == "M"]


def test_pins_and_netlist_are_the_staggered_blocks():
    """A drop-in for `StaggeredIoColumnSpec`: same pins, same devices."""
    for spec in (FOUR_WIDE, SIXTEEN):
        staggered = StaggeredIoColumnSpec(selects=spec.selects)
        assert set(spec.pin_positions) == set(io_column_pins(spec)) == set(io_column_pins(staggered))
        body = _devices(sideways_block_netlist(spec, name="b"))
        assert body == _devices(staggered_block_netlist(staggered, name="b"))


def test_compact_block_is_the_staggered_one_but_for_its_latch():
    """Same pins, the same leaves, amplifier and write driver; the latch is the 270 nm one's devices."""
    staggered = StaggeredIoColumnSpec(selects=4)
    assert set(FOUR.pin_positions) == set(io_column_pins(FOUR)) == set(io_column_pins(staggered))
    body = _devices(sideways_block_netlist(FOUR, name="b"))
    reference = _devices(staggered_block_netlist(staggered, name="b"))
    assert [d for d in body if not d.startswith("Mol_")] == [d for d in reference if not d.startswith("Mol_")]
    assert len([d for d in body if d.startswith("Mol_")]) == len(OutputLatch270Spec().devices)


def test_compact_block_puts_the_driver_under_the_amplifier_and_the_latch_beside():
    spec = FOUR
    assert spec.is_compact and not FOUR_WIDE.is_compact and not EIGHT.is_compact
    place = spec.placements
    wd_x, wd_y = place["write_driver"]
    sa_x, sa_y = place["sense_amp"]
    assert wd_x == sa_x and sa_y == wd_y + 270
    assert spec.logic_top <= spec.height
    # The latch on the leaves' rails, past the logic column's taps and the inner straps over them.
    assert place["output_latch"] == (spec.latch_x, spec.grid_offset)
    assert all(sa_x + spec.sense_amp.width < x < spec.latch_x for x in spec.inner_straps.values())
    assert spec.width == spec.latch_x + 432 <= FOUR_WIDE.width - 900
    with pytest.raises(ValueError, match="4:1 block or taller"):
        SidewaysIoColumnSpec(selects=2, compact=True)


def test_deep_muxes_stack_the_logic_in_one_column():
    """8:1 and up are tall enough for driver, amplifier and latch one above the other."""
    assert not FOUR.stacked and EIGHT.stacked and SIXTEEN.stacked
    for spec in (EIGHT, SIXTEEN):
        place = spec.placements
        xs = {place[k][0] for k in ("write_driver", "sense_amp", "output_latch")}
        assert len(xs) == 1
        ys = [place[k][1] for k in ("write_driver", "sense_amp", "output_latch")]
        assert ys == [spec.logic_y + k * spec.pair_height for k in range(3)]
        assert spec.logic_top <= spec.height
        assert spec.width <= spec.logic_x + spec.output_latch.width + 108
    assert SidewaysIoColumnSpec(selects=8, stack_logic=False).width > EIGHT.width + 1000


def test_rejected_specs():
    with pytest.raises(ValueError, match="even"):
        SidewaysIoColumnSpec(selects=3)
    with pytest.raises(ValueError, match="land at"):
        SidewaysIoColumnSpec(bitline_entry=(190, 83.5))


def test_a_local_ysel_narrows_the_deep_blocks_and_drops_their_ysel_pins():
    for spec, plain, saved in ((EIGHT_LY, EIGHT, 216), (SIXTEEN_LY, SIXTEEN, 540)):
        assert spec.width == plain.width - saved
        pins = set(io_column_pins(spec))
        assert set(spec.pin_positions) == pins
        assert pins == {p for p in io_column_pins(plain) if not p.startswith("YSEL[")}


@pytest.mark.parametrize("spec", [FOUR, FOUR_WIDE, EIGHT, SIXTEEN, EIGHT_LY, SIXTEEN_LY], ids=lambda spec: spec.cell_name)
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


@pytest.mark.parametrize("spec", [FOUR, FOUR_WIDE, EIGHT, SIXTEEN, EIGHT_LY, SIXTEEN_LY], ids=lambda spec: spec.cell_name)
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
