"""Geometry and netlist of the 270 nm (6T row) bitline leaf and its stacked group."""

from __future__ import annotations

import gdspy
import pytest

from chipforge_asap7.devices import (
    BITLINE_MUX_PINS,
    BitlineMuxSpec,
    SidewaysMuxSpec,
    build_sideways_mux,
    build_sideways_mux_group,
    sideways_mux_group_pins,
)


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def test_the_leaf_is_one_6t_row_tall():
    spec = SidewaysMuxSpec()
    cell = build_sideways_mux(spec, lib=_library())
    (_, y0), (_, y1) = cell.get_bounding_box()
    assert spec.height == 270
    # Gates run 5 nm past the rails and the gate cuts 22 nm, as in a 7.5-track cell.
    assert (y0, y1) == (-22, 292)


@pytest.mark.parametrize("selects, width", [(1, 378), (2, 432), (4, 486), (8, 756), (16, 1350)])
def test_the_leaf_widens_by_two_tracks_a_select(selects, width):
    spec = SidewaysMuxSpec(selects=selects)
    assert spec.width == width
    tracks = spec.track_x
    assert tracks["PRECHN"] == 51 and tracks["YSELN[0]"] == 123
    assert tracks["SA"] - tracks["SAN"] == 36
    assert tracks["SA"] + 9 <= width  # the last track inside the edge


@pytest.mark.parametrize("selects, width", [(4, 540), (8, 540), (16, 756)])
def test_a_local_ysel_drops_the_ysel_tracks(selects, width):
    """Only YSELN crosses the column: from 8:1 the leaf is narrower; at 4:1 the inverter sets it."""
    spec = SidewaysMuxSpec(selects=selects, local_ysel=True)
    assert spec.width == width
    assert not any(net.startswith("YSEL[") for net in spec.track_x)
    assert "YSEL" not in spec.pins and "YSEL" not in spec.pin_positions
    assert set(spec.devices) - set(SidewaysMuxSpec(selects=selects).devices) == {
        ("MNI", "YSEL", "YSELN", "VSS", "n", 3), ("MPI", "YSEL", "YSELN", "VDD", "p", 3)}  # fmt: skip
    assert not any(pin.startswith("YSEL[") for pin in spec.group_pin_positions())


def test_predecode_shares_eight_select_tracks_at_16_to_1():
    """Two one-hot groups instead of 16 YSELN: the NAND2 + inverter set the width, 648 nm."""
    spec = SidewaysMuxSpec(selects=16, select=6, predecode=True)
    assert spec.width == 648
    assert spec.group_nets == (*(f"YPA[{k}]" for k in range(4)), *(f"YPB[{k}]" for k in range(4)))
    assert spec.select_inputs == ("YPA[2]", "YPB[1]")
    assert not any(net.startswith(("YSEL[", "YSELN[")) for net in spec.track_x)
    assert "YSEL" not in spec.pins and "YSELN" not in spec.pins
    gates = {d[2] for d in spec.devices}
    assert {"YPA[2]", "YPB[1]", "YSELN", "YSEL"} <= gates
    with pytest.raises(ValueError, match="fours"):
        SidewaysMuxSpec(selects=4, predecode=True)


def test_the_bitlines_land_where_the_6t_cell_puts_them():
    # BL above BLN, half a fin pitch up on the array's grid: the leaf's M2 bars.
    spec = SidewaysMuxSpec()
    assert spec.pin_positions["BL"] == ("M2", (9, 173))
    assert spec.pin_positions["BLN"] == ("M2", (9, 70))
    assert tuple(y + spec.grid_offset for y in (173, 70)) == spec.bitline_entry


def test_a_group_swaps_alternate_leaves_as_the_array_mirrors_its_rows():
    pins = SidewaysMuxSpec().group_pin_positions()
    assert pins["BL[0]"][1][1] == 173 and pins["BLN[0]"][1][1] == 70
    # Leaf 1 is swapped, not reflected: BL on its lower strip, still on the grid.
    assert pins["BL[1]"][1][1] == 270 + 70 and pins["BLN[1]"][1][1] == 270 + 173
    # A mirrored array row's bitlines, half a fin pitch up: where they land.
    offset = SidewaysMuxSpec().grid_offset
    assert pins["BL[1]"][1][1] + offset == 540 - 186.5
    assert pins["BLN[1]"][1][1] + offset == 540 - 83.5


def test_a_swapped_leaf_swaps_its_nets_and_rails():
    plain, swapped = SidewaysMuxSpec(), SidewaysMuxSpec(swapped=True)
    assert swapped.devices == plain.devices
    for a, b in (("BL", "BLN"), ("SA", "SAN"), ("VDD", "VSS")):
        assert swapped.pin_positions[a][1][1] == plain.pin_positions[b][1][1]


def test_it_is_bitline_mux_specs_six_devices_and_pins():
    spec = SidewaysMuxSpec(selects=8, select=3)
    reference = BitlineMuxSpec(selects=8, select=3)
    assert spec.devices == reference.devices
    assert set(spec.pin_positions) == set(BITLINE_MUX_PINS)
    netlist = spec.netlist()
    assert f".SUBCKT {spec.cell_name} {' '.join(BITLINE_MUX_PINS)}" in netlist
    assert netlist.count("nfin=3") == 6


def test_each_leaf_taps_only_its_own_select_tracks(boxes_on):
    spec = SidewaysMuxSpec(selects=4, select=2)
    cell = build_sideways_mux(spec, lib=_library())
    centres = {((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in boxes_on(cell, "V2")}
    tracks = spec.track_x
    assert (tracks["YSEL[2]"], 217) in centres and (tracks["YSELN[2]"], 135) in centres
    others = {tracks[f"{r}[{j}]"] for r in ("YSEL", "YSELN") for j in (0, 1, 3)}
    assert not any(x in others for x, _ in centres)


def test_a_group_keeps_its_sense_lines_off_its_ends(boxes_on):
    spec = SidewaysMuxSpec(selects=4)
    group = build_sideways_mux_group(spec, lib=_library())
    top = 4 * 270
    for net in ("SA", "SAN"):
        x = spec.track_x[net]
        runs = [b for b in boxes_on(group, "M3") if (b[0] + b[2]) / 2 == x]
        assert min(b[1] for b in runs) > 0 and max(b[3] for b in runs) < top
    assert sideways_mux_group_pins(spec)[:4] == ("BL[0]", "BLN[0]", "YSEL[0]", "YSELN[0]")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"fins": 2}, "three fins"),
        ({"bitline_entry": (190, 83.5)}, "land at"),
        ({"select": 4}, "not one of"),
        ({"selects": 0}, "at least one"),
    ],
)
def test_undrawable_specs_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SidewaysMuxSpec(**kwargs)
