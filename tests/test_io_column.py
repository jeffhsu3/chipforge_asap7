"""Placement and routing of the column IO block: no external tools needed."""

from __future__ import annotations

import gdspy
import pytest

from chipforge_asap7.devices import (
    BitlineMuxSpec,
    IoColumnSpec,
    OutputLatchSpec,
    SenseAmpRowSpec,
    WriteDriverSpec,
    block_netlist,
    build_io_column,
    io_column_pins,
)


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def test_default_block_is_the_port_a_group_and_a_logic_column():
    spec = IoColumnSpec()
    assert (spec.width, spec.height) == (1620, 2376)
    assert spec.pairs == 4 and spec.pair_height == 594
    # The widest cell is 756; 756 - 702 would leave a 54 nm spare, which no
    # support cell fits, so the column grows to 864 and every spare is whole.
    assert spec.logic_width == 864
    assert spec.placements == {
        "mux": (0, 0),
        "sense_amp": (756, 0),
        "output_latch": (756, 594),
        "write_driver": (756, 1782),
    }
    assert spec.support_pairs == [2]
    assert spec.cell_name == "iocol_x4_r2_h135x162_m2in3485x2455_sa2t2_wd1_ol2"


def test_pins_are_the_leaves_per_leaf_pins_the_controls_and_the_rails():
    spec = IoColumnSpec()
    pins = io_column_pins(spec)
    assert len(pins) == 4 * 4 + 11
    positions = spec.pin_positions
    assert {p.split(".")[0] for p in positions} == set(pins)
    # Bitlines at the left edge, on each leaf's entry.
    assert positions["BL[0]"] == ("M2", (17, 348.5)) and positions["BLN[1]"] == (
        "M2",
        (17, 1188 - 245.5),
    )
    # Controls on the cells' own M3.
    assert positions["SAE"][0] == positions["D"][0] == positions["Q"][0] == "M3"
    assert positions["D"][1][1] > 1782 and positions["Q"][1][1] < 1188
    # With one sense phase SAE is pinned on the amplifier's top stub, clear of the block's M4 lines.
    one = IoColumnSpec(one_sense_phase=True).pin_positions
    assert "SAPRECHN" not in one and one["SAE"][1][1] > positions["SAE"][1][1]
    # Nine rails, each labelled.
    assert sum(1 for p in positions if p.startswith(("VDD", "VSS"))) == 9


def test_two_sided_mux_uses_separate_array_pins_and_shared_sense_lines():
    spec = IoColumnSpec(two_sided=True)
    pins = io_column_pins(spec)
    assert {"BL_R[0]", "YSEL_R[0]", "PRECHN_R"} <= set(pins)
    assert set(pins) <= {pin.split(".")[0] for pin in spec.pin_positions}

    devices = {
        line.split()[0]: line.split()
        for line in block_netlist(spec).splitlines()
        if line.startswith("M")
    }
    assert devices["M0_NT"][1:4] == ["BL[0]", "YSEL[0]", "SA"]
    assert devices["M0R_NT"][1:4] == ["BL_R[0]", "YSEL_R[0]", "SA"]
    assert devices["M0R_PPT"][1:4] == ["BL_R[0]", "PRECHN_R", "VDD"]


def test_routes_join_the_tracks_where_both_ends_exist():
    spec = IoColumnSpec()
    routes = spec.routes
    sa, wd = spec.sense_amp, spec.write_driver
    assert routes["SA"] == [
        (96, [270, 756 + sa.track_x["SA"]]),
        (1782 + 96, [270, 756 + wd.track_x["SA"]]),
    ]
    assert routes["SAN"][0][0] == 144 and routes["SAN"][1][0] == 1782 + 144
    ((y_qa, xs_qa),) = routes["QA"]
    ((y_qan, xs_qan),) = routes["QAN"]
    assert (y_qa, y_qan) == (594 + 48, 594 + 96)
    assert xs_qan[0] == xs_qan[1]  # riser and stub share a column: bridged on M3, no M4
    assert xs_qa[0] != xs_qa[1]


def test_the_block_draws_m4_only_for_its_routes(boxes_on):
    spec = IoColumnSpec()
    cell = build_io_column(spec, lib=_library())
    m4 = boxes_on(cell, "M4")
    ys = sorted({(y0 + y1) / 2 for _, y0, _, y1 in m4})
    assert ys == [96, 144, 642, 1878, 1926]
    assert all(y1 - y0 == 24 for _, y0, _, y1 in m4)
    v3 = boxes_on(cell, "V3")
    assert len(v3) == 2 * len(m4) and all(
        (x1 - x0, y1 - y0) == (18, 24) for x0, y0, x1, y1 in v3
    )
    refs = {ref.ref_cell.name.split("_")[0] for ref in cell.references}
    assert {"blmux", "sarow", "wrdrv", "outlatch"} <= refs
    assert boxes_on(cell, "BOUNDARY") == [(0, 0, 1620, 2376)]


def _m4(cell):
    """Every M4 box of `cell`, its cells' too."""
    polygons = cell.get_polygons(by_spec=True).get((40, 0), [])
    return [(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()) for p in polygons]


@pytest.mark.parametrize("grid", [(0, 0), (12, 6), (5.5, 30.5), (23, 47)])
def test_an_m4_grid_puts_every_line_on_its_tracks(grid):
    spec = IoColumnSpec(m4_grid=grid)
    gx, gy = spec.m4_grid
    m4 = _m4(build_io_column(spec, lib=_library()))
    assert len(m4) == 5
    for x0, y0, x1, y1 in m4:
        assert ((y0 + y1) / 2 - gy) % 48 == 0
        assert (x0 - gx) % 24 == 0 and (x1 - gx) % 24 == 0
    nominal, routes = IoColumnSpec().routes, spec.routes
    # The amplifier's lines go down its stubs, the others the nearer way;
    # one cell's lines stay a pitch apart.
    assert nominal["SA"][0][0] - 48 < routes["SA"][0][0] <= nominal["SA"][0][0]
    assert abs(routes["SA"][1][0] - nominal["SA"][1][0]) <= 24
    assert abs(routes["QA"][0][0] - nominal["QA"][0][0]) <= 24
    for k in range(2):
        assert routes["SAN"][k][0] - routes["SA"][k][0] == 48
    assert routes["QAN"][0][0] - routes["QA"][0][0] == 48


def test_an_m4_grid_is_one_period_and_names_the_block():
    assert IoColumnSpec(m4_grid=(36, 100)).m4_grid == (12, 4)
    assert IoColumnSpec(m4_grid=(12, 19.5)).cell_name == IoColumnSpec().cell_name + "_g12x19p5"


@pytest.mark.parametrize("grid", [(12, 6), (0, 19.5)])
def test_placed_on_its_m4_grid_no_m4_grid_rule_fires(grid, asap7_drc):
    library = _library()
    spec = IoColumnSpec(m4_grid=grid)
    block = build_io_column(spec, lib=library)
    top = library.new_cell("placed")
    # The grid's point to an absolute one: x on 24 nm, y a track's centre (12 mod 48).
    top.add(gdspy.CellReference(block, origin=(-grid[0] % 24, (12 - grid[1]) % 48)))
    assert not {c for c in asap7_drc(library, top, tag="placed") if c.startswith("M4.")}


def test_a_taller_group_gets_more_support_pairs():
    spec = IoColumnSpec(
        mux=BitlineMuxSpec(rows=2, selects=8, bitline_entry=(348.5, 245.5))
    )
    assert spec.pairs == 8 and spec.support_pairs == [2, 4, 5, 6, 7]
    assert spec.height == 4752


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"mux": BitlineMuxSpec(rows=2, selects=2, bitline_entry=(348.5, 245.5))},
            "four row pairs",
        ),
        ({"mux": BitlineMuxSpec(rows=1, selects=7)}, "not whole row pairs"),
        (
            {"sense_amp": SenseAmpRowSpec(band_height=(162, 189), n_fins=4, p_fins=4)},
            "not on one row",
        ),
        ({"write_driver": WriteDriverSpec(vt="sram")}, "not on one row"),
        ({"output_latch": OutputLatchSpec(vt="lvt")}, "not on one row"),
    ],
)
def test_mismatched_cells_and_short_groups_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        IoColumnSpec(**kwargs)
