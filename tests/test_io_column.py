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
