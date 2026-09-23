"""Geometry of the parametric write driver: no external tools needed."""

from __future__ import annotations

import gdspy
import pytest

from chipforge_asap7.devices import (
    WRITE_DRIVER_PINS,
    WriteDriverSpec,
    build_write_driver,
)


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def test_default_is_two_leaf_rows_and_twelve_columns():
    spec = WriteDriverSpec()
    assert (spec.width, spec.height) == (648, 594)
    assert [net for _, net in spec.rails] == ["VSS", "VDD", "VSS"]
    assert spec.cell_name == "wrdrv_3n3p_k1_h135x162"
    assert spec.tie_levels == (116, 152, 188) and spec.n_level == 80
    assert set(spec.pin_positions) == set(WRITE_DRIVER_PINS)
    assert {metal for metal, _ in spec.pin_positions.values()} == {"M1", "M3"}


def test_netlist_is_the_released_topology_with_weak_keepers():
    spec = WriteDriverSpec()
    fins = {name: count for name, *_, count in spec.devices}
    assert len(fins) == 12
    assert fins["MPW"] == fins["MPWN"] == 1  # the keepers lose to a pass gate
    assert {name for name, *_ in spec.devices if name.startswith("MN")} == {
        "MNI", "MNPD", "MNPDN", "MNW", "MNWN", "MNTW", "MNTWN"
    }  # fmt: skip
    gates = {name: gate for name, _, gate, *_ in spec.devices}
    assert (
        gates["MNPD"] == gates["MNPDN"] == gates["MPTW"] == gates["MPTWN"] == "WRENAN"
    )
    assert gates["MNTW"] == gates["MNTWN"] == "WRENA"
    assert gates["MNW"] == "WN" and gates["MNWN"] == "W"  # cross-coupled
    netlist = spec.netlist()
    assert netlist.count("nfin=1 ") == 2 and netlist.count("nfin=3 ") == 10


def test_sense_lines_run_the_full_height_and_the_controls_leave_at_the_top(boxes_on):
    spec = WriteDriverSpec()
    cell = build_write_driver(spec, lib=_library())
    m3 = boxes_on(cell, "M3")
    by_x = {}
    for x0, y0, x1, y1 in m3:
        by_x.setdefault((x0 + x1) / 2, []).append((y0, y1))
    tx = spec.track_x
    for net in ("SA", "SAN"):
        assert (0, spec.height) in by_x[tx[net]]
    for net in ("D", "WRENA", "WRENAN"):
        (y0, y1) = max(by_x[tx[net]], key=lambda s: s[1] - s[0])
        # Short of the top edge by a tip-to-tip space, so a cell stacked
        # above can keep metal on its bottom edge; the pin is still on it.
        assert y1 == spec.height - 31 and 0 < y0 < spec.height / 2
        assert y0 <= spec.pin_positions[net][1][1] <= y1
    assert tx["WRENAN"] - tx["D"] >= 3 * 54 and tx["WRENAN"] - tx["WRENA"] >= 2 * 54
    # W and WN climb from the bottom row's 80 nm track into the latch above.
    climbs = [
        s
        for x, spans in by_x.items()
        if x not in tx.values()
        for s in spans
        if s[1] - s[0] > 300
    ]
    assert len(climbs) == 2 and all(y0 < 80 < spec.height / 2 < y1 for y0, y1 in climbs)
    assert boxes_on(cell, "BOUNDARY") == [(0, 0, spec.width, spec.height)]


def test_keepers_are_their_own_narrow_island(boxes_on):
    spec = WriteDriverSpec(keeper_fins=1)
    cell = build_write_driver(spec, lib=_library())
    heights = sorted({round(y1 - y0) for _, y0, _, y1 in boxes_on(cell, "ACTIVE")})
    assert heights == [27, 81]  # one fin, and the row's three
    k = spec.bands["k1"]
    assert k.fins == 1 and k.y0 == spec.bands["p1"].y0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"keeper_fins": 4}, "cannot exceed p_fins"),
        ({"band_height": (108, 162), "n_fins": 2}, "no room for the tie tracks"),
        ({"band_height": (135, 135), "p_fins": 2}, "no room for the tie tracks"),
    ],
)
def test_undrawable_specs_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        WriteDriverSpec(**kwargs)


def test_taller_rows_move_the_ties_with_the_seam():
    spec = WriteDriverSpec(n_fins=4, p_fins=4, keeper_fins=2, band_height=(162, 189))
    assert spec.height == 702 and spec.stack.seam_ys == (162, 540)
    assert spec.tie_levels == (143, 179, 215)
