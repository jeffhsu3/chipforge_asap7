"""Geometry of the row sense amplifier: no external tools needed."""

from __future__ import annotations

import gdspy
import pytest

from chipforge_asap7.devices import SENSE_AMP_PINS, SenseAmpRowSpec, build_sense_amp_row


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def test_default_is_two_leaf_rows_eleven_columns():
    spec = SenseAmpRowSpec()
    assert (spec.width, spec.height) == (702, 594)
    assert spec.cell_name == "sarow_3n3p_f2t2_h135x162"
    assert spec.columns == 11 and spec.tail == (4, 6) and spec.centre == 5
    assert set(spec.pin_positions) == set(SENSE_AMP_PINS)


def test_the_n_chain_interleaves_input_and_cross_coupled_devices():
    spec = SenseAmpRowSpec()
    nets = [spec.column_nets()[i] for i in range(spec.columns)]
    assert nets == [
        "52",
        "57_1",
        "QAN",
        "57_0",
        "52",
        "VSS",
        "52",
        "58_0",
        "QA",
        "58_1",
        "52",
    ]
    stripes = [spec.stripe_nets()[i] for i in range(spec.columns - 1)]
    assert stripes == ["SA", "QA", "QA", "SA", "SAE", "SAE", "SAN", "QAN", "QAN", "SAN"]
    assert spec.risers == {"QAN": 2, "QA": 8}


@pytest.mark.parametrize("fingers, tail", [(2, 2), (4, 2), (2, 4), (4, 6)])
def test_width_grows_by_four_columns_per_finger_and_one_per_tail_stripe(fingers, tail):
    spec = SenseAmpRowSpec(n_fingers=fingers, tail_fingers=tail)
    assert spec.columns == 4 * fingers + tail + 1
    devices = {name for name, *_ in spec.devices}
    assert len(devices) == 6 * fingers + tail + 4
    # Every 52 column is on the common node and every VSS column is a tail source.
    nets = spec.column_nets()
    assert nets[0] == nets[spec.columns - 1] == "52"
    assert [nets[i] for i in range(*spec.tail)] == ["52", "VSS"] * (tail // 2)


def test_netlist_is_the_released_amplifier_without_its_dummies():
    spec = SenseAmpRowSpec()
    dev = {name: (d, g, s) for name, d, g, s, *_ in spec.devices}
    assert dev["MN6_0"] == ("N57_0", "SA", "TAIL") and dev["MN8_0"] == (
        "QAN",
        "QA",
        "N57_0",
    )
    assert dev["MN7_1"] == ("N58_1", "SAN", "TAIL") and dev["MN9_1"] == (
        "QA",
        "QAN",
        "N58_1",
    )
    assert dev["MN10_0"] == dev["MN10_1"] == ("TAIL", "SAE", "VSS")
    assert dev["MP1"] == ("QAN", "SAPRECHN", "N59") and dev["MP2"] == (
        "N59",
        "SAPRECHN",
        "QA",
    )
    assert not any(g in ("VDD", "VSS") for _, g, _ in dev.values())  # no off dummies
    assert spec.series_nodes == {"N59", "N57_0", "N57_1", "N58_0", "N58_1"}
    assert (
        ".SUBCKT sarow_3n3p_f2t2_h135x162 SA SAN SAE SAPRECHN QA QAN VDD VSS"
        in spec.netlist()
    )


def test_pins_leave_on_m3_inputs_down_outputs_up(boxes_on):
    spec = SenseAmpRowSpec()
    cell = build_sense_amp_row(spec, lib=_library())
    spans = {}
    for x0, y0, x1, y1 in boxes_on(cell, "M3"):
        spans.setdefault((x0 + x1) / 2, []).append((y0, y1))
    tx = spec.track_x
    for net in ("SA", "SAN", "SAE"):
        assert any(y0 == 0 for y0, _ in spans[tx[net]]), net
    for net in ("QA", "QAN", "SAPRECHN"):
        assert any(y1 == spec.height for _, y1 in spans[tx[net]]), net
    assert tx["SAE"] == tx["SAPRECHN"]  # opposite edges of the same column
    assert boxes_on(cell, "BOUNDARY") == [(0, 0, spec.width, spec.height)]


def test_cross_coupled_pfets_share_the_nfets_stripes(boxes_on):
    """No PSELECT-only stripe: every gate over p diffusion in the bottom row is a QA or QAN gate."""
    spec = SenseAmpRowSpec()
    cell = build_sense_amp_row(spec, lib=_library())
    p_islands = [
        (x0, x1) for x0, y0, x1, y1 in boxes_on(cell, "ACTIVE") if 135 < y0 < 297
    ]
    assert (
        len(p_islands) == 2
    )  # one island per output, each over that output's finger stripes
    stripes = spec.stripe_nets()
    for x0, x1 in p_islands:
        under = [
            stripes[i] for i in range(spec.columns - 1) if x0 < spec.gate_x(i) < x1
        ]
        assert set(under) <= {"QA", "QAN"} and len(under) == spec.n_fingers


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_fingers": 1}, "even number"),
        ({"n_fingers": 3}, "even number"),
        ({"tail_fingers": 1}, "even number"),
        ({"band_height": (108, 162), "n_fins": 2}, "no room for the tie tracks"),
    ],
)
def test_undrawable_specs_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SenseAmpRowSpec(**kwargs)
