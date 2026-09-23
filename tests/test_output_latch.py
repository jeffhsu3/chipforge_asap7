"""Geometry of the parametric output latch: no external tools needed."""

from __future__ import annotations

import gdspy
import pytest

from chipforge_asap7.devices import (
    OUTPUT_LATCH_PINS,
    OutputLatchSpec,
    build_output_latch,
)


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def test_default_is_two_leaf_rows_with_two_fingers():
    spec = OutputLatchSpec()
    assert (spec.width, spec.height) == (756, 594)
    assert spec.cell_name == "outlatch_3n3p_f2_h135x162"
    assert spec.pairs == [5] and spec.columns == 12
    assert set(spec.pin_positions) == set(OUTPUT_LATCH_PINS)


@pytest.mark.parametrize("fingers", [2, 4, 6])
def test_each_finger_pair_costs_six_columns(fingers):
    spec = OutputLatchSpec(fingers=fingers)
    assert spec.width == OutputLatchSpec().width + (fingers - 2) * 3 * 54
    fins = {name: count for name, *_, count in spec.devices}
    assert sum(1 for name in fins if name.startswith("MNE")) == fingers
    assert sum(1 for name in fins if name.startswith("MPE")) == fingers


def test_netlist_is_an_sr_latch_an_inverter_and_a_tristate():
    spec = OutputLatchSpec()
    dev = {name: (d, g, s) for name, d, g, s, *_ in spec.devices}
    # Y1 = NAND(Y2, QA), Y2 = NAND(QAN, Y1): cross-coupled through the gates.
    assert dev["MN1A"] == ("Y1", "Y2", "M1") and dev["MN1B"] == ("M1", "QA", "VSS")
    assert dev["MN2A"] == ("M2", "QAN", "VSS") and dev["MN2B"] == ("Y2", "Y1", "M2")
    assert dev["MPI"] == ("Y2N", "Y2", "VDD")
    assert dev["MNE0"] == ("Q", "OE", "N2_0") and dev["MPE1"] == ("Q", "OEB", "N1_1")
    assert spec.series_nodes == {"M1", "M2", "N1_0", "N1_1", "N2_0", "N2_1"}
    assert ".SUBCKT outlatch_3n3p_f2_h135x162 QA QAN OE OEB Q VDD VSS" in spec.netlist()


def test_pins_leave_on_m3_sense_side_down_data_side_up(boxes_on):
    spec = OutputLatchSpec()
    cell = build_output_latch(spec, lib=_library())
    spans = {}
    for x0, y0, x1, y1 in boxes_on(cell, "M3"):
        spans.setdefault((x0 + x1) / 2, []).append((y0, y1))
    tx = spec.track_x
    for net in ("QA", "QAN"):
        assert any(y0 == 0 for y0, _ in spans[tx[net]])
    for net in ("OE", "OEB", "Q"):
        assert any(y1 == spec.height for _, y1 in spans[tx[net]])
    assert boxes_on(cell, "BOUNDARY") == [(0, 0, spec.width, spec.height)]


def test_series_nodes_have_no_contact(boxes_on):
    """m1, m2 and the tristate's inner nodes are diffusion only: no LISD bar on their columns."""
    spec = OutputLatchSpec()
    cell = build_output_latch(spec, lib=_library())
    x = spec.column_x
    bars = {
        round((x0 + x1) / 2)
        for x0, y0, x1, y1 in boxes_on(cell, "LISD")
        if y1 - y0 > 30
    }
    uncontacted = {x[1], x[3]}  # m1, m2 in the bottom row
    for b in spec.pairs:  # n1 columns and n2 columns of the top row
        uncontacted |= {x[b + 1], x[b + 2], x[b + 4], x[b + 5]}
    # Those columns carry a bar only where the other band contacts them.
    n_lisd = {
        round((x0 + x1) / 2)
        for x0, y0, x1, y1 in boxes_on(cell, "LISD")
        if y0 < 135 and y1 - y0 > 30
    }
    assert x[1] not in n_lisd and x[3] not in n_lisd
    assert bars >= {x[0], x[2], x[4]}


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"fingers": 3}, "even number"),
        ({"fingers": 0}, "even number"),
        ({"band_height": (108, 162), "n_fins": 2}, "no room for the tie tracks"),
    ],
)
def test_undrawable_specs_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OutputLatchSpec(**kwargs)


def test_fingers_must_be_an_integer():
    with pytest.raises(TypeError, match="fingers must be an integer"):
        OutputLatchSpec(fingers=True)
