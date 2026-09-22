"""DriverSliceSpec geometry: four NANDs under four interleaved drivers."""

from __future__ import annotations

import pytest

from chipforge_asap7.devices import (
    DRIVER_SLICE_PINS,
    DriverSliceSpec,
    InverterSpec,
    NandSpec,
    build_driver_slice,
    build_driver_slice_support,
    size_decoder,
)
from chipforge_asap7.layout import LAYERS

RELEASED = DriverSliceSpec()
SMALL = DriverSliceSpec(
    nand=NandSpec(rows=((6, 4),)), inverter=InverterSpec(rows=((8, 8), (4, 4)))
)


def _flat_boxes(cell, layer_name):
    spec = (LAYERS[layer_name]["layer"], LAYERS[layer_name]["datatype"])
    return [
        (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
        for pts in cell.get_polygons(by_spec=True).get(spec, [])
    ]


def test_released_slice_is_four_wordline_pitches_by_the_released_height():
    """432 nm wide; 2 x 675 of NAND under the 1890 nm driver, as ASU stack them."""
    assert (RELEASED.width, RELEASED.height) == (432, 3240)
    assert RELEASED.driver_y0 == 1350
    assert RELEASED.rails == (
        (0, "VSS"),
        (675, "VDD"),
        (1350, "VSS"),
        (2430, "VDD"),
        (3240, "VSS"),
    )
    assert RELEASED.cell_name == "wl_slice_nand14n7p_inv18n18p_13n13p"


def test_wordlines_leave_on_the_bitcell_pitch():
    assert RELEASED.wordline_xs == [54, 162, 270, 378]
    assert RELEASED.input_xs == [81, 135, 297, 351]


def test_each_nand_feeds_the_driver_above_it():
    """NAND 2k sits below NAND 2k+1; the upper one's output bar is already
    under its driver, the lower one's jogs one gate pitch on M2."""
    assert RELEASED.nand_placements == (
        (0, 0, False),
        (1, 0, True),
        (2, 216, False),
        (3, 216, True),
    )
    for wordline, origin_x, top in RELEASED.nand_placements:
        tap_x, _ = RELEASED.nand_point(origin_x, top, RELEASED.output_bar_x, 0)
        assert abs(tap_x - RELEASED.input_xs[wordline]) == (0 if top else 54)


def test_from_sizing_reproduces_the_released_slice():
    sizing = size_decoder(22.8, 32, stage_effort=4.43)
    assert DriverSliceSpec.from_sizing(sizing) == RELEASED
    assert DriverSliceSpec.from_sizing(sizing, vt="lvt").nand.vt == "lvt"


def test_from_sizing_floors_a_tiny_slice_and_refuses_an_undrawable_one():
    tiny = DriverSliceSpec.from_sizing(size_decoder(0.5, 8))
    assert tiny.nand.rows == ((4, 2),) and tiny.inverter.rows == ((2, 2),)
    with pytest.raises(ValueError, match="band limit"):
        DriverSliceSpec.from_sizing(size_decoder(60.0, 32))


def test_pins_sit_on_their_own_metal():
    for spec in (RELEASED, SMALL):
        cell = build_driver_slice(spec)
        assert {label.text for label in cell.labels} == set(DRIVER_SLICE_PINS)
        for pin, (metal, (x, y)) in spec.pin_positions.items():
            boxes = _flat_boxes(cell, metal)
            assert any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in boxes), pin
        # Wordline M3 reaches the top edge, where the array picks it up.
        tops = sorted(b[0] + 9 for b in _flat_boxes(cell, "M3") if b[3] == spec.height)
        assert tops == spec.wordline_xs


def test_cell_is_two_leaves_and_eight_references():
    cell = build_driver_slice(SMALL)
    names = sorted(ref.ref_cell.name for ref in cell.references)
    assert names == [f"{SMALL.cell_name}__inv"] * 4 + [f"{SMALL.cell_name}__nand"] * 4


def test_drivers_keep_only_the_input_metal_the_router_lands_on():
    """What the reducer did to a whole slice, asked of the generator instead.

    One landing per driver, in row 0, from the bottom of the strap -- where the
    two staggered landings are -- up to the pad of the gate contact.  The
    landings do not move, so the routing above them is the same either way.
    """
    inverter = SMALL.inverter
    assert inverter.input_rows is None and inverter.input_reach is None
    driver = SMALL.driver
    below, _ = inverter.full_input_reach(0)
    assert (driver.input_rows, driver.input_reach) == ((0,), (below, 14))
    assert driver.rows == inverter.rows and driver.netlist("X") == inverter.netlist("X")

    full = DriverSliceSpec(nand=SMALL.nand, inverter=inverter, trim_driver_input=False)
    assert full.driver is inverter
    assert full.landing_ys == SMALL.landing_ys
    assert SMALL.input_strap_y[0] == full.input_strap_y[0]
    assert SMALL.input_strap_y[1] == SMALL.driver_y0 + inverter.seam_y(0) + 14
    assert SMALL.input_strap_y[1] < full.input_strap_y[1]

    def input_m1(spec):
        """M1 on the drivers' input tracks, above the NAND rows that share them."""
        xs = {(x - 9, x + 9) for x in spec.input_xs}
        # Mirrored placements come back a float hair off the grid.
        boxes = [
            tuple(round(v) for v in b)
            for b in _flat_boxes(build_driver_slice(spec), "M1")
        ]
        return sorted(b for b in boxes if (b[0], b[2]) in xs and b[1] > spec.driver_y0)

    trimmed, untrimmed = input_m1(SMALL), input_m1(full)
    assert len(trimmed) == 4 and len(untrimmed) == 8  # four drivers, one row or two
    assert {(b[1], b[3]) for b in trimmed} == {SMALL.input_strap_y}
    assert sum(b[3] - b[1] for b in trimmed) < 0.5 * sum(b[3] - b[1] for b in untrimmed)


def test_a_short_driver_keeps_enough_strap_above_the_contact_for_both_landings():
    spec = DriverSliceSpec(
        nand=NandSpec(rows=((4, 2),)), inverter=InverterSpec(rows=((2, 2),))
    )
    lo, hi = spec.input_strap_y
    assert spec.driver.input_reach == (40, 24)
    assert lo <= spec.landing_ys[0] - 14 and spec.landing_ys[1] + 14 == hi


def test_a_driver_that_sets_its_own_input_knobs_is_drawn_as_given():
    inverter = InverterSpec(rows=((8, 8), (4, 4)), input_rows=(0, 1))
    spec = DriverSliceSpec(nand=SMALL.nand, inverter=inverter)
    assert spec.driver is inverter
    assert spec.input_strap_y == tuple(
        spec.driver_y0 + y for y in inverter.input_strap_y(0)
    )


def test_support_column_covers_the_full_slice_height():
    tap = build_driver_slice_support(SMALL, "tap")
    (x0, y0), (x1, y1) = tap.get_bounding_box()
    assert (x0, x1) == (0, 108)
    assert y0 <= 0 and y1 >= SMALL.height


def test_netlist_shares_sel_and_instantiates_both_leaves():
    text = RELEASED.netlist()
    assert (
        f".SUBCKT {RELEASED.cell_name} SEL B0 B1 B2 B3 WL0 WL1 WL2 WL3 VDD VSS" in text
    )
    assert text.count("XN") == 4 and text.count("XD") == 4
    assert f".SUBCKT {RELEASED.nand.cell_name}" in text
    assert f".SUBCKT {RELEASED.inverter.cell_name}" in text
    assert text.count(".END\n") == 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"nand": NandSpec(rows=((6, 4),), fingers=4)}, "two fingers"),
        ({"inverter": InverterSpec(rows=((8, 8),), fingers=4)}, "two fingers"),
        ({"inverter": InverterSpec(rows=((8, 8),), abut=False)}, "abutting"),
        ({"nand": NandSpec(rows=((6, 4),), vt="lvt")}, "one threshold flavor"),
        # The slice lands in row 0, low on the strap: a driver that offers only
        # row 1, or only a bare pad at the contact, gives it nothing to land on.
        (
            {"inverter": InverterSpec(rows=((8, 8), (4, 4)), input_rows=(1,))},
            "row-0 input",
        ),
        (
            {"inverter": InverterSpec(rows=((8, 8),), input_reach=(14, 14))},
            "does not hold",
        ),
    ],
)
def test_rejects_leaves_that_cannot_sit_on_the_wordline_pitch(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DriverSliceSpec(**kwargs)
