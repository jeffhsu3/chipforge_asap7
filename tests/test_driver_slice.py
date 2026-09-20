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
    ],
)
def test_rejects_leaves_that_cannot_sit_on_the_wordline_pitch(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DriverSliceSpec(**kwargs)
