"""Parametric sense-amplifier topology, geometry, and public API."""

from __future__ import annotations

import pytest

from chipforge_asap7.devices import (
    SENSE_AMP_PINS,
    SenseAmpSpec,
    build_sense_amp,
    sense_amp_transistors,
)
from chipforge_asap7.devices.finfet import MAX_FINS, MAX_VERIFIABLE_FINS
from chipforge_asap7.layout import FIN_PITCH, FIN_WIDTH, LAYERS
from chipforge_asap7.verification import render_sense_amp_lvs_schematic

Box = tuple[float, float, float, float]


def _direct_boxes(cell, layer_name: str) -> list[Box]:
    layer = LAYERS[layer_name]
    target = (layer["layer"], layer["datatype"])
    result: list[Box] = []
    for polygon_set in cell.polygons:
        for points, gds_layer, datatype in zip(
            polygon_set.polygons,
            polygon_set.layers,
            polygon_set.datatypes,
        ):
            if (gds_layer, datatype) != target:
                continue
            xs, ys = points[:, 0], points[:, 1]
            result.append((xs.min(), ys.min(), xs.max(), ys.max()))
    return sorted(result)


def test_topology_is_the_symmetric_openfinram_16t_latch():
    devices = sense_amp_transistors()

    assert len(devices) == 16
    assert sum(device.flavor == "n" for device in devices) == 8
    assert sum(device.flavor == "p" for device in devices) == 8
    assert {device.name for device in devices} == {
        *(f"P{index}" for index in (*range(6), 14, 15)),
        *(f"N{index}" for index in range(6, 14)),
    }
    assert {net for device in devices for net in device.terminals.values()} == {
        *SENSE_AMP_PINS,
        "N52",
        "N57",
        "N58",
        "N59",
    }


def test_spec_sizes_both_polarity_banks_in_integer_fins():
    spec = SenseAmpSpec(n_fins=6, p_fins=2, vt="lvt")

    assert spec.cell_name == "sense_amp_sram_lvt_n6_p2"
    assert spec.nmos.fins == 6
    assert spec.pmos.fins == 2
    assert spec.nmos.model == "nmos_lvt"
    assert spec.pmos.model == "pmos_lvt"
    assert spec.width % 54 == 0
    assert spec.height % FIN_PITCH == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_fins": 0},
        {"p_fins": 0},
        {"n_fins": MAX_FINS + 1},
        {"p_fins": True},
        {"vt": "ulvt"},
    ],
)
def test_spec_rejects_devices_outside_the_asap7_grid(kwargs):
    with pytest.raises((TypeError, ValueError)):
        SenseAmpSpec(**kwargs)


def test_spec_accepts_fin_counts_the_public_runset_cannot_check():
    """13 fins is a legal ASAP7 device -- the released SRAM is full of them.

    It used to be rejected here only because `FinFETSpec` inherited the KLayout
    deck's twelve-height enumeration as if it were a process rule.
    """
    spec = SenseAmpSpec(n_fins=MAX_VERIFIABLE_FINS + 1)

    assert spec.nmos.fins == MAX_VERIFIABLE_FINS + 1
    assert not spec.nmos.drc_verifiable
    assert spec.pmos.drc_verifiable  # p_fins keeps its default


def test_placement_keeps_each_matched_pair_mirrored():
    placements = SenseAmpSpec().placements
    index = {
        placement.transistor.name: slot for slot, placement in enumerate(placements)
    }

    for left, right in (("P14", "P15"), ("P0", "P3"), ("P1", "P2"), ("P4", "P5")):
        assert index[left] + index[right] == 7
    for left, right in (("N12", "N13"), ("N6", "N7"), ("N8", "N9"), ("N10", "N11")):
        assert index[left] + index[right] == 23


def test_layout_has_continuous_fins_16_tiles_and_eight_top_pins():
    spec = SenseAmpSpec(n_fins=3, p_fins=2)
    cell = build_sense_amp(spec)

    assert len(cell.references) == 16
    assert len({reference.ref_cell.name for reference in cell.references}) == 2
    assert all(not reference.ref_cell.labels for reference in cell.references)
    assert {label.text for label in cell.labels} == set(SENSE_AMP_PINS)

    fins = _direct_boxes(cell, "FIN")
    assert len(fins) == spec.height // FIN_PITCH
    assert [box[1] for box in fins] == spec.fin_grid_ys
    assert all(
        (x0, x1, y1 - y0) == (0, spec.width, FIN_WIDTH) for x0, y0, x1, y1 in fins
    )


def test_every_terminal_reaches_one_upper_metal_net_track():
    spec = SenseAmpSpec(n_fins=2, p_fins=2)
    cell = build_sense_amp(spec)

    m2_trunks = [
        rectangle
        for rectangle in _direct_boxes(cell, "M2")
        if rectangle[0] == 0 and rectangle[2] == spec.width
    ]
    assert len(m2_trunks) == 12
    assert {(y0 + y1) / 2 for _, y0, _, y1 in m2_trunks} == set(
        spec.route_track_ys.values()
    )
    assert len(_direct_boxes(cell, "V1")) == 16 * 4
    assert len(_direct_boxes(cell, "M3")) == 16 * 4
    assert len(_direct_boxes(cell, "V2")) == 16 * 4 * 2

    pin_shapes = _direct_boxes(cell, "M2_PIN")
    assert len(pin_shapes) == len(SENSE_AMP_PINS)
    for pin, position in spec.pin_positions.items():
        assert any(
            x0 <= position[0] <= x1 and y0 <= position[1] <= y1
            for x0, y0, x1, y1 in pin_shapes
        ), pin


def test_lvs_reference_expands_every_logical_device_to_unit_fins():
    spec = SenseAmpSpec(n_fins=3, p_fins=2, vt="sram")
    schematic = render_sense_amp_lvs_schematic(spec)
    mos_lines = [line for line in schematic.splitlines() if line.startswith("M")]

    assert len(mos_lines) == 8 * spec.n_fins + 8 * spec.p_fins
    assert sum(" nmos_sram " in line for line in mos_lines) == 8 * spec.n_fins
    assert sum(" pmos_sram " in line for line in mos_lines) == 8 * spec.p_fins
    assert all(" W=7n" in line and " L=20n" in line for line in mos_lines)
