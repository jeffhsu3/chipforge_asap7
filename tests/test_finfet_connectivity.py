"""Layout-level checks that the DRC-clean geometry is still one four-pin FET."""

from __future__ import annotations

from collections import defaultdict

import pytest

from chipforge_asap7.devices import FinFETSpec, build_finfet
from chipforge_asap7.layout import LAYERS

Box = tuple[float, float, float, float]
Node = tuple[str, int]


def _boxes(cell, layer_name: str) -> list[Box]:
    layer = LAYERS[layer_name]
    target = (layer["layer"], layer["datatype"])
    result: list[Box] = []
    for polygon_set in cell.polygons:
        for points, gds_layer, datatype in zip(
            polygon_set.polygons,
            polygon_set.layers,
            polygon_set.datatypes,
        ):
            if (gds_layer, datatype) == target:
                xs, ys = points[:, 0], points[:, 1]
                result.append((xs.min(), ys.min(), xs.max(), ys.max()))
    return sorted(result)


def _overlaps(a: Box, b: Box, *, touching: bool = False) -> bool:
    if touching:
        return max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _effective_gates(cell) -> list[Box]:
    """Subtract horizontal GCUT rectangles from the vertical GATE stripes."""
    result: list[Box] = []
    cuts = _boxes(cell, "GATE_CUT")
    for x0, y0, x1, y1 in _boxes(cell, "GATE"):
        blocked = sorted(
            (max(y0, cy0), min(y1, cy1))
            for cx0, cy0, cx1, cy1 in cuts
            if max(x0, cx0) < min(x1, cx1) and max(y0, cy0) < min(y1, cy1)
        )
        cursor = y0
        for cut_lo, cut_hi in blocked:
            if cut_lo > cursor:
                result.append((x0, cursor, x1, cut_lo))
            cursor = max(cursor, cut_hi)
        if cursor < y1:
            result.append((x0, cursor, x1, y1))
    return result


def _terminal_components(
    cell,
) -> tuple[dict[str, int], dict[Node, int], dict[str, list[Box]]]:
    geometry = {
        "M1": _boxes(cell, "M1"),
        "V0": _boxes(cell, "V0"),
        "LIG": _boxes(cell, "LIG"),
        "LISD": _boxes(cell, "LISD"),
        "GATE": _effective_gates(cell),
    }
    graph: dict[Node, set[Node]] = defaultdict(set)

    for layer_name, boxes in geometry.items():
        for index in range(len(boxes)):
            graph[(layer_name, index)]
        for left in range(len(boxes)):
            for right in range(left + 1, len(boxes)):
                if _overlaps(boxes[left], boxes[right], touching=True):
                    graph[(layer_name, left)].add((layer_name, right))
                    graph[(layer_name, right)].add((layer_name, left))

    for layer_a, layer_b in (
        ("V0", "M1"),
        ("V0", "LIG"),
        ("V0", "LISD"),
        ("LIG", "LISD"),
        ("LIG", "GATE"),
    ):
        for index_a, box_a in enumerate(geometry[layer_a]):
            for index_b, box_b in enumerate(geometry[layer_b]):
                if _overlaps(box_a, box_b):
                    graph[(layer_a, index_a)].add((layer_b, index_b))
                    graph[(layer_b, index_b)].add((layer_a, index_a))

    component_by_node: dict[Node, int] = {}
    for start in graph:
        if start in component_by_node:
            continue
        component = len(set(component_by_node.values()))
        pending = [start]
        while pending:
            node = pending.pop()
            if node in component_by_node:
                continue
            component_by_node[node] = component
            pending.extend(graph[node])

    pin_components: dict[str, int] = {}
    for label in cell.labels:
        if label.text not in {"B", "D", "G", "S"}:
            continue
        x, y = label.position
        containing = [
            index
            for index, (x0, y0, x1, y1) in enumerate(geometry["M1"])
            if x0 <= x <= x1 and y0 <= y <= y1
        ]
        assert len(containing) == 1
        pin_components[label.text] = component_by_node[("M1", containing[0])]
    return pin_components, component_by_node, geometry


@pytest.mark.parametrize(
    "spec",
    [
        FinFETSpec(),
        FinFETSpec(flavor="p", fins=2, fingers=2),
        FinFETSpec(fins=3, fingers=2, multipliers=3),
        FinFETSpec(flavor="p", fins=12, fingers=3, vt="slvt"),
    ],
)
def test_layout_has_four_separate_fully_connected_terminals(spec):
    cell = build_finfet(spec)
    pins, components, geometry = _terminal_components(cell)

    assert set(pins) == {"B", "D", "G", "S"}
    assert len(set(pins.values())) == 4

    lisd_by_center = {
        (x0 + x1) / 2: index
        for index, (x0, _y0, x1, _y1) in enumerate(geometry["LISD"])
    }
    for x_source in spec.source_xs:
        assert components[("LISD", lisd_by_center[x_source])] == pins["S"]
    for x_drain in spec.drain_xs:
        assert components[("LISD", lisd_by_center[x_drain])] == pins["D"]
    assert components[("LISD", lisd_by_center[54])] == pins["B"]

    gate_nodes = [
        ("GATE", index)
        for index, (x0, y0, x1, y1) in enumerate(geometry["GATE"])
        if (x0 + x1) / 2 in spec.gate_xs and y0 < spec.gate_contact_y() < y1
    ]
    assert len(gate_nodes) == spec.layout_fingers
    assert all(components[node] == pins["G"] for node in gate_nodes)


@pytest.mark.parametrize(
    "spec",
    [
        FinFETSpec(fins=1),
        FinFETSpec(fins=4, fingers=3),
        FinFETSpec(fins=2, fingers=2, multipliers=4),
        FinFETSpec(flavor="p", fins=12, fingers=2, multipliers=2),
    ],
)
def test_channel_intersections_match_the_requested_fin_drive(spec):
    cell = build_finfet(spec)
    device_active = next(
        box for box in _boxes(cell, "ACTIVE") if box[0] >= spec.device_x0
    )
    active_fins = [box for box in _boxes(cell, "FIN") if _overlaps(box, device_active)]
    active_gates = [
        box for box in _effective_gates(cell) if _overlaps(box, device_active)
    ]

    assert len(active_fins) == spec.fins
    assert len(active_gates) == spec.layout_fingers
    assert (
        sum(
            _overlaps(fin, device_active) and _overlaps(gate, device_active)
            for fin in active_fins
            for gate in active_gates
        )
        == spec.total_fins
    )

    # Every physical gate lies halfway between consecutive alternating S/D
    # regions, so all extracted channels connect the same two terminal nets.
    for gate_x, left_sd, right_sd in zip(spec.gate_xs, spec.sd_xs, spec.sd_xs[1:]):
        assert gate_x - left_sd == right_sd - gate_x
