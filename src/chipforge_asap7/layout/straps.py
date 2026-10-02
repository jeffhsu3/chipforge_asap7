"""ASAP7 power supply strapping and seam tie routines for macro integration.

Ties horizontal M1 standard-cell or leaf-block supply rails across multi-row
heights with vertical M3 straps, and jumpers array M2 supply bars to IO column
rails at placement seams.
"""

from __future__ import annotations

from typing import Any

from .layers import LAYERS
from .transform import bbox, edge_boundary, rect
from ..verification.connectivity import MetalGraph

__all__ = [
    "add_supply_straps",
    "io_supply_rails",
    "tie_seam_supplies",
]

M1 = LAYERS["M1"]["layer"]
V1 = LAYERS["V1"]["layer"]
M2 = LAYERS["M2"]["layer"]
V2 = LAYERS["V2"]["layer"]
M3 = LAYERS["M3"]["layer"]
V3 = LAYERS["V3"]["layer"]

STRAP_HALF = 0.009
PAD_HALF_X = 0.015
VIA_HALF = 0.009
M3_SPACE = 0.018
M2_SPACE = 0.018
VIA_SPACE = 0.018
RAIL_OVERHANG = 0.005
SEAM_TIE_INSET = 0.012
M2_END_CAP = 0.006


def io_supply_rails(cell: Any) -> dict[str, list[tuple[float, float, float]]]:
    """Find (x0, x1, y_center) of every M1 supply rail of a placed block by net."""
    graph = MetalGraph(cell)
    net_of: dict[int, str] = {}
    for label in cell.get_labels(depth=None):
        net = label.text.upper().rstrip("!")
        if net in ("VDD", "VSS") and label.layer == M1:
            try:
                net_of[graph.label_root(label)] = net
            except (KeyError, RuntimeError, ValueError):
                pass
    rails: dict[str, list[tuple[float, float, float]]] = {"VDD": [], "VSS": []}
    for index, poly in enumerate(graph.polygons):
        x0, y0, x1, y1 = bbox(poly)
        net = net_of.get(graph.root(index))
        if poly.layer == M1 and net and x1 - x0 > 0.3 and y1 - y0 < 0.03:
            rails[net].append((x0, x1, round((y0 + y1) / 2, 4)))
    return rails


def add_supply_straps(cell: Any, prefer: str) -> dict[str, float]:
    """Tie each supply's rails together with a full-height M3 strap; returns ``{net: x}``."""
    x_lo, y_lo, x_hi, y_hi = edge_boundary(cell)
    polys = cell.get_polygons(depth=None)
    m1 = [bbox(q) for q in polys if q.layer == M1]
    m2 = [bbox(q) for q in polys if q.layer == M2]
    m3 = [bbox(q) for q in polys if q.layer == M3]
    vias = [bbox(q) for q in polys if q.layer in (V1, V2, V3)]
    rails = {
        net: [r for r in found if y_lo + VIA_HALF <= r[2] <= y_hi - VIA_HALF]
        for net, found in io_supply_rails(cell).items()
    }

    def near(boxes: list[tuple[float, float, float, float]], x0: float, y0: float, x1: float, y1: float, space: float) -> bool:
        return any(
            a < x1 + space and c > x0 - space and b < y1 + space and d > y0 - space
            for a, b, c, d in boxes
        )

    def fits(x: float, net: str, taken: list[tuple[float, float, float, float]]) -> bool:
        if near(m3 + taken, x - STRAP_HALF, y_lo, x + STRAP_HALF, y_hi, M3_SPACE):
            return False
        ys = sorted({y for x0, x1, y in rails[net]})
        for y in ys:
            if not any(
                x0 <= x - VIA_HALF - RAIL_OVERHANG and x1 >= x + VIA_HALF + RAIL_OVERHANG
                for x0, x1, yy in rails[net]
                if abs(yy - y) < 1e-4
            ):
                return False
            if near(m2, x - PAD_HALF_X, y - VIA_HALF, x + PAD_HALF_X, y + VIA_HALF, M2_SPACE):
                return False
            if near(vias, x - VIA_HALF, y - VIA_HALF, x + VIA_HALF, y + VIA_HALF, VIA_SPACE):
                return False
        return bool(ys)

    steps = [
        round(x_lo + STRAP_HALF + k * 0.001, 4)
        for k in range(int((x_hi - x_lo - 2 * STRAP_HALF) * 1000) + 1)
    ]
    order = {
        "low": steps,
        "high": steps[::-1],
        "middle": sorted(steps, key=lambda x: abs(x - (x_lo + x_hi) / 2)),
    }[prefer]
    chosen: dict[str, float] = {}
    taken: list[tuple[float, float, float, float]] = []
    for net in ("VSS", "VDD"):
        x = next((cand for cand in order if fits(cand, net, taken)), None)
        if x is None:
            raise RuntimeError(f"{cell.name}: no room for a {net} strap reaching every rail")
        chosen[net] = x
        taken.append((x - STRAP_HALF, y_lo, x + STRAP_HALF, y_hi))
        rect(cell, (x - STRAP_HALF, y_lo, x + STRAP_HALF, y_hi), M3)
        for y in sorted({y for _, _, y in rails[net]}):
            rect(cell, (x - VIA_HALF, y - VIA_HALF, x + VIA_HALF, y + VIA_HALF), V1)
            rect(cell, (x - PAD_HALF_X, y - VIA_HALF, x + PAD_HALF_X, y + VIA_HALF), M2)
            rect(cell, (x - VIA_HALF, y - VIA_HALF, x + VIA_HALF, y + VIA_HALF), V2)
    return chosen


def tie_seam_supplies(cell: Any, seam_x: float, io_side: str, every: bool = True) -> int:
    """Tie array supply bars to IO block rails at a seam; returns number of ties created."""
    graph = MetalGraph(cell)
    net_of: dict[int, str] = {}
    for label in cell.get_labels(depth=None):
        net = label.text.upper().rstrip("!")
        if net in ("VDD", "VSS") and label.layer in (M1, M2):
            try:
                net_of[graph.label_root(label)] = net
            except (KeyError, RuntimeError, ValueError):
                pass
    sign = -1.0 if io_side == "left" else 1.0
    x_via = seam_x + sign * SEAM_TIE_INSET
    bars, rails = [], []
    for index, poly in enumerate(graph.polygons):
        net = net_of.get(graph.root(index))
        x0, y0, x1, y1 = bbox(poly)
        if net is None or y1 - y0 > 0.03 or x1 - x0 < 0.1:
            continue
        if poly.layer == M2:
            near_x = x0 if io_side == "left" else x1
            if abs(near_x - seam_x) < 0.1 and (x1 - seam_x if io_side == "left" else seam_x - x0) > 0.1:
                bars.append((net, x0, x1, round((y0 + y1) / 2, 4)))
        elif poly.layer == M1 and x0 <= x_via - VIA_HALF and x1 >= x_via + VIA_HALF:
            rails.append((net, round((y0 + y1) / 2, 4)))
    polys = cell.get_polygons(depth=None)
    others = {layer: [bbox(q) for q in polys if q.layer == layer] for layer in (M1, V1)}
    ties = 0
    for net, x0, x1, y in sorted(set(bars), key=lambda bar: bar[3]):
        candidates = [ry for rnet, ry in rails if rnet == net]
        if not candidates:
            raise RuntimeError(f"{cell.name}: no {net} rail at the seam x {seam_x:.3f} for the bar at y {y:.4f}")
        rail_y = min(candidates, key=lambda ry: abs(ry - y))
        via = (x_via - VIA_HALF, y - VIA_HALF, x_via + VIA_HALF, y + VIA_HALF)
        jumper = (
            x_via - VIA_HALF, min(y - VIA_HALF - RAIL_OVERHANG, rail_y - VIA_HALF),
            x_via + VIA_HALF, max(y + VIA_HALF + RAIL_OVERHANG, rail_y + VIA_HALF),
        )
        if any(
            a < box[2] + space and c > box[0] - space and b < box[3] + space and d > box[1] - space
            and not (layer == M1 and abs((b + d) / 2 - rail_y) < 1e-4)
            for layer, box, space in ((V1, via, VIA_SPACE), (M1, jumper, 0.018))
            for a, b, c, d in others[layer]
        ):
            if every:
                raise RuntimeError(f"{cell.name}: no room for the {net} seam tie at y {y:.4f}")
            continue
        rect(cell, via, V1)
        rect(cell, jumper, M1)
        if io_side == "left" and x0 > via[0] - M2_END_CAP:
            rect(cell, (via[0] - M2_END_CAP, y - VIA_HALF, x0 + 0.001, y + VIA_HALF), M2)
        elif io_side == "right" and x1 < via[2] + M2_END_CAP:
            rect(cell, (x1 - 0.001, y - VIA_HALF, via[2] + M2_END_CAP, y + VIA_HALF), M2)
        ties += 1
    if every and not ties:
        raise RuntimeError(f"{cell.name}: no supply bars at the seam x {seam_x:.3f}")
    return ties
