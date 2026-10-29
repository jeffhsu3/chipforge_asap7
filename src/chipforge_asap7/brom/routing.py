"""Metal-only connectivity for the rectangular peripheral and macro routes."""

from typing import Any

from ..layout.layers import LAYERS


def metal_components(top: Any):
    """Return a point-to-component lookup; ignore child FEOL and substrate."""
    metals = ("V1", "M2", "V2", "M3", "V3", "M4", "V4", "M5")
    numbers = {LAYERS[name]["layer"]: i for i, name in enumerate(metals)}
    rectangles = []
    for polygon in top.polygons:
        for points, layer, datatype in zip(
            polygon.polygons, polygon.layers, polygon.datatypes, strict=True
        ):
            if layer in numbers and datatype == 0:
                lo, hi = points.min(axis=0), points.max(axis=0)
                rectangles.append((numbers[layer], *lo, *hi))
    parents = list(range(len(rectangles)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    # Sweep by left edge, retaining only rectangles that can overlap in X.
    active = []
    for i in sorted(parents, key=lambda i: rectangles[i][1]):
        layer, x0, y0, _x1, y1 = rectangles[i]
        active = [j for j in active if rectangles[j][3] > x0]
        for j in active:
            other, _, by0, _, by1 = rectangles[j]
            if (other == layer or abs(other - layer) == 1) and min(y1, by1) > max(
                y0, by0
            ):
                parents[root(i)] = root(j)
        active.append(i)

    def component(metal, x, y):
        roots = {
            root(i)
            for i, (layer, x0, y0, x1, y1) in enumerate(rectangles)
            if layer == metals.index(metal) and x0 < x < x1 and y0 < y < y1
        }
        if len(roots) != 1:
            raise ValueError(f"missing or ambiguous metal route at {metal} ({x}, {y})")
        return roots.pop()

    return component
