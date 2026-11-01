"""Metal-only connectivity for the rectangular peripheral and macro routes."""

from typing import Any

from ..layout.layers import LAYERS


def metal_components(top: Any, *, recursive=False):
    """Return a point-to-component lookup; ignore child FEOL and substrate."""
    metals = ("M1", "V1", "M2", "V2", "M3", "V3", "M4", "V4", "M5", "V5", "M6")
    numbers = {LAYERS[name]["layer"]: i for i, name in enumerate(metals)}
    rectangles = []
    polygons = top.polygons
    if recursive:
        from ..layout.layers import require_gdspy

        gdspy = require_gdspy()
        polygons = [
            gdspy.Polygon(p, layer=l, datatype=d)
            for (l, d), ps in top.get_polygons(by_spec=True).items()
            if l in numbers and d == 0
            for p in ps
        ]
    for polygon in polygons:
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

    # A two-dimensional spatial index avoids quadratic comparisons between
    # unrelated rows in tall ROMs. Keep edge abutments on the same conductor.
    from collections import defaultdict
    from math import floor

    bins = defaultdict(list)
    grid = 108
    for i, (layer, x0, y0, x1, y1) in enumerate(rectangles):
        tiles = [
            (x, y)
            for x in range(floor(x0 / grid), floor(x1 / grid) + 1)
            for y in range(floor(y0 / grid), floor(y1 / grid) + 1)
        ]
        candidates = set()
        for other in (layer - 1, layer, layer + 1):
            for x, y in tiles:
                candidates.update(bins.get((other, x, y), ()))
        for j in candidates:
            other, bx0, by0, bx1, by1 = rectangles[j]
            dx = min(x1, bx1) - max(x0, bx0)
            dy = min(y1, by1) - max(y0, by0)
            if (dx > 0 and dy > 0) or (
                other == layer and dx >= 0 and dy >= 0 and dx + dy > 0
            ):
                parents[root(i)] = root(j)
        for x, y in tiles:
            bins[layer, x, y].append(i)

    def component(metal, x, y):
        target = metals.index(metal)
        roots = {
            root(i)
            for i in bins[target, floor(x / grid), floor(y / grid)]
            for layer, x0, y0, x1, y1 in [rectangles[i]]
            if x0 < x < x1 and y0 < y < y1
        }
        if len(roots) != 1:
            raise ValueError(f"missing or ambiguous metal route at {metal} ({x}, {y})")
        return roots.pop()

    return component
