"""Metal-only connectivity for the rectangular peripheral and macro routes."""

from collections import defaultdict
from typing import Any

from ..layout.layers import LAYERS

METALS = ("V1", "M2", "V2", "M3", "V3", "M4", "V4", "M5")
#: Side (nm) of the square bins that index rectangles. Rectangles that overlap
#: share a bin, and so does a point inside one, so only a bin's own entries
#: are ever compared. Small next to row and spine pitches, large next to vias.
BIN = 512


def _bins(x0, y0, x1, y1):
    for bx in range(int(x0 // BIN), int(x1 // BIN) + 1):
        for by in range(int(y0 // BIN), int(y1 // BIN) + 1):
            yield bx, by


def metal_components(top: Any):
    """Return a point-to-component lookup; ignore child FEOL and substrate.

    Rectangles join when they share area on one layer or on adjacent ones
    (a metal and its via). Both the joining and the lookups use the bin
    index, so the cost grows with the shapes, not with their square.
    """
    numbers = {LAYERS[name]["layer"]: i for i, name in enumerate(METALS)}
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

    index = defaultdict(list)
    for i, (layer, x0, y0, x1, y1) in enumerate(rectangles):
        for bx, by in _bins(x0, y0, x1, y1):
            for other in (layer - 1, layer, layer + 1):
                for j in index.get((other, bx, by), ()):
                    _, ax0, ay0, ax1, ay1 = rectangles[j]
                    if min(x1, ax1) > max(x0, ax0) and min(y1, ay1) > max(y0, ay0):
                        parents[root(i)] = root(j)
            index[layer, bx, by].append(i)

    def component(metal, x, y):
        layer = METALS.index(metal)
        roots = {
            root(i)
            for i in index.get((layer, int(x // BIN), int(y // BIN)), ())
            if rectangles[i][1] < x < rectangles[i][3]
            and rectangles[i][2] < y < rectangles[i][4]
        }
        if len(roots) != 1:
            raise ValueError(f"missing or ambiguous metal route at {metal} ({x}, {y})")
        return roots.pop()

    return component
