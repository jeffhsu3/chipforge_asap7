"""ASAP7 FinFET in-memory connectivity extraction, DSU graph assembly, and DRC auditing.

Extracts device channels, terminals, fin counts, and multi-metal net connectivity
from gdstk layout cells without requiring external EDA tools.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from ..layout.layers import LAYERS
from ..layout.transform import bbox, edge_boundary, is_box, require_gdstk

__all__ = [
    "ADJACENT",
    "DisjointSetUnion",
    "METALS",
    "MetalGraph",
    "VIAS",
    "extract_connectivity",
    "filtered_polygon_fingerprint",
    "hierarchical_fingerprint",
    "layer_polygons",
    "overlap",
    "polygon_fingerprint",
    "process_fingerprint",
    "verify_rules",
    "verify_tap_rules",
]

# Standard ASAP7 layer numbers
WELL = LAYERS["NWELL"]["layer"]
FIN = LAYERS["FIN"]["layer"]
GATE = LAYERS["GATE"]["layer"]
GCUT = LAYERS["GATE_CUT"]["layer"]
ACTIVE = LAYERS["ACTIVE"]["layer"]
NSELECT = LAYERS["NSELECT"]["layer"]
PSELECT = LAYERS["PSELECT"]["layer"]
LIG = LAYERS["LIG"]["layer"]
LISD = LAYERS["LISD"]["layer"]
V0 = LAYERS["V0"]["layer"]
M1 = LAYERS["M1"]["layer"]
M2 = LAYERS["M2"]["layer"]
V1 = LAYERS["V1"]["layer"]
V2 = LAYERS["V2"]["layer"]
M3 = LAYERS["M3"]["layer"]
V3 = LAYERS["V3"]["layer"]
M4 = LAYERS["M4"]["layer"]
V4 = LAYERS["V4"]["layer"]
M5 = LAYERS["M5"]["layer"]
SDT = LAYERS["SDT"]["layer"]
SRAMDRC = LAYERS["SRAMDRC"]["layer"]
BOUNDARY = LAYERS["BOUNDARY"]["layer"]
SRAMVT = LAYERS["SRAMVT"]["layer"]

PIN_TEXTTYPE = 251
MARKER_LAYERS = {SRAMDRC, BOUNDARY, SRAMVT}
PROCESS_LAYERS = {WELL, FIN, GATE, GCUT, NSELECT, PSELECT, SRAMDRC, BOUNDARY, SRAMVT}

MARKER = (0.000, -0.162, 0.108, 0.432)
TAP_SLOTS = 2
TAP_MARKER = (0.000, -0.162, TAP_SLOTS * 0.108, 0.432)
GATE_A = (0.017, -0.181, 0.037, 0.439)
GATE_B = (0.071, -0.169, 0.091, 0.451)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


METALS = (19, 20, 30, 40, 50, 60, 70, 80, 90)
VIAS = (21, 25, 35, 45, 55, 65, 75, 85)
ADJACENT = {via: (METALS[i], METALS[i + 1]) for i, via in enumerate(VIAS)}


class MetalGraph:
    """Interconnect polygon graph with spatial hash grids for fast multi-metal collision detection."""

    def __init__(self, cell: Any):
        gdstk = require_gdstk()
        self.polygons = [
            p
            for p in cell.get_polygons()
            if p.datatype == 0 and p.layer in (*METALS, *VIAS)
        ]
        self.parent = list(range(len(self.polygons)))
        self.buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        self.boxes = []
        for i, p in enumerate(self.polygons):
            box = p.bounding_box()
            self.boxes.append(box)
            candidates = set()
            layers = [p.layer]
            layers += list(ADJACENT.get(p.layer, ()))
            layers += [v for v, pair in ADJACENT.items() if p.layer in pair]
            for x, y in self.tiles(box):
                for layer in layers:
                    candidates.update(self.buckets[layer, x, y])
            for j in candidates:
                q = self.polygons[j]
                a, b = box, self.boxes[j]
                if (
                    a[1][0] + 1e-7 < b[0][0]
                    or b[1][0] + 1e-7 < a[0][0]
                    or a[1][1] + 1e-7 < b[0][1]
                    or b[1][1] + 1e-7 < a[0][1]
                ):
                    continue
                # Tiny expansion includes legal edge abutment in the same
                # layer. Vias must overlap both adjacent conductor layers.
                test = (
                    gdstk.offset([p], 1e-6, precision=1e-7)
                    if p.layer == q.layer
                    else [p]
                )
                if gdstk.boolean(test, [q], "and", precision=1e-7):
                    self.parent[self.root(i)] = self.root(j)
            for x, y in self.tiles(box):
                self.buckets[p.layer, x, y].append(i)

    @staticmethod
    def tiles(box):
        for x in range(
            math.floor((box[0][0] - 1e-7) / 0.25),
            math.floor((box[1][0] + 1e-7) / 0.25) + 1,
        ):
            for y in range(
                math.floor((box[0][1] - 1e-7) / 0.25),
                math.floor((box[1][1] + 1e-7) / 0.25) + 1,
            ):
                yield x, y

    def root(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def at(self, layer: int, point: tuple[float, float]) -> set[int]:
        gdstk = require_gdstk()
        x, y = (math.floor(v / 0.25) for v in point)
        return {
            self.root(i)
            for i in self.buckets[layer, x, y]
            if gdstk.inside([point], [self.polygons[i]])[0]
        }

    def label_root(self, label: Any) -> int:
        roots = self.at(label.layer, (float(label.origin[0]), float(label.origin[1])))
        if len(roots) != 1:
            raise RuntimeError(
                f"{label.text}: expected one conductor at {label.origin} "
                f"on layer {label.layer}, found {len(roots)}"
            )
        return next(iter(roots))


class DisjointSetUnion:
    """Disjoint Set Union (DSU) data structure for electrical net clustering."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def layer_polygons(cell: Any, layer: int, datatype: int = 0) -> list[Any]:
    """Retrieve polygons on a specific layer and datatype."""
    return [p for p in cell.polygons if p.layer == layer and p.datatype == datatype]


def overlap(a: Any, b: Any) -> bool:
    """Return true for positive-area overlap (not mere edge contact)."""
    gdstk = require_gdstk()
    result = gdstk.boolean(a, b, "and", precision=1e-6)
    return bool(result) and sum(abs(p.area()) for p in result) > 1e-12


def polygon_fingerprint(cell: Any) -> str:
    """Stable geometry/label digest, independent of GDS timestamps."""
    records: list[tuple] = []
    for poly in cell.polygons:
        points = tuple(
            (round(float(x), 6), round(float(y), 6)) for x, y in poly.points
        )
        records.append(("P", poly.layer, poly.datatype, points))
    for label in cell.labels:
        records.append((
            "L",
            label.text,
            round(float(label.origin[0]), 6),
            round(float(label.origin[1]), 6),
            label.layer,
            label.texttype,
        ))
    digest = hashlib.sha256()
    for record in sorted(records, key=repr):
        digest.update(repr(record).encode("utf-8"))
    return digest.hexdigest()


def process_fingerprint(cell: Any) -> str:
    """Stable digest of process/frame layers only."""
    records: list[tuple] = []
    for poly in cell.polygons:
        if poly.layer in PROCESS_LAYERS:
            records.append((
                poly.layer,
                poly.datatype,
                tuple(
                    (round(float(x), 6), round(float(y), 6))
                    for x, y in poly.points
                ),
            ))
    digest = hashlib.sha256()
    for record in sorted(records, key=repr):
        digest.update(repr(record).encode("utf-8"))
    return digest.hexdigest()


def filtered_polygon_fingerprint(cell: Any, predicate: Callable[[Any], bool]) -> str:
    """Stable digest of polygons matching a custom predicate."""
    records: list[tuple] = []
    for poly in cell.polygons:
        if predicate(poly):
            records.append((
                poly.layer,
                poly.datatype,
                tuple(
                    (round(float(x), 6), round(float(y), 6))
                    for x, y in poly.points
                ),
            ))
    digest = hashlib.sha256()
    for record in sorted(records, key=repr):
        digest.update(repr(record).encode("utf-8"))
    return digest.hexdigest()


def hierarchical_fingerprint(cell: Any) -> str:
    """SHA-256 fingerprint including polygons, labels, and hierarchical sub-cell references."""
    records: list[str] = []
    for polygon in cell.polygons:
        records.append(f"P:{polygon.layer}:{polygon.datatype}:{bbox(polygon)}")
    for label in cell.labels:
        records.append(
            f"L:{label.text}:{label.layer}:{label.texttype}:"
            f"{tuple(round(float(value), 7) for value in label.origin)}"
        )
    for reference in cell.references:
        records.append(
            f"R:{reference.cell_name}:"
            f"{tuple(round(float(value), 7) for value in reference.origin)}:"
            f"{round(float(reference.rotation or 0), 7)}:"
            f"{int(reference.x_reflection)}"
        )
    return hashlib.sha256("\n".join(sorted(records)).encode()).hexdigest()


def extract_connectivity(
    cell: Any,
    expected_channels: int = 8,
) -> tuple[list[dict], dict[str, set[int]]]:
    """Extract device/net connectivity to audit the intended circuit graph."""
    gdstk = require_gdstk()
    gates = gdstk.boolean(
        layer_polygons(cell, GATE),
        layer_polygons(cell, GCUT),
        "not",
        precision=1e-6,
    )
    active = layer_polygons(cell, ACTIVE)
    channels = gdstk.boolean(active, gates, "and", precision=1e-6)
    diffusions = gdstk.boolean(active, gates, "not", precision=1e-6)
    _assert(
        len(channels) == expected_channels,
        f"expected {expected_channels} transistor channels, found {len(channels)}",
    )

    objects: list[tuple[str | int, Any]] = []
    objects.extend(("gate", p) for p in gates)
    objects.extend(("diff", p) for p in diffusions)
    for layer in (LIG, LISD, V0, M1, V1, M2, V2, M3, V3, M4, V4, M5):
        objects.extend((layer, p) for p in layer_polygons(cell, layer))

    same_layer = {"gate", "diff", LIG, LISD, M1, M2, M3, M4, M5}
    cross_layers = {
        frozenset(("gate", LIG)),
        frozenset(("diff", LISD)),
        frozenset((LIG, LISD)),
        frozenset((LIG, V0)),
        frozenset((LISD, V0)),
        frozenset((V0, M1)),
        frozenset((M1, V1)),
        frozenset((V1, M2)),
        frozenset((M2, V2)),
        frozenset((V2, M3)),
        frozenset((M3, V3)),
        frozenset((V3, M4)),
        frozenset((M4, V4)),
        frozenset((V4, M5)),
    }
    dsu = DisjointSetUnion(len(objects))
    for i, (left_layer, left_poly) in enumerate(objects):
        for j in range(i):
            right_layer, right_poly = objects[j]
            connect = (left_layer == right_layer and left_layer in same_layer)
            connect = connect or frozenset((left_layer, right_layer)) in cross_layers
            if connect and overlap(left_poly, right_poly):
                dsu.union(i, j)

    labels: dict[str, set[int]] = defaultdict(set)
    pin_layers = {LIG, LISD, M1, M2, M3, M4, M5}
    for label in cell.labels:
        if label.texttype != PIN_TEXTTYPE or label.layer not in pin_layers:
            continue
        hits = []
        for index, (layer, poly) in enumerate(objects):
            if layer == label.layer and poly.contain(label.origin):
                hits.append(dsu.find(index))
        _assert(bool(hits), f"pin {label.text} is not on drawing geometry")
        labels[label.text].update(hits)

    devices: list[dict] = []
    wells = layer_polygons(cell, WELL)
    fins = layer_polygons(cell, FIN)
    for channel in channels:
        (x0, y0), (x1, y1) = channel.bounding_box()
        gate_hits = [
            dsu.find(i)
            for i, (layer, poly) in enumerate(objects)
            if layer == "gate" and overlap(channel, poly)
        ]
        _assert(len(set(gate_hits)) == 1, "channel does not have one gate net")

        terminals: list[int] = []
        for side_x in (x0 - 0.0005, x1 + 0.0005):
            probe = gdstk.rectangle(
                (side_x - 0.00025, y0 + 0.0005),
                (side_x + 0.00025, y1 - 0.0005),
            )
            hits = [
                dsu.find(i)
                for i, (layer, poly) in enumerate(objects)
                if layer == "diff" and overlap(probe, poly)
            ]
            _assert(len(set(hits)) == 1, "channel does not have two diffusion terminals")
            terminals.append(hits[0])

        is_pmos = any(overlap(channel, well) for well in wells)
        nfin = sum(1 for fin in fins if overlap(channel, fin))
        _assert(nfin > 0, "channel is not crossed by any fin")
        devices.append({
            "gate": gate_hits[0],
            "terminals": tuple(terminals),
            "pmos": is_pmos,
            "nfin": nfin,
            "center": ((x0 + x1) / 2, (y0 + y1) / 2),
        })
    return devices, labels


def verify_rules(cell: Any, expected_marker: tuple[float, float, float, float] = MARKER) -> None:
    """Check the exact-grid and focused routing rules on an ASAP7 bitcell."""
    gdstk = require_gdstk()
    markers = {
        layer: [bbox(p) for p in layer_polygons(cell, layer)]
        for layer in MARKER_LAYERS
    }
    for layer, boxes in markers.items():
        _assert(boxes == [expected_marker], f"layer {layer} marker is not {expected_marker}")

    fins = sorted(layer_polygons(cell, FIN), key=lambda p: bbox(p)[1])
    centers = [round((bbox(p)[1] + bbox(p)[3]) / 2, 6) for p in fins]
    _assert(
        all(round(bbox(p)[3] - bbox(p)[1], 6) == 0.007 for p in fins),
        "FIN width is not exactly 7 nm",
    )
    _assert(
        all(round(right - left, 6) == 0.027 for left, right in zip(centers, centers[1:])),
        "FIN pitch is not exactly 27 nm",
    )

    gates = layer_polygons(cell, GATE)
    _assert(len(gates) == 2, "expected the two official gate columns")
    _assert(
        sorted(bbox(poly) for poly in gates) == sorted((GATE_A, GATE_B)),
        "GATE columns do not span the complete north/south boundary",
    )
    gate_centers = sorted(round((bbox(p)[0] + bbox(p)[2]) / 2, 6) for p in gates)
    _assert(gate_centers == [0.027, 0.081], "gate centers are off the 54 nm pitch")
    _assert(
        all(round(bbox(p)[2] - bbox(p)[0], 6) == 0.020 for p in gates),
        "GATE width is not exactly 20 nm",
    )

    for layer in (M1, M2, M3):
        for poly in layer_polygons(cell, layer):
            x0, y0, x1, y1 = bbox(poly)
            _assert(
                min(x1 - x0, y1 - y0) >= 0.018 - 1e-9,
                f"layer {layer} has a sub-18 nm shape",
            )

    m4 = layer_polygons(cell, M4)
    for poly in m4:
        x0, y0, x1, y1 = bbox(poly)
        _assert(round(y1 - y0, 6) == 0.024, "M4 route is not 24 nm wide")
        _assert(
            abs(y0 / 0.024 - round(y0 / 0.024)) < 1e-6
            and abs(y1 / 0.024 - round(y1 / 0.024)) < 1e-6,
            "M4 horizontal edges are off the 24 nm grid",
        )
        _assert(
            x1 - x0 >= 0.044 - 1e-9,
            "M4 horizontal length is below 44 nm (M4.W.5)",
        )
    m4_y = sorted((bbox(p)[1], bbox(p)[3]) for p in m4)
    _assert(
        all(
            next_y0 - y1 >= 0.024 - 1e-9
            for (_, y1), (next_y0, _) in zip(m4_y, m4_y[1:])
        ),
        "M4 tracks violate 24 nm spacing",
    )

    for poly in layer_polygons(cell, M5):
        x0, _y0, x1, _y1 = bbox(poly)
        _assert(round(x1 - x0, 6) == 0.024, "M5 route is not 24 nm wide")
        _assert(
            abs(x0 / 0.024 - round(x0 / 0.024)) < 1e-6
            and abs(x1 / 0.024 - round(x1 / 0.024)) < 1e-6,
            "M5 vertical edges are off the 24 nm grid",
        )

    lisd = layer_polygons(cell, LISD)
    for index, left in enumerate(lisd):
        for right in lisd[:index]:
            _assert(
                not overlap(left, right),
                "LISD contains redundant overlapping polygons",
            )
    if cell.name == "sram_cell_8t":
        wla_lig = {
            (0.016, -0.0115, 0.037, 0.0045),
            (0.071, 0.2655, 0.092, 0.2815),
        }
        _assert(
            wla_lig <= {bbox(poly) for poly in layer_polygons(cell, LIG)},
            "WLA LIG landings do not preserve 15 nm LISD spacing",
        )
        x_lo, _, x_hi, _ = edge_boundary(cell)
        for x0, _y0, x1, _y1 in wla_lig:
            _assert(
                x0 - x_lo >= 0.016 - 1e-9 and x_hi - x1 >= 0.016 - 1e-9,
                "a WLA LIG landing is within 16 nm of a placement seam",
            )
        storage_lisd = {
            (0.042, -0.0945, 0.066, -0.0405),
            (0.042, 0.0235, 0.066, 0.118),
            (0.042, 0.152, 0.066, 0.2465),
            (0.042, 0.3105, 0.066, 0.3645),
        }
        _assert(
            storage_lisd <= {bbox(poly) for poly in lisd},
            "storage LISD terminals are not canonical 24 nm shapes",
        )
        storage_v0 = {
            (0.045, -0.0765, 0.063, -0.0585),
            (0.045, 0.043, 0.063, 0.061),
            (0.045, 0.2195, 0.063, 0.2375),
            (0.045, 0.3285, 0.063, 0.3465),
        }
        _assert(
            storage_v0 <= {bbox(poly) for poly in layer_polygons(cell, V0)},
            "storage V0 landings are not centered at x=54 nm",
        )
        _assert(
            (-0.009, -0.0885, 0.009, -0.0705)
            in {bbox(poly) for poly in layer_polygons(cell, V0)},
            "BLBN V0 landing protrudes beyond its LISD terminal",
        )
    lisd_gate_overlap = gdstk.boolean(lisd, gates, "and", precision=1e-6)
    _assert(not lisd_gate_overlap, "LISD must not overlap either GATE column")
    lisd_union = gdstk.boolean(lisd, [], "or", precision=1e-6)
    uncovered_sdt = gdstk.boolean(
        layer_polygons(cell, SDT), lisd_union, "not", precision=1e-6
    )
    _assert(not uncovered_sdt, "SDT must be completely contained by LISD")


def verify_tap_rules(
    cell: Any,
    expected_marker: tuple[float, float, float, float] = TAP_MARKER,
    tap_slots: int = TAP_SLOTS,
) -> None:
    """Grid rules for the tap: the bitcell's frame without its devices."""
    for layer in (SRAMDRC, BOUNDARY):
        _assert(
            [bbox(p) for p in layer_polygons(cell, layer)] == [expected_marker],
            f"tap layer {layer} marker is not {expected_marker}",
        )
    _assert(
        not layer_polygons(cell, SRAMVT),
        "tap must not carry the SRAM-Vt implant",
    )

    fins = sorted(layer_polygons(cell, FIN), key=lambda p: bbox(p)[1])
    centres = [round((bbox(p)[1] + bbox(p)[3]) / 2, 6) for p in fins]
    _assert(
        all(round(bbox(p)[3] - bbox(p)[1], 6) == 0.007 for p in fins),
        "tap FIN width is not exactly 7 nm",
    )
    _assert(
        all(round(right - left, 6) == 0.027 for left, right in zip(centres, centres[1:])),
        "tap FIN pitch is not exactly 27 nm",
    )
    wanted = sorted(
        (g[0] + k * 0.108, g[1], g[2] + k * 0.108, g[3])
        for k in range(tap_slots)
        for g in (GATE_A, GATE_B)
    )
    _assert(
        sorted(bbox(p) for p in layer_polygons(cell, GATE)) == wanted,
        "tap does not reuse the bitcell gate columns in every slot",
    )
