"""ASAP7 GDS geometry transformations, orientation reflection, and shape adjustments.

These utilities operate on axis-aligned shapes and gdstk structures.
Import of gdstk is lazy so the module remains usable without gdstk installed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

__all__ = [
    "bbox",
    "clip_gcut_to_cell",
    "clone_label",
    "clone_polygons",
    "copy_labels",
    "direct_label",
    "dummy_array_name",
    "edge_boundary",
    "have_gdstk",
    "indexed_labels",
    "is_box",
    "oriented_cell",
    "oriented_name",
    "placed_def_instances",
    "rect",
    "require_gdstk",
    "snap_sdt_heights",
    "topbot_name",
]

_gdstk_module: ModuleType | None = None


def require_gdstk() -> ModuleType:
    """Return the gdstk module, importing it on first use."""
    global _gdstk_module
    if _gdstk_module is None:
        try:
            import gdstk
        except ImportError as exc:
            raise ImportError(
                "gdstk is required for layout transformation helpers; "
                "install it with 'pip install gdstk'"
            ) from exc
        _gdstk_module = gdstk
    return _gdstk_module


def have_gdstk() -> bool:
    """True when gdstk is installed and importable in this environment."""
    try:
        require_gdstk()
        return True
    except ImportError:
        return False


def rect(
    cell: Any,
    box: tuple[float, float, float, float],
    layer: int,
    datatype: int = 0,
) -> None:
    """Add an axis-aligned drawing rectangle to a cell."""
    gdstk = require_gdstk()
    x0, y0, x1, y1 = box
    cell.add(gdstk.rectangle((x0, y0), (x1, y1), layer=layer, datatype=datatype))


def bbox(poly: Any) -> tuple[float, float, float, float]:
    """Bounding box of a polygon as (x0, y0, x1, y1), rounded to 7 decimal places."""
    (x0, y0), (x1, y1) = poly.bounding_box()
    return tuple(round(float(v), 7) for v in (x0, y0, x1, y1))


def is_box(poly: Any, expected: tuple[float, float, float, float]) -> bool:
    """True when `poly`'s bounding box matches `expected` within 7 decimal places."""
    return bbox(poly) == tuple(round(v, 7) for v in expected)


def edge_boundary(cell: Any, boundary_layer: int = 100) -> tuple[float, float, float, float]:
    """Bounding box of the cell's single direct BOUNDARY polygon."""
    boxes = [bbox(p) for p in cell.polygons if p.layer == boundary_layer and p.datatype == 0]
    if len(boxes) != 1:
        raise ValueError(f"{cell.name}: expected one direct BOUNDARY on layer {boundary_layer}")
    return boxes[0]


def oriented_name(base: str, mirror_x: bool = False, mirror_y: bool = False) -> str:
    """_lr reverses X; _v2 reverses Y, both about the placement boundary."""
    return base + ("_v2" if mirror_y else "") + ("_lr" if mirror_x else "")


def topbot_name(mirror_x: bool = False, mirror_y: bool = False) -> str:
    """Col-cap orientation naming convention."""
    return f"dummy_topbot_8t_v{2 if mirror_y else 1}" + ("_lr" if mirror_x else "")


def dummy_array_name(
    rows: int,
    tap_pitch: int = 0,
    mirror_x: bool = False,
    mirror_y: bool = False,
) -> str:
    """Name of a parameterized dummy vertical array."""
    tap = f"_tap{tap_pitch}" if tap_pitch else ""
    return oriented_name(f"dummy_vertical_array_X{rows}{tap}_8t", mirror_x, mirror_y)


def clone_label(
    label: Any,
    text: str | None = None,
    origin: tuple[float, float] | None = None,
    layer: int | None = None,
    texttype: int | None = None,
) -> Any:
    """Create a copy of a gdstk Label with an optional text, origin, layer, or texttype override."""
    gdstk = require_gdstk()
    return gdstk.Label(
        text if text is not None else label.text,
        origin if origin is not None else tuple(label.origin),
        anchor=label.anchor,
        rotation=label.rotation,
        magnification=label.magnification,
        x_reflection=label.x_reflection,
        layer=layer if layer is not None else label.layer,
        texttype=texttype if texttype is not None else label.texttype,
    )


def direct_label(cell: Any, name: str, layer: int | None = None) -> Any:
    """Find a unique direct label in cell matching text `name` and optional `layer`."""
    labels = [
        label for label in cell.labels
        if label.text == name and (layer is None or label.layer == layer)
    ]
    if len(labels) != 1:
        raise RuntimeError(
            f"{cell.name}: expected one direct {name!r} label, found {len(labels)}"
        )
    return labels[0]


def indexed_labels(cell: Any, prefix: str) -> dict[int, Any]:
    """Find labels matching `prefix[index]` and return `{index: label}`."""
    pattern = re.compile(rf"{re.escape(prefix)}\[([0-9]+)\]")
    result: dict[int, Any] = {}
    for label in cell.labels:
        match = pattern.fullmatch(label.text)
        if not match:
            continue
        index = int(match.group(1))
        if index in result:
            raise RuntimeError(f"{cell.name}: duplicate {label.text}")
        result[index] = label
    return result


def placed_def_instances(
    def_path: Path,
    cells: dict[str, Any],
    master_sizes: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """Parse placed and fixed instances from a DEF file and return {instance_name: gdstk.Reference}."""
    gdstk = require_gdstk()
    text = def_path.read_text()
    dbu_match = re.search(r"UNITS DISTANCE MICRONS (\d+)", text)
    if not dbu_match:
        raise RuntimeError(f"{def_path}: missing UNITS DISTANCE MICRONS statement")
    dbu = int(dbu_match.group(1))
    if "COMPONENTS " not in text or "END COMPONENTS" not in text:
        raise RuntimeError(f"{def_path}: missing COMPONENTS section")
    section = text.split("COMPONENTS ", 1)[1].split("END COMPONENTS", 1)[0]
    result: dict[str, Any] = {}
    for match in re.finditer(
        r"-\s+(\S+)\s+(\S+).*?\+\s+(?:PLACED|FIXED)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\w+)\s*;",
        section,
        re.S,
    ):
        name, master, x_str, y_str, orient = match.groups()
        name = re.sub(r"\\(.)", r"\1", name)
        x, y = int(x_str) / dbu, int(y_str) / dbu
        if master not in master_sizes:
            raise RuntimeError(f"unknown master {master} in {def_path}")
        width, height = master_sizes[master]
        transforms = {
            "N": ((x, y), 0, False),
            "FS": ((x, y + height), 0, True),
            "FN": ((x + width, y), math.pi, True),
            "S": ((x + width, y + height), math.pi, False),
        }
        if orient not in transforms:
            raise RuntimeError(f"unsupported standard-cell orientation {orient}")
        origin, rotation, reflection = transforms[orient]
        if master not in cells:
            raise RuntimeError(f"missing cell master {master} for instance {name}")
        result[name] = gdstk.Reference(
            cells[master], origin=origin, rotation=rotation, x_reflection=reflection
        )
    if not result:
        raise RuntimeError(f"{def_path}: routed DEF has no placed instances")
    return result


def clone_polygons(
    source: Any,
    target: Any,
    predicate: Callable[[Any], bool],
    dx: float = 0.0,
) -> None:
    """Copy matching polygons from source to target, optionally translating in X."""
    gdstk = require_gdstk()
    for poly in source.polygons:
        if predicate(poly):
            clone = gdstk.Polygon(
                poly.points.copy(), layer=poly.layer, datatype=poly.datatype
            )
            if dx:
                clone.translate(dx, 0.0)
            target.add(clone)


def copy_labels(
    source: Any,
    target: Any,
    names: set[str],
    rename: dict[str, str] | None = None,
) -> None:
    """Copy matching labels from source to target, applying optional rename."""
    rename = rename or {}
    for label in source.labels:
        if label.text in names:
            target.add(clone_label(label, rename.get(label.text, label.text)))


def oriented_cell(
    source: Any,
    name: str,
    mirror_x: bool = False,
    mirror_y: bool = False,
    boundary_layer: int = 100,
) -> Any:
    """Materialize all polygons and pins, reflecting around the canonical BOUNDARY center.

    Reflecting around the boundary center (never the raw geometry bounding box)
    preserves correct process overhangs and implant band alignment.
    """
    gdstk = require_gdstk()
    x0, y0, x1, y1 = edge_boundary(source, boundary_layer)
    cell = gdstk.Cell(name)
    for poly in source.get_polygons():
        points = poly.points.copy()
        if mirror_x:
            points[:, 0] = x0 + x1 - points[:, 0]
        if mirror_y:
            points[:, 1] = y0 + y1 - points[:, 1]
        cell.add(gdstk.Polygon(points, layer=poly.layer, datatype=poly.datatype))
    for label in source.get_labels():
        copied = clone_label(label)
        x, y = map(float, label.origin)
        copied.origin = (
            x0 + x1 - x if mirror_x else x,
            y0 + y1 - y if mirror_y else y,
        )
        cell.add(copied)
    return cell


def clip_gcut_to_cell(
    cell: Any,
    x0: float = 0.0,
    x1: float = 0.108,
    gcut_layer: int = 10,
) -> None:
    """Trim GCUT polygons to the cell's horizontal [x0, x1] boundary.

    Prevents sub-grid overhang slivers beside taps or cap cells while
    allowing mirrored neighbour cells to abut cleanly at the seam.
    """
    for poly in [p for p in cell.polygons if p.layer == gcut_layer]:
        px0, py0, px1, py1 = bbox(poly)
        if px0 < x0 - 1e-9 or px1 > x1 + 1e-9:
            cell.remove(poly)
            rect(cell, (max(px0, x0), py0, min(px1, x1), py1), gcut_layer)


def snap_sdt_heights(
    cell: Any,
    sdt_pitch: float = 0.027,
    sdt_lig_space: float = 0.014,
    lisd_lig_space: float = 0.015,
    active_layer: int = 11,
    lig_layer: int = 16,
    lisd_layer: int = 17,
    sdt_layer: int = 88,
) -> None:
    """Snap every SDT contact trench to an integer multiple of fin pitches (SDT.W.3).

    Each trench becomes the shortest fin-pitch multiple that still covers the
    original, keeping minimum clearance from unrelated gate contacts, and
    growing LISD as needed to cover the trench without violating LISD-to-LIG spacing.
    """
    def box_gap(a: Sequence[float], b: Sequence[float]) -> float:
        dx = max(0.0, b[0] - a[2], a[0] - b[2])
        dy = max(0.0, b[1] - a[3], a[1] - b[3])
        return math.hypot(dx, dy)

    actives = [bbox(p) for p in cell.polygons if p.layer == active_layer]
    ligs = [bbox(p) for p in cell.polygons if p.layer == lig_layer]
    for poly in [p for p in cell.polygons if p.layer == sdt_layer]:
        x0, y0, x1, y1 = bbox(poly)
        pitches = (y1 - y0) / sdt_pitch
        if abs(pitches - round(pitches)) < 1e-6:
            continue
        height = math.ceil(pitches - 1e-6) * sdt_pitch
        lisd_poly = next(
            p for p in cell.polygons
            if p.layer == lisd_layer
            and bbox(p)[0] <= x0 + 1e-9
            and bbox(p)[2] >= x1 - 1e-9
            and bbox(p)[1] <= y0 + 1e-9
            and bbox(p)[3] >= y1 - 1e-9
        )
        lisd = bbox(lisd_poly)
        others = [lig for lig in ligs if box_gap(lig, (x0, y0, x1, y1)) > 1e-9]
        lisd_others = [lig for lig in ligs if box_gap(lig, lisd) > 1e-9]

        def on_active(lo: float, hi: float) -> float:
            return sum(
                max(0.0, min(hi, a[3]) - max(lo, a[1]))
                for a in actives if a[0] < x1 and a[2] > x0
            )

        covered = on_active(y0, y1)
        best = None
        for step in range(round((height - (y1 - y0)) / 0.0005) + 1):
            lo = round(y1 - height + step * 0.0005, 4)
            window = (x0, lo, x1, lo + height)
            if any(box_gap(window, lig) < sdt_lig_space - 1e-9 for lig in others):
                continue
            growth = max(0.0, lisd[1] - lo) + max(0.0, window[3] - lisd[3])
            grown = (lisd[0], min(lisd[1], lo), lisd[2], max(lisd[3], window[3]))
            if growth and any(box_gap(grown, lig) < lisd_lig_space - 1e-9 for lig in lisd_others):
                continue
            area = on_active(lo, window[3])
            if area < covered - 1e-9:
                continue
            key = (round(growth, 4), -round(area, 4))
            if best is None or key < best[0]:
                best = (key, window)
        if best is None:
            raise RuntimeError(f"{cell.name}: no {height * 1000:.0f} nm SDT window for x {x0:.3f} y {y0:.4f}")
        cell.remove(poly)
        rect(cell, best[1], sdt_layer)
        if best[0][0] > 0:
            cell.remove(lisd_poly)
            rect(cell, (lisd[0], min(lisd[1], best[1][1]), lisd[2], max(lisd[3], best[1][3])), lisd_layer)
