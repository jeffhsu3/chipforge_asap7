"""Shared-diffusion BROM peripheral cells and their actual physical plans."""

from hashlib import sha256
from typing import Any

from .dense_bank import DenseBank, build_dense_bank
from .dense_cgen import build_dense_lane
from .dense_macro import canonicalize_cells, verify_dense_routes


def geometry_digest(cell: Any) -> str:
    """Short hash of a flat cell's drawn shapes, independent of polygon order."""
    shapes = sorted(
        (layer, datatype, tuple(map(tuple, points.round(3).tolist())))
        for (layer, datatype), polygons in cell.get_polygons(by_spec=True).items()
        for points in polygons
    )
    return sha256(repr(shapes).encode()).hexdigest()[:12]


def periphery_plan(circuit):
    if circuit.array is not None:
        raise ValueError("use build_macro for a circuit with a selector array")
    if not circuit.name.startswith("brom_cgen_"):
        return build_dense_bank(circuit, name=circuit.name)
    lane = build_dense_lane(circuit.n.fins, circuit.n.vt)
    lane.cell.name = circuit.name
    pins = {**lane.inputs, **lane.outputs}
    checkpoints = [(n, *p) for n, p in pins.items()]
    for net, metal, a, b in lane.routes:
        if not net.startswith("cell") or metal != "M3":
            continue
        slot, n = net.split(":")
        k = int(slot[4:])
        if n not in ("G", "SEL", "C"):
            continue
        name = f"{n}<{k}>" if n in ("G", "C") else n
        checkpoints.append((name, metal, a[0], (a[1] + b[1]) / 2))
    return DenseBank(
        lane.cell,
        pins,
        lane.width,
        432,
        [],
        lane.routes,
        checkpoints,
        lane.certificates,
    )


def build_periphery(circuit, *, lib=None, label_pins=True):
    plan = periphery_plan(circuit)
    top = plan.cell
    verify_dense_routes(top, plan.checkpoints)
    if not label_pins:
        top.labels.clear()
    canonicalize_cells(top)
    if lib is not None:
        if lib.unit != 1e-9:
            raise ValueError("BROM layout requires lib.unit=1e-9")
        lib.add(top)
    return top


def bus_positions(circuit):
    return {n: p[2] for n, p in periphery_plan(circuit).pins.items()}


def verify_periphery_routes(circuit, top, *, plan=None):
    plan = plan or periphery_plan(circuit)
    verify_dense_routes(top, plan.checkpoints)
