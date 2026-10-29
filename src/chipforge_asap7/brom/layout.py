"""Compact selector-array macro assembly and peripheral routing.

The macro always uses the periodic selector core. M2 candidates leave on
private M3 access columns, M4 candidates leave on their native tracks, and
M5 trunks join every array interface to the peripheral bank above the core.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from ..devices.rowcell import label, square
from ..layout.layers import box, require_gdspy
from .circuit import Circuit
from .periphery import build_periphery, bus_positions
from .routing import metal_components
from .selector import build_selector_array


@dataclass(frozen=True)
class ArrayPort:
    net: str
    layer: str
    x: float
    y: float


@dataclass(frozen=True)
class MacroPlan:
    ports: tuple[ArrayPort, ...]
    bank_y: int
    access_x: tuple[int, ...]
    trunk_x: tuple[int, ...]
    route_y: tuple[float, ...]
    array_width: int
    array_height: int


def macro_plan(circuit: Circuit) -> MacroPlan:
    if circuit.array is None:
        raise ValueError("build_macro requires a macro_circuit with a selector array")
    array = circuit.array
    blocks, columns = len(array.mapping), len(array.mapping[0])
    width, height = columns * array.plan.width, blocks * array.plan.height
    ports = [
        ArrayPort(f"C<{b},{k}>", metal, -24, b * array.plan.height + y)
        for b in range(blocks)
        for k, (metal, y) in enumerate(array.plan.tracks)
    ]
    ports += [
        ArrayPort(f"BL<{c}>", "M3", c * array.plan.width + 81, 0)
        for c in range(columns)
    ]
    ports.append(ArrayPort("VSS", "M2", 0, height + 81))
    access = tuple(-144 - 72 * i for i in range(len(ports)))
    # M5 centres lie on 12 + 48*k. Twice-minimum pitch separates unequal tips.
    first_trunk = ((min(access) - 192 - 12) // 48) * 48 + 12
    trunks = tuple(first_trunk - 96 * i for i in range(len(ports)))
    routes = tuple(
        p.y if p.layer == "M4" else -84 - 96 * i for i, p in enumerate(ports)
    )
    return MacroPlan(tuple(ports), height + 432, access, trunks, routes, width, height)


def macro_periphery(circuit: Circuit) -> Circuit:
    plan = macro_plan(circuit)
    bank = Circuit(
        circuit.name + "_periphery",
        tuple(dict.fromkeys([*circuit.pins, *(p.net for p in plan.ports)])),
        circuit.n.fins,
        circuit.n.vt,
    )
    bank.array_width = plan.array_width
    bank.placement_groups = list(circuit.placement_groups)
    bank.devices = [
        d for d in circuit.devices if d.name not in circuit.array.device_names
    ]
    return bank


def _m4(top, x0, x1, y):
    lo, hi = sorted((x0, x1))
    box(top, "M4", lo - 24, y - 12, hi + 24, y + 12)


def _v3(top, x, y):
    box(top, "V3", x - 9, y - 12, x + 9, y + 12)


def _v4(top, x, y):
    box(top, "V4", x - 12, y - 12, x + 12, y + 12)


def build_macro(circuit: Circuit, *, lib: Any = None) -> Any:
    """Place the compact programmed array and route every peripheral interface."""
    plan = macro_plan(circuit)
    gdspy = require_gdspy()
    if lib is not None and lib.unit != 1e-9:
        raise ValueError("BROM layout requires lib.unit=1e-9")
    top = gdspy.Cell(circuit.name, exclude_from_current=True)
    if lib is not None:
        lib.add(top)
    array = build_selector_array(circuit.array.plan, circuit.array.mapping, lib=lib)
    array.labels.clear()  # Array interface nets are internal to this macro.
    top.add(gdspy.CellReference(array))
    bank = macro_periphery(circuit)
    bank_cell = build_periphery(bank, lib=lib, label_pins=False)
    top.add(gdspy.CellReference(bank_cell, (0, plan.bank_y)))
    buses = bus_positions(bank)
    for i, port in enumerate(plan.ports):
        access, trunk, y = plan.access_x[i], plan.trunk_x[i], plan.route_y[i]
        bank_y = plan.bank_y + buses[port.net]
        if port.layer == "M2":
            box(top, "M2", access - 17, port.y - 9, port.x + 24, port.y + 9)
            square(top, "V2", access, port.y)
            box(
                top,
                "M3",
                access - 9,
                min(y, port.y) - 17,
                access + 9,
                max(y, port.y) + 17,
            )
            _v3(top, access, y)
            _m4(top, trunk, access, y)
        elif port.layer == "M3":
            box(top, "M3", port.x - 9, y - 17, port.x + 9, 14)
            _v3(top, port.x, y)
            _m4(top, trunk, port.x, y)
        else:
            _m4(top, trunk, 0, y)
        _v4(top, trunk, y)
        box(top, "M5", trunk - 12, min(y, bank_y) - 24, trunk + 12, max(y, bank_y) + 24)
        _v4(top, trunk, bank_y)
        _m4(top, trunk, 0, bank_y)
    for pin in circuit.pins:
        label(top, pin, "M4", (24, plan.bank_y + buses[pin]))
    bounds = top.get_bounding_box()
    box(top, "BOUNDARY", bounds[0, 0], bounds[0, 1], bounds[1, 0], bounds[1, 1])
    verify_macro_routes(circuit, top)
    return top


def build_circuit(circuit: Circuit, *, lib: Any = None) -> Any:
    """Deprecated API: dispatches to the new macro/peripheral builders only."""
    warnings.warn(
        "build_circuit is deprecated; use build_macro or build_periphery",
        DeprecationWarning,
        stacklevel=2,
    )
    if circuit.array is not None:
        return build_macro(circuit, lib=lib)
    return build_periphery(circuit, lib=lib)


def verify_macro_routes(circuit: Circuit, top: Any) -> None:
    """Check actual macro-owned metal routes without substrate connectivity.

    Child cells are qualified by DRC/LVS separately. These rectangles must
    connect each array port to its bank bus, with no shared route components.
    """
    component = metal_components(top)

    plan = macro_plan(circuit)
    buses = bus_positions(macro_periphery(circuit))
    seen = {}
    for port in plan.ports:
        source = component(port.layer, port.x, port.y)
        sink = component("M4", 0, plan.bank_y + buses[port.net])
        if source != sink:
            raise ValueError(f"disconnected macro route: {port.net}")
        if source in seen:
            raise ValueError(f"shorted macro routes: {seen[source]} and {port.net}")
        seen[source] = port.net
