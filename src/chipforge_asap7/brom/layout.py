"""Compact selector-array macro assembly and peripheral routing.

The macro always uses the periodic selector core. M2 candidates leave on
private M3 access columns, M4 candidates leave on their native tracks, and
M5 trunks join every array interface to the peripheral bank above the core.
"""

from __future__ import annotations

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


def selector_placements(circuit: Circuit):
    """Keep replica BL escapes to the right of the data core, candidates above it."""
    if circuit.array is None:
        raise ValueError("build_macro requires a macro_circuit with a selector array")
    yield "", circuit.array, 0, 0
    if circuit.replica is not None:
        width = len(circuit.array.mapping[0]) * circuit.array.plan.width
        height = len(circuit.array.mapping) * circuit.array.plan.height
        x = ((width + 432 + 431) // 432) * 432
        yield "R", circuit.replica.selector, x, height + 432


def macro_plan(circuit: Circuit) -> MacroPlan:
    placements = list(selector_placements(circuit))
    ports = []
    bank_y = 0
    for prefix, array, ox, oy in placements:
        blocks, columns = len(array.mapping), len(array.mapping[0])
        height = blocks * array.plan.height
        bank_y = max(bank_y, oy + height + 432)

        def net(name, prefix=prefix):
            return circuit.replica.port_nets[name] if prefix else name

        ports += [
            ArrayPort(
                net(f"C<{b},{k}>"), metal, ox - 24, oy + b * array.plan.height + y
            )
            for b in range(blocks)
            for k, (metal, y) in enumerate(array.plan.tracks)
        ]
        ports += [
            ArrayPort(net(f"BL<{col}>"), "M3", ox + col * array.plan.width + 81, oy)
            for col in range(columns)
        ]
        ports.append(ArrayPort("VSS", "M2", ox, oy + height + 81))
    access = tuple(-144 - 72 * i for i in range(len(ports)))
    first_trunk = ((min(access) - 192 - 12) // 48) * 48 + 12
    trunks = tuple(first_trunk - 96 * i for i in range(len(ports)))
    routes = tuple(
        p.y if p.layer == "M4" else -84 - 96 * i for i, p in enumerate(ports)
    )
    array = circuit.array
    return MacroPlan(
        tuple(ports),
        bank_y,
        access,
        trunks,
        routes,
        len(array.mapping[0]) * array.plan.width,
        len(array.mapping) * array.plan.height,
    )


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
    array_devices = set(circuit.array.device_names)
    if circuit.replica is not None:
        array_devices.update(circuit.replica.selector.device_names)
    bank.devices = [d for d in circuit.devices if d.name not in array_devices]
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
    for prefix, instance, x, y in selector_placements(circuit):
        array = build_selector_array(instance.plan, instance.mapping)
        array.name += "_replica" if prefix else "_core"
        array.labels.clear()
        if lib is not None:
            if array.name in lib.cells:
                array = lib.cells[array.name]
            else:
                lib.add(array)
        top.add(gdspy.CellReference(array, (x, y)))
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
            box(top, "M3", port.x - 9, y - 17, port.x + 9, port.y + 14)
            _v3(top, port.x, y)
            _m4(top, trunk, port.x, y)
        else:
            _m4(top, trunk, port.x + 24, y)
        _v4(top, trunk, y)
        box(top, "M5", trunk - 12, min(y, bank_y) - 24, trunk + 12, max(y, bank_y) + 24)
        _v4(top, trunk, bank_y)
        _m4(top, trunk, 0, bank_y)
    bounds = top.get_bounding_box()
    for pin in circuit.pins:
        y = plan.bank_y + buses[pin]
        # Carry the pin's M4 bus out to the left edge, where a router can reach it.
        box(top, "M4", bounds[0, 0], y - 12, 48, y + 12)
        label(top, pin, "M4", (24, y))
    box(top, "BOUNDARY", bounds[0, 0], bounds[0, 1], bounds[1, 0], bounds[1, 1])
    verify_macro_routes(circuit, top)
    return top


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
        if source in seen and seen[source] != port.net:
            raise ValueError(f"shorted macro routes: {seen[source]} and {port.net}")
        seen[source] = port.net
