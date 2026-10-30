"""Routed peripheral banks for the compact-array BROM macro.

These device-level layouts contain only peripheral logic. Array transistors
must be placed by the periodic selector generator, never by this builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from ..devices.finfet import build_finfet
from ..devices.rowcell import label, square
from ..layout.layers import LAYERS, box, require_gdspy
from .circuit import Circuit, Device
from .routing import metal_components


def geometry_digest(cell: Any) -> str:
    """Short hash of a flat cell's drawn shapes, independent of polygon order."""
    shapes = sorted(
        (layer, datatype, tuple(map(tuple, points.round(3).tolist())))
        for (layer, datatype), polygons in cell.get_polygons(by_spec=True).items()
        for points in polygons
    )
    return sha256(repr(shapes).encode()).hexdigest()[:12]


def _device(spec: Any, lib: Any) -> Any:
    """Single-finger access: narrow drain column and a local source pad.

    The standalone FinFET's wide folded-drain landing violates the current
    M1 short-edge rule for this single-finger use. Keep its device/tap
    geometry, but use vertical drain access and stop the source metal at
    its contact instead of extending it beneath the drain.

    The cell is named after its drawn geometry as well as its spec, so two
    macros whose device cells differ (say, from different compiler versions)
    can never meet under one name when a chip merges their GDS.
    """
    cell = build_finfet(spec, name=f"brom_{spec.cell_name}", draw_pin_labels=False)
    act_lo, act_hi = spec.active_span()
    drain = spec.drain_xs[0]
    targets = {
        (spec.device_x0, -9, spec.width, 9),
        (drain - 14, act_lo, drain + 14, act_hi),
    }
    cell.remove_polygons(
        lambda points, layer, datatype: (
            layer == LAYERS["M1"]["layer"]
            and datatype == 0
            and (
                min(points[:, 0]),
                min(points[:, 1]),
                max(points[:, 0]),
                max(points[:, 1]),
            )
            in targets
        )
    )
    source = spec.source_xs[0]
    box(cell, "M1", source - 14, -9, source + 14, 9)
    _, dy = spec.pin_positions["D"]
    box(cell, "M1", drain - 9, min(act_lo, dy - 14), drain + 9, max(act_hi, dy + 14))
    cell.name += f"_{geometry_digest(cell)}"
    if lib is not None:
        if cell.name in lib.cells:
            return lib.cells[cell.name]
        lib.add(cell)
    return cell


@dataclass(frozen=True)
class PeripheralRow:
    devices: tuple[Device, ...]
    y: int
    buses: dict[str, int]
    ceiling: int


@dataclass(frozen=True)
class PeripheralPlan:
    rows: tuple[PeripheralRow, ...]
    device_pitch: int
    row_capacity: int
    spine_x: dict[str, int]
    pins: dict[str, int]
    width: int
    height: int
    optimization: dict | None = None


def _ceil(value: int, pitch: int) -> int:
    return ((value + pitch - 1) // pitch) * pitch


def bus_span(xs, pin, compact=False):
    """Drawn M4 extent; aligned tips satisfy S.2/S.3/S.4/S.5 at 48 nm pitch."""
    left, right = (0 if pin else min(xs) - 24), max(xs) + 24
    if compact:
        left = (left // 48) * 48
        right = _ceil(right, 48)
    return left, right


def periphery_plan(circuit: Circuit) -> PeripheralPlan:
    """Fold functional groups on integer multiples of the selector lattice.

    Local nets stay within their row. Only cross-row nets and external pins
    receive vertical M5 distribution. Pins escape directly from their owning
    row on M4, avoiding an extra global route for every candidate and bitline.
    """
    if circuit.array is not None:
        raise ValueError(
            "macro circuits must use build_macro and the compact selector array"
        )
    if not circuit.devices:
        raise ValueError("peripheral circuit must contain devices")
    pitch = _ceil(max(d.spec.width for d in circuit.devices) + 432, 108)
    # A narrow array still needs room for a complete eight-transistor latch.
    capacity = max(8, min(32, (circuit.array_width or 16 * pitch) // pitch))
    groups = {}
    for name, members in circuit.placement_groups:
        if len(members) > capacity:
            raise ValueError(f"peripheral group {name} exceeds row capacity")
        for member in members:
            if member in groups:
                raise ValueError(f"overlapping peripheral group: {member}")
            groups[member] = members
    by_name = {d.name: d for d in circuit.devices}
    packed, current, placed = [], [], set()
    for device in circuit.devices:
        if device.name in placed:
            continue
        members = groups.get(device.name, (device.name,))
        if any(m not in by_name for m in members):
            raise ValueError("peripheral group contains an absent device")
        if len(current) + len(members) > capacity:
            packed.append(tuple(current))
            current = []
        current.extend(by_name[m] for m in members)
        placed.update(members)
    if current:
        packed.append(tuple(current))
    owners = {}
    for index, devices in enumerate(packed):
        for net in sorted({n for d in devices for n in d.nodes}):
            owners.setdefault(net, []).append(index)
    shared = sorted(n for n, rs in owners.items() if len(rs) > 1)
    occupied = max(map(len, packed)) if circuit.compact_periphery else capacity
    first_spine = _ceil(108 + occupied * pitch + 192, 48) + 12
    track_pitch = 48 if circuit.compact_periphery else 96
    spine_pitch = 48 if circuit.compact_periphery else 96
    spines = {net: first_spine + spine_pitch * i for i, net in enumerate(shared)}
    rows, certificates = [], []
    y = 0
    for index, devices in enumerate(packed):
        nets = sorted({n for d in devices for n in d.nodes})
        first = _ceil(y + max(d.spec.height for d in devices) + 192, 48) + 12
        tracks = tuple(range(len(nets)))
        if circuit.compact_periphery:
            from .compact import assign_tracks

            accesses = {n: [] for n in nets}
            for i, d in enumerate(devices):
                cursor = 108 + i * pitch
                columns = (
                    cursor + d.spec.width + 144,
                    cursor + d.spec.width + 216,
                    cursor + d.spec.width + 72,
                    cursor - 72,
                )
                for net, x in zip(d.nodes, columns, strict=True):
                    accesses[net].append(x)
            intervals = []
            for net in nets:
                xs = accesses[net]
                if net in spines:
                    xs.append(spines[net])
                pin = net in circuit.pins and owners[net][0] == index
                left, right = bus_span(xs, pin, True)
                # M4.S.2: 40 nm tip spacing (aligned endpoints round to 48).
                intervals.append((left, right + 40))
            tracks, clique = assign_tracks(tuple(intervals), end_space=40)
            certificates.append(
                {
                    "status": "OPTIMAL",
                    "tracks": max(tracks) + 1,
                    "lower_bound": max(tracks) + 1,
                    "clique_lower_bound": len(clique),
                    "clique": [nets[i] for i in clique],
                    "intervals_nm": dict(zip(nets, intervals, strict=True)),
                    "assignment": dict(zip(nets, tracks, strict=True)),
                }
            )
        buses = {net: first + track_pitch * tracks[i] for i, net in enumerate(nets)}
        ceiling = max(buses.values()) + 36
        rows.append(PeripheralRow(devices, y, buses, ceiling))
        y = _ceil(ceiling + 162, 432)
    if circuit.compact_periphery:
        # M5 uses a 48 nm center grid. Runs whose ends just meet need a
        # spare track for 40 nm corner clearance; ordinary parallel runs do
        # not. Moving spines outward preserves the row conflict graph: all
        # spine-connected routes already overlap beyond every device access.
        prior_span = None
        x = first_spine
        for net in shared:
            ys = [rows[i].buses[net] for i in owners[net]]
            span = (min(ys) - 24, max(ys) + 24)
            if prior_span is not None:
                overlap = min(span[1], prior_span[1]) - max(span[0], prior_span[0])
                x += 96 if -40 < overlap < 44 else 48
            spines[net] = x
            prior_span = span
        for certificate in certificates:
            for net, (left, right) in certificate["intervals_nm"].items():
                if net in spines:
                    certificate["intervals_nm"][net] = (
                        left,
                        _ceil(spines[net] + 24, 48) + 40,
                    )
    pins = {}
    for net in sorted(circuit.pins):
        if net in owners:
            pins[net] = rows[owners[net][0]].buses[net]
        else:
            pins[net] = y + 12
            y += 96
    width = _ceil(max(spines.values(), default=first_spine) + 108, 432)
    height = _ceil(max(y, max(pins.values(), default=y)) + 108, 432)
    optimization = None
    if circuit.compact_periphery:
        optimization = {
            "method": "cp_sat_interval_channels_v1",
            "scope": "minimum tracks for fixed devices, row packing and spines",
            "track_pitch_nm": track_pitch,
            "spine_grid_nm": spine_pitch,
            "spine_spacing": "48 nm; 96 nm for facing tips or short overlaps",
            "rows": certificates,
            "timing_qualified": False,
        }
    return PeripheralPlan(
        tuple(rows), pitch, capacity, spines, pins, width, height, optimization
    )


def bus_positions(circuit: Circuit) -> dict[str, int]:
    """External M4 pin locations of the folded bank."""
    return periphery_plan(circuit).pins


def build_periphery(
    circuit: Circuit, *, lib: Any = None, label_pins: bool = True
) -> Any:
    plan = periphery_plan(circuit)
    gdspy = require_gdspy()
    if lib is not None and lib.unit != 1e-9:
        raise ValueError("BROM layout requires lib.unit=1e-9")
    top = gdspy.Cell(circuit.name, exclude_from_current=True)
    if lib is not None:
        lib.add(top)
    cache = {}
    spine_ys = {net: [] for net in plan.spine_x}
    for row in plan.rows:
        accesses = {net: [] for net in row.buses}
        for i, device in enumerate(row.devices):
            spec = device.spec
            if spec not in cache:
                cache[spec] = _device(spec, lib)
            cursor = 108 + i * plan.device_pitch
            top.add(gdspy.CellReference(cache[spec], (cursor, row.y)))
            columns = {
                "B": cursor - 72,
                "S": cursor + spec.width + 72,
                "D": cursor + spec.width + 144,
                "G": cursor + spec.width + 216,
            }
            for pin, net in zip(("D", "G", "S", "B"), device.nodes, strict=True):
                px, py = spec.pin_positions[pin]
                px, py = px + cursor, py + row.y
                tx = columns[pin]
                square(top, "V1", px, py)
                lo, hi = sorted((px, tx))
                box(top, "M2", lo - 17, py - 9, hi + 17, py + 9)
                square(top, "V2", tx, py)
                # Stop terminal columns at their own net, not the bank ceiling.
                y = row.buses[net]
                box(top, "M3", tx - 9, py - 14, tx + 9, y + 17)
                box(top, "V3", tx - 9, y - 12, tx + 9, y + 12)
                accesses[net].append(tx)
        for net, y in row.buses.items():
            xs = accesses[net]
            if net in plan.spine_x:
                x = plan.spine_x[net]
                xs.append(x)
                box(top, "V4", x - 12, y - 12, x + 12, y + 12)
                spine_ys[net].append(y)
            left, right = bus_span(
                xs, plan.pins.get(net) == y, circuit.compact_periphery
            )
            box(top, "M4", left, y - 12, right, y + 12)
    for net, y in plan.pins.items():
        if not any(row.buses.get(net) == y for row in plan.rows):
            box(top, "M4", 0, y - 12, 48, y + 12)
        if label_pins:
            label(top, net, "M4", (24, y))
    for net, ys in spine_ys.items():
        x = plan.spine_x[net]
        box(top, "M5", x - 12, min(ys) - 24, x + 12, max(ys) + 24)
    box(top, "BOUNDARY", 0, -54, plan.width, plan.height)
    verify_periphery_routes(circuit, top, plan=plan)
    return top


def verify_periphery_routes(
    circuit: Circuit, top: Any, *, plan: PeripheralPlan | None = None
) -> None:
    """Require every terminal via and external pin to reach its intended net.

    This check does not allow substrate or well continuity to substitute for
    metal supply distribution. Device-internal geometry is checked by LVS.
    """
    plan = plan or periphery_plan(circuit)
    component = metal_components(top)
    nets, roots = {}, {}

    def check(net, metal, x, y):
        root = component(metal, x, y)
        if net in nets and nets[net] != root:
            raise ValueError(f"disconnected peripheral route: {net}")
        if root in roots and roots[root] != net:
            raise ValueError(f"shorted peripheral routes: {roots[root]} and {net}")
        nets[net], roots[root] = root, net

    for row in plan.rows:
        for i, device in enumerate(row.devices):
            for pin, net in zip(("D", "G", "S", "B"), device.nodes, strict=True):
                x, y = device.spec.pin_positions[pin]
                check(net, "V1", x + 108 + i * plan.device_pitch, y + row.y)
    for net, y in plan.pins.items():
        check(net, "M4", 24, y)
