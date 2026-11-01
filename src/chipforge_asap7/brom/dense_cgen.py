"""Dense 432 nm-high CGEN lane with sixteen six-transistor AND2 units.

Two candidate tracks use M2, eight M4 and six M6 over the lane. The M6
candidates descend outside the selector half, joining its unchanged M2 tracks.
This avoids running global candidates through the cells' local M2 wiring.

Every candidate trunk runs outward, from its cell to the selector half, so
each M4 and M6 track is free from the lane's inner end up to the cell it
serves. The parent runs the ``G<k>`` inputs inward on those free stretches
(`DenseLane.m4_free`, `m6_free`); the lifts' M4 jogs all sit on the first M4
track to keep the others clear.
"""

from dataclasses import dataclass, field

from ..devices.row import RowStack
from ..devices.row_support import RowSupportSpec, build_row_support
from ..devices.rowcell import box, label, square
from ..layout.layers import require_gdspy
from .circuit import Circuit
from .dense_bank import ceil_grid
from .dense_cell import build_dense_cell, cell_routes
from .selector import default_selector_plan


@dataclass
class DenseLane:
    cell: object
    width: int
    array_x: int
    outputs: dict
    inputs: dict
    routes: list
    certificates: list = field(default_factory=list)
    #: M4 track index -> largest x an inward trunk on it may reach.
    m4_free: dict = field(default_factory=dict)
    #: M6 track index -> (y, largest x an inward trunk on it may reach).
    m6_free: dict = field(default_factory=dict)
    #: x of every M5 column the lane draws over its row.
    columns: tuple = ()
    #: Slot order of the sixteen candidate cells, by k.
    order: tuple = ()
    #: x of every M4 line end the lane draws on its first track.
    tips: tuple = ()


def lane_order(fins):
    """Cell order along a lane: M2-direct candidates first, then M6, then M4."""
    direct = (0 if fins < 3 else 1, 6)
    upper = [k for k in range(8) if k not in direct]
    return tuple(list(direct) + upper + list(range(8, 16)))


def m4(top, x0, x1, y):
    box(top, "M4", min(x0, x1) - 24, y - 12, max(x0, x1) + 24, y + 12)


def v3(top, x, y):
    box(top, "V3", x - 9, y - 12, x + 9, y + 12)


def v4(top, x, y):
    box(top, "V4", x - 12, y - 12, x + 12, y + 12)


def lift6(top, cx, y4, sx, y6):
    v3(top, cx, y4)
    m4(top, cx, sx, y4)
    v4(top, sx, y4)
    box(top, "M5", sx - 12, min(y4 - 24, y6 - 27), sx + 12, max(y4 + 24, y6 + 27))
    box(top, "V5", sx - 12, y6 - 16, sx + 12, y6 + 16)


def lane_aliases(circuit, members):
    """Canonical candidate cell, including any parallel output fingers."""
    by_name = {d.name: d for d in circuit.devices}
    ds = [by_name[n] for n in members]
    names = {
        "VDD": "VDD",
        "VSS": "VSS",
        ds[0].nodes[1]: "G",
        ds[1].nodes[1]: "SEL",
        ds[4].nodes[0]: "C",
    }
    for d in ds:
        for n in d.nodes:
            if n not in names:
                names[n] = f"n{len(names)}"
    return names


def lane_devices(circuit, members):
    names = lane_aliases(circuit, members)
    ds = {d.name: d for d in circuit.devices}
    return tuple(
        (ds[n].spec.flavor, ds[n].spec.fins, *(names[v] for v in ds[n].nodes[:3]))
        for n in members
    )


def lane_cell(devices, vt, fins, flip):
    tracks = default_selector_plan(fins, vt).tracks
    direct = (0 if fins < 3 else 1, 6)
    select_y = 414 if flip else 18
    reserved = (
        select_y,
        *(432 - tracks[k][1] if flip else tracks[k][1] for k in direct),
    )
    unit, plan = build_dense_cell(
        devices,
        ("G", "SEL", "C"),
        vt=vt,
        reserved_m2=reserved,
        fixed_trunks=(("G", 243),),
    )
    unit.labels.clear()
    tx = dict(plan.trunks)["SEL"]
    cx = dict(plan.trunks)["C"]
    unit.remove_polygons(
        lambda p, l, d: (
            l == 30
            and d == 0
            and (min(p[:, 0]), max(p[:, 0])) in ((tx - 9, tx + 9), (cx - 9, cx + 9))
        )
    )
    for polygon in unit.polygons:
        polygon.polygons = list(polygon.polygons)
    ys = [ty for n, _, _, ty in plan.contacts if n == "SEL"]
    box(unit, "M3", tx - 9, min(select_y, *ys) - 14, tx + 9, max(select_y, *ys) + 14)
    box(unit, "M3", cx - 9, 16, cx + 9, 416)
    unit.name += "_lane_hi" if flip else "_lane_lo"
    return unit, plan


def build_dense_lane(
    fins=2, vt="rvt", *, block=0, columns=0, y_offset=0, cells=None, cell_width=None
):
    """Right-facing lane; mirror the complete half for the left side."""
    gdspy = require_gdspy()
    tracks = default_selector_plan(fins, vt).tracks
    direct = (0 if fins < 3 else 1, 6)
    upper = [k for k in range(8) if k not in direct]
    order = lane_order(fins)
    flip = bool(block % 2)
    c = Circuit("and2", (), fins, vt)
    c.and2("G", "SEL", "C")
    default = tuple((d.spec.flavor, d.spec.fins, *d.nodes[:3]) for d in c.devices)
    cells = cells or (default,) * 16
    cache = {d: lane_cell(d, vt, fins, flip) for d in set(cells)}
    unit_width = (
        max(plan.width for _, plan in cache.values())
        if cell_width is None
        else cell_width
    )
    if any(plan.width > unit_width for _, plan in cache.values()):
        raise ValueError("CGEN cell exceeds assigned lane pitch")
    origin = 432 * block
    lane_width = 16 * unit_width + 324
    array_x = lane_width + 108
    end = array_x + 108 * columns
    from hashlib import sha256

    digest = sha256(repr((cells, unit_width)).encode()).hexdigest()[:8]
    top = gdspy.Cell(
        f"brom_lane_{digest}_{vt}_{block}_{columns}_{y_offset % 64}",
        exclude_from_current=True,
    )
    outputs, inputs, routes = {}, {}, []
    certificates = []
    m4_free, m6_free, columns, tips = {}, {}, [], []
    stack = RowStack(((fins, fins),), vt=vt, band_height=216)
    filler = build_row_support(RowSupportSpec(stack, "filler"), draw_pin_labels=False)
    for slot, k in enumerate(order):
        x = slot * unit_width
        unit, plan = cache[cells[k]]
        certificates.append(
            {
                "unit": f"candidate_{k}",
                "candidate": k,
                "status": plan.placement_status,
                "gate_slot_lower_bound": plan.placement_bound,
                "width_nm": plan.width,
                "slot_width_nm": unit_width,
                "height_nm": 432,
                "devices": cells[k],
                "placement": plan.devices,
                "origin_nm": (x, origin + 432 if flip else origin),
                "mirror_y": flip,
            }
        )
        tx = dict(plan.trunks)["SEL"]
        top.add(
            gdspy.CellReference(
                unit, (x, origin + 432 if flip else origin), x_reflection=flip
            )
        )
        for pad in range(plan.width, unit_width, 108):
            top.add(
                gdspy.CellReference(
                    filler,
                    (x + pad, origin + 432 if flip else origin),
                    x_reflection=flip,
                )
            )

        def transform(p, x=x):
            return x + p[0], origin + 432 - p[1] if flip else origin + p[1]

        for n, metal, a, b in cell_routes(plan, ("G", "SEL", "C")):
            if metal == "M3" and n == "SEL":
                ys = [ty for net, _, _, ty in plan.contacts if net == "SEL"]
                select_y = 414 if flip else 18
                a = (a[0], min(select_y, *ys) - 14)
                b = (b[0], max(select_y, *ys) + 14)
            elif metal == "M3" and n == "C":
                a = (a[0], 16)
                b = (b[0], 416)
            routes.append((f"cell{k}:{n}", metal, transform(a), transform(b)))
        gx = x + 243
        inputs[f"G<{k}>"] = ("M3", gx, origin + 216)
        # Legal join of identical G rails across independently cut logic rows.
        box(top, "M3", gx - 9, origin - 14, gx + 9, origin + 446)
        routes.append((f"cell{k}:G", "M3", (gx, origin - 14), (gx, origin + 446)))
        sx = x + tx
        sy = origin + 18
        square(top, "V2", sx, sy)
        routes.append((f"cell{k}:SEL", "V2", (sx, sy), (sx, sy)))
        cx = x + dict(plan.trunks)["C"]
        _native, y = tracks[k]
        y += origin
        if k in direct:
            square(top, "V2", cx, y)
            box(top, "M2", cx - 19, y - 9, end + 24, y + 9)
            routes.extend(
                [
                    (f"C<{k}>", "M2", (cx, y), (end, y)),
                    (f"C<{k}>", "V2", (cx, y), (cx, y)),
                ]
            )
            outputs[f"C<{k}>"] = ("M2", end, y)
        elif k >= 8:
            v3(top, cx, y)
            m4(top, cx, end, y)
            routes.extend(
                [
                    (f"C<{k}>", "M4", (cx, y), (end, y)),
                    (f"C<{k}>", "V3", (cx, y), (cx, y)),
                ]
            )
            outputs[f"C<{k}>"] = ("M4", end, y)
            # An inward trunk ends a lattice step short of this candidate.
            m4_free[k - 8] = cx - 72
        else:
            j = upper.index(k)
            y6 = 16 + ceil_grid(origin + y_offset, 64) - y_offset + 64 * j
            y4 = origin + 60
            sx = ceil_grid(cx - 12, 48) + 12
            lift6(top, cx, y4, sx, y6)
            rx = ceil_grid(end + 120, 48) + 12 + 96 * j
            box(top, "M6", sx - 24, y6 - 16, rx + 24, y6 + 16)
            lift6(top, rx, y4, rx, y6)
            m6_free[j] = (y6, sx - 72)
            columns += [sx, rx]
            tips += [min(cx, sx) - 24, max(cx, sx) + 24, rx - 24, rx + 24]
            box(top, "M3", rx - 9, min(y, y4) - 17, rx + 9, max(y, y4) + 17)
            square(top, "V2", rx, y)
            box(top, "M2", end - 24, y - 9, rx + 19, y + 9)
            routes += [
                (f"C<{k}>", "M6", (sx, y6), (rx, y6)),
                (f"C<{k}>", "M2", (end, y), (rx, y)),
            ]
            for a, b, z in ((cx, sx, y4), (rx, rx, y4)):
                routes += [
                    (f"C<{k}>", "V3", (a, z), (a, z)),
                    (f"C<{k}>", "M4", (a, z), (b, z)),
                    (f"C<{k}>", "V4", (b, z), (b, z)),
                    (f"C<{k}>", "M5", (b, z), (b, y6)),
                    (f"C<{k}>", "V5", (b, y6), (b, y6)),
                ]
            routes += [
                (f"C<{k}>", "M3", (rx, y4), (rx, y)),
                (f"C<{k}>", "V2", (rx, y), (rx, y)),
            ]
            outputs[f"C<{k}>"] = ("M2", end, y)
    sy = origin + 18
    box(top, "M2", 0, sy - 9, 16 * unit_width, sy + 9)
    inputs["SEL"] = ("M2", 24, sy)
    routes.append(("SEL", "M2", (0, sy), (16 * unit_width, sy)))
    cache = {}
    stack = RowStack(((fins, fins),), vt=vt, band_height=216)
    for i, kind in enumerate(("filler", "tap", "filler")):
        if kind not in cache:
            cache[kind] = build_row_support(
                RowSupportSpec(stack, kind), draw_pin_labels=False
            )
        top.add(
            gdspy.CellReference(
                cache[kind],
                (16 * unit_width + 108 * i, origin + 432 if flip else origin),
                x_reflection=flip,
            )
        )
    inputs["VSS"] = ("M1", lane_width - 54, origin + 432 if flip else origin)
    inputs["VDD"] = ("M1", lane_width - 54, origin if flip else origin + 432)
    for net, (metal, x, y) in {**inputs, **outputs}.items():
        label(top, net, metal, (x, y))
    width = ceil_grid(end + 768, 108)
    box(top, "BOUNDARY", 0, origin - 27, width, origin + 459)
    return DenseLane(
        top,
        width,
        array_x,
        outputs,
        inputs,
        routes,
        certificates,
        m4_free,
        m6_free,
        tuple(columns),
        order,
        tuple(tips),
    )
