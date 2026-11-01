"""Abutting, alternating CMOS rows with M4-M6 routing over their cells."""

from dataclasses import dataclass, field

from ..devices.row import RowStack
from ..devices.row_support import RowSupportSpec, build_row_support
from ..devices.rowcell import box, draw_frame, draw_rails, label, square
from ..layout.layers import require_gdspy
from .dense_bank_router import BOTTOM, route_bank, route_header
from .dense_cell import cell_routes, wall_gap
from .dense_place import TAP_PERIOD, TAP_SEGMENT, anchored_rows, unit_gap
from .dense_units import dense_units

#: Placement spreads tried in turn for each row count; see `anchored_rows`.
SPREAD = (0.0, 0.5)
#: Row fills tried in turn by the electrical-order fallback at each width.
FILL = (1.0, 0.75, 0.5)


def ceil_grid(n, grid):
    return ((int(n) + grid - 1) // grid) * grid


@dataclass
class DenseBank:
    cell: object
    pins: dict
    width: int
    height: int
    units: list
    routes: list
    checkpoints: list
    certificates: list = field(default_factory=list)
    # Port tracks: net -> (y, left end, right end)
    header: dict = field(default_factory=dict)
    # Anchored ports: net -> x of the column the parent joins above the bank
    anchored: dict = field(default_factory=dict)


def _layout(units, width=None, pad=0):
    """Lay ``units`` out left to right with legal edge spacers and tap gaps.

    ``pad`` (a multiple of 108 nm, which keeps every spacer legal) is added
    between neighbours. With ``width``, stop before the first unit that
    would end past it and return (row, units left over).
    """
    row, end = [], 0
    for k, unit in enumerate(units):
        w = unit.plan.width
        if row:
            x = end + unit_gap(row[-1][0], unit)[0] + pad
        else:
            x = wall_gap(unit.plan.edges[0])
        # Filler/tap/filler slots keep every transistor within 30 um of a
        # body contact even if pin congestion widens a bank substantially.
        if x % TAP_PERIOD + w > TAP_SEGMENT:
            x = ceil_grid(x + 1, TAP_PERIOD)
        if width is not None and row and x + w > width:
            return row, units[k:]
        row.append((unit, x))
        end = x + w
    return row, []


def _pack(units, width, fill=1.0):
    """Fill rows to ``fill`` of ``width`` in order, then spread each across it."""
    packed, rest = [], list(units)
    while rest:
        row, rest = _layout(rest, ceil_grid(width * fill, 108))
        order = [u for u, _ in row]
        # Alternate rows run right to left, so electrical neighbours stay
        # close across a row end.
        if len(packed) % 2:
            order.reverse()
        row = _layout(order)[0]
        if fill < 1 and len(order) > 1:
            end = row[-1][1] + order[-1].plan.width
            pad = (width - end) // (len(order) - 1) // 108 * 108
            while pad > 0:
                spread = _layout(order, pad=pad)[0]
                if spread[-1][1] + order[-1].plan.width <= width:
                    row = spread
                    break
                pad -= 108
        packed.append(row)
    return packed


def _accesses(packed, row_width):
    points = []
    for index, row in enumerate(packed):
        y, flip = index * 432, bool(index % 2)
        for unit, x in row:
            mapping = dict(unit.nets)
            for port, tx in unit.plan.trunks:
                if port in (*unit.ports, "VDD", "VSS"):
                    points.append((mapping[port], index, x + tx, y + 216))
        for net, x, rail in (("VSS", row_width + 54, 0), ("VDD", row_width + 270, 432)):
            points.append((net, index, x, y + (432 - rail if flip else rail)))
    return tuple(points)


def _route_plan(
    points, pins, row_width, rows, width, spans, windows, bottom, blocked, external
):
    """`route_bank` with the port classes the parent's drops imply."""
    edges = set(pins if external is None else external) - set(bottom)
    return route_bank(
        points,
        tuple(pins),
        row_width,
        rows,
        width,
        anchors=tuple(sorted((n, *window) for n, window in windows.items())),
        bottom=tuple(sorted(bottom)),
        spans=tuple(sorted((n, tuple(xs)) for n, xs in spans.items() if xs)),
        edges=tuple(sorted(edges)),
        blocked=tuple(sorted(blocked)),
    )


def _anchored(
    units,
    pins,
    spans,
    external,
    width,
    extra_rows=6,
    cost=None,
    windows=None,
    bottom=(),
    blocked=(),
):
    """Place units near their parent drops; keep the cheapest routed bank.

    ``cost(header_top, height)`` ranks candidates; by default the bank
    height. A parent whose array origin rounds up to a grid passes its own.

    Rows stay within ``width`` less the end tap, so the bank keeps the
    parent's width and the drop positions remain valid. Each candidate row
    count is fully routed. If a placement crowds more escapes into a row
    than its M4 tracks hold, it is retried with its cells spread toward
    uniform density. Return None if no candidate fits.
    """
    usable = width - 324
    anchors = spans(width) or {}
    windows = (windows(width) if callable(windows) else windows) or {}
    total = sum(u.plan.width for u in units)
    first = max(1, -(-total // usable))
    best = None
    for rows in range(first, first + 1 + extra_rows):
        for spread in SPREAD:
            packed = anchored_rows(
                units, anchors, usable, rows, pins, spread=spread, sinks=bottom
            )
            if packed is None:
                continue
            packed = [row for row in packed if row]
            occupied = max(
                (x + u.plan.width for row in packed for u, x in row), default=0
            )
            row_width = ceil_grid(occupied, 108)
            plan = _route_plan(
                _accesses(packed, row_width),
                pins,
                row_width,
                len(packed),
                width,
                anchors,
                windows,
                bottom,
                blocked,
                external,
            )
            if plan is None or plan["width"] != width:
                continue
            header, height = route_header(plan)
            header_top = max((y for y, _, _ in header.values()), default=0)
            price = (cost or (lambda _top, h: h))(header_top, height), height
            if best is None or price < best[0]:
                best = (price, (packed, row_width, plan, header, height))
            break
    return best and best[1]


def build_dense_bank(
    circuit,
    *,
    width=5184,
    exclude=(),
    units=None,
    name=None,
    pin_width=0,
    header_spans=None,
    external=None,
    cost=None,
    anchors=None,
    bottom=(),
    blocked=(),
):
    """Place and route a bank; only ``external`` pins (default all) get edge ports.

    ``header_spans`` maps other pins to the x positions where a parent drops
    onto them. It may be a callable of the final bank width. ``cost`` ranks
    anchored candidates; see `_anchored`. ``anchors`` maps ports to the
    (x lo, x hi, kind) window a column must sit in for the parent to join it
    directly, with ``blocked`` the x positions such a column may not take;
    columns of different kinds never sit on adjacent lattice slots.
    ``bottom`` names the external pins placed on the lower edge.
    """
    gdspy = require_gdspy()
    units = dense_units(circuit, exclude=exclude) if units is None else units
    spans = header_spans if callable(header_spans) else lambda _: header_spans
    windows = anchors if callable(anchors) else lambda _: (anchors or {})
    bottom = set(bottom) & set(circuit.pins)
    result = None
    if header_spans is not None and pin_width:
        result = _anchored(
            units,
            circuit.pins,
            spans,
            external,
            pin_width,
            cost=cost,
            windows=windows,
            bottom=bottom,
            blocked=blocked,
        )
    if result is None:
        # Preserve the electrical placement order. When the escapes do not
        # fit in the eight M4 tracks of each row, first trade rows for
        # whitespace, then widen.
        while True:
            for fill in FILL:
                packed = _pack(units, width, fill)
                occupied = max(
                    (x + u.plan.width for row in packed for u, x in row), default=0
                )
                row_width = ceil_grid(occupied, 108)
                bank_width = ceil_grid(max(pin_width, row_width + 324), 432)
                points = _accesses(packed, row_width)
                plan = _route_plan(
                    points,
                    circuit.pins,
                    row_width,
                    len(packed),
                    pin_width,
                    spans(bank_width) or {},
                    windows(bank_width) or {},
                    bottom,
                    blocked,
                    external,
                )
                if plan is not None:
                    break
            if plan is not None:
                break
            if width > 16 * max(pin_width, sum(u.plan.width for u in units)):
                raise ValueError(f"dense bank {circuit.name} does not route")
            width = ceil_grid(max(width + 108, width * 1.1), 108)
        header, height = route_header(plan)
    else:
        packed, row_width, plan, header, height = result
    top = gdspy.Cell(name or circuit.name + "_dense", exclude_from_current=True)
    pins, placed, routes, checkpoints, certificates = {}, [], [], [], []
    cache = {}
    stack = RowStack(
        ((circuit.n.fins, circuit.p.fins),), vt=circuit.n.vt, band_height=216
    )
    support = {
        kind: build_row_support(RowSupportSpec(stack, kind), draw_pin_labels=False)
        for kind in ("filler", "tap")
    }
    # One empty gate pitch: frame and rails only.
    spacer = gdspy.Cell(f"{top.name}_spacer", exclude_from_current=True)
    draw_frame(spacer, stack, 54, circuit.n.vt)
    draw_rails(spacer, stack, 54)
    box(spacer, "BOUNDARY", 0, 0, 54, 432)
    for index, row in enumerate(packed):
        y, flip = index * 432, bool(index % 2)
        origin = y + 432 if flip else y
        transform = lambda v, y=y, flip=flip: y + (432 - v if flip else v)
        for unit, x in row:
            key = (unit.devices, unit.ports, unit.vt)
            if key not in cache:
                cache[key] = unit.build(abut_y=True)
                cache[key].labels.clear()
            top.add(gdspy.CellReference(cache[key], (x, origin), x_reflection=flip))
            record = {
                "unit": unit.name,
                "members": unit.members,
                "origin_nm": (x, origin),
                "mirror_y": flip,
                "width_nm": unit.plan.width,
                "height_nm": 432,
            }
            placed.append(record)
            certificates.append(
                {
                    **record,
                    "status": unit.plan.placement_status,
                    "gate_slot_lower_bound": unit.plan.placement_bound,
                    "devices": unit.devices,
                    "placement": unit.plan.devices,
                }
            )
            mapping = dict(unit.nets)
            routes += [
                (
                    mapping[n],
                    metal,
                    (a[0] + x, transform(a[1])),
                    (b[0] + x, transform(b[1])),
                )
                for n, metal, a, b in cell_routes(unit.plan, unit.ports, abut_y=True)
            ]
            for port, tx in unit.plan.trunks:
                if port in (*unit.ports, "VDD", "VSS"):
                    checkpoints.append((mapping[port], "M3", x + tx, y + 216))
        # Fill every gap: a 54 nm spacer beside a cell when the gap is an
        # odd number of gate pitches, then 108 nm fillers. A tap needs a
        # filler on either side to enclose the adjacent logic diffusion with
        # the correct implant polarity.
        cursor, gaps = 0, []
        for a, b in sorted((x, x + u.plan.width) for u, x in row):
            if a > cursor:
                gaps.append((cursor, a))
            cursor = max(cursor, b)
        gaps.append((cursor, row_width + 108))
        for a, b in gaps:
            if (b - a) % 108:
                top.add(gdspy.CellReference(spacer, (a, origin), x_reflection=flip))
                a += 54
            taps = set(range(a + 108, b - 215, TAP_PERIOD)) if b - a >= 324 else ()
            for x in range(a, b, 108):
                top.add(
                    gdspy.CellReference(
                        support["tap" if x in taps else "filler"],
                        (x, origin),
                        x_reflection=flip,
                    )
                )
        for kind, x in (("tap", row_width + 108), ("filler", row_width + 216)):
            top.add(gdspy.CellReference(support[kind], (x, origin), x_reflection=flip))
        for net, x, rail in (
            ("VSS", row_width + 54, transform(0)),
            ("VDD", row_width + 270, transform(432)),
        ):
            square(top, "V1", x, rail)
            square(top, "V2", x, rail)
            box(top, "M2", x - 19, rail - 9, x + 19, rail + 9)
            checkpoints.append((net, "M1", x, rail))
    for i, by, accesses, lo, hi in plan["escapes"]:
        net = plan["groups"][i][0]
        sx = plan["columns"][i]
        for x, sy in accesses:
            box(top, "M3", x - 9, min(sy, by) - 17, x + 9, max(sy, by) + 17)
            box(top, "V3", x - 9, by - 12, x + 9, by + 12)
            routes += [(net, "M3", (x, sy), (x, by)), (net, "V3", (x, by), (x, by))]
        box(top, "M4", lo, by - 12, hi, by + 12)
        routes.append((net, "M4", (lo, by), (hi, by)))
        if sx is not None:
            box(top, "V4", sx - 12, by - 12, sx + 12, by + 12)
            routes.append((net, "V4", (sx, by), (sx, by)))
    for net, by, lo, hi, columns in plan["jogs"]:
        box(top, "M4", lo, by - 12, hi, by + 12)
        routes.append((net, "M4", (lo, by), (hi, by)))
        for x in columns:
            box(top, "V4", x - 12, by - 12, x + 12, by + 12)
            routes.append((net, "V4", (x, by), (x, by)))
    # Columns span only their rows; port spines end together above the
    # header, bottom pins at the lower boundary. These real wire lengths are
    # included in the physical RC record.
    row_top = len(packed) * 432
    bus_top = {
        (net, x): (y + 38) // 48 * 48 + 36
        for net, y, _, _, columns in plan["buses"]
        for x in columns
    }
    for net, x, low, high in plan["column_shapes"]:
        if high > height + 24:
            high = max(
                row_top + 36
                if net in plan["anchored"] and x == plan["anchored"][net]
                else 0,
                header[net][0] + 24 if net in header else 0,
                bus_top.get((net, x), 0),
            ) or (height + 24)
        low, high = max(low, BOTTOM), min(high, height + 24)
        box(top, "M5", x - 12, low, x + 12, high)
        routes.append((net, "M5", (x, low), (x, high)))
    for net, y, lo, hi, columns in plan["buses"]:
        box(top, "M6", lo, y - 16, hi, y + 16)
        routes.append((net, "M6", (lo, y), (hi, y)))
        for x in columns:
            box(top, "V5", x - 12, y - 16, x + 12, y + 16)
            routes.append((net, "V5", (x, y), (x, y)))
    edge = circuit.pins if external is None else external
    for net, (by, lo, hi) in header.items():
        columns = plan["port_columns"].get(net) or plan["m4_buses"][net][2]
        for x in columns:
            box(top, "V4", x - 12, by - 12, x + 12, by + 12)
            routes.append((net, "V4", (x, by), (x, by)))
        box(top, "M4", lo, by - 12, hi, by + 12)
        routes.append((net, "M4", (lo, by), (hi, by)))
        if net in bottom or net not in circuit.pins:
            continue
        # An external pin sits at the bank edge its track reaches.
        if net not in edge:
            x = lo + 24
        else:
            x = 24 if lo < 0 else plan["width"] - 24
        pins[net] = ("M4", x, by)
        label(top, net, "M4", pins[net][1:])
    # A bottom pin is its column, which reaches the lower boundary.
    for net, x in plan["bottom"].items():
        pins[net] = ("M5", x, BOTTOM // 2)
        label(top, net, "M5", pins[net][1:])
    box(top, "BOUNDARY", 0, BOTTOM, plan["width"], height)
    checkpoints += [(n, *point) for n, point in pins.items()]
    checkpoints += [(n, "M5", x, row_top) for n, x in plan["anchored"].items()]
    return DenseBank(
        top,
        pins,
        plan["width"],
        height,
        placed,
        routes,
        checkpoints,
        certificates,
        header,
        dict(plan["anchored"]),
    )
