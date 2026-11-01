"""Central-CGEN/bottom-decoder assembly using dense CMOS cells."""

from collections import defaultdict
from dataclasses import replace

from ..devices.rowcell import box, label, square
from ..layout.layers import require_gdspy
from .circuit import Circuit
from .dense_bank import build_dense_bank, ceil_grid
from .dense_cgen import (
    build_dense_lane,
    lane_aliases,
    lane_cell,
    lane_devices,
    m4,
    v3,
    v4,
)
from .selector import build_selector_array


def lane_trunks(lanes):
    """Give each ``G<k>`` an inward lane track shared with the outward C trunks.

    ``lanes`` holds the block-0 lane of each side; both have the same cell
    order and pitch. A trunk on M4 track ``i`` may run from the centre up to
    the cell it serves as long as it ends a lattice step before candidate
    ``8 + i``'s own trunk; the first M4 track carries the lifts' jogs and is
    kept clear. An M6 track likewise serves cells inside the lift that
    starts it, reached from the cell's G rail by a jog on the first M4 track
    and an M5 stub. Outer cells are tried first, each taking the tightest
    track that reaches it. Return {k: (layer, track, x)} in lane coordinates,
    where ``x`` is the G rail for M4 and the stub column for M6; a k with no
    track keeps its header track.
    """
    order = next(iter(lanes.values())).order
    m4 = {i: min(lane.m4_free[i] for lane in lanes.values()) for i in range(1, 8)}
    m4[0] = min(lane.inputs[f"G<{order[1]}>"][1] - 72 for lane in lanes.values())
    m6 = {j: min(lane.m6_free[j][1] for lane in lanes.values()) for j in range(6)}
    # Where the outward trunks start on each track: the candidate's M4 a
    # lattice step past the free limit, the lift's M6 at its column.
    tips4 = {i: {lane.m4_free[i] + 48 for lane in lanes.values()} for i in range(8)}
    tips4[0].update(t for lane in lanes.values() for t in lane.tips)
    tips6 = {j: {lane.m6_free[j][1] + 48 for lane in lanes.values()} for j in range(6)}

    def clear(tips, track, end):
        """No tip on the next track within a corner spacing of ``end``."""
        return all(
            abs(end - tip) >= 40
            for near in (track - 1, track + 1)
            for tip in tips.get(near, ())
        )

    plan = {}
    for slot in reversed(range(16)):
        k = order[slot]
        rails = {lane.inputs[f"G<{k}>"][1] for lane in lanes.values()}
        if len(rails) != 1:
            continue
        gx = rails.pop()
        end = ceil_grid(gx + 24, 48)
        fits = sorted(
            (limit, i)
            for i, limit in m4.items()
            if end <= limit and clear(tips4, i, end)
        )
        if fits:
            _, i = fits[0]
            del m4[i]
            tips4[i].add(end)
            plan[k] = ("M4", i, gx)
            continue
        sx = ceil_grid(gx - 12, 48) + 12
        fits = sorted(
            (limit, j)
            for j, limit in m6.items()
            if sx + 24 <= limit and clear(tips6, j, sx + 24)
        )
        if fits:
            _, j = fits[0]
            del m6[j]
            tips6[j].add(sx + 24)
            tips4[0].update((min(gx, sx) - 24, max(gx, sx) + 24))
            plan[k] = ("M6", j, sx)
    return plan


def prepare_dense_macro(c, spec):
    """Duplicate half-lane drivers in the schematic before sizing or layout."""
    if getattr(c, "dense_spec", None) is not None:
        return c
    if c.distribution is not None:
        raise ValueError("dense floorplan must be selected before control sizing")
    c.dense_spec = spec
    left = (spec.bits // 2) * spec.ways
    c.dense_left_columns = left
    if left:
        for b in range(spec.blocks):
            for k in range(16):
                start = len(c.devices)
                c.and2(f"G<{k}>", f"SEL<{b}>", f"LC<{b},{k}>")
                c.group_since(f"left_cgen_{b}_{k}", start)

        def retarget(d):
            if d.name in c.array.device_names and d.nodes[0].startswith("BL<"):
                col = int(d.nodes[0][3:-1])
                if col < left and d.nodes[1].startswith("C<"):
                    return replace(
                        d, nodes=(d.nodes[0], "L" + d.nodes[1], *d.nodes[2:])
                    )
            return d

        c.devices = [retarget(d) for d in c.devices]
    c.name += "_dense"
    return c


def _strip(c, excluded):
    remaining = [d for d in c.devices if d.name not in excluded]
    used = {n for d in remaining for n in d.nodes}
    boundary = {n for d in c.devices if d.name in excluded for n in d.nodes} & used
    bank = Circuit(
        c.name + "_bottom",
        tuple(dict.fromkeys([*c.pins, *sorted(boundary)])),
        c.n.fins,
        c.n.vt,
    )
    bank.devices = remaining
    bank.logic_units = c.logic_units
    bank.placement_groups = c.placement_groups
    return bank


def build_dense_macro(c, spec=None):
    if spec is not None:
        prepare_dense_macro(c, spec)
    spec = c.dense_spec
    gdspy = require_gdspy()
    left = c.dense_left_columns
    right = spec.columns - left
    candidate_cells = {
        role: lane_devices(c, members)
        for role, members in c.placement_groups
        if role.startswith(("cgen_", "left_cgen_"))
    }
    candidate_nets = {
        role: {v: k for k, v in lane_aliases(c, members).items()}
        for role, members in c.placement_groups
        if role in candidate_cells
    }
    variants = set(candidate_cells.values())
    pitch = max(
        lane_cell(ds, spec.vt, spec.fins, flip)[1].width
        for ds in variants
        for flip in range(min(spec.blocks, 2))
    )
    first_cells = tuple(candidate_cells[f"cgen_0_{k}"] for k in range(16))
    lane0 = build_dense_lane(
        spec.fins,
        spec.vt,
        columns=max(left, right),
        cells=first_cells,
        cell_width=pitch,
    )
    half_width = ceil_grid(lane0.width + 432, 432)
    middle = ceil_grid((spec.blocks + 4) * 48, 432)
    center = half_width + 216 if left else 0
    right_origin = ceil_grid(center + middle, 432)
    footprint = right_origin + half_width
    excluded = set(c.array.device_names)
    if c.replica:
        excluded.update(c.replica.selector.device_names)
    excluded.update(
        n
        for role, ns in c.placement_groups
        if role.startswith(("cgen_", "left_cgen_"))
        for n in ns
    )
    bank_c = _strip(c, excluded)
    sides = [
        (side, lo, hi, origin, mirror)
        for side, lo, hi, origin, mirror in (
            ("L", 0, left, center, True),
            ("R", left, spec.columns, right_origin, False),
        )
        if lo != hi
    ]

    replica_origin = ceil_grid(right_origin + lane0.width + 540, 432)
    bank_footprint = (
        ceil_grid(replica_origin + 108 * spec.columns + 324, 432)
        if c.replica
        else footprint
    )

    def replica_x(bank_width):
        if bank_width <= bank_footprint:
            return replica_origin
        return ceil_grid(max(footprint, bank_width) + 432, 432)

    def side_transform(origin, mirror):
        return lambda x: origin - x if mirror else origin + x

    # The block-0 lane of each side fixes where its G rails and lifts lie.
    lanes0 = {
        side: build_dense_lane(
            spec.fins,
            spec.vt,
            columns=hi - lo,
            cells=tuple(
                candidate_cells[f"{'left_' if mirror else ''}cgen_0_{k}"]
                for k in range(16)
            ),
            cell_width=pitch,
        )
        for side, lo, hi, _origin, mirror in sides
    }
    trunks = lane_trunks(lanes0)

    def header_spans(bank_width):
        """Every x where join() below drops onto a bank port track."""
        spans = defaultdict(list)
        ax = lane0.array_x
        for side, lo, hi, origin, mirror in sides:
            transform = side_transform(origin, mirror)
            lane = lanes0[side]
            for k in range(16):
                spans[f"G<{k}>"].append(transform(lane.inputs[f"G<{k}>"][1]))
            if not mirror:
                for block in range(spec.blocks):
                    spans[f"SEL<{block}>"].append(
                        ceil_grid(center, 48) + 12 + 48 * block
                    )
            for net, dx in (("VDD", 96), ("VSS", 240)):
                out = transform(lane.width + dx)
                spans[net].append(ceil_grid(out - 12, 48) + 12)
            for col in range(hi - lo):
                spans[f"BL<{lo + col}>"].append(transform(ax + 108 * col + 81))
            spans["VSS"].append(transform(lane0.width + 240))
        if c.replica:
            ox = replica_x(bank_width)
            for b, row in enumerate(c.replica.selector.mapping):
                for k in set(row):
                    net = c.replica.port_nets[f"C<{b},{k}>"]
                    spans[net].append(
                        ox - 132 if net != "VSS" else ox + 108 * spec.columns + 228
                    )
            for col in range(spec.columns):
                spans[f"RBL<{col}>"].append(ox + 108 * col + 81)
            spans["VSS"].append(ox + 108 * spec.columns + 228)
        return spans

    def array_base(header_top, height):
        """Array origin over a bank: the join stubs below reach down to
        base - 132 on M4; keep them two track pitches above the highest header
        track, and the arrays above the bank. Fins and M4 tracks lie on
        absolute 27 and 48 nm grids, so the arrays start on a multiple of
        432."""
        return ceil_grid(max(header_top + 228, height), 432)

    # Anchored ports: the bank puts the port's column where the join stub
    # below the array can reach it, so the port needs no header track. A
    # bitline column sits on the lattice slot nearest its drop, which keeps
    # the stubs of same-level neighbours 72 nm apart; a select's column is
    # its own M3 riser slot, with the stub's V3 and V4 stacked.
    anchors = {}
    for _side, lo, hi, origin, mirror in sides:
        for col in range(hi - lo):
            dx = lane0.array_x + 108 * col + 81
            x = origin - dx if mirror else origin + dx
            anchors[f"BL<{lo + col}>"] = (x - 24, x + 24, None)
    if c.replica:
        for col in range(spec.columns):
            x = replica_origin + 108 * col + 81
            anchors[f"RBL<{col}>"] = (x - 24, x + 24, None)
    for block in range(spec.blocks):
        sx = ceil_grid(center, 48) + 12 + 48 * block
        anchors[f"SEL<{block}>"] = (sx, sx, None)
    # A G trunk on a lane track must contain its riser: between the two
    # lanes' rails or stubs, or anywhere inside the strip when one lane
    # stands alone. The riser keeps a lattice step from the lanes' own M5.
    blocked = set()
    for side, _lo, _hi, origin, mirror in sides:
        transform = side_transform(origin, mirror)
        stubs = [x for layer, _, x in trunks.values() if layer == "M6"]
        for x in (*lanes0[side].columns, *stubs):
            blocked.update(transform(x) + d for d in (-48, 0, 48))
    # Risers to M4 and M6 trunks end at heights that may differ by less than
    # a tip stagger, so the two kinds never take adjacent lattice slots.
    for k, (layer, _track, x) in trunks.items():
        ends = [side_transform(origin, mirror)(x) for _, _, _, origin, mirror in sides]
        lo = (min(ends) - 24) // 48 * 48 + 24 if len(ends) > 1 else 12
        anchors[f"G<{k}>"] = (lo, ceil_grid(max(ends) + 24, 48) - 24, layer)

    bank = build_dense_bank(
        bank_c,
        width=bank_footprint,
        pin_width=bank_footprint,
        header_spans=header_spans,
        external=c.pins,
        cost=array_base,
        anchors=anchors,
        bottom=c.pins,
        blocked=blocked,
    )
    top = gdspy.Cell(c.name, exclude_from_current=True)
    bank.cell.labels.clear()
    top.add(gdspy.CellReference(bank.cell))
    base = array_base(
        max((y for y, _, _ in bank.header.values()), default=0), bank.height
    )
    routes = list(bank.routes)
    checkpoints = list(bank.checkpoints)
    certificates = list(bank.certificates)
    ports = {}
    arrays = []

    # M4 line ends on the lanes' first track, which a trunk on the second
    # track must keep a corner spacing from.
    tips0 = []
    for side, _lo, _hi, origin, mirror in sides:
        tips0 += [side_transform(origin, mirror)(x) for x in lanes0[side].tips]

    row_top = max(
        (
            u["origin_nm"][1] if u["mirror_y"] else u["origin_nm"][1] + 432
            for u in bank.units
        ),
        default=432,
    )

    def rise(net, x, y):
        """Join the M3 drop at ``x`` to the port's anchored column on an M4 stub."""
        cx = bank.anchored[net]
        v3(top, x, y)
        lo = (min(x, cx) - 24) // 48 * 48
        hi = ceil_grid(max(x, cx) + 24, 48)
        box(top, "M4", lo, y - 12, hi, y + 12)
        box(top, "V4", cx - 12, y - 12, cx + 12, y + 12)
        box(top, "M5", cx - 12, row_top - 24, cx + 12, y + 24)
        routes.extend(
            [
                (net, "V3", (x, y), (x, y)),
                (net, "M4", (x, y), (cx, y)),
                (net, "V4", (cx, y), (cx, y)),
                (net, "M5", (cx, row_top - 24), (cx, y)),
            ]
        )

    def join(net, x, y, *, source="M3", local_x=None):
        """Connect an M4 bank header through the clear M3 space above the cells."""
        if net in bank.anchored:
            rise(net, x, y)
            return
        bx = x if local_x is None else local_x
        by, left_end, right_end = bank.header[net]
        if not (left_end <= bx - 24 and bx + 24 <= right_end):
            raise ValueError(f"bank port track {net} does not reach x={bx}")
        if bx == x and source == "M3":
            v3(top, bx, by)
            box(top, "M3", bx - 9, min(y, by) - 17, bx + 9, max(y, by) + 17)
            routes.extend(
                [
                    (net, "M3", (bx, by), (bx, y)),
                    (net, "V3", (bx, by), (bx, by)),
                ]
            )
            return
        if source == "M3":
            v3(top, x, y)
        elif source == "M2":
            square(top, "V2", x, y)
        lo = (min(x, bx) - 24) // 48 * 48
        hi = ceil_grid(max(x, bx) + 24, 48)
        box(top, "M4", lo, y - 12, hi, y + 12)
        v3(top, bx, y)
        v3(top, bx, by)
        box(top, "M3", bx - 9, min(y, by) - 17, bx + 9, max(y, by) + 17)
        routes.extend([(net, "M4", (x, y), (bx, y)), (net, "M3", (bx, by), (bx, y))])
        routes.extend(
            [
                (net, "V3", (bx, by), (bx, by)),
                (net, "V3", (bx, y), (bx, y)),
            ]
        )
        if source == "M3":
            routes.append((net, "V3", (x, y), (x, y)))

    for side, lo, hi, origin, mirror in sides:
        columns = hi - lo
        mapping = tuple(tuple(row[lo:hi]) for row in c.array.mapping)
        array = build_selector_array(c.array.plan, mapping)
        array.labels.clear()
        array.name += "_dense_" + side
        ax = lane0.array_x
        transform = lambda x, origin=origin, mirror=mirror: (
            origin - x if mirror else origin + x
        )
        top.add(
            gdspy.CellReference(
                array,
                (transform(ax), base),
                rotation=180 if mirror else None,
                x_reflection=mirror,
            )
        )
        arrays.append(
            {
                "side": side,
                "columns": (lo, hi),
                "origin_nm": (transform(ax), base),
                "mirror_x": mirror,
            }
        )
        routes += selector_routes(
            c.array.plan,
            mapping,
            transform,
            ax,
            base,
            lambda b, k, mirror=mirror: f"{'LC' if mirror else 'C'}<{b},{k}>",
            lambda col, lo=lo: f"BL<{lo + col}>",
        )
        for block in range(spec.blocks):
            cells = tuple(
                candidate_cells[f"{'left_' if mirror else ''}cgen_{block}_{k}"]
                for k in range(16)
            )
            lane = build_dense_lane(
                spec.fins,
                spec.vt,
                block=block,
                columns=columns,
                y_offset=base,
                cells=cells,
                cell_width=pitch,
            )
            lane.cell.labels.clear()
            top.add(
                gdspy.CellReference(
                    lane.cell,
                    (origin, base),
                    rotation=180 if mirror else None,
                    x_reflection=mirror,
                )
            )
            for cert in lane.certificates:
                role = f"{'left_' if mirror else ''}cgen_{block}_{cert['candidate']}"
                certificates.append(
                    {
                        **cert,
                        "unit": role,
                        "members": dict(c.placement_groups)[role],
                        "origin_nm": (
                            transform(cert["origin_nm"][0]),
                            base + cert["origin_nm"][1],
                        ),
                        "mirror_x": mirror,
                    }
                )
            for net, layer, a, b in lane.routes:
                if net.startswith("cell"):
                    slot, canonical = net.split(":")
                    k = int(slot[4:])
                    name = candidate_nets[
                        f"{'left_' if mirror else ''}cgen_{block}_{k}"
                    ][canonical]
                elif net == "SEL":
                    name = f"SEL<{block}>"
                else:
                    k = int(net[2:-1])
                    name = f"{'LC' if mirror else 'C'}<{block},{k}>"
                routes.append(
                    (
                        name,
                        layer,
                        (transform(a[0]), base + a[1]),
                        (transform(b[0]), base + b[1]),
                    )
                )
                if (
                    net.startswith("cell")
                    and layer == "M3"
                    and net.split(":")[1] in ("G", "SEL", "C")
                ):
                    checkpoints.append(
                        (name, layer, transform(a[0]), base + (a[1] + b[1]) / 2)
                    )
            for net, (metal, x, y) in lane.outputs.items():
                k = int(net[2:-1])
                checkpoints.append(
                    (
                        f"{'LC' if mirror else 'C'}<{block},{k}>",
                        metal,
                        transform(x),
                        base + y,
                    )
                )
            for net, (metal, x, y) in lane.inputs.items():
                checkpoints.append(
                    (
                        f"SEL<{block}>" if net == "SEL" else net,
                        metal,
                        transform(x),
                        base + y,
                    )
                )
            sy = base + block * 432 + 18
            sx = ceil_grid(center, 48) + 12 + 48 * block
            # Shared SEL enters both halves directly on the local M2 band.
            edge = origin
            box(top, "M2", min(sx, edge) - 19, sy - 9, max(sx, edge) + 19, sy + 9)
            square(top, "V2", sx, sy)
            py = (
                base + block * 432 + 60
                if block and f"SEL<{block}>" in bank.anchored
                else base - 36 - 96 * (block % 3)
            )
            box(top, "M3", sx - 9, min(py, sy) - 17, sx + 9, max(py, sy) + 17)
            routes += [
                (f"SEL<{block}>", "M2", (sx, sy), (edge, sy)),
                (f"SEL<{block}>", "V2", (sx, sy), (sx, sy)),
                (f"SEL<{block}>", "M3", (sx, py), (sx, sy)),
            ]
            if side == "R":
                join(f"SEL<{block}>", sx, py, local_x=sx)
            if block == 0:
                for k in range(16):
                    net = f"G<{k}>"
                    _, gx, _ = lane.inputs[net]
                    gx = transform(gx)
                    if k in trunks and net in bank.anchored:
                        # The rail meets its inward lane trunk inside the row.
                        layer, track, x = trunks[k]
                        if layer == "M4":
                            y = base + 60 + 48 * track
                            v3(top, gx, y)
                            routes.append((net, "V3", (gx, y), (gx, y)))
                            continue
                        y4 = base + 60
                        y6 = 16 + ceil_grid(base, 64) + 64 * track
                        sx = transform(x)
                        v3(top, gx, y4)
                        m4(top, gx, sx, y4)
                        tips0 += [min(gx, sx) - 24, max(gx, sx) + 24]
                        v4(top, sx, y4)
                        box(
                            top,
                            "M5",
                            sx - 12,
                            min(y4 - 24, y6 - 27),
                            sx + 12,
                            max(y4 + 24, y6 + 27),
                        )
                        box(top, "V5", sx - 12, y6 - 16, sx + 12, y6 + 16)
                        routes += [
                            (net, "V3", (gx, y4), (gx, y4)),
                            (net, "M4", (gx, y4), (sx, y4)),
                            (net, "V4", (sx, y4), (sx, y4)),
                            (net, "M5", (sx, y4), (sx, y6)),
                            (net, "V5", (sx, y6), (sx, y6)),
                        ]
                        continue
                    py = base - 36
                    box(top, "M3", gx - 9, py - 17, gx + 9, base + 14)
                    routes.append((net, "M3", (gx, py), (gx, base + 14)))
                    join(net, gx, py)
            for net in ("VDD", "VSS"):
                _, px, ry = lane.inputs[net]
                # Cross the selector on the free M4 boundary track. Three-fin
                # selectors have an M2 candidate only 29 nm above the rail.
                px = transform(px)
                ry += base
                out = transform(lane.width + (96 if net == "VDD" else 240))
                square(top, "V1", px, ry)
                box(top, "M2", px - 19, ry - 9, px + 19, ry + 9)
                square(top, "V2", px, ry)
                py = ry + 12
                box(top, "M3", px - 9, ry - 17, px + 9, py + 17)
                join(net, px, py, local_x=ceil_grid(out - 12, 48) + 12)
        for col in range(columns):
            x = transform(ax + 108 * col + 81)
            y = base - 36 - 96 * (col % 2)
            box(top, "M3", x - 9, y - 17, x + 9, base + 14)
            routes.append(
                (f"BL<{lo + col}>", "M3", (x, y), (x, base + spec.blocks * 432))
            )
            join(f"BL<{lo + col}>", x, y)
            checkpoints.append((f"BL<{lo + col}>", "M3", x, base + 216))
        # The selector's own substrate rail is above its last block.
        x = transform(ax)
        sy = base + spec.blocks * 432 + 81
        target = transform(lane0.width + 240)
        box(top, "M2", min(x, target) - 19, sy - 9, max(x, target) + 19, sy + 9)
        square(top, "V2", target, sy)
        py = base + spec.blocks * 432 + 108
        box(top, "M3", target - 9, min(py, sy) - 17, target + 9, max(py, sy) + 17)
        join("VSS", target, py)
        checkpoints.append(("VSS", "M2", x, sy))
    # Inward G trunks on the block-0 lane tracks, from one lane's rail or
    # stub to the other's, with the bank's riser joining in between. With
    # one lane the trunk ends at its riser instead; that end moves inward
    # until no trunk on the next track ends on the same lattice line.
    drawn = defaultdict(list)
    for k, (layer, track, x) in sorted(trunks.items(), key=lambda t: t[1][1]):
        net = f"G<{k}>"
        if net not in bank.anchored:
            continue
        cx = bank.anchored[net]
        ends = [cx, *(side_transform(o, m)(x) for _, _, _, o, m in sides)]
        if layer == "M4":
            y = base + 60 + 48 * track
            lo = (min(ends) - 24) // 48 * 48
            hi = ceil_grid(max(ends) + 24, 48)
            near = [
                t for k in (track - 1, track + 1) for tips in drawn[k] for t in tips
            ]
            if track == 1:
                near += tips0
            # Only a riser end may move; a rail end is where the cell is.
            while len(ends) == 2 and lo > 0 and any(abs(lo - t) < 40 for t in near):
                lo -= 48
            drawn[track].append({lo, hi})
            box(top, "M4", lo, y - 12, hi, y + 12)
            v4(top, cx, y)
            box(top, "M5", cx - 12, row_top - 24, cx + 12, y + 24)
            routes += [
                (net, "M4", (lo, y), (hi, y)),
                (net, "V4", (cx, y), (cx, y)),
                (net, "M5", (cx, row_top - 24), (cx, y)),
            ]
        else:
            y6 = 16 + ceil_grid(base, 64) + 64 * track
            lo, hi = min(ends) - 24, max(ends) + 24
            box(top, "M6", lo, y6 - 16, hi, y6 + 16)
            box(top, "V5", cx - 12, y6 - 16, cx + 12, y6 + 16)
            box(top, "M5", cx - 12, row_top - 24, cx + 12, y6 + 27)
            routes += [
                (net, "M6", (lo, y6), (hi, y6)),
                (net, "V5", (cx, y6), (cx, y6)),
                (net, "M5", (cx, row_top - 24), (cx, y6)),
            ]
    # Independent replica columns occupy the far edge. Only selected native
    # candidate tracks need a connection: unselected tracks contain no gates.
    if c.replica:
        inst = c.replica.selector
        array = build_selector_array(inst.plan, inst.mapping)
        array.labels.clear()
        array.name += "_dense_replica"
        ox = replica_x(bank.width)
        oy = base
        top.add(gdspy.CellReference(array, (ox, oy)))
        routes += selector_routes(
            inst.plan,
            inst.mapping,
            lambda x: x,
            ox,
            oy,
            lambda b, k: c.replica.port_nets[f"C<{b},{k}>"],
            lambda col: f"RBL<{col}>",
        )
        for b, row in enumerate(inst.mapping):
            for k in sorted(set(row)):
                layer, dy = inst.plan.tracks[k]
                net = c.replica.port_nets[f"C<{b},{k}>"]
                y = oy + b * 432 + dy
                checkpoints.append((net, layer, ox, y))
                x = ox - 132 if net != "VSS" else ox + 108 * spec.columns + 228
                if layer == "M2":
                    box(
                        top,
                        "M2",
                        min(ox - 24, x - 19),
                        y - 9,
                        max(ox + 108 * spec.columns + 24, x + 19),
                        y + 9,
                    )
                    square(top, "V2", x, y)
                    py = oy + b * 432 + 60
                    box(top, "M3", x - 9, min(y, py) - 17, x + 9, max(y, py) + 17)
                    routes += [
                        (net, "M2", (ox - 24, y), (x, y)),
                        (net, "V2", (x, y), (x, y)),
                        (net, "M3", (x, y), (x, py)),
                    ]
                    join(net, x, py)
                else:
                    m4(top, x, ox, y)
                    routes.append((net, "M4", (x, y), (ox, y)))
                    join(net, x, y, source="M4", local_x=x)
        for col in range(spec.columns):
            x = ox + 108 * col + 81
            py = oy - 36 - 96 * (col % 2)
            box(top, "M3", x - 9, py - 17, x + 9, oy + 14)
            routes.append((f"RBL<{col}>", "M3", (x, py), (x, oy + 14)))
            join(f"RBL<{col}>", x, py)
            checkpoints.append((f"RBL<{col}>", "M3", x, oy + 216))
        sy = oy + len(inst.mapping) * 432 + 81
        x = ox + 108 * spec.columns + 228
        box(top, "M2", ox - 24, sy - 9, x + 19, sy + 9)
        square(top, "V2", x, sy)
        py = sy + 27
        box(top, "M3", x - 9, sy - 17, x + 9, py + 17)
        join("VSS", x, py)
        checkpoints.append(("VSS", "M2", ox, sy))
    # Check programmed gate stems as well as bitline and candidate sources.
    # A missing programming via can leave the horizontal track intact.
    for net, metal, a, b in routes:
        if metal == "M3" and net.startswith(("C<", "LC<", "RC<")):
            checkpoints.append((net, metal, a[0], (a[1] + b[1]) / 2))
    for net in c.pins:
        metal, x, y = bank.pins[net]
        label(top, net, metal, (x, y))
        ports[net] = (metal, x, y)
    bounds = top.get_bounding_box()
    box(top, "BOUNDARY", *bounds[0], *bounds[1])
    return top, {
        "certificates": certificates,
        "ports": ports,
        "arrays": arrays,
        "bottom_units": bank.units,
        "routes": routes,
        "checkpoints": checkpoints,
        "array_y_nm": base,
        "center_nm": (center, right_origin),
        "timing_qualified": False,
    }


def canonicalize_cells(top):
    """Intern equal named cells so repeated rows remain hierarchical in GDS."""
    cells = {}
    digests = {}

    def digest(cell):
        from hashlib import sha256

        if id(cell) not in digests:
            h = sha256()
            for layer, polygons in sorted(cell.get_polygons(by_spec=True).items()):
                h.update(repr(layer).encode())
                for polygon in polygons:
                    h.update(polygon.tobytes())
            digests[id(cell)] = h.digest()
        return digests[id(cell)]

    def visit(cell):
        for ref in cell.references:
            child = ref.ref_cell
            if child.name in cells:
                if digest(child) != digest(cells[child.name]):
                    raise ValueError(f"conflicting dense cell geometry: {child.name}")
                ref.ref_cell = cells[child.name]
            else:
                cells[child.name] = child
                visit(child)

    visit(top)


def dense_macro_layout(c):
    """One physical result shared by GDS, LEF, RC and report provenance."""
    from copy import deepcopy
    from hashlib import sha256

    if c.array is None or getattr(c, "dense_spec", None) is None:
        raise ValueError("dense layout requires a macro_circuit")
    key = sha256(
        repr(
            (
                c.devices,
                c.pins,
                c.placement_groups,
                c.dense_spec,
                c.array.plan,
                c.array.mapping,
                sorted(c.array.device_names),
                (c.replica.selector.mapping, sorted(c.replica.port_nets.items()))
                if c.replica
                else None,
            )
        ).encode()
    ).hexdigest()
    cached = getattr(c, "_dense_layout", None)
    if cached is None or cached[0] != key:
        top, meta = build_dense_macro(c)
        canonicalize_cells(top)
        certificates = meta.pop("certificates")
        meta["schema"] = "brom_dense_physical_v1"
        geometry = sha256()
        for layer, polygons in sorted(top.get_polygons(by_spec=True).items()):
            geometry.update(repr(layer).encode())
            for polygon in polygons:
                geometry.update(polygon.tobytes())
        meta["geometry_sha256"] = geometry.hexdigest()
        meta["physical_sha256"] = sha256(repr((key, meta)).encode()).hexdigest()
        meta["placement_certificates"] = certificates
        c._dense_layout = (key, top, meta)
    _, top, meta = c._dense_layout
    result = deepcopy(top)
    result.name = c.name
    return result, meta


def selector_routes(plan, mapping, transform, ox, oy, candidate, bitline):
    routes = []

    def add(net, metal, a, b):
        routes.append(
            (
                net,
                metal,
                (transform(ox + a[0]), oy + a[1]),
                (transform(ox + b[0]), oy + b[1]),
            )
        )

    h = len(mapping) * 432
    w = len(mapping[0]) * 108
    for col in range(len(mapping[0])):
        add(bitline(col), "M3", (108 * col + 81, 0), (108 * col + 81, h))
    for b, row in enumerate(mapping):
        for k, (metal, y) in enumerate(plan.tracks):
            add(candidate(b, k), metal, (-48, b * 432 + y), (w + 48, b * 432 + y))
        for col, k in enumerate(row):
            x = 108 * col
            y = b * 432
            net = candidate(b, k)
            gy = y + plan.gate_y
            dy = y + plan.drain_y
            add(net, "V1", (x + 45, gy), (x + 45, gy))
            add(net, "M2", (x + 10, gy), (x + 62, gy))
            add(net, "V2", (x + 27, gy), (x + 27, gy))
            add(
                net,
                "M3",
                (x + 27, y + plan.stem_span[0]),
                (x + 27, y + plan.stem_span[1]),
            )
            metal, ty = plan.tracks[k]
            add(
                net, "V2" if metal == "M2" else "V3", (x + 27, y + ty), (x + 27, y + ty)
            )
            add(bitline(col), "V1", (x + 54, dy), (x + 54, dy))
            add(bitline(col), "M2", (x + 37, dy), (x + 98, dy))
            add(bitline(col), "V2", (x + 81, dy), (x + 81, dy))
    return routes


def verify_dense_routes(top, checkpoints):
    """Audit actual hierarchical metal, including power and M6 candidates."""
    from .routing import metal_components

    component = metal_components(top, recursive=True)
    nets = {}
    owners = {}
    for net, metal, x, y in checkpoints:
        node = component(metal, x, y)
        if net in nets and nets[net] != node:
            raise ValueError(f"disconnected dense route: {net}")
        if node in owners and owners[node] != net:
            raise ValueError(f"shorted dense routes: {owners[node]} and {net}")
        nets[net] = node
        owners[node] = net
