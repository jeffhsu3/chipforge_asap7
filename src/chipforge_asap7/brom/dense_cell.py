"""Shared-diffusion CMOS cells on the ASAP7 gate grid.

Placement is a transistor graph problem: diffusion may be shared only when
the touching terminals and fin counts agree. Opposite bands may share a gate
column only when their gate nets agree. Routing is solved against real boxes.
"""

from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256

from ..devices.row import RowStack
from ..devices.rowcell import (
    draw_frame,
    draw_rails,
    gate_contact,
    island,
    label,
    sd_contact,
    square,
    supply_contact,
    via_y,
)
from ..layout.layers import box, require_gdspy


@dataclass(frozen=True)
class DenseCellPlan:
    width: int
    height: int
    # (device index, gate slot, flipped source/drain)
    devices: tuple
    # (net, M3 x)
    trunks: tuple
    # (net, x, intrinsic y, access y)
    contacts: tuple
    placement_status: str
    placement_bound: int
    # x of the first gate slot's left contact
    offset: int = 54
    # (left, right): per band (flavor, nm from the edge to the outermost
    # contact, its net, fins)
    edges: tuple = ()


#: The rail net of each band; only these may share diffusion across cells.
SUPPLY = {"n": "VSS", "p": "VDD"}


def _solver(model, seconds=30):
    from ortools.sat.python import cp_model

    s = cp_model.CpSolver()
    s.parameters.num_search_workers = 1
    s.parameters.random_seed = 0
    s.parameters.max_time_in_seconds = seconds
    status = s.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise ValueError(f"dense cell solver: {s.status_name(status)}")
    return s, s.status_name(status)


@lru_cache(maxsize=2048)
def place_devices(devices, share=False):
    """Place (flavor, fins, drain, gate, source) with exact shared terminals.

    Return (placement, status, gate-slot bound, (left ready, right ready)).
    With ``share``, an edge is ready when a neighbour may share its diffusion.
    """
    from ortools.sat.python import cp_model

    m = cp_model.CpModel()
    nets = sorted({n for d in devices for n in d[2:]})
    ids = {n: i for i, n in enumerate(nets)}
    positions, left, right, flips = [], [], [], []
    for i, (_, _, d, _, s) in enumerate(devices):
        positions.append(m.new_int_var(0, 2 * len(devices), f"x{i}"))
        left.append(m.new_int_var(0, len(nets) - 1, f"l{i}"))
        right.append(m.new_int_var(0, len(nets) - 1, f"r{i}"))
        flips.append(m.new_bool_var(f"flip{i}"))
        m.add_allowed_assignments(
            [flips[-1], left[-1], right[-1]], [(0, ids[s], ids[d]), (1, ids[d], ids[s])]
        )
    adjacency = {}
    for i, a in enumerate(devices):
        for j, b in enumerate(devices[:i]):
            if a[0] != b[0]:
                if a[3] != b[3]:
                    m.add(positions[i] != positions[j])
                continue
            m.add(positions[i] != positions[j])
            for u, v in ((i, j), (j, i)):
                adjacent = m.new_bool_var(f"adj{u}_{v}")
                adjacency[u, v] = adjacent
                m.add(positions[v] == positions[u] + 1).only_enforce_if(adjacent)
                m.add(positions[v] != positions[u] + 1).only_enforce_if(adjacent.Not())
                if a[1] != b[1]:
                    m.add(adjacent == 0)
                else:
                    m.add(right[u] == left[v]).only_enforce_if(adjacent)
    # A diffusion break between different nets needs 92 nm. One empty
    # gate gives only 38 nm; two empty gate sites give exactly 92 nm.
    for u, a in enumerate(devices):
        for v, b in enumerate(devices):
            if u == v or a[0] != b[0]:
                continue
            near = m.new_bool_var(f"near{u}_{v}")
            m.add(positions[v] == positions[u] + 2).only_enforce_if(near)
            m.add(positions[v] != positions[u] + 2).only_enforce_if(near.Not())
            same = m.new_bool_var(f"same{u}_{v}")
            m.add(right[u] == left[v]).only_enforce_if(same)
            m.add(right[u] != left[v]).only_enforce_if(same.Not())
            middle = [
                adjacency[u, w]
                for w in range(len(devices))
                if w not in (u, v) and (u, w) in adjacency
            ]
            m.add_bool_or([near.Not(), same, *middle])
    last = m.new_int_var(0, 2 * len(devices), "last")
    m.add_max_equality(last, positions)
    m.minimize(last)
    s, status = _solver(m)
    bound = int(s.best_objective_bound)
    placement = tuple(
        (i, s.value(x), bool(s.value(flips[i]))) for i, x in enumerate(positions)
    )
    if not share:
        return placement, status, bound, (False, False)
    # Keep the gate-slot count; then make each edge shareable if possible:
    # every diffusion in its outer slot ends on the band's supply, and no
    # band starts one slot in (its contact would face a neighbour's at 54 nm).
    end = s.value(last)
    m.add(last == end)
    for i, x in enumerate(positions):
        m.add_hint(x, s.value(x))
        m.add_hint(flips[i], s.value(flips[i]))
    ready = [m.new_bool_var("ready_left"), m.new_bool_var("ready_right")]
    for r, outer, (edge, inner) in zip(ready, (left, right), ((0, 1), (end, end - 1))):
        at = {}
        for i in range(len(devices)):
            at[i] = m.new_bool_var(f"at{edge}_{i}")
            m.add(positions[i] == edge).only_enforce_if(at[i])
            m.add(positions[i] != edge).only_enforce_if(at[i].Not())
        for i, d in enumerate(devices):
            supply = ids.get(SUPPLY[d[0]])
            if supply is None:
                m.add_bool_or([at[i].Not(), r.Not()])
            else:
                m.add(outer[i] == supply).only_enforce_if([at[i], r])
            nxt = m.new_bool_var(f"at{inner}_{i}")
            m.add(positions[i] == inner).only_enforce_if(nxt)
            m.add(positions[i] != inner).only_enforce_if(nxt.Not())
            same = [at[j] for j, e in enumerate(devices) if e[0] == d[0]]
            m.add_bool_or([nxt.Not(), r.Not(), *same])
    m.maximize(sum(ready))
    s, _ = _solver(m)
    return (
        tuple(
            (i, s.value(x), bool(s.value(flips[i]))) for i, x in enumerate(positions)
        ),
        status,
        bound,
        tuple(bool(s.value(r)) for r in ready),
    )


def terminal_frame(devices, placement, height=432, ports=(), offset=54):
    """Return shared diffusion islands and intrinsic M1 contact coordinates.

    Gate slot ``x`` lies between contacts ``offset + 54 * x`` and the next.
    """
    maxfin = {
        p: max((d[1] for d in devices if d[0] == p), default=1) for p in ("n", "p")
    }
    stack = RowStack(((maxfin["n"], maxfin["p"]),), band_height=height // 2)
    bands = {b.flavor: b for b in stack.bands()}
    gates, chains = {}, []
    for flavor in ("n", "p"):
        ordered = sorted(
            (x, i, flip) for i, x, flip in placement if devices[i][0] == flavor
        )
        chain = []
        for x, i, flip in ordered:
            _, fins, d, g, s = devices[i]
            lo, hi = (d, s) if flip else (s, d)
            gates[offset + 27 + 54 * x] = g
            entry = (x, fins, lo, hi)
            if chain and (x != chain[-1][0] + 1 or fins != chain[-1][1]):
                chains.append((flavor, tuple(chain)))
                chain = []
            chain.append(entry)
        if chain:
            chains.append((flavor, tuple(chain)))
    terminals = []
    for flavor, chain in chains:
        band = replace(
            bands[flavor], spec=replace(bands[flavor].spec, fins=chain[0][1])
        )
        endpoints = [(offset + 54 * chain[0][0], chain[0][2])]
        endpoints += [(offset + 54 + 54 * x, hi) for x, _, _, hi in chain]
        for x, net in endpoints:
            if net != band.rail_net:
                terminals.append((net, x, via_y(band), flavor))
    counts = defaultdict(int)
    for net, _, _, _ in terminals:
        counts[net] += 1
    terminals = [
        t
        for t in terminals
        if counts[t[0]] > 1
        or t[0] in ports
        or t[0] in gates.values()
        or t[0] in ("VDD", "VSS")
    ]
    terminals += [(net, x, height // 2, "g") for x, net in sorted(gates.items())]
    return stack, chains, gates, tuple(dict.fromkeys(terminals))


def _close(a, b, layer):
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    # Parallel long edges need 18 nm; tips/corners need 25 nm.
    space = (
        30
        if layer.startswith("V") and dx and dy
        else 18
        if layer.startswith("V")
        else 25
    )
    if layer == "M1" and dy == 0 and min(a[3], b[3]) > max(a[1], b[1]):
        space = 18
    if layer == "M2" and dx == 0 and min(a[2], b[2]) > max(a[0], b[0]):
        space = 18
    return dx * dx + dy * dy < space * space


def _edges(chains, offset, width):
    left, right = {}, {}
    for flavor, chain in chains:
        lo = (offset + 54 * chain[0][0], chain[0][2], chain[0][1])
        hi = (width - offset - 54 - 54 * chain[-1][0], chain[-1][3], chain[-1][1])
        if flavor not in left or lo[0] < left[flavor][0]:
            left[flavor] = lo
        if flavor not in right or hi[0] < right[flavor][0]:
            right[flavor] = hi
    return tuple(
        tuple(sorted((f, *e) for f, e in side.items())) for side in (left, right)
    )


def edge_gap(right, left, share=True):
    """Spacer (join, free) in nm between one cell's ``right`` edge and the next's ``left``.

    ``join`` is the least legal spacer; any spacer of at least ``free`` is
    legal too. Diffusion facing diffusion needs 108 nm between contact
    centres (a 92 nm break) unless both contacts sit on the boundary on the
    same supply with equal fins, which merges the two islands.
    """
    a = {f: e for f, *e in right}
    b = {f: e for f, *e in left}
    common = a.keys() & b.keys()
    free = max([0] + [108 - a[f][0] - b[f][0] for f in common])
    shared = share and all(
        a[f][0] == b[f][0] == 0 and a[f][1:] == b[f][1:] and a[f][1] == SUPPLY[f]
        for f in common
    )
    return (0 if shared else free), free


def wall_gap(left):
    """Spacer before a row's first cell: keep diffusion 46 nm inside the bank."""
    return max([0] + [54 - e for _, e, _, _ in left])


@lru_cache(maxsize=2048)
def cell_plan(devices, ports, reserved_m2=(), fixed_trunks=(), share=False):
    """Place and route a cell.

    Without ``share`` every cell keeps two empty gate sites from its
    neighbours, a full diffusion break. With it, a ready edge (see
    `place_devices`) puts its supply contacts on the cell boundary, where a
    neighbour's may coincide; `edge_gap` gives the spacer any pair needs.
    """
    from ortools.sat.python import cp_model

    placement, status, bound, ready = place_devices(devices, share)
    height = 432
    slots = max(x for _, x, _ in placement) + 1
    if share:
        core_width = 54 * (slots + 2 - sum(ready))
    else:
        core_width = 54 * (slots + 2)
    nets = sorted({t[0] for t in terminal_frame(devices, placement, height, ports)[3]})
    # Signal-only trunks; power contacts connect directly to shared M1 rails.
    for extra in range(9):
        width = max(core_width, 54 * len(nets)) + extra * 54
        if share:
            # Widening goes to an unready side, where it only adds clearance.
            offset = 0 if ready[0] else width - 54 * slots if ready[1] else 54
        else:
            width, offset = -(-width // 108) * 108, 54
        _stack, chains, _gates, terminals = terminal_frame(
            devices, placement, height, ports, offset
        )
        xs = tuple(range(27, width - 13, 54))
        m = cp_model.CpModel()
        trunk = {n: m.new_int_var(0, len(xs) - 1, f"trunk{n}") for n in nets}
        m.add_all_different(list(trunk.values()))
        for net, x in fixed_trunks:
            if x not in xs:
                raise ValueError("fixed trunk outside cell")
            m.add(trunk[net] == xs.index(x))
        choices, shapes = [], []
        for k, (net, x, y, flavor) in enumerate(terminals):
            ys = (
                [height // 2 + 36 * i for i in (-2, -1, 0, 1, 2)]
                if flavor == "g"
                else [
                    y + 36 * i
                    for i in (-2, -1, 0, 1, 2)
                    if (
                        48 <= y + 36 * i <= height // 2 - 36
                        if flavor == "n"
                        else height // 2 + 36 <= y + 36 * i <= height - 48
                    )
                ]
            )
            alternatives = []
            for ti, tx in enumerate(xs):
                for ty in ys:
                    if any(abs(ty - ry) < 36 for ry in reserved_m2):
                        continue
                    # M1 access, V1, horizontal M2, V2. M3 is a net trunk.
                    rects = {
                        "M1": (x - 9, min(y, ty) - 14, x + 9, max(y, ty) + 14),
                        "M2": (min(x, tx) - 19, ty - 9, max(x, tx) + 19, ty + 9),
                        "V1": (x - 9, ty - 9, x + 9, ty + 9),
                        "V2": (tx - 9, ty - 9, tx + 9, ty + 9),
                    }
                    if rects["M2"][0] < 0 or rects["M2"][2] > width:
                        continue
                    # Never let an access touch either power rail.
                    if rects["M1"][1] < 34 or rects["M1"][3] > height - 34:
                        continue
                    lo, hi = rects["M2"][0], rects["M2"][2]
                    if share:
                        # Within 17 nm of an edge, M2 keeps to the lower half
                        # on the left and the upper half on the right. Any
                        # two abutting tips then clear 25 nm.
                        if lo < 8 or hi > width - 8:
                            continue
                        if lo < 17 and ty > 197 or hi > width - 17 and ty < 235:
                            continue
                    elif lo < 13 or hi > width - 13:
                        # Abutted cells may carry unrelated nets on the same
                        # M2 track. Reserve half the 25 nm tip spacing.
                        continue
                    v = m.new_bool_var(f"a{k}_{ti}_{ty}")
                    m.add(trunk[net] == ti).only_enforce_if(v)
                    alternatives.append((v, ti, ty))
                    shapes.append((net, v, rects))
            m.add_exactly_one([a[0] for a in alternatives])
            choices.append(alternatives)
        for i, (net, v, rects) in enumerate(shapes):
            for other, u, boxes in shapes[:i]:
                conflict = False
                for layer in rects:
                    a, b = rects[layer], boxes[layer]
                    if net == other:
                        if a == b:
                            continue
                        if layer == "M2" and a[1] == b[1]:
                            continue
                        if layer == "M1" and a[0] == b[0]:
                            continue
                    if _close(a, b, layer):
                        conflict = True
                        break
                if conflict:
                    m.add_bool_or([v.Not(), u.Not()])
        m.minimize(
            sum(
                abs(xs[ti] - terminals[i][1]) * v
                for i, alts in enumerate(choices)
                for v, ti, _ in alts
            )
        )
        try:
            s, _ = _solver(m, 10)
        except ValueError:
            continue
        contacts = tuple(
            (net, x, y, next(ty for v, _, ty in choices[i] if s.value(v)))
            for i, (net, x, y, _) in enumerate(terminals)
        )
        return DenseCellPlan(
            width,
            height,
            placement,
            tuple((n, xs[s.value(trunk[n])]) for n in nets),
            contacts,
            status,
            bound,
            offset,
            _edges(chains, offset, width),
        )
    raise ValueError(f"dense cell has no routed placement: {devices}")


def build_dense_cell(
    devices,
    ports,
    *,
    vt="rvt",
    name=None,
    reserved_m2=(),
    fixed_trunks=(),
    abut_y=False,
    share=False,
):
    """Draw a tapless cell; parent must provide legal shared taps.

    With ``share``, see `cell_plan`: the parent must also keep the spacers
    `edge_gap` asks for.

    With ``abut_y``, signal trunk tips leave 32 nm across a shared row edge.
    Power trunks are exposed for parent routing instead of crossing the rail
    into a neighbour's signal trunk. Diffusion sources still use the M1 rails.
    """
    plan = cell_plan(
        tuple(devices), tuple(ports), tuple(reserved_m2), tuple(fixed_trunks), share
    )
    gdspy = require_gdspy()
    digest = sha256(repr((devices, ports, vt, plan, abut_y)).encode()).hexdigest()[:12]
    top = gdspy.Cell(name or f"brom_dense_{digest}", exclude_from_current=True)
    stack, chains, gates, intrinsic = terminal_frame(
        devices, plan.devices, plan.height, ports, plan.offset
    )
    retained = {(n, x, y) for n, x, y, _ in intrinsic}
    stack = replace(stack, vt=vt)
    draw_frame(top, stack, plan.width, vt)
    draw_rails(top, stack, plan.width)
    bands = {b.flavor: b for b in stack.bands()}
    for flavor, chain in chains:
        band = replace(
            bands[flavor], spec=replace(bands[flavor].spec, fins=chain[0][1])
        )
        nodes = [(plan.offset + 54 * chain[0][0], chain[0][2])]
        nodes += [(plan.offset + 54 + 54 * x, hi) for x, _, _, hi in chain]
        kept = [
            x for x, n in nodes if n == band.rail_net or (n, x, via_y(band)) in retained
        ]
        island(top, band, [x for x, _ in nodes], contacted=kept)
        for x, net in nodes:
            if net == band.rail_net:
                supply_contact(top, band, x)
            elif x in kept:
                sd_contact(top, band, x)
    for x in gates:
        gate_contact(top, plan.height // 2, [x], x, plan.height // 2)
    trunks = dict(plan.trunks)
    routed_ports = (*ports, "VDD", "VSS") if abut_y else ports
    margin = 16 if abut_y else 14
    for net, x, y, ty in plan.contacts:
        tx = trunks[net]
        box(top, "M1", x - 9, min(y, ty) - 14, x + 9, max(y, ty) + 14)
        square(top, "V1", x, ty)
        box(top, "M2", min(x, tx) - 19, ty - 9, max(x, tx) + 19, ty + 9)
        square(top, "V2", tx, ty)
    for net, x in plan.trunks:
        ys = [ty for n, _, _, ty in plan.contacts if n == net]
        lo, hi = (
            (margin, plan.height - margin)
            if net in routed_ports
            else (min(ys) - 14, max(ys) + 14)
        )
        box(top, "M3", x - 9, lo, x + 9, hi)
        if net in ("VDD", "VSS") and not abut_y:
            rail = 0 if net == "VSS" else plan.height
            square(top, "V1", x, rail)
            square(top, "V2", x, rail)
            box(top, "M2", x - 19, rail - 9, x + 19, rail + 9)
            box(top, "M3", x - 9, min(rail, lo) - 14, x + 9, max(rail, hi) + 14)
        if net in ports:
            label(top, net, "M3", (x, plan.height // 2))
    for net, y in [("VSS", 0), ("VDD", plan.height)]:
        label(top, net, "M1", (24, y))
    box(top, "BOUNDARY", 0, 0, plan.width, plan.height)
    return top, plan


def cell_routes(plan, ports, *, abut_y=False):
    """Centerlines and vias of local signal metal, in cell coordinates."""
    routes = []
    routed_ports = (*ports, "VDD", "VSS") if abut_y else ports
    margin = 16 if abut_y else 14
    trunks = dict(plan.trunks)
    for net, x, y, ty in plan.contacts:
        tx = trunks[net]
        routes.extend(
            [
                (net, "M1", (x, y), (x, ty)),
                (net, "V1", (x, ty), (x, ty)),
                (net, "M2", (x, ty), (tx, ty)),
                (net, "V2", (tx, ty), (tx, ty)),
            ]
        )
    for net, x in plan.trunks:
        ys = [ty for n, _, _, ty in plan.contacts if n == net]
        lo, hi = (
            (margin, plan.height - margin)
            if net in routed_ports
            else (min(ys) - 14, max(ys) + 14)
        )
        routes.append((net, "M3", (x, lo), (x, hi)))
    return routes
