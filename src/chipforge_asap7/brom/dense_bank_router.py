"""Over-cell routing for abutting CMOS rows.

M4 carries pin escapes inside each row. Same-net pins within a short x window
form a group with one M5 column across the rows it touches. An M6 bus joins a
net's groups in the row that needs the least column extension and has room;
rows above the bank carry buses only. Only one column per bank port rises to
the port's M4 header track above the rows. Columns span only the rows they
serve, so columns in disjoint row ranges share an x position.

A port may instead be *anchored*: its riser column is asked to sit within an
x range where the parent connects to it directly, so it needs no header
track; the row escape absorbs the offset between the cell and the column,
or, when that row is full, a *jog*: a column near the cell rises to a row
with room, an M4 segment there reaches the anchored column, which rises on
from that row. An anchor that finds no free column in its range, or no row
for its run, falls back to a header track. A *bottom* port's column runs
down to the bank's lower edge, where its pin lies.

Every M5 and M6 end lies on a 48 nm lattice. Tip stagger, parallel run and
same-track gaps are then multiples of 48 nm (ASAP7 M5/M6.S.2, S.4 and S.5
need 40, 40 and 44 nm). The remaining corner rule (S.3) forbids an end that
exactly meets an end on the adjacent track; the packers check it directly.
This is a routing heuristic, not an area optimum.
"""

from collections import defaultdict
from functools import lru_cache

from .compact import assign_tracks

#: Rising port columns end above the header; compare as unbounded.
TOP = 1 << 40
#: Bottom pin columns start at the bank's lower boundary.
BOTTOM = -54
#: Same-net pins within this x window share one M5 column; M4 escapes in
#: each row reach it.
GROUP_SPAN = 864
#: A one-row internal net within this span needs only its M4 escape.
LOCAL_SPAN = 864
#: Rows above the bank available to buses that do not fit over the cells.
BUS_ROWS_ABOVE = 64
#: Relief passes over overfull rows before giving up. Each pass relieves
#: every crowded row at once; passes with anchored ports count separately.
MAX_SPLITS = 64


def _ceil(n, grid):
    return -(-n // grid) * grid


def _band(row):
    """M5 extent covering every M4 escape track of a row, V4 enclosure included."""
    return row * 432 + 36, row * 432 + 420


def _fits(spans, at, lo, hi):
    """Can (lo, hi) join ``spans[at]`` without breaking same/adjacent-track rules?"""
    if any(lo < b + 48 and a < hi + 48 for a, b in spans.get(at, ())):
        return False
    return not any(
        hi == a or lo == b for k in (at - 1, at + 1) for a, b in spans.get(k, ())
    )


def _pack_columns(demands, limit, blocked=frozenset()):
    """Greedy x for (target x, lo, hi, net, window, kind) demands.

    Tallest first, nearest free slot; a demand with a ``window`` (x lo, x hi)
    tries the slots inside it before any outside, and never takes a slot in
    ``blocked``, where the parent's own M5 lies above the bank. Two columns
    of one net never take adjacent tracks over a shared y range: their vias
    onto one bus or escape would sit 24 nm apart. Nor do two anchored
    columns of different ``kind``: the parent ends them at heights that can
    differ by less than a tip stagger.
    """
    spans, nets, kinds = defaultdict(list), defaultdict(list), defaultdict(list)
    result: list[int | None] = [None] * len(demands)
    last = (limit - 12) // 48
    order = sorted(
        range(len(demands)),
        key=lambda i: (
            demands[i][2] < TOP,
            demands[i][4] is None,
            (demands[i][4][1] - demands[i][4][0]) if demands[i][4] is not None else 0,
            -(demands[i][2] - demands[i][1]),
            demands[i][0],
            i,
        ),
    )
    for i in order:
        target, lo, hi, net, window, kind = demands[i]
        home = min(max(round((target - 12) / 48), 0), last)
        slots = (
            k
            for step in range(2 * last + 2)
            if 0 <= (k := home + (step + 1) // 2 * (1 if step % 2 else -1)) <= last
        )
        if window is not None:
            inside = [k for k in slots if window[0] <= 48 * k + 12 <= window[1]]
            seen = set(inside)
            slots = inside + [
                k
                for step in range(2 * last + 2)
                if 0 <= (k := home + (step + 1) // 2 * (1 if step % 2 else -1)) <= last
                and k not in seen
            ]
        for k in slots:
            if window is not None and 48 * k + 12 in blocked:
                continue
            if not _fits(spans, k, lo, hi):
                continue
            if any(
                n == net and lo < b and a < hi
                for j in (k - 1, k + 1)
                for a, b, n in nets[j]
            ):
                continue
            if kind is not None and any(
                other != kind for j in (k - 1, k + 1) for other in kinds[j]
            ):
                continue
            spans[k].append((lo, hi))
            nets[k].append((lo, hi, net))
            if kind is not None:
                kinds[k].append(kind)
            result[i] = 48 * k + 12
            break
        else:
            return None
    return result


def _pack_buses(demands, spans=None):
    """Greedy M6 tracks for (x lo, x hi, y ranges in preference order).

    Longest buses go first; each takes the track nearest the middle of its
    first range with room. Return (y, range index) per bus, None if none fit.
    ``spans`` carries the tracks already taken, and is extended in place.
    """
    spans = defaultdict(list) if spans is None else spans
    result: list[tuple[int, int] | None] = [None] * len(demands)
    order = sorted(
        range(len(demands)), key=lambda i: (demands[i][0] - demands[i][1], i)
    )
    for i in order:
        lo, hi, ranges = demands[i]
        for choice, (ylo, yhi) in enumerate(ranges):
            first, last = _ceil(ylo - 16, 64) // 64, (yhi - 16) // 64
            middle = (ylo + yhi) / 2
            tracks = (
                range(first, last + 1)
                if yhi - ylo > 432
                else sorted(
                    range(first, last + 1), key=lambda k: abs(16 + 64 * k - middle)
                )
            )
            k = next((k for k in tracks if _fits(spans, k, lo, hi)), None)
            if k is not None:
                spans[k].append((lo, hi))
                result[i] = (16 + 64 * k, choice)
                break
    return result


@lru_cache(maxsize=128)
def route_bank(
    points,
    pins,
    row_width,
    row_count,
    pin_width,
    anchors=(),
    bottom=(),
    spans=(),
    edges=None,
    blocked=(),
):
    """Plan from (net, row, x, y) accesses; return None if the bank cannot route.

    ``anchors`` holds (net, x lo, x hi, kind) windows for anchored ports,
    ``bottom`` the ports pinned at the lower edge, ``spans`` the (net, xs)
    where a parent drops onto a port's header track, ``edges`` the ports
    whose track runs on to the nearer bank edge (every port when None) and
    ``blocked`` the x positions an anchored column may not take. Anchored
    columns of different kinds never sit on adjacent lattice slots.

    A row whose escapes need more than its eight M4 tracks is relieved first
    by moving the run of the anchored port with the widest escape crossing
    the crowded x up a row: a short escape to a nearby column, an M4 jog in
    the row above to the anchored column, which rises from there. A run
    with no row left is released to a header track. Failing that, the
    widest group crossing it is split into two groups, each with its own
    column, and the bank is routed again.
    """
    breaks = defaultdict(set)
    anchors = {net: (lo, hi, kind) for net, lo, hi, kind in anchors}
    jogs, tried = {}, defaultdict(set)

    def load(groups, columns, jog_spans, r, lo, hi):
        """Most M4 escapes live at once in row ``r`` over x in [lo, hi]."""
        spans = []
        for i, (_net, ps) in enumerate(groups):
            xs = [x for rr, x, _ in ps if rr == r]
            if not xs:
                continue
            if columns[i] is not None:
                xs.append(columns[i])
            spans.append(((min(xs) - 24) // 48 * 48, _ceil(max(xs) + 24, 48) + 40))
        spans += [(a, b + 40) for rr, a, b in jog_spans.values() if rr == r]
        events = sorted(
            (x, d)
            for a, b in spans
            if a < hi and b > lo
            for x, d in ((max(a, lo), 1), (min(b, hi), -1))
        )
        live = most = 0
        for _, d in events:
            live += d
            most = max(most, live)
        return most

    def relieve(row, at, groups, columns, jog_spans):
        """Relieve one crowded x: move a jog, split a group, or release.

        Anchored ports crossing the crowded x, via a jog or a direct escape,
        can move their run to another row with room. Prefer the move that
        lands in the emptiest row; failing any, split the widest other
        group crossing; failing that, release the widest anchored port to
        a header track. Return False when nothing applies.
        """
        crossing, split = [], None
        for net, (r, lo, hi) in jog_spans.items():
            if r == row and lo <= at < hi + 40:
                crossing.append((hi - lo, net, lo, hi))
        for i, (net, ps) in enumerate(groups):
            xs = [x for r, x, _ in ps if r == row]
            if not xs or at in breaks[net]:
                continue
            if columns[i] is not None:
                xs.append(columns[i])
            # The escape this group draws in the row, end space included.
            lo, hi = (min(xs) - 24) // 48 * 48, _ceil(max(xs) + 24, 48) + 40
            if not lo <= at < hi:
                continue
            # A jogged port's short escape to its first column stays put.
            if net in anchors and net not in jogs:
                crossing.append((hi - lo, net, lo, hi - 40))
            # Only a group with points on both sides of ``at`` can split.
            splits = min(x for _, x, _ in ps) < at <= max(x for _, x, _ in ps)
            if splits and net not in anchors and (split is None or hi - lo > split[0]):
                split = (hi - lo, net)
        moves = []
        for width_, net, lo, hi in crossing:
            first = min(r for n, ps in groups if n == net for r, _, _ in ps)
            tried[net].add(jogs.get(net, first))
            for r in range(first, row_count):
                if r not in tried[net]:
                    moves.append(
                        (
                            load(groups, columns, jog_spans, r, lo, hi + 40),
                            -width_,
                            net,
                            r,
                            first,
                            lo,
                            hi,
                        )
                    )
        moves.sort()
        if moves and moves[0][0] < 8:
            _, _, net, r, first, lo, hi = moves[0]
            if r == first:
                jogs.pop(net, None)
                jog_spans.pop(net, None)
            else:
                jogs[net] = r
                jog_spans[net] = (r, lo, hi)
            return True
        if split is not None:
            breaks[split[1]].add(at)
            return True
        if crossing:
            net = max(crossing)[1]
            del anchors[net]
            jogs.pop(net, None)
            jog_spans.pop(net, None)
            return True
        return False

    budget = MAX_SPLITS
    for _ in range(2 * MAX_SPLITS):
        result = _route(
            points,
            pins,
            row_width,
            row_count,
            pin_width,
            breaks,
            anchors,
            frozenset(bottom),
            dict(spans),
            None if edges is None else frozenset(edges),
            frozenset(blocked),
            jogs,
        )
        if result is None or "overfull" not in result:
            return result
        events, groups, columns, jog_spans = result["overfull"]
        jog_spans = dict(jog_spans)
        if anchors and budget == 0:
            # Out of passes with anchors in play: route as before, without.
            anchors.clear()
            jogs.clear()
            continue
        if anchors:
            budget -= 1
        for row, at in events:
            if not relieve(row, at, groups, columns, jog_spans):
                if anchors:
                    anchors.clear()
                    jogs.clear()
                    break
                return None
    return None


def _route(
    points,
    pins,
    row_width,
    row_count,
    pin_width,
    breaks,
    anchors,
    bottom,
    spans,
    edges,
    blocked,
    jogs,
):
    by_net = defaultdict(list)
    for net, row, x, y in points:
        by_net[net].append((row, x, y))
    groups = []
    for net, ps in sorted(by_net.items()):
        group = []
        for point in sorted(set(ps), key=lambda p: (p[1], p[0], p[2])):
            if group and (
                point[1] - group[0][1] > GROUP_SPAN
                or any(group[-1][1] < b <= point[1] for b in breaks[net])
            ):
                groups.append((net, tuple(group)))
                group = []
            group.append(point)
        if group:
            groups.append((net, tuple(group)))
    groups.sort(key=lambda g: (sum(x for _, x, _ in g[1]) / len(g[1]), g))
    # A one-row internal net within one window is joined by its M4 escape.
    local_only = {
        n
        for n, ps in by_net.items()
        if n not in pins
        and not breaks[n]
        and len({r for r, _, _ in ps}) == 1
        and max(x for _, x, _ in ps) - min(x for _, x, _ in ps) <= LOCAL_SPAN
    }
    by_group = defaultdict(list)
    for i, (net, _) in enumerate(groups):
        if net not in local_only:
            by_group[net].append(i)
    nets = sorted(n for n, members in by_group.items() if len(members) > 1)
    if edges is None:
        edges = frozenset(pins)
    # A port rises to the header unless it is only a bottom pin.
    rising = {n for n in pins if n in spans or n in anchors or n not in bottom}
    risers = {}
    for net, members in by_group.items():
        if net not in pins:
            continue
        if net in rising:
            # The highest group's column rises to the header; a bus joins
            # the rest. Lower groups never block the rows above them.
            risers[net] = max(
                members, key=lambda i: (max(r for r, _, _ in groups[i][1]), -i)
            )
        else:
            # A column that only descends to a bottom pin starts from the
            # lowest group.
            risers[net] = min(
                members, key=lambda i: (min(r for r, _, _ in groups[i][1]), i)
            )
    rows = {i: [r for r, _, _ in groups[i][1]] for ms in by_group.values() for i in ms}
    targets = {
        i: sorted(x for _, x, _ in groups[i][1])[len(groups[i][1]) // 2] for i in rows
    }
    # A jogged anchor keeps its cell's column ordinary and puts the window
    # on a second column, which rises from the jog row.
    jog_rows = {
        net: r
        for net, r in jogs.items()
        if net in anchors
        and net in risers
        and len(by_group[net]) == 1
        and max(rows[risers[net]]) < r < row_count
    }
    windows, kinds = {}, {}
    for net, (lo, hi, kind) in anchors.items():
        if net not in risers:
            continue
        key = ("J", net) if net in jog_rows else risers[net]
        windows[key] = (lo, hi)
        kinds[key] = kind
        targets[key] = min(max(targets[risers[net]], lo), hi)
    limit = max(pin_width, row_width + 324) - 108
    # Each bus prefers the row needing the least column extension.
    choices = {}
    for net in nets:
        own = [(min(rows[i]), max(rows[i])) for i in by_group[net]]
        choices[net] = sorted(
            range(row_count),
            key=lambda b, own=own: (
                sum(max(0, b - hi) + max(0, lo - b) for lo, hi in own),
                -b,
            ),
        ) + [row_count]
    header = _ceil(row_count * 432 + 48, 48) + 12
    width = _ceil(max(pin_width, row_width + 324), 432)

    def _bus_range(b):
        if b < row_count:
            return _band(b)[0] + 27, _band(b)[1] - 27
        return _band(row_count)[0] + 27, _band(row_count + BUS_ROWS_ABOVE)[1] - 27

    def settle(on_m4):
        """Pack columns and M6 buses until every bus keeps its row.

        Buses in ``on_m4`` take an M4 header track instead, so their columns
        rise to the top.
        """
        bus_row = {net: rows_[0] for net, rows_ in choices.items()}
        bus_y = {}
        rounds = 2 * row_count + 8
        stable = defaultdict(int)
        for step in range(rounds + 2):
            # A bus that has kept its row for two rounds is held there, so
            # the loop settles; the last rounds hold every bus, and one that
            # no longer fits its row goes to a band above the rows instead.
            final = step >= rounds
            column_spans, jog_columns = {}, {}
            for net, members in by_group.items():
                bus = bus_row.get(net)
                for i in members:
                    lo, hi = min(rows[i]), max(rows[i])
                    if bus is not None and bus < row_count:
                        lo, hi = min(lo, bus), max(hi, bus)
                    if net in jog_rows:
                        hi = max(hi, jog_rows[net])
                    high = _band(hi)[1]
                    if bus is not None and bus >= row_count:
                        high = max(
                            high,
                            _ceil(bus_y.get(net, _band(row_count)[1]) - 9, 48) + 36,
                        )
                    column_spans[i] = (_band(lo)[0], high)
                    if net in on_m4:
                        column_spans[i] = (column_spans[i][0], TOP)
                if net in risers:
                    low, high = column_spans[risers[net]]
                    if net in rising and net not in jog_rows:
                        high = TOP
                    if net in bottom:
                        low = BOTTOM
                    column_spans[risers[net]] = (low, high)
                if net in jog_rows:
                    column_spans["J", net] = (_band(jog_rows[net])[0], TOP)
            routed = sorted(k for k in column_spans if isinstance(k, int))
            extra = sorted(k for k in column_spans if not isinstance(k, int))
            placed = _pack_columns(
                [
                    (
                        targets[k],
                        *column_spans[k],
                        groups[k][0] if isinstance(k, int) else k[1],
                        windows.get(k),
                        kinds.get(k),
                    )
                    for k in routed + extra
                ],
                limit,
                blocked,
            )
            if placed is None:
                return None
            columns: list[int | None] = [None] * len(groups)
            for k, x in zip(routed + extra, placed):
                if isinstance(k, int):
                    columns[k] = x
                else:
                    jog_columns[k[1]] = x
            demands, order, on_m6 = [], {}, []
            for net in nets:
                if net in on_m4:
                    continue
                on_m6.append(net)
                xs = [columns[i] for i in by_group[net]]
                # Keep the current row while it has room, so the loop settles.
                held = final or stable[net] >= 2
                order[net] = [bus_row[net]] + [
                    b for b in choices[net] if b != bus_row[net] and not held
                ]
                # V5 needs 27 nm of M5 beyond its centre, and M6 23 nm beyond.
                ranges = [_bus_range(b) for b in order[net]]
                demands.append((min(xs) - 24, max(xs) + 24, ranges))
            taken = defaultdict(list)
            packed = _pack_buses(demands, taken)
            if any(p is None for p in packed):
                if step > rounds:
                    return None
                # Free the stuck buses into any band above the rows, around
                # the buses already placed.
                stuck = [k for k, p in enumerate(packed) if p is None]
                for k in stuck:
                    net = on_m6[k]
                    order[net] = [row_count]
                    ranges = [_bus_range(row_count)]
                    demands[k] = (*demands[k][:2], ranges)
                    stable[net] = 0
                freed = _pack_buses([demands[k] for k in stuck], taken)
                if any(p is None for p in freed):
                    return None
                for k, p in zip(stuck, freed):
                    packed[k] = p
            bus_y = {net: p[0] for net, p in zip(on_m6, packed) if p}
            rows_now = dict(bus_row)
            rows_now.update(
                {net: order[net][p[1]] for net, p in zip(on_m6, packed) if p}
            )
            if rows_now == bus_row:
                for net, p in zip(on_m6, packed):
                    if p and bus_row[net] >= row_count:
                        for i in by_group[net]:
                            if column_spans[i][1] < TOP:
                                hi = max(rows[i])
                                if net in jog_rows:
                                    hi = max(hi, jog_rows[net])
                                column_spans[i] = (
                                    column_spans[i][0],
                                    max(_band(hi)[1], _ceil(p[0] - 9, 48) + 36),
                                )
                return (
                    columns,
                    jog_columns,
                    column_spans,
                    routed,
                    bus_row,
                    on_m6,
                    packed,
                    demands,
                )
            for net in on_m6:
                stable[net] = stable[net] + 1 if rows_now[net] == bus_row[net] else 0
            bus_row = rows_now
        return None

    def extent(xs):
        return (min(xs) - 24) // 48 * 48, _ceil(max(xs) + 24, 48)

    def riser_x(net, columns, jog_columns):
        """The column of ``net`` that rises above the rows."""
        return jog_columns.get(net, columns[risers[net]])

    def port_columns(net, columns, jog_columns, on_m4):
        """Columns of ``net`` that rise to its header track."""
        if net in on_m4:
            return tuple(columns[i] for i in by_group[net])
        return (riser_x(net, columns, jog_columns),)

    def port_spans(columns, jog_columns, on_m4=frozenset()):
        """Header span of every port that needs a track.

        A port whose own bus moved onto M4 has every column rise to its one
        header track, which then spans them all.
        """
        result = {}
        for net in pins:
            if net not in risers:
                if spans.get(net):
                    raise ValueError(f"bank port {net} has no access")
                continue
            column = riser_x(net, columns, jog_columns)
            lo, hi, _ = anchors.get(net, (None, None, None))
            if lo is not None and lo <= column <= hi and net not in on_m4:
                continue
            if net in bottom and not spans.get(net) and net not in on_m4:
                continue
            lo, hi = extent(
                [*port_columns(net, columns, jog_columns, on_m4), *spans.get(net, ())]
            )
            if net in edges and net not in bottom:
                if lo < width - hi:
                    lo = -48
                else:
                    hi = max(hi, width + 48)
            result[net] = (lo, hi)
        return result

    settled = settle(frozenset())
    if settled is None:
        return None
    columns, jog_columns, column_spans, routed, bus_row, on_m6, packed, demands = (
        settled
    )
    # Buses that overflowed above the rows cost a 432 nm band of M6 per five;
    # an M4 header track costs 48 nm. Move the highest down onto M4 tracks
    # until the two layers end at about the same height.
    ports = port_spans(columns, jog_columns)
    bus_y = {
        net: p[0]
        for net, p in zip(on_m6, packed)
        if p and bus_row[net] >= row_count
    }
    above = sorted(bus_y, key=lambda net: (-bus_y[net], net))
    on_m4 = set()
    m4_margin = 148 if anchors or spans else 0
    while above:
        top_m6 = bus_y[above[0]]
        trial = on_m4 | {n for n in above if bus_y[n] == top_m6}
        channel = list(ports.values()) + [
            extent([columns[i] for i in by_group[net]]) for net in trial
        ]
        tracks = 0
        if channel:
            assigned, _ = assign_tracks(
                tuple((lo, hi + 40) for lo, hi in channel), end_space=40
            )
            tracks = max(assigned) + 1
        if top_m6 <= header + 48 * max(tracks - 1, 0) + m4_margin:
            break
        on_m4 = trial
        above = [n for n in above if bus_y[n] != top_m6]
    if on_m4:
        settled = settle(frozenset(on_m4))
        if settled is None:
            return None
        columns, jog_columns, column_spans, routed, bus_row, on_m6, packed, demands = (
            settled
        )
        ports = port_spans(columns, jog_columns, frozenset(on_m4))
    spans_by_group = column_spans
    displacement = max(
        (max(abs(columns[i] - x) for _, x, _ in groups[i][1]) for i in routed),
        default=0,
    )
    # A jog joins a jogged anchor's two columns on its row's M4 channel.
    jog_spans = {
        net: (r, *extent([columns[risers[net]], jog_columns[net]]))
        for net, r in jog_rows.items()
    }
    # Every row's M4 channel: escapes to columns, and jogs between columns.
    channels, overfull, peaks = {}, [], {}
    for row in range(row_count):
        local = []
        for i, (net, ps) in enumerate(groups):
            accesses = tuple((x, y) for r, x, y in ps if r == row)
            if not accesses:
                continue
            xs = [x for x, _ in accesses]
            if columns[i] is not None:
                xs.append(columns[i])
            lo = (min(xs) - 24) // 48 * 48
            hi = _ceil(max(xs) + 24, 48)
            local.append((i, accesses, lo, hi))
        local += [
            (("J", net), (), lo, hi)
            for net, (r, lo, hi) in jog_spans.items()
            if r == row
        ]
        channels[row] = local
        # Reject an overfull row before asking CP-SAT to minimize a channel
        # that cannot fit. End events precede starts at a shared endpoint.
        events = sorted(
            (x, delta) for _, _, a, b in local for x, delta in ((a, 1), (b + 40, -1))
        )
        live, peak = 0, (0, 0)
        for x, delta in events:
            live += delta
            if live > 8:
                overfull.append((row, x))
                break
            peak = max(peak, (live, x))
        peaks[row] = peak[1]
    if overfull:
        return {"overfull": (overfull, groups, columns, jog_spans)}
    escapes, jog_tracks = [], []
    for row, local in channels.items():
        tracks, _ = assign_tracks(
            tuple((a, b + 40) for _, _, a, b in local), end_space=40
        )
        if max(tracks) >= 8:
            # Aligned tips need a spare track: relieve the busiest x.
            overfull.append((row, peaks[row]))
            continue
        for (key, accesses, lo, hi), track in zip(local, tracks):
            y = row * 432 + 60 + 48 * track
            if isinstance(key, int):
                escapes.append((key, y, accesses, lo, hi))
            else:
                net = key[1]
                jog_tracks.append(
                    (net, y, lo, hi, (columns[risers[net]], jog_columns[net]))
                )
    if overfull:
        return {"overfull": (overfull, groups, columns, jog_spans)}
    buses = tuple(
        (net, p[0], lo, hi, tuple(columns[i] for i in by_group[net]))
        for net, p, (lo, hi, _) in zip(on_m6, packed, demands)
        if p
    )
    anchored = {
        net: riser_x(net, columns, jog_columns)
        for net, (lo, hi, _) in anchors.items()
        if net in risers and lo <= riser_x(net, columns, jog_columns) <= hi
    }
    return {
        "groups": tuple(groups),
        "columns": tuple(columns),
        "column_shapes": tuple(
            (groups[i][0], columns[i], *spans_by_group[i]) for i in routed
        )
        + tuple(
            (net, x, *spans_by_group["J", net])
            for net, x in sorted(jog_columns.items())
        ),
        "escapes": tuple(escapes),
        "jogs": tuple(jog_tracks),
        "buses": buses,
        "m4_buses": {
            net: (
                *extent([columns[i] for i in by_group[net]]),
                tuple(columns[i] for i in by_group[net]),
            )
            for net in sorted(on_m4)
            if net not in pins
        },
        "port_spans": ports,
        "header": header,
        "port_columns": {
            n: port_columns(n, columns, jog_columns, on_m4)
            if n in risers and (n in rising or n in on_m4)
            else ()
            for n in pins
        },
        "anchored": anchored,
        "bottom": {n: columns[risers[n]] for n in pins if n in bottom and n in risers},
        "width": width,
        "max_pin_displacement_nm": displacement,
    }


def route_header(plan):
    """Assign the M4 tracks above the rows; return (tracks, height).

    Every port that needs a track spans its M5 column and the x positions
    where a parent drops onto it (`port_spans`), and every bus moved onto
    M4 spans its columns. Disjoint spans share tracks. ``tracks`` maps each
    net to (y, left end, right end).
    """
    local = [(net, lo, hi) for net, (lo, hi) in plan["port_spans"].items()]
    local += [(net, lo, hi) for net, (lo, hi, _) in plan["m4_buses"].items()]
    tracks = {}
    if local:
        assigned, _ = assign_tracks(
            tuple((lo, hi + 40) for _, lo, hi in local), end_space=40
        )
        tracks = {
            net: (plan["header"] + 48 * track, lo, hi)
            for (net, lo, hi), track in zip(local, assigned)
        }
    top = max(
        [y + 48 for y, _, _ in tracks.values()]
        + [y + 48 for _, y, _, _, _ in plan["buses"]]
        # Columns may climb above the rows to a bus there.
        + [hi for _, _, _, hi in plan["column_shapes"] if hi < TOP],
        default=plan["header"],
    )
    return tracks, _ceil(top, 54)
