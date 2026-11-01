"""Anchor-driven row placement for the dense peripheral bank.

A quadratic (clique-model) placement pulls each unit toward the units it
shares nets with and toward the x positions where the parent macro drops onto
its boundary nets. Abacus-style legalization then packs the units into
rows, keeping the spacer each pair of neighbouring cell edges needs. This
is a wirelength heuristic, not an area or routing optimum.
"""

import re

import numpy as np

from .dense_cell import edge_gap, wall_gap

# Body-tap reach: rows keep logic within each 20412 nm segment and leave the
# following three 108 nm slots empty for a filler/tap/filler.
TAP_PERIOD = 20736
TAP_SEGMENT = 20412

_RECEIVER = re.compile(r"sample_receiver_\w+?_(\d+)$")


def _blocks(units):
    """Keep each SAMPLE receiver, latch and observer as one ordered block."""
    spans = {}
    for i, unit in enumerate(units):
        match = _RECEIVER.match(unit.name)
        if match:
            lo, _ = spans.get(match[1], (i, i))
            spans[match[1]] = (lo, i)
    owner = {}
    for lo, hi in spans.values():
        if any(i in owner for i in range(lo, hi + 1)):
            raise ValueError("overlapping SAMPLE receiver placement runs")
        owner.update({i: lo for i in range(lo, hi + 1)})
    blocks, seen = [], set()
    for i in range(len(units)):
        start = owner.get(i, i)
        if start in seen:
            continue
        seen.add(start)
        blocks.append([j for j in range(start, len(units)) if owner.get(j, j) == start])
    return blocks


def _targets(units, anchors, width):
    """Clique-model quadratic placement of unit centers."""
    n = len(units)
    a = np.zeros((n, n))
    b = np.zeros(n)
    members = {}
    for i, unit in enumerate(units):
        mapping = dict(unit.nets)
        for port in unit.ports:
            net = mapping[port]
            if net not in ("VDD", "VSS"):
                members.setdefault(net, set()).add(i)
    for net, ms in members.items():
        ms = sorted(ms)
        xs = [x for x in anchors.get(net, ()) if x is not None]
        pins = len(ms) + len(xs)
        if pins < 2:
            continue
        w = 1 / (pins - 1)
        for i in ms:
            a[i, i] += w * (pins - 1)
            b[i] += w * sum(xs)
            for j in ms:
                if j != i:
                    a[i, j] -= w
    # A weak spring to the electrical order fixes unanchored components.
    total = sum(u.plan.width for u in units)
    used, eps = 0, 1e-2
    for i, unit in enumerate(units):
        order = (used + unit.plan.width / 2) * width / max(total, 1)
        used += unit.plan.width
        a[i, i] += eps
        b[i] += eps * order
    return np.linalg.solve(a, b)


def _segments(rows, width):
    result = []
    for row in range(rows):
        for lo in range(0, width, TAP_PERIOD):
            hi = min(width, lo + TAP_SEGMENT) // 108 * 108
            if hi > lo:
                result.append((row, lo, hi))
    return result


def unit_gap(a, b):
    """(join, free) spacer between unit ``a`` and unit ``b`` to its right."""
    return edge_gap(a.plan.edges[1], b.plan.edges[0], share=a.vt == b.vt)


def _legalize(blocks, widths, targets, segments, climb, gap, wall, sink=None):
    """Abacus: blocks in target order, each into its cheapest segment.

    ``climb[k]`` is the cost per row below the top row for block k, whose port
    columns rise through every row above it; ``sink[k]`` the cost per row
    above the bottom row, for columns that descend to a bottom pin. ``gap(j,
    k)`` is the (join, free) spacer between blocks j and k; ``wall(k)`` the
    spacer block k needs at the start of a row.
    """
    sink = sink or [0] * len(blocks)
    top = max(row for row, _, _ in segments)
    # Per segment: clusters [x, width, weight, q, [(block, offset)]]
    state = [[] for _ in segments]
    used = [0] * len(segments)

    def collapse(clusters, lo, hi):
        while True:
            c = clusters[-1]
            x = round(c[3] / c[2] / 108) * 108
            low = lo + (wall(c[4][0][0]) if lo == 0 else 0)
            c[0] = min(max(x, low), hi - c[1])
            if len(clusters) < 2:
                return c[0] >= low
            p = clusters[-2]
            join, free = gap(p[4][-1][0], c[4][0][0])
            space = c[0] - p[0] - p[1]
            if space >= free or space == join:
                return True
            shift = p[1] + join
            for block, offset in c[4]:
                p[4].append((block, shift + offset))
            p[3] += c[3] - c[2] * shift
            p[2] += c[2]
            p[1] = shift + c[1]
            clusters.pop()

    def append(clusters, block, lo, hi):
        t = targets[block]
        clusters.append([t, widths[block], 1, t, [(block, 0)]])
        return collapse(clusters, lo, hi)

    for block in sorted(range(len(blocks)), key=lambda k: (targets[k], k)):
        best = None
        for s, (_, lo, hi) in enumerate(segments):
            if used[s] + widths[block] > hi - lo:
                continue
            trial = [[*c[:4], list(c[4])] for c in state[s]]
            if not append(trial, block, lo, hi):
                continue
            x = next(trial[-1][0] + off for b, off in trial[-1][4] if b == block)
            row = segments[s][0]
            cost = abs(x - targets[block]) + climb[block] * (top - row)
            cost += sink[block] * row
            if best is None or cost < best[0]:
                best = (cost, s)
        if best is None:
            return None
        s = best[1]
        append(state[s], block, *segments[s][1:])
        used[s] += widths[block]
    placed = {}
    for s, clusters in enumerate(state):
        for x, _, _, _, entries in clusters:
            for block, offset in entries:
                placed[block] = (segments[s][0], x + offset)
    return placed


def anchored_rows(
    units,
    anchors,
    width,
    rows,
    ports=(),
    climb=216,
    spread=0.0,
    sinks=(),
):
    """Return rows of (unit, x), or None if the units do not fit.

    ``anchors`` maps boundary nets to parent drop x positions. Units on the
    bank's ``ports`` rise to the header above the rows; each such net costs
    ``climb`` nm of displacement per row below the top. Ports in ``sinks`` descend to a bottom pin instead and
    cost the same per row above the bottom. ``spread`` blends each target
    toward the position of uniform cell density (0 to 1), so anchors cannot
    pile every row's cells, and their columns, into one place.
    """
    width = width // 108 * 108
    blocks = _blocks(units)
    centers = _targets(units, anchors, width)
    widths, targets, inner = [], [], []
    for block in blocks:
        offsets, left, xs = [], 0, []
        for k, i in enumerate(block):
            if k:
                left += unit_gap(units[block[k - 1]], units[i])[0]
            xs.append(left)
            offsets.append(centers[i] - left - units[i].plan.width / 2)
            left += units[i].plan.width
        widths.append(left)
        targets.append(float(np.mean(offsets)))
        inner.append(xs)
    if spread:
        used, total = 0, sum(widths)
        for k in sorted(range(len(blocks)), key=lambda k: (targets[k], k)):
            uniform = (used + widths[k] / 2) * width / total - widths[k] / 2
            targets[k] += spread * (uniform - targets[k])
            used += widths[k]
    sinks = set(sinks) - {"VDD", "VSS"}
    ports = set(ports) - {"VDD", "VSS"} - sinks
    climbs, sunk = [], []
    for block in blocks:
        nets = {dict(units[i].nets)[p] for i in block for p in units[i].ports}
        climbs.append(climb * len(nets & ports))
        sunk.append(climb * len(nets & sinks))
    placed = _legalize(
        blocks,
        widths,
        targets,
        _segments(rows, width),
        climbs,
        lambda j, k: unit_gap(units[blocks[j][-1]], units[blocks[k][0]]),
        lambda k: wall_gap(units[blocks[k][0]].plan.edges[0]),
        sunk,
    )
    if placed is None:
        return None
    packed = [[] for _ in range(rows)]
    for k, block in enumerate(blocks):
        row, x = placed[k]
        for i, dx in zip(block, inner[k]):
            packed[row].append((units[i], x + dx))
    for row in packed:
        row.sort(key=lambda p: p[1])
    return packed
