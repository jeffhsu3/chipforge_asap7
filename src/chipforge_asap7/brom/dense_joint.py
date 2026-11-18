"""Bounded joint placement and routing repair for a peripheral bank.

Each solve keeps rows, cell order, net groups, and M5 vertical extents fixed.
Cell x positions, M5 columns, and M6 tracks change together. M4 occupancy is
minimized in one window; the full bank must still pass exact track assignment
and physical verification. Solver optimality applies only to this subproblem.
"""

from collections import defaultdict
from copy import deepcopy
from math import ceil

from .dense_cell import wall_gap
from .dense_place import TAP_PERIOD, TAP_SEGMENT, unit_gap
from .dense_slices import _BIT_UNIT, _LATCH


def congestion(candidate):
    """Return M4 overflow area and each row's worst point, including keepouts."""
    rows = defaultdict(list)
    for i, (_net, points) in enumerate(candidate["groups"]):
        by_row = defaultdict(list)
        for row, x, _y in points:
            by_row[row].append(x)
        for row, xs in by_row.items():
            column = candidate["columns"][i]
            if column is not None:
                xs.append(column)
            lo = (min(xs) - 24) // 48 * 48
            hi = -(-(max(xs) + 24) // 48) * 48 + 40
            rows[row].extend(((lo, 1), (hi, -1)))
    for net, row in candidate.get("jog_rows", {}).items():
        xs = (
            candidate["columns"][candidate["risers"][net]],
            candidate["jog_columns"][net],
        )
        lo = (min(xs) - 24) // 48 * 48
        hi = -(-(max(xs) + 24) // 48) * 48 + 40
        rows[row].extend(((lo, 1), (hi, -1)))
    overflow, peaks = 0, []
    for row, events in rows.items():
        live, last, peak = 0, 0, (0, 0)
        for x, delta in sorted(events):
            overflow += max(0, live - 8) * (x - last)
            live += delta
            if live > peak[0]:
                peak = live, x
            last = x
        if peak[0] > 8:
            peaks.append((*peak, row))
    return overflow, sorted(peaks, reverse=True)


def refine_route(packed, failure, *, max_windows=None, seconds=60):
    """Repair the worst region first; accept only a complete routed bank.

    Every model includes fixed cells/columns/buses outside its window as
    obstacles. Changes are checked across the full bank before continuing.
    None means this bounded search did not find a route, not infeasibility.
    """
    from .dense_bank_router import finish_route

    candidate = failure["candidate"]
    if max_windows is None:
        max_windows = 2 * max(8, ceil(candidate["width"] / 7000))
    reports, tried = [], defaultdict(int)
    for _ in range(max_windows):
        before, peaks = congestion(candidate)
        if peaks:
            count, at, _row = peaks[0]
        else:
            # Track assignment can also fail on the corner-spacing rule.
            _row, at = failure["overfull"][0][0]
            count = 9
        bucket = at // 1000
        attempt = tried[bucket]
        if attempt >= 3:
            return None
        tried[bucket] += 1
        span = (11000 if count >= 11 else 7000) + 4000 * attempt
        left = max(0, (at - span // 2) // 1000 * 1000)
        right = min(candidate["width"] - 324, left + span)
        placed, updated, report = solve_window(
            packed, candidate, left, right, move=864 * (attempt + 1), seconds=seconds
        )
        report["overflow_before_nm"] = before
        if report["solver_status"] not in ("OPTIMAL", "FEASIBLE"):
            reports.append(report)
            continue
        after, _ = congestion(updated)
        report["overflow_after_nm"] = after
        reports.append(report)
        if after > before:
            continue
        result = finish_route(updated)
        if "overfull" not in result:
            result["joint_optimization"] = reports
            return placed, result
        if after < before:
            packed, candidate, failure = placed, updated, result
    return None


def solve_window(packed, candidate, left, right, *, move=864, seconds=60):
    """Return (placement, routing candidate, report), including on timeout.

    The initial placement/route supplies a complete solver hint. One worker
    and a deterministic budget make successful model searches reproducible;
    a wall limit remains a backstop. Infeasibility refers to this fixed
    topology and movement window, never to the macro as a whole.
    """
    from ortools.sat.python import cp_model

    p, s = packed, candidate
    width = s["width"]
    row_width = s.get(
        "row_width", -(-max(x + u.plan.width for row in p for u, x in row) // 108) * 108
    )
    bits = {
        int(match[1])
        for row in p
        for u, _ in row
        if (match := _LATCH.fullmatch(u.name))
    }
    report = {
        "method": "cp_sat_joint",
        "window_nm": (left, right),
        "move_limit_nm": move,
        "scope": "fixed rows, cell order, net grouping and M5 vertical extents",
        "global_optimal": False,
    }
    # Jog topology is retained by the ordinary router. The current joint
    # model operates on candidates after those jogs have been released.
    if s["jog_columns"]:
        return p, s, {**report, "solver_status": "UNSUPPORTED_JOGS"}
    m = cp_model.CpModel()

    def variable(lo, hi, name):
        return m.new_int_var(int(lo), int(hi), name)

    def extremum(exprs, lo, hi, name, ismax):
        if lo == hi:
            return int(lo)
        if all(isinstance(v, int) for v in exprs):
            return (max if ismax else min)(exprs)
        v = variable(lo, hi, name)
        (m.add_max_equality if ismax else m.add_min_equality)(v, exprs)
        return v

    # Cells retain their row and order. Movable controls use legal 54nm sites.
    origins = {}
    pins = {}
    cellbounds = {}
    moving = []
    for r, row in enumerate(p):
        blocks = defaultdict(list)
        for i, (u, x) in enumerate(row):
            match = _BIT_UNIT.fullmatch(u.name)
            key = (
                ("bit", int(match[1]))
                if match and int(match[1]) in bits
                else ("unit", i)
            )
            blocks[key].append(i)
        for key, ids in blocks.items():
            base = min(row[i][1] for i in ids)
            end = max(row[i][1] + row[i][0].plan.width for i in ids)
            movable = (
                move
                and left <= base
                and end <= right
                and not any(row[i][0].name.startswith("readout_") for i in ids)
            )
            if movable:
                lo = max(wall_gap(row[ids[0]][0].plan.edges[0]), left, base - move)
                hi = min(right - (end - base), row_width - (end - base), base + move)
                allowed = []
                for a in range(0, width, TAP_PERIOD):
                    l = max(lo, a)
                    h = min(hi, a + TAP_SEGMENT - (end - base))
                    if -(-l // 54) <= h // 54:
                        allowed.append((-(-l // 54), h // 54))
                v = m.new_int_var_from_domain(
                    cp_model.Domain.from_intervals(allowed), f"cell_{r}_{key}"
                )
                m.add_hint(v, base // 54)
                for i in ids:
                    dx = row[i][1] - base
                    origins[r, i] = 54 * v + dx
                    cellbounds[r, i] = (
                        54 * allowed[0][0] + dx,
                        54 * allowed[-1][1] + dx,
                    )
                    moving.append((r, i, v))
            else:
                for i in ids:
                    origins[r, i] = row[i][1]
                    cellbounds[r, i] = (row[i][1], row[i][1])
        for i, (u, x) in enumerate(row):
            mapping = dict(u.nets)
            for port, dx in u.plan.trunks:
                if port in (*u.ports, "VDD", "VSS"):
                    pins[mapping[port], r, x + dx, r * 432 + 216] = (
                        origins[r, i] + dx,
                        cellbounds[r, i][0] + dx,
                        cellbounds[r, i][1] + dx,
                    )
        for i in range(len(row) - 1):
            u, x = row[i]
            v, y = row[i + 1]
            if isinstance(origins[r, i], int) and isinstance(origins[r, i + 1], int):
                continue
            join, free = unit_gap(u, v)
            gap = origins[r, i + 1] - origins[r, i] - u.plan.width
            m.add(gap >= join)
            if free > join:
                joined = m.new_bool_var(f"join_{r}_{i}")
                m.add(gap == join).only_enforce_if(joined)
                m.add(gap >= free).only_enforce_if(joined.Not())

    # Move columns in regions with crowded escapes, while retaining vertical spans.
    zones = [(left, right)]
    col = {}
    colbounds = {}
    colvar = {}
    pointdata = {}
    for i, (net, ps) in enumerate(s["groups"]):
        pointdata[i] = [(r, *pins.get((net, r, x, y), (x, x, x)), y) for r, x, y in ps]
        if s["columns"][i] is None:
            continue
        old = s["columns"][i]
        target = s["targets"][i]
        change = any(
            lo <= old <= hi or any(lo <= x <= hi for r, x, y in ps) for lo, hi in zones
        )
        if change:
            low = max(0, (min(old, target) - 864 - 12) // 48)
            high = min((s["limit"] - 12) // 48, -(-(max(old, target) + 864 - 12) // 48))
            window = s["windows"].get(i)
            if window and window[0] <= old <= window[1]:
                low = max(low, -(-(window[0] - 12) // 48))
                high = min(high, (window[1] - 12) // 48)
            allowed = [
                k
                for k in range(low, high + 1)
                if not window or 48 * k + 12 not in s["blocked"]
            ]
            v = m.new_int_var_from_domain(
                cp_model.Domain.from_values(allowed), f"col_{i}"
            )
            colvar[i] = v
            col[i] = 48 * v + 12
            colbounds[i] = (48 * min(allowed) + 12, 48 * max(allowed) + 12)
            m.add_hint(v, (old - 12) // 48)
        else:
            col[i] = old
            colbounds[i] = (old, old)

    # M5 occupancy, including same-track clearance.
    events = []
    for i in s["routed"]:
        lo, hi = s["column_spans"][i]
        events.extend(((lo, 1, i), (hi + 48, -1, i)))
    active = set()
    added = False
    for y, d, i in sorted(events):
        if d < 0:
            if added and any(k in colvar for k in active):
                m.add_all_different(
                    [
                        colvar[k] if k in colvar else (col[k] - 12) // 48
                        for k in sorted(active)
                    ]
                )
            active.remove(i)
            added = False
        else:
            active.add(i)
            added = True
    ordered = sorted(s["routed"], key=lambda i: colbounds[i][0])
    for pos, i in enumerate(ordered):
        ilo, ihi = s["column_spans"][i]
        ni = s["groups"][i][0]
        for j in ordered[pos + 1 :]:
            if colbounds[j][0] > colbounds[i][1] + 48:
                break
            if i not in colvar and j not in colvar:
                continue
            jlo, jhi = s["column_spans"][j]
            adjacent = (
                ihi == jlo
                or ilo == jhi
                or (ni == s["groups"][j][0] and ilo < jhi and jlo < ihi)
                or (
                    s["kinds"].get(i) is not None
                    and s["kinds"].get(j) is not None
                    and s["kinds"][i] != s["kinds"][j]
                )
            )
            if adjacent:
                m.add(col[i] != col[j] + 48)
                m.add(col[i] != col[j] - 48)

    # M4 intervals are functions of both cell and column coordinates.
    intervals = defaultdict(list)
    initial_events = defaultdict(list)
    for i, (net, ps) in enumerate(s["groups"]):
        for r in sorted({r for r, x, y in ps}):
            rowpoints = [
                (expr, lo, hi) for rr, expr, lo, hi, y in pointdata[i] if rr == r
            ]
            if i in col:
                rowpoints.append((col[i], *colbounds[i]))
            low = min(lo for _, lo, _ in rowpoints)
            high = max(hi for _, _, hi in rowpoints)
            if high + 96 < left or low - 96 > right:
                continue
            mn = extremum(
                [v for v, _, _ in rowpoints],
                min(lo for _, lo, _ in rowpoints),
                min(hi for _, _, hi in rowpoints),
                f"min_{i}_{r}",
                False,
            )
            mx = extremum(
                [v for v, _, _ in rowpoints],
                max(lo for _, lo, _ in rowpoints),
                max(hi for _, _, hi in rowpoints),
                f"max_{i}_{r}",
                True,
            )
            if isinstance(mn, int) and isinstance(mx, int):
                a = (mn - 24) // 48
                b = -(-(mx + 24) // 48)
            else:
                a = variable((low - 24) // 48, (high - 24) // 48, f"a_{i}_{r}")
                b = variable(-(-(low + 24) // 48), -(-(high + 24) // 48), f"b_{i}_{r}")
                m.add(48 * a <= mn - 24)
                m.add(48 * a >= mn - 71)
                m.add(48 * b >= mx + 24)
                m.add(48 * b <= mx + 71)
            cliplo = left // 48
            cliphi = -(-right // 48)
            original_xs = [x for rr, x, _y in ps if rr == r]
            if s["columns"][i] is not None:
                original_xs.append(s["columns"][i])
            original_lo = min(cliphi, max(cliplo, (min(original_xs) - 24) // 48))
            original_hi = max(
                original_lo, min(cliphi, -(-(max(original_xs) + 24) // 48) + 1)
            )
            if original_lo < original_hi:
                initial_events[r].extend(((original_lo, 1), (original_hi, -1)))
            a = extremum(
                [a, cliplo],
                min((low - 24) // 48, cliplo),
                max((high - 24) // 48, cliplo),
                f"ac1_{i}_{r}",
                True,
            )
            a = extremum([a, cliphi], cliplo, cliphi, f"ac2_{i}_{r}", False)
            end = extremum(
                [b + 1, cliphi],
                min(-(-(low + 24) // 48) + 1, cliphi),
                max(-(-(high + 24) // 48) + 1, cliphi),
                f"bc1_{i}_{r}",
                False,
            )
            end = extremum([end, a], cliplo, cliphi, f"bc2_{i}_{r}", True)
            b = end - 1
            length = b - a + 1
            size = (
                length
                if isinstance(length, int)
                else variable(0, cliphi - cliplo, f"sz_{i}_{r}")
            )
            if not isinstance(length, int):
                m.add(size == length)
            intervals[r].append(m.new_interval_var(a, size, b + 1, f"m4_{i}_{r}"))
    caps = {}
    for r, ins in intervals.items():
        cap = variable(8, max(8, len(ins)), f"capacity_{r}")
        caps[r] = cap
        m.add_cumulative(ins, [1] * len(ins), cap)
        live, peak = 0, 8
        for _x, delta in sorted(initial_events[r]):
            live += delta
            peak = max(peak, live)
        m.add_hint(cap, peak)

    # M6 buses may change tracks within the existing M5 vertical spans.
    buses = []
    busvar = {}
    bx = []
    by = []
    busbounds = {}
    for net, p_ in zip(s["on_m6"], s["packed"]):
        if not p_:
            continue
        ids = s["by_group"][net]
        low = min(colbounds[i][0] for i in ids)
        high = max(colbounds[i][1] for i in ids)
        a = (
            extremum(
                [col[i] for i in ids],
                low,
                min(colbounds[i][1] for i in ids),
                f"bus_lo_{net}",
                False,
            )
            - 24
        )
        b = (
            extremum(
                [col[i] for i in ids],
                max(colbounds[i][0] for i in ids),
                high,
                f"bus_hi_{net}",
                True,
            )
            + 24
        )
        busbounds[net] = (
            (low - 24, min(colbounds[i][1] for i in ids) - 24),
            (max(colbounds[i][0] for i in ids) + 24, high + 24),
        )
        yl = max(s["column_spans"][i][0] + 27 for i in ids)
        yh = min(s["column_spans"][i][1] - 27 for i in ids)
        kl = -(-(yl - 16) // 64)
        kh = (yh - 16) // 64
        if any(i in colvar for i in ids):
            v = variable(kl, kh, f"bus_y_{net}")
            m.add_hint(v, (p_[0] - 16) // 64)
        else:
            v = (p_[0] - 16) // 64
            kl = kh = v
        busvar[net] = v
        length = variable(1, high - low + 96, f"bus_size_{net}")
        m.add(length == b + 48 - a)
        bx.append(m.new_interval_var(a, length, b + 48, f"bx_{net}"))
        by.append(m.new_fixed_size_interval_var(v, 1, f"by_{net}"))
        buses.append((net, v, a, b, low - 24, high + 24, kl, kh))
    m.add_no_overlap_2d(bx, by)
    for i, (n, y, a, b, lo, hi, yl, yh) in enumerate(buses):
        for n2, y2, a2, b2, lo2, hi2, yl2, yh2 in buses[:i]:
            if hi < lo2 or hi2 < lo or yh + 1 < yl2 or yh2 + 1 < yl:
                continue
            for tip, start, (tl, th), (sl, sh) in (
                (b, a2, busbounds[n][1], busbounds[n2][0]),
                (b2, a, busbounds[n2][1], busbounds[n][0]),
            ):
                if th < sl or sh < tl:
                    continue
                if isinstance(tip, int) and isinstance(start, int):
                    if tip == start:
                        m.add(y != y2 + 1)
                        m.add(y != y2 - 1)
                else:
                    meet = m.new_bool_var(f"tip_{i}_{n2}")
                    m.add(tip == start).only_enforce_if(meet)
                    m.add(tip != start).only_enforce_if(meet.Not())
                    m.add(y != y2 + 1).only_enforce_if(meet)
                    m.add(y != y2 - 1).only_enforce_if(meet)

    m.minimize(sum(caps.values()))

    seed = m.clone()
    for index, value in zip(m.proto.solution_hint.vars, m.proto.solution_hint.values):
        seed.add(seed.get_int_var_from_proto_index(index) == value)
    seeder = cp_model.CpSolver()
    seeder.parameters.num_search_workers = 1
    seeder.parameters.random_seed = 0
    seeder.parameters.max_deterministic_time = 2
    seeder.parameters.max_time_in_seconds = min(15, seconds / 4)
    seed_status = seeder.solve(seed)
    if seed_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        m.clear_hints()
        for i in range(len(m.proto.variables)):
            m.add_hint(
                m.get_int_var_from_proto_index(i),
                seeder.value(seed.get_int_var_from_proto_index(i)),
            )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.linearization_level = 0
    solver.parameters.max_deterministic_time = 20
    solver.parameters.max_time_in_seconds = seconds
    status = solver.solve(m)
    if status == cp_model.MODEL_INVALID:
        raise ValueError(f"invalid joint routing model: {m.validate()}")
    report.update(
        solver_status=solver.status_name(status),
        capacity_floor=8 * len(caps),
        lower_bound=ceil(solver.best_objective_bound),
        movable_cells=len(moving),
        movable_columns=len(colvar),
        optimal=status == cp_model.OPTIMAL,
    )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return p, s, report
    report["capacity_sum"] = round(solver.objective_value)
    new_packed = [
        [(u, solver.value(origins[r, i])) for i, (u, _) in enumerate(row)]
        for r, row in enumerate(p)
    ]
    updated = deepcopy(s)
    updated["row_width"] = row_width
    updated["groups"] = [
        (
            net,
            tuple((r, solver.value(expr), y) for r, expr, _lo, _hi, y in pointdata[i]),
        )
        for i, (net, _) in enumerate(s["groups"])
    ]
    updated["columns"] = [
        solver.value(col[i]) if i in col else None for i in range(len(s["groups"]))
    ]
    updated["packed"] = [
        (64 * solver.value(busvar[n]) + 16, old[1]) if old else old
        for n, old in zip(s["on_m6"], s["packed"])
    ]
    updated["demands"] = [
        (
            min(updated["columns"][i] for i in s["by_group"][n]) - 24,
            max(updated["columns"][i] for i in s["by_group"][n]) + 24,
            old[2],
        )
        for n, old in zip(s["on_m6"], s["demands"])
    ]
    updated["targets"] = {
        i: sorted(x for _r, x, _y in updated["groups"][i][1])[
            len(updated["groups"][i][1]) // 2
        ]
        for i in s["routed"]
    }
    return new_packed, updated, report
