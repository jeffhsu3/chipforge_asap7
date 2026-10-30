"""CP-SAT channel compaction for ASAP7 peripheral functional units.

Port the GT2N constrained-routing approach, not its technology geometry.
The optimization is exact for fixed terminal intervals, not a transistor-area
minimum. Keep functional placement groups and all device dimensions intact.
"""

from functools import lru_cache


def compact_periphery(circuit):
    """Select compact routing before sizing; return the same circuit."""
    if circuit.compact_periphery:
        return circuit
    if circuit.distribution is not None:
        raise ValueError("select compact periphery before control sizing")
    circuit.compact_periphery = True
    # Distinct physical tops and timing identities, even with identical devices.
    circuit.name += "_cp"
    return circuit


@lru_cache(maxsize=4096)
def assign_tracks(intervals, end_space=0):
    """Return optimal track indices and a clique certificate.

    Intervals include the metal's right-end spacing keepout. Touching keepout
    boundaries are legal. Interval-graph greedy coloring supplies an incumbent;
    a simultaneously occupied clique is an independent lower bound. Physical
    touching tips additionally require a two-track separation.
    CP-SAT enforces every conflict and certifies the objective, deterministically.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise ImportError(
            "compact BROM routing requires chipforge-asap7[optimize]"
        ) from exc

    if not intervals or any(a >= b for a, b in intervals):
        raise ValueError("expected nonempty positive-length route intervals")

    def clearance(i, j):
        lo, hi = intervals[i]
        a, b = intervals[j]
        if end_space and (hi - end_space == a or b - end_space == lo):
            # Real tips align in x: adjacent tracks have only 24 nm diagonal
            # clearance, below M4.S.3's 40 nm. Leave one empty track.
            return 2
        return int(lo < b and a < hi)

    hint = {}
    for i in sorted(range(len(intervals)), key=lambda i: (intervals[i], i)):
        hint[i] = next(
            t
            for t in range(2 * len(intervals))
            if all(abs(t - old) >= clearance(i, j) for j, old in hint.items())
        )
    upper = max(hint.values()) + 1
    witness = max(
        (
            tuple(j for j, (lo, hi) in enumerate(intervals) if lo <= a < hi)
            for a, _ in intervals
        ),
        key=len,
    )
    model = cp_model.CpModel()
    tracks = [model.new_int_var(0, upper - 1, f"t{i}") for i in range(len(intervals))]
    for i in range(len(intervals)):
        for j in range(i):
            gap = clearance(i, j)
            if gap == 1:
                model.add(tracks[i] != tracks[j])
            elif gap == 2:
                distance = model.new_int_var(2, upper, f"distance_{i}_{j}")
                model.add_abs_equality(distance, tracks[i] - tracks[j])
        model.add_hint(tracks[i], hint[i])
    count = model.new_int_var(len(witness), upper, "track_count")
    model.add_max_equality(count, [t + 1 for t in tracks])
    model.minimize(count)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = 5
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError(
            f"peripheral channel optimization: {solver.status_name(status)}"
        )
    return tuple(solver.value(t) for t in tracks), witness
