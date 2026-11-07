"""Channel track assignment for ASAP7 peripheral functional units.

Port the GT2N constrained-routing approach, not its technology geometry.
The result is a track minimum for fixed terminal intervals, unless CP-SAT's
budget runs out; it is not a transistor-area minimum. Keep functional placement
groups and all device dimensions intact.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from random import Random


def compact_periphery(circuit):
    """Deprecated compatibility helper: dense CMOS is now the default."""
    import warnings

    warnings.warn(
        "compact_periphery is deprecated; dense CMOS is the default",
        DeprecationWarning,
        stacklevel=2,
    )
    return circuit


#: Sweeps to try for the clique bound before falling back to CP-SAT. A stuck
#: sweep stops early, so a problem whose bound is out of reach costs little.
SWEEP_TRIES = 200


@dataclass(frozen=True)
class TrackAssignment:
    """Legal tracks and the evidence bounding their minimum channel height.

    Preserve the original ``(tracks, witness)`` sequence interface. The solver
    status is None for the fast paths; ``optimal`` also recognizes their clique
    proofs and a greedy fallback that meets a solver-proved lower bound.
    """

    tracks: tuple[int, ...]
    witness: tuple[int, ...]
    lower_bound: int
    method: str
    solver_status: str | None = None

    def __iter__(self):
        return iter((self.tracks, self.witness))

    def __getitem__(self, key):
        return (self.tracks, self.witness)[key]

    def __len__(self):
        return 2

    @property
    def track_count(self):
        return max(self.tracks) + 1

    @property
    def optimal(self):
        return self.track_count == self.lower_bound

    def certificate(self, intervals, end_space):
        return {
            "intervals_nm": intervals,
            "end_space_nm": end_space,
            "tracks": self.tracks,
            "clique_witness": self.witness,
            "track_count": self.track_count,
            "lower_bound": self.lower_bound,
            "optimal": self.optimal,
            "method": self.method,
            "solver_status": self.solver_status,
        }


def _clearance(intervals, end_space, i, j):
    """Tracks intervals i and j must keep apart: 0, 1, or 2 for touching tips."""
    lo, hi = intervals[i]
    a, b = intervals[j]
    if end_space and (hi - end_space == a or b - end_space == lo):
        # Real tips align in x: adjacent tracks have only 24 nm diagonal
        # clearance, below M4.S.3's 40 nm. Leave one empty track.
        return 2
    return int(lo < b and a < hi)


def _legal(intervals, end_space, tracks):
    return all(
        abs(tracks[i] - tracks[j]) >= _clearance(intervals, end_space, i, j)
        for i in range(len(intervals))
        for j in range(i)
    )


def _start_order(intervals):
    return sorted(range(len(intervals)), key=lambda i: (intervals[i], i))


def _greedy(intervals, end_space):
    """First-fit tracks in start order.

    A track is free when its last interval ends before this one starts and no
    real tip on it or the next tracks aligns with this one's; `_clearance`
    states the same rules pairwise.
    """
    hint, last_end, starts, tips = {}, {}, {}, {}
    for i in _start_order(intervals):
        lo, hi = intervals[i]
        for t in range(2 * len(intervals)):
            if lo < last_end.get(t, lo):
                continue
            if end_space and any(
                hi - end_space in starts.get(k, ()) or lo in tips.get(k, ())
                for k in (t - 1, t, t + 1)
            ):
                continue
            hint[i] = t
            last_end[t] = max(last_end.get(t, hi), hi)
            starts.setdefault(t, set()).add(lo)
            tips.setdefault(t, set()).add(hi - end_space)
            break
    return [hint[i] for i in range(len(intervals))]


def _clique(intervals):
    """The most intervals live at one x, a lower bound on the track count."""
    return max(
        (
            tuple(j for j, (lo, hi) in enumerate(intervals) if lo <= a < hi)
            for a, _ in intervals
        ),
        key=len,
    )


def _sweep(intervals, end_space, count, rng):
    """One pass in start order into ``count`` tracks; None if it gets stuck.

    Mostly takes the free track that filled up last, which keeps the longer
    gaps open, and otherwise a random one so that retries differ.
    """
    last_end = [None] * count
    tracks, tips = [None] * len(intervals), {}
    for i in _start_order(intervals):
        lo, hi = intervals[i]
        banned = {x for t in tips.get(lo, ()) for x in (t - 1, t, t + 1)}
        free = [
            t
            for t in range(count)
            if t not in banned and (last_end[t] is None or last_end[t] <= lo)
        ]
        if not free:
            return None
        rng.shuffle(free)
        if rng.random() < 0.8:
            free.sort(key=lambda t: -1 if last_end[t] is None else last_end[t])
            tracks[i] = free[-1]
        else:
            tracks[i] = free[0]
        last_end[tracks[i]] = hi
        tips.setdefault(hi - end_space, []).append(tracks[i])
    return tracks


def solve_tracks(intervals, end_space=0):
    """Fewest tracks by CP-SAT, seeded with the greedy result.

    Return a TrackAssignment. If the budget runs out before an incumbent is
    found, retain the validated greedy route and any solver lower bound.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise ImportError(
            "compact BROM routing requires chipforge-asap7[optimize]"
        ) from exc

    hint = _greedy(intervals, end_space)
    upper = max(hint) + 1
    witness = _clique(intervals)
    model = cp_model.CpModel()
    tracks = [model.new_int_var(0, upper - 1, f"t{i}") for i in range(len(intervals))]
    for i in range(len(intervals)):
        for j in range(i):
            gap = _clearance(intervals, end_space, i, j)
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
    # A deterministic-time budget keeps the result independent of machine
    # load; a feasible assignment within it is still a legal channel.
    solver.parameters.max_deterministic_time = 30
    solver.parameters.max_time_in_seconds = 600
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE, cp_model.UNKNOWN):
        raise RuntimeError(
            f"peripheral channel optimization: {solver.status_name(status)}"
        )
    lower_bound = max(len(witness), ceil(solver.best_objective_bound))
    if status == cp_model.UNKNOWN:
        if not _legal(intervals, end_space, hint):
            raise RuntimeError("peripheral channel optimization: illegal greedy route")
        return TrackAssignment(
            tuple(hint), witness, lower_bound, "greedy", solver.status_name(status)
        )
    return TrackAssignment(
        tuple(solver.value(t) for t in tracks),
        witness,
        lower_bound,
        "cp_sat",
        solver.status_name(status),
    )


@lru_cache(maxsize=4096)
def assign_tracks(intervals, end_space=0):
    """Return a TrackAssignment with track indices and optimality evidence.

    Intervals include the metal's right-end spacing keepout. Touching keepout
    boundaries are legal. Physical touching tips additionally require a
    two-track separation.

    Any assignment that uses as many tracks as the clique has intervals is
    optimal, whatever produced it. So try greedy first-fit, then seeded sweeps
    for that bound. CP-SAT only runs when both miss, since the tip rule can put
    the true minimum above the bound and only a solver can show that.
    """
    if not intervals or any(a >= b for a, b in intervals):
        raise ValueError("expected nonempty positive-length route intervals")
    witness = _clique(intervals)
    hint = _greedy(intervals, end_space)
    if max(hint) + 1 == len(witness):
        return TrackAssignment(tuple(hint), witness, len(witness), "greedy")
    rng = Random(0)
    for _ in range(SWEEP_TRIES):
        tracks = _sweep(intervals, end_space, len(witness), rng)
        if tracks is not None and _legal(intervals, end_space, tracks):
            return TrackAssignment(tuple(tracks), witness, len(witness), "sweep")
    return solve_tracks(intervals, end_space)
