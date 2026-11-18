"""Track assignment bounds, actual geometry, and unchanged peripheral topology."""

import importlib.util
import json
import random
from copy import deepcopy
from pathlib import Path

import pytest

from chipforge_asap7.brom import (
    BromSpec,
    compact,
    compact_periphery,
    dense_bank_router,
    self_timed_macro_circuit,
    size_control_distribution,
)
from chipforge_asap7.brom.compact import assign_tracks, solve_tracks
from chipforge_asap7.brom.dense_macro import dense_macro_layout

# Captured from real builds.
TRACK_PROBLEMS = json.loads(
    (Path(__file__).parent / "data" / "brom_track_problems.json").read_text()
)
# Fallbacks this large are where CP-SAT can spend 100 s, so fail on one.
LARGEST_FALLBACK = 100


def problem(entry):
    return tuple(tuple(i) for i in entry["intervals"]), entry["end_space"]


def count_solver_calls(monkeypatch):
    calls = []
    real = compact.solve_tracks

    def counted(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(compact, "solve_tracks", counted)
    compact.assign_tracks.cache_clear()
    return calls


@pytest.fixture(autouse=True)
def solver(external_tool):
    external_tool(
        importlib.util.find_spec("ortools") is not None, "OR-Tools is missing"
    )
    compact.assign_tracks.cache_clear()
    yield
    compact.assign_tracks.cache_clear()


def test_interval_bound_and_endpoint_clearance():
    intervals = ((0, 48), (48, 96), (24, 72), (20, 80))
    result = assign_tracks(intervals)
    tracks, witness = result
    assert max(tracks) + 1 == len(witness) == 3
    assert result.optimal and result.lower_bound == result.track_count == 3
    assert result.method == "greedy" and result.solver_status is None
    assert tracks[0] == tracks[1]  # precisely legal end spacing can reuse a track
    for i, (lo, hi) in enumerate(intervals):
        for j, (a, b) in enumerate(intervals[:i]):
            if lo < b and a < hi:
                assert tracks[i] != tracks[j]


def test_adjacent_tracks_cannot_have_touching_real_tips():
    # Drawn spans 0..48 and 48..96, including a 40 nm right keepout.
    tracks, _ = assign_tracks(((0, 88), (48, 136)), end_space=40)
    assert abs(tracks[0] - tracks[1]) == 2


@pytest.mark.parametrize("entry", TRACK_PROBLEMS["reach_bound"])
def test_sweep_reaches_the_bound_where_cp_sat_runs_out(entry, monkeypatch):
    calls = count_solver_calls(monkeypatch)
    intervals, end_space = problem(entry)
    result = assign_tracks(intervals, end_space)
    tracks, witness = result
    assert not calls
    assert result.optimal and result.lower_bound == entry["clique"]
    assert result.method == "sweep" and result.solver_status is None
    assert max(tracks) + 1 == len(witness) == entry["clique"]
    assert compact._legal(intervals, end_space, tracks)


@pytest.mark.parametrize("entry", TRACK_PROBLEMS["above_bound"])
def test_fallback_proves_a_minimum_above_the_bound(entry, monkeypatch):
    calls = count_solver_calls(monkeypatch)
    intervals, end_space = problem(entry)
    result = assign_tracks(intervals, end_space)
    tracks, witness = result
    assert len(calls) == 1
    assert result.method == "cp_sat" and result.solver_status == "OPTIMAL"
    assert result.optimal and result.lower_bound == entry["minimum"]
    assert len(witness) == entry["clique"]
    assert max(tracks) + 1 == entry["minimum"] > entry["clique"]
    assert compact._legal(intervals, end_space, tracks)


def exhaust_solver_budget(monkeypatch):
    from ortools.sat.python import cp_model

    real_solve = cp_model.CpSolver.solve

    def limited(self, model):
        self.parameters.max_deterministic_time = 1e-6
        return real_solve(self, model)

    def no_incumbent(*args):
        pytest.fail("UNKNOWN has no solver assignment to read")

    monkeypatch.setattr(cp_model.CpSolver, "solve", limited)
    monkeypatch.setattr(cp_model.CpSolver, "value", no_incumbent)


def test_unknown_keeps_legal_greedy_route(monkeypatch):
    exhaust_solver_budget(monkeypatch)
    intervals, end_space = problem(TRACK_PROBLEMS["above_bound"][0])
    result = assign_tracks(intervals, end_space)
    assert result.tracks == tuple(compact._greedy(intervals, end_space))
    assert compact._legal(intervals, end_space, result.tracks)
    assert result.solver_status == "UNKNOWN" and result.method == "greedy"
    assert result.lower_bound == len(result.witness) == 2
    assert result.track_count == 3 and not result.optimal


@pytest.mark.parametrize("bound,optimal", [(2, False), (3, True)])
def test_feasible_retains_solver_bound_and_proof_status(monkeypatch, bound, optimal):
    from ortools.sat.python import cp_model

    real_solve = cp_model.CpSolver.solve

    def feasible(self, model):
        assert real_solve(self, model) == cp_model.OPTIMAL
        return cp_model.FEASIBLE

    # Keep an actual legal solver assignment while exercising both unfinished
    # search and a closed bound, without a timing-dependent FEASIBLE fixture.
    monkeypatch.setattr(cp_model.CpSolver, "solve", feasible)
    monkeypatch.setattr(
        cp_model.CpSolver, "best_objective_bound", property(lambda self: float(bound))
    )
    intervals, end_space = problem(TRACK_PROBLEMS["above_bound"][0])
    result = solve_tracks(intervals, end_space)
    assert compact._legal(intervals, end_space, result.tracks)
    assert result.track_count == 3 and result.lower_bound == bound
    assert result.solver_status == "FEASIBLE" and result.method == "cp_sat"
    assert result.optimal is optimal


@pytest.mark.parametrize("status", ["INFEASIBLE", "MODEL_INVALID"])
def test_solver_errors_do_not_use_greedy_fallback(monkeypatch, status):
    from ortools.sat.python import cp_model

    monkeypatch.setattr(
        cp_model.CpSolver, "solve", lambda self, model: getattr(cp_model, status)
    )
    with pytest.raises(RuntimeError, match=status):
        solve_tracks(*problem(TRACK_PROBLEMS["above_bound"][0]))


def test_header_retains_unknown_certificate(monkeypatch):
    exhaust_solver_budget(monkeypatch)
    intervals, end_space = problem(TRACK_PROBLEMS["above_bound"][0])
    plan = {
        "port_spans": {
            f"n{i}": (a, b - end_space) for i, (a, b) in enumerate(intervals)
        },
        "m4_buses": {},
        "header": 492,
        "buses": (),
        "column_shapes": (),
    }
    tracks, height, certificate = dense_bank_router.route_header(plan)
    assert certificate["solver_status"] == "UNKNOWN"
    assert certificate["method"] == "greedy" and not certificate["optimal"]
    assert certificate["lower_bound"] == 2 and certificate["track_count"] == 3
    assert certificate["intervals_nm"] == intervals
    assert certificate["end_space_nm"] == end_space
    assert height >= max(y for y, _, _ in tracks.values()) + 48
    for net, track, (a, b) in zip(
        certificate["nets"], certificate["tracks"], certificate["intervals_nm"]
    ):
        assert tracks[net] == (492 + 48 * track, a, b - end_space)


def test_proof_metadata_does_not_change_physical_identity(monkeypatch):
    from chipforge_asap7.brom import dense_macro

    spec = BromSpec(4, 2, registered_output=True)
    words = [0, 1, 2, 3]
    _, before = dense_macro_layout(self_timed_macro_circuit(spec, words))
    real_build = dense_macro.build_dense_macro

    def different_evidence(circuit):
        top, meta = real_build(circuit)
        # A solve can prove the same route with a different method/status.
        # Copy the certificates to leave cached routing plans intact.
        meta["routing_certificates"] = [
            {**cert, "method": "cp_sat", "solver_status": "OPTIMAL"}
            for cert in meta["routing_certificates"]
        ]
        meta["joint_optimization"] = [
            {"solver_status": "OPTIMAL", "global_optimal": False}
        ]
        return top, meta

    monkeypatch.setattr(dense_macro, "build_dense_macro", different_evidence)
    _, after = dense_macro_layout(self_timed_macro_circuit(spec, words))
    assert before["routing_certificates"] != after["routing_certificates"]
    assert before["joint_optimization"] != after["joint_optimization"]
    assert before["geometry_sha256"] == after["geometry_sha256"]
    assert before["physical_sha256"] == after["physical_sha256"]


def test_sweep_is_repeatable():
    intervals, end_space = problem(TRACK_PROBLEMS["reach_bound"][0])
    count = TRACK_PROBLEMS["reach_bound"][0]["clique"]
    runs = [
        compact._sweep(intervals, end_space, count, random.Random(0)) for _ in range(2)
    ]
    assert runs[0] == runs[1]


def test_clique_is_the_most_intervals_at_one_x():
    assert len(compact._clique(((0, 10), (5, 15), (9, 20), (15, 30)))) == 3


@pytest.mark.slow
def test_tracks_match_cp_sat_on_a_real_build(monkeypatch):
    """Periodic check: every track problem in a build that CP-SAT finds hard."""
    seen = {}
    real = dense_bank_router.assign_tracks

    def capture(intervals, end_space=0):
        seen[intervals, end_space] = None
        return real(intervals, end_space)

    monkeypatch.setattr(dense_bank_router, "assign_tracks", capture)
    dense_bank_router.route_bank.cache_clear()
    calls = count_solver_calls(monkeypatch)
    rng = random.Random(1)
    words = [rng.getrandbits(30) for _ in range(64)]
    circuit = self_timed_macro_circuit(
        BromSpec(64, 30, 4, registered_output=True), words
    )
    dense_macro_layout(circuit)
    assert seen

    fallbacks = {call[0] for call in calls}
    assert all(len(intervals) <= LARGEST_FALLBACK for intervals in fallbacks)
    for intervals, end_space in seen:
        tracks, witness = compact.assign_tracks(intervals, end_space)
        assert compact._legal(intervals, end_space, tracks)
        used = max(tracks) + 1
        if used == max(compact._greedy(intervals, end_space)) + 1:
            continue  # nothing beat the greedy start, so there is no sweep to check
        reference = solve_tracks(intervals, end_space)
        assert used <= reference.track_count
        if reference.optimal:
            assert used == reference.track_count
        # A fallback the solver then closes at the bound means the sweep fell short.
        assert intervals not in fallbacks or reference.track_count > len(witness)


def test_compact_helper_is_deprecated_idempotent_noop():
    c = self_timed_macro_circuit(BromSpec(4, 4, registered_output=True), [10, 12, 0, 0])
    before = deepcopy(c)
    with pytest.warns(DeprecationWarning, match="dense CMOS"):
        assert compact_periphery(c) is c
    assert c.devices == before.devices and c.name == before.name
    size_control_distribution(c)
    with pytest.warns(DeprecationWarning):
        assert compact_periphery(c) is c


def test_cli_legacy_flag_reports_dense_geometry(tmp_path):
    from test_brom_cli import compile_words

    with pytest.warns(DeprecationWarning):
        _, report = compile_words(
            tmp_path, "--self-timed", "--registered-output", "--compact-periphery"
        )
    assert report["periphery_layout"] == "shared_diffusion_cmos"
    assert report["physical_sha256"]
    assert all(r["placement_certificates"] for r in report["artifacts"].values())
