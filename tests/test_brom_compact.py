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


def test_interval_bound_and_endpoint_clearance():
    intervals = ((0, 48), (48, 96), (24, 72), (20, 80))
    tracks, witness = assign_tracks(intervals)
    assert max(tracks) + 1 == len(witness) == 3
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
    tracks, witness = assign_tracks(intervals, end_space)
    assert not calls
    assert max(tracks) + 1 == len(witness) == entry["clique"]
    assert compact._legal(intervals, end_space, tracks)


@pytest.mark.parametrize("entry", TRACK_PROBLEMS["above_bound"])
def test_fallback_proves_a_minimum_above_the_bound(entry, monkeypatch):
    calls = count_solver_calls(monkeypatch)
    intervals, end_space = problem(entry)
    tracks, witness = assign_tracks(intervals, end_space)
    assert len(calls) == 1
    assert len(witness) == entry["clique"]
    assert max(tracks) + 1 == entry["minimum"] > entry["clique"]
    assert compact._legal(intervals, end_space, tracks)


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
        reference = max(solve_tracks(intervals, end_space)[0]) + 1
        assert used <= reference
        # A fallback the solver then closes at the bound means the sweep fell short.
        assert intervals not in fallbacks or reference > len(witness)


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
