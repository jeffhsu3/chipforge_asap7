"""Joint repair must move both a cell and its column, and respect search limits."""

from types import SimpleNamespace

import pytest

pytest.importorskip("ortools")

from chipforge_asap7.brom.dense_bank_router import BOTTOM, finish_route
from chipforge_asap7.brom.dense_joint import congestion, solve_window


@pytest.fixture
def crowded():
    # Eight fixed connections occupy a channel. A ninth connection's cell
    # and column must both move past the crowded interval to make it fit.
    unit = SimpleNamespace(
        name="buffer_0",
        ports=("P",),
        nets=(("P", "signal"),),
        vt="rvt",
        plan=SimpleNamespace(width=162, trunks=(("P", 27),), edges=((), ())),
    )
    packed = [[(unit, 1512)]]
    groups = [
        (f"fixed_{i}", ((0, 999 - 54 * i, 216), (0, 1809 + 54 * i, 216)))
        for i in range(8)
    ]
    groups.append(("signal", ((0, 1539, 216), (0, 2535, 216))))
    candidate = {
        "groups": groups,
        "columns": [None] * 8 + [1404],
        "column_spans": {8: (BOTTOM, 420)},
        "routed": [8],
        "by_group": {"signal": [8]},
        "targets": {8: 2037},
        "windows": {},
        "kinds": {},
        "blocked": (),
        "jog_columns": {},
        "jog_rows": {},
        "risers": {"signal": 8},
        "rising": set(),
        "on_m4": set(),
        "on_m6": [],
        "packed": [],
        "demands": [],
        "anchors": {},
        "spans": {},
        "bottom": {"signal"},
        "edges": set(),
        "pins": ("signal",),
        "width": 4752,
        "row_width": 4428,
        "limit": 4644,
        "row_count": 1,
        "header": 492,
    }
    return packed, candidate


def test_joint_repair_moves_cell_and_column_and_routes(crowded):
    packed, candidate = crowded
    before = finish_route(candidate)
    assert "overfull" in before
    assert congestion(candidate)[0] > 0
    placed, updated, report = solve_window(packed, candidate, 1080, 3500, seconds=5)
    assert report["solver_status"] == "OPTIMAL"
    assert report["capacity_sum"] == report["capacity_floor"] == report["lower_bound"]
    assert report["global_optimal"] is False
    assert placed[0][0][0] is packed[0][0][0]
    assert placed[0][0][1] > packed[0][0][1]
    assert placed[0][0][1] % 54 == 0
    assert updated["columns"][8] > candidate["columns"][8]
    assert updated["groups"][:8] == candidate["groups"][:8]
    assert congestion(updated)[0] == 0
    result = finish_route(updated)
    assert "overfull" not in result
    assert result["routing_certificates"][0]["track_count"] <= 8
    # Input candidates can be cached and reused by other placement searches.
    assert candidate["columns"][8] == 1404
    assert packed[0][0][1] == 1512


def test_fixed_cells_can_prove_excess_capacity_without_proving_macro_infeasible(
    crowded,
):
    packed, candidate = crowded
    placed, updated, report = solve_window(
        packed, candidate, 1080, 3500, move=0, seconds=5
    )
    assert report["solver_status"] == "OPTIMAL"
    assert report["capacity_sum"] == report["lower_bound"] > report["capacity_floor"]
    assert report["global_optimal"] is False
    assert placed[0][0][1] == packed[0][0][1]
    assert "overfull" in finish_route(updated)


def test_unknown_retains_original_candidate(crowded, monkeypatch):
    from ortools.sat.python import cp_model

    original = cp_model.CpSolver.solve

    def limited(self, model):
        self.parameters.max_deterministic_time = 1e-9
        return original(self, model)

    monkeypatch.setattr(cp_model.CpSolver, "solve", limited)
    packed, candidate = crowded
    placed, updated, report = solve_window(packed, candidate, 1080, 3500, seconds=5)
    assert report["solver_status"] == "UNKNOWN"
    assert report["optimal"] is False
    assert placed is packed
    assert updated is candidate
