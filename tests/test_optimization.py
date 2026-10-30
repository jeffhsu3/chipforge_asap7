"""Contracts used by physical planners: bounded, reproducible, non-regressing."""

import math

import pytest

from chipforge_asap7.optimization import bounded_search, pareto_front


def test_search_keeps_the_best_feasible_state_with_a_hard_budget():
    calls = []

    def score(state):
        calls.append(state)
        return (abs(state - 3),) if state >= 0 else None

    def mutate(state, rng):
        return state + rng.choice((-2, -1, 1, 2))

    first = bounded_search(0, mutate, score, evaluations=8, seed=4)
    assert first.cost <= (3,)
    assert first.state >= 0
    assert first.evaluations == len(set(calls)) == len(calls) <= 8
    calls.clear()
    assert bounded_search(0, mutate, score, evaluations=8, seed=4) == first


def test_duplicate_proposals_terminate_and_one_evaluation_keeps_the_baseline():
    assert (
        bounded_search(7, lambda s, r: s, lambda s: (s,), evaluations=20).evaluations
        == 1
    )
    result = bounded_search(7, lambda s, r: 0, lambda s: (s,), evaluations=1)
    assert (result.state, result.evaluations) == (7, 1)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_search_budgets_fail(budget):
    with pytest.raises(ValueError, match="positive integer"):
        bounded_search(0, lambda s, r: s, lambda s: (s,), evaluations=budget)


def test_infeasible_baseline_and_nonfinite_costs_fail():
    for cost in (None, (), (math.nan,), (math.inf,)):
        with pytest.raises(ValueError):
            bounded_search(0, lambda s, r: s, lambda s, cost=cost: cost)


def test_cost_dimensions_cannot_change_between_trials():
    with pytest.raises(ValueError, match="equal dimensions"):
        bounded_search(0, lambda s, r: 1, lambda s: (s,) if s == 0 else (s, s))


def test_pareto_keeps_tradeoffs_and_ties_without_reordering():
    items = [(2, 2), (1, 4), (3, 3), (2, 2), (4, 1)]
    assert pareto_front(items, key=lambda x: x) == [(2, 2), (1, 4), (2, 2), (4, 1)]
    assert pareto_front([], key=lambda x: x) == []
    with pytest.raises(ValueError):
        pareto_front([(1,), (1, 2)], key=lambda x: x)
