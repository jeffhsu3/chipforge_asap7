"""Small, technology-independent search helpers for physical planners.

The caller supplies immutable states, legal mutations and a cost function.
There are no device, GDS, routing-layer or electrical-model dependencies here.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

__all__ = ["SearchResult", "bounded_search", "pareto_front"]

State = TypeVar("State", bound=Hashable)
Item = TypeVar("Item")
Cost = tuple[float, ...]


@dataclass(frozen=True)
class SearchResult(Generic[State]):
    state: State
    cost: Cost
    evaluations: int


def bounded_search(
    initial: State,
    mutate: Callable[[State, random.Random], State],
    score: Callable[[State], Cost | None],
    *,
    evaluations: int = 64,
    seed: int = 0,
) -> SearchResult[State]:
    """Seeded local search; return an incumbent no worse than ``initial``.

    ``None`` rejects an infeasible state. The budget includes the initial
    evaluation and counts unique states, including rejected ones. Equal-cost
    moves permit exploration; restarts return to the incumbent or initial
    state. Duplicate proposals cannot make the search run without a bound.
    """
    if (
        isinstance(evaluations, bool)
        or not isinstance(evaluations, int)
        or evaluations < 1
    ):
        raise ValueError("evaluations must be a positive integer")
    rng = random.Random(seed)
    cache: dict[State, Cost | None] = {}
    dimensions: int | None = None

    def evaluate(state: State) -> Cost | None:
        nonlocal dimensions
        if state not in cache:
            cost = score(state)
            if cost is not None and (
                not cost or not all(math.isfinite(v) for v in cost)
            ):
                raise ValueError("search costs must be nonempty and finite")
            if cost is not None:
                if dimensions is not None and len(cost) != dimensions:
                    raise ValueError("search costs must have equal dimensions")
                dimensions = len(cost)
            cache[state] = cost
        return cache[state]

    best, current = initial, initial
    best_cost = current_cost = evaluate(initial)
    if best_cost is None:
        raise ValueError("the initial search state is infeasible")
    for _ in range(8 * evaluations):
        if len(cache) >= evaluations:
            break
        if rng.random() < 0.2:
            current = initial if rng.random() < 0.5 else best
            current_cost = cache[current]
        proposal = mutate(current, rng)
        cost = evaluate(proposal)
        if cost is None:
            continue
        if cost < best_cost:
            best, best_cost = proposal, cost
        if cost <= current_cost:
            current, current_cost = proposal, cost
    return SearchResult(best, best_cost, len(cache))


def pareto_front(items: Iterable[Item], key: Callable[[Item], Cost]) -> list[Item]:
    """Nondominated items, lower costs preferred; preserve ties and input order."""
    entries = [(item, key(item)) for item in items]
    dimensions = {len(cost) for _, cost in entries}
    if (
        len(dimensions) > 1
        or 0 in dimensions
        or any(not all(math.isfinite(v) for v in cost) for _, cost in entries)
    ):
        raise ValueError(
            "Pareto costs must have equal, nonzero dimensions and be finite"
        )
    return [
        item
        for item, cost in entries
        if not any(
            all(a <= b for a, b in zip(other, cost))
            and any(a < b for a, b in zip(other, cost))
            for _, other in entries
        )
    ]
