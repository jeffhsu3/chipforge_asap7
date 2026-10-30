"""Geometry search for the transposed NOR-ROM, with an optional sizing model.

The shared optimizer knows only states and costs. This adapter keeps PA gates
in their wordline pair, fills both rows of every predecoder column, and scores
the actual ASAP7 M3/M4 routing plan. No GDS library or electrical simulator is
needed to search. Search feasibility does not imply DRC or timing signoff.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from ..optimization import bounded_search, pareto_front
from .rom_decoder import RomDecoderSpec
from .rom_predecoder import Placement, network
from .rom_sizing import RomDecoderSizing, _wire_lengths, size_rom_decoder

__all__ = [
    "RomCandidate",
    "RomMetrics",
    "RomPackingResult",
    "RomRejectedShape",
    "RomShapeSearch",
    "optimize_rom_packing",
    "rom_metrics",
    "search_rom_shapes",
]

# One ordered list per pair. Parallel fill copies are materialized separately.
PlacementState = tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class RomMetrics:
    """Array + decoder envelope and estimated M3/M4 predecode wire length.

    The envelope includes row-pair routing space and end cells. Sensing,
    precharge and output circuitry are not part of this ROM generator yet.
    Wire length follows the sizing model, including merged shared stubs.
    """

    width_nm: int
    height_nm: int
    wire_length_um: float

    @property
    def area_um2(self) -> float:
        return self.width_nm * self.height_nm / 1e6

    @property
    def cost(self) -> tuple[float, float]:
        return self.area_um2, self.wire_length_um


def rom_metrics(spec: RomDecoderSpec) -> RomMetrics:
    lo, hi = spec.rows_span
    return RomMetrics(hi - lo, spec.y0 + spec.height, sum(_wire_lengths(spec).values()))


@dataclass(frozen=True)
class RomCandidate:
    spec: RomDecoderSpec
    metrics: RomMetrics
    sizing: RomDecoderSizing | None = None
    packing_evaluations: int = 0

    def report(self) -> dict[str, Any]:
        return {
            "cell": self.spec.cell_name,
            "parameters": asdict(self.spec),
            "envelope": {**asdict(self.metrics), "area_um2": self.metrics.area_um2},
            "packing_evaluations": self.packing_evaluations,
            "gate_delay_estimate_ps": self.sizing.gate_delay_ps
            if self.sizing
            else None,
            "wordline_load_fF": self.sizing.wl_load_fF if self.sizing else None,
        }


def _evaluate(spec: RomDecoderSpec, sizing: Mapping[str, Any] | None) -> RomCandidate:
    # Re-size every placement: moving gates changes wire loads and fill copies.
    sized = size_rom_decoder(spec, **sizing) if sizing is not None else None
    spec = sized.spec if sized else spec
    return RomCandidate(spec, rom_metrics(spec), sized)


@dataclass(frozen=True)
class RomPackingResult:
    initial: RomCandidate
    selected: RomCandidate
    evaluations: int

    @property
    def spec(self) -> RomDecoderSpec:
        return self.selected.spec


def optimize_rom_packing(
    spec: RomDecoderSpec,
    *,
    evaluations: int = 64,
    seed: int = 0,
    sizing: Mapping[str, Any] | None = None,
) -> RomPackingResult:
    """Search gate order and pair assignment using routed area, then wire length.

    PA gates stay in their required pair. Other gates may move between pairs;
    odd counts get a parallel copy as in the original placer. The existing
    placement is the incumbent, so a search cannot worsen its geometry cost.
    ``sizing=None`` preserves explicit cell sizes; a mapping (including ``{}``)
    supplies keyword arguments to ``size_rom_decoder`` for every trial.
    """
    if (
        isinstance(evaluations, bool)
        or not isinstance(evaluations, int)
        or evaluations < 1
    ):
        raise ValueError("evaluations must be a positive integer")
    baseline = _evaluate(spec, sizing)
    if not spec.predecode:
        return RomPackingResult(baseline, baseline, 1)
    gates = network(spec.address_bits, spec.pa_bits, spec.row_bits)
    fanout = Counter(line for gate in gates.values() for line in set(gate.ins))
    groups: list[list[str]] = [[] for _ in range(spec.pairs)]
    seen = set()
    for p in sorted(spec.predecoder, key=lambda p: (p.pair, p.column, p.row)):
        if p.gate.out not in seen:
            groups[p.pair].append(p.gate.out)
            seen.add(p.gate.out)
    initial = tuple(tuple(group) for group in groups)
    candidates: dict[PlacementState, RomCandidate] = {initial: baseline}

    def materialize(state: PlacementState) -> tuple[Placement, ...]:
        placements = []
        for pair, group in enumerate(state):
            names = list(group)
            if len(names) % 2:
                names.append(max(names, key=lambda name: fanout[name]))
            placements.extend(
                Placement(gates[name], pair, i % 2, i // 2)
                for i, name in enumerate(names)
            )
        return tuple(placements)

    def mutate(state: PlacementState, rng: random.Random) -> PlacementState:
        groups = [list(group) for group in state]
        movable = [
            (p, i)
            for p, group in enumerate(groups)
            for i, name in enumerate(group)
            if not name.startswith("PA<")
        ]
        reorder = [p for p, group in enumerate(groups) if len(group) > 1]
        if reorder and (not movable or rng.random() < 0.6):
            group = groups[rng.choice(reorder)]
            a, b = rng.sample(range(len(group)), 2)
            if rng.random() < 0.5:
                group[a], group[b] = group[b], group[a]
            else:
                group.insert(b, group.pop(a))
        elif movable:
            p, i = rng.choice(movable)
            name = groups[p].pop(i)
            destination = rng.randrange(len(groups))
            groups[destination].insert(
                rng.randrange(len(groups[destination]) + 1), name
            )
        return tuple(tuple(group) for group in groups)

    def score(state: PlacementState) -> tuple[float, float] | None:
        if state not in candidates:
            try:
                trial = replace(spec, predecoder_placement=materialize(state))
                candidates[state] = _evaluate(trial, sizing)
            except ValueError:
                return None
        return candidates[state].metrics.cost

    result = bounded_search(initial, mutate, score, evaluations=evaluations, seed=seed)
    selected = replace(candidates[result.state], packing_evaluations=result.evaluations)
    return RomPackingResult(baseline, selected, result.evaluations)


@dataclass(frozen=True)
class RomRejectedShape:
    interleave: int
    split: int | None
    strap_every: int
    reason: str


@dataclass(frozen=True)
class RomShapeSearch:
    candidates: tuple[RomCandidate, ...]
    rejected: tuple[RomRejectedShape, ...]
    seed: int
    packing_budget: int

    @property
    def best(self) -> RomCandidate:
        if not self.candidates:
            reasons = "; ".join(dict.fromkeys(r.reason for r in self.rejected))
            raise ValueError(f"no feasible ROM shapes: {reasons}")
        return min(self.candidates, key=lambda c: c.metrics.cost)

    @property
    def pareto(self) -> list[RomCandidate]:
        def cost(c: RomCandidate) -> tuple[float, ...]:
            return (
                (*c.metrics.cost, c.sizing.gate_delay_ps)
                if c.sizing
                else c.metrics.cost
            )

        return pareto_front(self.candidates, cost)

    def report(self) -> dict[str, Any]:
        return {
            "format": "chipforge_asap7.rom_shape_search.v1",
            "objective": "minimize array/decoder envelope area, then predecode wire length",
            "limits": [
                "Precharge, sensing and output circuitry are excluded.",
                "Gate-delay estimates exclude distributed wire resistance.",
                "Candidates require gdscheck and extracted electrical validation.",
            ],
            "seed": self.seed,
            "packing_budget_per_shape": self.packing_budget,
            "selected": self.best.spec.cell_name if self.candidates else None,
            "pareto": [c.spec.cell_name for c in self.pareto],
            "candidates": [c.report() for c in self.candidates],
            "rejected": [asdict(r) for r in self.rejected],
        }


def search_rom_shapes(
    spec: RomDecoderSpec,
    *,
    interleaves: Sequence[int] | None = None,
    splits: Sequence[int] | None = None,
    strap_intervals: Sequence[int] | None = None,
    packing_evaluations: int = 64,
    seed: int = 0,
    sizing: Mapping[str, Any] | None = None,
) -> RomShapeSearch:
    """Enumerate split/interleave/strap choices at fixed word count and width.

    Defaults retain the caller's strap interval and try its interleave plus
    8 and 16. An explicit ``spec.split`` restricts the split unless ``splits``
    is supplied. Every candidate is packed; a budget of 1 keeps its existing
    greedy placement. Invalid physical combinations are reported, not clipped.
    Banking and column muxing are not implemented by the NOR-ROM generator.
    """
    if (
        isinstance(packing_evaluations, bool)
        or not isinstance(packing_evaluations, int)
        or packing_evaluations < 1
    ):
        raise ValueError("packing_evaluations must be a positive integer")
    interleaves = tuple(
        dict.fromkeys(
            interleaves if interleaves is not None else (spec.interleave, 8, 16)
        )
    )
    straps = tuple(
        dict.fromkeys(
            strap_intervals
            if strap_intervals is not None
            else (spec.array.strap_every,)
        )
    )
    if splits is not None:
        splits = tuple(dict.fromkeys(splits))
    for name, values in (
        ("interleaves", interleaves),
        ("strap_intervals", straps),
        ("splits", splits),
    ):
        if values is not None and (
            not values
            or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values
            )
        ):
            raise ValueError(f"{name} must contain positive integers")
    candidates, rejected = [], []
    for interleave in interleaves:
        for strap in straps:
            try:
                shape = replace(
                    spec,
                    interleave=interleave,
                    split=None,
                    predecoder_placement=None,
                    array=replace(spec.array, strap_every=strap),
                )
            except ValueError as exc:
                rejected.append(RomRejectedShape(interleave, None, strap, str(exc)))
                continue
            choices = (
                splits
                if splits is not None
                else (
                    (spec.split,)
                    if spec.split is not None
                    else range(shape.row_bits, shape._max_pa_bits + 1)
                )
            )
            for split in choices:
                try:
                    trial = replace(shape, split=split)
                    if (interleave, split, strap) == (
                        spec.interleave,
                        spec.pa_bits,
                        spec.array.strap_every,
                    ):
                        trial = replace(
                            trial, predecoder_placement=spec.predecoder_placement
                        )
                    result = optimize_rom_packing(
                        trial, evaluations=packing_evaluations, seed=seed, sizing=sizing
                    )
                    candidates.append(result.selected)
                except ValueError as exc:
                    rejected.append(
                        RomRejectedShape(interleave, split, strap, str(exc))
                    )
    return RomShapeSearch(tuple(candidates), tuple(rejected), seed, packing_evaluations)
