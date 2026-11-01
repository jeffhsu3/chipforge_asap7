"""Routed-length RC estimates and bounded physical control-driver search.

This is not field-solver PEX. R uses the open stack (including its extrapolated
upper metals); C/length is an explicit assumption. Rpath*C is a deliberately
pessimistic routing cost, not an extracted delay or an operating frequency.
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite

from ..devices.sizing import LogicalEffortModel
from ..verification.stack import STACK_CONDUCTORS
from .distribution import size_control_distribution


@dataclass(frozen=True)
class WireRCSpec:
    wire_cap_ff_per_um: float = 0.2
    gate_cap_ff_per_fin: float = 0.0417
    max_rounds: int = 3
    max_parallel: int = 32
    max_added_devices: int = 4096

    def __post_init__(self):
        for key in ("wire_cap_ff_per_um", "gate_cap_ff_per_fin"):
            value = getattr(self, key)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("max_rounds", "max_parallel", "max_added_devices"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")


def routed_control_rc(c, model=None):
    """Dense M1-M6 centerline model, including local accesses and vias.

    Total net resistance bounds the receiving path; capacitance per length
    remains an explicit assumption. This is not extracted timing.
    """
    from .dense_rc import dense_control_rc

    return dense_control_rc(c, model or WireRCSpec())


def _units(c):
    groups = {
        member: members for _, members in c.placement_groups for member in members
    }
    by_name = {d.name: d for d in c.devices}
    seen, units = set(), []
    for d in c.devices:
        if d.name not in seen:
            members = groups.get(d.name, (d.name,))
            units.append([by_name[n] for n in members])
            seen.update(members)
    return units


def _place(c, mode):
    """Keep source stages by the driver and final stages by median sink group."""
    units = _units(c)
    added = {
        n
        for r in c.distribution["nets"]
        if not r.get("embedded_candidate")
        for s in r["stages"]
        for n in s["devices"]
    }
    fixed = [u for u in units if not any(d.name in added for d in u)]
    owner = {d.name: i for i, u in enumerate(fixed) for d in u}
    by_name = {d.name: d for d in c.devices}
    inserts = defaultdict(list)
    for r in c.distribution["nets"]:
        if r.get("embedded_candidate"):
            continue
        source = r["stages"][0]["input"]
        driver = [owner[d.name] for u in fixed for d in u if d.nodes[0] == source]
        sinks = sorted(
            owner[d.name] for u in fixed for d in u if d.nodes[1] == r["receiver_net"]
        )
        origin = min(driver, default=min(sinks, default=0))
        target = sinks[len(sinks) // 2] if sinks else origin
        for j, stage in enumerate(r["stages"]):
            at = origin if mode == "source" or j == 0 else target
            inserts[at].extend(by_name[n] for n in stage["devices"])
    c.devices = [d for i, u in enumerate(fixed) for d in [*inserts[i], *u]]


def _score(c, model, rc, nets):
    """Wire RC cost plus inverter drive cost, for a fixed original-net set."""
    records = {r["net"]: r for r in c.distribution["nets"]}
    unit_c = c.distribution["unit_input_fins"] * model.gate_cap_ff_per_fin
    costs = []
    for net in nets:
        record = records.get(net)
        stages = record["stages"] if record else []
        outputs = [(s["output"], s["parallel_inverters"]) for s in stages]
        outputs.insert(0, (stages[0]["input"] if stages else net, 1))
        cost = 0
        for output, count in outputs:
            if output in rc:
                v = rc[output]
                cost += v["wire_rc_cost_ps"] + LogicalEffortModel().tau_ps * (
                    (v["wire_cap_ff"] + v["gate_cap_ff"]) / (count * unit_c) + 0.59
                )
        costs.append(cost)
    return sum(costs)


def size_wire_rc_distribution(c, model=None):
    """Bounded sizing/placement search; commit the best circuit atomically.

    All candidates use unchanged real transistor topology except added buffers.
    The cost is an estimate; no fixed stage-effort or timing closure is claimed.
    """
    if c.array is None or c.distribution is not None:
        raise ValueError("wire-RC sizing needs an unsized macro")
    model = model or WireRCSpec()
    baseline = deepcopy(c)
    initial = size_control_distribution(deepcopy(baseline))
    if len(initial.devices) - len(baseline.devices) > model.max_added_devices or any(
        stage["parallel_inverters"] > model.max_parallel
        for record in initial.distribution["nets"]
        for stage in record["stages"]
    ):
        raise ValueError(
            "wire-RC resource limit is smaller than the gate-sized baseline"
        )
    # Probe eligible gate-only nets with bounded wire load so lightly loaded
    # but physically long candidate nets participate as well.
    raw = routed_control_rc(baseline, model)
    unit = c.n.fins + c.p.fins
    gate_sized = {r["net"] for r in initial.distribution["nets"]}
    controls = {
        n
        for n in raw
        if n in gate_sized
        or n in ("PRECHN", "EVAL", "SAMPLE")
        or n.startswith(("C<", "RC<", "SEL<", "G<", "BLOCK<", "CSEL<"))
    }

    def wire_load(net, rc):
        gate_fins = raw[net]["gate_cap_ff"] / model.gate_cap_ff_per_fin
        return min(
            rc["wire_cap_ff"] / model.gate_cap_ff_per_fin,
            max(0, model.max_parallel * unit * 4 - gate_fins),
        )

    overrides = {n: wire_load(n, raw[n]) for n in controls}
    probe = size_control_distribution(deepcopy(baseline), wire_gate_fins=overrides)
    nets = sorted(
        {r["net"] for r in probe.distribution["nets"]}
        | {r["net"] for r in initial.distribution["nets"]}
    )
    initial_rc = routed_control_rc(initial, model)
    best, best_rc = initial, initial_rc
    best_mode = "appended"
    best_score = _score(initial, model, initial_rc, nets)
    history = [
        {
            "round": 0,
            "placement": "appended",
            "score_ps": best_score,
            "devices": len(initial.devices),
        }
    ]
    best_loads = {}
    seen = set()
    for round_index in range(model.max_rounds + 1):
        choices = [{}] if round_index == 0 else []
        if round_index:
            receivers = {r["net"]: r["receiver_net"] for r in best.distribution["nets"]}
            ordered = sorted(
                nets,
                key=lambda n: (
                    -best_rc.get(receivers.get(n, n), raw[n])["wire_rc_cost_ps"],
                    n,
                ),
            )
            for net in ordered[:4]:
                full = wire_load(net, best_rc.get(receivers.get(net, net), raw[net]))
                for fraction in (0.1, 0.25, 1.0):
                    choices.append({**best_loads, net: fraction * full})
        prior = best_score
        for loads in choices:
            signature = tuple(sorted(loads.items()))
            if signature in seen:
                continue
            seen.add(signature)
            trial = size_control_distribution(deepcopy(baseline), wire_gate_fins=loads)
            if len(trial.devices) - len(
                baseline.devices
            ) > model.max_added_devices or any(
                stage["parallel_inverters"] > model.max_parallel
                for record in trial.distribution["nets"]
                for stage in record["stages"]
            ):
                history.append(
                    {"round": round_index + 1, "rejected": "device/parallel limit"}
                )
                continue
            for mode in ("source", "median"):
                candidate = deepcopy(trial)
                _place(candidate, mode)
                rc = routed_control_rc(candidate, model)
                score = _score(candidate, model, rc, nets)
                history.append(
                    {
                        "round": round_index + 1,
                        "placement": mode,
                        "score_ps": score,
                        "devices": len(candidate.devices),
                        "wire_load_overrides": loads,
                    }
                )
                if score < best_score:
                    best, best_rc, best_score = candidate, rc, score
                    best_mode, best_loads = mode, loads
        if round_index and best_score >= prior:
            break
    stage_audit = []
    for record in best.distribution["nets"]:
        outputs = [(record["stages"][0]["input"], record["original_driver_fins"])]
        outputs += [
            (stage["output"], stage["parallel_inverters"] * unit)
            for stage in record["stages"]
        ]
        for net, drive in outputs:
            rc = best_rc[net]
            effort = (rc["wire_cap_ff"] + rc["gate_cap_ff"]) / (
                drive * model.gate_cap_ff_per_fin
            )
            stage_audit.append(
                {
                    "net": net,
                    "estimated_stage_effort": effort,
                    "target_met": effort <= 4,
                }
            )
    best.name += "_wirerc"
    best.distribution.update(
        method="routed_length_rc_search",
        wire_rc_included=True,
        extracted=False,
        timing_qualified=False,
        model=asdict(model),
        search_history=history,
        initial_score_ps=history[0]["score_ps"],
        selected_score_ps=best_score,
        selected_placement=best_mode,
        objective_nets=nets,
        final_stage_audit=stage_audit,
        stage_effort_target_met=all(s["target_met"] for s in stage_audit),
        routed_rc=best_rc,
        scope="M1-M6 centerlines and via R; total-net R bound; no poly/coupling/via C",
        resistance_provenance={
            s.name: s.provenance
            for s in STACK_CONDUCTORS
            if s.name in ("M1", "M2", "M3", "M4", "M5", "M6")
        },
    )
    c.__dict__.update(best.__dict__)
    return c
