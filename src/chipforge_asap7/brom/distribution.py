"""Gate-load-sized control buffers; interconnect RC is not characterized."""

from collections import defaultdict
from dataclasses import dataclass, replace
from math import ceil, isfinite

from .circuit import Circuit


@dataclass(frozen=True)
class DistributionSpec:
    max_stage_effort: int = 4

    def __post_init__(self):
        if type(self.max_stage_effort) is not int or self.max_stage_effort < 2:
            raise ValueError("max_stage_effort must be an integer >= 2")


def size_control_distribution(
    c: Circuit, spec: DistributionSpec | None = None, *, wire_gate_fins=None
):
    """Insert noninverting tapered chains on overloaded gate-only nets.

    A unit inverter has the circuit's base n/p fin counts. Load is the sum of
    receiving gate fins, normalized to that inverter's input fins. This is a
    reproducible fanout heuristic, not a capacitance or slew characterization.
    Parallel single-finger devices preserve the qualified peripheral geometry.
    Call once, after the controller and replica have been added.
    """
    if c.distribution is not None:
        raise ValueError("control distribution has already been sized")
    spec = spec or DistributionSpec()
    wire_gate_fins = dict(wire_gate_fins or {})
    if any(not isfinite(v) or v < 0 for v in wire_gate_fins.values()):
        raise ValueError("wire gate-equivalent loads must be finite and nonnegative")
    snapshot = list(c.devices)
    gates, terminals = defaultdict(list), defaultdict(list)
    for d in snapshot:
        gates[d.nodes[1]].append(d)
        for net in {d.nodes[0], d.nodes[2], d.nodes[3]}:
            terminals[net].append(d)
    unit = c.n.fins + c.p.fins
    records = []
    for net in sorted(gates):
        if net in ("VDD", "VSS"):
            continue
        load = sum(d.spec.total_fins for d in gates[net])
        effective_load = load + wire_gate_fins.get(net, 0)
        drivers = terminals[net]
        external = net in ("PRECHN", "EVAL", "SAMPLE") and not drivers
        # Only a complementary inverter output with no other diffusion loads
        # can be safely isolated. This excludes latch storage and data paths.
        inverter = (
            len(drivers) == 2
            and {d.spec.flavor for d in drivers} == {"n", "p"}
            and len({d.nodes[1] for d in drivers}) == 1
            and all(
                d.nodes[0] == net
                and d.nodes[2] == d.nodes[3]
                and d.nodes[2] in ("VDD", "VSS")
                for d in drivers
            )
        )
        if not (external or inverter):
            continue
        strength = unit if external else sum(d.spec.total_fins for d in drivers)
        if effective_load <= spec.max_stage_effort * strength:
            continue
        stages = [ceil(effective_load / (unit * spec.max_stage_effort))]
        while stages[0] * unit > spec.max_stage_effort * strength:
            stages.insert(0, ceil(stages[0] / spec.max_stage_effort))
        # Even inversion count preserves the controller and candidate polarity.
        if len(stages) % 2:
            stages.insert(0, ceil(stages[0] / spec.max_stage_effort))
        source, receiver = net, c.node() if external else net
        if external:
            names = {d.name for d in gates[net]}
            c.devices = [
                replace(d, nodes=(d.nodes[0], receiver, *d.nodes[2:]))
                if d.name in names
                else d
                for d in c.devices
            ]
        else:
            source = c.node()
            names = {d.name for d in drivers}
            c.devices = [
                replace(d, nodes=(source, *d.nodes[1:])) if d.name in names else d
                for d in c.devices
            ]
        stage_records = []
        current = source
        for i, count in enumerate(stages):
            target = receiver if i == len(stages) - 1 else c.node()
            members = []
            for copy in range(count):
                start = len(c.devices)
                c.inv(current, target)
                c.group_since(f"distribution_{net}_{i}_{copy}", start)
                members.extend(d.name for d in c.devices[start:])
            stage_records.append(
                {
                    "input": current,
                    "output": target,
                    "parallel_inverters": count,
                    "output_load_gate_fins": (
                        load if i == len(stages) - 1 else stages[i + 1] * unit
                    ),
                    "stage_effort": (
                        load if i == len(stages) - 1 else stages[i + 1] * unit
                    )
                    / (count * unit),
                    "devices": members,
                }
            )
            current = target
        records.append(
            {
                "net": net,
                "receiver_net": receiver,
                "external_input": external,
                "load_gate_fins": load,
                "wire_gate_equivalent_fins": wire_gate_fins.get(net, 0),
                "original_driver_fins": strength,
                "original_stage_effort": load / strength,
                "sized_source_effort": stages[0] * unit / strength,
                "stages": stage_records,
            }
        )
    c.distribution = {
        "method": "gate_fin_fanout",
        "wire_rc_included": False,
        "timing_qualified": False,
        "max_stage_effort": spec.max_stage_effort,
        "unit_input_fins": unit,
        "nets": records,
    }
    c.name += f"_dist{spec.max_stage_effort}"
    return c
