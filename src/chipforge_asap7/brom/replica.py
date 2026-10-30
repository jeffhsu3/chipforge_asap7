"""Loaded selector replica and internal completion feedback for BROM.

The replica is a separate compact selector array, not a fixed-time behavioral
source. Geometry and intrinsic device loads track the requested macro shape.
Guard stages provide explicit transistor delay; their adequacy still requires
transient checks and, for physical timing, extracted interconnect qualification.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .circuit import Circuit, ReplicaInstance, SelectorInstance
from .compiler import BromSpec
from .controller import clocked_macro_circuit


@dataclass(frozen=True)
class ReplicaSpec:
    guard_pairs: int = 4
    extra_blocks: int = 1

    def __post_init__(self):
        for name in ("guard_pairs", "extra_blocks"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def add_replica(c: Circuit, spec: BromSpec, replica: ReplicaSpec) -> None:
    if c.array is None or c.replica is not None:
        raise ValueError(
            "replica requires a macro with one unreplicated selector array"
        )
    blocks = spec.blocks + replica.extra_blocks
    if blocks * c.array.plan.height + 81 > 30_000:
        raise ValueError(
            "replica exceeds single tap row reach; reduce depth or extra_blocks"
        )
    last = blocks - 1
    mapping = tuple(
        tuple(15 if b == last else 0 for _ in range(spec.columns))
        for b in range(blocks)
    )
    aliases = {
        f"C<{b},{k}>": f"RC<{b},{k}>" if b == last else "VSS"
        for b in range(blocks)
        for k in range(16)
    }
    aliases.update({f"BL<{col}>": f"RBL<{col}>" for col in range(spec.columns)})
    aliases["VSS"] = "VSS"

    def gate(kind, *nets):
        start = len(c.devices)
        getattr(c, kind)(*nets)
        c.group_since(f"replica_{start}", start)

    def delay(source, output, pairs):
        for stage in range(pairs):
            target = output if stage == pairs - 1 else c.node()
            gate("buffer", source, target)
            source = target

    # Match EVAL -> evaluate -> block SEL -> candidate, including the full
    # sixteen-AND SEL load. Every replica column selects candidate 15.
    evaluate, sel = c.node(), c.node()
    gate("and2", "EVAL", "PRECHN", evaluate)
    gate("and2", "VDD", evaluate, sel)
    for k in range(16):
        gate("and2", "VDD" if k == 15 else "VSS", sel, f"RC<{last},{k}>")
    devices = set()
    for b, row in enumerate(mapping):
        for col, k in enumerate(row):
            c.fet(False, f"RBL<{col}>", aliases[f"C<{b},{k}>"], "VSS")
            devices.add(c.devices[-1].name)
            c.fet(False, f"RBL<{col}>", "VSS", "VSS")
            devices.add(c.devices[-1].name)
    for col in range(spec.columns):
        start = len(c.devices)
        c.fet(True, f"RBL<{col}>", "PRECHN", "VDD")
        c.inv(f"RBL<{col}>", f"RDOUT<{col}>")
        c.group_since(f"replica_readout_{col}", start)
    data = f"RDOUT<{spec.columns - 1}>"
    if spec.ways == 4:
        terms = []
        for way in range(4):
            term = c.node()
            gate(
                "and2", data if way == 0 else "VSS", "VDD" if way == 0 else "VSS", term
            )
            terms.append(term)
        data = terms[0]
        for term in terms[1:]:
            out = c.node()
            gate("or2", data, term, out)
            data = out
    # Include a real 8T latch input load on the replicated readout/mux path.
    members = next(m for name, m in c.placement_groups if name == "output_latch_0")
    by_name = {d.name: d for d in c.devices}
    sampleb = by_name[members[1]].nodes[1]
    gate("latch", data, "RQ", "SAMPLE", sampleb)
    delay(data, "RVALID", replica.guard_pairs)
    gate("and2", "EVAL", "RVALID", "READ_READY")
    # Fast assertion, guarded release. The direct EVAL term covers startup
    # before the loaded candidate has risen; the candidate tail covers return.
    candidate = f"RC<{last},15>"
    tail = c.node()
    delay(candidate, tail, replica.guard_pairs)
    pending = "EVAL"
    for i, signal in enumerate((evaluate, sel, candidate, tail)):
        target = "CANDIDATE_BUSY" if i == 3 else c.node()
        gate("or2", pending, signal, target)
        pending = target
    # Place observers last before any optional distribution pass. Added
    # buffers can extend the bank beyond these taps. Wire delay is not
    # qualified merely by adding them; PEX must establish the receiving margins.
    delay("PRECHN", "PRECHN_FAR", 2)
    sample_low, closed = c.node(), c.node()
    gate("inv", "SAMPLE", sample_low)
    gate("and2", sample_low, sampleb, closed)
    delay(closed, "SAMPLE_CLOSED", 2)
    c.replica = ReplicaInstance(
        SelectorInstance(c.array.plan, mapping, frozenset(devices)),
        aliases,
        replica.guard_pairs,
        replica.extra_blocks,
    )


def self_timed_macro_circuit(
    spec: BromSpec, words: Sequence[int], *, replica: ReplicaSpec | None = None
) -> Circuit:
    """Experimental closed-loop macro; only address, CLK, RESETN, Q and supplies."""
    c = clocked_macro_circuit(spec, words)
    add_replica(c, spec, replica or ReplicaSpec())
    c.name = c.name.replace("brom_clocked_", "brom_selftimed_")
    c.name += f"_g{c.replica.guard_pairs}_e{c.replica.extra_blocks}"
    feedback = {"PRECHN_FAR", "READ_READY", "SAMPLE_CLOSED", "CANDIDATE_BUSY"}
    c.pins = tuple(pin for pin in c.pins if pin not in feedback)
    return c
