"""Semantic peripheral units with canonical, reusable dense cell definitions."""

from collections import Counter
from dataclasses import dataclass

from .dense_cell import build_dense_cell, cell_plan


@dataclass(frozen=True)
class DenseUnit:
    name: str
    members: tuple[str, ...]
    devices: tuple
    ports: tuple[str, ...]
    nets: tuple[tuple[str, str], ...]
    vt: str

    @property
    def plan(self):
        return cell_plan(self.devices, self.ports, share=True)

    def build(self, *, abut_y=False):
        return build_dense_cell(
            self.devices, self.ports, vt=self.vt, abut_y=abut_y, share=True
        )[0]


def dense_units(circuit, *, exclude=()):
    """Preserve functional units; group column readouts including keepers."""
    excluded = set(exclude)
    external = set(circuit.pins) | {
        n for d in circuit.devices if d.name in excluded for n in d.nodes
    }
    devices = {d.name: d for d in circuit.devices if d.name not in excluded}
    groups = [
        (n, m)
        for n, m in circuit.placement_groups
        if n.startswith(("readout_", "replica_readout_"))
    ]
    groups += [
        (f"{kind}_{i}", members)
        for i, (kind, members) in enumerate(circuit.logic_units)
    ]
    seen, selected = set(), []
    for name, members in groups:
        if any(n in seen or n not in devices for n in members):
            continue
        seen.update(members)
        selected.append((name, members))
    selected += [(n, (n,)) for n in devices if n not in seen]
    owners = Counter()
    for _, members in selected:
        owners.update({n for d in members for n in devices[d].nodes})
    result = []
    role = {
        member: name for name, members in circuit.placement_groups for member in members
    }
    for name, members in selected:
        ds = [devices[n] for n in members]
        names = {"VDD": "VDD", "VSS": "VSS"}
        for d in ds:
            for net in d.nodes:
                if net not in names:
                    names[net] = f"N{len(names) - 2}"
        ports = tuple(
            names[n]
            for n in names
            if n not in ("VDD", "VSS") and (owners[n] > 1 or n in external)
        )
        shape = tuple(
            (d.spec.flavor, d.spec.fins, *(names[n] for n in d.nodes[:3])) for d in ds
        )
        result.append(
            DenseUnit(
                role.get(members[0], name),
                tuple(members),
                shape,
                ports,
                tuple((v, k) for k, v in names.items()),
                ds[0].spec.vt,
            )
        )
    # Respect the electrical placement pass: in particular a SAMPLE receiver,
    # its latch and closure observer must remain neighbours after folding.
    order = {d.name: i for i, d in enumerate(circuit.devices)}
    return sorted(result, key=lambda u: min(order[n] for n in u.members))
