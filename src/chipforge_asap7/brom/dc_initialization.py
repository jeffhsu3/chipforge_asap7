"""Legal idle-state NODESET hints for transistor transient DC initialization.

These Boolean values are Newton-solver guesses, never forced transient initial
conditions. Floating latch storage chooses zero; power-up behavior is unqualified.
"""


def idle_logic_state(c, address_bits):
    fixed = {
        "VDD": 1,
        "VSS": 0,
        "CLK": 1,
        "RESETN": 0,
        **{f"A[{i}]": 0 for i in range(address_bits)},
    }
    by_name = {d.name: d for d in c.devices}
    values = dict(fixed)
    for _, members in c.placement_groups:
        if len(members) == 8:
            for index, value in ((0, 0), (2, 1), (4, 0)):
                values[by_name[members[index]].nodes[0]] = value
    terminals = {n for d in c.devices for n in d.nodes}
    complements = []
    controls = {"PRECHN", "SAMPLE", "REG_M", "REG_S"}
    controls.update(b["enable"] for b in c.sample_distribution["banks"])
    for d in c.devices:
        out, gate, source, _ = d.nodes
        if (
            d.spec.flavor == "n"
            and source == "VSS"
            and gate in controls | fixed.keys()
            and any(
                p.spec.flavor == "p" and p.nodes[:3] == (out, gate, "VDD")
                for p in c.devices
            )
        ):
            complements.append((gate, out))
    for _ in range(400):
        for gate, out in complements:
            if gate in values:
                values[out] = 1 - values[gate]
        parent = {n: n for n in terminals}

        def root(n, parent=parent):
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        for d in c.devices:
            drain, gate, source, _ = d.nodes
            if values.get(gate) == (0 if d.spec.flavor == "p" else 1):
                parent[root(drain)] = root(source)
        driven = {}
        for net, value in fixed.items():
            driven.setdefault(root(net), set()).add(value)
        new = dict(values)
        for net in terminals:
            levels = driven.get(root(net), set())
            if len(levels) == 1:
                new[net] = next(iter(levels))
        new.update(fixed)
        if new == values:
            if any(len(v) > 1 for v in driven.values()):
                raise ValueError("DC hint logic has rail contention")
            if any(
                new.get(n) != v
                for n, v in {
                    "SAMPLE": 0,
                    "PRECHN": 0,
                    "EVAL": 0,
                    "SAMPLE_CLOSED": 1,
                }.items()
            ):
                raise ValueError("DC hint logic did not reach idle")
            return new
        values = new
    raise ValueError("DC hint logic did not settle")
