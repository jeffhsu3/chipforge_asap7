"""Local sampling receivers and all-latch closure completion feedback."""

from dataclasses import replace


def add_sample_receivers(c, latch_groups):
    """Buffer SAMPLE locally; return closure from every data/replica latch.

    ``latch_groups`` lists each receiver's latches: an 8T latch group name,
    or a tuple of names that one receiver drives. Its output and complement
    inverters have the base fin count times half its latches, rounded up.
    Keep each receiver, its latches and observer adjacent.
    The global return tree carries closure indications, never latch gate
    controls. RC/PEX qualification is still required for the receiving and
    return paths.
    """
    if c.sample_distribution is not None:
        raise ValueError("sampling receivers already installed")
    latch_groups = [(g,) if isinstance(g, str) else tuple(g) for g in latch_groups]
    groups = dict(c.placement_groups)
    by_name = {d.name: d for d in c.devices}
    old_bar = by_name[groups[latch_groups[0][0]][1]].nodes[1]
    complement = [
        d for d in c.devices if d.nodes[0] == old_bar and d.nodes[1] == "SAMPLE"
    ]
    if len(complement) != 2:
        raise ValueError("expected the original shared SAMPLE inverter")
    original = list(c.devices)
    placements, moved = {}, set()
    banks, wires = [], ["SAMPLE"]

    def gate(kind, group, *nets):
        start = len(c.devices)
        getattr(c, kind)(*nets)
        c.group_since(group, start)
        return list(c.devices[start:])

    def strong_inv(group, a, y, strength):
        if strength == 1:
            return gate("inv", group, a, y)
        start = len(c.devices)
        c.fet(False, y, a, "VSS", fins=c.n.fins * strength)
        c.fet(True, y, a, "VDD", fins=c.p.fins * strength)
        c.group_since(group, start)
        return list(c.devices[start:])

    for i, names in enumerate(latch_groups):
        if any(len(groups[name]) != 8 for name in names):
            raise ValueError("sampling receiver requires an 8T latch")
        # Two latches per base-strength driver, as one latch had before.
        strength = max(1, -(-len(names) // 2))
        mid, enable, inverse, off, quiet, closed = (c.node() for _ in range(6))
        local = gate("inv", f"sample_receiver_input_{i}", "SAMPLE", mid)
        local += strong_inv(f"sample_receiver_output_{i}", mid, enable, strength)
        if i == 0:
            inverse = old_bar
            reused = [
                replace(
                    d,
                    spec=replace(d.spec, fins=d.spec.fins * strength),
                    nodes=(d.nodes[0], enable, *d.nodes[2:]),
                )
                for d in complement
            ]
            c.placement_groups.append(
                (f"sample_receiver_complement_{i}", tuple(d.name for d in reused))
            )
            local += reused
        else:
            local += strong_inv(
                f"sample_receiver_complement_{i}", enable, inverse, strength
            )
        for name in names:
            for member in groups[name]:
                d = by_name[member]
                g = (
                    enable
                    if d.nodes[1] == "SAMPLE"
                    else inverse
                    if d.nodes[1] == old_bar
                    else d.nodes[1]
                )
                local.append(replace(d, nodes=(d.nodes[0], g, *d.nodes[2:])))
        local += gate("inv", f"sample_receiver_off_{i}", enable, off)
        # The first receiver stage must also be idle. Otherwise an in-flight
        # SAMPLE pulse could be acknowledged before reaching the latch gates.
        local += gate("and2", f"sample_receiver_quiet_{i}", mid, off, quiet)
        local += gate("and2", f"sample_receiver_closed_{i}", quiet, inverse, closed)
        placements[groups[names[0]][0]] = local
        moved.update(d.name for d in local)
        wires += [mid, enable, inverse, off, quiet, closed]
        banks.append(
            {
                "latch_group": names[0],
                "latch_groups": list(names),
                "command_inverted": mid,
                "enable": enable,
                "complement": inverse,
                "closed": closed,
            }
        )
    # Move existing and newly built devices without renaming them: selector
    # device inventories and every functional placement group remain valid.
    c.devices = []
    for d in original:
        if d.name in placements:
            c.devices.extend(placements[d.name])
        elif d.name not in moved:
            c.devices.append(d)
    level = [bank["closed"] for bank in banks]
    joins = []
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            if i + 1 == len(level):
                next_level.append(level[i])
                continue
            target = c.node()
            gate(
                "and2",
                f"sample_closed_join_{len(joins)}",
                level[i],
                level[i + 1],
                target,
            )
            joins.append(target)
            next_level.append(target)
        level = next_level
    wires += joins
    c.sample_distribution = {
        "method": "local_receivers_all_latch_closure",
        "banks": banks,
        "closure_join": level[0],
        "wire_nets": wires,
        "timing_qualified": False,
    }
    return level[0]
