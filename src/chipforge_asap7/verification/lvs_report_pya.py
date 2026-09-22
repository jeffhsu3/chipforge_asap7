"""Read a KLayout LVS database into JSON.  Run *by* KLayout, not imported:

    klayout -b -r lvs_report_pya.py -rd db_path=layout.lvsdb.gz -rd out_path=summary.json

KLayout's own Python carries the database API, so reading a report needs
nothing beyond the binary LVS already required.  `lvs_report.summarize_lvsdb`
is the front door.

Besides the cross-reference it answers one question KLayout cannot: whether a
circuit pair that fails to match differs *only in the order of a series stack*.
Released ASAP7 cells do -- ``AOI211xp5`` stacks its pull-up C, B, (A1|A2) in the
GDS and (A1|A2), B, C in the CDL; ``AO21x1`` has A1 and A2 the other way up its
pull-down -- and Calibre's gate recognition lets that pass where KLayout's
strictly topological compare does not.  Worse, KLayout *matches* ``AO21x1`` by
pairing A1 with A2, and the disagreement then surfaces in the parent as nets
that look identical and will not pair.  `series_parallel_signature` reduces
each pull network to a series/parallel tree with unordered series, names every
net from the pins inward, and compares; equal signatures and equal fins per
input are proof that pin-name comparison of that cell loses nothing.
"""

import json
from collections import Counter, defaultdict

import pya  # provided by KLayout

EXAMPLES = 12


# ── Series/parallel signature ────────────────────────────────────────────────
def _flatten(kind, children):
    out = []
    for child in children:
        if child[0] == kind:
            out.extend(child[1])
        else:
            out.append(child)
    return (kind, out)


def _canon(tree, names):
    """Canonical text of a tree, or None while a gate net is still unnamed."""
    if tree[0] == "L":
        gate = names.get(tree[2])
        return None if gate is None else f"{tree[1]}({gate})"
    parts = [_canon(child, names) for child in tree[1]]
    if any(part is None for part in parts):
        return None
    if tree[0] == "P":  # parallel copies are width, not logic: fins are counted apart
        parts = sorted(set(parts))
        return parts[0] if len(parts) == 1 else "P[" + ",".join(parts) + "]"
    return "S[" + ",".join(sorted(parts)) + "]"  # series order is not logic either


def series_parallel_signature(circuit):
    """``(signature, fins)`` of a transistor-level circuit, or None if it has no unique one."""
    names = {}
    for pin in circuit.each_pin():
        net = circuit.net_for_pin(pin.id())
        if net is not None:
            names[net.expanded_name()] = pin.name().upper()
    edges, gates, unit = [], set(), []
    for device in circuit.each_device():
        cls = device.device_class()
        if not all(cls.has_terminal(t) for t in ("S", "G", "D")):
            return None  # not a MOS: a resistor or capacitor has no pull network
        nets = [device.net_for_terminal(cls.terminal_id(t)) for t in ("S", "G", "D")]
        if any(net is None for net in nets):
            return None
        # Keyed by name: a schematic net has no cluster id to tell it apart by.
        source, gate, drain = (net.expanded_name() for net in nets)
        kind = cls.name.lower()
        edges.append([kind, ("L", kind, gate), source, drain])
        gates.add(gate)
        unit.append((kind, gate))
    if not edges:
        return None
    protected = set(names) | gates

    changed = True
    while changed:
        changed = False
        groups = defaultdict(list)
        for edge in edges:
            if edge[2] != edge[3]:
                groups[(edge[0], frozenset(edge[2:4]))].append(edge)
        for (kind, _), members in groups.items():
            if len(members) > 1:
                merged = [
                    kind,
                    _flatten("P", [m[1] for m in members]),
                    members[0][2],
                    members[0][3],
                ]
                edges = [e for e in edges if not any(e is m for m in members)] + [
                    merged
                ]
                changed = True
        incident = defaultdict(list)
        for edge in edges:
            incident[edge[2]].append(edge)
            incident[edge[3]].append(edge)
        for node, members in incident.items():
            if node in protected or len(members) != 2 or members[0] is members[1]:
                continue
            first, second = members
            if first[0] != second[0]:
                continue
            ends = [e[2] if e[3] == node else e[3] for e in members]
            if ends[0] == ends[1]:
                continue
            joined = [first[0], _flatten("S", [first[1], second[1]]), ends[0], ends[1]]
            edges = [e for e in edges if e is not first and e is not second] + [joined]
            changed = True
            break

    # Name the nets no pin names, from what is already named, inward.
    progress = True
    while progress:
        progress = False
        unnamed = {n for e in edges for n in e[2:4] if n not in names} | {
            g for g in gates if g not in names
        }
        for net in unnamed:
            around = []
            for edge in edges:
                if net not in edge[2:4]:
                    continue
                other = edge[3] if edge[2] == net else edge[2]
                text = _canon(edge[1], names)
                if text is None or other not in names:
                    around = None
                    break
                around.append(text + "@" + names[other])
            if around:
                names[net] = "{" + ";".join(sorted(around)) + "}"
                progress = True
    texts = []
    for edge in edges:
        text = _canon(edge[1], names)
        if text is None or edge[2] not in names or edge[3] not in names:
            return None  # feedback (a latch): no inward naming exists
        texts.append(text + ":" + "|".join(sorted((names[edge[2]], names[edge[3]]))))
    fins = Counter(f"{kind}({names[gate]})" for kind, gate in unit)
    return sorted(texts), dict(fins)


# ── Cross-reference ──────────────────────────────────────────────────────────
def _status(code):
    ref = pya.NetlistCrossReference
    return {
        ref.Match: "match",
        ref.MatchWithWarning: "match",
        ref.Mismatch: "mismatch",
        ref.NoMatch: "nomatch",
        ref.Skipped: "skipped",
    }.get(code, "none")


def _name(item):
    if item is None:
        return None
    for attribute in ("expanded_name", "name"):
        value = getattr(item, attribute, None)
        if value is not None:
            return value() if callable(value) else value
    return str(item)


SUPPLIES = {"VDD", "VSS", "GND", "VCC", "VPWR", "VGND"}


def _supply_pins(circuit):
    """``(supply, child circuit, pin) -> count`` for every instance pin on a supply net."""
    found = Counter()
    if circuit is None:
        return found
    for net in circuit.each_net():
        names = {
            part.strip().upper().rstrip("!") for part in net.expanded_name().split(",")
        }
        supply = sorted(names & SUPPLIES)
        if not supply:
            continue
        for ref in net.each_subcircuit_pin():
            pin = ref.pin().name().upper()
            if pin.rstrip("!") not in SUPPLIES:
                found[
                    (supply[0], ref.subcircuit().circuit_ref().name.upper(), pin)
                ] += 1
    return found


def supply_shorts(db, layout_name, reference_name):
    """Instance pins that sit on a supply in the layout and not in the reference.

    A short to a rail swallows a net whole, so the compare can only say that
    nothing matches; this says which pins went where.
    """
    ours = _supply_pins(db.netlist().circuit_by_name(layout_name))
    theirs = _supply_pins(db.reference.circuit_by_name(reference_name))
    return [
        {
            "supply": key[0],
            "circuit": key[1],
            "pin": key[2],
            "count": count - theirs.get(key, 0),
        }
        for key, count in sorted(ours.items())
        if count > theirs.get(key, 0)
    ]


def summarize(path):
    db = pya.LayoutVsSchematic()
    db.read(path)
    xref = db.xref()
    circuits = []
    for pair in xref.each_circuit_pair():
        layout, reference = pair.first(), pair.second()
        entry = {
            "layout": layout.name if layout is not None else None,
            "reference": reference.name if reference is not None else None,
            "status": _status(pair.status()),
            "counts": {},
            "examples": {},
            "renamed_pins": [],
        }
        for what, items in (
            ("nets", xref.each_net_pair(pair)),
            ("devices", xref.each_device_pair(pair)),
            ("pins", xref.each_pin_pair(pair)),
            ("subcircuits", xref.each_subcircuit_pair(pair)),
        ):
            tally, examples = Counter(), []
            for item in items:
                status = _status(item.status())
                tally[status] += 1
                first, second = _name(item.first()), _name(item.second())
                if status != "match" and len(examples) < EXAMPLES:
                    examples.append([first, second])
                if (
                    what == "pins"
                    and status == "match"
                    and first
                    and second
                    and first.upper() != second.upper()
                ):
                    entry["renamed_pins"].append([first, second])
            entry["counts"][what] = dict(tally)
            if examples:
                entry["examples"][what] = examples
        differs = entry["status"] != "match" or entry["renamed_pins"]
        if layout is not None and reference is not None and differs:
            # By name from the netlists: the cross-reference hands out views whose
            # devices do not answer for their nets.
            ours = series_parallel_signature(db.netlist().circuit_by_name(layout.name))
            theirs = series_parallel_signature(
                db.reference.circuit_by_name(reference.name)
            )
            entry["series_order_only"] = (
                None if ours is None or theirs is None else ours == theirs
            )
        if layout is not None and reference is not None and entry["status"] != "match":
            shorts = supply_shorts(db, layout.name, reference.name)
            if shorts:
                entry["supply_shorts"] = shorts
        circuits.append(entry)
    top = circuits[-1] if circuits else None
    return {
        "matched": bool(circuits) and all(c["status"] == "match" for c in circuits),
        "top": top["layout"] if top else None,
        "circuits": circuits,
    }


if "db_path" in globals():
    result = summarize(db_path)  # noqa: F821  (set by klayout -rd)
    with open(out_path, "w") as stream:  # noqa: F821
        json.dump(result, stream, indent=1)
