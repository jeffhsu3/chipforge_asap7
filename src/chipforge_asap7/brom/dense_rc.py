"""Conservative signal RC from the dense layout's recorded metal and vias.

Centerline overlaps are merged. Path resistance is bounded by all metal in
one connected net, so this is a pessimistic lumped stress model, not PEX.
Local M1-M3, candidate M6 and programmed selector accesses are included.
"""

from collections import defaultdict

from ..verification.stack import STACK_CONDUCTORS
from .dense_macro import dense_macro_layout


def dense_control_rc(c, model):
    _, plan = dense_macro_layout(c)
    metals = {m.name: m for m in STACK_CONDUCTORS}
    intervals = defaultdict(list)
    vias = defaultdict(set)
    for net, layer, a, b in plan["routes"]:
        if net in ("VDD", "VSS"):
            continue
        if layer.startswith("V"):
            vias[net].add((layer, *a))
            continue
        axis = 0 if a[1] == b[1] else 1
        if a[1 - axis] != b[1 - axis]:
            raise ValueError("non-Manhattan dense route")
        if a == b:
            continue
        intervals[net, layer, axis, a[1 - axis]].append(
            tuple(sorted((a[axis], b[axis])))
        )
    lengths = defaultdict(lambda: defaultdict(float))
    for (net, layer, axis, fixed), spans in intervals.items():
        merged = []
        for lo, hi in sorted(spans):
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(hi, merged[-1][1])
            else:
                merged.append([lo, hi])
        lengths[net][layer] += sum(hi - lo for lo, hi in merged) / 1000
    gate_fins = defaultdict(int)
    for d in c.devices:
        gate_fins[d.nodes[1]] += d.spec.total_fins
    result = {}
    for net, ls in lengths.items():
        resistance = sum(
            metals[m].sheet_resistance_ohm
            * length
            / (0.018 if m in ("M1", "M2", "M3") else 0.032 if m == "M6" else 0.024)
            for m, length in ls.items()
        )
        resistance += sum(
            metals["M" + v[1:]].via_resistance_ohm for v, _, _ in vias[net]
        )
        cap = sum(ls.values()) * model.wire_cap_ff_per_um
        gate = gate_fins[net] * model.gate_cap_ff_per_fin
        result[net] = {
            "length_um_by_layer": dict(ls),
            "wire_cap_ff": cap,
            "gate_cap_ff": gate,
            "resistance_sum_ohm": resistance,
            "max_driver_sink_path_ohm": resistance,
            "path_model": "total_net_resistance_upper_bound",
            "wire_rc_cost_ps": resistance * (cap + gate) * 1e-3,
            "physical_sha256": plan["physical_sha256"],
        }
    return result
