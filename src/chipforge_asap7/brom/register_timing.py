"""Checks for the physical rising-edge output bank in replica transients."""

import numpy as np

from .timing import settling_delay


def analyze_registered_output(time, traces, spec, words, vdd, *, phase_scale=1):
    from .replica_timing import crossing

    t = time * 1e12
    failures, reads = [], []
    active = t > 150 * phase_scale
    master_open = (traces["REG_M"] > 0.2 * vdd) | (traces["REG_MB"] < 0.8 * vdd)
    slave_open = (traces["REG_S"] > 0.2 * vdd) | (traces["REG_SB"] < 0.8 * vdd)
    overlap = active & master_open & slave_open
    edges = np.diff(np.r_[False, overlap, False].astype(int))
    intervals = [
        {"start_ps": float(t[a]), "observed_duration_ps": float(t[b - 1] - t[a])}
        for a, b in zip(
            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
        )
    ]
    if np.any(overlap):
        failures.append("output register master/slave input gates overlap")
    handoffs = []
    for cycle in range(len(words)):
        for edge_name, start, end, closing, opening in (
            ("falling", 200, 750, "REG_S", "REG_M"),
            ("rising", 750, 1200, "REG_M", "REG_S"),
        ):
            begin = (cycle * 1000 + start) * phase_scale
            finish = min((cycle * 1000 + end) * phase_scale, t[-1])
            if begin >= finish:
                continue
            closed = [
                crossing(t, traces[closing], begin, finish, 0.2 * vdd, False),
                crossing(t, traces[closing + "B"], begin, finish, 0.8 * vdd, True),
            ]
            opened = [
                crossing(t, traces[opening], begin, finish, 0.2 * vdd, True),
                crossing(t, traces[opening + "B"], begin, finish, 0.8 * vdd, False),
            ]
            margin = (
                None if None in closed or None in opened else min(opened) - max(closed)
            )
            handoffs.append(
                {"cycle": cycle, "clock_edge": edge_name, "nonoverlap_ps": margin}
            )
            if margin is None:
                failures.append(
                    f"cycle {cycle}: missing {edge_name} register phase handoff"
                )
            elif margin < 0:
                failures.append(
                    f"cycle {cycle}: negative {edge_name} register nonoverlap margin"
                )
    for address, word in enumerate(words):
        start = (address * 1000 + 750) * phase_scale
        end = (address * 1000 + 1000) * phase_scale
        edge = crossing(t, traces["CLK"], start, end, 0.5 * vdd, True)
        if edge is None:
            failures.append(f"address {address}: missing rising capture edge")
            continue
        master_off = crossing(t, traces["REG_M"], edge, end, 0.1 * vdd, False)
        master_bar = crossing(t, traces["REG_MB"], edge, end, 0.9 * vdd, True)
        if master_off is None or master_bar is None:
            failures.append(f"address {address}: master input did not close")
            continue
        master_closed = max(master_off, master_bar)
        margins, delays = [], []
        hold_end = (
            ((address + 1) * 1000 + 750) * phase_scale
            if address + 1 < len(words)
            else t[-1]
        )
        for bit in range(spec.bits):
            high = bool(word >> bit & 1)
            data_delay = settling_delay(
                t,
                traces[f"SAMPLED<{bit}>"],
                (address * 1000 + 200) * phase_scale,
                edge,
                high,
                vdd,
            )
            if data_delay is None:
                failures.append(
                    f"address {address}: sampled bit {bit} not valid before capture edge"
                )
            else:
                margins.append(
                    edge - ((address * 1000 + 200) * phase_scale + data_delay)
                )
            aperture = traces[f"SAMPLED<{bit}>"][(t >= edge) & (t <= master_closed)]
            if np.any(aperture < 0.9 * vdd if high else aperture > 0.1 * vdd):
                failures.append(
                    f"address {address}: sampled bit {bit} changed during capture hold aperture"
                )
            q = traces[f"Q[{bit}]"]
            valid = crossing(t, q, edge, end, (0.9 if high else 0.1) * vdd, high)
            if valid is None:
                failures.append(
                    f"address {address}: registered Q[{bit}] did not capture"
                )
                continue
            delay = valid - edge
            delays.append(delay)
            held = q[(t >= edge + delay + 1e-6) & (t < hold_end)]
            if not len(held) or not np.all(
                held >= 0.9 * vdd if high else held <= 0.1 * vdd
            ):
                failures.append(
                    f"address {address}: registered Q[{bit}] changed between rising edges"
                )
        reads.append(
            {
                "address": address,
                "data_before_edge_ps": min(margins)
                if len(margins) == spec.bits
                else None,
                "clk_to_valid_ps": max(delays) if len(delays) == spec.bits else None,
            }
        )
    return {
        "passed": not failures,
        "failures": failures,
        "reads": reads,
        "minimum_data_before_edge_ps": min(r["data_before_edge_ps"] for r in reads)
        if reads and not failures
        else None,
        "maximum_clk_to_valid_ps": max(r["clk_to_valid_ps"] for r in reads)
        if reads and not failures
        else None,
        "phase_overlap_intervals": intervals,
        "phase_handoffs": handoffs,
        "minimum_nonoverlap_ps": min(r["nonoverlap_ps"] for r in handoffs)
        if handoffs and all(r["nonoverlap_ps"] is not None for r in handoffs)
        else None,
        "setup_hold_qualified": False,
    }
