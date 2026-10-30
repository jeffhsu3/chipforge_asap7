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
    if np.any(active & master_open & slave_open):
        failures.append("output register master/slave input gates overlap")
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
        "setup_hold_qualified": False,
    }
