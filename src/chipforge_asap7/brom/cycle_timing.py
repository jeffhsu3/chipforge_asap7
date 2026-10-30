"""Measured read and recovery paths in a passing closed-loop transient."""

from .timing import settling_delay


def measure_cycle_paths(time, traces, spec, words, vdd, *, phase_scale=1):
    from .replica_timing import crossing

    t = time * 1e12
    reads = []
    for address, word in enumerate(words):
        base = address * 1000 * phase_scale
        fall = crossing(
            t,
            traces["CLK"],
            base + 200 * phase_scale,
            base + 300 * phase_scale,
            0.5 * vdd,
            False,
        )
        rise = crossing(
            t,
            traces["CLK"],
            base + 750 * phase_scale,
            base + 850 * phase_scale,
            0.5 * vdd,
            True,
        )
        deadline = base + 1180 * phase_scale
        row = {"address": address}
        if fall is None or rise is None:
            raise ValueError("missing clock edge in passing transient")
        for label, prefix in (
            ("fall_to_data_valid_ps", "DATA"),
            ("fall_to_sampled_valid_ps", "SAMPLED" if spec.registered_output else "Q"),
        ):
            delays = [
                settling_delay(
                    t,
                    traces[f"{prefix}<{bit}>"],
                    fall,
                    rise,
                    bool(word >> bit & 1),
                    vdd,
                )
                for bit in range(spec.bits)
            ]
            row[label] = max(delays) if all(d is not None for d in delays) else None
        recover = [
            settling_delay(t, traces[n], rise, deadline, False, vdd)
            for n in ("PRECHN", "EVAL", "SAMPLE", "READ_READY", "CANDIDATE_BUSY")
        ]
        recover += [
            settling_delay(t, traces[f"BL<{i}>"], rise, deadline, True, vdd)
            for i in range(spec.columns)
        ]
        row["rise_to_idle_and_precharged_ps"] = (
            max(recover) if all(d is not None for d in recover) else None
        )
        reads.append(row)
    keys = (
        "fall_to_data_valid_ps",
        "fall_to_sampled_valid_ps",
        "rise_to_idle_and_precharged_ps",
    )
    return {
        "reads": reads,
        "maxima_ps": {
            k: max(r[k] for r in reads)
            if reads and all(r[k] is not None for r in reads)
            else None
            for k in keys
        },
        "independent_minimum_pulse_widths": False,
    }
