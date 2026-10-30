"""Closed-loop replica transient: only CLK, RESETN and address are driven."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..devices.spice import (
    _osdi_command_path,
    _spice_path,
    find_default_model_card,
    translate_asap7_model,
)
from .compiler import BromSpec
from .distribution import size_control_distribution
from .replica import ReplicaSpec, self_timed_macro_circuit
from .timing import TimingConditions, _run_ngspice, pattern_words, settling_delay
from .wire_rc import routed_control_rc, size_wire_rc_distribution


def render_replica_bench(
    spec,
    words,
    replica,
    conditions,
    model_card,
    osdi,
    *,
    klu=False,
    sized_controls=False,
    wire_rc_controls=False,
    startup_ramp_ps=0,
):
    if (
        not math.isfinite(startup_ramp_ps)
        or not 0 <= startup_ramp_ps < 40 * conditions.phase_scale
    ):
        raise ValueError("startup ramp must finish before the first address stimulus")
    c = self_timed_macro_circuit(spec, words, replica=replica)
    if sized_controls and wire_rc_controls:
        raise ValueError("choose gate sizing or wire-RC sizing, not both")
    if wire_rc_controls:
        size_wire_rc_distribution(c)
    elif sized_controls:
        size_control_distribution(c)
    by_name = {d.name: d for d in c.devices}
    groups = dict(c.placement_groups)
    sampleb = by_name[groups["output_latch_0"][1]].nodes[1]
    aliases = {"SAMPLEB": sampleb}
    for bit in range(spec.bits):
        aliases[f"DATA<{bit}>"] = by_name[groups[f"output_latch_{bit}"][0]].nodes[2]
    nets = sorted({n for d in c.devices for n in d.nodes})
    nodes = {net: f"n{i}" for i, net in enumerate(nets)}
    nodes.update(VDD="vdd", VSS="0")
    signals = [
        "CLK",
        "RESETN",
        "PRECHN",
        "PRECHN_FAR",
        "EVAL",
        "READ_READY",
        "SAMPLE",
        "SAMPLEB",
        "SAMPLE_CLOSED",
        "CANDIDATE_BUSY",
        "RVALID",
    ]
    signals += [f"C<{b},{k}>" for b in range(spec.blocks) for k in range(16)]
    signals += [f"BL<{i}>" for i in range(spec.columns)]
    signals += [f"DATA<{i}>" for i in range(spec.bits)] + [
        f"Q<{i}>" for i in range(spec.bits)
    ]
    if spec.registered_output:
        signals += [f"SAMPLED<{i}>" for i in range(spec.bits)]
        signals += ["REG_M", "REG_MB", "REG_S", "REG_SB"]
    signals += [
        f"RBL<{spec.columns - 1}>",
        f"RC<{spec.blocks + replica.extra_blocks - 1},15>",
    ]
    initial = {
        "CLK": 1,
        "RESETN": 0,
        **{f"A<{i}>": 0 for i in range(spec.address_bits)},
    }
    changes = {net: [] for net in initial}
    changes["RESETN"] = [(100, 1), (spec.depth * 1000 + 260, 0)]
    for cycle, address in enumerate([*range(spec.depth), 0]):
        base = cycle * 1000
        changes["CLK"] += [(base + 200, 0), (base + 750, 1)]
        for bit in range(spec.address_bits):
            changes[f"A<{bit}>"] += [
                (base + 40, address >> bit & 1),
                (base + 600, 1 - (address >> bit & 1)),
            ]
            if spec.registered_output:
                # Exercise next-address launch during the output capture/hold
                # aperture; the existing address latch must isolate this edge.
                changes[f"A<{bit}>"] += [(base + 760, address >> bit & 1)]
    changes = {
        net: [(t * conditions.phase_scale, v) for t, v in edges]
        for net, edges in changes.items()
    }
    # Cancel 60 ps after the final falling clock edge regardless of period.
    changes["RESETN"][-1] = ((spec.depth * 1000 + 200) * conditions.phase_scale + 60, 0)
    lines = [
        "* BROM replica closed loop; real generated transistors; wire model recorded in report",
        f".include {_spice_path(model_card)}",
        f".temp {conditions.temperature_c:g}",
        (
            ".options "
            + ("klu " if klu else "")
            + "method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6 rshunt=1e15"
        ),
        (
            f"Vsupply vdd 0 PWL(0 0 {startup_ramp_ps:g}p {conditions.vdd:g})"
            if startup_ramp_ps
            else f"Vsupply vdd 0 {conditions.vdd:g}"
        ),
    ]
    rc_nets = {}
    if wire_rc_controls:
        for record in c.distribution["nets"]:
            for net in [
                record["stages"][0]["input"],
                *(stage["output"] for stage in record["stages"]),
            ]:
                rc_nets[net] = c.distribution["routed_rc"][net]
        if spec.registered_output:
            # These are gate-only clock nets; never split latch storage/data.
            routes = routed_control_rc(c)
            for net in ("CLK", "REG_CLKB", "REG_M", "REG_MB", "REG_S", "REG_SB"):
                rc_nets[net] = routes[net]
    for i, (net, edges) in enumerate(changes.items()):
        old = initial[net]
        pts = (
            ["0 0", f"{startup_ramp_ps:g}p {old * conditions.vdd:g}"]
            if startup_ramp_ps
            else [f"0 {old * conditions.vdd:g}"]
        )
        for t, v in sorted(edges):
            if v != old:
                pts += [
                    f"{t}p {old * conditions.vdd:g}",
                    f"{t + conditions.slew_ps:g}p {v * conditions.vdd:g}",
                ]
            old = v
        source = "rcdrv_" + nodes[net] if net in rc_nets else nodes[net]
        lines.append(f"Vstim{i} {source} 0 PWL({' '.join(pts)})")
    for d in c.devices:
        terminals = [nodes[n] for n in d.nodes]
        if d.nodes[0] in rc_nets:
            terminals[0] = "rcdrv_" + terminals[0]
        lines.append(
            f"N{d.name} {' '.join(terminals)} {d.spec.model} L=20n NFIN={d.spec.fins} NF=1"
        )
    # All sink gates remain on the original net; only inverter output drains
    # move to the source side. This Rpath pi model is an intentionally pessimistic
    # stress approximation, not the actual distributed extracted network.
    for i, (net, rc) in enumerate(sorted(rc_nets.items())):
        sink, source = nodes[net], "rcdrv_" + nodes[net]
        lines += [
            f"* estimated control wire: {net}",
            f"Rwire{i} {source} {sink} {rc['max_driver_sink_path_ohm']:.12g}",
            f"Cwire{i}a {source} 0 {rc['wire_cap_ff'] / 2:.12g}f",
            f"Cwire{i}b {sink} 0 {rc['wire_cap_ff'] / 2:.12g}f",
        ]
    if conditions.output_load_ff:
        for bit in range(spec.bits):
            lines.append(
                f"CQ{bit} {nodes[f'Q<{bit}>']} 0 {conditions.output_load_ff:g}f"
            )
    if conditions.bitline_load_ff:
        # Intentionally load real bitlines only: useful as an adversarial test
        # of insufficient replica margin, not an assumed matched wire model.
        for col in range(spec.columns):
            lines.append(
                f"CBL{col} {nodes[f'BL<{col}>']} 0 {conditions.bitline_load_ff:g}f"
            )
    for members in groups.values():
        if len(members) == 8:
            for index, voltage in ((0, 0), (2, conditions.vdd), (4, 0)):
                lines.append(
                    f".nodeset V({nodes[by_name[members[index]].nodes[0]]})={voltage:g}"
                )
    if spec.registered_output:
        # Legal startup hints for the interlocked clock loop (CLK initially high).
        # NODESET aids the DC solve without forcing transient state.
        for net, high in (
            ("REG_CLKB", False),
            ("REG_M", False),
            ("REG_MB", True),
            ("REG_S", True),
            ("REG_SB", False),
        ):
            lines.append(f".nodeset V({nodes[net]})={conditions.vdd if high else 0:g}")
    vectors = [f"v({nodes[aliases.get(net, net)]})" for net in signals]
    if "CLK" in rc_nets:
        # Measure latency from the macro pin, and retain the receiving clock.
        signals.append("CLK_RECEIVER")
        vectors.append(f"v({nodes['CLK']})")
        vectors[signals.index("CLK")] = f"v(rcdrv_{nodes['CLK']})"

    stop = ((spec.depth + 1) * 1000 + 200) * conditions.phase_scale
    lines += [
        ".control",
        "set num_threads=1",
        f"pre_osdi {_osdi_command_path(osdi)}",
        "set wr_singlescale",
        "set wr_vecnames",
        "set numdgt=12",
        "save " + " ".join(vectors),
        f"tran {conditions.step_ps:g}p {stop}p 0 {conditions.step_ps:g}p",
        "wrdata waveform.dat " + " ".join(vectors),
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines), signals


def crossing(time, values, start, end, threshold, rising):
    mask = (time > start) & (time < end)
    t = np.concatenate(([start], time[mask], [end]))
    v = np.interp(t, time, values)
    good = v >= threshold if rising else v <= threshold
    ids = np.flatnonzero(good)
    if not len(ids):
        return None
    i = ids[0]
    if i == 0:
        return float(start)
    return float(
        t[i - 1] + (threshold - v[i - 1]) * (t[i] - t[i - 1]) / (v[i] - v[i - 1])
    )


def analyze_replica(time, traces, spec, words, vdd, *, phase_scale=1):
    if spec.registered_output:
        from dataclasses import replace

        core = dict(traces)
        core.update({f"Q<{i}>": traces[f"SAMPLED<{i}>"] for i in range(spec.bits)})
        result = analyze_replica(
            time,
            core,
            replace(spec, registered_output=False),
            words,
            vdd,
            phase_scale=phase_scale,
        )
        from .register_timing import analyze_registered_output

        reg = analyze_registered_output(
            time, traces, spec, words, vdd, phase_scale=phase_scale
        )
        result["registered_output"] = reg
        result["failures"].extend(reg["failures"])
        result["passed"] = not result["failures"]
        if not result["passed"]:
            result["minimum_margins_ps"] = {}
        return result
    failures, records = [], []
    t = time * 1e12 / phase_scale
    candidates = np.maximum.reduce(
        [traces[f"C<{b},{k}>"] for b in range(spec.blocks) for k in range(16)]
    )
    active = t > 150
    if np.any(active & (traces["PRECHN"] < 0.8 * vdd) & (candidates > 0.2 * vdd)):
        failures.append("precharge overlaps active data candidates")
    if np.any(active & (traces["SAMPLE"] > 0.8 * vdd) & (traces["EVAL"] < 0.8 * vdd)):
        failures.append("sampling outside evaluation")
    if np.any(
        active
        & (traces["EVAL"] < 0.8 * vdd)
        & ((traces["SAMPLE"] > 0.2 * vdd) | (traces["SAMPLEB"] < 0.8 * vdd))
    ):
        failures.append("latch input open outside evaluation")
    starts = []
    for address in range(spec.depth):
        base = address * 1000
        sample = crossing(t, traces["SAMPLE"], base + 200, base + 750, 0.1 * vdd, True)
        starts.append(sample)
    for address, sample in enumerate(starts):
        base = address * 1000
        if sample is None:
            failures.append(
                f"address {address}: no sample (replica/controller stalled)"
            )
            continue
        for net in ("PRECHN", "EVAL", "SAMPLE", "READ_READY"):
            if np.interp(base + 180, t, traces[net]) > 0.1 * vdd:
                failures.append(
                    f"address {address}: {net} not idle before clock request"
                )
        for col in range(spec.columns):
            if np.interp(base + 180, t, traces[f"BL<{col}>"]) < 0.9 * vdd:
                failures.append(f"address {address}: bitline {col} not restored")
        # The next request starts at +1200 ps; its precharge checkpoint is
        # +1180 ps. Return may cross the nominal cycle boundary (+1000).
        end = base + 1180
        s_off = crossing(t, traces["SAMPLE"], base + 750, end, 0.1 * vdd, False)
        sb_on = crossing(t, traces["SAMPLEB"], base + 750, end, 0.9 * vdd, True)
        e_off = crossing(t, traces["EVAL"], base + 750, end, 0.9 * vdd, False)
        p_on = crossing(t, traces["PRECHN"], base + 750, end, 0.9 * vdd, False)
        if None in (s_off, sb_on, e_off, p_on):
            failures.append(f"address {address}: incomplete return to precharge")
            continue
        closure = max(s_off, sb_on)
        candidate_delay = settling_delay(t, candidates, base + 750, p_on, False, vdd)
        close_margin = e_off - closure
        if close_margin < 0:
            failures.append(
                f"address {address}: evaluation ended before latch input closure"
            )
        if candidate_delay is None:
            failures.append(f"address {address}: candidates not low before precharge")
        read_margins = []
        for bit in range(spec.bits):
            high = bool(words[address] >> bit & 1)
            valid_delay = settling_delay(
                t, traces[f"DATA<{bit}>"], base + 200, sample, high, vdd
            )
            if valid_delay is None:
                failures.append(
                    f"address {address}: SAMPLE precedes DATA<{bit}> validity"
                )
            else:
                read_margins.append(sample - (base + 200 + valid_delay))
            hold_end = (
                starts[address + 1]
                if address + 1 < len(starts) and starts[address + 1] is not None
                else t[-1]
            )
            q = traces[f"Q<{bit}>"][(t >= closure) & (t < hold_end)]
            good = q >= 0.9 * vdd if high else q <= 0.1 * vdd
            if not len(q) or not np.all(good):
                failures.append(f"address {address}: Q<{bit}> read/hold failure")
        records.append(
            {
                "address": address,
                "sample_ps": sample - base,
                "read_margin_ps": min(read_margins)
                if len(read_margins) == spec.bits
                else None,
                "closure_margin_ps": close_margin,
                "candidate_off_margin_ps": None
                if candidate_delay is None
                else p_on - (base + 750 + candidate_delay),
            }
        )
    abort = (t >= spec.depth * 1000) & (t <= t[-1])
    if np.any(traces["SAMPLE"][abort] > 0.1 * vdd):
        failures.append("reset-canceled request sampled data")
    for net in ("PRECHN", "EVAL", "SAMPLE", "READ_READY", "CANDIDATE_BUSY"):
        if traces[net][-1] > 0.1 * vdd:
            failures.append(f"{net} did not return idle after reset")
    margins = {}
    if not failures:
        margins = {
            key: min(r[key] for r in records)
            for key in (
                "read_margin_ps",
                "closure_margin_ps",
                "candidate_off_margin_ps",
            )
        }
    for record in records:
        for key in (
            "sample_ps",
            "read_margin_ps",
            "closure_margin_ps",
            "candidate_off_margin_ps",
        ):
            if record[key] is not None:
                record[key] *= phase_scale
    margins = {key: value * phase_scale for key, value in margins.items()}
    return {
        "passed": not failures,
        "failures": failures,
        "minimum_margins_ps": margins,
        "reads": records,
    }


def characterize_replica(
    spec,
    *,
    out,
    model_card,
    osdi,
    conditions=None,
    replica=None,
    words=None,
    klu=False,
    sized_controls=False,
    wire_rc_controls=False,
    startup_ramp_ps=0,
):
    conditions = conditions or TimingConditions(step_ps=0.5)
    replica = replica or ReplicaSpec()
    words = pattern_words(spec) if words is None else words
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "passed": False,
        "timing_qualified": False,
        "scope": (
            "selected control nets: routed-length Rpath pi stress model"
            if wire_rc_controls
            else "internally generated acknowledgments, ideal wires"
        ),
        "wire_rc_controls": wire_rc_controls,
        "sized_controls": sized_controls,
        "linear_solver": "KLU" if klu else "SPARSE",
        "startup_ramp_ps": startup_ramp_ps,
        "spec": asdict(spec),
        "replica": asdict(replica),
        "conditions": asdict(conditions),
        "words": words,
    }
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        card = Path(model_card).read_text()
        model = out / "models.pm"
        model.write_text(
            "\n".join(
                translate_asap7_model(card, f"{kind}mos_{spec.vt}")
                for kind in ("n", "p")
            )
        )
        physical = self_timed_macro_circuit(spec, words, replica=replica)
        if sized_controls:
            size_control_distribution(physical)
        if wire_rc_controls:
            physical = size_wire_rc_distribution(physical)
            distribution_path = out / "distribution.json"
            distribution_path.write_text(
                json.dumps(physical.distribution, indent=2) + "\n"
            )
            report["distribution_sha256"] = hashlib.sha256(
                distribution_path.read_bytes()
            ).hexdigest()
            report["extracted"] = False
            report["register_clock_rc_included"] = spec.registered_output
            if spec.registered_output:
                routes = routed_control_rc(physical)
                report["register_clock_rc"] = {
                    net: routes[net]
                    for net in ("CLK", "REG_CLKB", "REG_M", "REG_MB", "REG_S", "REG_SB")
                }
        report["circuit_sha256"] = hashlib.sha256(
            physical.netlist().encode()
        ).hexdigest()
        deck, signals = render_replica_bench(
            spec,
            words,
            replica,
            conditions,
            model,
            osdi,
            klu=klu,
            sized_controls=sized_controls,
            wire_rc_controls=wire_rc_controls,
            startup_ramp_ps=startup_ramp_ps,
        )
        spice = out / "replica.sp"
        spice.write_text(deck)
        _run_ngspice(spice, "ngspice", [out / "waveform.dat"])
        values = np.loadtxt(out / "waveform.dat", skiprows=1)
        if (
            values.ndim != 2
            or values.shape[1] != len(signals) + 1
            or not np.isfinite(values).all()
            or values[-1, 0]
            < ((spec.depth + 1) * 1000 + 200)
            * 1e-12
            * conditions.phase_scale
            * (1 - 1e-6)
            or np.any(np.diff(values[:, 0]) <= 0)
        ):
            raise RuntimeError("incomplete/invalid replica transient")
        (out / "signals.json").write_text(json.dumps(signals, indent=2) + "\n")
        report.update(
            analyze_replica(
                values[:, 0],
                dict(zip(signals, values[:, 1:].T, strict=True)),
                spec,
                words,
                conditions.vdd,
                phase_scale=conditions.phase_scale,
            )
        )
        if report["passed"]:
            from .cycle_timing import measure_cycle_paths

            report["cycle_paths"] = measure_cycle_paths(
                values[:, 0],
                dict(zip(signals, values[:, 1:].T, strict=True)),
                spec,
                words,
                conditions.vdd,
                phase_scale=conditions.phase_scale,
            )
        report.update(
            status="passed" if report["passed"] else "failed",
            model_sha256=hashlib.sha256(Path(model_card).read_bytes()).hexdigest(),
            osdi_sha256=hashlib.sha256(Path(osdi).read_bytes()).hexdigest(),
            deck_sha256=hashlib.sha256(spice.read_bytes()).hexdigest(),
        )
    except Exception as exc:
        report.update(status="error", error=str(exc))
        path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--bits", type=int, default=2)
    p.add_argument("--registered-output", action="store_true")
    p.add_argument("--keeper-fins", type=int, choices=(0, 1, 2, 3), default=0)
    p.add_argument("--keeper-stack", type=int, choices=(1, 2, 3, 4), default=2)
    p.add_argument("--fins", type=int, choices=(1, 2, 3), default=2)
    p.add_argument("--ways", type=int, choices=(1, 4), default=1)
    p.add_argument(
        "--klu", action="store_true", help="Use KLU (requires a KLU-enabled ngspice)"
    )
    sizing = p.add_mutually_exclusive_group()
    sizing.add_argument("--sized-controls", action="store_true")
    sizing.add_argument("--wire-rc-controls", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--osdi", type=Path, required=True)
    p.add_argument("--model-card", type=Path, default=find_default_model_card())
    p.add_argument("--vdd", type=float, default=0.7)
    p.add_argument("--temperature-c", type=float, default=25)
    p.add_argument("--guard-pairs", type=int, default=4)
    p.add_argument("--extra-blocks", type=int, default=1)
    p.add_argument("--phase-scale", type=float, default=1)
    p.add_argument("--step-ps", type=float, default=0.5)
    p.add_argument("--bitline-load-ff", type=float, default=0)
    p.add_argument("--output-load-ff", type=float, default=1)
    p.add_argument("--slew-ps", type=float, default=5)
    p.add_argument("--startup-ramp-ps", type=float, default=0)
    args = p.parse_args()
    if args.model_card is None:
        p.error("model card required")
    r = characterize_replica(
        BromSpec(
            args.depth,
            args.bits,
            args.ways,
            fins=args.fins,
            keeper_fins=args.keeper_fins,
            keeper_stack=args.keeper_stack,
            registered_output=args.registered_output,
        ),
        out=args.out,
        model_card=args.model_card,
        osdi=args.osdi,
        replica=ReplicaSpec(args.guard_pairs, args.extra_blocks),
        klu=args.klu,
        sized_controls=args.sized_controls,
        wire_rc_controls=args.wire_rc_controls,
        startup_ramp_ps=args.startup_ramp_ps,
        conditions=TimingConditions(
            vdd=args.vdd,
            temperature_c=args.temperature_c,
            step_ps=args.step_ps,
            phase_scale=args.phase_scale,
            bitline_load_ff=args.bitline_load_ff,
            output_load_ff=args.output_load_ff,
            slew_ps=args.slew_ps,
        ),
    )
    print(json.dumps(r, indent=2))
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
