"""Sampled setup/hold and pulse-width bounds for the standalone output register.

All trials recondition the same physical register. Bounds use a fixed 10%
clock-to-valid degradation criterion and both data polarities, not metastability
qualification or Liberty constraints. Wire parasitics are excluded here.
"""

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
from .output_register import output_register_circuit
from .replica_timing import crossing
from .timing import TimingConditions, _run_ngspice

OFFSETS_PS = (-40, -20, -10, 0, 10, 20, 40, 80, 120)
WIDTHS_PS = (10, 20, 40, 80, 160, 320)
SIGNALS = ("CLK", "D<0>", "Q<0>", "REG_M", "REG_MB", "REG_S", "REG_SB")


def trials():
    result = []
    for kind, samples in (
        ("reference", (200,)),
        ("setup", OFFSETS_PS),
        ("hold", OFFSETS_PS),
        ("high_width", WIDTHS_PS),
        ("low_width", WIDTHS_PS),
    ):
        for value in samples:
            for high in (False, True):
                result.append({"kind": kind, "value_ps": value, "high": high})
    return result


def render_register_bench(
    conditions, model_card, osdi, *, fins=2, vt="rvt", klu=False, startup_ramp_ps=0
):
    if (
        not math.isfinite(startup_ramp_ps)
        or not 0 <= startup_ramp_ps < 50 - conditions.slew_ps / 2
    ):
        raise ValueError("startup ramp must finish before the first register data edge")
    c = output_register_circuit(fins=fins, vt=vt)
    nodes = {
        n: f"n{i}"
        for i, n in enumerate(sorted({n for d in c.devices for n in d.nodes}))
    }
    nodes.update(VDD="vdd", VSS="0")
    slew = conditions.slew_ps
    # Times below denote ramp midpoints; offsets and pulse widths are 50%-50%.
    edges = {"CLK": [(0, 0)], "D<0>": [(0, 0)]}
    records = trials()
    for i, r in enumerate(records):
        base = i * 2800
        kind, width, high = r["kind"], r["value_ps"], int(r["high"])
        low_edge = 1500 - width if kind == "low_width" else 800
        high_edge = 1500 + width if kind == "high_width" else 1900
        edges["CLK"] += [
            (base + 400, 1),
            (base + low_edge, 0),
            (base + 1500, 1),
            (base + high_edge, 0),
        ]
        # First capture deliberately establishes the opposite value.
        edges["D<0>"] += [(base + 50, 1 - high)]
        change = (
            1500 - width
            if kind == "setup"
            else low_edge + slew
            if kind == "low_width"
            else 900
        )
        edges["D<0>"] += [(base + change, high)]
        if kind == "hold":
            edges["D<0>"] += [(base + 1500 + width, 1 - high)]
    lines = [
        "* Standalone BROM output register timing trials; ideal wires",
        f".include {_spice_path(model_card)}",
        f".temp {conditions.temperature_c:g}",
        ".options "
        + ("klu " if klu else "")
        + "method=gear reltol=1e-3 abstol=1e-12 vntol=1e-6 rshunt=1e15",
        (
            f"Vsupply vdd 0 PWL(0 0 {startup_ramp_ps:g}p {conditions.vdd:g})"
            if startup_ramp_ps
            else f"Vsupply vdd 0 {conditions.vdd:g}"
        ),
    ]
    for i, (net, events) in enumerate(edges.items()):
        pts, old = ["0 0"], 0
        for t, value in events[1:]:
            if value != old:
                pts += [
                    f"{t - slew / 2:.12g}p {old * conditions.vdd:g}",
                    f"{t + slew / 2:.12g}p {value * conditions.vdd:g}",
                ]
            old = value
        lines.append(f"Vstim{i} {nodes[net]} 0 PWL({' '.join(pts)})")
    for d in c.devices:
        lines.append(
            f"N{d.name} {' '.join(nodes[n] for n in d.nodes)} {d.spec.model} L=20n NFIN={d.spec.fins} NF=1"
        )
    if conditions.output_load_ff:
        lines.append(f"CQ {nodes['Q<0>']} 0 {conditions.output_load_ff:g}f")
    by_name = {d.name: d for d in c.devices}
    for _, members in c.placement_groups:
        if len(members) == 8:
            for index, v in ((0, 0), (2, conditions.vdd), (4, 0)):
                lines.append(
                    f".nodeset V({nodes[by_name[members[index]].nodes[0]]})={v:g}"
                )
    vectors = " ".join(f"v({nodes[n]})" for n in SIGNALS)
    stop = len(records) * 2800
    lines += [
        ".control",
        "set num_threads=1",
        f"pre_osdi {_osdi_command_path(osdi)}",
        "set wr_singlescale",
        "set wr_vecnames",
        "set numdgt=17",
        "save " + vectors,
        f"tran {conditions.step_ps:g}p {stop}p 0 {conditions.step_ps:g}p",
        "wrdata waveform.dat " + vectors,
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines), records


def sampled_bound(records, kind):
    """Only accept a passing suffix; expose failures/nonmonotonicity and bounds."""
    values = sorted({r["value_ps"] for r in records if r["kind"] == kind})
    passing = {}
    for v in values:
        group = [r for r in records if r["kind"] == kind and r["value_ps"] == v]
        passing[v] = {r["high"] for r in group} == {False, True} and all(
            r["passed"] for r in group
        )
    suffix = [v for v in values if all(passing[w] for w in values if w >= v)]
    upper = min(suffix) if suffix else None
    failed = [v for v in values if not passing[v]]
    return {
        "passing_bound_ps": upper,
        "largest_failing_sample_ps": max(failed) if failed else None,
        "boundary_bracketed": upper is not None and bool(failed),
        "nonmonotonic": any(
            passing[a] and not passing[b] for a in values for b in values if a < b
        ),
        "samples_ps": values,
    }


def analyze_register_trials(time, traces, records, vdd):
    t = np.asarray(time) * 1e12
    # ngspice's text precision can merge adjacent sub-attosecond timesteps.
    # Coalesce only voltage-equivalent duplicates; never hide a discontinuity.
    duplicate = np.flatnonzero(np.diff(t) == 0)
    for n in SIGNALS:
        if duplicate.size and np.any(
            np.abs(traces[n][duplicate + 1] - traces[n][duplicate]) > 1e-6 * vdd
        ):
            raise ValueError("discontinuous duplicate register timestamps")
    keep = np.r_[np.diff(t) != 0, True]
    t = t[keep]
    traces = {n: np.asarray(v)[keep] for n, v in traces.items()}
    if (
        len(t) < 2
        or not np.isfinite(t).all()
        or np.any(np.diff(t) <= 0)
        or t[0] > 1e-3
        or t[-1] < len(records) * 2800 - 1e-3
        or any(
            len(traces[n]) != len(t) or not np.isfinite(traces[n]).all()
            for n in SIGNALS
        )
    ):
        raise ValueError("incomplete/invalid register transient")
    q = traces["Q<0>"]
    result, refs = [], {}
    for i, trial in enumerate(records):
        r = dict(trial)
        base, high = i * 2800, r["high"]
        edge = crossing(t, traces["CLK"], base + 1490, base + 1510, 0.5 * vdd, True)
        failures = []
        prior = q[(t >= base + 700) & (t < (edge if edge is not None else base + 1490))]
        if not np.all(prior <= 0.1 * vdd if high else prior >= 0.9 * vdd):
            failures.append("conditioning or pre-edge retention")
        valid = (
            None
            if edge is None
            else crossing(t, q, edge, base + 2200, (0.9 if high else 0.1) * vdd, high)
        )
        delay = None if valid is None else valid - edge
        if valid is None:
            failures.append("missing capture")
        else:
            held = q[(t >= valid + 1e-6) & (t <= base + 2200)]
            if not len(held) or not np.all(
                held >= 0.9 * vdd if high else held <= 0.1 * vdd
            ):
                failures.append("output failed to hold captured data")
        active = (t >= base + 700) & (t <= base + 2200)
        master = (traces["REG_M"] > 0.2 * vdd) | (traces["REG_MB"] < 0.8 * vdd)
        slave = (traces["REG_S"] > 0.2 * vdd) | (traces["REG_SB"] < 0.8 * vdd)
        if np.any(active & master & slave):
            failures.append("master/slave input overlap")
        if r["kind"] == "reference":
            refs[high] = delay if not failures else None
        elif refs.get(high) is None:
            failures.append("reference invalid")
        elif delay is not None and delay > 1.1 * refs[high]:
            failures.append("clock-to-valid degraded over 10%")
        r.update(passed=not failures, failures=failures, clk_to_valid_ps=delay)
        result.append(r)
    bounds = {
        kind: sampled_bound(result, kind)
        for kind in ("setup", "hold", "high_width", "low_width")
    }
    return {
        "passed": all(v is not None for v in refs.values())
        and len(refs) == 2
        and all(
            b["passing_bound_ps"] is not None and not b["nonmonotonic"]
            for b in bounds.values()
        ),
        "reference_clk_to_valid_ps": {
            "rise" if k else "fall": v for k, v in refs.items()
        },
        "bounds": bounds,
        "trials": result,
    }


def characterize_register(
    *,
    out,
    model_card,
    osdi,
    conditions=None,
    fins=2,
    vt="rvt",
    klu=False,
    startup_ramp_ps=0,
):
    conditions = conditions or TimingConditions()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "passed": False,
        "timing_qualified": False,
        "setup_hold_qualified": False,
        "scope": "standalone register, ideal wires, sampled bounds",
        "conditions": asdict(conditions),
        "fins": fins,
        "vt": vt,
        "linear_solver": "KLU" if klu else "SPARSE",
        "delay_degradation_limit": 0.1,
        "startup_ramp_ps": startup_ramp_ps,
        "register_cell_sha256": hashlib.sha256(
            output_register_circuit(fins=fins, vt=vt).netlist().encode()
        ).hexdigest(),
    }
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        model = out / "models.pm"
        card = Path(model_card).read_text()
        model.write_text(
            "\n".join(
                translate_asap7_model(card, f"{kind}mos_{vt}") for kind in ("n", "p")
            )
        )
        deck, records = render_register_bench(
            conditions,
            model,
            osdi,
            fins=fins,
            vt=vt,
            klu=klu,
            startup_ramp_ps=startup_ramp_ps,
        )
        spice = out / "register.sp"
        spice.write_text(deck)
        report.update(
            model_sha256=hashlib.sha256(Path(model_card).read_bytes()).hexdigest(),
            osdi_sha256=hashlib.sha256(Path(osdi).read_bytes()).hexdigest(),
            deck_sha256=hashlib.sha256(spice.read_bytes()).hexdigest(),
        )
        _run_ngspice(spice, "ngspice", [out / "waveform.dat"])
        values = np.loadtxt(out / "waveform.dat", skiprows=1)
        if values.ndim != 2 or values.shape[1] != len(SIGNALS) + 1:
            raise ValueError("invalid register waveform columns")
        report.update(
            analyze_register_trials(
                values[:, 0],
                dict(zip(SIGNALS, values[:, 1:].T, strict=True)),
                records,
                conditions.vdd,
            )
        )
        report["status"] = "passed" if report["passed"] else "failed"
        (out / "signals.json").write_text(json.dumps(SIGNALS) + "\n")
    except Exception as exc:
        report.update(status="error", error=str(exc))
        path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--osdi", type=Path, required=True)
    p.add_argument("--model-card", type=Path, default=find_default_model_card())
    p.add_argument("--vdd", type=float, default=0.7)
    p.add_argument("--temperature-c", type=float, default=25)
    p.add_argument("--slew-ps", type=float, default=5)
    p.add_argument("--output-load-ff", type=float, default=1)
    p.add_argument("--step-ps", type=float, default=0.5)
    p.add_argument("--klu", action="store_true")
    p.add_argument("--startup-ramp-ps", type=float, default=0)
    args = p.parse_args()
    r = characterize_register(
        out=args.out,
        model_card=args.model_card,
        osdi=args.osdi,
        klu=args.klu,
        startup_ramp_ps=args.startup_ramp_ps,
        conditions=TimingConditions(
            vdd=args.vdd,
            temperature_c=args.temperature_c,
            slew_ps=args.slew_ps,
            output_load_ff=args.output_load_ff,
            step_ps=args.step_ps,
        ),
    )
    print(json.dumps(r, indent=2))
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
