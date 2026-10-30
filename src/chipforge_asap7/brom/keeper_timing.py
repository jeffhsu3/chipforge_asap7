"""Bitline-level keeper retention, discharge and contention characterization.

Real selector/isolation device loads and readout transistors; ideal wires.
Optional imposed leakage is a labeled stress, separate from model leakage.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..devices.spice import _osdi_command_path, _spice_path, translate_asap7_model
from .compiler import BromSpec, output_circuit
from .timing import _run_ngspice


def render_keeper_bench(
    spec,
    model,
    osdi,
    *,
    read,
    vdd=0.7,
    temperature_c=25,
    hold_ns=100,
    bitline_load_ff=0,
    leakage_na=0,
    klu=False,
):
    c = output_circuit(
        spec.fins, spec.vt, keeper_fins=spec.keeper_fins, keeper_stack=spec.keeper_stack
    )
    # One selected live FET plus all other live/isolation drains. In retention
    # all selector gates are grounded; this also covers an all-zero word.
    for i in range(2 * spec.blocks):
        c.fet(False, "BL", "SEL" if i == 0 else "VSS", "VSS")
    lines = [
        "* BROM bitline keeper comparison; ideal wires",
        f".include {_spice_path(model)}",
        f".temp {temperature_c:g}",
        ".options "
        + ("klu " if klu else "")
        + "method=gear reltol=1e-4 abstol=1e-13 vntol=1e-7 rshunt=1e15",
        f"Vsupply VDD 0 {vdd:g}",
        "Vkeeper VDD vk 0",
        f"Vpre PRECHN 0 PWL(0 0 200p 0 205p {vdd:g})",
        f"Vsel SEL 0 PWL(0 0 400p 0 405p {vdd:g})" if read else "Vsel SEL 0 0",
        "CQ DOUT 0 1f",
        f".nodeset V(BL)={vdd:g} V(DOUT)=0",
    ]
    if bitline_load_ff:
        lines.append(f"CBL BL 0 {bitline_load_ff:g}f")
    if leakage_na:
        lines.append(f"Ileak BL 0 {leakage_na:g}n")
    for d in c.devices:
        nodes = ["0" if n == "VSS" else n for n in d.nodes]
        if d.spec.flavor == "p" and d.nodes[1:3] == ("DOUT", "VDD"):
            nodes[2] = "vk"
        lines.append(
            f"N{d.name} {' '.join(nodes)} {d.spec.model} L=20n NFIN={d.spec.fins} NF=1"
        )
    stop_ps = 2000 if read else 200 + hold_ns * 1000
    step_ps = 0.5 if read else 5
    lines += [
        ".control",
        "set num_threads=1",
        f"pre_osdi {_osdi_command_path(osdi)}",
        "set wr_singlescale",
        "set wr_vecnames",
        "set numdgt=12",
        f"tran {step_ps}p {stop_ps:g}p 0 {step_ps}p",
        "wrdata waveform.dat v(BL) v(DOUT) i(Vkeeper) v(SEL)",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines), stop_ps


def first_crossing(t, v, start, threshold, rising):
    ids = np.flatnonzero(
        (t >= start) & ((v >= threshold) if rising else (v <= threshold))
    )
    if not len(ids):
        return None
    i = ids[0]
    if i == 0 or t[i - 1] < start:
        return float(t[i])
    return float(
        t[i - 1] + (threshold - v[i - 1]) * (t[i] - t[i - 1]) / (v[i] - v[i - 1])
    )


def analyze_keeper(values, *, read, vdd, stop_ps):
    if (
        values.ndim != 2
        or values.shape[1] != 5
        or not np.isfinite(values).all()
        or np.any(np.diff(values[:, 0]) <= 0)
        or values[-1, 0] * 1e12 < stop_ps * (1 - 1e-6)
    ):
        raise ValueError("incomplete/invalid keeper transient")
    t = values[:, 0] * 1e12
    bl, dout, current, sel = values[:, 1:].T
    if np.interp(195, t, bl) < 0.9 * vdd:
        raise ValueError("bitline did not precharge before release")
    if read:
        start = first_crossing(t, sel, 400, 0.1 * vdd, True)
        if start is None:
            raise ValueError("selector did not evaluate")
        low = first_crossing(t, bl, 400, 0.1 * vdd, False)
        valid = first_crossing(t, dout, 400, 0.9 * vdd, True)
        ready = bool(
            np.interp(395, t, bl) >= 0.9 * vdd and np.interp(395, t, dout) <= 0.1 * vdd
        )
        good = low is not None and valid is not None
        end = max(low, valid) if good else t[-1]
        mask = (t >= start) & (t <= end)
        positive = np.maximum(current[mask], 0)
        charge_fc = float(np.trapezoid(positive, t[mask] * 1e-12) * 1e15)
        held = (bl[t >= end] <= 0.1 * vdd) & (dout[t >= end] >= 0.9 * vdd)
        return {
            "read_passed": bool(ready and good and np.all(held)),
            "restored_before_read": ready,
            "bitline_discharge_ps": None if low is None else low - start,
            "data_valid_ps": None if valid is None else valid - start,
            "peak_keeper_contention_ua": float(np.max(positive, initial=0) * 1e6),
            "keeper_contention_charge_fc": charge_fc,
            "keeper_contention_energy_fj": charge_fc * vdd,
        }
    loss90 = first_crossing(t, bl, 205, 0.9 * vdd, False)
    loss50 = first_crossing(t, bl, 205, 0.5 * vdd, False)
    return {
        "retention_90_ns": None if loss90 is None else (loss90 - 205) / 1000,
        "retention_50_ns": None if loss50 is None else (loss50 - 205) / 1000,
        "observed_hold_ns": (t[-1] - 205) / 1000,
        "minimum_bitline_v": float(np.min(bl[t >= 205])),
        "end_keeper_current_na": float(np.mean(current[t >= 0.9 * t[-1]]) * 1e9),
    }


def characterize_keeper(
    spec,
    *,
    out,
    model_card,
    osdi,
    vdd=0.7,
    temperature_c=25,
    hold_ns=100,
    bitline_load_ff=0,
    leakage_na=0,
    klu=False,
):
    if (
        not all(
            np.isfinite(v)
            for v in (vdd, temperature_c, hold_ns, bitline_load_ff, leakage_na)
        )
        or vdd <= 0
        or hold_ns <= 0
        or min(bitline_load_ff, leakage_na) < 0
        or temperature_c <= -273.15
    ):
        raise ValueError("invalid keeper characterization conditions")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "timing_qualified": False,
        "spec": asdict(spec),
        "linear_solver": "KLU" if klu else "SPARSE",
        "scope": "isolated bitline/readout, intrinsic model leakage, ideal wires",
        "conditions": {
            "vdd": vdd,
            "temperature_c": temperature_c,
            "hold_ns": hold_ns,
            "bitline_load_ff": bitline_load_ff,
            "imposed_leakage_na": leakage_na,
            "output_load_ff": 1,
        },
        "experiments": {},
    }
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        card = Path(model_card).read_text()
        model = out / "models.pm"
        model.write_text(
            "\n".join(
                translate_asap7_model(card, f"{k}mos_{spec.vt}") for k in ("n", "p")
            )
        )
        report["model_sha256"] = hashlib.sha256(
            Path(model_card).read_bytes()
        ).hexdigest()
        report["osdi_sha256"] = hashlib.sha256(Path(osdi).read_bytes()).hexdigest()
        for read in (True, False):
            folder = out / ("read" if read else "hold")
            folder.mkdir(exist_ok=True)
            deck, stop = render_keeper_bench(
                spec,
                model,
                osdi,
                read=read,
                vdd=vdd,
                temperature_c=temperature_c,
                hold_ns=hold_ns,
                bitline_load_ff=bitline_load_ff,
                leakage_na=leakage_na,
                klu=klu,
            )
            spice = folder / "keeper.sp"
            spice.write_text(deck)
            _run_ngspice(spice, "ngspice", [folder / "waveform.dat"])
            result = analyze_keeper(
                np.loadtxt(folder / "waveform.dat", skiprows=1),
                read=read,
                vdd=vdd,
                stop_ps=stop,
            )
            result["deck_sha256"] = hashlib.sha256(spice.read_bytes()).hexdigest()
            report["experiments"][folder.name] = result
        report["status"] = (
            "passed" if report["experiments"]["read"]["read_passed"] else "read_failed"
        )
    except Exception as exc:
        report.update(status="error", error=str(exc))
        path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--klu", action="store_true")
    p.add_argument("--blocks", type=int, default=16)
    p.add_argument("--fins", type=int, choices=(1, 2, 3), default=2)
    p.add_argument("--keeper-fins", type=int, choices=(0, 1, 2, 3), default=0)
    p.add_argument("--keeper-stack", type=int, choices=(1, 2, 3, 4), default=2)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-card", type=Path, required=True)
    p.add_argument("--osdi", type=Path, required=True)
    p.add_argument("--vdd", type=float, default=0.7)
    p.add_argument("--temperature-c", type=float, default=25)
    p.add_argument("--hold-ns", type=float, default=100)
    p.add_argument("--bitline-load-ff", type=float, default=0)
    p.add_argument("--leakage-na", type=float, default=0)
    args = p.parse_args()
    report = characterize_keeper(
        BromSpec(
            4 * args.blocks,
            1,
            fins=args.fins,
            keeper_fins=args.keeper_fins,
            keeper_stack=args.keeper_stack,
        ),
        out=args.out,
        model_card=args.model_card,
        osdi=args.osdi,
        vdd=args.vdd,
        temperature_c=args.temperature_c,
        hold_ns=args.hold_ns,
        bitline_load_ff=args.bitline_load_ff,
        leakage_na=args.leakage_na,
        klu=args.klu,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
