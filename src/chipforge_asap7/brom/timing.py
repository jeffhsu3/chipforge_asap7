"""Loaded transistor-level BROM transient characterization (no extracted wire RC)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..devices.spice import (
    _osdi_command_path,
    _spice_path,
    find_default_model_card,
    translate_asap7_model,
)
from .compiler import BromSpec, macro_circuit


def _run_ngspice(deck: Path, executable: str, expected_outputs, *, timeout_s=600):
    """Bound simulation runtime and retain a live log, including failed runs."""
    for output in expected_outputs:
        output.unlink(missing_ok=True)
    log = deck.with_suffix(".log")
    with log.open("w") as stream:
        result = subprocess.run(
            [executable, "-b", deck.name],
            cwd=deck.parent,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            check=False,
        )
    if result.returncode or any(not p.exists() for p in expected_outputs):
        raise RuntimeError(f"ngspice failed; see {log}")


@dataclass(frozen=True)
class TimingConditions:
    vdd: float = 0.7
    temperature_c: float = 25
    output_load_ff: float = 1
    bitline_load_ff: float = 0
    slew_ps: float = 5
    step_ps: float = 0.5
    phase_scale: float = 1

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("timing conditions must be finite")
        if min(self.vdd, self.slew_ps, self.step_ps, self.phase_scale) <= 0:
            raise ValueError("voltage, slew, timestep and phase scale must be positive")
        if min(self.output_load_ff, self.bitline_load_ff) < 0:
            raise ValueError("loads cannot be negative")
        if self.slew_ps >= 10 * self.phase_scale:
            raise ValueError("slew must fit inside phase guard intervals")
        if self.step_ps > self.slew_ps / 2:
            raise ValueError("timestep must resolve input edges")
        if self.temperature_c <= -273.15:
            raise ValueError("temperature must exceed absolute zero")


def pattern_words(spec: BromSpec) -> list[int]:
    """Rotate all sixteen four-bit selector patterns across columns/blocks."""
    words = []
    for address in range(spec.depth):
        block, offset, way = spec.decode(address)
        words.append(
            sum(
                ((((bit * spec.ways + way + 7 * block) % 16) >> offset) & 1) << bit
                for bit in range(spec.bits)
            )
        )
    return words


def settling_delay(time, voltage, start, end, target_high, vdd):
    """Delay to the last 10/90% threshold crossing, stable through window end.

    None means the window failed. A signal already valid has delay zero.
    Crossings are linearly interpolated; callers check the whole final window.
    """
    inside = (time > start) & (time < end)
    t = np.concatenate(([start], time[inside], [end]))
    v = np.interp(t, time, voltage)
    good = v >= 0.9 * vdd if target_high else v <= 0.1 * vdd
    if not good[-1]:
        return None
    bad = np.flatnonzero(~good)
    if not len(bad):
        return 0.0
    i = bad[-1]
    threshold = (0.9 if target_high else 0.1) * vdd
    crossing = t[i] + (threshold - v[i]) * (t[i + 1] - t[i]) / (v[i + 1] - v[i])
    return max(0.0, float(crossing - start))


def render_deck(spec, words, conditions, model_card, osdi, addresses=None):
    c = macro_circuit(spec, words)
    nodes = sorted({n for d in c.devices for n in d.nodes} | set(c.pins))
    names = {n: f"n{i}" for i, n in enumerate(nodes)}
    names.update(VSS="0", VDD="vdd")
    signals = ["PRECHN", "EVAL", "SAMPLE"]
    signals += [f"BL<{i}>" for i in range(spec.columns)]
    signals += [f"DOUT<{i}>" for i in range(spec.columns)]
    signals += [f"Q[{i}]" for i in range(spec.bits)]
    data = {}
    by_name = {d.name: d for d in c.devices}
    for group, members in c.placement_groups:
        if group.startswith("output_latch_"):
            bit = int(group.removeprefix("output_latch_"))
            data[f"DATA<{bit}>"] = by_name[members[0]].nodes[2]
    signals += list(data)
    addresses = (
        list(addresses)
        if addresses is not None
        else [*range(spec.depth), *reversed(range(spec.depth))]
    )
    if not addresses or any(
        type(a) is not int or not 0 <= a < spec.depth for a in addresses
    ):
        raise ValueError("read sequence must contain valid addresses")
    scale = conditions.phase_scale
    cycle = 400 * scale
    stop = len(addresses) * cycle + 280 * scale
    events = {pin: [(0.0, 0)] for pin in ("PRECHN", "EVAL", "SAMPLE")}
    events.update({f"A[{i}]": [(0.0, 0)] for i in range(spec.address_bits)})
    for index, address in enumerate([*addresses, 0]):
        base = index * cycle
        for name, points in {
            "PRECHN": [(0, 0), (120, 1)],
            "EVAL": [(140, 1), (360, 0)],
            "SAMPLE": [(280, 1), (340, 0)],
        }.items():
            for offset, value in points:
                if base + offset * scale < stop:
                    events[name].append((base + offset * scale, value))
        for bit in range(spec.address_bits):
            events[f"A[{bit}]"].append((base + 20 * scale, (address >> bit) & 1))
            # Deliberately disturb the external address after the latch closes.
            events[f"A[{bit}]"].append((base + 200 * scale, 1 - ((address >> bit) & 1)))
    lines = [
        "* BROM loaded BSIM-CMG timing; ideal wires, no layout RC",
        f".include {_spice_path(model_card)}",
        f".temp {conditions.temperature_c:g}",
        ".options reltol=1e-3 abstol=1e-12 vntol=1e-6 method=gear rshunt=1e15",
        f"Vsupply vdd 0 {conditions.vdd:g}",
    ]
    for i, (net, changes) in enumerate(events.items()):
        points, old = ["0 0"], 0
        for t, value in sorted(changes):
            if t > stop or value == old:
                continue
            points += [
                f"{t:g}p {old * conditions.vdd:g}",
                f"{t + conditions.slew_ps:g}p {value * conditions.vdd:g}",
            ]
            old = value
        lines.append(f"Vstim{i} {names[net]} 0 PWL({' '.join(points)})")
    # Select a legal initial state for closed feedback latches. NODESET guides
    # the DC solve; it does not force nodes during the transient.
    for group, members in c.placement_groups:
        if group.startswith(("address_", "output_latch_")):
            storage = by_name[members[0]].nodes[0]
            qb = by_name[members[2]].nodes[0]
            q = by_name[members[4]].nodes[0]
            lines.append(
                f".nodeset V({names[storage]})=0 V({names[qb]})={conditions.vdd:g} V({names[q]})=0"
            )
    for d in c.devices:
        for multiplier in range(d.spec.multipliers):
            lines.append(
                f"N{d.name}_{multiplier} {' '.join(names[n] for n in d.nodes)} "
                f"{d.spec.model} L={d.spec.gate_length}n NFIN={d.spec.fins} NF={d.spec.fingers}"
            )
    # Q is a macro pin, so its bits are bracketed; BL is internal.
    for prefix, net, count, load in (
        ("Q", "Q[{}]", spec.bits, conditions.output_load_ff),
        ("BL", "BL<{}>", spec.columns, conditions.bitline_load_ff),
    ):
        if load:
            lines += [
                f"C{prefix}{i} {names[net.format(i)]} 0 {load:g}f" for i in range(count)
            ]
    vectors = [f"v({names[data.get(n, n)]})" for n in signals]
    lines += [
        ".control",
        "set num_threads=1",
        f"pre_osdi {_osdi_command_path(osdi)}",
        "set wr_singlescale",
        "set wr_vecnames",
        "set numdgt=12",
        "save " + " ".join(vectors),
        f"tran {conditions.step_ps:g}p {stop:g}p 0 {conditions.step_ps:g}p",
        "wrdata waveform.dat " + " ".join(vectors),
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines), signals, addresses, names


def analyze(time, traces, spec, words, conditions, addresses):
    records, failures = [], []
    scale = conditions.phase_scale * 1e-12
    slew = conditions.slew_ps * 1e-12
    vdd = conditions.vdd
    for index, address in enumerate(addresses):
        base = index * 400 * scale
        eval_start = base + 140 * scale + slew / 2
        sample_start = base + 280 * scale + slew / 2
        record = {"address": address, "word": words[address]}
        pre, discharge, read, capture = [], [], [], []
        for col in range(spec.columns):
            pre.append(
                settling_delay(
                    time,
                    traces[f"BL<{col}>"],
                    base + slew / 2,
                    base + 120 * scale,
                    True,
                    vdd,
                )
            )
        for bit in range(spec.bits):
            high = bool(words[address] >> bit & 1)
            col = bit * spec.ways + spec.decode(address)[2]
            if high:
                discharge.append(
                    settling_delay(
                        time,
                        traces[f"BL<{col}>"],
                        eval_start,
                        base + 280 * scale,
                        False,
                        vdd,
                    )
                )
            read.append(
                settling_delay(
                    time,
                    traces[f"DATA<{bit}>"],
                    eval_start,
                    base + 280 * scale,
                    high,
                    vdd,
                )
            )
            capture.append(
                settling_delay(
                    time,
                    traces[f"Q[{bit}]"],
                    sample_start,
                    base + 340 * scale,
                    high,
                    vdd,
                )
            )
            # Hold through the following precharge/evaluation, before SAMPLE opens.
            start, end = base + 340 * scale + slew, base + 680 * scale
            if settling_delay(time, traces[f"Q[{bit}]"], start, end, high, vdd) != 0:
                failures.append(
                    f"read {index} address {address}: Q[{bit}] hold failure"
                )
        for metric, delays in (
            ("precharge_90_ps", pre),
            ("bitline_discharge_10_ps", discharge),
            ("eval_to_data_valid_ps", read),
            ("sample_to_q_valid_ps", capture),
        ):
            if any(d is None for d in delays):
                record[metric] = None
                failures.append(
                    f"read {index} address {address}: {metric} did not settle"
                )
            else:
                record[metric] = max(delays, default=0) * 1e12
        records.append(record)
    metrics = (
        "precharge_90_ps",
        "bitline_discharge_10_ps",
        "eval_to_data_valid_ps",
        "sample_to_q_valid_ps",
    )
    maxima = {
        m: max((r[m] for r in records if r[m] is not None), default=None)
        for m in metrics
    }
    if failures:
        maxima = {}  # Failed windows must never be summarized as passing timing.
    return {
        "passed": not failures,
        "failures": failures,
        "worst_ps": maxima,
        "reads": records,
    }


def characterize(
    spec, words, conditions, *, model_card, osdi, out, ngspice="ngspice", addresses=None
):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    report_path.write_text(
        json.dumps({"status": "running", "passed": False, "timing_qualified": False})
        + "\n"
    )
    report = {
        "status": "running",
        "scope": "loaded_transistor_level_ideal_wires",
        "bench_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing_qualified": False,
        "spec": asdict(spec),
        "conditions": asdict(conditions),
        "numerics": {
            "method": "gear",
            "reltol": 1e-3,
            "abstol_a": 1e-12,
            "vntol_v": 1e-6,
            "rshunt_ohm": 1e15,
            "dc_operating_point": True,
            "threads": 1,
        },
        "words": words,
        "model_card": str(Path(model_card).resolve()),
        "model_sha256": hashlib.sha256(Path(model_card).read_bytes()).hexdigest(),
        "osdi_sha256": hashlib.sha256(Path(osdi).read_bytes()).hexdigest(),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        version = subprocess.run(
            [ngspice, "--version"], capture_output=True, text=True, check=True
        )
        report["ngspice_version"] = version.stdout.strip()
        translated = out / "models.pm"
        card = Path(model_card).read_text()
        translated.write_text(
            "\n".join(
                translate_asap7_model(card, f"{p}mos_{spec.vt}") for p in ("n", "p")
            )
        )
        deck, signals, addresses, names = render_deck(
            spec, words, conditions, translated, osdi, addresses
        )
        report["read_sequence"] = addresses
        path = out / "timing.sp"
        path.write_text(deck)
        (out / "nodes.json").write_text(json.dumps(names, indent=2) + "\n")
        (out / "signals.json").write_text(json.dumps(signals, indent=2) + "\n")
        _run_ngspice(path, ngspice, [out / "waveform.dat"])
        samples = np.loadtxt(out / "waveform.dat", skiprows=1)
        if (
            samples.ndim != 2
            or samples.shape[1] != len(signals) + 1
            or not np.isfinite(samples).all()
        ):
            raise RuntimeError("invalid waveform data")
        time = samples[:, 0]
        expected_end = (len(addresses) * 400 + 280) * conditions.phase_scale * 1e-12
        if time[-1] < expected_end * (1 - 1e-6) or np.any(np.diff(time) <= 0):
            raise RuntimeError("incomplete or nonmonotonic transient")
        report.update(
            analyze(
                time,
                dict(zip(signals, samples[:, 1:].T, strict=True)),
                spec,
                words,
                conditions,
                addresses,
            )
        )
        report["status"] = "passed" if report["passed"] else "failed"
        report["deck_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception as exc:
        report.update(status="error", passed=False, error=str(exc))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def plot_waveform(out: Path) -> Path:
    """Save a shareable plot of the first two reads, including held outputs."""
    os.environ.setdefault("MPLCONFIGDIR", str(out.resolve() / "matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    report = json.loads((out / "report.json").read_text())
    signals = json.loads((out / "signals.json").read_text())
    data = np.loadtxt(out / "waveform.dat", skiprows=1)
    time = data[:, 0] * 1e12
    traces = dict(zip(signals, data[:, 1:].T, strict=True))
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 7), sharex=True, constrained_layout=True
    )
    bit = min(1, report["spec"]["bits"] - 1)
    spec = BromSpec(**report["spec"])
    sequence = report.get(
        "read_sequence", [*range(spec.depth), *reversed(range(spec.depth))]
    )
    cycle = 400 * report["conditions"]["phase_scale"]
    which = np.minimum((time / cycle).astype(int), len(sequence) - 1)
    columns = np.array(
        [bit * spec.ways + spec.decode(address)[2] for address in sequence]
    )
    traces["Selected BL"] = np.array(
        [traces[f"BL<{columns[index]}>"][i] for i, index in enumerate(which)]
    )
    for ax, names in zip(
        axes,
        [("PRECHN", "EVAL", "SAMPLE"), ("Selected BL", f"DATA<{bit}>"), (f"Q[{bit}]",)],
        strict=True,
    ):
        for name in names:
            ax.plot(time, traces[name], label=name)
        ax.set_ylabel("Voltage (V)")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.25)
    axes[-1].set(
        xlabel="Time (ps)", xlim=(0, 800 * report["conditions"]["phase_scale"])
    )
    fig.suptitle("BROM loaded transistor timing — ideal wires, no extracted RC")
    path = out / "waveform.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--words", type=Path, help="Optional JSON ROM image; overrides depth"
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--addresses",
        type=int,
        nargs="+",
        help="Optional read sequence (default: forward then reverse)",
    )
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--ways", type=int, choices=(1, 4), default=1)
    parser.add_argument("--fins", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--vt", default="rvt", choices=("rvt", "lvt", "slvt", "sram"))
    parser.add_argument("--model-card", type=Path, default=find_default_model_card())
    parser.add_argument("--osdi", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    for name, field in TimingConditions.__dataclass_fields__.items():
        parser.add_argument(
            "--" + name.replace("_", "-"), type=float, default=field.default
        )
    args = parser.parse_args()
    if args.model_card is None:
        parser.error("ASAP7 model card not found; pass --model-card")
    words = json.loads(args.words.read_text()) if args.words else None
    spec = BromSpec(
        len(words) if words is not None else args.depth,
        args.bits,
        args.ways,
        args.fins,
        args.vt,
    )
    conditions = TimingConditions(
        **{n: getattr(args, n) for n in TimingConditions.__dataclass_fields__}
    )
    report = characterize(
        spec,
        words if words is not None else pattern_words(spec),
        conditions,
        model_card=args.model_card,
        osdi=args.osdi,
        out=args.out,
        addresses=args.addresses,
    )
    print(
        json.dumps({k: report[k] for k in ("status", "worst_ps", "failures")}, indent=2)
    )
    if args.plot:
        plot_waveform(args.out)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
