"""Emit a programmed BROM and its standalone peripheral cells."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ..layout.layers import require_gdspy
from ..verification.drc import drc_counts, run_gdscheck
from ..verification.lvs import run_lvs
from ..verification.lvs_report import summarize_lvsdb
from . import (
    BromSpec,
    ReplicaSpec,
    build_macro,
    build_periphery,
    cgen_circuit,
    clocked_macro_circuit,
    controller_circuit,
    decoder_circuit,
    macro_circuit,
    output_circuit,
    selector_map,
    self_timed_macro_circuit,
)
from .layout import macro_periphery, macro_plan
from .periphery import periphery_plan
from .views import macro_lef, macro_verilog


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("words", type=Path, help="JSON array of unsigned integer words")
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--ways", type=int, choices=(1, 4), default=1)
    parser.add_argument("--fins", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--vt", choices=("rvt", "lvt", "slvt", "sram"), default="rvt")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--self-timed",
        action="store_true",
        help="Experimental internal selector-replica feedback",
    )
    parser.add_argument("--replica-guard-pairs", type=int)
    parser.add_argument("--replica-extra-blocks", type=int)
    mode.add_argument(
        "--clocked",
        action="store_true",
        help="Integrate clock controller; completion feedback remains external",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--drc",
        action="store_true",
        help="Run full ASAP7 gdscheck on every emitted top",
    )
    parser.add_argument(
        "--lvs",
        action="store_true",
        help="Run unit-fin KLayout LVS on every emitted top",
    )
    args = parser.parse_args(argv)
    if not args.self_timed and (
        args.replica_guard_pairs is not None or args.replica_extra_blocks is not None
    ):
        parser.error("replica options require --self-timed")
    words = json.loads(args.words.read_text())
    if not isinstance(words, list):
        parser.error("words must contain a JSON array")
    spec = BromSpec(len(words), args.bits, args.ways, args.fins, args.vt)
    macro = (
        self_timed_macro_circuit(
            spec,
            words,
            replica=ReplicaSpec(
                4 if args.replica_guard_pairs is None else args.replica_guard_pairs,
                1 if args.replica_extra_blocks is None else args.replica_extra_blocks,
            ),
        )
        if args.self_timed
        else (clocked_macro_circuit if args.clocked else macro_circuit)(spec, words)
    )
    circuits = {
        "macro": macro,
        "cgen": cgen_circuit(spec.fins, spec.vt),
        "decoder": decoder_circuit(spec),
        "output": output_circuit(spec.fins, spec.vt),
    }
    if args.clocked or args.self_timed:
        circuits["controller"] = controller_circuit(spec.fins, spec.vt)
    args.out.mkdir(parents=True, exist_ok=True)
    bank_plan = periphery_plan(macro_periphery(macro))
    report = {
        "spec": asdict(spec),
        "top_cell": macro.name,
        "selector_map": selector_map(spec, words),
        "pins": macro.pins,
        "controller": (
            "replica_feedback"
            if args.self_timed
            else "external_completion_handshake"
            if args.clocked
            else "external_phases"
        ),
        "replica_integrated": bool(macro.replica),
        "replica": (
            {
                "blocks": len(macro.replica.selector.mapping),
                "columns": spec.columns,
                "guard_pairs": macro.replica.guard_pairs,
                "extra_blocks": macro.replica.extra_blocks,
                "candidate": 15,
                "selector_devices": len(macro.replica.selector.device_names) // 2,
                "isolation_devices": len(macro.replica.selector.device_names) // 2,
                "feedback_timing_qualified": False,
            }
            if macro.replica
            else None
        ),
        "layout": "compact_selector_array",
        "selector_plan": asdict(macro.array.plan),
        "array_core_nm": [
            macro_plan(macro).array_width,
            macro_plan(macro).array_height,
        ],
        "periphery_layout": "folded_local_rows",
        "periphery_plan": {
            "row_count": len(bank_plan.rows),
            "device_pitch_nm": bank_plan.device_pitch,
            "row_capacity": bank_plan.row_capacity,
            "width_nm": bank_plan.width,
            "height_nm": bank_plan.height,
            "shared_spines": len(bank_plan.spine_x),
            "pins": bank_plan.pins,
        },
        "periphery_origin_nm": [0, macro_plan(macro).bank_y],
        "array_interfaces": [asdict(p) for p in macro_plan(macro).ports],
        "selector_devices": spec.blocks * spec.columns,
        "isolation_devices": spec.blocks * spec.columns,
        "periphery_devices": len(macro_periphery(macro).devices),
        "timing_qualified": False,
        "interface": (
            "CLK high precharges, low requests read; RESETN active low; internal replica feedback"
            if args.self_timed
            else "CLK high precharges, low requests read; RESETN active low; four external completion inputs"
            if args.clocked
            else "PRECHN active low; EVAL and SAMPLE active high; nonoverlapping read phases"
        ),
        "artifacts": {},
    }
    report_path = args.out / "manifest.json"
    # Invalidate a prior successful report before starting any physical checks.
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    failed = False
    for role, circuit in circuits.items():
        lib = require_gdspy().GdsLibrary(unit=1e-9, precision=1e-10)
        top = (
            build_macro(circuit, lib=lib)
            if role == "macro"
            else build_periphery(circuit, lib=lib)
        )
        gds = args.out / f"{role}.gds"
        lib.write_gds(str(gds))
        (args.out / f"{role}.cdl").write_text(circuit.netlist())
        reference = args.out / f"{role}.lvs.spice"
        reference.write_text(circuit.lvs_reference())
        bounds = top.get_bounding_box()
        record = {
            "cell": circuit.name,
            "transistors": len(circuit.devices),
            "gds": gds.name,
            "cdl": f"{role}.cdl",
            "width_nm": float(bounds[1, 0] - bounds[0, 0]),
            "height_nm": float(bounds[1, 1] - bounds[0, 1]),
            "periphery_route_connectivity": "passed",
            "drc": "not_run",
            "lvs": "not_run",
        }
        if role == "macro":
            record["array_route_connectivity"] = "passed"
            (args.out / "macro.lef").write_text(macro_lef(circuit, top))
            (args.out / "macro.v").write_text(macro_verilog(circuit, spec, words))
            record["lef"], record["verilog"] = "macro.lef", "macro.v"
        report["artifacts"][role] = record
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if args.drc:
            violations = run_gdscheck(gds, args.out / "drc" / role, cell_name=top.name)
            record["drc"] = dict(drc_counts(violations))
            failed |= bool(violations)
        if args.lvs:
            result = run_lvs(
                gds, reference, args.out / "lvs" / role, cell_name=top.name
            )
            summary = summarize_lvsdb(result.report)
            renamed = [p for c in summary["circuits"] for p in c["renamed_pins"]]
            pins = {
                line.removeprefix("* pin ").strip()
                for line in result.extracted_netlist.read_text().splitlines()
                if line.startswith("* pin ")
            }
            missing = sorted(set(circuit.pins) - pins)
            extra = sorted(pins - set(circuit.pins))
            record["missing_pins"] = missing
            record["extra_pins"] = extra
            matched = result.matched and not renamed and not missing and not extra
            record["lvs"] = "passed" if matched else "failed"
            record["renamed_pins"] = renamed
            failed |= not matched
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{macro.name}: {args.out / 'manifest.json'}")
    if failed:
        raise SystemExit("BROM physical checks failed; see manifest.json and reports")


if __name__ == "__main__":
    main()
