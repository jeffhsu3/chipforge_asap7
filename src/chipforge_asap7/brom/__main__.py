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
    output_register_circuit,
    selector_map,
    self_timed_macro_circuit,
    size_control_distribution,
    size_wire_rc_distribution,
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
    parser.add_argument(
        "--keeper-fins",
        type=int,
        choices=(0, 1, 2, 3),
        default=0,
        help="Feedback keeper fins; 0 disables",
    )
    parser.add_argument(
        "--keeper-stack",
        type=int,
        choices=(1, 2, 3, 4),
        default=2,
        help="Series keeper PMOS count",
    )
    parser.add_argument(
        "--registered-output",
        action="store_true",
        help="Rising-edge Q register; requires --clocked or --self-timed",
    )
    sizing = parser.add_mutually_exclusive_group()
    sizing.add_argument(
        "--wire-rc-controls",
        action="store_true",
        help="Optimize control sizing/placement with routed-length RC estimates",
    )
    sizing.add_argument(
        "--sized-controls",
        action="store_true",
        help="Insert gate-load-sized tapered control buffers (experimental)",
    )
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
    parser.add_argument(
        "--timing-report",
        type=Path,
        help="Attach a matching completed timing-estimate JSON",
    )
    args = parser.parse_args(argv)
    if not args.self_timed and (
        args.replica_guard_pairs is not None or args.replica_extra_blocks is not None
    ):
        parser.error("replica options require --self-timed")
    if args.registered_output and not (args.clocked or args.self_timed):
        parser.error("--registered-output requires --clocked or --self-timed")
    words = json.loads(args.words.read_text())
    if not isinstance(words, list):
        parser.error("words must contain a JSON array")
    spec = BromSpec(
        len(words),
        args.bits,
        args.ways,
        args.fins,
        args.vt,
        args.keeper_fins,
        args.keeper_stack,
        args.registered_output,
    )
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
    if args.wire_rc_controls:
        size_wire_rc_distribution(macro)
    elif args.sized_controls:
        size_control_distribution(macro)
    estimate = None
    if args.timing_report:
        from .timing_estimate import attach_estimate

        # Check the report before writing anything, and fail as a usage error.
        try:
            estimate = attach_estimate(args.timing_report, macro, spec, words)
        except (OSError, ValueError) as exc:
            parser.error(f"--timing-report {args.timing_report}: {exc}")
    circuits = {
        "macro": macro,
        "cgen": cgen_circuit(spec.fins, spec.vt),
        "decoder": decoder_circuit(spec),
        "output": output_circuit(
            spec.fins,
            spec.vt,
            keeper_fins=spec.keeper_fins,
            keeper_stack=spec.keeper_stack,
        ),
    }
    if args.registered_output:
        circuits["output_register"] = output_register_circuit(
            spec.bits, spec.fins, spec.vt
        )
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
        "registered_output": bool(macro.output_register),
        "output_register": macro.output_register,
        "control_distribution": macro.distribution,
        "keeper": (
            {
                "fins": spec.keeper_fins,
                "series_devices": spec.keeper_stack,
                "replica_matched": bool(macro.replica),
                "timing_qualified": False,
            }
            if spec.keeper_fins
            else None
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
    if estimate is not None:
        report["timing_estimate"] = estimate
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
