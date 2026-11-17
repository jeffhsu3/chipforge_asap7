"""Emit a programmed BROM and its standalone peripheral cells."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from ..layout.layers import LAYERS, require_gdspy
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
    compact_periphery,
    controller_circuit,
    decoder_circuit,
    macro_circuit,
    output_circuit,
    output_register_circuit,
    pruned_macro_circuit,
    selector_map,
    self_timed_macro_circuit,
    size_control_distribution,
    size_wire_rc_distribution,
)
from .abstract import PROGRAMMING_LAYERS, abstract_name, frozen_geometry_sha256
from .dense_macro import dense_macro_layout
from .periphery import periphery_plan
from .views import macro_lef, macro_verilog


def _write_abstract(out: Path, circuit, top, lib) -> dict:
    """Write the stable-named LEF and GDS, and return the manifest's record."""
    name = abstract_name(circuit.name)
    lef = macro_lef(circuit, top, name=name)
    (out / "abstract.lef").write_text(lef)
    lib.rename_cell(top, name)
    lib.write_gds(str(out / "abstract.gds"))
    lib.rename_cell(top, circuit.name)
    return {
        "cell": name,
        "lef": "abstract.lef",
        "gds": "abstract.gds",
        "lef_sha256": sha256(lef.encode()).hexdigest(),
        "frozen_geometry_sha256": frozen_geometry_sha256(top),
        "programming_layers": {
            layer: [LAYERS[layer]["layer"], LAYERS[layer]["datatype"]]
            for layer in PROGRAMMING_LAYERS
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Flag names follow chipforge_gt2n's `gt2n rom brom`; the old ASAP7 names
    # (--bits, --ways, --out and the positional words file) still work.
    parser.add_argument(
        "words",
        type=Path,
        nargs="?",
        help="JSON array of unsigned integer words (or --words-json)",
    )
    parser.add_argument(
        "--words-json", type=Path, help="JSON array of unsigned integer words"
    )
    parser.add_argument(
        "--word-bits",
        "--bits",
        dest="bits",
        type=int,
        required=True,
        help="logical word width (--bits is the old name)",
    )
    parser.add_argument(
        "--column-mux",
        "--ways",
        dest="ways",
        type=int,
        choices=(1, 4),
        default=1,
        help="logical words per physical row; the macro selects the word "
        "(--ways is the old name)",
    )
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
    parser.add_argument(
        "--compact-periphery",
        action="store_true",
        help="Deprecated no-op: dense CMOS is always enabled (requires optimize extra)",
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
    parser.add_argument(
        "--replica-share",
        type=int,
        help="Output latches per local SAMPLE receiver (default 1)",
    )
    mode.add_argument(
        "--clocked",
        action="store_true",
        help="Integrate clock controller; completion feedback remains external",
    )
    mode.add_argument(
        "--pruned",
        action="store_true",
        help="Data-pruned single-sided CGEN and decoupled-BL custom-mask macro",
    )
    parser.add_argument(
        "--output-dir",
        "--out",
        dest="out",
        type=Path,
        required=True,
        help="directory for the GDS, views and manifest (--out is the old name)",
    )
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
        args.replica_guard_pairs is not None
        or args.replica_extra_blocks is not None
        or args.replica_share is not None
    ):
        parser.error("replica options require --self-timed")
    if args.registered_output and not (args.clocked or args.self_timed):
        parser.error("--registered-output requires --clocked or --self-timed")
    if args.pruned and (args.wire_rc_controls or args.sized_controls):
        parser.error("--pruned sizes its own controls and cannot combine with --sized-controls/--wire-rc-controls")
    if args.pruned and args.keeper_fins and args.ways != 1:
        parser.error("--pruned ways=4 uses gated tristate sense without bitline keeper")
    if (args.words is None) == (args.words_json is None):
        parser.error("give the words file once: positionally or as --words-json")
    words = json.loads((args.words or args.words_json).read_text())
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
                1 if args.replica_share is None else args.replica_share,
            ),
        )
        if args.self_timed
        else clocked_macro_circuit(spec, words)
        if args.clocked
        else pruned_macro_circuit(spec, words)
        if args.pruned
        else macro_circuit(spec, words)
    )
    if args.compact_periphery:
        compact_periphery(macro)
    if args.wire_rc_controls:
        size_wire_rc_distribution(macro)
    elif args.sized_controls:
        size_control_distribution(macro)
    circuits = (
        {"macro": macro}
        if args.pruned
        else {
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
    )
    if args.registered_output:
        circuits["output_register"] = output_register_circuit(
            spec.bits, spec.fins, spec.vt
        )
    if args.clocked or args.self_timed:
        circuits["controller"] = controller_circuit(spec.fins, spec.vt)
    if args.compact_periphery:
        for circuit in circuits.values():
            compact_periphery(circuit)
    args.out.mkdir(parents=True, exist_ok=True)
    _, physical = dense_macro_layout(macro)
    report = {
        # The shape keys chipforge_gt2n's ROM manifests carry, by the same names.
        "architecture": "brom",
        "platform": "asap7",
        "depth": spec.depth,
        "num_words": spec.depth,
        "word_bits": spec.bits,
        "physical_word_bits": spec.columns,
        "column_mux": spec.ways,
        "column_mux_in_macro": spec.ways > 1,
        "address_bits": spec.address_bits,
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
        "sample_distribution": macro.sample_distribution,
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
                "columns": len(macro.replica.selector.mapping[0]),
                "candidate_gates": macro.replica.dummy_loads + 1,
                "dummy_loads": macro.replica.dummy_loads,
                "guard_pairs": macro.replica.guard_pairs,
                "extra_blocks": macro.replica.extra_blocks,
                "latches_per_sample_receiver": max(
                    len(bank["latch_groups"])
                    for bank in macro.sample_distribution["banks"]
                ),
                "candidate": 15,
                "selector_devices": len(macro.replica.selector.device_names) // 2,
                "isolation_devices": len(macro.replica.selector.device_names) // 2,
                "feedback_timing_qualified": False,
            }
            if macro.replica
            else None
        ),
        "layout": "pruned_dense_cmos" if args.pruned else "dense_cmos_split_selector",
        "selector_plan": asdict(macro.array.plan),
        "array_core_nm": [spec.columns * 108, spec.blocks * 432],
        "periphery_layout": "shared_diffusion_cmos",
        "physical_plan": physical,
        "physical_sha256": physical["physical_sha256"],
        "selector_devices": spec.blocks * spec.columns,
        "isolation_devices": spec.blocks * spec.columns,
        "periphery_devices": len(macro.devices)
        - len(macro.array.device_names)
        - (len(macro.replica.selector.device_names) if macro.replica else 0),
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
    if args.timing_report:
        from .timing_estimate import attach_estimate

        report["timing_estimate"] = attach_estimate(
            args.timing_report, macro, spec, words
        )
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
        plan = periphery_plan(circuit) if role != "macro" else None
        record = {
            "placement_certificates": physical["placement_certificates"]
            if role == "macro"
            else plan.certificates,
            "routing_certificates": physical["routing_certificates"]
            if role == "macro"
            else plan.routing_certificates,
            "cell": circuit.name,
            "transistors": len(circuit.devices),
            "gds": gds.name,
            "cdl": f"{role}.cdl",
            "width_nm": float(bounds[1, 0] - bounds[0, 0]),
            "height_nm": float(bounds[1, 1] - bounds[0, 1]),
            "periphery_route_connectivity": "pending_lvs",
            "drc": "not_run",
            "lvs": "not_run",
        }
        if role == "macro":
            record["array_route_connectivity"] = "pending_lvs"
            (args.out / "macro.lef").write_text(macro_lef(circuit, top))
            (args.out / "macro.v").write_text(macro_verilog(circuit, spec, words))
            record["lef"], record["verilog"] = "macro.lef", "macro.v"
            report["abstract"] = (
                None if args.pruned else _write_abstract(args.out, circuit, top, lib)
            )
        report["artifacts"][role] = record
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if args.drc:
            violations = run_gdscheck(gds, args.out / "drc" / role, cell_name=top.name)
            record["drc"] = dict(drc_counts(violations))
            failed |= bool(violations)
        if args.lvs:
            result = run_lvs(
                gds,
                reference,
                args.out / "lvs" / role,
                cell_name=top.name,
                merge_fin_diffusion=True,
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
            record["periphery_route_connectivity"] = record["lvs"]
            if role == "macro":
                record["array_route_connectivity"] = record["lvs"]
            record["renamed_pins"] = renamed
            failed |= not matched
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{macro.name}: {args.out / 'manifest.json'}")
    if failed:
        raise SystemExit("BROM physical checks failed; see manifest.json and reports")


if __name__ == "__main__":
    main()
