"""Command-line entry points for ASAP7 LVS, open PEX, and correlation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ..devices.spice import find_default_model_card
from .lvs import run_lvs
from .pex import result_as_json, run_open_pex
from .reference import (
    DEFAULT_REFERENCE_CELL,
    find_asap7_release,
    validate_released_inverter,
)


def _add_pex_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fastercap", type=Path)
    parser.add_argument(
        "--fastercap-lib",
        type=Path,
        help="directory containing FasterCap's private wx shared libraries",
    )
    parser.add_argument("--fastercap-threads", type=int)
    parser.add_argument("--klayout", type=Path)
    parser.add_argument("--model-card", type=Path)
    parser.add_argument(
        "--dialect", choices=("ngspice_osdi", "hspice"), default="ngspice_osdi"
    )
    parser.add_argument("--substrate-net")
    parser.add_argument("--cap-tolerance", type=float, default=0.08)
    parser.add_argument("--cap-mesh-area", type=float, default=0.1, metavar="UM2")
    parser.add_argument("--cap-mesh-quality", type=float, default=1.0)
    parser.add_argument("--cap-refinement", type=float, default=0.5)
    parser.add_argument("--cap-threshold", type=float, default=0.0, metavar="F")
    parser.add_argument("--res-mesh-area", type=float, default=0.0, metavar="UM2")
    parser.add_argument("--timeout", type=float, default=900.0)


def _run_pex_from_args(args: argparse.Namespace, *, standard_cell: bool) -> object:
    cap_tolerance = 0.15 if getattr(args, "smoke", False) else args.cap_tolerance
    cap_mesh_area = 4.0 if getattr(args, "smoke", False) else args.cap_mesh_area
    return run_open_pex(
        args.gds,
        args.schematic,
        args.out,
        cell_name=args.cell,
        klayout=args.klayout,
        fastercap=args.fastercap,
        fastercap_library_dir=args.fastercap_lib,
        fastercap_threads=args.fastercap_threads,
        asap7_standard_cell=standard_cell,
        substrate_net=args.substrate_net,
        model_card=args.model_card,
        dialect=args.dialect,
        capacitance_tolerance=cap_tolerance,
        capacitance_mesh_area_um2=cap_mesh_area,
        capacitance_mesh_quality=args.cap_mesh_quality,
        capacitance_refinement=args.cap_refinement,
        capacitance_threshold_farad=args.cap_threshold,
        resistance_mesh_area_um2=args.res_mesh_area,
        timeout=args.timeout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research-grade open-source LVS/PEX for public ASAP7 layouts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    lvs = subparsers.add_parser("lvs", help="run KLayout LVS")
    lvs.add_argument("--gds", type=Path, required=True)
    lvs.add_argument("--schematic", type=Path, required=True)
    lvs.add_argument("--cell", required=True)
    lvs.add_argument("--out", type=Path, required=True)
    lvs.add_argument("--klayout", type=Path)
    lvs.add_argument("--standard-cell", action="store_true")

    pex = subparsers.add_parser("pex", help="run LVS + KPEX R + FasterCap C")
    pex.add_argument("--gds", type=Path, required=True)
    pex.add_argument("--schematic", type=Path, required=True)
    pex.add_argument("--cell", required=True)
    pex.add_argument("--out", type=Path, required=True)
    pex.add_argument("--standard-cell", action="store_true")
    pex.add_argument("--smoke", action="store_true", help="use a coarse CI mesh")
    _add_pex_options(pex)

    validate = subparsers.add_parser(
        "validate", help="compare an extracted INVxp33 to released views"
    )
    validate.add_argument("--pex-report", type=Path, required=True)
    validate.add_argument("--netlist", type=Path, required=True)
    validate.add_argument("--out", type=Path, required=True)
    validate.add_argument("--asap7-root", type=Path)
    validate.add_argument("--cell", default=DEFAULT_REFERENCE_CELL)
    validate.add_argument("--osdi", type=Path)
    validate.add_argument("--bsimcmg-source", type=Path)
    validate.add_argument("--ngspice", default="ngspice")
    validate.add_argument("--no-bsim", action="store_true")

    reference = subparsers.add_parser(
        "reference-inverter",
        help="extract and correlate the released INVxp33 end to end",
    )
    reference.add_argument("--asap7-root", type=Path)
    reference.add_argument("--out", type=Path, required=True)
    reference.add_argument("--osdi", type=Path)
    reference.add_argument("--bsimcmg-source", type=Path)
    reference.add_argument("--ngspice", default="ngspice")
    reference.add_argument("--no-bsim", action="store_true")
    reference.add_argument("--smoke", action="store_true", help="use a coarse CI mesh")
    _add_pex_options(reference)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "lvs":
        result = run_lvs(
            args.gds,
            args.schematic,
            args.out,
            cell_name=args.cell,
            klayout=args.klayout,
            asap7_standard_cell=args.standard_cell,
        )
        print(
            json.dumps(
                {"matched": result.matched, "report": str(result.report)}, indent=2
            )
        )
        return 0 if result.matched else 1

    if args.command == "pex":
        result = _run_pex_from_args(args, standard_cell=args.standard_cell)
        print(json.dumps(result_as_json(result), indent=2))
        return 0

    if args.command == "validate":
        result = validate_released_inverter(
            args.pex_report,
            args.netlist,
            args.out,
            asap7_root=args.asap7_root,
            cell_name=args.cell,
            osdi=args.osdi,
            bsimcmg_source=args.bsimcmg_source,
            ngspice=args.ngspice,
            run_bsim=not args.no_bsim,
        )
        print(result.report.read_text(), end="")
        return 0 if result.passed_research_bounds else 1

    if args.command == "reference-inverter":
        release = find_asap7_release(args.asap7_root)
        library = release / "asap7sc7p5t_28"
        args.gds = library / "GDS/asap7sc7p5t_28_R_220121a.gds"
        args.schematic = library / "CDL/LVS/asap7sc7p5t_28_R.cdl"
        args.cell = DEFAULT_REFERENCE_CELL
        args.substrate_net = args.substrate_net or "VSS"
        args.model_card = args.model_card or find_default_model_card()
        if args.model_card is None and not args.no_bsim:
            parser.error("ASAP7 model card not found; pass --model-card")
        pex_result = _run_pex_from_args(args, standard_cell=True)
        validation = validate_released_inverter(
            pex_result.report,
            pex_result.post_layout_netlist,
            Path(args.out) / "validation",
            asap7_root=release,
            osdi=args.osdi,
            bsimcmg_source=args.bsimcmg_source,
            ngspice=args.ngspice,
            run_bsim=not args.no_bsim,
        )
        print(validation.report.read_text(), end="")
        return 0 if validation.passed_research_bounds else 1

    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
