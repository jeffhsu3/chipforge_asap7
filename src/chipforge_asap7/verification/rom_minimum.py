"""A scoped minimum-area proof for the existing periodic NOR-ROM topology.

The lower-bound model uses only necessary FEOL inequalities, with continuous
coordinates and no routing constraints. UNSAT in this relaxed model excludes
smaller tiles in the stated topology. A separate drawn witness must attain
the bound, pass the pinned gdscheck deck, and have the expected extracted
connectivity. This is not a search over other NOR-ROM architectures.

Run with ``python -m chipforge_asap7.verification.rom_minimum --help``.
Z3 is invoked through SMT-LIB; no Python solver bindings are required.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..devices.rom_bitcell import (
    RomArraySpec,
    RomBitcellSpec,
    build_rom_array,
    build_rom_bitcell,
)
from ..layout.layers import LAYERS, require_gdspy
from .drc import drc_counts, find_gdscheck, run_gdscheck
from .minimum_proof import dimension_queries, run_query
from .minimum_proof import smt_assert as _assert
from .minimum_proof import tool_info as _tool_info
from .minimum_proof import validate_rule_source as validate_pinned_source

ASSUMPTIONS = (
    "Periodic rectangular two-word, one-bit tile with continuous diffusion along X.",
    "Gate order is WLa, WLb, ISO, then the next tile's WLa; ISO is held at VSS.",
    "Two NMOS words share a grounded source and have independently V1-programmed drains.",
    "Each transistor has the requested number of distinct parallel fins.",
    "Different bit rows have separate ACTIVE strips and bitlines.",
    "Ordinary ASAP7 rules apply: no SRAMDRC marker or SRAM rule exceptions.",
)
LIMITS = (
    "Area is the periodic core area per bit; straps, taps, edges and periphery are excluded.",
    "Alternative topologies, programming layers and diffusion-sharing schemes are outside the proof.",
    "Switches are ideal; leakage, read current, timing, noise margins and PVT are not proved.",
    "The SMT model is a necessary-rule relaxation, not an encoding of the full DRC deck.",
    "Rule transcription and the topology-to-constraint mapping are explicit trusted assumptions.",
    "Z3 proof terms are saved; they are not independently checked by a second proof checker.",
    "DRC checks the finite witness; it does not certify every array size or programmed image.",
)


def rule_manifest() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("rom_minimum_rules.json").read_text())


def validate_rule_source(pdk: Path) -> dict[str, Any]:
    return validate_pinned_source(pdk, rule_manifest())


def _facts() -> dict[str, int]:
    return {name: fact["value"] for name, fact in rule_manifest()["facts"].items()}


def minimum_dimensions(fins: int) -> tuple[int, int]:
    """Lower bounds on tile width and row pitch, independently of the generator."""
    if isinstance(fins, bool) or not isinstance(fins, int) or fins < 1:
        raise ValueError("fins must be a positive integer")
    r = _facts()
    width = 3 * (r["gate_width_nm"] + r["gate_space_nm"])
    height = (
        (fins - 1) * r["fin_pitch_nm"]
        + r["fin_width_nm"]
        + 2 * r["active_fin_enclosure_nm"]
        + r["active_space_nm"]
    )
    return width, height


def layout_constraints(fins: int) -> str:
    """Necessary linear inequalities; neither minimum dimension is asserted.

    The wraparound gate spacing and inter-row ACTIVE spacing encode the two
    periodic boundaries. Fin grid membership is relaxed to minimum spacing
    between distinct fins, making the modeled space larger, not smaller.
    """
    minimum_dimensions(fins)  # validate the count
    r = _facts()
    pitch = r["gate_width_nm"] + r["gate_space_nm"]
    names = ["width", "height", "ga", "gb", "gi", "alo", "ahi"]
    names += [f"fin{i}" for i in range(fins)]
    lines = [
        "; Units: nm. See rom_minimum_rules.json for the audited rule facts.",
        "; Three gates per period is a topology assumption, not a DRC rule.",
        "(set-option :produce-proofs true)",
        "(set-logic QF_LRA)",
        *(f"(declare-const {name} Real)" for name in names),
        _assert("positive_width", "(> width 0)"),
        _assert("positive_height", "(> height 0)"),
        _assert("gate_ab", f"(>= (- gb ga) {pitch})"),
        _assert("gate_bi", f"(>= (- gi gb) {pitch})"),
        _assert("gate_wrap", f"(>= (- (+ ga width) gi) {pitch})"),
        _assert("active_lower", f"(>= (- fin0 alo) {r['active_fin_enclosure_nm']})"),
        _assert(
            "active_upper",
            f"(>= (- ahi fin{fins - 1}) {r['fin_width_nm'] + r['active_fin_enclosure_nm']})",
        ),
        _assert("active_wrap", f"(>= (- (+ alo height) ahi) {r['active_space_nm']})"),
    ]
    for i in range(1, fins):
        lines.append(
            _assert(
                f"fin_spacing_{i}", f"(>= (- fin{i} fin{i - 1}) {r['fin_pitch_nm']})"
            )
        )
    return "\n".join(lines) + "\n"


def measure_tile(cell: Any, fins: int) -> dict[str, float]:
    """Read the actual polygons, not the spec's width/height properties."""
    polygons = cell.get_polygons(by_spec=True)

    def rectangles(layer: str) -> list[tuple[float, float, float, float]]:
        key = (LAYERS[layer]["layer"], LAYERS[layer]["datatype"])
        boxes = []
        for points in polygons.get(key, []):
            x0, y0 = map(float, points.min(axis=0))
            x1, y1 = map(float, points.max(axis=0))
            if len(points) != 4 or set(map(tuple, points)) != {
                (x0, y0),
                (x0, y1),
                (x1, y0),
                (x1, y1),
            }:
                raise ValueError(f"proof witness requires rectangular {layer}")
            boxes.append((x0, y0, x1, y1))
        return sorted(boxes)

    if rectangles("SRAMDRC") or rectangles("GATE_CUT"):
        raise ValueError("proof witness excludes SRAMDRC and gate cuts in the tile")
    boundaries, active = rectangles("BOUNDARY"), rectangles("ACTIVE")
    if len(boundaries) != 1 or len(active) != 1:
        raise ValueError(
            "proof witness needs one boundary and one continuous ACTIVE strip"
        )
    x0, y0, x1, y1 = boundaries[0]
    ax0, alo, ax1, ahi = active[0]
    if (x0, y0) != (0, 0) or (ax0, ax1) != (x0, x1):
        raise ValueError(
            "proof witness must start at zero and span the tile with ACTIVE"
        )
    r = _facts()
    gates = rectangles("GATE")
    if len(gates) != 3 or any(
        b - a != r["gate_width_nm"] or (lo, hi) != (y0, y1) for a, lo, b, hi in gates
    ):
        raise ValueError("proof witness requires three full-height, rule-width gates")
    active_fins = [box for box in rectangles("FIN") if box[1] < ahi and box[3] > alo]
    active_fins.sort(key=lambda box: box[1])
    if len(active_fins) != fins or any(
        (a, b) != (x0, x1) or hi - lo != r["fin_width_nm"]
        for a, lo, b, hi in active_fins
    ):
        raise ValueError("proof witness has the wrong fin count, width or span")
    return {
        "width": x1 - x0,
        "height": y1 - y0,
        **{
            name: (gate[0] + gate[2]) / 2
            for name, gate in zip(("ga", "gb", "gi"), gates)
        },
        "alo": alo,
        "ahi": ahi,
        **{f"fin{i}": fin[1] for i, fin in enumerate(active_fins)},
    }


def proof_queries(fins: int, witness: dict[str, float]) -> dict[str, tuple[str, str]]:
    width, height = minimum_dimensions(fins)
    return dimension_queries(layout_constraints(fins), width, height, witness)


def topology_check() -> dict[str, Any]:
    """Exhaust all programming and wordline states of two periodic tiles.

    BL and VSS are nodes 0 and 1; the four distinct drains are nodes 2..5.
    Omitting ISO merges drains across each tile boundary and must produce a
    counterexample. This is an ideal-switch model, not extracted analog LVS.
    """

    def conducts(program: int, selected: int, isolated: bool) -> bool:
        parent = list(range(6))

        def find(node: int) -> int:
            while parent[node] != node:
                node = parent[node]
            return node

        edges = []
        for word in range(4):
            if program & (1 << word):
                edges.append((0, word + 2))
            if selected & (1 << word):
                edges.append((1, word + 2))
        if not isolated:
            edges.extend(((3, 4), (5, 2)))
        for a, b in edges:
            parent[find(a)] = find(b)
        return find(0) == find(1)

    counterexample = None
    for program in range(16):
        for selected in range(16):
            expected = bool(program & selected)
            if conducts(program, selected, True) != expected:
                raise ValueError(
                    "the stated topology does not implement independent NOR bits"
                )
            if (
                counterexample is None
                and conducts(program, selected, False) != expected
            ):
                counterexample = {
                    "program_mask": program,
                    "selected_wordlines": selected,
                    "expected_discharge": expected,
                    "actual_discharge": not expected,
                }
    if counterexample is None:
        raise ValueError(
            "removing isolation did not produce the required counterexample"
        )
    return {
        "words": 4,
        "programming_patterns": 16,
        "wordline_states": 16,
        "cases_passed": 256,
        "without_isolation_counterexample": counterexample,
    }


def extract_witness(gds: Path, cell_name: str, spec: RomArraySpec) -> dict[str, int]:
    """Check the drawn four-word array containing all 16 programming patterns."""
    import gdstk

    from . import connectivity as cv

    top = next(
        c for c in gdstk.read_gds(str(gds), unit=1e-6).cells if c.name == cell_name
    )
    flat = top.copy("flat").flatten()
    merged = gdstk.Cell("merged")
    for layer, datatype in {(p.layer, p.datatype) for p in flat.polygons}:
        same = [p for p in flat.polygons if (p.layer, p.datatype) == (layer, datatype)]
        merged.add(*gdstk.boolean(same, [], "or", layer=layer, datatype=datatype))
    merged.add(*flat.labels)
    devices, labels = cv.extract_connectivity(merged, expected_channels=96)
    if any(len(ids) != 1 for ids in labels.values()):
        raise ValueError("a witness pin is split across nets")
    nets = {name: next(iter(ids)) for name, ids in labels.items()}
    if len(set(nets.values())) != len(nets):
        raise ValueError("witness pins are shorted")
    word_gates = {nets[f"WL<{word}>"]: word for word in range(4)}
    seen: Counter[tuple[int, int]] = Counter()
    floating = set()
    isolation = 0
    for device in devices:
        if device["pmos"] or device["nfin"] != spec.bitcell.fins:
            raise ValueError("witness has an unexpected device type or fin count")
        if device["gate"] not in word_gates:
            if device["gate"] != nets["VSS"]:
                raise ValueError("isolation gate is not grounded")
            isolation += 1
            continue
        word = word_gates[device["gate"]]
        y = device["center"][1] * 1000
        rows = [
            bit
            for bit in range(16)
            if spec.row_y(bit) <= y < spec.row_y(bit) + spec.bitcell.height
        ]
        if len(rows) != 1:
            raise ValueError("word device is outside its bit row")
        bit = rows[0]
        seen[bit, word] += 1
        terminals = set(device["terminals"])
        if nets["VSS"] not in terminals or len(terminals) != 2:
            raise ValueError(
                "word device lacks its grounded source or has a shorted drain"
            )
        (drain,) = terminals - {nets["VSS"]}
        if bit & (1 << word):
            if drain != nets[f"BL<{bit}>"]:
                raise ValueError("programmed drain does not reach its own bitline")
        else:
            if drain in nets.values() or drain in floating:
                raise ValueError("unprogrammed drain is not independently floating")
            floating.add(drain)
    if (
        seen != Counter({(bit, word): 1 for bit in range(16) for word in range(4)})
        or isolation != 32
    ):
        raise ValueError(
            "witness transistor inventory differs from the stated topology"
        )
    return {
        "programming_patterns": 16,
        "word_devices": sum(seen.values()),
        "grounded_isolation_devices": isolation,
    }


def _draw_witness(fins: int, gds: Path) -> tuple[dict[str, float], RomArraySpec, str]:
    gdspy = require_gdspy()
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    previous = gdspy.current_library
    gdspy.current_library = library
    try:
        bitcell = RomBitcellSpec(fins=fins)
        tile = build_rom_bitcell(bitcell, lib=library)
        witness = measure_tile(tile, fins)
        # Distinct bitlines carry all four-word programming patterns. Both
        # within-pair and between-pair neighbours see every bit combination.
        array = RomArraySpec(words=4, bits=16, bitcell=bitcell, strap_every=16)
        sites = [
            (word, bit) for bit in range(16) for word in range(4) if bit & (1 << word)
        ]
        top = build_rom_array(array, sites, lib=library)
        library.write_gds(str(gds))
        return witness, array, top.name
    finally:
        gdspy.current_library = previous


def prove_minimum(
    fins: int, out: Path, *, pdk: Path, z3: Path, gdscheck: Path
) -> dict[str, Any]:
    """Produce the bound, solver logs and fully checked constructive witness.

    Failure leaves a failed report; a previous successful report cannot survive
    a failed rerun. Every success requires source hashes, SAT/UNSAT, extraction
    and DRC. No missing verification tool is treated as a pass.
    """
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "format": "chipforge_asap7.rom_minimum.v1",
        "status": "incomplete",
        "fins": fins,
        "assumptions": ASSUMPTIONS,
        "limits": LIMITS,
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        width, height = minimum_dimensions(fins)
        report["rule_source"] = validate_rule_source(pdk)
        package = Path(__file__).resolve().parents[1]
        report["implementation_sha256"] = {
            str(path.relative_to(package)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(package.rglob("*.py"))
        }
        report["tools"] = {
            "z3": _tool_info(z3, "-version"),
            "gdscheck": _tool_info(gdscheck, "--version"),
        }
        report["topology"] = topology_check()
        gds = out / "witness.gds"
        witness, array, cell_name = _draw_witness(fins, gds)
        if (witness["width"], witness["height"]) != (width, height):
            raise ValueError("drawn tile does not attain the independent lower bound")
        report["drawn_tile_nm"] = witness
        report["queries"] = {
            name: run_query(z3, source, out / f"{name}.smt2", expected)
            for name, (source, expected) in proof_queries(fins, witness).items()
        }
        report["gds"] = {
            "path": gds.name,
            "cell": cell_name,
            "sha256": sha256(gds.read_bytes()).hexdigest(),
        }
        violations = run_gdscheck(
            gds,
            out / "drc",
            cell_name=cell_name,
            binary=gdscheck,
            pdk_file=pdk / "pdk.yml",
        )
        report["drc"] = {"suite": "main", "violations": dict(drc_counts(violations))}
        if violations:
            raise ValueError("the constructive witness fails gdscheck")
        report["extracted_connectivity"] = extract_witness(gds, cell_name, array)
        report["minimum"] = {
            "tile_width_nm": width,
            "row_pitch_nm": height,
            "bits_per_tile": 2,
            "core_area_per_bit_nm2": width * height // 2,
        }
        report["area_argument"] = (
            "Positive width >= W and height >= H imply width*height/2 >= W*H/2. The drawn witness attains both bounds."
        )
        report["status"] = "proved_for_stated_topology"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fins", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--gdscheck-root",
        type=Path,
        required=True,
        help="checkout containing the pinned pdks/asap7 sources",
    )
    parser.add_argument("--gdscheck", type=Path)
    parser.add_argument("--z3", type=Path, default=shutil.which("z3"))
    parser.add_argument("--out", type=Path, default=Path("build/rom_minimum"))
    args = parser.parse_args(argv)
    if args.z3 is None:
        parser.error("Z3 is required; install its executable or supply --z3")
    try:
        binary = find_gdscheck(args.gdscheck)
        for fins in dict.fromkeys(args.fins):
            destination = args.out / f"f{fins}"
            print(f"Checking {fins}-fin ROM minimum...", flush=True)
            result = prove_minimum(
                fins,
                destination,
                pdk=args.gdscheck_root / "pdks/asap7",
                z3=Path(args.z3),
                gdscheck=binary,
            )
            print(
                f"  {result['status']}: {result['minimum']['core_area_per_bit_nm2']} nm²/bit -> {destination / 'report.json'}",
                flush=True,
            )
    except (
        ValueError,
        RuntimeError,
        OSError,
        subprocess.SubprocessError,
        ImportError,
    ) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    main()
