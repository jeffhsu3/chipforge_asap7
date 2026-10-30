"""Conditional minimum-area proof for the ASAP7 6T_122 bitcell template.

SRAMDRC exempts the ordinary gate/fin/ACTIVE rules. The retained 54/27 nm
grids, 7 nm fins and 10 nm ACTIVE enclosure are therefore explicit template
assumptions, not a claim about the minimum over every SRAM-rule layout.
The proof checks measured GDS, exact 6T connectivity, ideal hold/read paths,
and the pinned gdscheck suite on the leaf and a supplied array witness.
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

from ..layout.layers import LAYERS
from . import connectivity as cv
from .drc import drc_counts, find_gdscheck, run_gdscheck
from .minimum_proof import (
    dimension_queries,
    run_query,
    smt_assert,
    smt_number,
    tool_info,
    validate_rule_source,
)

CELL_NAME = "sram_cell_6t_122"
ARRAY_NAME = "array_x2x4"
ASSUMPTIONS = (
    "One 6T bit per rectangular periodic tile, with two distinct gate columns on a retained 54 nm lattice.",
    "Four separate horizontal ACTIVE bands ordered N/P/P/N; each N band shares diffusion between PD and PG.",
    "The band fin counts are 2/1/1/2: PU=1, PD=2, PG=2, on a retained 27 nm fin lattice.",
    "FIN width is 7 nm and ACTIVE retains 10 nm enclosure of the outer fins of each band.",
    "ACTIVE bands remain strictly separate, including the vertical periodic boundary; they cannot be merged or staggered into fewer bands.",
    "The middle P bands share one N-well; the outer N bands are outside it.",
    "Repeated supply pin labels are tied externally; the source leaf has two separate VSS landing roots.",
)
LIMITS = (
    "This is a conditional template minimum, NOT a minimum implied by the SRAM DRC rules alone.",
    "SRAMDRC exempts ordinary gate/fin grid and ACTIVE enclosure/spacing checks; these template policies are not mislabeled as applicable rules.",
    "Alternative folding, diffusion sharing, band order, fin counts, enclosure policies and transistor topologies are excluded.",
    "The lower-bound model omits routing; a smaller model after relaxing an assumption is not a validated manufacturable layout.",
    "Core area excludes taps, dummies, array edges and all peripheral circuitry.",
    "Ideal-switch checks establish hold/read paths, not read stability, write margin, delay, leakage, PVT or analog bistability.",
    "Z3 proof terms are saved but not independently checked. Rule transcription, template mapping and verification tools are trusted.",
    "Full gdscheck runs validate the supplied finite leaf and array witnesses, not every possible array assembly.",
)


def rule_manifest() -> dict[str, Any]:
    return json.loads(
        Path(__file__).with_name("sram_6t_minimum_rules.json").read_text()
    )


def _template() -> dict[str, Any]:
    return {name: value["value"] for name, value in rule_manifest()["template"].items()}


def minimum_dimensions() -> tuple[int, int]:
    t = _template()
    # With fin width + two enclosures equal to a fin pitch, strict band
    # separation on the retained lattice requires one unused fin track per band.
    if t["fin_width_nm"] + 2 * t["active_fin_enclosure_nm"] != t["fin_pitch_nm"]:
        raise ValueError("the template arithmetic needs reauditing")
    return 2 * t["gate_pitch_nm"], (sum(t["band_fins"]) + len(t["band_fins"])) * t[
        "fin_pitch_nm"
    ]


def layout_constraints(*, retain_enclosure: bool = True) -> str:
    """Mixed integer/real model of the retained template and applicable rules.

    No width or height lower bound is assumed. Integer lattice indices and
    real ACTIVE/well edges permit coordinate movement and asymmetric gaps.
    The enclosure-relaxed model is an explicit scope counterexample.
    """
    t = _template()
    r = rule_manifest()["applicable_rules"]
    cpp, fp = t["gate_pitch_nm"], t["fin_pitch_nm"]
    enclosure = t["active_fin_enclosure_nm"] if retain_enclosure else 0
    edge = smt_number(t["fin_width_nm"] / 2 + enclosure)
    clearance = smt_number(r["well_clearance_nm"]["value"])
    fin_names = [
        f"f{band}_{i}"
        for band, count in enumerate(t["band_fins"])
        for i in range(count)
    ]
    reals = ["width", "height", "well_lo", "well_hi"] + [
        f"{edge}{band}" for band in range(4) for edge in ("lo", "hi")
    ]
    lines = [
        "; nm; lattice indices are integers. See the manifest for policy vs DRC scope.",
        "(set-option :produce-proofs true)",
        "(set-logic QF_LIRA)",
        *(f"(declare-const {name} Real)" for name in reals),
        *(
            f"(declare-const {name} Int)"
            for name in ["period_x", "period_y", "ga", "gb", *fin_names]
        ),
        smt_assert("positive_width", "(> width 0)"),
        smt_assert("positive_height", "(> height 0)"),
        # Remove translation symmetry. Every periodic solution can be shifted
        # by whole lattice tracks to put its first gate/fin at these indices.
        smt_assert("gate_frame", "(= ga 0)"),
        smt_assert("fin_frame", "(= f0_0 1)"),
        smt_assert("gate_period", f"(= width (* {cpp} period_x))"),
        smt_assert("fin_period", f"(= height (* {fp} period_y))"),
        smt_assert("gate_order", "(> gb ga)"),
        smt_assert("gate_wrap", "(> (+ ga period_x) gb)"),
    ]
    for band, count in enumerate(t["band_fins"]):
        lines.extend(
            [
                smt_assert(
                    f"enclose_low_{band}",
                    f"(>= (- (* {fp} f{band}_0) lo{band}) {edge})",
                ),
                smt_assert(
                    f"enclose_high_{band}",
                    f"(>= (- hi{band} (* {fp} f{band}_{count - 1})) {edge})",
                ),
                smt_assert(
                    f"contact_overlap_{band}",
                    f"(>= (- hi{band} lo{band}) {r['contact_active_overlap_nm']['value']})",
                ),
            ]
        )
        for i in range(1, count):
            lines.append(
                smt_assert(
                    f"distinct_fins_{band}_{i}", f"(> f{band}_{i} f{band}_{i - 1})"
                )
            )
        if band:
            lines.append(
                smt_assert(f"separate_bands_{band}", f"(> lo{band} hi{band - 1})")
            )
    lines.extend(
        [
            smt_assert("active_wrap", "(> (+ lo0 height) hi3)"),
            smt_assert("well_bottom_outside", f"(>= (- well_lo hi0) {clearance})"),
            smt_assert("well_bottom_inside", f"(>= (- lo1 well_lo) {clearance})"),
            smt_assert("well_top_inside", f"(>= (- well_hi hi2) {clearance})"),
            smt_assert("well_top_outside", f"(>= (- lo3 well_hi) {clearance})"),
        ]
    )
    return "\n".join(lines) + "\n"


def _boxes(cell: Any, layer: str) -> list[tuple[float, float, float, float]]:
    """Rectangles in nm; exact 0.1 nm witness grid, separate from model bounds."""
    boxes = []
    key = LAYERS[layer]
    for polygon in cell.get_polygons(layer=key["layer"], datatype=key["datatype"]):
        points = {
            (round(float(x) * 1000, 7), round(float(y) * 1000, 7))
            for x, y in polygon.points
        }
        (x0, y0), (x1, y1) = polygon.bounding_box()
        box = tuple(round(float(v) * 1000, 7) for v in (x0, y0, x1, y1))
        a, b, c, d = box
        if len(polygon.points) != 4 or points != {(a, b), (a, d), (c, b), (c, d)}:
            raise ValueError(f"the retained template requires rectangular {layer}")
        boxes.append(box)
    return sorted(boxes)


def _track(value: float, pitch: float, offset: float = 0) -> int:
    index = (value - offset) / pitch
    if abs(index - round(index)) > 1e-7:
        raise ValueError("drawn geometry is off the retained template lattice")
    return round(index)


def measure_tile(cell: Any) -> dict[str, float]:
    """Bind the proof to actual ACTIVE, FIN, GATE, NWELL and boundary polygons."""
    t = _template()
    boundaries = _boxes(cell, "BOUNDARY")
    if len(boundaries) != 1 or boundaries[0][:2] != (0, 0):
        raise ValueError("one zero-origin 6T placement boundary is required")
    if _boxes(cell, "SRAMDRC") != boundaries:
        raise ValueError("the witness must use SRAMDRC on its placement boundary")
    _, _, width, height = boundaries[0]
    gates, fins = _boxes(cell, "GATE"), _boxes(cell, "FIN")
    active = sorted(_boxes(cell, "ACTIVE"), key=lambda b: b[1])
    wells = _boxes(cell, "NWELL")
    if len(gates) != 2 or len(active) != 4 or len(wells) != 1:
        raise ValueError(
            "expected two gate columns and four N/P/P/N ACTIVE bands in one well"
        )
    if any(b[2] - b[0] != t["gate_width_nm"] for b in gates):
        raise ValueError("drawn gate length differs from the retained 20 nm device")
    if any(abs(b[3] - b[1] - t["fin_width_nm"]) > 1e-6 for b in fins):
        raise ValueError("drawn FIN width differs from the retained template")
    fin_tracks = [(_track((b[1] + b[3]) / 2, t["fin_pitch_nm"]), b) for b in fins]
    result = {
        "width": width,
        "height": height,
        "period_x": _track(width, t["gate_pitch_nm"]),
        "period_y": _track(height, t["fin_pitch_nm"]),
        "ga": _track(
            (gates[0][0] + gates[0][2]) / 2, t["gate_pitch_nm"], t["gate_offset_nm"]
        ),
        "gb": _track(
            (gates[1][0] + gates[1][2]) / 2, t["gate_pitch_nm"], t["gate_offset_nm"]
        ),
        "well_lo": wells[0][1],
        "well_hi": wells[0][3],
    }
    for band, ((x0, lo, x1, hi), count) in enumerate(zip(active, t["band_fins"])):
        used = sorted(
            index
            for index, b in fin_tracks
            if b[1] < hi and b[3] > lo and b[0] <= x0 and b[2] >= x1
        )
        if len(used) != count or len(set(used)) != count:
            raise ValueError("ACTIVE band fin counts differ from the 2/1/1/2 template")
        inside = (
            wells[0][0] <= x0
            and wells[0][2] >= x1
            and wells[0][1] <= lo
            and wells[0][3] >= hi
        )
        outside = hi <= wells[0][1] or lo >= wells[0][3]
        if (band in (1, 2) and not inside) or (band in (0, 3) and not outside):
            raise ValueError(
                "ACTIVE bands do not have the retained N/P/P/N well arrangement"
            )
        result.update({f"lo{band}": lo, f"hi{band}": hi})
        result.update({f"f{band}_{i}": index for i, index in enumerate(used)})
    return result


def proof_queries(witness: dict[str, float]) -> dict[str, tuple[str, str]]:
    width, height = minimum_dimensions()
    queries = dimension_queries(layout_constraints(), width, height, witness)
    # A smaller fixed-width model when the enclosure policy is removed
    # establishes why the template qualifier is necessary. It is not a GDS.
    relaxed = layout_constraints(retain_enclosure=False)
    relaxed += f"(assert (= width {width}))\n(assert (< height {height}))\n(check-sat)\n(get-model)\n"
    queries["relaxed_enclosure_smaller_height"] = relaxed, "sat"
    return queries


def verify_topology(cell: Any) -> dict[str, Any]:
    """Extract exactly the 6T_122 schematic, allowing external supply aliases."""
    devices, labels = cv.extract_connectivity(cell, expected_channels=6)
    pin_names = {"WL": "WL", "BL": "BL", "BLN": "BLN", "vdd!": "VDD", "vss!": "VSS"}
    if set(labels) != set(pin_names) or any(not ids for ids in labels.values()):
        raise ValueError("6T pin interface differs from WL/BL/BLN/VDD/VSS")
    names = {}
    for label, canonical in pin_names.items():
        if canonical not in ("VDD", "VSS") and len(labels[label]) != 1:
            raise ValueError("a 6T signal pin is split across disconnected nets")
        for net in labels[label]:
            if net in names:
                raise ValueError("different 6T pins are shorted")
            names[net] = canonical
    access = [d for d in devices if names.get(d["gate"]) == "WL" and not d["pmos"]]
    if len(access) != 2:
        raise ValueError("6T topology requires two wordline-controlled access devices")
    for bitline, storage in (("BL", "Q"), ("BLN", "QB")):
        matches = [
            d for d in access if any(names.get(n) == bitline for n in d["terminals"])
        ]
        if len(matches) != 1:
            raise ValueError("6T access devices do not connect separate bitlines")
        remaining = [n for n in matches[0]["terminals"] if names.get(n) != bitline]
        if len(remaining) != 1 or remaining[0] in names:
            raise ValueError(
                "6T storage nodes are shorted or exposed on a signal/supply pin"
            )
        names[remaining[0]] = storage
    try:
        actual = Counter(
            (
                d["pmos"],
                d["nfin"],
                names[d["gate"]],
                tuple(sorted(names[n] for n in d["terminals"])),
            )
            for d in devices
        )
    except KeyError as exc:
        raise ValueError("6T topology has an unexpected internal net") from exc
    expected = Counter(
        [
            (False, 2, "WL", ("BL", "Q")),
            (False, 2, "WL", ("BLN", "QB")),
            (False, 2, "QB", ("Q", "VSS")),
            (False, 2, "Q", ("QB", "VSS")),
            (True, 1, "QB", ("Q", "VDD")),
            (True, 1, "Q", ("QB", "VDD")),
        ]
    )
    if actual != expected:
        raise ValueError(
            "extracted 6T topology or 1/2/2 sizing differs from the reference schematic"
        )
    checks = []
    for q in (0, 1):
        for wl in (0, 1):
            gates = {"Q": q, "QB": 1 - q, "WL": wl}
            parent = {
                name: name for name in ("Q", "QB", "WL", "BL", "BLN", "VDD", "VSS")
            }

            def find(name: str, parent=parent) -> str:
                while name != parent[name]:
                    name = parent[name]
                return name

            for pmos, _, gate, (a, b) in actual:
                if gates[gate] == (0 if pmos else 1):
                    parent[find(a)] = find(b)
            if (
                find("VDD") == find("VSS")
                or find("Q") != find("VDD" if q else "VSS")
                or find("QB") != find("VSS" if q else "VDD")
            ):
                raise ValueError(
                    "extracted 6T network does not support the complementary hold state"
                )
            for bitline, storage in (("BL", "Q"), ("BLN", "QB")):
                if (find(bitline) == find(storage)) != bool(wl):
                    raise ValueError(
                        "6T read access/isolation differs from the schematic"
                    )
            checks.append(
                {"Q": q, "QB": 1 - q, "WL": wl, "mode": "read_path" if wl else "hold"}
            )
    return {
        "transistors": 6,
        "pu_fins": 1,
        "pd_fins": 2,
        "pg_fins": 2,
        "external_supply_landing_roots": {
            pin_names[k]: len(v)
            for k, v in labels.items()
            if pin_names[k] in ("VDD", "VSS")
        },
        "ideal_switch_checks": checks,
    }


def prove_minimum(
    source: Path,
    out: Path,
    *,
    pdk: Path,
    z3: Path,
    gdscheck: Path,
    cell_name: str = CELL_NAME,
    array_name: str = ARRAY_NAME,
) -> dict[str, Any]:
    """Write replayable proof artifacts; any failure replaces a previous success."""
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "format": "chipforge_asap7.sram_6t_minimum.v1",
        "status": "incomplete",
        "assumptions": ASSUMPTIONS,
        "limits": LIMITS,
    }
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        import gdstk

        report["rule_source"] = validate_rule_source(pdk, rule_manifest())
        report["tools"] = {
            "z3": tool_info(z3, "-version"),
            "gdscheck": tool_info(gdscheck, "--version"),
        }
        package = Path(__file__).resolve().parents[1]
        report["implementation_sha256"] = {
            str(p.relative_to(package)): sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob("*.py"))
        }
        report["source"] = {
            "path": str(source.resolve()),
            "sha256": sha256(source.read_bytes()).hexdigest(),
            "cell": cell_name,
            "array": array_name,
        }
        library = gdstk.read_gds(str(source), unit=1e-6)
        cells = {cell.name: cell for cell in library.cells}
        cell, array = cells[cell_name], cells[array_name]
        if cell not in array.dependencies(True):
            raise ValueError("array witness does not instantiate the measured leaf")
        flat = cell.copy("measured_6t").flatten()
        witness = measure_tile(flat)
        width, height = minimum_dimensions()
        if (witness["width"], witness["height"]) != (width, height):
            raise ValueError(
                "drawn 6T cell does not attain the retained template's lower bound"
            )
        report["model_witness"] = witness
        report["extracted_topology"] = verify_topology(flat)
        report["queries"] = {
            name: run_query(z3, script, out / f"{name}.smt2", status)
            for name, (script, status) in proof_queries(witness).items()
        }
        output = gdstk.Library(unit=1e-6, precision=1e-10)
        output.add(array, *sorted(array.dependencies(True), key=lambda c: c.name))
        gds = out / "witness.gds"
        output.write_gds(str(gds))
        report["gds"] = {
            "path": gds.name,
            "sha256": sha256(gds.read_bytes()).hexdigest(),
        }
        report["drc"] = {}
        for name in (cell_name, array_name):
            found = run_gdscheck(
                gds,
                out / "drc",
                cell_name=name,
                binary=gdscheck,
                pdk_file=pdk / "pdk.yml",
            )
            report["drc"][name] = {
                "suite": "main",
                "violations": dict(drc_counts(found)),
            }
            if found:
                raise ValueError(f"6T witness {name} fails the pinned gdscheck suite")
        report["minimum"] = {
            "width_nm": width,
            "height_nm": height,
            "area_nm2": width * height,
            "area_um2": width * height / 1e6,
        }
        report["area_argument"] = (
            "Positive width >= 108 and height >= 270 imply area >= 29160 nm². The measured leaf attains both bounds under the stated template assumptions."
        )
        report["status"] = "proved_for_retained_6t_122_template"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gds", type=Path, required=True)
    parser.add_argument("--gdscheck-root", type=Path, required=True)
    parser.add_argument("--cell", default=CELL_NAME)
    parser.add_argument("--array-cell", default=ARRAY_NAME)
    parser.add_argument("--gdscheck", type=Path)
    parser.add_argument("--z3", type=Path, default=shutil.which("z3"))
    parser.add_argument("--out", type=Path, default=Path("build/sram_6t_minimum"))
    args = parser.parse_args(argv)
    if args.z3 is None:
        parser.error("Z3 is required on PATH or through --z3")
    try:
        result = prove_minimum(
            args.gds,
            args.out,
            pdk=args.gdscheck_root / "pdks/asap7",
            z3=Path(args.z3),
            gdscheck=find_gdscheck(args.gdscheck),
            cell_name=args.cell,
            array_name=args.array_cell,
        )
    except (
        ValueError,
        RuntimeError,
        OSError,
        KeyError,
        ImportError,
        subprocess.SubprocessError,
    ) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(
        f"{result['status']}: {result['minimum']['area_um2']:.5f} µm²/bit -> {args.out / 'report.json'}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
