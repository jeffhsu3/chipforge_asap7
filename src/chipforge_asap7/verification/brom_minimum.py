"""CP-SAT area bound and physically checked witness for a periodic BROM core.

The bound is a necessary-rule relaxation. A separate routing-constrained
solve constructs a witness at that bound. Neither solver success nor DRC
alone is reported as a proved physical minimum.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Any

from ..brom.selector import SelectorPlan, build_selector_array, selector_lvs_reference
from ..layout.layers import LAYERS, require_gdspy
from .drc import drc_counts, find_gdscheck, run_gdscheck
from .lvs import find_klayout, run_lvs
from .lvs_report import summarize_lvsdb
from .minimum_proof import tool_info, validate_rule_source

SCALE = 2  # half-nanometre coordinates; the drawn fin centres can be half-integers
EXTENT = 2048 * SCALE
ASSUMPTIONS = (
    "One selector and one grounded isolation gate per rectangular X period, continuous ACTIVE.",
    "One independent selector per block/column stores four logical bits.",
    "Sixteen distinct full-width horizontal candidate conductors use only M2 and M4.",
    "Each M2 conductor is at least 18 nm wide; M4 is minimum-width 24 nm on 48 nm tracks.",
    "Identical blocks repeat in Y on the global 27 nm FIN lattice and, when used, the M4 lattice.",
    "Candidate tracks keep their identity across X; no multiplexing, jogs or additional routing layers.",
    "Coordinates lie on a 0.5 nm grid; ordinary ASAP7 rules apply, without SRAMDRC exceptions.",
)
LIMITS = (
    "Core area only; array edges, taps, power closure, CGEN and other periphery are excluded.",
    "This is not a minimum over other selector topologies, layer assignments or multi-block supercells.",
    "The lower bound omits contact/routing details; only an independently checked witness can attain it.",
    "CP-SAT OPTIMAL/INFEASIBLE results are trusted solver conclusions, not independently checked proof certificates.",
    "Finite DRC/LVS witnesses do not establish every array size, pattern, timing, leakage or PVT behavior.",
)


def rule_manifest() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("brom_minimum_rules.json").read_text())


def model(fins: int, *, routed: bool = False, omit: frozenset[str] = frozenset()):
    """Necessary constraints, without inserting the expected optimal dimensions."""
    from ortools.sat.python import cp_model

    if type(fins) is not int or not 1 <= fins <= 3:
        raise ValueError("the constructive proof supports 1..3 fins")
    facts = {k: v["value"] * SCALE for k, v in rule_manifest()["facts"].items()}
    m = cp_model.CpModel()
    v = {
        k: m.new_int_var(0, EXTENT, k)
        for k in ("width", "height", "ga", "gi", "alo", "ahi")
    }
    w, h, ga, gi = (v[k] for k in ("width", "height", "ga", "gi"))
    m.add(w >= 1)
    m.add(h >= 1)
    m.add(ga < w)
    m.add(gi < w)
    gate_pitch = facts["gate_width_nm"] + facts["gate_space_nm"]
    m.add(gi - ga >= gate_pitch)
    if "gate_wrap" not in omit:
        m.add(w + ga - gi >= gate_pitch)
    fs = [m.new_int_var(0, EXTENT, f"fin{i}") for i in range(fins)]
    v.update({f"fin{i}": f for i, f in enumerate(fs)})
    m.add(fs[0] - v["alo"] >= facts["active_fin_enclosure_nm"])
    m.add(v["ahi"] - fs[-1] >= facts["fin_width_nm"] + facts["active_fin_enclosure_nm"])
    for a, b in pairwise(fs):
        m.add(b - a >= facts["fin_pitch_nm"])
    m.add(h + v["alo"] - v["ahi"] >= facts["active_space_nm"])
    if "fin_period" not in omit:
        periods = m.new_int_var(1, EXTENT, "fin_periods")
        m.add(h == facts["fin_pitch_nm"] * periods)
    layers = [m.new_bool_var(f"m4_{i}") for i in range(16)]
    ys = [m.new_int_var(0, EXTENT, f"y{i}") for i in range(16)]
    any_m4 = m.new_bool_var("any_m4")
    m.add_max_equality(any_m4, layers)
    if "m4_period" not in omit:
        periods = m.new_int_var(1, EXTENT, "m4_periods")
        m.add(h == facts["m4_pitch_nm"] * periods).only_enforce_if(any_m4)
    for i, (layer, y) in enumerate(zip(layers, ys)):
        m.add(y < h)
        track = m.new_int_var(0, EXTENT, f"m4_track_{i}")
        m.add(
            y == facts["m4_offset_nm"] + facts["m4_pitch_nm"] * track
        ).only_enforce_if(layer)
        if i:
            # Interchangeable candidates: M2 first, then M4, each ordered by Y.
            m.add(layers[i - 1] <= layer)
        for j in range(i):
            for on_m4, pitch in (
                (False, facts["m2_pitch_nm"]),
                (True, facts["m4_pitch_nm"]),
            ):
                conditions = [layer, layers[j]] if on_m4 else [~layer, ~layers[j]]
                m.add(y - ys[j] >= pitch).only_enforce_if(conditions)
                if "track_wrap" not in omit:
                    m.add(h - (y - ys[j]) >= pitch).only_enforce_if(conditions)
    if routed:
        # Constructive route template: M2 drain/gate hooks and one bounded M3 stem.
        gy, dy = 27 * (fins + 3) * SCALE, int((27 + 13.5 * fins) * SCALE)
        for i, (layer, y) in enumerate(zip(layers, ys)):
            m.add(y >= 17 * SCALE)
            m.add(y <= h - 17 * SCALE)
            for name, hook in (("gate", gy), ("drain", dy)):
                delta = m.new_int_var(0, EXTENT, f"{name}_clearance_{i}")
                m.add_abs_equality(delta, y - hook)
                m.add(delta >= facts["m2_pitch_nm"]).only_enforce_if(~layer)
        low, high = (
            m.new_int_var(0, EXTENT, "stem_low"),
            m.new_int_var(0, EXTENT, "stem_high"),
        )
        m.add_min_equality(low, [*ys, gy])
        m.add_max_equality(high, [*ys, gy])
        m.add(h - (high - low) >= 2 * 17 * SCALE + facts["m3_tip_space_nm"])
        # Pin the FEOL witness placement, not its periodic width/height.
        m.add(ga == 27 * SCALE)
        m.add(gi == 81 * SCALE)
        m.add(v["alo"] == 27 * SCALE)
        m.add(v["ahi"] == 27 * (fins + 1) * SCALE)
        for i, f in enumerate(fs):
            m.add(f == (37 + 27 * i) * SCALE)
    area = m.new_int_var(1, EXTENT * EXTENT, "area")
    m.add_multiplication_equality(area, [w, h])
    v.update(area=area, layers=layers, ys=ys)
    return m, v


def solve(m, v, out: Path, *, optimize: bool = True) -> dict[str, Any]:
    import ortools
    from ortools.sat.python import cp_model

    out.parent.mkdir(parents=True, exist_ok=True)
    if optimize:
        m.minimize(v["area"])
    m.export_to_file(str(out.with_suffix(".pbtxt")))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = 60
    logs = []
    solver.parameters.log_search_progress = True
    solver.parameters.log_to_stdout = False
    solver.log_callback = logs.append
    status = solver.solve(m)
    out.with_suffix(".log").write_text("\n".join(logs) + "\n" + solver.response_stats())
    out.with_suffix(".response.txt").write_text(str(solver.response_proto))
    result = {
        "status": solver.status_name(status),
        "ortools": ortools.__version__,
        "bound_nm2": solver.best_objective_bound / SCALE**2,
        "model_sha256": sha256(out.with_suffix(".pbtxt").read_bytes()).hexdigest(),
    }
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result.update(
            width_nm=solver.value(v["width"]) / SCALE,
            height_nm=solver.value(v["height"]) / SCALE,
            area_nm2=solver.value(v["area"]) / SCALE**2,
            tracks=[
                ("M4" if solver.value(layer) else "M2", solver.value(y) / SCALE)
                for layer, y in zip(v["layers"], v["ys"])
            ],
        )
    return result


def measure_witness(
    top: Any, plan: SelectorPlan, blocks: int, columns: int
) -> dict[str, Any]:
    """Recover the periodic dimensions and candidate positions from polygons."""
    polys = top.get_polygons(by_spec=True)

    def rectangles(layer):
        spec = LAYERS[layer]
        result = []
        for points in polys.get((spec["layer"], spec["datatype"]), []):
            lo, hi = points.min(axis=0), points.max(axis=0)
            x0, y0, x1, y1 = (*map(float, lo), *map(float, hi))
            if len(points) != 4 or set(map(tuple, points)) != {
                (x0, y0),
                (x0, y1),
                (x1, y0),
                (x1, y1),
            }:
                raise ValueError(f"nonrectangular {layer} in witness")
            result.append((x0, y0, x1, y1))
        return sorted(result)

    if blocks < 2 or rectangles("SRAMDRC"):
        raise ValueError("need repeated ordinary-rule blocks")
    active = [p for p in rectangles("ACTIVE") if p[2] - p[0] > 16]
    active.sort(key=lambda p: p[1])
    if len(active) != blocks or any(p[0] != -8 for p in active):
        raise ValueError("wrong continuous diffusion inventory")
    width = (active[0][2] - 8) / columns
    height = active[1][1] - active[0][1]
    if any(
        p
        != (
            active[0][0],
            active[0][1] + b * height,
            active[0][2],
            active[0][3] + b * height,
        )
        for b, p in enumerate(active)
    ):
        raise ValueError("nonperiodic ACTIVE")
    core_w = width * columns
    gates = [p for p in rectangles("GATE") if 0 <= (p[0] + p[2]) / 2 < core_w]
    if len(gates) != 2 * columns or any(p[2] - p[0] != 20 for p in gates):
        raise ValueError("wrong periodic gate inventory")
    ga, gi = [(p[0] + p[2]) / 2 for p in gates[:2]]
    if any(
        (p[0] + p[2]) / 2 != (ga if i % 2 == 0 else gi) + (i // 2) * width
        for i, p in enumerate(gates)
    ):
        raise ValueError("nonperiodic gate positions")
    alo, ahi = active[0][1], active[0][3]
    fins = [p for p in rectangles("FIN") if p[1] < ahi and p[3] > alo]
    if len(fins) != plan.fins or any(p[3] - p[1] != 7 for p in fins):
        raise ValueError("wrong fin inventory")
    measured_tracks = []
    for metal in ("M2", "M4"):
        tracks = [p for p in rectangles(metal) if (p[0], p[2]) == (-48, core_w + 48)]
        first = [p for p in tracks if 0 <= (p[1] + p[3]) / 2 < height]
        expected = [
            (p[0], p[1] + b * height, p[2], p[3] + b * height)
            for b in range(blocks)
            for p in first
        ]
        if sorted(tracks) != sorted(expected):
            raise ValueError("nonperiodic candidate conductors")
        for p in first:
            if p[3] - p[1] != (18 if metal == "M2" else 24):
                raise ValueError("wrong candidate width")
            measured_tracks.append((metal, (p[1] + p[3]) / 2))
    if tuple(measured_tracks) != plan.tracks:
        raise ValueError("drawn candidates differ from solver plan")
    return {
        "width": width,
        "height": height,
        "ga": ga,
        "gi": gi,
        "alo": alo,
        "ahi": ahi,
        **{f"fin{i}": p[1] for i, p in enumerate(fins)},
        "tracks": measured_tracks,
    }


def extract_witness(
    gds: Path, name: str, plan: SelectorPlan, mapping
) -> dict[str, int]:
    """Independent polygon connectivity; no inferred or promoted input pins."""
    import gdstk

    from . import connectivity as cv

    flat = next(
        c for c in gdstk.read_gds(str(gds), unit=1e-6).cells if c.name == name
    ).flatten()
    merged = gdstk.Cell("merged")
    for layer, datatype in {(p.layer, p.datatype) for p in flat.polygons}:
        same = [p for p in flat.polygons if (p.layer, p.datatype) == (layer, datatype)]
        merged.add(
            *gdstk.boolean(
                same, [], "or", precision=1e-6, layer=layer, datatype=datatype
            )
        )
    merged.add(*flat.labels)
    blocks, columns = len(mapping), len(mapping[0])
    devices, labels = cv.extract_connectivity(
        merged, expected_channels=2 * blocks * columns
    )
    expected_pins = {
        *(f"C<{b},{k}>" for b in range(blocks) for k in range(16)),
        *(f"BL<{c}>" for c in range(columns)),
        "VSS",
    }
    if labels.keys() != expected_pins or any(len(ids) != 1 for ids in labels.values()):
        raise ValueError("missing or split witness pins")
    nets = {name: next(iter(ids)) for name, ids in labels.items()}
    if len(set(nets.values())) != len(nets):
        raise ValueError("witness pins are shorted")
    expected = Counter()
    for b, row in enumerate(mapping):
        for col, k in enumerate(row):
            for gate in (nets[f"C<{b},{k}>"], nets["VSS"]):
                expected[
                    (
                        False,
                        plan.fins,
                        gate,
                        frozenset((nets[f"BL<{col}>"], nets["VSS"])),
                    )
                ] += 1
    actual = Counter(
        (d["pmos"], d["nfin"], d["gate"], frozenset(d["terminals"])) for d in devices
    )
    if actual != expected:
        raise ValueError(
            "extracted selector/isolation connectivity or fin counts differ"
        )
    # Exhaust every block-enable combination and offset on the extracted switches.
    checked = 0
    for enable in range(1 << blocks):
        for offset in range(4):
            high = {
                nets[f"C<{b},{k}>"]
                for b in range(blocks)
                for k in range(16)
                if enable >> b & 1 and k >> offset & 1
            }
            for col in range(columns):
                actual_bit = any(
                    d["gate"] in high
                    and nets[f"BL<{col}>"] in d["terminals"]
                    and nets["VSS"] in d["terminals"]
                    for d in devices
                )
                expected_bit = any(
                    enable >> b & 1 and mapping[b][col] >> offset & 1
                    for b in range(blocks)
                )
                if actual_bit != expected_bit:
                    raise ValueError("extracted BROM truth table differs")
                checked += 1
    return {
        "selectors": blocks * columns,
        "isolation_devices": blocks * columns,
        "patterns": 16,
        "switch_observations": checked,
        "distinct_pins": len(nets),
    }


def extracted_pins(path: Path) -> set[str]:
    return {
        line.removeprefix("* pin ").strip()
        for line in path.read_text().splitlines()
        if line.startswith("* pin ")
    }


def prove_minimum(fins: int, out: Path, *, pdk: Path) -> dict[str, Any]:
    """Require optimum, strict lower-bound exclusion and a DRC/LVS witness."""
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "chipforge_asap7.brom_minimum.v1",
        "status": "incomplete",
        "fins": fins,
        "assumptions": ASSUMPTIONS,
        "limits": LIMITS,
    }
    report_path = out / "report.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        report["rules"] = validate_rule_source(pdk, rule_manifest())
        report["tools"] = {
            "gdscheck": tool_info(find_gdscheck(), "--version"),
            "klayout": tool_info(find_klayout(), "-v"),
        }
        from ..brom import selector

        report["implementation_sha256"] = {
            str(p): sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(selector.__file__),
                *(
                    Path(__file__).parent / name
                    for name in (
                        "asap7.lvs",
                        "connectivity.py",
                        "drc.py",
                        "lvs.py",
                        "lvs_report.py",
                        "lvs_report_pya.py",
                        "minimum_proof.py",
                    )
                ),
            )
        }
        m, v = model(fins)
        optimum = solve(m, v, out / "lower_bound")
        report["lower_bound"] = optimum
        if optimum["status"] != "OPTIMAL":
            raise RuntimeError("lower-bound solve did not prove optimality")
        area = optimum["area_nm2"]
        # Every smaller-area tile must fit in the finite search domain:
        # W >= 108 and H >= 27*(fins+1) follow independently from FEOL.
        facts = {k: v["value"] for k, v in rule_manifest()["facts"].items()}
        min_w = 2 * (facts["gate_width_nm"] + facts["gate_space_nm"])
        min_h = (
            (fins - 1) * facts["fin_pitch_nm"]
            + facts["fin_width_nm"]
            + 2 * facts["active_fin_enclosure_nm"]
            + facts["active_space_nm"]
        )
        if not area < min(min_w, min_h) * (EXTENT / SCALE):
            raise RuntimeError("finite solver domain does not cover every smaller tile")
        report["domain_coverage"] = {
            "extent_nm": EXTENT / SCALE,
            "necessary_width_nm": min_w,
            "necessary_height_nm": min_h,
            "smaller_area_requires_width_below_nm": area / min_h,
            "smaller_area_requires_height_below_nm": area / min_w,
        }
        m, v = model(fins)
        m.add(v["area"] < int(area * SCALE**2))
        exclusion = solve(m, v, out / "strictly_smaller", optimize=False)
        report["strictly_smaller"] = exclusion
        if exclusion["status"] != "INFEASIBLE":
            raise RuntimeError("strict smaller-area exclusion failed")
        m, v = model(fins, routed=True)
        routed = solve(m, v, out / "routed_witness")
        report["routed_witness"] = routed
        if routed["status"] != "OPTIMAL" or routed["area_nm2"] != area:
            raise RuntimeError("constructive routing does not attain lower bound")
        plan = SelectorPlan(
            fins,
            int(routed["width_nm"]),
            int(routed["height_nm"]),
            tuple((m, y) for m, y in routed["tracks"]),
        )
        (out / "plan.json").write_text(json.dumps(asdict(plan), indent=2) + "\n")
        mapping = (tuple(range(16)), tuple(reversed(range(16))))
        lib = require_gdspy().GdsLibrary(unit=1e-9, precision=1e-10)
        top = build_selector_array(plan, mapping, lib=lib)
        gds, reference = out / "witness.gds", out / "witness.spice"
        lib.write_gds(str(gds))
        reference.write_text(selector_lvs_reference(plan, mapping, top.name))
        drawn = measure_witness(top, plan, 2, 16)
        report["drawn"] = drawn
        m, v = model(fins, routed=True)
        for key, value in drawn.items():
            if key != "tracks":
                if value * SCALE != int(value * SCALE):
                    raise ValueError(
                        "drawn geometry is off the modeled coordinate grid"
                    )
                m.add(v[key] == int(value * SCALE))
        for i, (metal, y) in enumerate(drawn["tracks"]):
            m.add(v["ys"][i] == int(y * SCALE))
            m.add(v["layers"][i] == int(metal == "M4"))
        measured = solve(m, v, out / "drawn_witness", optimize=False)
        report["drawn_model"] = measured
        if measured["status"] != "OPTIMAL":
            raise RuntimeError("drawn polygons do not satisfy the model")
        violations = run_gdscheck(
            gds, out / "drc", cell_name=top.name, pdk_file=pdk / "pdk.yml"
        )
        report["drc"] = dict(drc_counts(violations))
        if violations:
            raise RuntimeError("witness DRC failed")
        lvs = run_lvs(gds, reference, out / "lvs", cell_name=top.name)
        summary = summarize_lvsdb(lvs.report)
        renamed = [p for c in summary["circuits"] for p in c["renamed_pins"]]
        expected_pins = set(reference.read_text().splitlines()[0].split()[2:])
        pin_inventory = extracted_pins(lvs.extracted_netlist) == expected_pins
        report["lvs"] = {
            "matched": lvs.matched,
            "renamed_pins": renamed,
            "pin_inventory_matches": pin_inventory,
        }
        if not lvs.matched or renamed or not pin_inventory:
            raise RuntimeError("witness LVS/pin-name check failed")
        report["extraction"] = extract_witness(gds, top.name, plan, mapping)
        controls = {}
        for constraint in ("gate_wrap", "fin_period", "m4_period"):
            m, v = model(fins, omit=frozenset([constraint]))
            m.add(v["area"] < int(area * SCALE**2))
            result = solve(m, v, out / f"without_{constraint}", optimize=False)
            controls[constraint] = result
            if result["status"] not in ("OPTIMAL", "FEASIBLE"):
                raise RuntimeError(f"negative control {constraint} failed")
        report["negative_controls"] = controls
        # A missing programming via must fail against the same reference.
        metal, y = plan.tracks[0]
        via = LAYERS["V2" if metal == "M2" else "V3"]["layer"]
        removed = [0]

        def remove(points, layer, datatype):
            match = (
                layer == via
                and datatype == 0
                and float(points[:, 0].mean()) == 27
                and float(points[:, 1].mean()) == y
            )
            removed[0] += int(match)
            return match

        top.remove_polygons(remove)
        if removed != [1]:
            raise RuntimeError("missing-via control did not remove exactly one cut")
        bad_gds = out / "missing_via.gds"
        lib.write_gds(str(bad_gds))
        bad = run_lvs(bad_gds, reference, out / "missing_via_lvs", cell_name=top.name)
        bad_pins = extracted_pins(bad.extracted_netlist)
        try:
            extract_witness(bad_gds, top.name, plan, mapping)
        except ValueError as exc:
            report["missing_via_extraction_error"] = str(exc)
        else:
            raise RuntimeError("missing programming via escaped extraction")
        report["missing_via_rejected"] = not bad.matched or bad_pins != expected_pins
        report["missing_via_raw_lvs_matched"] = bad.matched
        if not report["missing_via_rejected"]:
            raise RuntimeError("missing programming via escaped LVS/pin inventory")
        report["artifact_sha256"] = {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in (gds, reference, out / "plan.json", bad_gds)
        }
        report["minimum"] = {
            "width_nm": drawn["width"],
            "height_nm": drawn["height"],
            "selector_area_nm2": area,
            "core_area_per_bit_nm2": area / 4,
        }
        report["status"] = "proved_for_stated_topology"
        save()
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fins", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--gdscheck-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("build/brom_minimum"))
    args = parser.parse_args(argv)
    for fins in args.fins:
        report = prove_minimum(
            fins, args.out / f"f{fins}", pdk=args.gdscheck_root / "pdks/asap7"
        )
        print(f"{fins} fins: {report['status']}: {report['minimum']}", flush=True)


if __name__ == "__main__":
    main()
