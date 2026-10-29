"""CP-SAT bound, finite witness and deliberately broken proof controls."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from chipforge_asap7.brom.selector import SelectorPlan, build_selector_array
from chipforge_asap7.layout.layers import LAYERS, box
from chipforge_asap7.verification.brom_minimum import (
    measure_witness,
    model,
    prove_minimum,
    rule_manifest,
    solve,
)
from chipforge_asap7.verification.minimum_proof import validate_rule_source


@pytest.fixture
def solver(external_tool):
    external_tool(
        importlib.util.find_spec("ortools") is not None, "OR-Tools is missing"
    )


@pytest.fixture
def pdk(external_tool):
    path = Path(__file__).resolve().parents[2] / "gdscheck/pdks/asap7"
    external_tool(path.is_dir(), "pinned ASAP7 rule source is missing")
    return path


def example_plan():
    return SelectorPlan(
        2,
        108,
        432,
        tuple(
            [("M2", y) for y in (90, 171, 207, 243, 279, 315, 351, 387)]
            + [("M4", 60 + 48 * i) for i in range(8)]
        ),
    )


@pytest.mark.parametrize("fins", [1, 2, 3])
def test_complete_physical_minimum_proof(
    fins, solver, pdk, tmp_path, external_tool, require_klayout
):
    from chipforge_asap7.verification.drc import find_gdscheck

    try:
        find_gdscheck()
    except FileNotFoundError:
        external_tool(False, "gdscheck is missing")
    external_tool(importlib.util.find_spec("gdstk") is not None, "gdstk is missing")
    result = prove_minimum(fins, tmp_path / f"f{fins}", pdk=pdk)
    assert result["status"] == "proved_for_stated_topology"
    assert result["minimum"] == {
        "width_nm": 108,
        "height_nm": 432,
        "selector_area_nm2": 46656,
        "core_area_per_bit_nm2": 11664,
    }
    assert result["lower_bound"]["bound_nm2"] == 46656
    assert result["strictly_smaller"]["status"] == "INFEASIBLE"
    assert result["drawn_model"]["status"] == "OPTIMAL"
    assert result["drc"] == {}
    assert result["lvs"] == {
        "matched": True,
        "renamed_pins": [],
        "pin_inventory_matches": True,
    }
    assert result["extraction"]["selectors"] == 32
    assert result["extraction"]["switch_observations"] == 256
    assert result["missing_via_rejected"]
    assert "connectivity" in result["missing_via_extraction_error"]
    for negative in result["negative_controls"].values():
        assert negative["status"] in ("OPTIMAL", "FEASIBLE")
        assert negative["area_nm2"] < 46656


def test_all_m2_requires_a_larger_period(solver, tmp_path):
    m, v = model(2)
    for layer in v["layers"]:
        m.add(layer == 0)
    result = solve(m, v, tmp_path / "all_m2")
    assert result["status"] == "OPTIMAL"
    assert result["height_nm"] == 594  # ceil(16 * 36 / 27) * 27
    assert result["area_nm2"] > 46656


def test_base_geometry_is_data_independent():
    plan = example_plan()
    a = build_selector_array(plan, (tuple(range(16)), tuple(range(16))))
    b = build_selector_array(
        plan, (tuple(reversed(range(16))), tuple(reversed(range(16))))
    )
    pa, pb = a.get_polygons(by_spec=True), b.get_polygons(by_spec=True)
    cuts = {LAYERS[m]["layer"] for m in ("V2", "V3")}
    from collections import Counter

    for layer in pa.keys() | pb.keys():
        if layer[0] not in cuts:
            assert Counter(p.tobytes() for p in pa[layer]) == Counter(
                p.tobytes() for p in pb[layer]
            )
    assert any(
        Counter(p.tobytes() for p in pa[layer])
        != Counter(p.tobytes() for p in pb[layer])
        for layer in pa
        if layer[0] in cuts
    )


def test_drawn_witness_is_measured_and_rejects_tampering():
    plan = example_plan()
    top = build_selector_array(plan, (tuple(range(16)), tuple(range(16))))
    assert measure_witness(top, plan, 2, 16)["height"] == 432
    box(top, "M4", -48, 408, 16 * 108 + 48, 432)
    with pytest.raises(ValueError, match="nonperiodic"):
        measure_witness(top, plan, 2, 16)


def test_source_drift_invalidates_previous_success(tmp_path, pdk):
    out = tmp_path / "proof"
    out.mkdir()
    (out / "report.json").write_text('{"status":"proved_for_stated_topology"}')
    with pytest.raises(ValueError, match="differs from the proof"):
        prove_minimum(2, out, pdk=tmp_path / "missing_pdk")
    assert json.loads((out / "report.json").read_text())["status"] == "failed"
    manifest = rule_manifest()
    manifest["sha256"]["decks/m4.yml"] = "changed"
    with pytest.raises(ValueError, match="m4.yml"):
        validate_rule_source(pdk, manifest)


@pytest.mark.parametrize(
    "change",
    [
        {"height": 0},
        {"height": 216},
        {"fins": True},
        {"fins": 4},
        {"width": 54},
        {"tracks": (("M5", 12),) * 16},
    ],
)
def test_rejects_unsupported_witnesses(change):
    with pytest.raises(ValueError):
        replace(example_plan(), **change)
