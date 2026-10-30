"""Conditional 6T bounds, scope counterexamples and measured GDS witnesses."""

import json
import shutil
from hashlib import sha256
from pathlib import Path

import pytest

from chipforge_asap7.layout.layers import LAYERS
from chipforge_asap7.verification.drc import find_gdscheck
from chipforge_asap7.verification.minimum_proof import run_query, validate_rule_source
from chipforge_asap7.verification.sram_6t_minimum import (
    ARRAY_NAME,
    CELL_NAME,
    layout_constraints,
    measure_tile,
    minimum_dimensions,
    proof_queries,
    prove_minimum,
    rule_manifest,
    verify_topology,
)


@pytest.fixture
def z3(external_tool):
    executable = shutil.which("z3")
    external_tool(executable is not None, "Z3 is not installed")
    return Path(executable)


@pytest.fixture
def pdk(external_tool):
    root = Path(__file__).resolve().parents[2] / "gdscheck/pdks/asap7"
    external_tool(root.is_dir(), "the pinned gdscheck checkout is missing")
    return root


@pytest.fixture
def require_gdscheck(external_tool):
    try:
        return find_gdscheck()
    except FileNotFoundError:
        external_tool(False, "gdscheck is not installed")


@pytest.fixture
def source_gds(external_tool):
    path = (
        Path(__file__).resolve().parents[2]
        / "OpenFinRAM/tech/gds/srambank_32b_boundary_2.gds"
    )
    external_tool(path.is_file(), "the OpenFinRAM 6T reference GDS is missing")
    return path


@pytest.fixture
def leaf(source_gds):
    gdstk = pytest.importorskip("gdstk")
    library = gdstk.read_gds(str(source_gds), unit=1e-6)
    cell = next(cell for cell in library.cells if cell.name == CELL_NAME)
    return cell.copy("test_6t").flatten()


def test_measured_gds_attains_template_bound_and_extracts_exact_six_devices(leaf):
    witness = measure_tile(leaf)
    assert (witness["width"], witness["height"]) == minimum_dimensions() == (108, 270)
    assert [witness[f"f{band}_0"] for band in range(4)] == [1, 4, 6, 8]
    extracted = verify_topology(leaf)
    assert extracted["transistors"] == 6
    assert (extracted["pu_fins"], extracted["pd_fins"], extracted["pg_fins"]) == (
        1,
        2,
        2,
    )
    assert extracted["external_supply_landing_roots"] == {"VDD": 1, "VSS": 2}
    assert {(case["Q"], case["WL"]) for case in extracted["ideal_switch_checks"]} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }


def test_no_smaller_dimension_but_relaxing_enclosure_permits_smaller_cell(
    leaf, z3, tmp_path
):
    queries = proof_queries(measure_tile(leaf))
    assert {name: expected for name, (_, expected) in queries.items()} == {
        "smaller_width": "unsat",
        "smaller_height": "unsat",
        "smaller_either_dimension": "unsat",
        "drawn_witness": "sat",
        "relaxed_enclosure_smaller_height": "sat",
    }
    for name, (source, expected) in queries.items():
        result = run_query(z3, source, tmp_path / f"{name}.smt2", expected)
        if expected == "unsat":
            assert "(proof" in (tmp_path / result["output"]).read_text()


@pytest.mark.parametrize(
    ("constraint", "query"),
    [
        ("gate_wrap", "(< width 108)"),
        ("active_wrap", "(< height 270)"),
        ("separate_bands_2", "(< height 270)"),
        ("distinct_fins_0_1", "(< height 270)"),
    ],
)
def test_removing_essential_template_constraints_allows_a_smaller_cell(
    constraint, query, z3, tmp_path
):
    # Guard against an inconsistent model or asserting the desired conclusion.
    source = "\n".join(
        line
        for line in layout_constraints().splitlines()
        if f":named {constraint})" not in line
    )
    source += f"\n(assert {query})\n(check-sat)\n(get-model)\n"
    run_query(z3, source, tmp_path / "relaxed.smt2", "sat")


@pytest.mark.parametrize(
    ("dimension", "period"), [("width", "period_x"), ("height", "period_y")]
)
def test_shrunken_witness_is_rejected_even_with_a_matching_period(
    dimension, period, leaf, z3, tmp_path
):
    witness = measure_tile(leaf)
    witness[dimension] -= {"width": 54, "height": 27}[dimension]
    witness[period] -= 1
    source, _ = proof_queries(witness)["drawn_witness"]
    source = source.replace("(get-model)", "(get-proof)")
    run_query(z3, source, tmp_path / "shrunk.smt2", "unsat")


def test_marker_and_retained_lattices_are_checked_in_the_gds(leaf):
    marker = LAYERS["SRAMDRC"]
    leaf.remove(*(p for p in leaf.polygons if p.layer == marker["layer"]))
    with pytest.raises(ValueError, match="SRAMDRC"):
        measure_tile(leaf)


def test_off_grid_gate_is_rejected(leaf):
    gate = next(p for p in leaf.polygons if p.layer == LAYERS["GATE"]["layer"])
    gate.translate(0.001, 0)
    with pytest.raises(ValueError, match="lattice"):
        measure_tile(leaf)


def test_missing_fin_is_rejected(leaf):
    fin = next(
        p
        for p in leaf.polygons
        if p.layer == LAYERS["FIN"]["layer"]
        and abs(sum(y for _, y in p.points) / 4 - 0.027) < 1e-8
    )
    leaf.remove(fin)
    with pytest.raises(ValueError, match="fin counts"):
        measure_tile(leaf)


def test_broken_feedback_is_rejected_by_extraction(leaf):
    feedback = next(
        p
        for p in leaf.polygons
        if p.layer == LAYERS["LIG"]["layer"]
        and abs(p.bounding_box()[0][1] - 0.0935) < 1e-8
    )
    leaf.remove(feedback)
    # Area and lattice checks alone cannot detect a missing cross-coupling wire.
    assert measure_tile(leaf)["height"] == 270
    with pytest.raises(ValueError, match="6T topology"):
        verify_topology(leaf)


def test_sram_rule_change_requires_reauditing_the_proof(pdk, tmp_path):
    clone = tmp_path / "asap7"
    shutil.copytree(pdk, clone)
    assert validate_rule_source(clone, rule_manifest()) == rule_manifest()
    deck = clone / "decks/active.yml"
    deck.write_text(deck.read_text().replace("value: 0.0135", "value: 0.014"))
    with pytest.raises(ValueError, match="decks/active.yml"):
        validate_rule_source(clone, rule_manifest())


def test_failed_rerun_replaces_a_previous_success_report(tmp_path):
    pytest.importorskip("gdstk")
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"status": "proved_for_retained_6t_122_template"}))
    with pytest.raises(ValueError, match="rule source"):
        prove_minimum(
            Path("unused"),
            tmp_path,
            pdk=tmp_path / "missing",
            z3=Path("unused"),
            gdscheck=Path("unused"),
        )
    assert json.loads(path.read_text())["status"] == "failed"


def test_full_proof_binds_smt_topology_drc_and_input_hashes(
    source_gds, pdk, z3, require_gdscheck, tmp_path
):
    pytest.importorskip("gdstk")
    result = prove_minimum(
        source_gds, tmp_path, pdk=pdk, z3=z3, gdscheck=require_gdscheck
    )
    assert result["status"] == "proved_for_retained_6t_122_template"
    assert result["minimum"] == {
        "width_nm": 108,
        "height_nm": 270,
        "area_nm2": 29160,
        "area_um2": 0.02916,
    }
    assert result["drc"] == {
        CELL_NAME: {"suite": "main", "violations": {}},
        ARRAY_NAME: {"suite": "main", "violations": {}},
    }
    assert result["queries"]["relaxed_enclosure_smaller_height"]["status"] == "sat"
    assert result["source"]["sha256"] == sha256(source_gds.read_bytes()).hexdigest()
    assert (
        result["gds"]["sha256"]
        == sha256((tmp_path / "witness.gds").read_bytes()).hexdigest()
    )
    for query in result["queries"].values():
        assert (
            query["query_sha256"]
            == sha256((tmp_path / query["query"]).read_bytes()).hexdigest()
        )
