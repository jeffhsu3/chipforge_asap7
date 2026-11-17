"""Scoped proof, counterexamples, source drift and constructive GDS witnesses."""

import json
import shutil
from pathlib import Path

import pytest

from chipforge_asap7.devices import (
    RomArraySpec,
    RomBitcellSpec,
    build_rom_array,
    build_rom_bitcell,
)
from chipforge_asap7.layout.layers import box, require_gdspy
from chipforge_asap7.verification.drc import find_gdscheck
from chipforge_asap7.verification.rom_minimum import (
    extract_witness,
    layout_constraints,
    measure_routing,
    measure_tile,
    minimum_dimensions,
    proof_queries,
    prove_minimum,
    routing_constraints,
    rule_manifest,
    run_query,
    topology_check,
    validate_rule_source,
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


@pytest.mark.parametrize("fins", [1, 2, 3])
def test_no_smaller_dimension_and_drawn_witness_is_satisfiable(fins, z3, tmp_path):
    witness = measure_tile(build_rom_bitcell(RomBitcellSpec(fins=fins)), fins)
    assert (witness["width"], witness["height"]) == minimum_dimensions(fins)
    for name, (source, expected) in proof_queries(fins, witness).items():
        result = run_query(z3, source, tmp_path / f"{name}.smt2", expected)
        if expected == "unsat":
            assert "(proof" in (tmp_path / result["output"]).read_text()


def _routed_witness(fins, tmp_path):
    """Both tile measurements, with V1s read from a fully programmed row."""
    gdspy = require_gdspy()
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    bitcell = RomBitcellSpec(fins=fins)
    tile = build_rom_bitcell(bitcell, lib=library)
    spec = RomArraySpec(words=2, bits=1, bitcell=bitcell)
    top = build_rom_array(spec, [(0, 0), (1, 0)], lib=library)
    return measure_tile(tile, fins), measure_routing(tile, top, spec, bit=0)


def _without(source, *names):
    return (
        "\n".join(
            line
            for line in source.splitlines()
            if not any(f":named {name})" in line for name in names)
        )
        + "\n"
    )


@pytest.mark.parametrize("fins", [1, 2, 3])
def test_drawn_routing_satisfies_the_routing_rules_at_the_bound(fins, z3, tmp_path):
    witness, routing = _routed_witness(fins, tmp_path)
    source, expected = proof_queries(fins, witness, routing)["routed_drawn_witness"]
    assert expected == "sat"
    run_query(z3, source, tmp_path / "routed.smt2", "sat")


@pytest.mark.parametrize(
    ("change", "rules"),
    [
        # M1.S.2: a stub's 18 nm tip 24 nm from the VSS column's long side.
        ({"m1_da_x1": 21}, ("m1_space_da_s",)),
        # M1.S.4: two 18 nm tips 29 nm apart across the tile boundary.
        ({"m1_db_x1": 124}, ("m1_space_db_da",)),
        # V0.M1.AUX.3: the full-ACTIVE-height pad the FinFET used to draw.
        ({"m1_da_y0": 0, "m1_da_y1": 54}, ("m1_stub_flush_da",)),
        # M1.A.1: a 27 x 18 nm stub is 486 nm².
        ({"m1_da_x1": 18}, ("m1_stub_area_da",)),
        # V0.LISD.EN.2: 2 nm of LISD either side of the V0.
        (
            {"lisd_da_x0": -11, "lisd_da_x1": 11},
            ("v0_lisd_enclosure_da", "lisd_width_da"),
        ),
    ],
)
def test_each_routing_rule_rejects_its_own_violation(change, rules, z3, tmp_path):
    witness, routing = _routed_witness(2, tmp_path)
    source, _ = proof_queries(2, witness, {**routing, **change})["routed_drawn_witness"]
    run_query(
        z3,
        source.replace("(get-model)", "(get-proof)"),
        tmp_path / "broken.smt2",
        "unsat",
    )
    # And only that rule: without it the same geometry is accepted.
    run_query(z3, _without(source, *rules), tmp_path / "relaxed.smt2", "sat")


@pytest.mark.parametrize("fins", [1, 2, 3])
def test_routing_alone_holds_the_row_pitch_at_the_minimum_width(fins, z3, tmp_path):
    # Without ACTIVE.S.1, LISD row spacing still needs 27 nm between the bars:
    # gates 54 nm apart leave a bar 34 nm wide at most, so its ends are `mid`.
    width, height = minimum_dimensions(fins)
    source = _without(layout_constraints(fins), "active_wrap") + routing_constraints(
        fins
    )
    bound = f"(assert (<= width {width}))\n(assert (< height {height}))\n(check-sat)\n"
    run_query(z3, source + bound, tmp_path / "pitch.smt2", "unsat")
    # A wider tile lets the bars grow long-edged, and 18 nm is then enough.
    wider = f"(assert (< height {height}))\n(check-sat)\n"
    run_query(z3, source + wider, tmp_path / "wider.smt2", "sat")


def test_routing_measurement_needs_both_programming_vias(tmp_path):
    gdspy = require_gdspy()
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    bitcell = RomBitcellSpec(fins=2)
    tile = build_rom_bitcell(bitcell, lib=library)
    spec = RomArraySpec(words=2, bits=1, bitcell=bitcell)
    top = build_rom_array(spec, [(0, 0)], lib=library)
    with pytest.raises(ValueError, match="both words"):
        measure_routing(tile, top, spec, bit=0)


@pytest.mark.parametrize(
    ("constraint", "query"),
    [
        ("gate_wrap", "(< width 162)"),
        ("active_wrap", "(< height 81)"),
        ("active_lower", "(< height 81)"),
        ("fin_spacing_1", "(< height 81)"),
    ],
)
def test_removing_a_necessary_constraint_allows_a_smaller_tile(
    constraint, query, z3, tmp_path
):
    # These must become SAT. Otherwise the model could contain its own
    # conclusion, omit the periodic boundary, or be UNSAT for unrelated reasons.
    source = "\n".join(
        line
        for line in layout_constraints(2).splitlines()
        if f":named {constraint})" not in line
    )
    source += f"\n(assert {query})\n(check-sat)\n(get-model)\n"
    run_query(z3, source, tmp_path / "relaxed.smt2", "sat")


@pytest.mark.parametrize("dimension", ["width", "height"])
def test_a_shrunken_drawn_witness_is_rejected(dimension, z3, tmp_path):
    witness = measure_tile(build_rom_bitcell(RomBitcellSpec(fins=2)), 2)
    witness[dimension] -= 0.25
    source, _ = proof_queries(2, witness)["drawn_witness"]
    source = source.replace("(get-model)", "(get-proof)")
    run_query(z3, source, tmp_path / "shrunk.smt2", "unsat")


def test_drawn_gate_and_rule_flavor_changes_cannot_pass_as_the_same_topology():
    cell = build_rom_bitcell(RomBitcellSpec())
    box(cell, "SRAMDRC", 0, 0, 162, 81)
    with pytest.raises(ValueError, match="SRAMDRC"):
        measure_tile(cell, 2)
    cell = build_rom_bitcell(RomBitcellSpec())
    box(cell, "GATE", 179, 0, 199, 81)
    with pytest.raises(ValueError, match="three full-height"):
        measure_tile(cell, 2)


@pytest.mark.parametrize("fins", [0, -1, True, 1.5])
def test_invalid_fin_counts_are_not_proved(fins):
    with pytest.raises(ValueError, match="positive integer"):
        minimum_dimensions(fins)


def test_independent_programming_requires_the_isolation_in_this_topology():
    result = topology_check()
    assert result["cases_passed"] == 256
    example = result["without_isolation_counterexample"]
    assert example["program_mask"] & example["selected_wordlines"] == 0
    assert not example["expected_discharge"] and example["actual_discharge"]


def test_rule_changes_require_reauditing_the_proof(pdk, tmp_path):
    clone = tmp_path / "asap7"
    shutil.copytree(pdk, clone)
    assert validate_rule_source(clone) == rule_manifest()
    active = clone / "decks/active.yml"
    active.write_text(active.read_text().replace("value: 0.027", "value: 0.026"))
    with pytest.raises(ValueError, match="decks/active.yml"):
        validate_rule_source(clone)


def test_a_failed_rerun_replaces_a_previous_success_report(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"status": "proved_for_stated_topology"}))
    with pytest.raises(ValueError, match="rule source"):
        prove_minimum(
            2,
            tmp_path,
            pdk=tmp_path / "missing",
            z3=Path("unused"),
            gdscheck=Path("unused"),
        )
    assert json.loads(report.read_text())["status"] == "failed"


def test_solver_sat_cannot_be_reported_as_a_proof(z3, tmp_path):
    with pytest.raises(RuntimeError, match="expected unsat, got sat"):
        run_query(
            z3, "(set-logic QF_LRA)\n(check-sat)\n", tmp_path / "sat.smt2", "unsat"
        )


def test_extraction_rejects_a_missing_programming_via(tmp_path):
    pytest.importorskip("gdstk")
    gdspy = require_gdspy()
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    spec = RomArraySpec(words=4, bits=16, bitcell=RomBitcellSpec(fins=1))
    sites = [
        (word, bit)
        for bit in range(16)
        for word in range(4)
        if bit & (1 << word) and (word, bit) != (0, 1)
    ]
    top = build_rom_array(spec, sites, lib=library)
    gds = tmp_path / "wrong_program.gds"
    library.write_gds(str(gds))
    with pytest.raises(ValueError, match="programmed drain"):
        extract_witness(gds, top.name, spec)


@pytest.mark.parametrize("fins", [1, 2, 3])
def test_complete_proof_includes_pinned_drc_and_extracted_all_pattern_witness(
    fins, z3, pdk, tmp_path, external_tool
):
    pytest.importorskip("gdstk")
    try:
        binary = find_gdscheck()
    except FileNotFoundError:
        external_tool(False, "gdscheck is not installed")
    result = prove_minimum(fins, tmp_path, pdk=pdk, z3=z3, gdscheck=binary)
    assert result["status"] == "proved_for_stated_topology"
    assert result["minimum"]["core_area_per_bit_nm2"] == 2187 * (fins + 1)
    assert result["drc"]["violations"] == {}
    assert result["extracted_connectivity"] == {
        "programming_patterns": 16,
        "word_devices": 64,
        "grounded_isolation_devices": 32,
    }
    assert result["queries"]["smaller_either_dimension"]["status"] == "unsat"
    assert result["queries"]["routed_drawn_witness"]["status"] == "sat"
    assert result["drawn_routing_nm"]["m1_s_y1"] == 27 * (fins + 1)
