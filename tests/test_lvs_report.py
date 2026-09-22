"""LVS of layouts the package did not write the reference for.

Three things an assembled macro needs that a generated cell does not: a
source/drain model that agrees with released CDL, flattening of cells whose
devices are completed by their neighbours, and a way past released cells whose
GDS and CDL order a series stack differently.  Each is reproduced here on a
cell small enough to read.  Skips without KLayout (``ASAP7_REQUIRE_TOOLS=1``
makes that a failure).
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import (
    InverterSpec,
    NandSpec,
    build_inverter_row,
    build_nand,
)
from chipforge_asap7.verification import run_hierarchical_lvs, run_lvs, summarize_lvsdb


@pytest.fixture(autouse=True)
def _isolated_library():
    gdspy.current_library = gdspy.GdsLibrary()


def _write(make, path: Path):
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    cell = make(library)
    library.write_gds(str(path))
    return cell


def _nand_cdl(name: str, *, near_output: str) -> str:
    """A NAND2 as a designer writes it: ``NFIN`` devices, one net inside the stack."""
    far = "B" if near_output == "A" else "A"
    return (
        f".SUBCKT {name} A B Y VDD VSS\n"
        f"MN0 Y {near_output} mid VSS nmos_rvt w=81n l=20n nfin=3\n"
        f"MN1 mid {far} VSS VSS nmos_rvt w=81n l=20n nfin=3\n"
        "MP0 Y A VDD VDD pmos_rvt w=81n l=20n nfin=3\n"
        "MP1 Y B VDD VDD pmos_rvt w=81n l=20n nfin=3\n"
        ".ENDS\n"
    )


NAND = NandSpec(rows=((3, 3),), fingers=1, abut=False)


def _pins_kept_their_names(summary: dict) -> bool:
    return not any(circuit["renamed_pins"] for circuit in summary["circuits"])


def test_a_designers_netlist_needs_the_merged_source_drain(
    tmp_path: Path, require_klayout
):
    """On the fins, the inside of a stack is a net per fin; in released CDL it is one."""
    gds = tmp_path / "nand.gds"
    _write(lambda lib: build_nand(NAND, name="ND2", lib=lib), gds)
    found = {}
    for order in "AB":
        cdl = tmp_path / f"nand_{order}.cdl"
        cdl.write_text(_nand_cdl("ND2", near_output=order))
        per_fin = run_lvs(gds, cdl, tmp_path / f"fin_{order}", cell_name="ND2",
                          asap7_standard_cell=True, merge_fin_diffusion=False)  # fmt: skip
        assert not per_fin.matched
        merged = run_lvs(gds, cdl, tmp_path / f"merged_{order}", cell_name="ND2",
                         asap7_standard_cell=True)  # fmt: skip
        assert merged.matched
        found[order] = summarize_lvsdb(merged.report)

    # One of the two orders is the layout's.  The other also "matches", but only
    # because KLayout paired A with B to make it: true of the cell, and false
    # the moment a parent wires A and B to different nets.
    faithful = [
        order for order, summary in found.items() if _pins_kept_their_names(summary)
    ]
    assert len(faithful) == 1
    (swapped,) = set("AB") - set(faithful)
    (circuit,) = found[swapped]["circuits"]
    assert sorted(circuit["renamed_pins"]) == [["A", "B"], ["B", "A"]]
    assert circuit["series_order_only"] is True


def test_a_different_gate_is_not_mistaken_for_a_reordered_one(
    tmp_path: Path, require_klayout
):
    gds = tmp_path / "nand.gds"
    _write(lambda lib: build_nand(NAND, name="ND2", lib=lib), gds)
    nor = tmp_path / "nor.cdl"
    nor.write_text(
        ".SUBCKT ND2 A B Y VDD VSS\n"
        "MN0 Y A VSS VSS nmos_rvt w=81n l=20n nfin=3\n"
        "MN1 Y B VSS VSS nmos_rvt w=81n l=20n nfin=3\n"
        "MP0 Y A mid VDD pmos_rvt w=81n l=20n nfin=3\n"
        "MP1 mid B VDD VDD pmos_rvt w=81n l=20n nfin=3\n"
        ".ENDS\n"
    )
    result = run_lvs(
        gds, nor, tmp_path / "nor", cell_name="ND2", asap7_standard_cell=True
    )
    assert not result.matched
    (circuit,) = summarize_lvsdb(result.report)["circuits"]
    assert circuit["status"] != "match" and circuit["series_order_only"] is False

    narrow = tmp_path / "narrow.cdl"  # the right gate, one fin short on one input
    narrow.write_text(_nand_cdl("ND2", near_output="A").replace("MP1 Y B VDD VDD pmos_rvt w=81n l=20n nfin=3",
                                                                   "MP1 Y B VDD VDD pmos_rvt w=54n l=20n nfin=2"))  # fmt: skip
    result = run_lvs(
        gds, narrow, tmp_path / "narrow", cell_name="ND2", asap7_standard_cell=True
    )
    assert not result.matched
    assert summarize_lvsdb(result.report)["circuits"][0]["series_order_only"] is False


def test_a_tile_completed_by_its_neighbour_has_to_be_flattened(
    tmp_path: Path, require_klayout
):
    """The abutting inverter tile owns two gates and draws three: like a bitcell,
    the cell on its own is not the circuit its schematic says it is."""
    spec = InverterSpec(rows=((4, 6),), fingers=2)
    gds = tmp_path / "row.gds"
    _write(
        lambda lib: build_inverter_row(spec, 2, name="ROW", lib=lib, leaf_name="TILE"),
        gds,
    )
    reference = tmp_path / "row.cdl"
    reference.write_text(
        ".SUBCKT TILE A Y VDD VSS\n"
        "MN Y A VSS VSS nmos_rvt w=108n l=20n nfin=8\n"
        "MP Y A VDD VDD pmos_rvt w=162n l=20n nfin=12\n"
        ".ENDS\n"
        ".SUBCKT ROW A0 Y0 A1 Y1 VDD VSS\n"
        "X0 A0 Y0 VDD VSS TILE\n"
        "X1 A1 Y1 VDD VSS TILE\n"
        ".ENDS\n"
    )
    kept = run_hierarchical_lvs(gds, reference, tmp_path / "kept", cell_name="ROW")
    assert not kept.matched and "TILE" in kept.failing
    flat = run_hierarchical_lvs(
        gds, reference, tmp_path / "flat", cell_name="ROW", flatten_circuits=("TILE",)
    )
    assert flat.matched and flat.failing == () and flat.series_order_cells == ()
    assert "LVS MATCH" in flat.describe()


def test_a_reordered_library_cell_is_excused_by_proof_and_by_name(
    tmp_path: Path, require_klayout
):
    """The released AO21x1 case in miniature: the cell's CDL has its stack the
    other way up, and KLayout matches the cell by pairing A with B.  In a macro
    those nets fan out to other cells and the parent then cannot be paired; here
    the point is the mechanism: the swap is noticed, the cell is proven
    equivalent, compared by pin name, and reported."""
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    leaf = build_nand(NAND, name="ND2_LIB", lib=library, draw_pin_labels=True)
    top = library.new_cell("TOP")
    top.add(gdspy.CellReference(leaf, origin=(0, 0)))
    for pin, (x, y) in NAND.pin_positions.items():
        top.add(
            gdspy.Label(
                {"A": "IN0", "B": "IN1", "Y": "OUT"}.get(pin, pin),
                (x, y),
                layer=19,
                texttype=251,
            )
        )
    gds = tmp_path / "top.gds"
    library.write_gds(str(gds))

    def reference(near_output: str) -> Path:
        path = tmp_path / f"top_{near_output}.cdl"
        path.write_text(_nand_cdl("ND2_LIB", near_output=near_output)
                        + ".SUBCKT TOP IN0 IN1 OUT VDD VSS\nX0 IN0 IN1 OUT VDD VSS ND2_LIB\n.ENDS\n")  # fmt: skip
        return path

    results = {order: run_hierarchical_lvs(gds, reference(order), tmp_path / order, cell_name="TOP",
                                           library_cells="*_LIB") for order in "AB"}  # fmt: skip
    assert all(result.matched for result in results.values())
    excused = sorted(result.series_order_cells for result in results.values())
    assert excused == [
        (),
        ("ND2_LIB",),
    ]  # only the order that is not the layout's needed it

    # What triggered it: the first pass matched the cell only by pairing A with B.
    (needed,) = [result for result in results.values() if result.series_order_cells]
    first_pass = summarize_lvsdb(
        needed.lvs.report.parents[1] / "pass1" / "layout.lvsdb.gz"
    )
    cell = next(c for c in first_pass["circuits"] if c["layout"] == "ND2_LIB")
    assert (
        sorted(cell["renamed_pins"]) == [["A", "B"], ["B", "A"]]
        and cell["series_order_only"] is True
    )
    # ... and nothing is excused that is not declared a library cell.
    undeclared = run_hierarchical_lvs(gds, reference("A"), tmp_path / "undeclared", cell_name="TOP",
                                      library_cells="nothing_is_a_library_cell")  # fmt: skip
    assert undeclared.series_order_cells == ()
