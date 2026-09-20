"""The layout reducer: its readers and rules without tools, then a real reduction.

The end-to-end tests need KLayout, the public ASAP7 DRC deck and KPEX, and skip
without them (``ASAP7_REQUIRE_TOOLS=1`` makes that a failure); the one that
runs converged FasterCap solves takes minutes and also wants
``ASAP7_SLOW_TESTS=1``.  Everything above them runs anywhere.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import gdspy
import pytest
from conftest import find_asap7_drc_deck

from chipforge_asap7.devices import InverterSpec, build_inverter
from chipforge_asap7.layout.layers import LAYERS
from chipforge_asap7.verification.drc import drc_counts, parse_report
from chipforge_asap7.verification.lvs import render_inverter_lvs_schematic
from chipforge_asap7.verification.parasitics import (
    Parasitics,
    effective_resistance,
    pairwise_resistance,
)
from chipforge_asap7.verification.pex import find_fastercap
from chipforge_asap7.verification.reduce import (
    LayoutModel,
    ReductionConfig,
    _subckt_pins,
    parasitic_objection,
    reduce_layout,
)

HAVE_KPEX = importlib.util.find_spec("klayout_pex") is not None


# ── Parasitics ───────────────────────────────────────────────────────────────
def test_effective_resistance_solves_series_parallel_and_ignores_other_islands():
    edges = [
        ("P", "m", 10.0),
        ("m", "t1", 20.0),
        ("m", "t1", 20.0),  # two 20s in parallel: 10
        ("m", "t2", 5.0),
        ("x", "t3", 1.0),  # not connected to P
    ]
    found = effective_resistance(edges, "P", {"t1", "t2", "t3"})
    assert found == pytest.approx({"t1": 20.0, "t2": 15.0})


def test_effective_resistance_sees_a_redundant_path():
    ladder = [("P", "a", 10.0), ("a", "t", 10.0)]
    assert effective_resistance(ladder, "P", {"t"})["t"] == pytest.approx(20.0)
    strapped = [*ladder, ("P", "t", 20.0)]
    assert effective_resistance(strapped, "P", {"t"})["t"] == pytest.approx(10.0)


POST_LAYOUT = """\
* header
.SUBCKT INV A Y
+ VDD VSS
Rext_0 A A.$0.27 100
Rext_1 A.$0.27 A.N0.1 50
Rext_2 A.$0.27 A.P0.1 150
Rext_3 Y Y.N0.0 40
Cext_0 A Y 100a
Cext_1 A VSS 0.2f
Cext_2 Y VDD 300a
Cext_3 Y A 50a
NPEX_M_0 Y.N0.0 A.N0.1 VSS VSS nmos_rvt l=20n nfin=1
NPEX_M_1 Y.N0.0 A.P0.1 VDD VDD pmos_rvt l=20n nfin=1
.ENDS
"""


def test_post_layout_netlist_becomes_capacitance_and_pin_resistance(tmp_path: Path):
    netlist = tmp_path / "inv.post_layout.spice"
    netlist.write_text(POST_LAYOUT)
    found = Parasitics.from_post_layout(netlist)

    assert found.pins == ("A", "Y", "VDD", "VSS")
    assert found.signals() == ("A", "Y")
    assert found.between_fF("Y", "A") == pytest.approx(0.15)  # both orders accumulate
    assert found.capacitance_fF("A") == pytest.approx(0.35)
    assert found.capacitance_fF("Y") == pytest.approx(0.45)
    # A: 0.2 to ground + 0.15 x 2 (Miller); Y: 0.3 + 0.15 x 2.
    assert found.cost_fF() == pytest.approx(0.2 + 0.3 + 2 * 2 * 0.15)
    assert found.cost_fF(miller=1.0) == pytest.approx(0.35 + 0.45)
    assert found.resistance_ohm["A"] == pytest.approx((250.0, 200.0))
    assert found.resistance_ohm["Y"] == pytest.approx((40.0, 40.0))
    assert "VDD" not in found.resistance_ohm  # no extracted resistor reaches a terminal


def test_pairwise_resistance_is_per_island_and_symmetric():
    edges = [("d", "m", 10.0), ("m", "g1", 30.0), ("m", "g2", 5.0), ("x", "g3", 1.0)]
    found = pairwise_resistance(edges, {"d"}, {"g1", "g2", "g3"})
    assert sorted(found) == pytest.approx([15.0, 40.0])  # g3 is on another island
    assert pairwise_resistance(edges, {"g1"}, {"d"}) == pytest.approx([40.0])


SLICE = """\
.SUBCKT SLICE SEL B0 B1 WL0 WL1 VDD VSS
Rext_0 \\$7.P0.29 \\$7.$0.29 100
Rext_1 \\$7.$0.29 \\$7.P1.27 50
Rext_2 \\$4.P0.29 \\$4.P1.27 70
Rext_3 \\$I9.P0.29 \\$I9.P1.29 0.001
NPEX_M_0 \\$7.P0.29 B0.P0.27 \\$I9.P0.29 VSS nmos_rvt
NPEX_M_1 \\$I9.P1.29 SEL.P0.27 VSS VSS nmos_rvt
NPEX_M_2 WL0.P0.29 \\$7.P1.27 VSS VSS nmos_rvt
NPEX_M_3 \\$4.P0.29 B1.P0.27 VSS VSS nmos_rvt
NPEX_M_4 WL1.P0.29 \\$4.P1.27 VSS VSS nmos_rvt
.ENDS
"""


def test_a_net_without_a_pin_is_named_by_the_pins_around_it(tmp_path: Path):
    """KLayout numbers internal nets as it meets them; an edit can renumber them."""
    netlist = tmp_path / "slice.post_layout.spice"
    netlist.write_text(SLICE)
    found = Parasitics.from_post_layout(netlist).resistance_ohm
    # Driving diffusion to driven gate; the series stack's milliohm node is nothing to guard.
    assert found == {
        "int[B0,WL0]": pytest.approx((150.0, 150.0)),
        "int[B1,WL1]": pytest.approx((70.0, 70.0)),
    }
    renumbered = tmp_path / "renumbered.spice"
    renumbered.write_text(SLICE.replace("$7", "$70").replace("$4", "$7"))
    assert Parasitics.from_post_layout(renumbered).resistance_ohm == found


def _parasitics(a_ohm: float, a_y_fF: float) -> Parasitics:
    return Parasitics(
        pins=("A", "Y", "VDD", "VSS"),
        coupling_fF={("A", "Y"): a_y_fF, ("A", "VSS"): 0.2},
        resistance_ohm={"A": (a_ohm, a_ohm / 2)},
    )


def test_extraction_objects_to_resistance_and_to_cost_but_not_to_noise():
    cfg = ReductionConfig()
    base = _parasitics(100.0, 0.10)
    cost = base.cost_fF()
    assert parasitic_objection(cfg, base, cost, _parasitics(100.0, 0.05)) == ""
    assert (
        parasitic_objection(cfg, base, cost, _parasitics(104.0, 0.101)) == ""
    )  # within tolerance
    assert "resistance" in parasitic_objection(
        cfg, base, cost, _parasitics(150.0, 0.05)
    )
    assert "cost" in parasitic_objection(cfg, base, cost, _parasitics(100.0, 0.20))
    lost = Parasitics(pins=base.pins, coupling_fF=base.coupling_fF, resistance_ohm={})
    assert "no longer reaches" in parasitic_objection(cfg, base, cost, lost)


# ── Report readers ───────────────────────────────────────────────────────────
def test_drc_report_gives_rule_names_and_boxes_in_nm(tmp_path: Path):
    report = tmp_path / "cell.lyrdb"
    report.write_text(
        "<report-database><items>"
        "<item><category>'M1.A.1'</category><values>"
        "<value>polygon: (0.126,0.068;0.126,0.31;0.144,0.31;0.144,0.068)</value></values></item>"
        "<item><category>'V0.AUX.1-2V0.AUX.1-2 : V0 must be aligned'</category><values/></item>"
        "<item><category>'M1.A.1'</category><values><value>edge-pair: (0,0;0,0.018)|(0.03,0;0.03,0.018)"
        "</value></values></item>"
        "</items></report-database>"
    )
    found = parse_report(report)
    assert drc_counts(found) == {"M1.A.1": 2, "V0.AUX.1-2V0.AUX.1-2": 1}
    assert found[0].bbox == pytest.approx((126, 68, 144, 310))
    assert found[1].bbox is None
    assert found[2].bbox == pytest.approx((0, 0, 30, 18))


def test_pins_are_read_from_spice_headers_and_from_klayout_comments(tmp_path: Path):
    plain = tmp_path / "ref.sp"
    plain.write_text(
        ".SUBCKT OTHER X\n.ENDS\n.subckt inv A Y\n+ VDD\n+VSS\nM0 Y A VSS VSS n\n.ENDS\n"
    )
    assert _subckt_pins(plain, "INV") == {"A", "Y", "VDD", "VSS"}

    extracted = tmp_path / "extracted.spice"
    extracted.write_text(
        "* cell LEAF\n* pin Q\n.SUBCKT LEAF 1\n.ENDS\n"
        "* cell INV\n* pin VSS\n* pin Y\n* pin VDD\n.SUBCKT INV 1 2 3\n* net 1 VSS\n.ENDS\n"
    )
    assert _subckt_pins(extracted, "INV") == {"VSS", "Y", "VDD"}  # A lost its metal


# ── Layout model ─────────────────────────────────────────────────────────────
def _layer(name: str) -> dict:
    return {"layer": LAYERS[name]["layer"], "datatype": LAYERS[name]["datatype"]}


def test_model_pins_the_shape_under_a_label_through_a_rotated_reference(tmp_path: Path):
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    leaf = gdspy.Cell("LEAF", exclude_from_current=True)
    leaf.add(gdspy.Rectangle((0, 0), (200, 100), **_layer("BOUNDARY")))
    leaf.add(
        gdspy.Rectangle((20, 40), (120, 58), **_layer("M1"))
    )  # index 1: carries the pin
    leaf.add(gdspy.Rectangle((20, 70), (120, 88), **_layer("M1")))  # index 2: free
    leaf.add(
        gdspy.Rectangle((0, 10), (200, 28), **_layer("M1"))
    )  # index 3: spans the boundary
    leaf.add(
        gdspy.Polygon(
            [(150, 40), (180, 40), (180, 90), (165, 90), (165, 60), (150, 60)],
            **_layer("M1"),
        )
    )
    top = gdspy.Cell("TOP", exclude_from_current=True)
    top.add(gdspy.CellReference(leaf, origin=(1000, 0), rotation=90))
    # Leaf (100, 50) lands at top (1000 - 50, 100).
    top.add(gdspy.Label("A", (950, 100), layer=LAYERS["M1"]["layer"], texttype=251))
    library.add(top)
    library.add(leaf)
    gds = tmp_path / "top.gds"
    library.write_gds(str(gds))

    model = LayoutModel(gds, "TOP")
    assert list(model.pinned) == [("LEAF", 1)]
    assert model.pinned[("LEAF", 1)][0] == pytest.approx((100, 50))
    # Deletable: the free bar only.  Not the pinned bar, the rail, or the L-shape.
    assert model.candidates(("M1",), None, {}) == [("LEAF", 2)]
    assert model.candidates(("M1",), None, {}, pinned=True) == [
        ("LEAF", 1),
        ("LEAF", 2),
    ]
    assert model.candidates(("M1",), None, {("LEAF", 2): None}) == []

    edited = tmp_path / "edited.gds"
    model.write(edited, {("LEAF", 2): None, ("LEAF", 1): (60, 40, 120, 58)})
    again = LayoutModel(edited, "TOP")
    boxes = sorted(
        p[3]
        for p in again.polys["LEAF"]
        if p[3] and again.layer(("LEAF", 0)) and p[0] == 19
    )
    assert boxes == [(0, 10, 200, 28), (60, 40, 120, 58)]
    assert (
        sum(1 for p in again.polys["LEAF"] if p[3] is None) == 1
    )  # the L-shape went through verbatim
    assert again.refs["TOP"][0].rotation == 90
    assert list(again.pinned) == [("LEAF", 1)]


# ── End to end ───────────────────────────────────────────────────────────────
def _fastercap() -> Path | None:
    try:
        return find_fastercap()
    except FileNotFoundError:
        return None


def _inverter(tmp_path: Path) -> tuple[InverterSpec, Path, Path]:
    spec = InverterSpec(rows=((4, 6), (3, 3)), fingers=2, abut=False)
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    build_inverter(spec, name="INV", lib=library)
    gds, reference = tmp_path / "inv.gds", tmp_path / "inv.sp"
    library.write_gds(str(gds))
    reference.write_text(render_inverter_lvs_schematic(spec, cell_name="INV"))
    return spec, gds, reference


@pytest.mark.skipif(not HAVE_KPEX, reason="KPEX is not installed")
def test_inverter_gives_up_its_dead_strap_but_extraction_keeps_the_gate_tie(
    tmp_path: Path, external_tool, require_klayout
):
    """Two rows, two fingers, one pin on row 0.

    Row 1's M1 input strap and its V0 are a second landing nothing uses: they
    go.  Row 1's LIG pad is not needed for connectivity either -- the gates run
    through both rows -- and LVS with DRC lets it go too; but it ties the two
    fingers' gates together at the far end, and extraction refuses the jump in
    gate resistance.  Resistance floors only: no field solve, so this is quick.
    """
    external_tool(
        find_asap7_drc_deck() is not None, "the public ASAP7 DRC deck is not installed"
    )
    spec, gds, reference = _inverter(tmp_path)
    only = {"layers": ("V0", "M1", "LIG"), "shrink": False, "max_passes": 2}

    blind = reduce_layout(
        gds,
        reference,
        tmp_path / "blind",
        cell_name="INV",
        config=ReductionConfig(pex=False, **only),
    )
    assert {(e.kind, e.layer) for e in blind.edits} == {
        ("delete", "V0"),
        ("delete", "M1"),
        ("delete", "LIG"),
    }

    result = reduce_layout(
        gds,
        reference,
        tmp_path / "held",
        cell_name="INV",
        config=ReductionConfig(capacitance=False, **only),
    )
    assert sorted((e.kind, e.layer) for e in result.edits) == [
        ("delete", "M1"),
        ("delete", "V0"),
    ]
    strap = next(e for e in result.edits if e.layer == "M1")
    assert strap.before[1] > spec.stack.bands()[1].y0  # the strap of row 1
    assert not strap.covered
    assert [kept["layer"] for kept in result.kept_for_parasitics] == ["LIG"]
    assert "A worst pin-to-device resistance" in result.kept_for_parasitics[0]["reason"]
    assert result.before is not None and result.after is not None
    # What went was dead-end metal: no resistance moves.
    for net, held in result.before.resistance_ohm.items():
        assert result.after.resistance_ohm[net] == pytest.approx(held)
    assert result.gds.is_file() and result.report.is_file()


@pytest.mark.skipif(not HAVE_KPEX, reason="KPEX is not installed")
@pytest.mark.skipif(
    os.environ.get("ASAP7_SLOW_TESTS") != "1",
    reason="minutes of field solving; set ASAP7_SLOW_TESTS=1",
)
def test_full_reduction_lowers_the_switched_capacitance(
    tmp_path: Path, external_tool, require_klayout
):
    """Deletion and shrinking under all three floors, with converged field solves.

    What is left of the input's M1 is one landing at the row-0 contact, and the
    input-to-output coupling that the two straps carried goes with the rest.
    """
    external_tool(
        find_asap7_drc_deck() is not None, "the public ASAP7 DRC deck is not installed"
    )
    external_tool(_fastercap() is not None, "FasterCap is not built")
    _, gds, reference = _inverter(tmp_path)
    result = reduce_layout(
        gds,
        reference,
        tmp_path / "out",
        cell_name="INV",
        config=ReductionConfig(layers=("V0", "M1", "LIG"), shrink_layers=("M1",)),
    )
    assert sorted((e.kind, e.layer) for e in result.edits) == [
        ("delete", "M1"),
        ("delete", "V0"),
        ("shrink", "M1"),
    ]
    landing = next(e for e in result.edits if e.kind == "shrink")
    assert landing.after is not None and landing.after[3] - landing.after[1] == 36
    assert result.before is not None and result.after is not None
    assert result.after.cost_fF() < 0.95 * result.before.cost_fF()
    assert result.after.between_fF("A", "Y") < 0.95 * result.before.between_fF("A", "Y")
