"""Dense CMOS: independent logic, placement bounds and physical qualification."""

import json
from dataclasses import asdict
from itertools import pairwise

import gdspy
import pytest
from test_brom import settle

from chipforge_asap7.brom import (
    BromSpec,
    macro_circuit,
    self_timed_macro_circuit,
    size_control_distribution,
)
from chipforge_asap7.brom.circuit import Circuit
from chipforge_asap7.brom.dense_cell import cell_plan
from chipforge_asap7.brom.dense_macro import dense_macro_layout, verify_dense_routes


def test_mux4_fusion_exhaustive_including_non_one_hot():
    c = Circuit("mux", (), 2)
    c.mux4(["a", "b", "c", "d"], ["s0", "s1", "s2", "s3"], "q")
    assert len(c.devices) == 20
    for data in range(16):
        for sel in range(16):
            values = {n: (data >> i) & 1 for i, n in enumerate("abcd")}
            values.update({f"s{i}": (sel >> i) & 1 for i in range(4)})
            assert settle(c, values)["q"] == bool(data & sel)


def test_diffusion_bound_and_gate_pairing():
    c = Circuit("and", (), 2)
    c.and2("a", "b", "q")
    devices = tuple((d.spec.flavor, d.spec.fins, *d.nodes[:3]) for d in c.devices)
    p = cell_plan(devices, ("a", "b", "q"))
    assert p.placement_status == "OPTIMAL"
    assert max(x for _, x, _ in p.devices) == p.placement_bound
    assert p.width <= 432 and p.height == 432
    for i, x, flip in p.devices:
        a = devices[i]
        _al, ar = (a[2], a[4]) if flip else (a[4], a[2])
        for j, y, rev in p.devices:
            b = devices[j]
            bl, _br = (b[2], b[4]) if rev else (b[4], b[2])
            if a[0] == b[0] and y == x + 1:
                assert a[1] == b[1] and ar == bl
            if a[0] != b[0] and x == y:
                assert a[3] == b[3]


def test_supply_edges_share_diffusion():
    from chipforge_asap7.brom.dense_cell import edge_gap, wall_gap

    c = Circuit("nand", (), 2)
    c.nand("a", "b", "q")
    devices = tuple((d.spec.flavor, d.spec.fins, *d.nodes[:3]) for d in c.devices)
    padded = cell_plan(devices, ("a", "b", "q"))
    shared = cell_plan(devices, ("a", "b", "q"), share=True)
    # Sharing keeps the minimum gate-slot count.
    assert shared.placement_bound == padded.placement_bound
    assert max(x for _, x, _ in shared.devices) == shared.placement_bound
    # Both bands end on their supply at the left edge, right on the boundary.
    assert shared.edges[0] == (("n", 0, "VSS", 2), ("p", 0, "VDD", 2))
    assert shared.offset == 0 and shared.width < padded.width
    assert wall_gap(shared.edges[0]) == 54
    # Facing ready edges merge their islands: the spacer is 0 or a full break.
    assert edge_gap(shared.edges[0], shared.edges[0]) == (0, 108)
    assert edge_gap(shared.edges[0], shared.edges[0], share=False) == (108, 108)
    # Diffusion on a signal net stays 54 nm inside the cell.
    assert all(
        e >= 54
        for side in shared.edges
        for _, e, net, _ in side
        if net not in ("VDD", "VSS")
    )


# All 16 selector patterns are exercised whenever the shape has enough columns.
SHAPES = [
    (4, 1, 1),
    (4, 4, 1),
    (4, 16, 1),
    (16, 9, 1),
    (16, 4, 4),
    (64, 16, 4),
    pytest.param(1024, 16, 4, marks=pytest.mark.slow),
]


@pytest.mark.parametrize("depth,bits,ways", SHAPES)
def test_dense_shape_matrix(depth, bits, ways, tmp_path, asap7_drc, require_klayout):
    from chipforge_asap7.verification.lvs import run_lvs
    from chipforge_asap7.verification.lvs_report import summarize_lvsdb

    s = BromSpec(depth, bits, ways)
    words = [
        sum(
            (((bit * ways + addr % ways) % 16) >> (addr // ways % 4) & 1) << bit
            for bit in range(bits)
        )
        for addr in range(depth)
    ]
    c = macro_circuit(s, words)
    top, plan = dense_macro_layout(c)
    verify_dense_routes(top, plan["checkpoints"])
    lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    lib.add(top)
    assert not asap7_drc(lib, top)
    path = tmp_path / "macro.gds"
    ref = tmp_path / "macro.sp"
    lib.write_gds(str(path))
    ref.write_text(c.lvs_reference())
    result = run_lvs(
        path, ref, tmp_path / "lvs", cell_name=top.name, merge_fin_diffusion=True
    )
    assert result.matched, result.log.read_text()
    assert not any(
        r["renamed_pins"] for r in summarize_lvsdb(result.report)["circuits"]
    )
    assert c.array.plan.width == 108 and c.array.plan.height == 432
    (x0, y0), (x1, y1) = top.get_bounding_box()
    (tmp_path / "area.json").write_text(
        json.dumps(
            {
                "spec": asdict(s),
                "area_um2": (x1 - x0) * (y1 - y0) / 1e6,
                "physical_sha256": plan["physical_sha256"],
            }
        )
    )


def test_dense_rc_uses_layout_and_m6():
    from chipforge_asap7.brom import routed_control_rc

    c = self_timed_macro_circuit(BromSpec(4, 2, registered_output=True), [0, 1, 2, 3])
    _, plan = dense_macro_layout(c)
    rc = routed_control_rc(c)
    assert any("M6" in r["length_um_by_layer"] for r in rc.values())
    assert all(r["physical_sha256"] == plan["physical_sha256"] for r in rc.values())
    for n in c.sample_distribution["wire_nets"] + c.output_register["clock_nets"]:
        assert rc[n]["wire_cap_ff"] > 0 and rc[n]["max_driver_sink_path_ohm"] > 0


def test_resized_candidates_stay_inside_the_lane():
    c = macro_circuit(BromSpec(16, 16, ways=4), [65535] * 16)
    size_control_distribution(c)
    embedded = [r for r in c.distribution["nets"] if r.get("embedded_candidate")]
    assert {r["net"] for r in embedded} == {"C<0,15>", "LC<0,15>"}
    assert all(r["stages"][0]["parallel_inverters"] == 4 for r in embedded)
    top, plan = dense_macro_layout(c)
    verify_dense_routes(top, plan["checkpoints"])
    members = {m for u in plan["bottom_units"] for m in u["members"]}
    assert all(m not in members for r in embedded for m in r["stages"][0]["devices"])


@pytest.mark.parametrize("fins,keeper,stack", [(1, 3, 4), (3, 1, 4), (2, 3, 2)])
def test_mixed_fin_fused_readout(
    fins, keeper, stack, tmp_path, asap7_drc, require_klayout
):
    from chipforge_asap7.brom import build_periphery, output_circuit
    from chipforge_asap7.verification.lvs import run_lvs

    c = output_circuit(fins, keeper_fins=keeper, keeper_stack=stack)
    lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    top = build_periphery(c, lib=lib)
    assert not asap7_drc(lib, top)
    gds = tmp_path / "leaf.gds"
    ref = tmp_path / "leaf.sp"
    lib.write_gds(str(gds))
    ref.write_text(c.lvs_reference())
    result = run_lvs(
        gds, ref, tmp_path / "lvs", cell_name=top.name, merge_fin_diffusion=True
    )
    assert result.matched, result.log.read_text()


def test_registered_bank_rows_abut_and_share_power_rails():
    from collections import defaultdict

    from chipforge_asap7.brom.routing import metal_components

    c = self_timed_macro_circuit(BromSpec(4, 4, registered_output=True), [10, 12, 0, 0])
    devices = tuple(c.devices)
    top, plan = dense_macro_layout(c)
    assert tuple(c.devices) == devices
    verify_dense_routes(top, plan["checkpoints"])
    rows = defaultdict(list)
    for unit in plan["bottom_units"]:
        x, origin = unit["origin_nm"]
        bottom = origin - unit["height_nm"] if unit["mirror_y"] else origin
        rows[bottom].append((x, x + unit["width_nm"]))
        assert unit["mirror_y"] == bool(bottom // 432 % 2)
    assert sorted(rows) == [0, 432, 864]
    # Spacers, fillers and taps close every gap between logic cells, so each
    # row is one continuous frame.
    bank = next(
        r.ref_cell for r in top.references if r.ref_cell.name.endswith("_dense")
    )
    frames = defaultdict(list)
    for ref in bank.references:
        (outline,) = ref.ref_cell.get_polygons(by_spec=True)[(100, 0)]
        x, y = ref.origin
        bottom = y - 432 if ref.x_reflection else y
        frames[bottom].append((x + outline[:, 0].min(), x + outline[:, 0].max()))
    assert sorted(frames) == sorted(rows)
    for bottom, spans in frames.items():
        ordered = sorted(spans)
        assert ordered[0][0] == 0
        assert all(a[1] == b[0] for a, b in pairwise(ordered))
        assert set(rows[bottom]) <= set(ordered)
    # Mirroring makes the touching rails VDD/VDD and VSS/VSS. These points
    # must reach the public supplies through the new upper-metal routing.
    component = metal_components(top, recursive=True)
    for y, net in ((0, "VSS"), (432, "VDD"), (864, "VSS"), (1296, "VDD")):
        assert component("M1", 24, y) == component(*plan["ports"][net])
    lo, hi = top.get_bounding_box()
    assert (hi[0] - lo[0]) * (hi[1] - lo[1]) / 1e6 < 90
