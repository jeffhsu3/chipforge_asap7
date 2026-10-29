"""Independent switch-level and physical checks of the BROM port."""

from collections import Counter

import gdspy
import pytest

from chipforge_asap7.brom import (
    BromSpec,
    build_macro,
    build_periphery,
    cgen_circuit,
    decoder_circuit,
    macro_circuit,
    output_circuit,
    selector_map,
)


def settle(circuit, inputs, previous=None):
    """Ideal transistor switches, retaining charge on isolated nodes.

    Only supply-connected components drive a new value. Input inverters settle
    first (ideal complementary controls, with no clock skew model).
    This checks logic/holding, not analog timing or leakage.
    """
    fixed = {"VDD": 1, "VSS": 0, **inputs}
    values = {**(previous or {}), **fixed}
    for n in circuit.devices:
        d, g, s, _ = n.nodes
        if (
            n.spec.flavor == "n"
            and s == "VSS"
            and g in fixed
            and any(
                p.spec.flavor == "p" and p.nodes[:3] == (d, g, "VDD")
                for p in circuit.devices
            )
        ):
            values[d] = 1 - fixed[g]
    nets = {n for d in circuit.devices for n in d.nodes}
    for _ in range(200):
        parent = {n: n for n in nets}

        def root(n, parent=parent):
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        for device in circuit.devices:
            d, g, s, _ = device.nodes
            if values.get(g) == (0 if device.spec.flavor == "p" else 1):
                parent[root(d)] = root(s)
        driven = {}
        for net, value in fixed.items():
            driven.setdefault(root(net), set()).add(value)
        new = dict(values)
        for net in nets:
            levels = driven.get(root(net), set())
            if len(levels) == 1:
                new[net] = next(iter(levels))
        new.update(fixed)
        if new == values:
            assert all(len(v) == 1 for v in driven.values()), "settled contention"
            return values
        values = new
    raise AssertionError("circuit did not settle")


@pytest.mark.parametrize("ways,depth", [(1, 4), (1, 12), (1, 32), (4, 16), (4, 48)])
def test_encoding(ways, depth):
    spec = BromSpec(depth, 7, ways)
    words = [(i * 43 + 9) % 128 for i in range(depth)]
    mapping = selector_map(spec, words)
    for address, expected in enumerate(words):
        block, offset, way = spec.decode(address)
        result = sum(
            ((mapping[block][b * ways + way] >> offset) & 1) << b for b in range(7)
        )
        assert result == expected


def test_cgen_all_candidates_and_disabled_state():
    c = cgen_circuit()
    assert len(c.devices) == 96
    for sel in (0, 1):
        for offset in range(4):
            inputs = {"SEL": sel, **{f"G<{k}>": k >> offset & 1 for k in range(16)}}
            values = settle(c, inputs)
            assert [values[f"C<{k}>"] for k in range(16)] == [
                sel & (k >> offset & 1) for k in range(16)
            ]


@pytest.mark.parametrize("spec", [BromSpec(4, 1), BromSpec(12, 1), BromSpec(48, 1, 4)])
def test_decoder_transistors(spec):
    c = decoder_circuit(spec)
    for address in range(1 << spec.address_bits):
        for prechn, evaluate in ((0, 0), (0, 1), (1, 0), (1, 1)):
            v = settle(
                c,
                {
                    "PRECHN": prechn,
                    "EVAL": evaluate,
                    **{f"A<{i}>": address >> i & 1 for i in range(spec.address_bits)},
                },
            )
            row, way = divmod(address, spec.ways)
            block, offset = divmod(row, 4)
            assert [v[f"SEL<{b}>"] for b in range(spec.blocks)] == [
                int(b == block and prechn and evaluate) for b in range(spec.blocks)
            ]
            assert [v[f"G<{k}>"] for k in range(16)] == [
                k >> offset & 1 for k in range(16)
            ]
            assert [v[f"CSEL<{w}>"] for w in range(spec.ways)] == [
                int(w == way) for w in range(spec.ways)
            ]


@pytest.mark.parametrize("ways", [1, 4])
def test_macro_reads_and_holds_every_four_bit_pattern(ways):
    # Each output stores one of the 16 candidate patterns; every way differs.
    spec = BromSpec(8 * ways, 16, ways)
    words = [
        sum((((k + (a % ways)) % 16) >> ((a // ways) % 4) & 1) << k for k in range(16))
        for a in range(spec.depth)
    ]
    c = macro_circuit(spec, words)
    state = None
    held = None
    for address, expected in enumerate(words):
        pins = {f"A<{i}>": address >> i & 1 for i in range(spec.address_bits)}
        state = settle(c, {**pins, "PRECHN": 0, "EVAL": 0, "SAMPLE": 0}, state)
        if held is not None:
            assert sum(state[f"Q<{i}>"] << i for i in range(spec.bits)) == held
        assert all(state[f"BL<{i}>"] == 1 for i in range(spec.columns))
        state = settle(c, {**pins, "PRECHN": 1, "EVAL": 1, "SAMPLE": 0}, state)
        # External address changes must not disturb the closed address latches.
        noise = {key: 1 - value for key, value in pins.items()}
        state = settle(c, {**noise, "PRECHN": 1, "EVAL": 1, "SAMPLE": 1}, state)
        held = sum(state[f"Q<{i}>"] << i for i in range(spec.bits))
        assert held == expected
        state = settle(c, {**pins, "PRECHN": 1, "EVAL": 1, "SAMPLE": 0}, state)
        state = settle(c, {**pins, "PRECHN": 1, "EVAL": 0, "SAMPLE": 0}, state)


@pytest.mark.parametrize(
    "kwargs", [{"ways": 2}, {"depth": 6}, {"bits": 0}, {"depth": True}, {"fins": 0}]
)
def test_invalid_specs(kwargs):
    with pytest.raises((ValueError, TypeError)):
        BromSpec(**kwargs)


def test_programming_changes_only_v2_v3():
    from chipforge_asap7.layout.layers import LAYERS

    spec = BromSpec(4, 1)
    cells = [build_macro(macro_circuit(spec, words)) for words in ([0] * 4, [1] * 4)]
    polygons = [c.get_polygons(by_spec=True) for c in cells]
    cuts = {(LAYERS[m]["layer"], 0) for m in ("V2", "V3")}
    for layer in polygons[0].keys() | polygons[1].keys():
        if layer not in cuts:
            assert Counter(p.tobytes() for p in polygons[0][layer]) == Counter(
                p.tobytes() for p in polygons[1][layer]
            ), layer
    assert any(
        Counter(p.tobytes() for p in polygons[0][layer])
        != Counter(p.tobytes() for p in polygons[1][layer])
        for layer in cuts
    )


@pytest.mark.parametrize(
    "make",
    [
        output_circuit,
        cgen_circuit,
        lambda: output_circuit(1, "lvt"),
        lambda: output_circuit(4, "slvt"),
        lambda: output_circuit(3, "sram"),
        lambda: macro_circuit(BromSpec(4, 2), [0, 1, 2, 3]),
        lambda: macro_circuit(BromSpec(16, 3, 4), [i % 8 for i in range(16)]),
    ],
)
def test_periphery_drc(make, asap7_drc):
    lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    circuit = make()
    top = (build_macro if circuit.array is not None else build_periphery)(
        circuit, lib=lib
    )
    assert asap7_drc(lib, top, tag=top.name) == []


def test_output_layout_matches_transistors(tmp_path):
    gdstk = pytest.importorskip("gdstk")
    from chipforge_asap7.verification.connectivity import extract_connectivity

    lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    circuit = output_circuit()
    top = build_periphery(circuit, lib=lib)
    path = tmp_path / "output.gds"
    lib.write_gds(str(path))
    flat = next(
        c for c in gdstk.read_gds(path, unit=1e-6).cells if c.name == top.name
    ).flatten()
    devices, labels = extract_connectivity(flat, expected_channels=3)
    assert all(len(ids) == 1 for ids in labels.values())
    nets = {name: next(iter(ids)) for name, ids in labels.items()}
    assert len(set(nets.values())) == len(nets)
    expected = Counter(
        (
            d.spec.flavor == "p",
            d.spec.fins,
            nets[d.nodes[1]],
            frozenset((nets[d.nodes[0]], nets[d.nodes[2]])),
        )
        for d in circuit.devices
    )
    actual = Counter(
        (d["pmos"], d["nfin"], d["gate"], frozenset(d["terminals"])) for d in devices
    )
    assert actual == expected


@pytest.mark.parametrize(
    "make",
    [
        output_circuit,
        cgen_circuit,
        lambda: decoder_circuit(BromSpec(12, 1)),
        lambda: macro_circuit(BromSpec(4, 2), [0, 1, 2, 3]),
        lambda: macro_circuit(BromSpec(16, 3, 4), [i % 8 for i in range(16)]),
    ],
)
def test_periphery_and_macro_lvs(make, tmp_path, require_klayout):
    from chipforge_asap7.verification.lvs import run_lvs
    from chipforge_asap7.verification.lvs_report import summarize_lvsdb

    lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    c = make()
    top = (build_macro if c.array is not None else build_periphery)(c, lib=lib)
    gds, reference = tmp_path / "dut.gds", tmp_path / "dut.spice"
    lib.write_gds(str(gds))
    reference.write_text(c.lvs_reference())
    result = run_lvs(gds, reference, tmp_path / "lvs", cell_name=top.name)
    assert result.matched, result.log.read_text()
    summary = summarize_lvsdb(result.report)
    assert not any(c["renamed_pins"] for c in summary["circuits"])


def test_wrong_programming_fails_lvs(tmp_path, require_klayout):
    from chipforge_asap7.verification.lvs import run_lvs

    spec = BromSpec(4, 1)
    correct = macro_circuit(spec, [0, 0, 0, 0])
    wrong = macro_circuit(spec, [0, 1, 0, 0])
    wrong.name = correct.name
    lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    build_macro(correct, lib=lib)
    gds, reference = tmp_path / "dut.gds", tmp_path / "wrong.spice"
    lib.write_gds(str(gds))
    reference.write_text(wrong.lvs_reference())
    assert not run_lvs(gds, reference, tmp_path / "lvs", cell_name=correct.name).matched


def test_cli_emits_artifacts(tmp_path):
    import json

    from chipforge_asap7.brom.__main__ import main

    words = tmp_path / "words.json"
    words.write_text("[0, 1, 2, 3]")
    out = tmp_path / "compiled"
    main([str(words), "--bits", "2", "--out", str(out)])
    report = json.loads((out / "manifest.json").read_text())
    assert report["timing_qualified"] is False
    assert report["layout"] == "compact_selector_array"
    assert report["periphery_layout"] == "folded_local_rows"
    assert report["periphery_plan"]["row_count"] > 1
    assert report["array_core_nm"] == [216, 432]
    assert report["isolation_devices"] == 2
    assert report["selector_map"] == [[10, 12]]
    for role, artifact in report["artifacts"].items():
        assert artifact["drc"] == artifact["lvs"] == "not_run"
        assert (out / artifact["gds"]).is_file()
        assert (out / artifact["cdl"]).is_file()
        assert (out / f"{role}.lvs.spice").is_file()


@pytest.mark.parametrize(
    "words", [[0] * 3, [0, 0, 0, 4], [0, 0, 0, 1.0], [0, 0, 0, True]]
)
def test_bad_contents_rejected(words):
    with pytest.raises(ValueError):
        macro_circuit(BromSpec(4, 2), words)


def test_macro_cannot_use_flat_peripheral_builder():
    c = macro_circuit(BromSpec(4, 1), [0, 1, 0, 1])
    with pytest.raises(ValueError, match="compact selector array"):
        build_periphery(c)
    with pytest.raises(ValueError, match="macro_circuit"):
        build_macro(output_circuit())


def test_deprecated_builder_routes_to_compact_macro():
    from chipforge_asap7.brom import build_circuit

    c = macro_circuit(BromSpec(4, 1), [0, 1, 0, 1])
    with pytest.warns(DeprecationWarning, match="build_macro"):
        top = build_circuit(c)
    assert (
        sum(r.ref_cell.name.startswith("brom_selector_") for r in top.references) == 1
    )
    assert len(top.references) == 2


@pytest.mark.parametrize("fins", [1, 2, 3])
def test_macro_contains_compact_array_and_no_discrete_array_devices(fins):
    from chipforge_asap7.brom.layout import macro_periphery, macro_plan

    spec = BromSpec(12, 3, fins=fins)
    c = macro_circuit(spec, [i % 8 for i in range(12)])
    assert len(c.array.device_names) == 2 * spec.blocks * spec.columns
    bank = macro_periphery(c)
    assert {d.name for d in bank.devices}.isdisjoint(c.array.device_names)
    assert len(bank.devices) + len(c.array.device_names) == len(c.devices)
    floorplan = macro_plan(c)
    assert (floorplan.array_width, floorplan.array_height) == (324, 1296)
    assert len(floorplan.ports) == 16 * spec.blocks + spec.columns + 1
    for b in range(spec.blocks):
        for k, (metal, y) in enumerate(c.array.plan.tracks):
            port = floorplan.ports[16 * b + k]
            assert (port.net, port.layer, port.y) == (f"C<{b},{k}>", metal, b * 432 + y)
    top = build_macro(c)
    array = next(
        r.ref_cell
        for r in top.references
        if r.ref_cell.name.startswith("brom_selector_")
    )
    assert (
        not array.references
    )  # compact FEOL is drawn directly, with no isolated-device tiles
    assert not array.labels  # internal C/BL ports must not become macro pins
    assert {p.text for p in top.labels} == set(c.pins)


@pytest.mark.parametrize(
    "spec",
    [
        BromSpec(8, 3, fins=1, vt="lvt"),
        BromSpec(12, 1, fins=3),
        BromSpec(32, 2, ways=4),
    ],
)
def test_integrated_shapes_drc_lvs(spec, tmp_path, asap7_drc, require_klayout):
    from chipforge_asap7.verification.lvs import run_lvs
    from chipforge_asap7.verification.lvs_report import summarize_lvsdb

    c = macro_circuit(spec, [(i * 3 + 1) % (1 << spec.bits) for i in range(spec.depth)])
    lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    top = build_macro(c, lib=lib)
    assert asap7_drc(lib, top, tag=top.name) == []
    gds, ref = tmp_path / "macro.gds", tmp_path / "macro.spice"
    lib.write_gds(str(gds))
    ref.write_text(c.lvs_reference())
    result = run_lvs(gds, ref, tmp_path / "lvs", cell_name=top.name)
    assert result.matched, result.log.read_text()
    assert not any(
        p["renamed_pins"] for p in summarize_lvsdb(result.report)["circuits"]
    )
    pins = {
        line.removeprefix("* pin ").strip()
        for line in result.extracted_netlist.read_text().splitlines()
        if line.startswith("* pin ")
    }
    assert pins == set(c.pins)


@pytest.mark.parametrize("port_name", ["C<0,2>", "C<0,8>", "BL<0>", "VSS"])
def test_disconnected_array_interface_is_rejected(port_name, tmp_path, require_klayout):
    from chipforge_asap7.brom.layout import macro_plan, verify_macro_routes
    from chipforge_asap7.layout.layers import LAYERS
    from chipforge_asap7.verification.lvs import run_lvs

    words = [sum((k >> offset & 1) << k for k in range(16)) for offset in range(4)]
    c = macro_circuit(BromSpec(4, 16), words)
    lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    top = build_macro(c, lib=lib)
    plan = macro_plan(c)
    i = next(i for i, p in enumerate(plan.ports) if p.net == port_name)
    x, y = plan.trunk_x[i], plan.route_y[i]
    removed = []

    def remove(points, layer, datatype):
        match = (
            layer == LAYERS["V4"]["layer"]
            and datatype == 0
            and points[:, 0].mean() == x
            and points[:, 1].mean() == y
        )
        if match:
            removed.append(True)
        return match

    top.remove_polygons(remove)
    assert len(removed) == 1
    with pytest.raises(ValueError, match="disconnected macro route"):
        verify_macro_routes(c, top)
    gds, ref = tmp_path / "bad.gds", tmp_path / "expected.spice"
    lib.write_gds(str(gds))
    ref.write_text(c.lvs_reference())
    result = run_lvs(gds, ref, tmp_path / "lvs", cell_name=top.name)
    # Substrate continuity can conceal a missing VSS metal bridge in raw LVS.
    if port_name != "VSS":
        assert not result.matched


@pytest.mark.parametrize("kwargs", [{"fins": 4}, {"depth": 280}])
def test_unsupported_compact_shape_has_no_legacy_fallback(kwargs):
    with pytest.raises(ValueError):
        BromSpec(**kwargs)


def test_short_between_array_routes_is_rejected():
    from chipforge_asap7.brom.layout import macro_plan, verify_macro_routes
    from chipforge_asap7.layout.layers import box

    c = macro_circuit(BromSpec(4, 1), [0, 1, 0, 1])
    top = build_macro(c)
    plan = macro_plan(c)
    # Bridge two independent M5 trunks in the open routing channel.
    box(top, "M5", plan.trunk_x[1] - 12, 500, plan.trunk_x[0] + 12, 524)
    with pytest.raises(ValueError, match="shorted macro routes"):
        verify_macro_routes(c, top)


@pytest.mark.parametrize("bits", [3, 128])
def test_folded_periphery_preserves_groups_and_selector_lattice(bits):
    from chipforge_asap7.brom.layout import macro_periphery
    from chipforge_asap7.brom.periphery import periphery_plan

    c = macro_circuit(BromSpec(12, bits), [i % 8 for i in range(12)])
    bank = macro_periphery(c)
    plan = periphery_plan(bank)
    placed = [d.name for row in plan.rows for d in row.devices]
    assert Counter(placed) == Counter(d.name for d in bank.devices)
    assert set(placed).isdisjoint(c.array.device_names)
    assert plan.device_pitch % c.array.plan.width == 0
    assert all(row.y % c.array.plan.height == 0 for row in plan.rows)
    for _, members in bank.placement_groups:
        assert any(set(members) <= {d.name for d in row.devices} for row in plan.rows)
    # No single-row local net should get a global spine.
    for net in plan.spine_x:
        assert sum(net in row.buses for row in plan.rows) > 1
    # Array ports escape directly at the row that owns their devices.
    for pin, y in plan.pins.items():
        assert any(row.buses.get(pin) == y for row in plan.rows)
    if bits == 128:
        assert plan.row_capacity > 8


@pytest.mark.parametrize("net", ["SEL", "VSS", "VDD"])
def test_periphery_metal_audit_rejects_missing_shared_connection(net):
    from chipforge_asap7.brom.periphery import periphery_plan, verify_periphery_routes
    from chipforge_asap7.layout.layers import LAYERS

    c = cgen_circuit()
    top = build_periphery(c)
    plan = periphery_plan(c)
    x = plan.spine_x[net]
    y = next(row.buses[net] for row in plan.rows if net in row.buses)
    removed = []

    def remove(points, layer, datatype):
        match = (
            layer == LAYERS["V4"]["layer"]
            and datatype == 0
            and points[:, 0].mean() == x
            and points[:, 1].mean() == y
        )
        if match:
            removed.append(True)
        return match

    top.remove_polygons(remove)
    assert len(removed) == 1
    with pytest.raises(ValueError, match="disconnected peripheral route"):
        verify_periphery_routes(c, top)


def test_periphery_metal_audit_rejects_shared_net_short():
    from chipforge_asap7.brom.periphery import periphery_plan, verify_periphery_routes
    from chipforge_asap7.layout.layers import box

    c = cgen_circuit()
    top = build_periphery(c)
    plan = periphery_plan(c)
    lo, hi = sorted((plan.spine_x["VDD"], plan.spine_x["VSS"]))
    y = max(plan.rows[0].buses.values()) + 48
    box(top, "M5", lo - 12, y - 12, hi + 12, y + 12)
    with pytest.raises(ValueError, match="shorted peripheral routes"):
        verify_periphery_routes(c, top)


def test_folded_mux4_macro_area_regression():
    c = macro_circuit(BromSpec(16, 3, ways=4), [i % 8 for i in range(16)])
    bounds = build_macro(c).get_bounding_box()
    area_um2 = (bounds[1, 0] - bounds[0, 0]) * (bounds[1, 1] - bounds[0, 1]) / 1e6
    # The prior single-strip implementation occupied 8826.31 square microns.
    assert area_um2 < 3000
