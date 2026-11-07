"""Wide-word placement: local data paths, legal taps, and complete routing."""

from collections import Counter, defaultdict
from itertools import pairwise

import pytest

pytest.importorskip("gdspy")
pytest.importorskip("ortools")

from chipforge_asap7.brom import BromSpec, self_timed_macro_circuit
from chipforge_asap7.brom import dense_bank as db
from chipforge_asap7.brom import dense_macro as dm
from chipforge_asap7.brom.dense_place import TAP_PERIOD, TAP_SEGMENT, unit_gap
from chipforge_asap7.brom.dense_slices import sliced_rows
from chipforge_asap7.brom.dense_units import dense_units


def bank_inputs(monkeypatch, depth, bits, ways):
    captured = {}

    class Captured(Exception):
        pass

    def capture(circuit, **kwargs):
        captured.update(circuit=circuit, **kwargs)
        raise Captured

    circuit = self_timed_macro_circuit(
        BromSpec(depth, bits, ways, registered_output=True), [0] * depth
    )
    with monkeypatch.context() as patch:
        patch.setattr(dm, "build_dense_bank", capture)
        with pytest.raises(Captured):
            dm.dense_macro_layout(circuit)
    return captured


@pytest.mark.parametrize("depth,bits,ways", [(16, 148, 1), (64, 148, 4), (16, 9, 4)])
def test_slices_keep_data_local_and_every_cell_legal(monkeypatch, depth, bits, ways):
    inputs = bank_inputs(monkeypatch, depth, bits, ways)
    units = dense_units(inputs["circuit"])
    width = inputs["width"]
    usable = width - 324
    spans = inputs["header_spans"](width)
    rows = -(-sum(u.plan.width for u in units) // usable) + 4
    packed = sliced_rows(units, spans, usable, rows, sinks=inputs["bottom"])
    assert packed is not None
    assert Counter(id(u) for row in packed for u, _ in row) == Counter(map(id, units))
    locations = {}
    points = defaultdict(list)
    for r, row in enumerate(packed):
        for unit, x in row:
            assert x >= 0 and x % 54 == 0
            assert x + unit.plan.width <= usable
            assert x % TAP_PERIOD + unit.plan.width <= TAP_SEGMENT
            locations[unit.name] = (r, x)
            mapping = dict(unit.nets)
            for port, dx in unit.plan.trunks:
                if port in unit.ports:
                    points[mapping[port]].append((r, x + dx))
        for (left, x), (right, y) in pairwise(row):
            gap = y - x - left.plan.width
            join, free = unit_gap(left, right)
            assert gap == join or gap >= free
    for bit in range(bits):
        row = locations[f"output_latch_{bit}"][0]
        assert locations[f"register_master_{bit}"][0] == row
        assert locations[f"register_slave_{bit}"][0] == row
        for stage in ("input", "output", "complement", "off", "quiet", "closed"):
            assert locations[f"sample_receiver_{stage}_{bit}"][0] == row
        for net in (f"SAMPLED<{bit}>", f"REG_DATA<{bit}>"):
            assert {r for r, _ in points[net]} == {row}
            xs = [x for _, x in points[net]]
            assert max(xs) - min(xs) < 1500
    if ways == 4:
        for column in range(bits * ways):
            ((_, x),) = points[f"BL<{column}>"]
            # Tap gaps can displace a readout, but never by a bank-wide run.
            assert abs(x - spans[f"BL<{column}>"][0]) <= 432


@pytest.mark.slow
@pytest.mark.parametrize("depth,ways", [(16, 1), (64, 4), (1024, 4)])
def test_wide_registered_macro_uses_slices_and_routes(monkeypatch, depth, ways):
    called = []
    original = db.sliced_rows

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        called.append(result is not None)
        return result

    monkeypatch.setattr(db, "sliced_rows", record)
    circuit = self_timed_macro_circuit(
        BromSpec(depth, 148, ways, registered_output=True), [0] * depth
    )
    top, plan = dm.dense_macro_layout(circuit)
    assert called == ([True, True] if depth == 1024 else [True])
    if depth == 1024:
        assert plan["joint_optimization"]
        assert all(
            record["capacity_sum"] == record["capacity_floor"]
            and record["global_optimal"] is False
            for record in plan["joint_optimization"]
        )
    dm.verify_dense_routes(top, plan["checkpoints"])
    # Local placement keeps the bank inside the original array footprint.
    assert top.get_bounding_box()[1][0] < 100_000
    assert {f"Q[{i}]" for i in range(148)} <= plan["ports"].keys()
