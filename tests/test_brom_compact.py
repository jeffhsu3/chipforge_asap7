"""CP-SAT routing bounds, actual geometry, and unchanged peripheral topology."""

import importlib.util
from copy import deepcopy

import pytest
from test_brom_physical import CELLS
from test_brom_physical import test_drc_and_lvs as check_physical

from chipforge_asap7.brom import (
    BromSpec,
    cgen_circuit,
    compact_periphery,
    routed_control_rc,
    self_timed_macro_circuit,
    size_control_distribution,
)
from chipforge_asap7.brom.compact import assign_tracks
from chipforge_asap7.brom.layout import macro_periphery
from chipforge_asap7.brom.periphery import periphery_plan


@pytest.fixture(autouse=True)
def solver(external_tool):
    external_tool(
        importlib.util.find_spec("ortools") is not None, "OR-Tools is missing"
    )


def test_interval_bound_and_endpoint_clearance():
    intervals = ((0, 48), (48, 96), (24, 72), (20, 80))
    tracks, witness = assign_tracks(intervals)
    assert max(tracks) + 1 == len(witness) == 3
    assert tracks[0] == tracks[1]  # precisely legal end spacing can reuse a track
    for i, (lo, hi) in enumerate(intervals):
        for j, (a, b) in enumerate(intervals[:i]):
            if lo < b and a < hi:
                assert tracks[i] != tracks[j]


def test_adjacent_tracks_cannot_have_touching_real_tips():
    # Drawn spans 0..48 and 48..96, including a 40 nm right keepout.
    tracks, _ = assign_tracks(((0, 88), (48, 136)), end_space=40)
    assert abs(tracks[0] - tracks[1]) == 2


@pytest.mark.parametrize("name", CELLS)
def test_compact_units_and_macro_drc_lvs(
    name, tmp_path, asap7_drc, require_klayout, monkeypatch
):
    original = CELLS[name]

    def compact():
        # Some original matrix factories already size. Select the layout on a
        # copy of that circuit for DRC/LVS only; sizing validity is tested below.
        c = original()
        c.compact_periphery = True
        c.name += "_cp"
        return c

    monkeypatch.setitem(CELLS, name, compact)
    check_physical(name, tmp_path, asap7_drc, require_klayout)


def test_area_certificates_and_circuit_preservation():
    c = self_timed_macro_circuit(BromSpec(4, 4, registered_output=True), [10, 12, 0, 0])
    before = deepcopy(c)
    old = periphery_plan(macro_periphery(c))
    assert compact_periphery(c) is c
    assert c.devices == before.devices and c.placement_groups == before.placement_groups
    assert c.sample_distribution == before.sample_distribution
    assert c.array == before.array and c.replica == before.replica
    plan = periphery_plan(macro_periphery(c))
    assert plan.width * plan.height < 0.6 * old.width * old.height
    for row in plan.optimization["rows"]:
        assert row["status"] == "OPTIMAL"
        assert row["tracks"] == row["lower_bound"]
        assert row["clique_lower_bound"] == len(row["clique"]) <= row["tracks"]
        intervals = row["intervals_nm"]
        assert max(intervals[n][0] for n in row["clique"]) < min(
            intervals[n][1] for n in row["clique"]
        )
    assert periphery_plan(macro_periphery(c)) == plan
    assert c.name != before.name
    assert compact_periphery(c).name == c.name
    # The geometry consumer must use the same compact plan, not old RC lengths.
    a, b = routed_control_rc(before), routed_control_rc(c)
    assert b["SAMPLE"]["wire_cap_ff"] < a["SAMPLE"]["wire_cap_ff"]


def test_layout_must_be_selected_before_sizing():
    c = self_timed_macro_circuit(BromSpec(4, 2), [0, 1, 2, 3])
    size_control_distribution(c)
    with pytest.raises(ValueError, match="before control sizing"):
        compact_periphery(c)
    # Standalone output-like units should not reserve an empty 16-slot row.
    c = cgen_circuit()
    old = periphery_plan(c)
    new = periphery_plan(compact_periphery(c))
    assert new.width <= old.width and new.height < old.height


def test_cli_records_certificates_for_every_unit(tmp_path):
    from test_brom_cli import compile_words

    _, report = compile_words(
        tmp_path,
        "--self-timed",
        "--registered-output",
        "--sized-controls",
        "--compact-periphery",
    )
    assert report["periphery_layout"] == "cp_sat_channels"
    assert report["periphery_optimization"]["rows"]
    assert not report["timing_qualified"]
    for artifact in report["artifacts"].values():
        assert "_cp" in artifact["cell"]
        assert artifact["periphery_optimization"]["rows"]


def test_timing_bench_uses_compact_sizing_and_distinct_identity(tmp_path):
    import hashlib

    from chipforge_asap7.brom import ReplicaSpec
    from chipforge_asap7.brom.replica_timing import render_replica_bench
    from chipforge_asap7.brom.timing import TimingConditions

    spec, words = BromSpec(4, 2, registered_output=True), [0, 1, 2, 3]
    c = self_timed_macro_circuit(spec, words)
    old = hashlib.sha256(c.netlist().encode()).hexdigest()
    compact_periphery(c)
    assert hashlib.sha256(c.netlist().encode()).hexdigest() != old
    deck, signals = render_replica_bench(
        spec,
        words,
        ReplicaSpec(),
        TimingConditions(),
        tmp_path / "models.pm",
        tmp_path / "model.osdi",
        compact_layout=True,
        wire_rc_controls=True,
    )
    assert "rcdrv_" in deck and "SAMPLE_RX<2>" in signals
