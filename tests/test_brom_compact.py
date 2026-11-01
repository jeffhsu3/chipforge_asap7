"""CP-SAT routing bounds, actual geometry, and unchanged peripheral topology."""

import importlib.util
from copy import deepcopy

import pytest

from chipforge_asap7.brom import (
    BromSpec,
    compact_periphery,
    self_timed_macro_circuit,
    size_control_distribution,
)
from chipforge_asap7.brom.compact import assign_tracks


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


def test_compact_helper_is_deprecated_idempotent_noop():
    c = self_timed_macro_circuit(BromSpec(4, 4, registered_output=True), [10, 12, 0, 0])
    before = deepcopy(c)
    with pytest.warns(DeprecationWarning, match="dense CMOS"):
        assert compact_periphery(c) is c
    assert c.devices == before.devices and c.name == before.name
    size_control_distribution(c)
    with pytest.warns(DeprecationWarning):
        assert compact_periphery(c) is c


def test_cli_legacy_flag_reports_dense_geometry(tmp_path):
    from test_brom_cli import compile_words

    with pytest.warns(DeprecationWarning):
        _, report = compile_words(
            tmp_path, "--self-timed", "--registered-output", "--compact-periphery"
        )
    assert report["periphery_layout"] == "shared_diffusion_cmos"
    assert report["physical_sha256"]
    assert all(r["placement_certificates"] for r in report["artifacts"].values())
