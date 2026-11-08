"""The NOR ROM's sense lane: its plan, DRC beside the half arrays, and LVS.

DRC and LVS skip when gdscheck or KLayout is missing; ``ASAP7_REQUIRE_TOOLS=1``
makes that a failure.
"""

from __future__ import annotations

import random
from itertools import pairwise
from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import RomDecoderSpec, build_rom
from chipforge_asap7.devices.rom_bitcell import (
    RomArraySpec,
    RomBitcellSpec,
    build_rom_array,
)
from chipforge_asap7.devices.rom_lane import RomLaneSpec, build_rom_lane
from chipforge_asap7.devices.rom_pins import _crossing, _m3_at, _rects
from chipforge_asap7.verification.lvs import run_lvs


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def _half(words: int, bits: int, fins: int = 2) -> RomArraySpec:
    return RomArraySpec(words // 2, bits, RomBitcellSpec(fins=fins))


@pytest.mark.parametrize("sample", [False, True], ids=["leaves", "sampled"])
@pytest.mark.parametrize(
    ("words", "bits", "fins"),
    [(16, 8, 2), (64, 32, 2), (128, 33, 2), (64, 40, 3), (1024, 148, 2)],
)
def test_plan_gives_every_bit_its_own_track_and_lift(words, bits, fins, sample):
    lane = RomLaneSpec(_half(words, bits, fins), sample=sample)
    assert [p.bit for p in lane.plan] == list(range(bits))
    assert len({p.track for p in lane.plan}) == bits
    assert len({(p.row, p.slot) for p in lane.plan}) == bits
    for p in lane.plan:
        assert p.track in lane.tracks(p.row)
        assert p.slot < lane.row_leaves(p.row)
    columns: dict[int, list[tuple[int, int]]] = {}
    for p in lane.plan:
        columns.setdefault(p.lift, []).append(
            lane.lift_span(lane.bitline_y(p.bit), p.track)
        )
    for spans in columns.values():
        spans.sort()
        for (_, hi), (lo, _) in pairwise(spans):
            assert lo - hi >= 32
    # The rows' ends leave the lifts a gate pitch on either side.
    assert lane.lift_x(0, lane.lifts - 1) < lane.x0 and lane.x1 < lane.lift_x(
        1, lane.lifts - 1
    )
    # The top rail clears the array's VSS bar, or is a VSS rail on it.
    clear = lane.half.height - lane.height
    assert clear >= 36 or (clear == 0 and lane.rows % 2 == 0)


def test_the_top_rail_keeps_off_the_vss_bar():
    """A 4-bit half is exactly one 513 nm sample row high: its VDD rail would
    land on the bar, so the row does not fit."""
    half = RomArraySpec(8, 4, RomBitcellSpec(fins=2), strap_every=4)
    assert half.height == RomLaneSpec(_half(64, 32), sample=True).row_height
    with pytest.raises(ValueError, match="no room"):
        RomLaneSpec(half, sample=True)


def test_a_one_fin_array_is_too_dense_for_the_tracks():
    with pytest.raises(ValueError, match="denser than the lane's tracks"):
        RomLaneSpec(_half(64, 40, fins=1)).plan  # noqa: B018


@pytest.mark.parametrize(
    ("words", "bits", "fins", "sample", "replica"),
    [
        (16, 8, 2, False, False),
        (64, 32, 2, False, False),
        (128, 33, 2, False, False),
        (64, 40, 3, False, False),
        (64, 32, 2, True, False),
        (128, 33, 2, True, False),
        (64, 33, 2, True, True),
        (128, 34, 2, True, True),
        (64, 32, 2, "register", False),
        (128, 34, 2, "register", True),
    ],
)
def test_lane_between_programmed_halves_is_clean(
    words, bits, fins, sample, replica, asap7_drc
):
    half = _half(words, bits, fins)
    lane = RomLaneSpec(
        half, sample=bool(sample), register=sample == "register", replica=replica
    )
    library = _library()
    rng = random.Random(words * bits)
    sites = {(w, b) for w in range(words) for b in range(bits) if rng.random() < 0.5}
    n = half.words
    top = library.new_cell("lane_top")
    for side, (x, label, edge) in enumerate(
        ((0, "BLL", "right"), (lane.x_right, "BLR", "left"))
    ):
        array = build_rom_array(
            half,
            [(w - side * n, b) for w, b in sites if (w >= n) == side],
            name=f"half_{side}",
            lib=library,
            bitline=label,
            bitline_side=edge,  # type: ignore[arg-type]
            first_word=side * n,
        )
        top.add(gdspy.CellReference(array, (x, 0)))
    top.add(gdspy.CellReference(build_rom_lane(lane, lib=library)))
    assert asap7_drc(library, top, tag="lane") == []


@pytest.mark.parametrize(
    "sample", [False, True, "register"], ids=["leaves", "sampled", "registered"]
)
def test_lane_matches_its_unit_fin_reference(sample, tmp_path: Path, require_klayout):
    lane = RomLaneSpec(
        _half(64, 32), sample=bool(sample), register=sample == "register"
    )
    library = _library()
    cell = build_rom_lane(lane, lib=library)
    gds = tmp_path / "lane.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(lane.lvs_reference())
    result = run_lvs(gds, reference, tmp_path / "lvs", cell_name=cell.name)
    assert result.matched
    fins = sum(device[5] for device in lane.devices())
    per_bit = {False: 12, True: 14 * 4, "register": 22 * 4}[sample]
    assert fins == 32 * per_bit + (2 * 4 * lane.rows if sample else 0)
    assert result.extracted_netlist.read_text().count("M$") == fins


def test_replica_lane_matches_its_unit_fin_reference(tmp_path: Path, require_klayout):
    """The replica's leaf output is READY and it has no Q pin."""
    lane = RomLaneSpec(_half(64, 33), sample=True, replica=True)
    assert "READY" in lane.pins and "Q<32>" not in lane.pins
    assert "D<32>" not in lane.pins
    library = _library()
    cell = build_rom_lane(lane, lib=library)
    gds = tmp_path / "lane.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(lane.lvs_reference())
    assert run_lvs(gds, reference, tmp_path / "lvs", cell_name=cell.name).matched


def test_ready_keeps_its_own_track_off_the_replicas():
    lane = RomLaneSpec(_half(64, 33), sample=True, replica=True)
    p = lane.plan[lane.replica_bit]
    assert lane.ready_track in lane.tracks(p.row)
    assert lane.ready_track not in {q.track for q in lane.plan}
    assert abs(lane.ready_track - p.track) >= 96
    assert lane.ready_x % 48 == 12


def test_lane_with_two_rows_sampleb_joined_fails(tmp_path: Path, require_klayout):
    """Each row's SAMPLEB is its own: a reference that joins two rows' must not match."""
    lane = RomLaneSpec(_half(64, 32), sample=True)
    library = _library()
    cell = build_rom_lane(lane, lib=library)
    gds = tmp_path / "lane.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    text = lane.lvs_reference()
    assert "SAMPLEB_R1" in text
    reference.write_text(text.replace("SAMPLEB_R1", "SAMPLEB_R0"))
    assert not run_lvs(gds, reference, tmp_path / "lvs", cell_name=cell.name).matched


def test_lane_with_a_bit_crossed_fails(tmp_path: Path, require_klayout):
    """Two bits' BLL nets swapped in the reference must not match."""
    lane = RomLaneSpec(_half(64, 32))
    library = _library()
    cell = build_rom_lane(lane, lib=library)
    gds = tmp_path / "lane.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    text = lane.lvs_reference()
    text = text.replace("MPL_B5_f0 BLL<5>", "MPL_B5_f0 BLL<6>")
    reference.write_text(text)
    assert not run_lvs(gds, reference, tmp_path / "lvs", cell_name=cell.name).matched


@pytest.mark.parametrize(
    ("words", "bits", "periphery", "sample"),
    [
        (16, 8, False, False),
        (64, 32, False, False),
        (16, 8, True, False),
        (64, 32, True, False),
        (128, 33, True, False),
        (16, 8, False, True),
        (64, 32, True, True),
        (128, 33, True, True),
        (16, 6, True, "control"),
        (64, 32, True, "control"),
        (128, 33, True, "control"),
        (16, 6, True, "register"),
        (64, 32, True, "register"),
        (128, 33, True, "register"),
    ],
    ids=[
        "16x8",
        "64x32",
        "16x8_latched",
        "64x32_latched",
        "128x33_latched",
        "16x8_sampled",
        "64x32_latched_sampled",
        "128x33_latched_sampled",
        "16x6_self_timed",
        "64x32_self_timed",
        "128x33_self_timed",
        "16x6_registered",
        "64x32_registered",
        "128x33_registered",
    ],
)
def test_rom_with_its_sense_lane_is_clean(words, bits, periphery, sample, asap7_drc):
    """The ROM with its sense lane, with the EVAL gate and address latches,
    and with the sample latches."""
    spec = RomDecoderSpec(
        array=RomArraySpec(words, bits),
        sense=True,
        enable=periphery,
        latch=periphery,
        sample=bool(sample),
        control=sample in ("control", "register"),
        register=sample == "register",
    )
    plain = RomDecoderSpec(array=RomArraySpec(words, bits), enable=periphery)
    assert spec.right_x >= plain.right_x
    assert spec.right_x >= spec.sense_lane.min_right_x
    library = _library()
    rng = random.Random(7)
    sites = {(w, b) for w in range(words) for b in range(bits) if rng.random() < 0.5}
    top = build_rom(spec, sites, lib=library)
    assert asap7_drc(library, top, tag="rom_sense") == []


def test_sense_needs_the_lane():
    with pytest.raises(ValueError, match="sense needs lane"):
        RomDecoderSpec(array=RomArraySpec(16, 8), lane=False, sense=True)


# ── Room for the macro's pins ─────────────────────────────────────────────────
def _macro(
    bits: int, strap_every: int = 16, *, register: bool = True
) -> RomDecoderSpec:
    return RomDecoderSpec(
        array=RomArraySpec(64, bits, strap_every=strap_every),
        sense=True,
        enable=True,
        latch=True,
        sample=True,
        control=True,
        register=register,
        edge_pins=True,
    )


def _q_track_is_open(lane: RomLaneSpec, row: int) -> bool:
    """A track across the row's Q columns that no bitline runs on."""
    taken = {p.track for p in lane.plan}
    return bool(lane.q_tracks(row) - taken) or (
        lane.ready_track in lane.q_tracks(row)
        and lane.plan[lane.replica_bit].row == row
    )


@pytest.mark.parametrize("bits", [12, 13, 14, 24, 27, 37, 48, 74, 96, 132])
def test_bitlines_leave_each_row_a_track_for_its_q_columns(bits):
    """These widths once took every track the Q columns cross in some row."""
    lane = _macro(bits).sense_lane
    assert all(_q_track_is_open(lane, row) for row in range(lane.rows))


@pytest.mark.parametrize("row", [0, 1])
def test_q_tracks_are_the_ones_its_column_crosses(row):
    """The plan's idea of the Q column against the column as drawn."""
    spec = _macro(16)
    lane = spec.sense_lane
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    top = build_rom_lane(lane, lib=library)
    m3 = _rects(top, "M3")
    bit = next(p.bit for p in lane.plan if p.row == row and p.bit in lane.data_bits)
    shape = _m3_at(m3, *lane.q_pin(bit))
    assert lane.q_tracks(row) == set(_crossing(shape))


def test_a_lane_with_q_labels_reserves_nothing():
    """Without pins nothing lands on those tracks, so the bitlines may use them."""
    half = RomArraySpec(32, 13)
    labelled = RomLaneSpec(half, sample=True, register=True, replica=True)
    pinned = RomLaneSpec(half, sample=True, register=True, replica=True, q_labels=False)
    assert not _q_track_is_open(labelled, 1)
    assert _q_track_is_open(pinned, 1)


@pytest.mark.parametrize(
    "bits, every", [(2, 1), (3, 2), (8, 8), (9, 8), (15, 8), (6, 16), (148, 16)]
)
def test_only_an_array_too_short_for_its_tracks_adds_straps(bits, every):
    spec = _macro(bits)
    assert spec.cells.strap_every == every
    assert spec.array.strap_every == 16


def test_a_tighter_strap_interval_is_kept():
    spec = _macro(30, strap_every=4)
    assert spec.cells.strap_every == 4


def test_one_bit_has_no_room_for_a_lane_row():
    with pytest.raises(ValueError, match="no room"):
        _macro(1).sense_lane  # noqa: B018


@pytest.mark.parametrize(
    "bits, register",
    [(8, True), (12, True), (24, True), (48, True), (164, True), (150, False)],
)
def test_widths_once_refused_get_their_pins(bits, register):
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    top = build_rom(_macro(bits, register=register), lib=library)
    labels = {label.text for label in top.get_labels()}
    assert {f"Q[{b}]" for b in range(bits)} <= labels


def test_a_pin_shut_out_of_the_first_routing_pass_goes_first(monkeypatch):
    """One pass is what refused this width; the retry is what builds it."""
    monkeypatch.setattr("chipforge_asap7.devices.rom_pins.MAX_ROUTING_PASSES", 1)
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    with pytest.raises(ValueError, match="in 1 passes"):
        build_rom(_macro(164), lib=library)
