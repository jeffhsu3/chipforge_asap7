"""The shared row stack: band positions, rail polarity, and what validates when.

`RowStack` is the one place that decides where a row's transistor bands sit.
A logic cell and the tap or filler beside it both read it, so a disagreement
here is a row that does not abut.
"""

from itertools import pairwise

import pytest

from chipforge_asap7.devices import RowStack
from chipforge_asap7.devices.finfet import MAX_FINS, SELECT_Y_ENC
from chipforge_asap7.layout import FIN_PITCH


def test_reproduces_the_released_decoder_inverter_bands():
    """The band boundaries read out of `dec_inv_62f_halved_AND`."""
    stack = RowStack(rows=((18, 18), (13, 13)))
    assert stack.height == 1890
    assert stack.row_ys == (0, 1080, 1890)
    assert stack.seam_ys == (540, 1485)
    assert [band.active_span for band in stack.bands()] == [
        (27, 513),
        (567, 1053),
        (1107, 1458),
        (1512, 1863),
    ]


def test_a_standard_cell_row_is_three_fins_per_band():
    """Two 135 nm bands, which is where the released 270 nm row comes from."""
    stack = RowStack(rows=((3, 3),))
    assert stack.height == 270
    assert [band.height for band in stack.bands()] == [135, 135]
    assert stack.seam_ys == (135,)
    assert stack.rails == ((0, "VSS"), (270, "VDD"))


def test_rows_flip_so_every_rail_carries_one_supply():
    stack = RowStack(rows=((2, 2), (3, 3), (4, 4)))
    assert [band.flavor for band in stack.bands()] == list("np" + "pn" + "np")
    assert [net for _, net in stack.rails] == ["VSS", "VDD", "VSS", "VDD"]
    for y_rail, net in stack.rails:
        touching = [b for b in stack.bands() if y_rail in (b.y0, b.y1)]
        assert {b.rail_net for b in touching} == {net}


def test_each_band_faces_its_own_rail():
    """Contacting the far ACTIVE edge would run a conductor down the channel."""
    for band in RowStack(rows=((4, 6), (5, 5))).bands():
        lo, _hi = band.active_span
        assert band.rail_y in (band.y0, band.y1)
        assert band.rail_below == (band.rail_y == band.y0)
        assert (lo < band.contact_y < lo + FIN_PITCH) == band.rail_below


def test_a_tap_is_doped_the_opposite_way_to_the_devices_it_serves():
    """The one inversion that separates the released TAPCELL from FILLER."""
    for band in RowStack(rows=((4, 6),)).bands():
        assert {band.implant, band.tap_implant} == {"NSELECT", "PSELECT"}
        assert band.implant != band.tap_implant
        assert band.in_nwell == (band.flavor == "p")
        assert (band.implant == "PSELECT") == band.in_nwell


def test_bands_carry_the_finger_count_through_to_their_device_spec():
    assert [b.spec.fingers for b in RowStack(rows=((2, 2),)).bands(4)] == [4, 4]
    assert [b.spec.fingers for b in RowStack(rows=((2, 2),)).bands()] == [1, 1]
    # Fingers change the columns, never the band's height or its ACTIVE.
    one, four = RowStack(rows=((2, 2),)).bands(), RowStack(rows=((2, 2),)).bands(4)
    assert [b.active_span for b in one] == [b.active_span for b in four]


def test_band_heights_leave_room_for_a_tap_in_every_band():
    """A tap fills the band between its two select enclosures."""
    for stack in (RowStack(rows=((1, 1),)), RowStack(rows=((18, 18), (13, 13)))):
        for band in stack.bands():
            tie_height = band.height - 2 * SELECT_Y_ENC
            assert tie_height > 0
            assert tie_height % FIN_PITCH == 0


def test_code_carries_the_rows_not_the_total():
    assert RowStack(rows=((18, 18), (13, 13))).code == "18n18p_13n13p"
    assert RowStack(rows=((4, 6),)).code == "4n6p"
    # Same 31+31 fins, different geometry, different code.
    assert RowStack(rows=((16, 18), (15, 13))).code != (
        RowStack(rows=((18, 18), (13, 13))).code
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rows": ()}, "at least one row"),
        ({"rows": ((MAX_FINS + 1, 2),)}, "tallest device"),
        ({"rows": ((0, 2),)}, "must be >= 1"),
        ({"rows": ((2,),)}, "integer pairs"),
        ({"rows": 4}, "integer pairs"),
        ({"vt": "xvt"}, "vt must be one of"),
        ({"rows": ((4, 4),), "band_height": 135}, "too short"),
        ({"rows": ((2, 2),), "band_height": 140}, "not a multiple of fin pitch"),
        ({"band_height": True}, "band_height must be an integer"),
    ],
)
def test_rejects_impossible_stacks_at_construction(kwargs, match):
    """Eagerly, not at draw time.

    `bands` is a method, so a stack that only asserted its truthiness would
    validate nothing at all -- a bound method is always truthy.
    """
    with pytest.raises((ValueError, TypeError), match=match):
        RowStack(**kwargs)


def test_rows_are_normalised_so_the_stack_stays_hashable():
    stack = RowStack(rows=[[4, 6], [5, 5]])
    assert stack.rows == ((4, 6), (5, 5))
    assert hash(stack) == hash(RowStack(rows=((4, 6), (5, 5))))


def test_fin_grid_spans_the_whole_stack():
    stack = RowStack(rows=((4, 6), (5, 5)))
    ys = stack.fin_grid_ys
    assert len(ys) == stack.height // FIN_PITCH
    assert ys[0] == 10 and all(b - a == FIN_PITCH for a, b in pairwise(ys))


def test_band_height_pins_every_band_to_one_height():
    """``band_height=135`` puts a row of any legal fin count on the 270 nm row.

    Left alone, a band is as tall as its fins make it, so a 2-fin row is
    216 nm and does not sit on the released 7.5-track rails.
    """
    assert RowStack(rows=((2, 2),)).height == 216
    for fins in (1, 2, 3):
        stack = RowStack(rows=((fins, fins),), band_height=135)
        assert stack.height == 270
        assert [band.height for band in stack.bands()] == [135, 135]
        assert stack.rails == ((0, "VSS"), (270, "VDD"))
        assert stack.seam_ys == (135,)
        assert [band.spec.row_height for band in stack.bands()] == [135, 135]


def test_rails_follow_the_drawn_band_height():
    """The row is as tall as its bands are drawn, not as tall as their fins imply.

    `bands` used to sum the fin-default heights for the row height, so a
    pinned 2-fin row reported its VDD rail at 216 nm while `row_ys` and the
    drawn rails said 270: every source stub and tap tie aimed at `rail_y`
    stopped 54 nm short of the rail.
    """
    n_band, p_band = RowStack(rows=((2, 2),), band_height=135).bands()
    assert (n_band.rail_y, p_band.rail_y) == (0, 270)
    for stack in (
        RowStack(rows=((2, 2),), band_height=135),
        RowStack(rows=((1, 1), (2, 3)), band_height=162),
        RowStack(rows=((4, 6), (5, 5))),
    ):
        for band in stack.bands():
            assert band.rail_y in stack.row_ys
            assert band.rail_y in (band.y0, band.y1)


def test_band_height_changes_the_code():
    """Two geometries must not claim one cell name."""
    assert RowStack(rows=((2, 2),), band_height=135).code == "2n2p_h135"
    assert (
        RowStack(rows=((3, 3),), band_height=135).code != RowStack(rows=((3, 3),)).code
    )
