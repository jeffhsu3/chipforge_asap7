"""NandSpec: the released post-decode NAND's geometry, and what generalises."""

from __future__ import annotations

import pytest

from chipforge_asap7.devices import NandSpec, build_nand, build_nand_row
from chipforge_asap7.devices.nand import NAND_PINS, TIE_PITCH
from chipforge_asap7.layout import GATE_PITCH

RELEASED = NandSpec(rows=((14, 7),), fingers=2)


def test_reproduces_the_released_post_decode_tile():
    """216 x 675 nm, four gates, five columns, seam at 432."""
    assert (RELEASED.width, RELEASED.height) == (216, 675)
    assert RELEASED.cell_name == "nand2_fin_14n7p_2f"
    assert RELEASED.gate_xs == [27, 81, 135, 189]
    assert RELEASED.sd_xs == [0, 54, 108, 162, 216]
    assert RELEASED.seam_y == 432
    assert RELEASED.rails == ((0, "VSS"), (675, "VDD"))
    assert RELEASED.n_band.active_span == (27, 405)
    assert RELEASED.p_band.active_span == (459, 648)


def test_gates_read_rail_b_a_output_in_every_stack():
    assert RELEASED.gate_roles == ("B", "A", "A", "B")
    assert NandSpec(rows=((6, 4),), fingers=4).gate_roles == tuple("BAABBAAB")
    assert NandSpec(rows=((3, 3),), fingers=1, abut=False).gate_roles == ("B", "A")
    # Every A gate is adjacent to an output column, every B gate to a rail column.
    for spec in (RELEASED, NandSpec(rows=((6, 4),), fingers=4)):
        n_roles = dict(zip(spec.sd_xs, spec.column_roles("n")))
        for x, role in zip(spec.gate_xs, spec.gate_roles):
            neighbours = {n_roles[x - GATE_PITCH // 2], n_roles[x + GATE_PITCH // 2]}
            assert ("Y" in neighbours) == (role == "A")
            assert ("S" in neighbours) == (role == "B")


def test_column_roles():
    assert RELEASED.column_roles("n") == ("S", "x", "Y", "x", "S")
    assert RELEASED.column_roles("p") == ("S", "Y", "S", "Y", "S")
    assert RELEASED.columns("n", "Y") == [108]
    assert RELEASED.columns("p", "S") == [0, 108, 216]
    # Both edge columns of an abutting tile are sources in both bands.
    for spec in (RELEASED, NandSpec(rows=((6, 4),), fingers=4)):
        for flavor in "np":
            roles = spec.column_roles(flavor)
            assert roles[0] == roles[-1] == "S"


def test_a_shares_one_pad_and_b_is_tied_on_m2():
    """The released cell's two B gates are joined by its parent; ours joins them."""
    assert RELEASED.pads == (("B", (27,)), ("A", (81, 135)), ("B", (189,)))
    assert RELEASED.pad_gate_xs("A") == [81]
    assert RELEASED.pad_gate_xs("B") == [27, 189]
    assert RELEASED.tie_ys == {"B": 432 - TIE_PITCH}
    wide = NandSpec(rows=((6, 4),), fingers=4)
    assert wide.pad_gate_xs("A") == [81, 297] and wide.pad_gate_xs("B") == [
        27,
        189,
        405,
    ]
    assert set(wide.tie_ys) == {"A", "B"}
    assert wide.tie_ys["A"] - wide.tie_ys["B"] == -TIE_PITCH
    assert NandSpec(rows=((3, 3),), fingers=1, abut=False).tie_ys == {}


def test_isolated_tile_adds_the_edge_dummies():
    spec = NandSpec(rows=((3, 3),), fingers=1, abut=False)
    assert (spec.width, spec.height) == (216, 270)  # the NAND2xp33 footprint
    assert spec.gate_grid_xs == [27, 81, 135, 189]
    assert spec.gate_xs == [81, 135]
    assert spec.sd_xs == [54, 108, 162]
    assert spec.active_x == (46, 170)
    assert spec.cell_name == "nand2_fin_3n3p_1f_iso"


def test_sizing_totals():
    assert (RELEASED.nfet_fins, RELEASED.pfet_fins) == (56, 28)
    assert RELEASED.drc_verifiable is False
    assert NandSpec(rows=((6, 4),)).drc_verifiable


def test_pins_sit_on_their_own_m1(boxes_on):
    for spec in (
        RELEASED,
        NandSpec(rows=((6, 4),), fingers=4),
        NandSpec(rows=((3, 3),), fingers=1, abut=False),
    ):
        cell = build_nand(spec)
        m1 = boxes_on(cell, "M1")
        for pin, (x, y) in spec.pin_positions.items():
            assert any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in m1), (
                spec,
                pin,
            )
        assert {label.text for label in cell.labels} == set(NAND_PINS)


def test_series_nodes_carry_no_contact(boxes_on):
    cell = build_nand(RELEASED)
    n_lo, n_hi = RELEASED.n_band.active_span
    n_columns = {
        round((b[0] + b[2]) / 2)
        for b in boxes_on(cell, "SDT")
        if b[1] < n_hi and b[3] > n_lo
    }
    assert n_columns == {0, 108, 216}
    assert not any(
        b[1] < n_hi and b[3] > n_lo and 40 < (b[0] + b[2]) / 2 < 70
        for b in boxes_on(cell, "LISD")
    )


def test_netlist_is_one_series_pair_and_two_pull_ups_per_finger():
    text = RELEASED.netlist()
    assert ".SUBCKT nand2_fin_14n7p_2f A B Y VDD VSS" in text
    assert text.count("nmos_rvt") == 4 and text.count("pmos_rvt") == 4
    assert "MNa0 Y A n0 VSS" in text and "MNb0 n0 B VSS VSS" in text
    assert "MPa1 Y A VDD VDD" in text and "nfin=7" in text and "nfin=14" in text


def test_row_butts_tiles_without_mirroring(boxes_on):
    row = build_nand_row(RELEASED, 3)
    assert [round(r.origin[0]) for r in row.references] == [0, 216, 432]
    assert all(not r.x_reflection and not r.rotation for r in row.references)
    assert {label.text for label in row.labels} == {
        "A0",
        "B0",
        "Y0",
        "A1",
        "B1",
        "Y1",
        "A2",
        "B2",
        "Y2",
        "VDD",
        "VSS",
    }
    assert boxes_on(row, "BOUNDARY") == [(0, 0, 648, 675)]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rows": ((14, 7), (14, 7))}, "one row"),
        ({"fingers": 1}, "even finger count"),
        ({"fingers": 0}, "must be >= 1"),
        ({"fingers": True}, "must be an integer"),
        ({"rows": ((2, 2),)}, "n band is too short"),
        ({"rows": ((14, 1),)}, "p band is too short"),
    ],
)
def test_rejects_what_cannot_be_drawn(kwargs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        NandSpec(**kwargs)


def test_band_height_lets_a_small_nand_sit_on_the_standard_cell_row():
    spec = NandSpec(rows=((2, 2),), band_height=135)
    assert spec.height == 270 and spec.cell_name == "nand2_fin_2n2p_h135_2f"
    with pytest.raises(ValueError, match="n band is too short"):
        NandSpec(rows=((2, 2),))  # 108 nm bands leave no room below the seam
