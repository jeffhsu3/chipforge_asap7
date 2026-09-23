"""Tap, filler and decap geometry.

The reference is the released `TAPCELL_ASAP7_75t_R` and `FILLER_ASAP7_75t_R`,
which `test_reference_validation.py` compares against shape for shape.  What
is asserted here is the reasoning those shapes encode, so a change that drifts
away from them fails with an explanation rather than a diff.
"""

import gdspy
import pytest

from chipforge_asap7.devices import RowStack, build_row_support
from chipforge_asap7.devices.finfet import (
    DEVICE_GATE_CUT_HEIGHT,
    GATE_LIG_HEIGHT,
    LI_RAIL_HEIGHT,
    MAX_VERIFIABLE_FINS,
    SD_BAR_WIDTH,
    SELECT_X_ENC,
    SELECT_Y_ENC,
    TAP_ACTIVE_WIDTH,
)
from chipforge_asap7.devices.row_support import (
    DEFAULT_WIDTH_CPP,
    MINIMUM_WIDTH_CPP,
    ROW_SUPPORT_KINDS,
    RowSupportSpec,
    main,
)
from chipforge_asap7.layout import FIN_PITCH, GATE_PITCH, LAYERS
from chipforge_asap7.layout.rules import CONTACT_SIZE, M1_WIDTH

#: A 3-fin/3-fin stack is two 135 nm bands: the released 270 nm cell row.
RELEASED_ROW = RowStack(rows=((3, 3),))


@pytest.fixture(autouse=True)
def _isolated_library():
    gdspy.current_library = gdspy.GdsLibrary()


def polys(cell, layer_name):
    spec = (LAYERS[layer_name]["layer"], LAYERS[layer_name]["datatype"])
    out = []
    for pset in cell.polygons:
        for pts, lyr, dt in zip(pset.polygons, pset.layers, pset.datatypes):
            if (lyr, dt) == spec:
                xs, ys = pts[:, 0], pts[:, 1]
                out.append((xs.min(), ys.min(), xs.max(), ys.max()))
    return sorted(out)


# ── Spec ──────────────────────────────────────────────────────────────────────
def test_cell_names_carry_the_stack_and_the_width():
    assert (
        RowSupportSpec(stack=RowStack(rows=((18, 18), (13, 13))), kind="tap").cell_name
        == "tap_fin_18n18p_13n13p_2cpp"
    )
    assert (
        RowSupportSpec(stack=RowStack(rows=((4, 6),), vt="lvt"), kind="decap").cell_name
        == "decap_fin_4n6p_6cpp_lvt"
    )


def test_defaults_match_the_released_widths():
    assert DEFAULT_WIDTH_CPP["filler"] == DEFAULT_WIDTH_CPP["tap"] == 2
    for kind in ROW_SUPPORT_KINDS:
        spec = RowSupportSpec(stack=RELEASED_ROW, kind=kind)
        assert spec.height == 270


def test_a_pinned_stack_ties_its_bands_to_the_real_rails():
    """`band_height` moves the upper rail, and the tap's ties must follow it.

    With the row height summed from fin-default band heights, a pinned 2-fin
    stack put `rail_y` at 216 while the rails were drawn at 270, so the well
    tie stopped 54 nm short of the VDD rail it exists to reach.  DRC cannot
    see that; the LISD extents can.
    """
    pinned = RowStack(rows=((2, 2),), band_height=135)
    spec = RowSupportSpec(stack=pinned, kind="tap")
    assert spec.height == 270
    assert spec.cell_name == "tap_fin_2n2p_h135_2cpp"
    lisd = polys(build_row_support(spec), "LISD")
    assert any(y1 == 270 for _x0, _y0, _x1, y1 in lisd), "p-band tie must reach VDD"
    assert any(y0 == 0 for _x0, y0, _x1, _y1 in lisd), "n-band tie must reach VSS"


def test_a_single_pitch_cell_cannot_stand_alone():
    """FIN, NWELL, NSELECT and PSELECT all want 108 nm of horizontal width.

    A 54 nm cell is short on four rules at once.  `FILLERxp5_ASAP7_75t_R` is
    exactly that width and passes only because its layers merge with the cells
    it is placed between, which is not something a generator can promise.
    """
    assert min(MINIMUM_WIDTH_CPP.values()) == 2
    for kind in ROW_SUPPORT_KINDS:
        with pytest.raises(ValueError, match="width_cpp >="):
            RowSupportSpec(stack=RELEASED_ROW, kind=kind, width_cpp=1)


def test_a_decap_needs_a_gate_between_two_columns():
    assert MINIMUM_WIDTH_CPP["decap"] == 3
    with pytest.raises(ValueError, match="width_cpp >= 3"):
        RowSupportSpec(stack=RELEASED_ROW, kind="decap", width_cpp=2)
    assert RowSupportSpec(stack=RELEASED_ROW, kind="decap", width_cpp=3).gate_xs


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"kind": "spacer"}, "kind must be one of"), ({"width_cpp": 2.5}, "integer")],
)
def test_rejects_impossible_specs(kwargs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        RowSupportSpec(stack=RELEASED_ROW, **kwargs)


def test_contacts_sit_between_two_gates():
    spec = RowSupportSpec(stack=RELEASED_ROW, kind="decap", width_cpp=6)
    assert spec.gate_grid_xs == [27, 81, 135, 189, 243, 297]
    assert spec.gate_xs == [81, 135, 189, 243]  # the two edge tracks are dummies
    assert spec.tie_xs == [54, 108, 162, 216, 270]


def test_only_a_decap_has_devices():
    stack = RowStack(rows=((4, 6),))
    assert RowSupportSpec(stack=stack, kind="tap").devices == ()
    assert RowSupportSpec(stack=stack, kind="filler").devices == ()
    decap = RowSupportSpec(stack=stack, kind="decap", width_cpp=6)
    assert [count for _band, count in decap.devices] == [4 * 4, 6 * 4]


def test_drc_verifiable_follows_the_tie_height_not_the_fin_count():
    """A tap fills its band, so a tall stack puts its tie past the deck's list."""
    tall = RowStack(rows=((18, 18), (13, 13)))
    assert not RowSupportSpec(stack=tall, kind="tap").drc_verifiable
    assert not RowSupportSpec(stack=tall, kind="decap").drc_verifiable
    # A filler has no ACTIVE at all, so there is nothing to be unsure about.
    assert RowSupportSpec(stack=tall, kind="filler").drc_verifiable
    short = RowStack(rows=((MAX_VERIFIABLE_FINS, MAX_VERIFIABLE_FINS),))
    assert RowSupportSpec(stack=short, kind="tap").drc_verifiable


# ── Drawn geometry ────────────────────────────────────────────────────────────
def test_a_filler_carries_no_diffusion_at_all():
    cell = build_row_support(RowSupportSpec(stack=RELEASED_ROW, kind="filler"))
    for layer in ("ACTIVE", "SDT", "LISD"):
        assert polys(cell, layer) == [], layer
    # ... but it still stitches its LI rail to the power rail above it.
    assert len(polys(cell, "V0")) == len(RELEASED_ROW.rails)


def test_a_tap_is_doped_opposite_to_the_row_it_serves():
    """Which is why it cannot sit straight against a logic cell."""
    spec = RowSupportSpec(stack=RELEASED_ROW, kind="tap")
    tap = build_row_support(spec)
    filler = build_row_support(RowSupportSpec(stack=RELEASED_ROW, kind="filler"))
    for band in spec.bands:
        strip = (0, band.y0, spec.width, band.y1)
        assert strip in polys(tap, band.tap_implant)
        assert strip in polys(filler, band.implant)
    # The well follows the bands, not the implant: an n-well tie has to be
    # inside the well it ties.
    assert polys(tap, "NWELL") == polys(filler, "NWELL")


def test_tap_ties_fill_their_band_between_the_select_enclosures():
    spec = RowSupportSpec(stack=RowStack(rows=((4, 6),)), kind="tap")
    cell = build_row_support(spec)
    ties = polys(cell, "ACTIVE")
    assert len(ties) == len(spec.bands) * len(spec.tie_xs)
    for band, tie in zip(spec.bands, ties):
        assert tie[2] - tie[0] == TAP_ACTIVE_WIDTH
        assert (tie[1], tie[3]) == (band.y0 + SELECT_Y_ENC, band.y1 - SELECT_Y_ENC)
        assert (tie[3] - tie[1]) % FIN_PITCH == 0


def test_ties_reach_their_rail_on_lisd_and_contact_it_there():
    """The released TAPCELL's arrangement, and two shapes fewer than a stub."""
    spec = RowSupportSpec(stack=RowStack(rows=((4, 6), (5, 5))), kind="tap")
    cell = build_row_support(spec)
    bars = polys(cell, "LISD")
    assert len(bars) == len(spec.bands) * len(spec.tie_xs)
    for band, bar in zip(spec.bands, bars):
        assert bar[2] - bar[0] == SD_BAR_WIDTH
        assert band.rail_y in (bar[1], bar[3])

    # One contact per rail per column -- not one per band, or two stacked rows
    # would each place a via on the rail they share.
    vias = polys(cell, "V0")
    assert len(vias) == len(spec.rails) * len(spec.tie_xs)
    assert {(v[1] + v[3]) / 2 for v in vias} == {y for y, _net in spec.rails}


def test_a_rail_contact_lands_on_local_interconnect_because_of_the_li_rail():
    """V0.AUX.1-2 oversizes LIG by 1 nm, so a 16 nm rail covers an 18 nm via.

    Without it the contact hangs over the end of its LISD bar -- the released
    cells carry these rails for exactly this reason, not for decoration.
    """
    spec = RowSupportSpec(stack=RowStack(rows=((4, 6),)), kind="tap")
    cell = build_row_support(spec)
    assert LI_RAIL_HEIGHT + 2 >= CONTACT_SIZE
    rails = polys(cell, "LIG")
    assert len(rails) == len(spec.rails)
    for via in polys(cell, "V0"):
        rail = next(r for r in rails if r[1] <= (via[1] + via[3]) / 2 <= r[3])
        assert rail[0] <= via[0] and rail[2] >= via[2]
        assert rail[1] - 1 <= via[1] and rail[3] + 1 >= via[3]


def test_power_rails_span_the_full_width_so_a_row_stays_continuous():
    for kind in ROW_SUPPORT_KINDS:
        spec = RowSupportSpec(stack=RELEASED_ROW, kind=kind)
        cell = build_row_support(spec)
        for layer, height in (("M1", M1_WIDTH), ("LIG", LI_RAIL_HEIGHT)):
            rails = [p for p in polys(cell, layer) if p[3] - p[1] == height]
            assert len(rails) == len(spec.rails)
            assert all((r[0], r[2]) == (0, spec.width) for r in rails)


def test_poly_is_cut_at_every_rail_and_at_a_seam_only_where_it_is_dummy():
    """A decap's plates have to stay continuous through the seam to share a gate.

    So only its two edge tracks are cut there -- the same split the released
    DECAPx4 draws -- while a tap or filler, whose poly is dummy throughout,
    is cut right across.
    """
    stack = RowStack(rows=((4, 6), (5, 5)))
    for kind in ("tap", "filler"):
        spec = RowSupportSpec(stack=stack, kind=kind)
        cuts = polys(build_row_support(spec), "GATE_CUT")
        assert sorted((c[1] + c[3]) / 2 for c in cuts) == sorted(
            [y for y, _ in spec.rails] + list(stack.seam_ys)
        )
        assert all((c[0], c[2]) == (0, spec.width) for c in cuts)
        assert all(c[3] - c[1] == DEVICE_GATE_CUT_HEIGHT for c in cuts)

    spec = RowSupportSpec(stack=stack, kind="decap", width_cpp=6)
    cuts = polys(build_row_support(spec), "GATE_CUT")
    seam_cuts = [c for c in cuts if (c[1] + c[3]) / 2 in stack.seam_ys]
    assert len(seam_cuts) == 2 * len(stack.seam_ys)
    assert {(c[0], c[2]) for c in seam_cuts} == {
        (0, GATE_PITCH),
        (spec.width - GATE_PITCH, spec.width),
    }
    for gate_x in spec.gate_xs:
        assert all(not (c[0] < gate_x < c[2]) for c in seam_cuts)


def test_a_decaps_plates_share_one_floating_gate_per_row():
    """No via on the strap: the node between the two capacitors is the point.

    Both bands of a row hang off it, so neither oxide sees the full supply.
    """
    stack = RowStack(rows=((4, 6), (5, 5)))
    spec = RowSupportSpec(stack=stack, kind="decap", width_cpp=6)
    cell = build_row_support(spec)
    straps = polys(cell, "LIG")
    seam_straps = [s for s in straps if s[3] - s[1] == GATE_LIG_HEIGHT]
    assert len(seam_straps) == len(stack.rows)
    for strap, y_seam in zip(seam_straps, stack.seam_ys):
        assert (strap[1] + strap[3]) / 2 == y_seam
        for gate_x in spec.gate_xs:
            assert strap[0] <= gate_x - GATE_LIG_HEIGHT / 2
            assert strap[2] >= gate_x + GATE_LIG_HEIGHT / 2
        assert not any(
            strap[1] <= (v[1] + v[3]) / 2 <= strap[3] for v in polys(cell, "V0")
        )

    # The plate diffusion is inset far enough for a full select enclosure.
    plates = polys(cell, "ACTIVE")
    assert len(plates) == len(spec.bands)
    assert all(
        (p[0], p[2]) == (SELECT_X_ENC, spec.width - SELECT_X_ENC) for p in plates
    )


@pytest.mark.parametrize("rows", [((4, 6),), ((1, 1),), ((1, 3), (2, 1))])
def test_a_decaps_plates_stay_on_the_fin_grid(rows):
    """It is a MOS capacitor, so its plate is a real fin-quantized channel.

    A one-fin band is where this separates from a tap: `RowStack` floors a
    band at four fin pitches, so the band has room the device does not use.
    A tap takes all of it -- more tie area for the same cell -- while a decap
    keeps to the fins, because a plate wider than its channel is not a
    capacitor.
    """
    stack = RowStack(rows=rows)
    spec = RowSupportSpec(stack=stack, kind="decap")
    tap = RowSupportSpec(stack=stack, kind="tap")
    for band, plate in zip(spec.bands, polys(build_row_support(spec), "ACTIVE")):
        assert (plate[1], plate[3]) == band.active_span
        assert plate[3] - plate[1] == band.fins * FIN_PITCH
        tie_lo, tie_hi = tap.active_y(band)
        assert tie_hi - tie_lo >= plate[3] - plate[1]
        if band.fins < 2:
            assert tie_hi - tie_lo > plate[3] - plate[1]


def test_labels_name_every_rail():
    spec = RowSupportSpec(stack=RowStack(rows=((4, 6), (5, 5))), kind="tap")
    cell = build_row_support(spec)
    assert sorted(label.text for label in cell.labels) == ["VDD", "VSS", "VSS"]
    assert build_row_support(spec, draw_pin_labels=False).labels == []


def test_boundary_is_drawn_one_row_at_a_time():
    stack = RowStack(rows=((4, 6), (5, 5)))
    spec = RowSupportSpec(stack=stack, kind="filler")
    boxes = polys(build_row_support(spec), "BOUNDARY")
    assert [(b[1], b[3]) for b in boxes] == [
        (stack.row_ys[i], stack.row_ys[i + 1]) for i in range(len(stack.rows))
    ]


def test_cells_do_not_collide_in_the_global_library():
    before = set(gdspy.current_library.cells)
    build_row_support(RowSupportSpec(stack=RELEASED_ROW))
    build_row_support(RowSupportSpec(stack=RELEASED_ROW))
    assert set(gdspy.current_library.cells) == before


def test_cli_writes_a_gds(tmp_path):
    out = tmp_path / "tap.gds"
    main(["--kind", "tap", "--stack.rows", "4", "6", "--out", str(out)])
    assert out.is_file()
    assert "tap_fin_4n6p_2cpp" in gdspy.GdsLibrary(infile=str(out)).cells
