"""Stacked-band inverter: band arithmetic, abutment, and drawn geometry.

The reference for all of it is `dec_inv_62f_halved_AND` in the released ASAP7
SRAM.  Numbers taken from that cell are marked as such, so a change that drifts
away from it fails here rather than in a viewer.
"""

import math

import gdspy
import pytest

from chipforge_asap7.devices import (
    InverterSpec,
    build_inverter,
    build_inverter_row,
)
from chipforge_asap7.devices.finfet import (
    ACTIVE_ENC,
    CONTACT_SIZE,
    GATE_LIG_HEIGHT,
    M1_MIN_SPACE,
    M1_V0_ENCLOSURE,
    M1_WIDTH,
    MAX_FINS,
    MAX_VERIFIABLE_FINS,
    SELECT_X_ENC,
)
from chipforge_asap7.devices.inverter import (
    ACTIVE_ABUT_OVERHANG,
    GATE_STRAP_CLEARANCE,
    M2_V1_ENCLOSURE,
    SELECT_ABUT_OVERHANG,
    main,
)
from chipforge_asap7.layout import GATE_PITCH, LAYERS

#: The released decoder inverter, as `InverterSpec` describes it.
RELEASED = InverterSpec(rows=((18, 18), (13, 13)), fingers=2)


@pytest.fixture(autouse=True)
def _isolated_library():
    """Give every test in this file its own gdspy default library.

    `GdsLibrary.new_cell` registers the cell in the process-global library as
    well as the target one, so two tests drawing the same cell name would
    otherwise collide.  conftest's autouse fixture puts the real one back.
    """
    gdspy.current_library = gdspy.GdsLibrary()


def polys(cell, layer_name):
    """Bounding boxes of every polygon on `layer_name`, sorted."""
    spec = (LAYERS[layer_name]["layer"], LAYERS[layer_name]["datatype"])
    out = []
    for pset in cell.polygons:
        for pts, lyr, dt in zip(pset.polygons, pset.layers, pset.datatypes):
            if (lyr, dt) == spec:
                xs, ys = pts[:, 0], pts[:, 1]
                out.append((xs.min(), ys.min(), xs.max(), ys.max()))
    return sorted(out)


def gap(a, b):
    """Euclidean spacing between two boxes, which is what M1.S.1 measures."""
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return math.hypot(dx, dy)


# ── Fidelity to the released cell ─────────────────────────────────────────────
def test_reproduces_the_released_decoder_inverter_band_stack():
    """Every band boundary and ACTIVE span in `dec_inv_62f_halved_AND`.

    These are read straight out of the released GDS: a 162 x 1890 nm tile,
    row boundaries at 0/1080/1890, and four ACTIVE bands of 486, 486, 351 and
    351 nm.  `test_inverter_drc` checks the same numbers against the file
    itself; this keeps them asserted on a machine without the SRAM release.
    """
    assert (RELEASED.width, RELEASED.height) == (162, 1890)
    assert RELEASED.row_ys == (0, 1080, 1890)
    assert [band.active_span for band in RELEASED.bands] == [
        (27, 513),
        (567, 1053),
        (1107, 1458),
        (1512, 1863),
    ]
    assert [band.flavor for band in RELEASED.bands] == ["n", "p", "p", "n"]
    # 18 + 18 + 13 + 13 -- the "62f" the released cell is named after.
    assert sum(band.fins for band in RELEASED.bands) == 62


def test_columns_and_gates_land_on_the_released_tracks():
    assert RELEASED.gate_grid_xs == [27, 81, 135]
    assert RELEASED.gate_xs == [27, 81]  # the third belongs to the neighbour
    assert RELEASED.sd_xs == [0, 54, 108]
    assert RELEASED.source_xs == [0, 108]
    assert RELEASED.drain_xs == [54]
    assert RELEASED.output_strap_x(54) == (27, 45)
    assert RELEASED.input_gate_x == 81


def test_row_placements_match_how_asu_place_the_wordline_drivers():
    """Four instances at 0/216/216/432, alternately mirrored.

    Read out of `post_Decode_and_size_reduced_x1_sram_6t122_P1N1_32bit`.  The
    tiles overlap by one gate track, which is what "halved" means: each holds
    half of every finger pair.
    """
    assert RELEASED.row_placements(4) == (
        (0, False),
        (216, True),
        (216, False),
        (432, True),
    )
    assert RELEASED.device_pitch == 108  # not `width`; tiles overlap a track
    assert RELEASED.row_width(4) == 432


def test_abutted_tiles_produce_one_continuous_gate_and_column_array():
    """The interleave has to leave no track empty and no track claimed twice."""
    spec = InverterSpec(rows=((2, 3),), fingers=4)
    owned, columns = [], set()
    for origin, mirrored in spec.row_placements(3):
        for x in spec.gate_xs:
            owned.append(origin - x if mirrored else origin + x)
        for x in spec.sd_xs:
            columns.add(origin - x if mirrored else origin + x)

    assert len(owned) == len(set(owned)) == 3 * spec.fingers
    assert sorted(owned) == [27 + i * GATE_PITCH for i in range(3 * spec.fingers)]
    assert sorted(columns) == [i * GATE_PITCH for i in range(3 * spec.fingers + 1)]


# ── Spec arithmetic ───────────────────────────────────────────────────────────
def test_cell_name_carries_the_whole_row_stack():
    """Two stacks can share a fin total and still be different cells."""
    assert RELEASED.cell_name == "inv_fin_18n18p_13n13p_2f"
    assert InverterSpec(rows=((4, 6),)).cell_name == "inv_fin_4n6p_2f"
    assert (
        InverterSpec(rows=((4, 6),), abut=False, vt="lvt").cell_name
        == "inv_fin_4n6p_2f_lvt_iso"
    )
    # 31 pull-down fins either way, two different cells.
    assert InverterSpec(rows=((18, 18), (13, 13))).cell_name != (
        InverterSpec(rows=((16, 18), (15, 13))).cell_name
    )


def test_rails_alternate_because_rows_do():
    """Flipping alternate rows is what gives each rail a single supply."""
    three = InverterSpec(rows=((2, 2), (3, 3), (4, 4)))
    assert [net for _, net in three.rails] == ["VSS", "VDD", "VSS", "VDD"]
    assert [band.flavor for band in three.bands] == list("npp" + "nnp")
    for y_rail, _ in three.rails:
        touching = [b for b in three.bands if y_rail in (b.y0, b.y0 + b.height)]
        assert len({b.flavor for b in touching}) == 1


def test_drive_strength_is_fins_times_fingers():
    assert RELEASED.pull_down_fins == RELEASED.pull_up_fins == 62
    assert RELEASED.total_fins == 124
    wide = InverterSpec(rows=((2, 3),), fingers=4)
    assert (wide.pull_down_fins, wide.pull_up_fins) == (8, 12)


def test_fin_ceiling_and_deck_ceiling_are_reported_separately():
    """The released bands are legal ASAP7 and beyond what the runset decides."""
    assert not RELEASED.drc_verifiable
    assert InverterSpec(
        rows=((MAX_VERIFIABLE_FINS, MAX_VERIFIABLE_FINS),)
    ).drc_verifiable
    assert not InverterSpec(rows=((MAX_VERIFIABLE_FINS + 1, 2),)).drc_verifiable
    assert InverterSpec(rows=((MAX_FINS, MAX_FINS),))  # accepted, just uncheckable


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fingers": 3}, "even"),
        ({"fingers": 0}, "even"),
        ({"rows": ()}, "at least one row"),
        ({"rows": ((MAX_FINS + 1, 2),)}, "tallest device"),
        ({"rows": ((0, 2),)}, "must be >= 1"),
        ({"vt": "xvt"}, "vt must be one of"),
    ],
)
def test_rejects_impossible_specs(kwargs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        InverterSpec(**kwargs)


def test_isolated_placement_has_no_abutment_geometry():
    island = InverterSpec(rows=((4, 6),), abut=False)
    with pytest.raises(ValueError, match="abut=True"):
        island.row_placements(2)


def test_netlist_is_sized_by_nfin_not_width():
    text = RELEASED.netlist()
    assert ".SUBCKT inv_fin_18n18p_13n13p_2f A Y VDD VSS" in text
    assert "M0 Y A VSS VSS nmos_rvt nfin=18 l=20n nf=2 m=1" in text
    assert "M2 Y A VDD VDD pmos_rvt nfin=13 l=20n nf=2 m=1" in text
    assert text.count("\nM") == 4


# ── Drawn geometry ────────────────────────────────────────────────────────────
def test_every_active_finger_reaches_the_input_strap():
    """A pad on one finger would leave the rest of the device on floating poly.

    This is the bug the released cell's single 22 nm LIG pad looks like it
    has: contacting only `input_gate_x` leaves the other fingers undriven.
    """
    spec = InverterSpec(rows=((2, 3),), fingers=4)
    cell = build_inverter(spec)
    straps = polys(cell, "LIG")
    assert len(straps) == len(spec.rows)
    for strap in straps:
        for gate_x in spec.gate_xs:
            assert strap[0] <= gate_x - GATE_LIG_HEIGHT / 2
            assert strap[2] >= gate_x + GATE_LIG_HEIGHT / 2
        # ... and stop short of the shared edge gate, which is not ours to drive.
        assert strap[2] < spec.gate_grid_xs[-1] - GATE_LIG_HEIGHT / 2


def test_gate_strap_sits_where_no_sd_bar_can_reach_it():
    """The n/p seam is the only Y in the cell with no LISD across the columns."""
    cell = build_inverter(RELEASED)
    strap = polys(cell, "LIG")[0]
    for bar in polys(cell, "LISD"):
        assert bar[3] <= strap[1] or bar[1] >= strap[3]


def test_output_and_input_straps_keep_their_m1_spacing():
    """The output cannot sit on the drain: 27 nm of track leaves 9 nm of gap.

    Tucked beside the drain via instead -- the released arrangement -- it gets
    27 nm from the input strap and exactly `M1_MIN_SPACE` from the source stub.
    """
    for spec in (RELEASED, InverterSpec(rows=((2, 3),), fingers=4, abut=False)):
        cell = build_inverter(spec)
        metal = polys(cell, "M1")
        strap_x0s = {spec.output_strap_x(x)[0] for x in spec.drain_xs}
        outputs = [p for p in metal if p[0] in strap_x0s and p[3] - p[1] > M1_WIDTH]
        inputs = [p for p in metal if p[0] == spec.input_gate_x - M1_WIDTH / 2]
        stubs = [
            p
            for p in metal
            if p[0] in {x - M1_WIDTH / 2 for x in spec.source_xs}
            and p[2] - p[0] == M1_WIDTH
        ]
        assert outputs and inputs and stubs
        for other in (inputs, stubs):
            for output in outputs:
                assert min(gap(output, p) for p in other) >= M1_MIN_SPACE
        for strap in inputs:
            assert min(gap(strap, p) for p in stubs) >= M1_MIN_SPACE


def test_input_strap_buys_its_spacing_from_the_via_rows_in_y():
    """Only 4 nm separates the strap from a drain pad in X.

    The whole Euclidean margin therefore comes from `GATE_STRAP_CLEARANCE`;
    shrink it and the cell stops being DRC clean without any shape moving.
    """
    cell = build_inverter(RELEASED)
    band = RELEASED.bands[0]
    strap = min(
        (p for p in polys(cell, "M1") if p[0] == RELEASED.input_gate_x - M1_WIDTH / 2),
        key=lambda p: p[1],
    )
    assert strap[1] == band.contact_y + CONTACT_SIZE / 2 + GATE_STRAP_CLEARANCE
    pads = [p for p in polys(cell, "M1") if p[1] == band.contact_y - M1_WIDTH / 2]
    assert pads and min(gap(strap, pad) for pad in pads) >= M1_MIN_SPACE


def test_source_stubs_reach_the_far_edge_of_their_rail():
    """Stopping on the rail centre line leaves a 9 nm concave step.

    On the column an abutting cell puts at x=0 that step is a real M1.W.1, so
    the stub has to cross the rail rather than land on it.
    """
    cell = build_inverter(RELEASED)
    for band in RELEASED.bands:
        for x_source in RELEASED.source_xs:
            stub = next(
                p
                for p in polys(cell, "M1")
                if p[0] == x_source - M1_WIDTH / 2
                and p[1] <= band.contact_y <= p[3]
                and p[3] - p[1] < 4 * M1_WIDTH
            )
            assert band.rail_y - M1_WIDTH / 2 >= stub[1]
            assert band.rail_y + M1_WIDTH / 2 <= stub[3]


def test_source_and_drain_vias_face_their_own_rail():
    """Contacting the far ACTIVE edge would run a conductor down the channel."""
    cell = build_inverter(RELEASED)
    vias = polys(cell, "V0")
    for band in RELEASED.bands:
        act_lo, act_hi = band.active_span
        expected = act_lo if band.rail_y <= band.y0 else act_hi - CONTACT_SIZE
        rows = {p[1] for p in vias if act_lo <= p[1] and p[3] <= act_hi}
        assert rows == {expected}


def test_fin_grid_covers_an_abutting_cells_active_overhang():
    """ACTIVE.FIN.EX.1 wants 10 nm of ACTIVE past FIN.

    An abutting cell cannot give that at its own edge -- its ACTIVE overhangs
    on purpose -- so FIN runs past the overhang instead and crosses the
    boundary rather than ending just inside it.
    """
    cell = build_inverter(RELEASED)
    ax0, ax1 = RELEASED.active_x
    assert (ax0, ax1) == (-ACTIVE_ABUT_OVERHANG, RELEASED.width + ACTIVE_ABUT_OVERHANG)
    for fin in polys(cell, "FIN"):
        assert fin[0] <= ax0 - ACTIVE_ENC and fin[2] >= ax1 + ACTIVE_ENC
    for active in polys(cell, "ACTIVE"):
        assert (active[0], active[2]) == (ax0, ax1)


def test_isolated_style_is_a_self_contained_island():
    """No overhang, a dummy gate on each side, and a full select enclosure."""
    island = InverterSpec(rows=((4, 6),), abut=False)
    assert island.width == RELEASED.width + GATE_PITCH
    assert island.select_x == (0, island.width)
    assert island.active_x == (SELECT_X_ENC, island.width - SELECT_X_ENC)
    assert island.gate_grid_xs[0] not in island.gate_xs
    assert island.gate_grid_xs[-1] not in island.gate_xs

    cell = build_inverter(island)
    for active in polys(cell, "ACTIVE"):
        for name in ("NSELECT", "PSELECT"):
            enclosing = [
                s for s in polys(cell, name) if s[1] <= active[1] and s[3] >= active[3]
            ]
            for select in enclosing:
                assert active[0] - select[0] >= SELECT_X_ENC
                assert select[2] - active[2] >= SELECT_X_ENC


def test_implant_and_well_tile_the_cell_one_band_at_a_time():
    cell = build_inverter(RELEASED)
    covered = sorted(
        polys(cell, "NSELECT") + polys(cell, "PSELECT"), key=lambda p: p[1]
    )
    assert [(p[1], p[3]) for p in covered] == [
        (band.y0, band.y0 + band.height) for band in RELEASED.bands
    ]
    wells = polys(cell, "NWELL")
    assert [(p[1], p[3]) for p in wells] == [
        (band.y0, band.y0 + band.height)
        for band in RELEASED.bands
        if band.flavor == "p"
    ]
    for select in covered:
        assert (select[0], select[2]) == RELEASED.select_x
    assert RELEASED.select_x == (
        -SELECT_ABUT_OVERHANG,
        RELEASED.width + SELECT_ABUT_OVERHANG,
    )


def test_m2_landings_clear_the_runsets_five_nanometre_ring():
    """V1.M2.EN.2 erases a strip that is exactly 5 nm, so 5 nm is not enough."""
    spec = InverterSpec(rows=((2, 3),), fingers=4)
    cell = build_inverter(spec)
    vias = polys(cell, "V1")
    assert vias
    for via in vias:
        landing = next(p for p in polys(cell, "M2") if gap(p, via) == 0)
        assert landing[0] <= via[0] - M2_V1_ENCLOSURE
        assert landing[2] >= via[2] + M2_V1_ENCLOSURE
        assert M2_V1_ENCLOSURE > M1_V0_ENCLOSURE


def test_output_crosses_a_rail_on_m3_because_m1_owns_that_track():
    two_rows = polys(build_inverter(RELEASED), "M3")
    assert len(two_rows) == len(RELEASED.rows) - 1
    assert polys(build_inverter(InverterSpec(rows=((4, 6),))), "M3") == []
    for jog in two_rows:
        assert jog[1] < 1080 < jog[3]  # the vdd rail it steps over


def test_one_gate_cut_at_each_boundary_and_none_between_rows():
    """A single uncut poly stripe is what makes one input drive every band."""
    cell = build_inverter(RELEASED)
    cuts = polys(cell, "GATE_CUT")
    assert [(c[1] + c[3]) / 2 for c in cuts] == [0, RELEASED.height]
    for gate in polys(cell, "GATE"):
        assert gate[1] < 0 and gate[3] > RELEASED.height


def test_pin_labels_name_the_pins_and_every_rail():
    """Two rows have two VSS rails, and LVS has to be told they are one net."""
    cell = build_inverter(RELEASED)
    names = sorted(label.text for label in cell.labels)
    assert names == ["A", "VDD", "VSS", "VSS", "Y"]
    ys = {label.text: set() for label in cell.labels}
    for label in cell.labels:
        ys[label.text].add(label.position[1])
    assert ys["VSS"] == {0, RELEASED.height}


def test_pin_labels_can_be_suppressed_for_composite_cells():
    assert build_inverter(RELEASED, draw_pin_labels=False).labels == []


def test_row_labels_separate_every_driver():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    row = build_inverter_row(RELEASED, 4, lib=library)
    names = sorted(label.text for label in row.labels)
    assert names == [
        "A0",
        "A1",
        "A2",
        "A3",
        "VDD",
        "VSS",
        "VSS",
        "Y0",
        "Y1",
        "Y2",
        "Y3",
    ]
    inputs = {
        label.text: label.position[0] for label in row.labels if label.text[0] == "A"
    }
    assert sorted(inputs.values()) == [81, 135, 297, 351]
    assert len(row.references) == 4


def test_row_boundary_spans_the_whole_row():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    row = build_inverter_row(RELEASED, 4, lib=library)
    boundaries = polys(row, "BOUNDARY")
    assert [(b[0], b[2]) for b in boundaries] == [(0, 432)] * len(RELEASED.rows)


def test_cells_do_not_collide_in_the_global_library():
    before = set(gdspy.current_library.cells)
    build_inverter(RELEASED)
    build_inverter(RELEASED)
    assert set(gdspy.current_library.cells) == before


def test_cli_writes_a_gds(tmp_path):
    out = tmp_path / "inv.gds"
    main(["--rows", "18:18,13:13", "--fingers", "2", "--out", str(out)])
    assert out.is_file()
    library = gdspy.GdsLibrary(infile=str(out))
    assert RELEASED.cell_name in library.cells
