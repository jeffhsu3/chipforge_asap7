"""Parametric FinFET: fin-count naming, discrete sizing, and drawn geometry."""

import gdspy
import pytest

from chipforge_asap7.devices import FinFETSpec, build_finfet, nmos_fin, pmos_fin
from chipforge_asap7.devices.finfet import (
    ACTIVE_ENC,
    CONTACT_SIZE,
    DEVICE_GATE_CUT_HEIGHT,
    GATE_CUT_MIN_SPACE,
    GATE_LIG_HEIGHT,
    GATE_SD_SPACE,
    GATE_V0_DX,
    M1_BOUNDARY_INSET,
    M1_MIN_SPACE,
    M1_V0_ENCLOSURE,
    M1_WIDTH,
    MAX_FINS,
    MAX_VERIFIABLE_FINS,
    POLY_OVERHANG,
    ROUTING_BAND_HEIGHT,
    SD_BAR_WIDTH,
    TAP_COLUMN_WIDTH,
    V0_LISD_ENCLOSURE,
    main,
)
from chipforge_asap7.layout import (
    FIN_PITCH,
    FIN_WIDTH,
    GATE_PITCH,
    GATE_WIDTH,
    LAYERS,
)


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


# ── Naming: the "122" nomenclature ────────────────────────────────────────────
def test_code_is_the_fin_count_digit_string():
    assert FinFETSpec(fins=1, fingers=2, multipliers=2).code == "122"
    assert FinFETSpec(fins=2, fingers=2, multipliers=1).code == "221"
    assert FinFETSpec().code == "111"


def test_cell_name_carries_flavor_and_code():
    assert FinFETSpec(fins=1, fingers=2, multipliers=2).cell_name == "nmos_fin_122"
    assert (
        FinFETSpec(flavor="p", fins=1, fingers=2, multipliers=2).cell_name
        == "pmos_fin_122"
    )
    assert FinFETSpec(vt="lvt").cell_name == "nmos_lvt_fin_111"
    assert FinFETSpec(flavor="p", vt="slvt").cell_name == "pmos_slvt_fin_111"


def test_cell_name_separates_a_non_default_row_height():
    """Two bands, same counts: the name has to tell them apart."""
    default = FinFETSpec(fins=1)
    taller = FinFETSpec(fins=1, row_height=135)

    assert default.height != taller.height
    assert taller.cell_name == "nmos_fin_111_h135"
    assert default.cell_name != taller.cell_name
    # Passing the default explicitly is not a "custom" band.
    assert FinFETSpec(fins=1, row_height=108).cell_name == default.cell_name

    lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    build_finfet(default, lib=lib)
    build_finfet(taller, lib=lib)  # would raise if the names collided
    assert set(lib.cells) == {default.cell_name, taller.cell_name}


def test_double_digit_counts_stay_unambiguous():
    """A bare digit string would make 1,12,1 and 11,2,1 collide."""
    assert FinFETSpec(fins=12, fingers=2, multipliers=1).code == "12_2_1"
    assert FinFETSpec(fins=1, fingers=12, multipliers=1).code == "1_12_1"


def test_built_cell_takes_the_code_name():
    assert nmos_fin(fins=1, fingers=2, multipliers=2).name == "nmos_fin_122"
    assert pmos_fin(fins=2).name == "pmos_fin_211"
    assert nmos_fin(vt="sram").name == "nmos_sram_fin_111"
    assert nmos_fin(name="custom").name == "custom"


# ── Discrete sizing ───────────────────────────────────────────────────────────
def test_sizing_is_quantized_to_fins():
    spec = FinFETSpec(fins=3, fingers=2, multipliers=2)
    assert spec.total_fins == 3 * 2 * 2
    assert spec.layout_fingers == spec.fingers * spec.multipliers
    assert spec.height_per_row == (3 + 2) * FIN_PITCH
    assert spec.height == spec.height_per_row + ROUTING_BAND_HEIGHT
    assert spec.device_width == (spec.layout_fingers + 2) * GATE_PITCH
    assert spec.width == TAP_COLUMN_WIDTH + spec.device_width


def test_gates_and_sd_columns_interleave():
    spec = FinFETSpec(fins=1, fingers=3)
    assert spec.gate_xs == [189, 243, 297]
    # One more S/D column than fingers: neighbours share the column between.
    assert spec.sd_xs == [162, 216, 270, 324]
    assert spec.source_xs == [162, 270]
    assert spec.drain_xs == [216, 324]
    for gate, left, right in zip(spec.gate_xs, spec.sd_xs, spec.sd_xs[1:]):
        assert gate - left == right - gate == spec.finger_pitch // 2


def test_off_grid_gate_pitch_is_rejected_instead_of_drawing_dirty_gds():
    with pytest.raises(ValueError, match="finger_pitch_cpp=1"):
        FinFETSpec(fingers=2, finger_pitch_cpp=2)


def test_active_fins_stay_quantized_when_multipliers_are_folded_in_x():
    spec = FinFETSpec(fins=2, multipliers=3)
    ys = spec.fin_ys()
    assert spec.layout_fingers == 3
    assert len(ys) == 2
    assert ys[1] - ys[0] == FIN_PITCH
    lo, hi = spec.active_span()
    assert ys[0] - lo == ACTIVE_ENC
    assert hi - (ys[-1] + FIN_WIDTH) == ACTIVE_ENC
    with pytest.raises(IndexError, match="folded"):
        spec.fin_ys(1)


def test_rejects_impossible_specs():
    """Each rejection is matched, so none of them can pass for another reason."""
    with pytest.raises(ValueError, match="fins must be >= 1"):
        FinFETSpec(fins=0)
    with pytest.raises(ValueError, match="flavor must be"):
        FinFETSpec(flavor="x")
    with pytest.raises(ValueError, match="gate_length=20"):
        FinFETSpec(gate_length=GATE_PITCH)  # only 20 nm poly is on the grid
    with pytest.raises(ValueError, match="too short"):
        FinFETSpec(fins=3, row_height=81)  # 3 fins + isolation need 108 nm
    with pytest.raises(ValueError, match="not DRC-safe"):
        FinFETSpec(fingers=2, sd_dx=36)  # shared S/D fixes the spacing
    with pytest.raises(ValueError, match="gate_length=20"):
        FinFETSpec(gate_length=0)
    with pytest.raises(ValueError, match="gate_length=20"):
        FinFETSpec(gate_length=-20)
    with pytest.raises(ValueError, match="vt must be one of"):
        FinFETSpec(vt="ulvt")
    with pytest.raises(ValueError, match="fin pitch"):
        FinFETSpec(row_height=100)  # off the 27 nm grid
    with pytest.raises(ValueError, match="released ASAP7 collateral"):
        FinFETSpec(fins=MAX_FINS + 1)
    with pytest.raises(ValueError, match="too tall"):
        FinFETSpec(row_height=(MAX_FINS + 3) * FIN_PITCH)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fins": 1.5},
        {"fingers": True},
        {"multipliers": 1.5},
        {"finger_pitch_cpp": 1.5},
        {"gate_length": 20.0},
        {"sd_dx": 36.0},
        {"row_height": 81.0},
    ],
)
def test_rejects_non_integer_discrete_geometry(kwargs):
    with pytest.raises(TypeError):
        FinFETSpec(**kwargs)


def test_fixed_sd_grid_has_exact_legal_gate_clearance():
    spec = FinFETSpec(fingers=4, multipliers=2)
    # The spec derives the clearance; the constant records what the released
    # cells use.  They have to agree, and the drawn columns have to match both.
    assert spec.gate_sd_clearance == GATE_SD_SPACE
    for gate, left, right in zip(spec.gate_xs, spec.sd_xs, spec.sd_xs[1:]):
        assert gate - left == right - gate == GATE_PITCH // 2
        clearance = gate - left - GATE_WIDTH / 2 - SD_BAR_WIDTH / 2
        assert clearance == spec.gate_sd_clearance
    with pytest.raises(ValueError, match="not DRC-safe"):
        FinFETSpec(sd_dx=GATE_PITCH // 2)
    with pytest.raises(ValueError, match="gate_length=20"):
        FinFETSpec(gate_length=GATE_WIDTH + 1)


# ── Netlist ───────────────────────────────────────────────────────────────────
def test_netlist_is_sized_by_nfin_not_width():
    spec = FinFETSpec(fins=3, fingers=2, multipliers=2)
    line = spec.netlist()
    assert line == "M0 D G S B nmos_rvt nfin=3 l=20n nf=2 m=2"
    assert "w=" not in line
    assert (
        FinFETSpec(flavor="p", vt="slvt")
        .netlist("M7", ("out", "in", "vdd", "vdd"))
        .startswith("M7 out in vdd vdd pmos_slvt nfin=1")
    )


@pytest.mark.parametrize(
    ("vt", "model", "marker"),
    [
        ("rvt", "nmos_rvt", None),
        ("lvt", "nmos_lvt", "LVT"),
        ("slvt", "nmos_slvt", "SLVT"),
        ("sram", "nmos_sram", "SRAMVT"),
    ],
)
def test_vt_selects_pdk_model_and_layout_marker(vt, model, marker):
    spec = FinFETSpec(vt=vt)
    cell = build_finfet(spec)

    assert spec.model == model
    for layer_name in ("LVT", "SLVT", "SRAMVT"):
        expected_count = 1 if layer_name == marker else 0
        assert len(polys(cell, layer_name)) == expected_count


# ── Drawn geometry ────────────────────────────────────────────────────────────
def test_draws_the_full_fixed_pitch_fin_grid():
    spec = FinFETSpec(fins=2, fingers=2, multipliers=3)
    cell = build_finfet(spec)
    fins = polys(cell, "FIN")
    assert len(fins) == len(spec.fin_grid_ys) == spec.height // FIN_PITCH
    assert [y0 for _, y0, _, _ in fins] == spec.fin_grid_ys
    assert all(
        (x0, x1, y1 - y0) == (0, spec.width, FIN_WIDTH) for x0, y0, x1, y1 in fins
    )
    assert set(spec.fin_ys()).issubset(spec.fin_grid_ys)


def test_gate_grid_contains_active_and_dummy_gates():
    spec = FinFETSpec(fins=1, fingers=2, multipliers=3)
    gates = polys(build_finfet(spec), "GATE")
    assert len(gates) == len(spec.gate_grid_xs)
    centers = [(x0 + x1) / 2 for x0, _, x1, _ in gates]
    assert centers == spec.gate_grid_xs
    assert set(spec.gate_xs).issubset(centers)
    for x0, y0, x1, y1 in gates:
        assert x1 - x0 == GATE_WIDTH
        assert (y0, y1) == (-POLY_OVERHANG, spec.height + POLY_OVERHANG)


def test_gate_cuts_leave_only_active_gates_uncut_at_access_band():
    spec = FinFETSpec(multipliers=3)
    cuts = polys(build_finfet(spec), "GATE_CUT")
    half = DEVICE_GATE_CUT_HEIGHT / 2
    assert cuts == [
        (0, -half, spec.width, half),
        (
            0,
            spec.height_per_row - half,
            spec.device_x0 + GATE_PITCH,
            spec.height_per_row + half,
        ),
        (0, spec.height - half, spec.width, spec.height + half),
        (
            spec.width - GATE_PITCH,
            spec.height_per_row - half,
            spec.width,
            spec.height_per_row + half,
        ),
    ]
    cut_centers = {(y0 + y1) / 2 for _, y0, _, y1 in cuts}
    assert cut_centers == {0, spec.height_per_row, spec.height}
    middle = [p for p in cuts if (p[1] + p[3]) / 2 == spec.height_per_row]
    assert all(
        not (x0 < x_gate < x1) for x0, _, x1, _ in middle for x_gate in spec.gate_xs
    )


def test_sd_bars_and_m1_terminal_shapes():
    spec = FinFETSpec(fins=2, fingers=2, multipliers=2)
    cell = build_finfet(spec)
    lisd = polys(cell, "LISD")
    sdt = polys(cell, "SDT")
    assert len(lisd) == len(spec.sd_xs) + 1  # S/D columns plus body tap
    assert len(sdt) == len(spec.sd_xs) + 1
    for x0, _, x1, _ in lisd:
        assert x1 - x0 == SD_BAR_WIDTH
    assert all(x1 - x0 == SD_BAR_WIDTH for x0, _, x1, _ in sdt)
    assert all((y1 - y0) % FIN_PITCH == 0 for _, y0, _, y1 in sdt)
    assert len(polys(cell, "M1")) == 4  # one contiguous B/S/D/G conductor each


def test_v0_connects_every_gate_and_sd_region_to_m1():
    spec = FinFETSpec(fins=2, fingers=3, multipliers=2)
    vias = polys(build_finfet(spec), "V0")

    assert len(vias) == len(spec.sd_xs) + 2  # all S/D plus one gate and one body
    centers = {((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in vias}
    expected = {(x, spec.source_contact_y()) for x in spec.source_xs}
    expected |= {(x, spec.drain_contact_y()) for x in spec.drain_xs}
    expected |= {(spec.sd_xs[0] + GATE_V0_DX, spec.gate_contact_y())}
    expected |= {(GATE_PITCH, spec.height_per_row)}
    assert centers == expected
    assert all(x1 - x0 == y1 - y0 == CONTACT_SIZE for x0, y0, x1, y1 in vias)


def test_single_fin_contacts_follow_released_lisd_lig_overlap_style():
    spec = FinFETSpec(fins=1)
    cell = build_finfet(spec)
    lisd_by_x = {
        (x0 + x1) / 2: (x0, y0, x1, y1) for x0, y0, x1, y1 in polys(cell, "LISD")
    }
    sd_vias = [via for via in polys(cell, "V0") if (via[0] + via[2]) / 2 in spec.sd_xs]

    assert len(sd_vias) == len(spec.sd_xs)
    for vx0, vy0, vx1, vy1 in sd_vias:
        x_center = (vx0 + vx1) / 2
        lx0, ly0, lx1, ly1 = lisd_by_x[x_center]
        assert lx0 <= vx0 < vx1 <= lx1
        if x_center in spec.source_xs:
            # Source V0 straddles the LISD end and is completed by the LIG rail,
            # exactly like the ASAP7 standard-cell source/body contact.
            assert (vy0, ly0, vy1) == (-9, 0, 9)
            assert any(y0 < vy1 and y1 > vy0 for _, y0, _, y1 in polys(cell, "LIG"))
        else:
            assert ly0 <= vy0 < vy1 <= ly1


def test_lig_strap_joins_all_gate_fingers():
    spec = FinFETSpec(fingers=3, multipliers=2)
    straps = polys(build_finfet(spec), "LIG")
    assert len(straps) == 3  # source rail, body rail, gate strap
    gate_strap = next(p for p in straps if p[0] >= spec.device_x0 and p[1] > 0)
    x0, y0, x1, y1 = gate_strap
    assert x0 < spec.gate_xs[0] - GATE_WIDTH / 2
    assert x1 > spec.gate_xs[-1] + GATE_WIDTH / 2
    assert (y0 + y1) / 2 == spec.gate_contact_y()
    assert y1 - y0 == 22
    assert all(x0 < x_gate < x1 for x_gate in spec.gate_xs)


def test_active_encloses_the_fins():
    spec = FinFETSpec(fins=2)
    lo, hi = spec.active_span()
    ys = spec.fin_ys()
    assert lo == ys[0] - ACTIVE_ENC
    assert hi == ys[-1] + FIN_WIDTH + ACTIVE_ENC


def test_gate_contact_clears_active_and_the_top_gate_cut():
    spec = FinFETSpec(fins=2, multipliers=2)
    _, act_hi = spec.active_span()
    y = spec.gate_contact_y()
    assert y - 11 - act_hi >= 15
    assert spec.height - DEVICE_GATE_CUT_HEIGHT / 2 - (y + 11) >= 5


def test_active_and_select_layers_encode_device_flavor():
    spec = FinFETSpec(fins=2, multipliers=2)
    nmos = build_finfet(spec)
    pmos = build_finfet(FinFETSpec(flavor="p", fins=2, multipliers=2))

    # Each four-terminal cell has transistor ACTIVE plus the opposite-polarity
    # body-tap ACTIVE.
    assert len(polys(nmos, "ACTIVE")) == 2
    assert len(polys(pmos, "ACTIVE")) == 2
    assert len(polys(nmos, "NSELECT")) == 1
    assert len(polys(nmos, "PSELECT")) == 1
    assert len(polys(pmos, "PSELECT")) == 1
    assert len(polys(pmos, "NSELECT")) == 1
    assert polys(nmos, "NSELECT")[0][0] == TAP_COLUMN_WIDTH
    assert polys(nmos, "PSELECT")[0][2] == TAP_COLUMN_WIDTH
    assert polys(pmos, "PSELECT")[0][0] == TAP_COLUMN_WIDTH
    assert polys(pmos, "NSELECT")[0][2] == TAP_COLUMN_WIDTH
    assert polys(nmos, "NWELL") == []
    assert len(polys(pmos, "NWELL")) == 1


def test_select_layers_tile_the_whole_cell():
    """No implant-less rows.

    Every released cell splits its full boundary between NSELECT and PSELECT
    (INVxp33 and NAND2xp33 horizontally, TAPCELL likewise); leaving the
    gate-access band bare was a deviation, and it made the top ~40% of a cell
    carry gate poly with no implant marker at all.
    """
    for spec in (
        FinFETSpec(),
        FinFETSpec(fins=2, fingers=2, multipliers=4),
        FinFETSpec(flavor="p", fins=12, fingers=3, vt="slvt"),
    ):
        cell = build_finfet(spec)
        n = polys(cell, "NSELECT")
        p_ = polys(cell, "PSELECT")
        assert len(n) == len(p_) == 1

        left, right = sorted(n + p_)
        # Abutting halves, together spanning the boundary exactly.
        assert left[0] == 0 and right[2] == spec.width
        assert left[2] == right[0] == TAP_COLUMN_WIDTH
        for box in (left, right):
            assert (box[1], box[3]) == (0, spec.height)


def test_drain_pad_spans_only_the_drain_columns():
    """The drain conductor must not be dragged across the source columns.

    Ending the pad at the cell edge shorted nothing, but it ran drain M1 over
    every source LISD bar -- pure Cds that PEX would faithfully report.
    """
    for spec in (
        FinFETSpec(),
        FinFETSpec(fingers=3),
        FinFETSpec(fins=2, fingers=2, multipliers=4),
    ):
        cell = build_finfet(spec)
        act_lo, act_hi = spec.active_span()
        pad = next(p for p in polys(cell, "M1") if (p[1], p[3]) == (act_lo, act_hi))
        end_cap = CONTACT_SIZE / 2 + M1_V0_ENCLOSURE

        assert pad[0] == spec.drain_xs[0] - end_cap
        assert pad[2] == spec.drain_xs[-1] + end_cap
        assert spec.width - pad[2] >= M1_BOUNDARY_INSET

        # A contiguous pad has to bridge the source columns *between* two
        # drains -- that is inherent to fingering.  What it must not do is
        # reach past the outermost drain and cover the outer sources too.
        outer_sources = [
            x for x in spec.source_xs if x < spec.drain_xs[0] or x > spec.drain_xs[-1]
        ]
        assert outer_sources  # the first column is always a source
        bars = {(b[0] + b[2]) / 2: b for b in polys(cell, "LISD")}
        for x_source in outer_sources:
            bar = bars[x_source]
            assert not (max(pad[0], bar[0]) < min(pad[2], bar[2]))


def test_pin_labels_use_the_m1_pin_purpose():
    spec = FinFETSpec(fingers=2, multipliers=2)
    labels = build_finfet(spec).labels
    pin = LAYERS["M1_PIN"]

    assert sorted(label.text for label in labels) == ["B", "D", "G", "S"]
    assert all(label.layer == pin["layer"] for label in labels)
    assert all(label.texttype == pin["datatype"] for label in labels)


def test_cells_do_not_collide_in_the_global_library():
    """Building the same device twice must not raise."""
    a, b = nmos_fin(fins=1), nmos_fin(fins=1)
    assert a.name == b.name == "nmos_fin_111"


def test_lib_argument_registers_the_cell():
    lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    cell = nmos_fin(fins=2, lib=lib)
    assert lib.cells[cell.name] is cell


def test_drain_contact_y_keeps_its_deprecated_alias():
    """`sd_contact_y` read as the source-or-drain accessor; it never was one."""
    spec = FinFETSpec(fins=2, fingers=2)
    act_lo, _ = spec.active_span()

    assert spec.drain_contact_y() == act_lo + CONTACT_SIZE / 2
    assert spec.sd_contact_y() == spec.drain_contact_y()
    assert spec.source_contact_y() != spec.drain_contact_y()
    with pytest.raises(IndexError, match="folded"):
        spec.drain_contact_y(1)


def test_pin_labels_can_be_suppressed_for_composite_cells():
    """`sense_amp` embeds leaf devices and exposes only its own top-level pins."""
    spec = FinFETSpec(fins=2, fingers=2)

    assert build_finfet(spec, draw_pin_labels=False).labels == []
    assert build_finfet(spec).labels != []
    # The convenience wrappers reach the flag too, rather than forwarding it
    # into FinFETSpec and raising an unrelated TypeError.
    assert nmos_fin(fins=2, draw_pin_labels=False).labels == []
    assert pmos_fin(fins=2, draw_pin_labels=False).labels == []
    assert nmos_fin(fins=3).labels != []


def test_fin_ceiling_separates_the_device_limit_from_the_deck_limit():
    """`MAX_VERIFIABLE_FINS` is the deck's enumeration; `MAX_FINS` is the device.

    ASAP7 documents no fin-count maximum, and the released SRAM banks ship 13-,
    14- and 18-fin devices.  The 12 that used to gate `FinFETSpec` came from the
    public KLayout runset spelling "integer multiple of 27nm" as a list of
    twelve heights, which is a property of the deck, not of the process.
    """
    assert MAX_VERIFIABLE_FINS < MAX_FINS

    # The heights the released decoder inverter is built from must be buildable.
    for fins in (13, 14, 18):
        spec = FinFETSpec(fins=fins, fingers=3)
        assert spec.fins == fins
        assert not spec.drc_verifiable
        assert build_finfet(spec).name == f"nmos_fin_{fins}_3_1"
        # The band still follows the released bands exactly.
        assert spec.height_per_row == (fins + 2) * FIN_PITCH
        lo, hi = spec.active_span()
        assert (lo, hi - lo) == (FIN_PITCH, fins * FIN_PITCH)

    assert FinFETSpec(fins=MAX_VERIFIABLE_FINS).drc_verifiable
    assert not FinFETSpec(fins=MAX_VERIFIABLE_FINS + 1).drc_verifiable
    # The body tap is height-checked by the same rules, so a tall band on a
    # short device is unverifiable too.
    tall = FinFETSpec(fins=1, row_height=(MAX_VERIFIABLE_FINS + 3) * FIN_PITCH)
    assert tall.tap_fins == MAX_VERIFIABLE_FINS + 1
    assert not tall.drc_verifiable


# ── Margins with no runset headroom ───────────────────────────────────────────
def test_source_to_drain_m1_spacing_sits_exactly_on_the_rule():
    """M1.S.1 has zero slack here: ACTIVE_ENC and the rail height set it."""
    for spec in (FinFETSpec(fins=1), FinFETSpec(fins=4, fingers=3)):
        cell = build_finfet(spec)
        act_lo, act_hi = spec.active_span()
        m1 = polys(cell, "M1")
        rail = next(p for p in m1 if p[1] < 0)
        drain = next(p for p in m1 if (p[1], p[3]) == (act_lo, act_hi))

        assert rail[3] - rail[1] == M1_WIDTH
        assert drain[1] - rail[3] == M1_MIN_SPACE


def test_drain_v0_enclosure_by_lisd_sits_exactly_on_the_rule():
    """V0.LISD.EN.2 wants 3 nm on two opposite sides; the bar gives exactly 3."""
    spec = FinFETSpec(fins=2, fingers=3)
    cell = build_finfet(spec)
    act_lo, _ = spec.active_span()
    bars = {(p[0] + p[2]) / 2: p for p in polys(cell, "LISD")}
    vias = [p for p in polys(cell, "V0") if p[1] == act_lo]

    assert len(vias) == len(spec.drain_xs)
    for via in vias:
        bar = bars[(via[0] + via[2]) / 2]
        assert via[0] - bar[0] == bar[2] - via[2] == V0_LISD_ENCLOSURE


def test_routing_band_height_is_what_keeps_the_upper_gate_cuts_apart():
    """GCUT.S.3 is why the band is three fin pitches, not two."""
    assert ROUTING_BAND_HEIGHT - DEVICE_GATE_CUT_HEIGHT >= GATE_CUT_MIN_SPACE
    assert 2 * FIN_PITCH - DEVICE_GATE_CUT_HEIGHT < GATE_CUT_MIN_SPACE

    spec = FinFETSpec(fins=2, multipliers=2)
    cuts = polys(build_finfet(spec), "GATE_CUT")
    access_top = max(
        y1 for _, y0, _, y1 in cuts if (y0 + y1) / 2 == spec.height_per_row
    )
    boundary_bottom = min(y0 for _, y0, _, y1 in cuts if (y0 + y1) / 2 == spec.height)
    assert boundary_bottom - access_top == ROUTING_BAND_HEIGHT - DEVICE_GATE_CUT_HEIGHT
    assert boundary_bottom - access_top >= GATE_CUT_MIN_SPACE


def test_gate_strap_and_rail_heights_match_their_named_constants():
    spec = FinFETSpec(fingers=2)
    cell = build_finfet(spec)
    strap = next(
        p
        for p in polys(cell, "LIG")
        if (p[1] + p[3]) / 2 == spec.gate_contact_y() and p[0] >= spec.device_x0
    )
    # The body rail shares this Y centre, so select on X as well.
    gate_pin = next(
        p
        for p in polys(cell, "M1")
        if (p[1] + p[3]) / 2 == spec.gate_contact_y() and p[0] >= spec.device_x0
    )

    assert strap[3] - strap[1] == GATE_LIG_HEIGHT
    assert gate_pin[2] - gate_pin[0] == M1_WIDTH
    assert (gate_pin[0] + gate_pin[2]) / 2 == spec.sd_xs[0] + GATE_V0_DX


def test_cli_writes_a_gds(tmp_path):
    out = tmp_path / "dev.gds"
    main(["--fins", "2", "--fingers", "2", "--vt", "lvt", "--out", str(out)])
    assert out.exists()
    lib = gdspy.GdsLibrary(infile=str(out))
    assert "nmos_lvt_fin_221" in lib.cells
    assert polys(lib.cells["nmos_lvt_fin_221"], "LVT")
