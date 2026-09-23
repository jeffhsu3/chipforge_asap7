"""Parsers, bounds and geometry correlation against the released INVxp33.

Skips when the public ASAP7 standard-cell release is not on this machine;
set ``ASAP7_REQUIRE_TOOLS=1`` to turn those skips into failures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chipforge_asap7.verification.reference import (
    parse_liberty_reference,
    parse_xact_reference,
)


def test_released_invxp33_xact_metrics_when_available(
    external_tool, released_library: Path
):
    path = released_library / "CDL/xAct3D_extracted/asap7sc7p5t_28_R.sp"
    external_tool(path.is_file(), "the released xACT reference is not available")
    metrics = parse_xact_reference(path, "INVxp33_ASAP7_75t_R")
    assert metrics["coupling_capacitor_count"] == 3
    assert metrics["coupling_capacitance_farad"] == pytest.approx(3.856488e-17)
    assert metrics["maximum_resistor_ohm"] == pytest.approx(19.3796)
    assert metrics["resistor_count"] == 33


def test_released_invxp33_liberty_metrics_when_available(
    external_tool, released_library: Path
):
    path = released_library / "LIB/NLDM/asap7sc7p5t_INVBUF_RVT_TT_nldm_220122.lib.7z"
    external_tool(path.is_file(), "the released Liberty reference is not available")
    metrics = parse_liberty_reference(path, "INVxp33_ASAP7_75t_R")
    assert metrics["input_capacitance_ff"] == pytest.approx(0.275723)
    assert metrics["rise_delay_ps"] == pytest.approx(9.11704)
    assert metrics["fall_delay_ps"] == pytest.approx(7.83205)
    assert metrics["selected_output_load_ff"] == pytest.approx(0.36)


# ── Dimensional correlation against the released standard cell ────────────────
#
# The xACT/Liberty parsers above check released *metrics*.  Nothing checked that
# the geometry this package draws matches the geometry ASAP7 ships, even though
# the release is the strongest oracle available and needs no EDA tool to read.
# These assertions are what caught a false positive during review: the 4 nm SDT
# overhang past ACTIVE looks like a bug until you find it in INVxp33 too.
_RELEASED_GDS = "GDS/asap7sc7p5t_28_R_220121a.gds"
_RELEASED_CELL = "INVxp33_ASAP7_75t_R"


def _released_boxes(path: Path, cell_name: str, layer: int, datatype: int = 0):
    """Bounding boxes on one layer of a released cell, converted to nm."""
    gdspy = pytest.importorskip("gdspy")
    cell = gdspy.GdsLibrary(infile=str(path)).cells[cell_name]
    polygons = cell.get_polygons(by_spec=True).get((layer, datatype), [])
    return sorted(
        (
            round(points[:, 0].min() * 1000),
            round(points[:, 1].min() * 1000),
            round(points[:, 0].max() * 1000),
            round(points[:, 1].max() * 1000),
        )
        for points in polygons
    )


def test_generated_finfet_stack_matches_released_invxp33_dimensions(
    external_tool, released_library: Path, boxes_on
):
    """Our one-fin device must draw the same stack ASAP7 released.

    INVxp33 is a one-fin nFET / one-fin pFET pair on the same grid, so its
    ACTIVE, SDT, LISD, V0, GCUT and LIG shapes are directly comparable to a
    `FinFETSpec(fins=1)` device even though the surrounding cell differs.
    """
    from chipforge_asap7.devices import FinFETSpec, build_finfet
    from chipforge_asap7.devices.finfet import (
        ACTIVE_ENC,
        DEVICE_GATE_CUT_HEIGHT,
        GATE_LIG_HEIGHT,
        LI_RAIL_HEIGHT,
        POLY_OVERHANG,
        SD_BAR_WIDTH,
        SELECT_Y_ENC,
    )
    from chipforge_asap7.layout import FIN_PITCH, FIN_WIDTH, GATE_WIDTH, LAYERS
    from chipforge_asap7.layout.rules import CONTACT_SIZE

    gds = released_library / _RELEASED_GDS
    external_tool(
        gds.is_file(), "the public ASAP7 standard-cell release is not available"
    )

    def released(name: str):
        return _released_boxes(
            gds, _RELEASED_CELL, LAYERS[name]["layer"], LAYERS[name]["datatype"]
        )

    spec = FinFETSpec(fins=1)
    cell = build_finfet(spec)

    def ours(name: str):
        return [tuple(round(v) for v in box) for box in boxes_on(cell, name)]

    # ACTIVE: the released nFET diffusion is 70 x 27 nm, 27 nm off the row edge.
    ref_active = released("ACTIVE")
    ref_nfet_active = ref_active[0]
    assert ref_nfet_active[2] - ref_nfet_active[0] == 70
    assert ref_nfet_active[3] - ref_nfet_active[1] == FIN_PITCH
    assert ref_nfet_active[1] == SELECT_Y_ENC

    our_active = next(b for b in ours("ACTIVE") if b[0] >= spec.device_x0)
    assert our_active[2] - our_active[0] == ref_nfet_active[2] - ref_nfet_active[0]
    assert our_active[3] - our_active[1] == ref_nfet_active[3] - ref_nfet_active[1]
    assert our_active[1] == ref_nfet_active[1]

    # SDT/LISD bars: 24 nm wide, and both overhang ACTIVE by 4 nm on the outside.
    ref_sdt = [b for b in released("SDT") if b[1] == SELECT_Y_ENC]
    assert all(b[2] - b[0] == SD_BAR_WIDTH for b in ref_sdt)
    assert ref_nfet_active[0] - ref_sdt[0][0] == 4
    assert ref_sdt[-1][2] - ref_nfet_active[2] == 4

    our_sdt = [b for b in ours("SDT") if b[0] >= spec.device_x0]
    assert all(b[2] - b[0] == SD_BAR_WIDTH for b in our_sdt)
    assert our_active[0] - our_sdt[0][0] == 4
    assert our_sdt[-1][2] - our_active[2] == 4

    # Drain V0: an 18 nm square sitting on the lower ACTIVE edge, zero enclosure.
    ref_drain_v0 = next(b for b in released("V0") if b[1] == ref_nfet_active[1])
    assert ref_drain_v0[2] - ref_drain_v0[0] == CONTACT_SIZE
    assert ref_drain_v0[3] - ref_drain_v0[1] == CONTACT_SIZE

    our_drain_v0 = next(b for b in ours("V0") if b[1] == our_active[1])
    assert our_drain_v0[2] - our_drain_v0[0] == CONTACT_SIZE
    assert our_drain_v0[3] - our_drain_v0[1] == CONTACT_SIZE
    assert our_drain_v0[1] + CONTACT_SIZE / 2 == spec.drain_contact_y()

    # GCUT height, LI rail height and the gate strap height are all released
    # dimensions, not choices this package is free to make.
    assert {b[3] - b[1] for b in released("GATE_CUT")} == {DEVICE_GATE_CUT_HEIGHT}
    assert {b[3] - b[1] for b in ours("GATE_CUT")} == {DEVICE_GATE_CUT_HEIGHT}

    ref_lig = released("LIG")
    assert {b[3] - b[1] for b in ref_lig} == {LI_RAIL_HEIGHT, GATE_LIG_HEIGHT}
    assert {b[3] - b[1] for b in ours("LIG")} == {LI_RAIL_HEIGHT, GATE_LIG_HEIGHT}

    # GATE: 20 nm poly, run past the placement boundary rather than ended on it.
    assert {b[2] - b[0] for b in released("GATE")} == {GATE_WIDTH}
    assert {b[2] - b[0] for b in ours("GATE")} == {GATE_WIDTH}
    assert {b[1] for b in released("GATE")} == {-POLY_OVERHANG}
    assert {b[1] for b in ours("GATE")} == {-POLY_OVERHANG}

    # FIN grid: 7 nm stripes on a 27 nm pitch, anchored ACTIVE_ENC off the edge.
    ref_fins = released("FIN")
    assert {b[3] - b[1] for b in ref_fins} == {FIN_WIDTH}
    assert ref_fins[0][1] == ACTIVE_ENC
    assert ref_fins[1][1] - ref_fins[0][1] == FIN_PITCH

    our_fins = ours("FIN")
    assert {b[3] - b[1] for b in our_fins} == {FIN_WIDTH}
    assert our_fins[0][1] == ACTIVE_ENC
    assert our_fins[1][1] - our_fins[0][1] == FIN_PITCH


# ── Dimensional correlation against the released decoder inverter ─────────────
#
# `dec_inv_62f_halved_AND` is the only tapless, stacked-band, shared-diffusion
# cell in the released collateral, and the one
# `chipforge_asap7.devices.inverter` is modelled on.  Comparing against it is
# what pins the band arithmetic and the abutting placement to something ASU
# actually taped out rather than to numbers this package chose.
_SRAM_CELL = "dec_inv_62f_halved_AND"


def test_generated_inverter_matches_the_released_decoder_inverter(
    external_tool, released_sram_gds: Path, boxes_on
):
    """Diffusion, implant, wells and contacts, shape for shape.

    Every layer here has to match exactly.  The three that do not are checked
    separately below, because each is a deliberate difference rather than a
    drift: the fin grid, the gate stripes and the gate cuts.
    """
    from chipforge_asap7.devices import InverterSpec, build_inverter
    from chipforge_asap7.layout import LAYERS

    external_tool(
        released_sram_gds.is_file(),
        "the public ASAP7 SRAM release is not available",
    )
    spec = InverterSpec(rows=((18, 18), (13, 13)), fingers=2)
    cell = build_inverter(spec)

    def released(name: str):
        return sorted(
            set(
                _released_boxes(
                    released_sram_gds,
                    _SRAM_CELL,
                    LAYERS[name]["layer"],
                    LAYERS[name]["datatype"],
                )
            )
        )

    def ours(name: str):
        return sorted({tuple(round(v) for v in box) for box in boxes_on(cell, name)})

    assert (spec.width, spec.height) == (162, 1890)
    for name in ("ACTIVE", "SDT", "LISD", "NWELL", "BOUNDARY"):
        assert ours(name) == released(name), name

    # The gate contact: a 22 nm LIG pad on the poly at each n/p seam.  Ours is
    # a strap across every active finger rather than a single pad -- see
    # `test_inverter.test_every_active_finger_reaches_the_input_strap` -- so
    # only its height and its Y are comparable.
    ref_lig = released("LIG")
    our_lig = ours("LIG")
    assert len(our_lig) == len(ref_lig) == len(spec.rows)
    for ref, got in zip(ref_lig, our_lig):
        assert (got[1], got[3]) == (ref[1], ref[3])
        assert got[0] <= ref[0] and got[2] >= ref[2]

    # Implant tiles the same bands; the released cell is a nanometre
    # asymmetric at its own edges, ours is not.
    for name in ("NSELECT", "PSELECT"):
        assert sorted((b[1], b[3]) for b in ours(name)) == sorted(
            (b[1], b[3]) for b in released(name)
        ), name

    # Contacts: same columns, same rows, except that the released cell places
    # its topmost band's via row 3 nm off the ACTIVE edge where its other
    # three bands sit flush.  Ours are flush throughout.
    ref_v0 = released("V0")
    our_v0 = ours("V0")
    assert len(our_v0) == len(ref_v0)
    assert {b[0] for b in our_v0} == {b[0] for b in ref_v0}
    assert sum(a != b for a, b in zip(sorted(our_v0), sorted(ref_v0))) == 3


def test_our_inverter_diverges_from_the_released_cell_only_where_intended(
    external_tool, released_sram_gds: Path, boxes_on
):
    """Three deliberate differences, each of which the released cell needs.

    They are asserted rather than tolerated so that a fourth one cannot appear
    silently.
    """
    from chipforge_asap7.devices import InverterSpec, build_inverter
    from chipforge_asap7.layout import LAYERS

    external_tool(
        released_sram_gds.is_file(),
        "the public ASAP7 SRAM release is not available",
    )
    spec = InverterSpec(rows=((18, 18), (13, 13)), fingers=2)
    cell = build_inverter(spec)

    def released(name: str):
        return sorted(
            set(
                _released_boxes(
                    released_sram_gds,
                    _SRAM_CELL,
                    LAYERS[name]["layer"],
                    LAYERS[name]["datatype"],
                )
            )
        )

    def ours(name: str):
        return sorted({tuple(round(v) for v in box) for box in boxes_on(cell, name)})

    # 1. FIN.  The released grid stops at the tile edge, leaving its own
    #    8 nm ACTIVE overhang uncovered -- an ACTIVE.FIN.EX.1 the neighbour
    #    happens to fix.  Ours runs past the overhang so the cell stands alone.
    ref_fin, our_fin = released("FIN"), ours("FIN")
    assert len(our_fin) == len(ref_fin)
    assert [(b[1], b[3]) for b in our_fin] == [(b[1], b[3]) for b in ref_fin]
    assert our_fin[0][0] < ref_fin[0][0] and our_fin[0][2] > ref_fin[0][2]

    # 2. GATE.  The released cell splits each stripe at the row boundary, with
    #    an 11 nm overlap that makes it one net anyway.  One stripe is simpler
    #    and identical electrically.
    ref_gate, our_gate = released("GATE"), ours("GATE")
    assert len(our_gate) * 2 == len(ref_gate)
    assert {(b[0], b[2]) for b in our_gate} == {(b[0], b[2]) for b in ref_gate}
    assert all(b[1] < 0 and b[3] > spec.height for b in our_gate)

    # 3. GCUT.  The released cell cuts only at the top, because its poly runs
    #    on into the bank below it.  A cell that has to be correct on its own
    #    cuts both ends.
    assert [(b[1] + b[3]) // 2 for b in ours("GATE_CUT")] == [0, spec.height]
    assert len(released("GATE_CUT")) == 1


# ── Dimensional correlation against the released tap and filler ───────────────
#
# These two are the only cells this package draws that are *identical* to
# released ASAP7 collateral rather than merely consistent with it, so the
# comparison is an equality rather than a set of dimensional claims.  A
# three-fin/three-fin stack is two 135 nm bands, which is exactly the released
# 270 nm standard-cell row.
def test_generated_tap_and_filler_match_the_released_cells_shape_for_shape(
    external_tool, released_library: Path, boxes_on
):
    from chipforge_asap7.devices import RowStack, RowSupportSpec, build_row_support
    from chipforge_asap7.layout import LAYERS

    gds = released_library / _RELEASED_GDS
    external_tool(
        gds.is_file(), "the public ASAP7 standard-cell release is not available"
    )

    stack = RowStack(rows=((3, 3),))
    assert stack.height == 270

    for kind, released_cell in (
        ("tap", "TAPCELL_ASAP7_75t_R"),
        ("filler", "FILLER_ASAP7_75t_R"),
    ):
        spec = RowSupportSpec(stack=stack, kind=kind)
        cell = build_row_support(spec)
        assert (spec.width, spec.height) == (108, 270)

        drawn = {
            name
            for name in LAYERS
            if not name.endswith("_PIN") and boxes_on(cell, name)
        }
        # CA is an alias of LIG on the same layer number, so it is not a
        # separate shape set; compare the layers, not the names.
        specs = {(LAYERS[name]["layer"], LAYERS[name]["datatype"]) for name in drawn}
        released_specs = {
            key
            for key in _released_layer_specs(gds, released_cell)
            if key[1] != 251  # pin text, which the released cells carry as labels
        }
        assert specs == released_specs, kind

        for layer, datatype in sorted(specs):
            ours = sorted(
                {
                    tuple(round(v) for v in b)
                    for b in _boxes_on_spec(cell, layer, datatype)
                }
            )
            theirs = sorted(set(_released_boxes(gds, released_cell, layer, datatype)))
            assert ours == theirs, f"{kind} layer {layer}/{datatype}"


def _released_layer_specs(path: Path, cell_name: str):
    gdspy = pytest.importorskip("gdspy")
    cell = gdspy.GdsLibrary(infile=str(path)).cells[cell_name]
    return {(int(spec[0]), int(spec[1])) for spec in cell.get_polygons(by_spec=True)}


def _boxes_on_spec(cell, layer: int, datatype: int):
    for polygon_set in cell.polygons:
        for points, gds_layer, gds_datatype in zip(
            polygon_set.polygons, polygon_set.layers, polygon_set.datatypes
        ):
            if (gds_layer, gds_datatype) == (layer, datatype):
                xs, ys = points[:, 0], points[:, 1]
                yield (xs.min(), ys.min(), xs.max(), ys.max())


# ── Dimensional correlation against the released post-decode NAND ─────────────
#
# `dec_nand_12f_12f_for_and_size_reduced_post_decode_P1N1` is the decoder's
# last NAND, drawn in the same abutting style as the inverter above.  Its
# routing is deliberately not compared: the released cell relies on its parent
# for the VSS ties of its edge columns and for joining its two B gates on M2,
# both of which `chipforge_asap7.devices.nand` draws itself.
_SRAM_NAND = "dec_nand_12f_12f_for_and_size_reduced_post_decode_P1N1"


def test_generated_nand_matches_the_released_post_decode_nand(
    external_tool, released_sram_gds: Path, boxes_on
):
    """Diffusion, wells, implant, gates and contact columns, shape for shape."""
    from chipforge_asap7.devices import NandSpec, build_nand
    from chipforge_asap7.layout import LAYERS

    external_tool(
        released_sram_gds.is_file(),
        "the public ASAP7 SRAM release is not available",
    )
    spec = NandSpec(rows=((14, 7),), fingers=2)
    cell = build_nand(spec)

    def released(name: str):
        return sorted(
            set(
                _released_boxes(
                    released_sram_gds,
                    _SRAM_NAND,
                    LAYERS[name]["layer"],
                    LAYERS[name]["datatype"],
                )
            )
        )

    def ours(name: str):
        return sorted({tuple(round(v) for v in box) for box in boxes_on(cell, name)})

    assert (spec.width, spec.height) == (216, 675)
    for name in ("ACTIVE", "NWELL", "BOUNDARY", "GATE"):
        assert ours(name) == released(name), name
    for name in ("NSELECT", "PSELECT"):
        assert sorted((b[1], b[3]) for b in ours(name)) == sorted(
            (b[1], b[3]) for b in released(name)
        ), name

    # The same columns are contacted in each band -- and the same ones are
    # not: the n band's series nodes carry no SDT in either cell.
    for band in spec.bands:
        lo, hi = band.active_span

        def columns(boxes, lo=lo, hi=hi):
            return sorted({(b[0], b[2]) for b in boxes if b[1] < hi and b[3] > lo})

        assert columns(ours("SDT")) == columns(released("SDT")), band.flavor

    # Gate contacts: one V0 on the seam at the same three gates, and LIG pads
    # over the same gate runs (the released cell draws one of its pads twice).
    seam = spec.seam_y

    def on_seam(boxes):
        return sorted(b[0] for b in boxes if b[1] < seam < b[3])

    def pads(boxes):
        return {(b[0], b[2]) for b in boxes if b[1] < seam < b[3]}

    assert on_seam(ours("V0")) == on_seam(released("V0")) == [18, 72, 180]
    assert pads(ours("LIG")) <= pads(released("LIG"))
    assert len(pads(ours("LIG"))) == 3

    # As with the inverter: the released cell cuts its poly only at the rail
    # it does not share with the cell below; a standalone cell cuts both.
    assert [(b[1] + b[3]) // 2 for b in ours("GATE_CUT")] == [0, spec.height]
    assert len(released("GATE_CUT")) == 1
