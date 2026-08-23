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
        CONTACT_SIZE,
        DEVICE_GATE_CUT_HEIGHT,
        GATE_LIG_HEIGHT,
        LI_RAIL_HEIGHT,
        POLY_OVERHANG,
        SD_BAR_WIDTH,
        SELECT_Y_ENC,
    )
    from chipforge_asap7.layout import FIN_PITCH, FIN_WIDTH, GATE_WIDTH, LAYERS

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
