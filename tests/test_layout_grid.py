"""Grid primitives must reproduce the geometry the chipforge NOR-ROM flow taped out."""

import gdspy
import pytest

from chipforge_asap7.layout import (
    COLUMN_PITCH,
    COLUMN_X_SHIFT,
    FIN_PITCH,
    GATE_PITCH,
    LAYERS,
    STD_CELL_HEIGHT,
    WORDLINE_PITCH,
    centered_fin_ys,
    column_gate_track,
    column_x,
    draw_fin_grid,
    draw_gate_cuts,
    draw_gate_grid,
    fin_ys,
    gate_center_x,
    gate_track_xs,
    row_boundary_ys,
    snap_to_fin,
    snap_to_gate,
    spans_excluding,
)


@pytest.fixture
def cell():
    return gdspy.Cell("TEST", exclude_from_current=True)


def _polys(cell, layer_name):
    key = (LAYERS[layer_name]["layer"], LAYERS[layer_name]["datatype"])
    return cell.get_polygons(by_spec=True).get(key, [])


def test_grid_constants_are_commensurate():
    assert WORDLINE_PITCH == 3 * FIN_PITCH  # 2 active fins + 1 isolation
    assert COLUMN_PITCH == 2 * GATE_PITCH  # bitcell is 2 CPP wide
    assert STD_CELL_HEIGHT % FIN_PITCH == 0


def test_fin_grid_matches_compiler_loop(cell):
    """rom_compiler draws `range(int(height / FIN_P) + 1)` fins from y=0."""
    height = 16 * WORDLINE_PITCH
    n = draw_fin_grid(cell, 0, 1000, 0, height)

    assert n == int(height / FIN_PITCH) + 1
    ys = sorted(p[:, 1].min() for p in _polys(cell, "FIN"))
    assert ys == [i * FIN_PITCH for i in range(n)]
    # Last fin covers the top of the array, so abutted rows never run dry.
    assert ys[-1] >= height - FIN_PITCH


def test_fin_stripes_span_requested_x(cell):
    draw_fin_grid(cell, 300, 900, 0, 81)
    for poly in _polys(cell, "FIN"):
        assert poly[:, 0].min() == 300
        assert poly[:, 0].max() == 900


def test_centered_fin_ys_leaves_isolation():
    ys = centered_fin_ys(WORDLINE_PITCH, fins=2)
    assert ys == [13 + 0 * FIN_PITCH, 13 + 1 * FIN_PITCH]
    # An isolation gap of >= one fin pitch above and below the active block.
    assert ys[0] >= FIN_PITCH // 2
    assert WORDLINE_PITCH - ys[-1] >= FIN_PITCH // 2


def test_centered_fin_ys_rejects_overfull_cell():
    with pytest.raises(ValueError):
        centered_fin_ys(WORDLINE_PITCH, fins=3)
    with pytest.raises(ValueError):
        centered_fin_ys(100, fins=2)  # not a fin-pitch multiple


def test_gate_grid_matches_compiler_loop(cell):
    width = 32 * COLUMN_PITCH
    n = draw_gate_grid(cell, 500, 500 + width, -1200, 1296)

    assert n == int(width / GATE_PITCH) + 1
    xs = sorted(p[:, 0].min() for p in _polys(cell, "GATE"))
    assert xs == [500 + j * GATE_PITCH for j in range(n)]
    for poly in _polys(cell, "GATE"):
        assert poly[:, 0].max() - poly[:, 0].min() == 20
        # Poly runs past the array bottom so no line terminates inside it.
        assert poly[:, 1].min() == -1200


def test_columns_land_on_gate_tracks():
    """COLUMN_X_SHIFT exists exactly so column b's gate is on track 2b+2."""
    for b in range(64):
        assert column_gate_track(b) == 2 * b + 2
    x0 = 1188  # a decoder-column left margin (multiple of COLUMN_PITCH)
    for b in range(8):
        track = column_gate_track(b, x0)
        assert gate_center_x(x0 + track * GATE_PITCH) == column_x(b, x0) - 36


def test_column_shift_is_load_bearing():
    with pytest.raises(ValueError):
        column_gate_track(0, shift=COLUMN_X_SHIFT + 1)


def test_column_x_matches_compiler_expression():
    x0 = 216
    for b in range(4):
        assert column_x(b, x0) == x0 + (b + 1) * COLUMN_PITCH + COLUMN_X_SHIFT


def test_snapping():
    assert snap_to_fin(0) == 0
    assert snap_to_fin(20) == FIN_PITCH
    assert snap_to_fin(40) == FIN_PITCH
    assert snap_to_gate(80) == GATE_PITCH  # 80 is 26 above 54, 28 below 108
    assert snap_to_gate(85) == 2 * GATE_PITCH
    assert snap_to_gate(26) == 0


def test_row_boundaries_sit_between_rows():
    ys = row_boundary_ys(4, WORDLINE_PITCH)
    assert ys == [0.5 * 81, 1.5 * 81, 2.5 * 81, 3.5 * 81, 4.5 * 81]
    # Every wordline (at (r+1)*pitch) is bracketed by two boundaries.
    for r in range(4):
        assert ys[r] < (r + 1) * WORDLINE_PITCH < ys[r + 1]


def test_spans_excluding_steps_over_buffer_strips():
    assert spans_excluding(0, 100, []) == [(0, 100)]
    assert spans_excluding(0, 100, [(40, 60)]) == [(0, 40), (60, 100)]
    # A strip flush against an edge yields no zero-width span.
    assert spans_excluding(0, 100, [(0, 20)]) == [(20, 100)]
    assert spans_excluding(0, 100, [(80, 100)]) == [(0, 80)]
    assert spans_excluding(0, 100, [(10, 30), (50, 70)]) == [
        (0, 10),
        (30, 50),
        (70, 100),
    ]


def test_gate_cuts_are_centered_and_split(cell):
    spans = spans_excluding(0, 1000, [(400, 500)])
    ys = row_boundary_ys(3, WORDLINE_PITCH)
    n = draw_gate_cuts(cell, spans, ys)

    assert n == len(spans) * len(ys) == 8
    for poly in _polys(cell, "GATE_CUT"):
        assert poly[:, 1].max() - poly[:, 1].min() == 24
        assert (poly[:, 1].max() + poly[:, 1].min()) / 2 in ys
        # Nothing lands on the buffer strip.
        assert not (400 < poly[:, 0].min() < 500)


def test_gate_cuts_skip_empty_spans(cell):
    assert draw_gate_cuts(cell, [(100, 100), (200, 150)], [40.5]) == 0
    assert _polys(cell, "GATE_CUT") == []


def test_gate_track_xs_covers_range():
    xs = gate_track_xs(0, 216)
    assert xs == [0, 54, 108, 162, 216]
    assert fin_ys(0, 81) == [0, 27, 54, 81]
