"""Geometry of the parametric bitline leaf: no external tools needed."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import gdspy
import pytest

from chipforge_asap7.devices import (
    BITLINE_MUX_PINS,
    BitlineMuxSpec,
    bitline_mux_group_pins,
    build_bitline_mux,
    build_bitline_mux_group,
)
from chipforge_asap7.layout.grid import GATE_PITCH


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def test_default_leaf_is_half_an_8t_bitcell_row():
    spec = BitlineMuxSpec()
    assert spec.height == 297
    assert spec.width == 702
    assert spec.cell_name == "blmux_3n3p_h135x162_s0of4"
    assert [net for _, net in spec.rails] == ["VSS", "VDD"]


@pytest.mark.parametrize("selects", [1, 2, 4, 8, 16])
def test_the_leaf_widens_by_one_track_a_side_per_select(selects):
    spec = BitlineMuxSpec(selects=selects)
    xs = sorted(spec.track_x.values())
    assert len(xs) == 3 + 2 * selects
    # 18 nm M3 on an 18 nm space.
    assert min(b - a for a, b in pairwise(xs)) >= 36
    # A select track takes a V2 on a 34 nm M2 pad, which keeps half an M2 space
    # from the cell edge; one gate pitch less on either side would not.
    left, right = xs[0] - 17, spec.width - xs[-1] - 17
    assert left >= 9 and right >= 9
    assert left - GATE_PITCH < 9 and right - GATE_PITCH < 9


def test_every_leaf_of_a_group_puts_its_tracks_in_the_same_place():
    """That is what lets them stack into a mux with no routing."""
    specs = [BitlineMuxSpec(selects=4, select=i) for i in range(4)]
    assert len({spec.cell_name for spec in specs}) == 4
    assert all(spec.track_x == specs[0].track_x for spec in specs)
    assert all(spec.width == specs[0].width for spec in specs)
    for i, spec in enumerate(specs):
        assert spec.pin_positions["YSEL"][1][0] == spec.track_x[f"YSEL[{i}]"]
        assert spec.pin_positions["YSELN"][1][0] == spec.track_x[f"YSELN[{i}]"]


def test_shared_tracks_run_the_full_height_on_m3(boxes_on):
    spec = BitlineMuxSpec(selects=2, select=1)
    cell = build_bitline_mux(spec, lib=_library())
    full_height = {
        (x0 + x1) / 2
        for x0, y0, x1, y1 in boxes_on(cell, "M3")
        if (y0, y1) == (0, spec.height)
    }
    assert full_height == set(spec.track_x.values())
    assert boxes_on(cell, "BOUNDARY") == [(0, 0, spec.width, spec.height)]


def test_bitlines_enter_on_m2_at_the_left_edge(boxes_on):
    spec = BitlineMuxSpec()
    cell = build_bitline_mux(spec, lib=_library())
    at_edge = {(y0 + y1) / 2 for x0, y0, _, y1 in boxes_on(cell, "M2") if x0 == 0}
    assert at_edge == {spec.tracks_y["BL"], spec.tracks_y["BLN"]}
    assert spec.tracks_y["BLN"] - spec.tracks_y["BL"] == 36


def test_pins_and_netlist_agree():
    spec = BitlineMuxSpec(n_fins=2, p_fins=4, band_height=(135, 189))
    assert set(spec.pin_positions) == set(BITLINE_MUX_PINS)
    nets = {
        net
        for _, drain, gate, source, _, _ in spec.devices
        for net in (drain, gate, source)
    }
    assert nets == set(BITLINE_MUX_PINS) - {"VSS"}  # VSS is the n body only
    fins = {name: count for name, *_, count in spec.devices}
    assert fins == {"MNT": 2, "MNC": 2, "MPT": 4, "MPC": 4, "MPPT": 4, "MPPC": 4}


def test_group_stacks_one_leaf_per_select(boxes_on):
    spec = BitlineMuxSpec(selects=4)
    library = _library()
    group = build_bitline_mux_group(spec, lib=library)
    assert boxes_on(group, "BOUNDARY") == [(0, 0, spec.width, 4 * spec.height)]
    placed = sorted(
        (ref.ref_cell.name, ref.origin[1], bool(ref.x_reflection))
        for ref in group.references
    )
    assert [(y, flipped) for _, y, flipped in placed] == [
        (0, False),
        (594, True),  # flipped about its own bottom rail, so it occupies 297..594
        (594, False),
        (1188, True),
    ]
    assert len(bitline_mux_group_pins(spec)) == 4 * 4 + 5
    labels = {label.text for label in group.labels}
    assert labels == set(bitline_mux_group_pins(spec))
    assert (
        build_bitline_mux_group(replace(spec, select=3), lib=_library()).name
        == group.name
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"selects": 0}, "selects must be >= 1"),
        ({"select": 4}, "select must be 0..3"),
        ({"band_height": (108, 162), "n_fins": 2}, "no room for both bitline"),
        ({"band_height": (135, 135), "p_fins": 2}, "no room for the select tie"),
    ],
)
def test_undrawable_specs_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BitlineMuxSpec(**kwargs)


def test_select_must_be_an_integer():
    with pytest.raises(TypeError, match="select must be an integer"):
        BitlineMuxSpec(select=True)


# ── Two rows, and bitlines that arrive where an array puts them ───────────────
PORT_A = BitlineMuxSpec(rows=2, bitline_entry=(348.5, 245.5))  # 8T port A, on M2
PORT_B = BitlineMuxSpec(rows=2, bitline_entry=(510, 78), bitline_layer="M4")


def _flat_boxes(cell, layer_name):
    from chipforge_asap7.layout import LAYERS

    key = (LAYERS[layer_name]["layer"], LAYERS[layer_name]["datatype"])
    return sorted(
        (
            float(p[:, 0].min()),
            float(p[:, 1].min()),
            float(p[:, 0].max()),
            float(p[:, 1].max()),
        )
        for p in cell.get_polygons(by_spec=True).get(key, [])
    )


def test_two_rows_are_a_whole_8t_bitcell_row_with_twice_the_fins():
    one, two = BitlineMuxSpec(), BitlineMuxSpec(rows=2)
    assert (two.height, two.row_height) == (594, 297)
    assert [net for _, net in two.rails] == ["VSS", "VDD", "VSS"]
    assert two.width == one.width + 54  # two entry columns cost one gate pitch at 4:1
    assert {name: fins for name, *_, fins in two.devices} == {
        name: 2 * fins for name, *_, fins in one.devices
    }
    assert "nfin=6" in two.netlist() and "nfin=3" not in two.netlist()
    assert two.cell_name == "blmux_3n3p_h135x162_r2_s0of4"
    assert PORT_A.cell_name == "blmux_3n3p_h135x162_r2_s0of4_m2in3485x2455"
    assert two.stack.height == two.height


def test_upper_row_is_the_lower_one_mirrored_about_the_vdd_rail():
    spec = BitlineMuxSpec(rows=2)
    cell = build_bitline_mux(spec, lib=_library())
    assert [(ref.origin[1], bool(ref.x_reflection)) for ref in cell.references] == [
        (0, False),
        (594, True),
    ]
    for layer in ("ACTIVE", "GATE", "M1", "LISD", "V0"):
        boxes = _flat_boxes(cell, layer)
        mirrored = sorted((x0, 594 - y1, x1, 594 - y0) for x0, y0, x1, y1 in boxes)
        assert boxes == mirrored, layer
    # The shared M3 tracks still run the full height, now as two halves.
    for x_track in spec.track_x.values():
        spans = [
            (y0, y1) for x0, y0, x1, y1 in _flat_boxes(cell, "M3") if x0 == x_track - 9
        ]
        assert spans == [(0, 297), (297, 594)]


@pytest.mark.parametrize("spec", [PORT_A, PORT_B], ids=["M2", "M4"])
def test_entry_columns_take_each_bitline_to_its_track_in_both_rows(spec):
    cell = build_bitline_mux(spec, lib=_library())
    layer = spec.bitline_layer
    half = 9 if layer == "M2" else 12
    for net in ("BL", "BLN"):
        x_col, y_in = spec.entry_x[net], spec.entry_y[net]
        # Outside every select track, a full M3 pitch from the nearest.
        assert x_col <= min(spec.track_x.values()) - 36
        # Reaches the left edge on the array's layer, at the array's height.
        stubs = [
            b
            for b in _flat_boxes(cell, layer)
            if b[0] == 0 and (b[1], b[3]) == (y_in - half, y_in + half)
        ]
        assert len(stubs) == 1 and stubs[0][2] >= x_col + 9
        # One column spanning the entry and both rows' tracks.
        (column,) = [b for b in _flat_boxes(cell, "M3") if b[0] == x_col - 9]
        for y in (*spec.track_ys(net), y_in):
            assert column[1] <= y - 9 and y + 9 <= column[3]
        vias = {(b[1] + b[3]) / 2 for b in _flat_boxes(cell, "V2") if b[0] == x_col - 9}
        assert vias == set(spec.track_ys(net)) | ({y_in} if layer == "M2" else set())
        assert spec.pin_positions[net] == (layer, (17, y_in))
    if layer == "M4":
        v3 = _flat_boxes(cell, "V3")
        assert [(b[2] - b[0], b[3] - b[1]) for b in v3] == [(18, 24), (18, 24)]
        assert all(
            b[2] % 24 == 0 for b in _flat_boxes(cell, "M4")
        )  # the deck's M4 x grid
    # No bitline M2 at the edge but the entries themselves.
    at_edge = {(b[1] + b[3]) / 2 for b in _flat_boxes(cell, "M2") if b[0] == 0}
    assert at_edge == (set(spec.entry_y.values()) if layer == "M2" else set())


def test_a_group_flips_alternate_leaves_as_an_array_flips_its_rows():
    library = _library()
    group = build_bitline_mux_group(PORT_A, lib=library)
    assert group.name == "blmux_3n3p_h135x162_r2_group4_m2in3485x2455"
    where = {label.text: label.position[1] for label in group.labels}
    # sram_cell_8t rows: BLA at 348.5 and BLAN at 245.5 from the bottom of an upright row.
    assert [where[f"BL[{i}]"] for i in range(4)] == [
        348.5,
        1188 - 348.5,
        1188 + 348.5,
        2376 - 348.5,
    ]
    assert [where[f"BLN[{i}]"] for i in range(4)] == [
        245.5,
        1188 - 245.5,
        1188 + 245.5,
        2376 - 245.5,
    ]
    rails = sorted(
        (label.position[1], label.text)
        for label in group.labels
        if label.text in ("VDD", "VSS")
    )
    assert rails == [(297 * k, "VDD" if k % 2 else "VSS") for k in range(9)]


def test_a_grid_offset_moves_the_landings_against_the_flip():
    # Half a fin pitch up puts the leaf's fins on an 8T array's grid; each
    # row's bitlines still come in where the array puts them, which is
    # 13.5 nm lower on the leaf's own edge for upright rows and 13.5 higher
    # for flipped ones.
    spec = replace(PORT_A, grid_offset=13.5)
    assert spec.entry_y == {"BL": 335.0, "BLN": 232.0}
    assert (
        spec.for_row(0).bitline_entry == (335.0, 232.0)
        and spec.for_row(0).grid_offset == 0
    )
    assert (
        spec.for_row(1).bitline_entry == (362.0, 259.0) and spec.for_row(1).select == 1
    )
    assert spec.for_row(2) == replace(spec.for_row(0), select=2)
    library = _library()
    group = build_bitline_mux_group(spec, lib=library)
    assert group.name == "blmux_3n3p_h135x162_r2_group4_m2in3485x2455g135"
    where = {label.text: label.position[1] for label in group.labels}
    # In the array's frame (13.5 nm below the group's) these are the rows' bars.
    assert [where[f"BL[{i}]"] + 13.5 for i in range(4)] == [
        348.5,
        1188 - 348.5,
        1188 + 348.5,
        2376 - 348.5,
    ]
    assert [where[f"BLN[{i}]"] + 13.5 for i in range(4)] == [
        245.5,
        1188 - 245.5,
        1188 + 245.5,
        2376 - 245.5,
    ]
    # Every leaf's landing is a real M2 bar at its edge, at that leaf's own entry.
    for i, leaf in enumerate(sorted(group.get_dependencies(), key=lambda c: c.name)):
        bars = {
            (x0, y0, x1, y1)
            for (x0, y0), (x1, y1) in (
                poly.get_bounding_box() for poly in leaf.polygons if poly.layers == [20]
            )
            if x0 == 0
        }
        wanted = spec.for_row(i).entry_y
        assert {round((y0 + y1) / 2, 1) for _, y0, _, y1 in bars} >= {
            wanted["BL"],
            wanted["BLN"],
        }
    with pytest.raises(ValueError, match="needs a bitline_entry"):
        BitlineMuxSpec(grid_offset=13.5)


def test_one_row_can_take_an_entry_too():
    spec = BitlineMuxSpec(bitline_entry=(224, 260))
    assert spec.has_entry_columns and spec.height == 297
    assert not BitlineMuxSpec().has_entry_columns


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"rows": 3}, "rows must be 1 or 2"),
        ({"bitline_layer": "M4"}, "need a bitline_entry"),
        ({"bitline_layer": "M6", "bitline_entry": (100, 200)}, "must be M2 or M4"),
        ({"bitline_entry": (100, 200, 300)}, "is \\(y_BL, y_BLN\\)"),
        ({"bitline_entry": (290, 245.5)}, "outside the 297 nm leaf"),
        ({"rows": 2, "bitline_entry": (348.5, 330)}, "within 36 nm of other M2"),
        (
            {"rows": 2, "bitline_entry": (348.5, 100)},
            "within 36 nm of other M2",
        ),  # BL track, 80
        (
            {"rows": 2, "bitline_entry": (160, 245.5)},
            "within 36 nm of other M2",
        ),  # YSEL tie, 152
        ({"rows": 2, "bitline_entry": (60, 245.5)}, "crowds its own track"),
        (
            {"rows": 2, "bitline_entry": (510, 480), "bitline_layer": "M4"},
            "need 48 nm between",
        ),
    ],
)
def test_entries_that_cannot_be_drawn_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BitlineMuxSpec(**kwargs)


def test_an_entry_on_the_track_itself_is_allowed():
    spec = BitlineMuxSpec(rows=2, bitline_entry=(80, 116))
    assert spec.entry_y == {"BL": 80, "BLN": 116}
