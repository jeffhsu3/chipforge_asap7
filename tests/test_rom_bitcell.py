"""The poly-wordline NOR-ROM: tile, strap row, programming sites and DRC.

The DRC half skips when gdscheck is missing; set ``ASAP7_REQUIRE_TOOLS=1`` to
make that a failure instead.
"""

from __future__ import annotations

import gdspy
import pytest

from chipforge_asap7.devices import (
    RomArraySpec,
    RomBitcellSpec,
    build_rom_array,
    build_rom_bitcell,
    build_rom_strap,
    build_rom_tap,
)
from chipforge_asap7.devices.rom_bitcell import (
    LATCHUP_REACH,
    STRAP_HEIGHT,
    TAP_HEIGHT,
    main,
)
from chipforge_asap7.layout import FIN_PITCH, GATE_PITCH, GATE_WIDTH


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


@pytest.mark.parametrize("fins", [1, 2, 3])
def test_pitches(fins):
    spec = RomBitcellSpec(fins=fins)
    assert spec.width == 3 * GATE_PITCH
    assert spec.word_pitch == 81
    assert spec.height == (fins + 1) * FIN_PITCH
    assert spec.active_span == (0, fins * FIN_PITCH)


def test_two_fins_is_an_81_nm_square_bit():
    spec = RomBitcellSpec()
    assert (spec.word_pitch, spec.height) == (81, 81)


def test_tile_sits_on_the_asap7_grids(boxes_on):
    """Gates on 27 + 54k (GATE.S.1), fins on 10 + 27k (FIN.S.1), S/D between gates."""
    spec = RomBitcellSpec(fins=2)
    cell = build_rom_bitcell(spec)
    for x0, _, x1, _ in boxes_on(cell, "GATE"):
        assert x1 - x0 == GATE_WIDTH
        assert ((x0 + x1) / 2 - GATE_PITCH / 2) % GATE_PITCH == 0
    for _, y0, _, _ in boxes_on(cell, "FIN"):
        assert (y0 - 10) % FIN_PITCH == 0
    for name, x in spec.sd_xs.items():
        assert x % GATE_PITCH == 0, name


def test_tile_has_no_gate_contact(boxes_on):
    """The wordline is the poly: no LIG, no gate cut, no V1 until programmed."""
    cell = build_rom_bitcell(RomBitcellSpec())
    for layer in ("LIG", "GATE_CUT", "V1"):
        assert boxes_on(cell, layer) == []


def test_sites_land_on_the_drain_stubs():
    spec = RomBitcellSpec()
    for word, drain in ((0, "Da"), (1, "Db")):
        x, y = spec.site(word)
        x0, x1 = spec.stubs[drain]
        assert (x - 9 - x0, x1 - x - 9) == (5, 5)  # V1.M1.EN.1 on both ends
        assert y == spec.contact_y
    with pytest.raises(IndexError):
        spec.site(2)


def test_array_places_one_v1_per_programmed_site(boxes_on):
    spec = RomArraySpec(6, 3, strap_every=2)
    _library()
    sites = {(0, 0), (3, 1), (5, 2)}
    cell = build_rom_array(spec, sites)
    vias = boxes_on(cell, "V1")
    assert len(vias) == len(sites)
    centres = {((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in vias}
    assert centres == {spec.site(w, b) for w, b in sites}


def test_rows_step_over_the_straps():
    spec = RomArraySpec(4, 5, RomBitcellSpec(fins=2), strap_every=2)
    assert spec.straps == 3
    assert spec.strap_ys == [0, 270, 540]
    assert [spec.row_y(b) for b in range(5)] == [108, 189, 378, 459, 648]
    assert spec.tap_y == 3 * STRAP_HEIGHT + 5 * 81
    assert spec.height == spec.tap_y + TAP_HEIGHT
    with pytest.raises(IndexError):
        spec.row_y(5)


def test_strap_contacts_every_gate_and_lands_wordlines_on_m3(boxes_on):
    """Three LIGs per tile, two wordline stacks up to V2, ISO down to the VSS track."""
    bitcell = RomBitcellSpec()
    strap = build_rom_strap(bitcell)
    ligs = boxes_on(strap, "LIG")
    assert len(ligs) == 3
    for x_gate in bitcell.gate_xs.values():
        assert any(x0 < x_gate - 10 and x_gate + 10 < x1 for x0, _, x1, _ in ligs)
    v2_xs = sorted((x0 + x1) / 2 for x0, _, x1, _ in boxes_on(strap, "V2"))
    spec = RomArraySpec(2, 1, bitcell)
    assert v2_xs == [spec.wordline_x(0), spec.wordline_x(1)]
    assert len(boxes_on(strap, "V1")) == 4  # two wordlines, VSS column, ISO


def test_tap_ties_p_plus_active_to_the_vss_column(boxes_on):
    """One 16 nm p+ tie per tile, on the VSS column and clear of every gate."""
    bitcell = RomBitcellSpec()
    tap = build_rom_tap(bitcell)
    (active,) = boxes_on(tap, "ACTIVE")
    x0, y0, x1, y1 = active
    x_s = bitcell.sd_xs["S"]
    assert (x0 + x1) / 2 == x_s and x1 - x0 == 16
    assert (y0, y1) == (27, TAP_HEIGHT)
    assert (x1 - x0) * (y1 - y0) >= 864  # ACTIVE.A.1A
    assert boxes_on(tap, "PSELECT") == [(0, 0, bitcell.width, TAP_HEIGHT)]
    assert boxes_on(tap, "NSELECT") == []
    (v0,) = boxes_on(tap, "V0")
    assert (v0[0] + v0[2]) / 2 == x_s
    for gx0, _, gx1, _ in boxes_on(tap, "GATE"):
        assert gx1 + 9 <= x0 or x1 + 9 <= gx0  # GATE.ACTIVE.S.4


def test_array_too_tall_for_one_tap_row():
    rows = LATCHUP_REACH // 81
    with pytest.raises(ValueError, match="ACTIVE.LUP.1"):
        RomArraySpec(2, rows, RomBitcellSpec(fins=2))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"words": 3, "bits": 1}, ValueError),
        ({"words": 2, "bits": 0}, ValueError),
        ({"words": 2, "bits": 1, "strap_every": 0}, ValueError),
        ({"words": 2.0, "bits": 1}, TypeError),
    ],
)
def test_array_spec_rejects_bad_shapes(kwargs, error):
    with pytest.raises(error):
        RomArraySpec(**kwargs)


def test_array_rejects_sites_outside_it():
    _library()
    with pytest.raises(IndexError):
        build_rom_array(RomArraySpec(2, 1), [(2, 0)])


def test_array_can_bring_its_bitlines_out_on_the_right():
    """The center lane's left half: bitline pins past its right edge, named by
    the caller, and wordline labels offset by `first_word`."""
    _library()
    spec = RomArraySpec(4, 2)
    cell = build_rom_array(
        spec, name="half", bitline="BLL", bitline_side="right", first_word=8
    )
    labels = {label.text: label.position for label in cell.get_labels()}
    for bit in range(2):
        x, _ = labels[f"BLL<{bit}>"]
        assert x > spec.width
    assert {f"WL<{w}>" for w in range(8, 12)} <= labels.keys()
    assert "BL<0>" not in labels and "WL<0>" not in labels


@pytest.mark.parametrize("fins", [0, True, 1.5])
def test_spec_rejects_bad_fins(fins):
    with pytest.raises((TypeError, ValueError)):
        RomBitcellSpec(fins=fins)


def test_netlist_floats_unprogrammed_drains():
    lines = RomBitcellSpec().netlist(programmed=(True, False)).splitlines()
    assert lines[0].startswith("M0 BL WLa VSS VSS nmos_rvt nfin=2")
    assert lines[1].startswith("M1 M1_float WLb VSS VSS")


def test_cli_writes_tile_and_array(tmp_path):
    main(["--fins", "1", "--out", str(tmp_path / "tile.gds")])
    main(
        [
            "--words",
            "4",
            "--bits",
            "3",
            "--strap-every",
            "2",
            "--out",
            str(tmp_path / "array.gds"),
        ]
    )
    assert (tmp_path / "tile.gds").stat().st_size > 0
    assert (tmp_path / "array.gds").stat().st_size > 0


@pytest.mark.parametrize("fins", [1, 2, 3])
@pytest.mark.parametrize("pattern", ["none", "all", "checker"])
def test_array_drc(asap7_drc, fins, pattern):
    spec = RomArraySpec(8, 6, RomBitcellSpec(fins=fins), strap_every=4)
    every = [(w, b) for w in range(spec.words) for b in range(spec.bits)]
    sites = {
        "none": [],
        "all": every,
        "checker": [(w, b) for w, b in every if (w + b) % 2],
    }[pattern]
    library = _library()
    top = build_rom_array(spec, sites, lib=library, name=f"rom_f{fins}_{pattern}")
    assert set(asap7_drc(library, top, tag=top.name)) == set()


def test_straps_wire_every_gate(tmp_path):
    """Extracted nets: WL<w> on its gates, ISO, sources and taps on VSS, bits on BL<b>.

    Read back from GDS in micrometres.  A word transistor's drain reaches its
    row's bitline exactly when the bit is programmed; otherwise it is a net of
    its own.
    """
    gdstk = pytest.importorskip("gdstk")
    from chipforge_asap7.verification.connectivity import extract_connectivity

    spec = RomArraySpec(4, 3, RomBitcellSpec(fins=1), strap_every=2)
    sites = {(0, 0), (1, 0), (3, 1), (2, 2)}
    library = _library()
    top = build_rom_array(spec, sites, lib=library)
    gds = tmp_path / "rom.gds"
    library.write_gds(str(gds))
    flat = next(
        c for c in gdstk.read_gds(str(gds), unit=1e-6).cells if c.name == top.name
    )
    flat = flat.copy("flat").flatten()
    # Tiles abut edge to edge, and the extractor only joins shapes that
    # overlap; merge each layer first, as a layout extractor would.
    cell = gdstk.Cell("merged")
    for layer, datatype in {(p.layer, p.datatype) for p in flat.polygons}:
        same = [p for p in flat.polygons if (p.layer, p.datatype) == (layer, datatype)]
        cell.add(*gdstk.boolean(same, [], "or", layer=layer, datatype=datatype))
    cell.add(*flat.labels)
    # Name each tap tie's LISD, so its net can be checked against VSS.
    from chipforge_asap7.verification.connectivity import LISD, PIN_TEXTTYPE

    y_tie = (spec.tap_y + TAP_HEIGHT - 27) / 1000
    for t in range(spec.tiles):
        x_tie = (t * spec.bitcell.width + spec.bitcell.sd_xs["S"]) / 1000
        cell.add(
            gdstk.Label(f"TAP{t}", (x_tie, y_tie), layer=LISD, texttype=PIN_TEXTTYPE)
        )

    iso_per_row = spec.tiles
    devices, labels = extract_connectivity(
        cell, expected_channels=spec.bits * (spec.words + iso_per_row)
    )
    net = {name: next(iter(ids)) for name, ids in labels.items() if len(ids) == 1}
    assert {f"WL<{w}>" for w in range(spec.words)} <= net.keys()
    assert {f"BL<{b}>" for b in range(spec.bits)} <= net.keys()
    vss = net["VSS"]
    assert all(net.pop(f"TAP{t}") == vss for t in range(spec.tiles))
    assert len(set(net.values())) == len(net)  # no two pins short

    gate_x = {}
    for word in range(spec.words):
        tile, tile_word = divmod(word, 2)
        gate = ("WLa", "WLb")[tile_word]
        gate_x[tile * spec.bitcell.width + spec.bitcell.gate_xs[gate]] = word
    for device in devices:
        x, y = (round(c * 1000) for c in device["center"])
        bit = next(
            b
            for b in range(spec.bits)
            if spec.row_y(b) <= y < spec.row_y(b) + spec.bitcell.height
        )
        if x not in gate_x:  # an ISO gate
            assert device["gate"] == vss
            continue
        word = gate_x[x]
        assert device["gate"] == net[f"WL<{word}>"]
        assert vss in device["terminals"]
        (drain,) = set(device["terminals"]) - {vss}
        assert (drain == net[f"BL<{bit}>"]) == ((word, bit) in sites)
