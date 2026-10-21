"""The ROM's transposed wordline decoder: plan, DRC and decode logic.

The DRC half skips when gdscheck is missing and the logic check without
gdstk; set ``ASAP7_REQUIRE_TOOLS=1`` to make a missing gdscheck a failure.
"""

from __future__ import annotations

import gdspy
import pytest

from chipforge_asap7.devices import (
    RomArraySpec,
    RomBitcellSpec,
    RomDecoderSpec,
    build_rom,
    build_rom_decoder,
)
from chipforge_asap7.devices.rom_decoder import MIN_GAP, RAIL_MARGIN, main
from chipforge_asap7.layout import GATE_PITCH


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def _spec(words=16, bits=4, fins=2, **kwargs):
    array = RomArraySpec(words, bits, RomBitcellSpec(fins=fins), strap_every=4)
    return RomDecoderSpec(array, **kwargs)


@pytest.mark.parametrize(
    ("words", "pa_bits", "rails"),
    [(8, 3, 3), (16, 3, 4), (64, 4, 8), (256, 5, 16)],
)
def test_split_takes_the_fewest_rails(words, pa_bits, rails):
    spec = _spec(words)
    assert (spec.pa_bits, spec.pair_rails) == (pa_bits, rails)
    assert all(len(spec.rails(p)) == rails for p in range(spec.pairs))


def test_every_word_has_one_pa_and_one_pb_line():
    spec = _spec(64)
    seen = {spec.lines(w) for w in range(64)}
    assert len(seen) == 64
    for w in range(64):
        pa, pb = spec.lines(w)
        assert pa in spec.rails(spec.row_of(w) // 2) and pb in spec.rails(0)


def test_slots_sit_on_the_gate_grid_under_their_wordlines():
    spec = _spec(64)
    for w in range(64):
        x = spec.slot_x(w)
        assert x % GATE_PITCH == 0
        assert x + spec.wordline_local(w % 2) == spec.array.wordline_x(w)
        assert spec.wordline_local(w % 2) + 17 <= spec.slot_width  # pad's end
    for w in range(0, 64, 2):  # a pair's two rows stand slot over slot
        assert spec.slot_x(w) == spec.slot_x(w + 1)


def test_stubs_keep_off_every_wordline_track():
    spec = _spec(64)
    for parity in (0, 1):
        tracks = spec.wordline_tracks(parity)
        for x in spec.stubs(parity).values():
            assert min(abs(x - t) for t in tracks) >= 36
    even, odd = spec.stubs(0), spec.stubs(1)
    assert odd["B"] == even["B"] and abs(odd["A"] - even["A"]) >= 36


def test_rails_sit_on_the_m4_lattice_inside_their_pair():
    spec = _spec(256)
    for pair in range(spec.pairs):
        ys = [spec.rail_y(pair, line) for line in spec.rails(pair)]
        assert len(set(ys)) == len(ys)
        assert all((y - 12) % 48 == 0 for y in ys)
        lo = spec.y0 + pair * spec.pair_height
        assert (
            lo + RAIL_MARGIN <= min(ys)
            and max(ys) <= lo + spec.pair_height - RAIL_MARGIN
        )


@pytest.mark.parametrize(("words", "interleave"), [(16, 8), (256, 8), (32, 16)])
def test_straps_stand_on_every_rows_end(words, interleave):
    """Rows are padded to one span; each supply has a strap inside each end."""
    spec = _spec(words, interleave=interleave)
    lo, hi = spec.rows_span
    left, right = spec.slots_span
    assert (left - lo, hi - right) == (6 * GATE_PITCH, 4 * GATE_PITCH)
    assert (right - left) % GATE_PITCH == 0
    rails = spec.supply_rails
    assert [net for _, net in rails] == ["VSS", "VDD", "VSS"] * spec.pairs
    for x_left, x_right in spec.strap_xs.values():
        assert lo < x_left < lo + 2 * GATE_PITCH  # on the outer filler
        assert hi - 2 * GATE_PITCH < x_right < hi


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"words": 12}, ValueError),
        ({"words": 4}, ValueError),
        ({"interleave": 4}, ValueError),
        ({"interleave": 6}, ValueError),
        ({"interleave": 8.0}, TypeError),
        ({"gap": 27}, ValueError),
        ({"gap": MIN_GAP + 1}, ValueError),
        ({"split": 2}, ValueError),
        ({"split": 5}, ValueError),
    ],
)
def test_spec_rejects(kwargs, error):
    words = kwargs.pop("words", 16)
    with pytest.raises(error):
        _spec(words, **kwargs)


def test_cli_writes_a_rom(tmp_path):
    out = tmp_path / "rom.gds"
    main(["--array.words", "16", "--array.bits", "2", "--out", str(out)])
    assert out.stat().st_size > 0


@pytest.mark.parametrize(
    ("words", "bits", "fins", "gap"),
    [(16, 6, 1, 54), (16, 6, 3, 81), (64, 8, 2, 54)],
)
def test_rom_drc(asap7_drc, words, bits, fins, gap):
    spec = _spec(words, bits, fins, gap=gap)
    sites = [
        (w, b) for w in range(words) for b in range(bits) if (w * 5 + b * 3) % 7 < 3
    ]
    library = _library()
    top = build_rom(spec, sites, lib=library)
    assert set(asap7_drc(library, top, tag=top.name)) == set()


def test_decoder_drc_with_taller_pairs(asap7_drc):
    """256 words: 16 rails per pair, more than two rows hold."""
    spec = _spec(256, 2)
    assert spec.pair_height > 2 * spec.row_height + 54
    library = _library()
    top = build_rom_decoder(spec, lib=library)
    assert set(asap7_drc(library, top, tag=top.name)) == set()


@pytest.mark.parametrize("words", [16, 64])
def test_each_wordline_is_the_and_of_its_lines(tmp_path, words):
    """Extracted nets: WL<w> is driven by one inverter whose input's pull-ups
    are gated by exactly PA and PB of `w`, with every pin -- VDD and VSS
    included -- a single net."""
    gdstk = pytest.importorskip("gdstk")
    from chipforge_asap7.verification import connectivity as cv

    spec = _spec(words, 2, 1)
    library = _library()
    top = build_rom(spec, [(w, w % 2) for w in range(words)], lib=library)
    gds = tmp_path / "rom.gds"
    library.write_gds(str(gds))
    flat = next(
        c for c in gdstk.read_gds(str(gds), unit=1e-6).cells if c.name == top.name
    )
    flat = flat.copy("flat").flatten()
    # Merge abutting shapes, which the extractor would otherwise keep apart.
    cell = gdstk.Cell("merged")
    for layer, datatype in {(p.layer, p.datatype) for p in flat.polygons}:
        same = [p for p in flat.polygons if (p.layer, p.datatype) == (layer, datatype)]
        cell.add(*gdstk.boolean(same, [], "or", layer=layer, datatype=datatype))
    cell.add(*flat.labels)
    gates = gdstk.boolean(
        cv.layer_polygons(cell, cv.GATE),
        cv.layer_polygons(cell, cv.GCUT),
        "not",
        precision=1e-6,
    )
    channels = gdstk.boolean(
        cv.layer_polygons(cell, cv.ACTIVE), gates, "and", precision=1e-6
    )
    devices, labels = cv.extract_connectivity(cell, expected_channels=len(channels))

    # Every pin is one net -- the supply grid joins every row's rails, and
    # the array's VSS columns, to the straps -- and no two pins short.
    net = {}
    for name, ids in labels.items():
        assert len(ids) == 1, name
        net[name] = next(iter(ids))
    assert len(set(net.values())) == len(net)
    assert {"VDD", "VSS"} <= net.keys()
    for w in range(words):
        drivers = [d for d in devices if net[f"WL<{w}>"] in d["terminals"]]
        assert {d["pmos"] for d in drivers} == {True, False}
        for d in drivers:
            assert net["VDD" if d["pmos"] else "VSS"] in d["terminals"]
        inputs = {d["gate"] for d in drivers}
        assert len(inputs) == 1
        (y,) = inputs
        pull_ups = {d["gate"] for d in devices if d["pmos"] and y in d["terminals"]}
        assert pull_ups == {net[line] for line in spec.lines(w)}
