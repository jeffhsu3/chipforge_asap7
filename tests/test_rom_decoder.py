"""The ROM's transposed wordline decoder: plan, DRC and decode logic.

The DRC half skips when gdscheck is missing and the logic check without
gdstk; set ``ASAP7_REQUIRE_TOOLS=1`` to make a missing gdscheck a failure.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import pairwise

import gdspy
import pytest

from chipforge_asap7.devices import (
    RomArraySpec,
    RomBitcellSpec,
    RomDecoderSpec,
    build_rom,
    build_rom_decoder,
    size_rom_decoder,
)
from chipforge_asap7.devices.rom_decoder import MIN_GAP, RAIL_MARGIN, main
from chipforge_asap7.devices.rom_predecoder import network
from chipforge_asap7.layout import GATE_PITCH


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


def _spec(words=16, bits=4, fins=2, **kwargs):
    array = RomArraySpec(words, bits, RomBitcellSpec(fins=fins), strap_every=4)
    return RomDecoderSpec(array, **kwargs)


@pytest.mark.parametrize(
    ("words", "predecode", "pa_bits", "rails"),
    [(8, False, 3, 3), (16, True, 3, 4), (64, True, 4, 8), (256, True, 5, 16)],
)
def test_split_takes_the_fewest_rails(words, predecode, pa_bits, rails):
    spec = _spec(words, predecode=predecode, lane=words >= 16)
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
        assert x + spec.wordline_local(w % 2) == spec.wordline_x(w)
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
    assert (right - lo) % GATE_PITCH == 0
    rails = spec.supply_rails
    assert [net for _, net in rails] == ["VSS", "VDD", "VSS"] * spec.pairs
    for x_left, x_right in spec.strap_xs.values():
        assert lo < x_left < lo + 2 * GATE_PITCH  # on the outer filler
        assert hi - 2 * GATE_PITCH < x_right < hi


def _evaluate(gates, address):
    """Every line's value for `address`, from the gate network alone."""
    value = {f"A<{i}>": (address >> i) & 1 for i in range(16)}

    def get(line):
        if line not in value:
            gate = gates[line]
            ins = [get(i) for i in gate.ins]
            value[line] = int(not ins[0]) if gate.kind == "inv" else ins[0] & ins[1]
        return value[line]

    return get


@pytest.mark.parametrize(
    ("words", "interleave"), [(16, 8), (64, 8), (256, 8), (128, 16)]
)
def test_predecoders_make_one_hot_pa_and_pb(words, interleave):
    spec = _spec(words, interleave=interleave)
    gates = network(spec.address_bits, spec.pa_bits, spec.row_bits)
    for address in range(words):
        get = _evaluate(gates, address)
        pa, pb = spec.lines(address)
        high = {line for line in gates if line[:2] in ("PA", "PB") and get(line)}
        assert high == {pa, pb}


@pytest.mark.parametrize(
    ("words", "interleave"), [(16, 8), (64, 8), (256, 8), (128, 16)]
)
def test_predecoders_sit_with_their_rails(words, interleave):
    """PA gates in the pair that uses them; both rows of every column filled."""
    spec = _spec(words, interleave=interleave)
    places = spec.predecoder
    for p in places:
        if p.gate.out.startswith("PA<"):
            assert p.pair == spec.row_of(int(p.gate.out[3:-1])) // 2
    cells = Counter((p.pair, p.column) for p in places)
    assert set(cells.values()) == {2}
    assert {p.gate.out for p in places} == set(
        network(spec.address_bits, spec.pa_bits, spec.row_bits)
    )


@pytest.mark.parametrize(("words", "interleave"), [(16, 8), (256, 8), (128, 16)])
def test_the_lane_splits_the_words_between_two_halves(words, interleave):
    """The right half is the left one a whole number of tiles further on, and
    the lane between them holds the predecoders and the join columns."""
    spec = _spec(words, interleave=interleave)
    n = words // 2
    assert spec.half.words == n and spec.right_x % spec.half.bitcell.width == 0
    for w in range(n):
        assert spec.wordline_x(w) == spec.half.wordline_x(w)
        assert spec.wordline_x(w + n) == spec.right_x + spec.half.wordline_x(w)
        assert spec.slot_x(w + n) == spec.slot_x(w) + spec.right_x
    (_, left_hi), (right_lo, _) = spec.side_span(0), spec.side_span(1)
    assert left_hi < spec.pre_x(0) and spec.pre_x(spec.pre_columns) < right_lo
    assert spec.half.width < spec.pre_x(0)
    # A PB line only the left half uses joins on the lane's left side.
    left_pb = {spec.lines(w)[1] for w in range(n)} - {
        spec.lines(w)[1] for w in range(n, words)
    }
    for line, x in spec.joins.items():
        if line in left_pb:
            assert left_hi < x < spec.pre_x(0), line
        else:
            assert spec.pre_x(spec.pre_columns) < x < right_lo, line
    # Every row's cells tile its span without overlaps.
    for row in range(interleave):
        cells = [
            (x, x + spec.slot_width)
            for x in (spec.slot_x(w) for w in range(row, words, interleave))
        ]
        cells += spec.fills(row)
        cells.append((spec.pre_x(0), spec.pre_x(spec.pre_columns)))
        cells.sort()
        assert cells[0][0] == spec.slots_span[0] and cells[-1][1] == spec.slots_span[1]
        assert all(a[1] == b[0] for a, b in pairwise(cells)), row


def test_m4_segments_keep_their_tracks_apart():
    """Same track: 48 nm between ends.  Next track: tips never corner to corner."""
    spec = _spec(256)
    for pair in range(spec.pairs):
        segs = [v for (p, _), v in spec.segments.items() if p == pair]
        for i, (y, x0, x1) in enumerate(segs):
            assert x0 % 48 == 0 and x1 % 48 == 0
            for y2, a, b in segs[i + 1 :]:
                if y2 == y:
                    assert x1 + 48 <= a or b + 48 <= x0
                elif abs(y2 - y) == 48:
                    assert x0 != b and x1 != a


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"words": 12}, ValueError),
        ({"words": 4}, ValueError),
        ({"words": 8}, ValueError),  # predecoders need a PB bit
        ({"words": 8, "predecode": False}, ValueError),  # each half a slot a row
        ({"interleave": 4}, ValueError),
        ({"interleave": 6}, ValueError),
        ({"interleave": 8.0}, TypeError),
        ({"gap": 27}, ValueError),
        ({"gap": MIN_GAP + 1}, ValueError),
        ({"split": 2}, ValueError),
        ({"split": 4}, ValueError),
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
    ("words", "bits", "fins", "gap", "lane"),
    [
        (16, 6, 1, 54, True),
        (16, 6, 3, 81, True),
        (64, 8, 2, 54, True),
        (16, 6, 2, 54, False),
    ],
)
def test_rom_drc(asap7_drc, words, bits, fins, gap, lane):
    spec = _spec(words, bits, fins, gap=gap, lane=lane)
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


def _extract(tmp_path, spec, sites):
    """``(devices, net)``: the ROM's extracted devices and each pin's one net."""
    gdstk = pytest.importorskip("gdstk")
    from chipforge_asap7.verification import connectivity as cv

    library = _library()
    top = build_rom(spec, sites, lib=library)
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
    return devices, net


def _switch_level(devices, net, inputs):
    """Each net's settled value with `inputs` driven: 1, 0 or None (floating).

    A net takes VDD's or VSS's value when a chain of on transistors reaches
    it, and gates read the values of the last pass until nothing changes.
    """
    value = dict(inputs)
    for _ in range(64):
        parent = {}

        def find(n, parent=parent):
            parent.setdefault(n, n)
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        for d in devices:
            g = value.get(d["gate"])
            if g is not None and g == (0 if d["pmos"] else 1):
                a, b = d["terminals"]
                parent[find(a)] = find(b)
        high, low = find(net["VDD"]), find(net["VSS"])
        assert high != low, "VDD shorted to VSS"
        new = dict(inputs)
        for n in list(parent):
            root = find(n)
            if root in (high, low) and n not in inputs:
                new[n] = int(root == high)
        if new == value:
            return value
        value = new
    raise AssertionError("the netlist did not settle")


@pytest.mark.parametrize("words", [16, 64])
def test_each_wordline_is_the_and_of_its_lines(tmp_path, words):
    """Extracted nets: WL<w> is driven by one inverter whose input's pull-ups
    are gated by exactly PA and PB of `w`, with every pin -- VDD and VSS
    included -- a single net."""
    spec = _spec(words, 2, 1)
    devices, net = _extract(tmp_path, spec, [(w, w % 2) for w in range(words)])
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


@pytest.mark.parametrize(
    ("words", "interleave", "lane"), [(16, 8, True), (32, 16, True), (16, 8, False)]
)
def test_each_address_raises_its_own_wordline(tmp_path, words, interleave, lane):
    """Switch-level simulation of the extracted ROM, from the address pins:
    for every address exactly its wordline is high and every other low, and
    only its programmed bit's bitline, in its own half, is pulled to VSS."""
    spec = _spec(words, 2, 1, interleave=interleave, lane=lane)
    devices, net = _extract(tmp_path, spec, [(w, w % 2) for w in range(words)])
    supplies = {net["VDD"]: 1, net["VSS"]: 0}
    halves = ("BLL", "BLR") if lane else ("BL",)
    for address in range(words):
        pins = {net[f"A<{i}>"]: (address >> i) & 1 for i in range(spec.address_bits)}
        value = _switch_level(devices, net, {**supplies, **pins})
        levels = [value.get(net[f"WL<{w}>"]) for w in range(words)]
        assert levels == [int(w == address) for w in range(words)], address
        side = address >= words // 2 if lane else 0
        bitlines = {
            (h, b): value.get(net[f"{name}<{b}>"])
            for h, name in enumerate(halves)
            for b in range(2)
        }
        assert bitlines == {
            (h, b): 0 if (h, b) == (side, address % 2) else None
            for h in range(len(halves))
            for b in range(2)
        }, address


def test_independent_sizes_keep_common_rails_and_have_distinct_cell_names():
    base = _spec()
    sized = replace(
        base,
        nand_fins=(4, 2),
        driver_fins=(8, 8),
        predecoder_nand_fins=(6, 3),
        predecoder_driver_fins=(12, 12),
    )
    assert (
        len(
            {
                base.cell_name,
                sized.cell_name,
                replace(sized, driver_fins=(10, 10)).cell_name,
                replace(sized, driver_fingers=4).cell_name,
            }
        )
        == 4
    )
    cells = (sized.nand, sized.inverter, sized.pre_slot.nand, sized.pre_slot.inverter)
    assert len({cell.rows for cell in cells}) == 4
    assert all(cell.stack.rails == sized.stack.rails for cell in cells)
    assert all(cell.rail_sources for cell in cells)
    library = _library()
    build_rom_decoder(base, lib=library)
    build_rom_decoder(sized, lib=library)
    assert base.cell_name in library.cells and sized.cell_name in library.cells


@pytest.mark.parametrize(
    ("words", "bits", "interleave", "load"),
    [
        (16, 8, 8, None),
        (64, 8, 8, None),
        (64, 64, 8, None),
        (256, 8, 8, None),
        (256, 64, 8, None),
        (16, 2, 8, 16),
        (64, 8, 16, None),
    ],
)
def test_sized_rom_drc(asap7_drc, words, bits, interleave, load):
    sizing = size_rom_decoder(
        RomDecoderSpec(RomArraySpec(words, bits), interleave=interleave),
        wl_load_fF=load,
    )
    library = _library()
    top = build_rom(sizing.spec, [(0, 0), (words - 1, bits - 1)], lib=library)
    assert set(asap7_drc(library, top, tag=top.name)) == set()


@pytest.mark.parametrize("load", [2.0, 16.0])
def test_sized_rom_extracts_the_requested_fins_and_decodes(tmp_path, load):
    sizing = size_rom_decoder(_spec(16, 2, 1), wl_load_fF=load)
    spec = sizing.spec
    devices, net = _extract(tmp_path, spec, [(w, w % 2) for w in range(16)])
    for word in range(16):
        drivers = [d for d in devices if net[f"WL<{word}>"] in d["terminals"]]
        for pmos, fins in (
            (False, spec.inverter.pull_down_fins),
            (True, spec.inverter.pull_up_fins),
        ):
            assert sum(d["nfin"] for d in drivers if d["pmos"] == pmos) == fins
        (nand_out,) = {d["gate"] for d in drivers}
        for line in spec.lines(word):
            pullups = [
                d
                for d in devices
                if d["pmos"] and d["gate"] == net[line] and nand_out in d["terminals"]
            ]
            assert (
                sum(d["nfin"] for d in pullups)
                == spec.nand.rows[0][1] * spec.nand.fingers
            )
    supplies = {net["VDD"]: 1, net["VSS"]: 0}
    for address in range(16):
        pins = {net[f"A<{i}>"]: (address >> i) & 1 for i in range(4)}
        value = _switch_level(devices, net, {**supplies, **pins})
        assert [value.get(net[f"WL<{w}>"]) for w in range(16)] == [
            int(w == address) for w in range(16)
        ]


def test_cli_can_size_from_a_supplied_wordline_load(tmp_path, capsys):
    out = tmp_path / "sized.gds"
    main(
        [
            "--array.words",
            "16",
            "--array.bits",
            "2",
            "--wl-load-ff",
            "8",
            "--out",
            str(out),
        ]
    )
    assert out.stat().st_size > 0
    assert "WL 8.000 fF (supplied total)" in capsys.readouterr().out
