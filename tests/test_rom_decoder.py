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
from chipforge_asap7.devices.rom_decoder import (
    MIN_GAP,
    RAIL_MARGIN,
    WELL_TAP_REACH,
    main,
)
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


def test_only_a_row_too_long_for_its_end_taps_gets_a_tap_in_the_lane():
    assert RomDecoderSpec(array=RomArraySpec(512, 16)).lane_tap_x is None
    spec = RomDecoderSpec(array=RomArraySpec(1024, 16))
    taps = spec.well_taps
    assert len(taps) == 3 and taps[1] == spec.lane_tap_x
    assert max(b - a for a, b in pairwise(taps)) <= 2 * WELL_TAP_REACH
    # Every row has filler where the tap stands, and its slots stay clear of it.
    tap = spec.lane_tap_x
    for row in range(spec.interleave):
        spans = spec.fills(row)
        assert not any(x0 < tap + 108 and tap < x1 for x0, x1 in spans), row
        assert any(x1 == tap and x1 - x0 >= 108 for x0, x1 in spans), row
        assert any(x0 == tap + 108 and x1 - x0 >= 108 for x0, x1 in spans), row


@pytest.mark.parametrize(
    "spec",
    [
        {"array": RomArraySpec(1024, 8), "lane": False},
        {"array": RomArraySpec(2048, 16)},
    ],
    ids=["no_lane_for_a_tap", "too_long_for_one_tap"],
)
def test_a_row_no_tap_can_reach_is_refused(spec):
    with pytest.raises(ValueError, match=r"N-well taps.*ACTIVE\.LUP\.1"):
        RomDecoderSpec(**spec)


@pytest.mark.parametrize(
    "flags",
    [
        {},
        {
            "sense": True,
            "enable": True,
            "latch": True,
            "sample": True,
            "register": True,
            "control": True,
        },
    ],
    ids=["array_and_decoder", "self_timed"],
)
def test_1024_words_are_drc_clean(asap7_drc, flags):
    spec = RomDecoderSpec(array=RomArraySpec(1024, 16), **flags)
    library = _library()
    top = build_rom(spec, lib=library)
    assert set(asap7_drc(library, top, tag=top.name)) == set()


def test_1024_words_without_the_lane_tap_break_the_latchup_rule(asap7_drc):
    """The negative control: this is what the tap is for."""
    spec = RomDecoderSpec(array=RomArraySpec(1024, 16))
    spec.__dict__["lane_tap_x"] = None
    library = _library()
    top = build_rom(spec, lib=library)
    assert set(asap7_drc(library, top, tag=top.name)) == {"ACTIVE.LUP.1"}


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


def _settle(devices, net, inputs, held):
    """`_switch_level` with charge: a net no source reaches keeps its `held` value.

    The sources are VDD, VSS and the driven inputs.  The on transistors join
    nets into groups (a source does not join the groups it drives); a group
    takes the value of the sources it reaches.  One that reaches neither
    keeps the charge its nets held, shared: one value if they held the same
    one, unknown otherwise.  If none of them held a value, it keeps what it
    had a moment ago instead.  A latch that closes on a value it never held
    before this step otherwise loses it: under a synchronous update the
    value and an unknown chase each other round its two inverters forever.  One that reaches both is a fight, unknown too:
    a latch's two gates overlap for the inverter delay of its clock, as they
    do in silicon.  A fight still standing once the nets settle is a short.
    """
    source = {net["VDD"]: 1, net["VSS"]: 0, **inputs}
    nets = {t for d in devices for t in (*d["terminals"], d["gate"])} | set(held)
    value = {**held, **source}
    for _ in range(256):  # a self-timed read is a long chain of gates
        parent: dict = {}

        def find(n, parent=parent):
            parent.setdefault(n, n)
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        on = [
            d["terminals"]
            for d in devices
            if (g := value.get(d["gate"])) is not None and g == (0 if d["pmos"] else 1)
        ]
        for a, b in on:
            if a not in source and b not in source:
                parent[find(a)] = find(b)
        reach: dict = {}
        for a, b in on:
            for n, other in ((a, b), (b, a)):
                if other in source and n not in source:
                    reach.setdefault(find(n), set()).add(source[other])
        charge: dict = {}
        recent: dict = {}
        for n in nets - source.keys():
            charge.setdefault(find(n), set()).add(held.get(n))
            recent.setdefault(find(n), set()).add(value.get(n))
        new = dict(source)
        fights = set()
        for n in nets - source.keys():
            root = find(n)
            levels = reach.get(root)
            if levels is None:
                levels = (charge[root] - {None}) or (recent[root] - {None})
            elif len(levels) > 1:
                fights.add(root)
            new[n] = next(iter(levels)) if len(levels) == 1 else None
        if new == value:
            assert not fights, "VDD shorted to VSS"
            return value
        value = new
    raise AssertionError("the netlist did not settle")


def test_a_read_puts_each_programmed_bit_on_its_leaf(tmp_path):
    """Switch-level reads of the extracted ROM with its EVAL gate and sense lane.

    Precharge with PRECHN and EVAL low: every bitline rises and no wordline
    does.  Then PRECHN high and EVAL high: the addressed word's programmed
    bits discharge their bitline, in that word's half only, the other
    bitlines hold their charge, and each ``D<b>`` reads the bit.  With EVAL
    held low nothing discharges and every ``D<b>`` stays 0.
    """
    words, bits = 16, 4
    spec = _spec(words, bits, 2, enable=True, sense=True)
    sites = {
        (w, b) for w in range(words) for b in range(bits) if (w * 5 + b * 3) % 7 < 3
    }
    devices, net = _extract(tmp_path, spec, sites)
    supplies = {net["VDD"]: 1, net["VSS"]: 0}
    for address in range(words):
        pins = {net[f"A<{i}>"]: (address >> i) & 1 for i in range(spec.address_bits)}
        pre = _settle(
            devices, net, {**supplies, **pins, net["PRECHN"]: 0, net["EVAL"]: 0}, {}
        )
        assert all(pre[net[f"WL<{w}>"]] == 0 for w in range(words)), address
        assert all(
            pre[net[f"{h}<{b}>"]] == 1 for h in ("BLL", "BLR") for b in range(bits)
        )
        assert all(pre[net[f"D<{b}>"]] == 0 for b in range(bits))
        for enable in (0, 1):
            read = _settle(
                devices,
                net,
                {**supplies, **pins, net["PRECHN"]: 1, net["EVAL"]: enable},
                pre,
            )
            levels = [read[net[f"WL<{w}>"]] for w in range(words)]
            assert levels == [enable * int(w == address) for w in range(words)]
            data = [read[net[f"D<{b}>"]] for b in range(bits)]
            assert data == [enable * int((address, b) in sites) for b in range(bits)]
            side = "BLR" if address >= words // 2 else "BLL"
            for h in ("BLL", "BLR"):
                for b in range(bits):
                    low = enable and h == side and (address, b) in sites
                    assert read[net[f"{h}<{b}>"]] == int(not low), (address, h, b)


def test_the_address_latches_hold_through_the_read(tmp_path):
    """Switch-level reads through the address latches.

    The address is loaded while PRECHN is low, the latches close when it
    rises, and only then do the pins move to another address and EVAL rise:
    the word read is the loaded one, from every address to its complement.
    """
    words, bits = 16, 4
    spec = _spec(words, bits, 2, enable=True, sense=True, latch=True)
    sites = {
        (w, b) for w in range(words) for b in range(bits) if (w * 5 + b * 3) % 7 < 3
    }
    devices, net = _extract(tmp_path, spec, sites)
    assert not any(k.startswith("AL<") for k in net)  # the decoder's lines are internal
    supplies = {net["VDD"]: 1, net["VSS"]: 0}
    mask = words - 1

    def pins(address):
        return {net[f"A<{i}>"]: (address >> i) & 1 for i in range(spec.address_bits)}

    for address in range(words):
        other = address ^ mask
        steps = [
            {**pins(address), net["PRECHN"]: 0, net["EVAL"]: 0},
            {**pins(address), net["PRECHN"]: 1, net["EVAL"]: 0},
            {**pins(other), net["PRECHN"]: 1, net["EVAL"]: 0},
            {**pins(other), net["PRECHN"]: 1, net["EVAL"]: 1},
        ]
        value: dict = {}
        for inputs in steps:
            value = _settle(devices, net, {**supplies, **inputs}, value)
        levels = [value[net[f"WL<{w}>"]] for w in range(words)]
        assert levels == [int(w == address) for w in range(words)], address
        data = [value[net[f"D<{b}>"]] for b in range(bits)]
        assert data == [int((address, b) in sites) for b in range(bits)], address


def test_the_sample_latches_hold_the_word_through_the_next_precharge(tmp_path):
    """Switch-level reads through the sample latches.

    Each read precharges, evaluates, opens SAMPLE and closes it, and then
    precharges again: D<b> falls back to 0 but Q<b> keeps the word.  The next
    address's evaluation, with SAMPLE low, leaves Q alone too; only opening
    SAMPLE brings its word in.
    """
    words, bits = (
        16,
        6,
    )  # four bits' array is one 513 nm row high, with no rail to spare
    spec = _spec(words, bits, 2, enable=True, sense=True, sample=True)
    sites = {
        (w, b) for w in range(words) for b in range(bits) if (w * 5 + b * 3) % 7 < 3
    }
    devices, net = _extract(tmp_path, spec, sites)
    supplies = {net["VDD"]: 1, net["VSS"]: 0}

    def word(address):
        return [int((address, b) in sites) for b in range(bits)]

    def pins(address, prechn, sample):
        return {
            **supplies,
            **{net[f"A<{i}>"]: (address >> i) & 1 for i in range(spec.address_bits)},
            net["PRECHN"]: prechn,
            net["EVAL"]: prechn,
            net["SAMPLE"]: sample,
        }

    value: dict = {}
    for address in [*range(words), 0]:
        held = [value.get(net[f"Q<{b}>"]) for b in range(bits)]
        for prechn, sample in ((0, 0), (1, 0)):
            value = _settle(devices, net, pins(address, prechn, sample), value)
            if address or held[0] is not None:
                assert [value[net[f"Q<{b}>"]] for b in range(bits)] == held
        assert [value[net[f"D<{b}>"]] for b in range(bits)] == word(address)
        for prechn, sample in ((1, 1), (1, 0), (0, 0)):
            value = _settle(devices, net, pins(address, prechn, sample), value)
            assert [value[net[f"Q<{b}>"]] for b in range(bits)] == word(address)
        assert all(value[net[f"D<{b}>"]] == 0 for b in range(bits)), address
    # Left open through the precharge, the latches follow D down to 0.
    value = _settle(devices, net, pins(1, 0, 1), value)
    assert all(value[net[f"Q<{b}>"]] == 0 for b in range(bits))


def test_a_self_timed_read_samples_each_word(tmp_path):
    """Switch-level reads of the extracted ROM driven by CLK, RESETN and the
    address alone.

    CLK high precharges and loads the address; CLK low reads: the controller
    lifts PRECHN, fires EVAL, waits for the replica's READY and opens
    SAMPLE, and Q<b> is the word.  Back to CLK high, Q keeps it through the
    precharge while the next address loads.  With RESETN low nothing reads.
    """
    words, bits = 16, 6
    spec = _spec(
        words, bits, 2, enable=True, sense=True, latch=True, sample=True, control=True
    )
    sites = {
        (w, b) for w in range(words) for b in range(bits) if (w * 5 + b * 3) % 7 < 3
    }
    devices, net = _extract(tmp_path, spec, sites)
    assert {"CLK", "RESETN", "READY", "EVAL", "SAMPLE"} <= net.keys()
    supplies = {net["VDD"]: 1, net["VSS"]: 0}

    def word(address):
        return [int((address, b) in sites) for b in range(bits)]

    def pins(address, clk, resetn=1):
        return {
            **supplies,
            **{net[f"A<{i}>"]: (address >> i) & 1 for i in range(spec.address_bits)},
            net["CLK"]: clk,
            net["RESETN"]: resetn,
        }

    def q(value):
        return [value[net[f"Q<{b}>"]] for b in range(bits)]

    value = _settle(devices, net, pins(0, 1), {})
    assert [value[net[n]] for n in ("PRECHN", "EVAL", "SAMPLE", "READY")] == [0] * 4
    for address in [*range(words), 3]:
        value = _settle(devices, net, pins(address, 0), value)
        assert [value[net[n]] for n in ("PRECHN", "EVAL", "SAMPLE", "READY")] == [1] * 4
        assert q(value) == word(address), address
        value = _settle(devices, net, pins(address ^ 5, 1), value)
        assert [value[net[n]] for n in ("PRECHN", "EVAL", "SAMPLE", "READY")] == [0] * 4
        assert q(value) == word(address), address
    # In reset CLK low reads nothing: no wordline, and Q keeps the last word.
    value = _settle(devices, net, pins(9, 1), value)
    value = _settle(devices, net, pins(9, 0, resetn=0), value)
    assert value[net["EVAL"]] == 0 and value[net["SAMPLE"]] == 0
    assert all(value[net[f"WL<{w}>"]] == 0 for w in range(words))
    assert q(value) == word(3)


def test_a_registered_read_changes_q_only_as_clk_rises(tmp_path):
    """The self-timed read with the output register, at switch level.

    While CLK is low the read runs and is sampled as before, but Q still
    shows the word read in the cycle before.  Q takes the new word only when
    CLK rises and SAMPLE falls, and holds it through the next read.  In reset
    nothing reads and Q keeps the last word.
    """
    words, bits = 16, 6
    spec = _spec(
        words,
        bits,
        2,
        enable=True,
        sense=True,
        latch=True,
        sample=True,
        register=True,
        control=True,
    )
    assert "_qr_c4" in spec.cell_name
    sites = {
        (w, b) for w in range(words) for b in range(bits) if (w * 5 + b * 3) % 7 < 3
    }
    devices, net = _extract(tmp_path, spec, sites)
    supplies = {net["VDD"]: 1, net["VSS"]: 0}

    def word(address):
        return [int((address, b) in sites) for b in range(bits)]

    def pins(address, clk, resetn=1):
        return {
            **supplies,
            **{net[f"A<{i}>"]: (address >> i) & 1 for i in range(spec.address_bits)},
            net["CLK"]: clk,
            net["RESETN"]: resetn,
        }

    def q(value):
        return [value[net[f"Q<{b}>"]] for b in range(bits)]

    value = _settle(devices, net, pins(0, 1), {})
    previous = None
    addresses = [*range(words), 3]
    for address in addresses:
        value = _settle(devices, net, pins(address, 0), value)
        assert [value[net[n]] for n in ("PRECHN", "EVAL", "SAMPLE", "READY")] == [1] * 4
        if previous is not None:
            assert q(value) == word(previous), address  # the read is not out yet
        value = _settle(devices, net, pins(address ^ 5, 1), value)
        assert value[net["SAMPLE"]] == 0
        assert q(value) == word(address), address
        previous = address
    # Two words that differ, so that a leak would show.
    assert word(addresses[-2]) != word(addresses[-1])
    value = _settle(devices, net, pins(9, 0, resetn=0), value)
    assert value[net["SAMPLE"]] == 0 and q(value) == word(3)
    value = _settle(devices, net, pins(9, 1, resetn=0), value)
    assert q(value) == word(3)


def test_register_needs_the_sample_latches():
    with pytest.raises(ValueError, match="register needs sample"):
        _spec(16, 6, 2, sense=True, register=True)


def test_control_needs_the_whole_read_path():
    with pytest.raises(ValueError, match="control needs enable, latch and sample"):
        _spec(16, 6, 2, enable=True, sense=True, latch=True, control=True)
    spec = _spec(
        16, 6, 2, enable=True, sense=True, latch=True, sample=True, control=True
    )
    assert spec.cells.bits == 7 and spec.replica_bit == 6
    assert spec.sense_lane.d_net(6) == "READY"
    tracks = spec.latch_tracks
    assert (
        tracks["SAMPLE"]
        < tracks["READY"]
        < min(tracks[line] for line in spec.channel_lines)
    )
    assert "EVAL" in spec.joins


def test_sample_needs_the_sense_lane():
    with pytest.raises(ValueError, match="sample needs sense"):
        _spec(16, 4, 2, sample=True)
    spec = _spec(16, 6, 2, sense=True, sample=True)
    assert spec.sense_lane.sample
    assert spec.cell_name.endswith("_s_q")


def test_latch_needs_the_sense_lane_and_grows_the_gap():
    with pytest.raises(ValueError, match="latch needs sense"):
        _spec(16, 4, 2, enable=True, latch=True)
    plain = _spec(16, 4, 2, enable=True, sense=True)
    latched = _spec(16, 4, 2, enable=True, sense=True, latch=True)
    assert latched.gap > plain.gap
    assert latched.address_line == "AL"
    assert latched.right_x >= latched.latch_ends[1] + GATE_PITCH
    assert set(latched.latch_tracks) == {
        f"AL<{i}>" for i in range(latched.address_bits)
    }
    assert {f"AL<{i}>" for i in range(latched.address_bits)} <= latched.joins.keys()


def test_enable_gates_the_top_pb_literals():
    gates = network(6, 4, 3, enable="EVAL")
    assert gates["EAN<5>"].ins == ("AN<5>", "EVAL")
    assert gates["EA<5>"].ins == ("A<5>", "EVAL")
    pb = [g for name, g in gates.items() if name.startswith("PB<")]
    assert len(pb) == 4 and all({"EA<5>", "EAN<5>"} & set(g.ins) for g in pb)
    # A one-bit PB group takes EVAL in its own AND2.
    one = network(5, 4, 3, enable="EVAL")
    assert one["PB<0>"].ins == ("AN<4>", "EVAL")
    with pytest.raises(ValueError, match="enable needs predecode"):
        _spec(16, predecode=False, lane=False, enable=True)


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
