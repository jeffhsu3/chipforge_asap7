"""The NOR-ROM read bench: its decks, and one real run when the tools exist.

The run compiles BSIM-CMG 107 with OpenVAF and simulates it in ngspice; it
skips when either, the ASAP7 model card or the BSIM-CMG source is missing,
and fails instead under ``ASAP7_REQUIRE_TOOLS=1``.
"""

import shutil
from pathlib import Path

import pytest

from chipforge_asap7.devices.rom_predecode_spice import predecode_network
from chipforge_asap7.devices.rom_read_spice import (
    RomReadConditions,
    decoder_driver,
    decoder_spec,
    render_rom_read_deck,
    run_rom_read,
    strap_stack_ohm,
    wordline_network,
)
from chipforge_asap7.devices.spice import (
    compile_bsimcmg107,
    find_default_bsimcmg107_source,
    find_default_model_card,
)

CONDITIONS = RomReadConditions()


def devices(deck):
    return [line.split() for line in deck.splitlines() if line.startswith("N")]


def deck_for(tmp_path, fins=1, words=64, **kwargs):
    return render_rom_read_deck(
        fins,
        words,
        models=tmp_path / "m.pm",
        osdi=tmp_path / "b.osdi",
        conditions=RomReadConditions(**kwargs),
    )


def ladder(deck, line):
    """Each line's word devices, resistors and capacitors, by node."""
    nodes = lambda name: name == f"BL{line}" or name.startswith(f"BL{line}_")
    parts = [x.split() for x in deck.splitlines()]
    words = [d for d in parts if d[0].startswith("N") and nodes(d[1])]
    rs = [float(d[3]) for d in parts if d[0].startswith(f"R{line}")]
    cs = [float(d[3].rstrip("f")) for d in parts if d[0].startswith(f"C{line}")]
    return words, rs, cs


@pytest.mark.parametrize("fins", [1, 2])
def test_both_lines_carry_every_other_word_as_an_off_device(tmp_path: Path, fins):
    deck = deck_for(tmp_path, fins=fins, words=64)
    off = ("0", "0", "0", "nmos_rvt", "L=20n", f"NFIN={fins}")
    blp, _, _ = ladder(deck, "P")
    blu, _, _ = ladder(deck, "U")
    # BLP: 63 off words, the selected word on WL at the far end, and the
    # precharge PMOS at the sense end.
    assert (
        sorted(tuple(d[2:]) for d in blp if d[0] not in ("NPP", "NPsel")) == [off] * 63
    )
    (selected,) = [d for d in blp if d[0] == "NPsel"]
    assert selected[1] == "BLP_63"
    assert selected[2] == wordline_network(fins, CONDITIONS, "nmos_rvt").selected_gate
    assert {d[1] for d in blp if d[0] == "NPP"} == {"BLP"}
    # BLU: the same 63 off words; the far word's drain floats.
    assert sorted(tuple(d[2:]) for d in blu if d[0] != "NPU") == [off] * 63
    assert "BLU_63" not in {d[1] for d in blu}


def test_the_bitline_is_a_distributed_m2_ladder(tmp_path: Path):
    deck = deck_for(tmp_path, words=1024)
    for line in ("P", "U"):
        _, rs, cs = ladder(deck, line)
        # 1024 words x 81 nm x 0.2 fF/um; 1023 segments of 81 nm of 18 nm M2.
        assert len(cs) == 1024 and sum(cs) == pytest.approx(16.5888)
        assert len(rs) == 1023
        assert sum(rs) == pytest.approx(1023 * 0.081 * 3.03147 / 0.018, rel=1e-5)
    # No resistance: one lumped node per line.
    _, rs, cs = ladder(deck_for(tmp_path, words=16, wire_ohm_per_um=0), "P")
    assert rs == [] and len(cs) == 16


def test_strap_stack_is_the_drawn_via_stack():
    # Gate link 0.1, 18 nm of 16 nm LIG at 8.33 ohm/sq, V0 19.38, V1, V2 19.53.
    assert strap_stack_ohm() == pytest.approx(
        0.1 + 8.33353 * 18 / 16 + 19.3796 + 2 * 19.5347
    )


def test_the_wordline_is_m3_straps_onto_continuous_poly(tmp_path: Path):
    c = RomReadConditions(bits=64, strap_every=16)
    net = wordline_network(2, c, "nmos_rvt")
    assert (net.rows, net.straps) == (64, 4)
    rs = {d[0]: d[1:] for d in (x.split() for x in net.lines) if d[0].startswith("RW")}
    # One stack from each strap's M3 node to its poly tap.
    stacks = [v for v in rs.values() if v[0].startswith("WM") and v[1].startswith("WT")]
    assert len(stacks) == 4
    assert all(
        float(v[2]) == pytest.approx(strap_stack_ohm(2), rel=1e-5) for v in stacks
    )
    # The poly runs through every row and tap: 64 + 4 nodes, 67 segments.
    poly = [
        v for v in rs.values() if v[0][:2] in ("WT", "WR") and v[1][:2] in ("WT", "WR")
    ]
    assert len(poly) == 67
    # Every row but the selected one hangs a gate load on the poly.
    loads = [x for x in net.lines if x.startswith("NG")]
    assert len(loads) == 63 and f"WR{net.selected_row} " not in " ".join(loads)
    # The worst row sits in the bottom group, furthest from the decoder.
    assert 0 <= net.selected_row < 16


def test_one_strap_row_makes_the_top_row_the_slowest():
    net = wordline_network(1, RomReadConditions(bits=16, strap_every=16), "nmos_rvt")
    assert (net.straps, net.selected_row) == (1, 15)


def test_the_ideal_wordline_drives_the_gate_directly(tmp_path: Path):
    deck = deck_for(tmp_path, wordline_rc=False)
    assert not any(x.startswith(("RW", "NG")) for x in deck.splitlines())
    assert " WL 0 0 nmos_rvt" in deck
    rc = deck_for(tmp_path)
    assert ".measure tran t_wordline TRIG v(WL)" in rc


def test_the_default_driver_is_the_drawn_nand_and_inverter(tmp_path: Path):
    # Without the predecoders, PA is the edge and PB is held high.
    deck = deck_for(tmp_path, predecode=False)
    drv = {d[0]: d for d in devices(deck) if d[0].startswith("ND")}
    # NAND: series NMOS from the output through PA then PB; parallel PMOS.
    assert [drv[n][1:4] for n in ("ND1", "ND2")] == [
        ["nandout", "PA", "nmid"],
        ["nmid", "PB", "0"],
    ]
    assert {tuple(drv[n][1:4]) for n in ("ND3", "ND4")} == {
        ("nandout", "PA", "VDD"),
        ("nandout", "PB", "VDD"),
    }
    # Inverter onto the top of the M3 strap; two fingers of 3 fins each.
    assert drv["ND5"][1:3] == drv["ND6"][1:3] == ["WL", "nandout"]
    assert {d[-1] for d in drv.values()} == {"NFIN=6"}
    assert "Vpa PA 0 PWL" in deck and "Vpb PB 0 0.7" in deck
    assert "Vwl " not in deck
    # Every time now starts at the predecoded PA line.
    assert ".measure tran t_sense TRIG v(PA)" in deck
    assert ".measure tran t_decoded TRIG v(PA) VAL=0.35 RISE=1 TARG v(WL)" in deck


def test_the_sized_driver_grows_with_the_wordline_load():
    def sized(fins, bits, **kw):
        c = RomReadConditions(bits=bits, driver="sized", **kw)
        return decoder_driver(fins, 256, c)

    # The wordline inverter's NMOS fins follow the bits and their fins.
    for predecode in (False, True):
        assert (
            sized(1, 256, predecode=predecode)[2] > sized(1, 16, predecode=predecode)[2]
        )
        assert (
            sized(2, 64, predecode=predecode)[2] > sized(1, 64, predecode=predecode)[2]
        )
    # Without predecoders a minimal decoder sizes the driver, widening the
    # interleave when it must; with them, the whole 512-word decoder sizes
    # its predecoders too, and for 2 fins x 256 bits they never converge.
    assert sized(2, 256, predecode=False)[2] > sized(1, 256, predecode=False)[2]
    with pytest.raises(ValueError, match="diverges"):
        sized(2, 256)
    assert decoder_driver(1, 64, RomReadConditions(driver="ideal")) is None
    explicit = RomReadConditions(driver=(2, 2, 4, 4))
    assert decoder_driver(1, 64, explicit) == (2, 2, 4, 4)


def test_the_predecoders_drive_the_words_nand(tmp_path: Path):
    deck = deck_for(tmp_path, words=256)
    pre = predecode_network(
        decoder_spec(1, 256, RomReadConditions()),
        0,
        nmos="nmos_rvt",
        pmos="pmos_rvt",
        wire_ff_per_um=0.2,
    )
    # Word 0 of the 512-word decoder: PA<0> and PB<0>, every address bit 0,
    # so all nine fall and each AN<i> inverter is on the path.
    assert len(pre.sources) == 9 and not any(up for _, up in pre.sources)
    assert {f"AN<{i}>" for i in range(9)} <= set(pre.tree)
    assert {"PA<0>", "PB<0>"} <= set(pre.tree)
    drv = {d[0]: d for d in devices(deck) if d[0].startswith("ND")}
    assert drv["ND1"][2] == pre.pa_pin and drv["ND2"][2] == pre.pb_pin
    # The address pins are the edges; nothing drives PA or PB directly.
    assert "Vpa " not in deck and "Vwl " not in deck
    assert sum(x.startswith("Va") for x in deck.splitlines()) == 9
    assert f".measure tran t_sense TRIG v({pre.sources[0][0]}) VAL=0.35 FALL=1" in deck
    # Every predecoder gate on the path, and an off-path gate load per input.
    gates = {d[0][:-1] for d in devices(deck) if d[0].startswith("NX")}
    assert len(gates) == pre.gates and pre.loads > 0
    assert sum(d[0].startswith("NL") for d in devices(deck)) == 2 * pre.loads


def test_predecoded_benches_need_a_decoder_word_count(tmp_path: Path):
    with pytest.raises(ValueError, match="power-of-two"):
        deck_for(tmp_path, words=48)
    assert "Vpa PA" in deck_for(tmp_path, words=48, predecode=False)


@pytest.mark.parametrize("driver", ["strong", (1, 2, 3), (0, 1, 1, 1)])
def test_rejects_bad_drivers(driver):
    with pytest.raises(ValueError):
        RomReadConditions(driver=driver)


def test_sense_target_counts_from_the_precharged_level(tmp_path: Path):
    paths = {"models": tmp_path / "m", "osdi": tmp_path / "o", "conditions": CONDITIONS}
    first = render_rom_read_deck(1, 16, **paths)
    second = render_rom_read_deck(1, 16, precharged_v=0.734, **paths)
    assert "TARG v(BLP) VAL=0.6 FALL=1" in first
    assert "TARG v(BLP) VAL=0.634 FALL=1" in second
    assert "FIND v(BLU) WHEN v(BLP)=0.634 FALL=1" in second
    assert "FIND v(BLU) WHEN v(BLP)=0.35 FALL=1" in second
    assert "pre_osdi" in second and second.rstrip().endswith(".end")


def test_vt_flavour_selects_its_models(tmp_path: Path):
    models = {d[5] for d in devices(deck_for(tmp_path, vt="slvt"))}
    assert models == {"nmos_slvt", "pmos_slvt"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vdd": 0},
        {"vt": "sram"},
        {"sense_margin_v": 0.4},
        {"min_sense_split": 0},
        {"step_ps": 0},
        {"wire_ff_per_um": -1},
        {"wire_ohm_per_um": -1},
        {"precharge_fins": True},
        {"bits": 0},
        {"strap_every": 0},
    ],
)
def test_rejects_bad_conditions(kwargs):
    with pytest.raises(ValueError):
        RomReadConditions(**kwargs)


@pytest.mark.parametrize(("fins", "words"), [(0, 16), (1, 0), (True, 16)])
def test_rejects_bad_fins_and_depth(tmp_path: Path, fins, words):
    with pytest.raises(ValueError):
        render_rom_read_deck(
            fins,
            words,
            models=tmp_path / "m",
            osdi=tmp_path / "o",
            conditions=CONDITIONS,
        )


def test_one_fin_reads_slower_than_two_and_depth_slows_both(
    tmp_path: Path, external_tool
):
    card, source = find_default_model_card(), find_default_bsimcmg107_source()
    tools = shutil.which("ngspice") and shutil.which("openvaf")
    external_tool(
        bool(tools and card and source),
        "ngspice, OpenVAF, the ASAP7 model card or the BSIM-CMG source is missing",
    )
    osdi = compile_bsimcmg107(source, tmp_path / "BSIMCMG107.osdi")
    results = run_rom_read(
        model_card=card,
        osdi=osdi,
        output_dir=tmp_path / "run",
        fins=(1, 2),
        words=(16, 256),
        # The ideal driver keeps this about the bitline and the wordline.
        conditions=RomReadConditions(driver="ideal"),
    )
    assert (tmp_path / "run/results.json").is_file()
    r = {(x.fins, x.words): x for x in results}
    assert all(x.passed_sense and x.passed_inverter for x in results)
    for fins in (1, 2):
        assert r[fins, 256].t_sense_s > 2 * r[fins, 16].t_sense_s
        assert r[fins, 256].t_half_s > r[fins, 256].t_sense_s
    # One fin pulls less current, so a long line reads slower...
    assert r[1, 256].t_sense_s > r[2, 256].t_sense_s
    for words in (16, 256):
        # ...but twice the fins is twice the gate load on the wordline.
        assert r[2, words].t_wordline_s > r[1, words].t_wordline_s
        # The poly wordline delays the gate, a little.
        assert 0 < r[1, words].t_wordline_s < r[1, words].t_sense_s
        # The released precharge kicks both lines above VDD.
        assert r[1, words].blp_precharged_v > CONDITIONS.vdd
