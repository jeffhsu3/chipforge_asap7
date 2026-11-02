"""The NOR-ROM read bench: its decks, and one real run when the tools exist.

The run compiles BSIM-CMG 107 with OpenVAF and simulates it in ngspice; it
skips when either, the ASAP7 model card or the BSIM-CMG source is missing,
and fails instead under ``ASAP7_REQUIRE_TOOLS=1``.
"""

import shutil
from pathlib import Path

import pytest

from chipforge_asap7.devices.rom_read_spice import (
    RomReadConditions,
    render_rom_read_deck,
    run_rom_read,
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
    assert selected[1:3] == ["BLP_63", "WL"]
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
    )
    assert (tmp_path / "run/results.json").is_file()
    r = {(x.fins, x.words): x for x in results}
    assert all(x.passed_sense and x.passed_inverter for x in results)
    for fins in (1, 2):
        assert r[fins, 256].t_sense_s > 2 * r[fins, 16].t_sense_s
        assert r[fins, 256].t_half_s > r[fins, 256].t_sense_s
    for words in (16, 256):
        assert r[1, words].t_sense_s > r[2, words].t_sense_s
        # The released precharge kicks both lines above VDD.
        assert r[1, words].blp_precharged_v > CONDITIONS.vdd
