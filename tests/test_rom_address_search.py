"""The address-step search: the generalised step, its deck, and its logic.

The search's own simulations are stubbed; the decks it writes are real.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from chipforge_asap7.devices import rom_address_search as search
from chipforge_asap7.devices.rom_predecode_spice import predecode_network
from chipforge_asap7.devices.rom_read_spice import (
    RomReadConditions,
    decoder_spec,
    render_decoder_deck,
    render_rom_read_deck,
)

C = RomReadConditions(bits=16)


def network(word, previous=None):
    spec = decoder_spec(1, 16, C)
    return predecode_network(
        spec,
        word,
        nmos="n",
        pmos="p",
        wire_ff_per_um=0.2,
        address_buffer="BUFx4",
        previous=previous,
    )


def test_only_the_bits_a_step_changes_get_edges():
    # A 32-word decoder: five address bits.  0b00101 from 0b00110 changes
    # bits 0 and 1; bits 2, 3 and 4 are held at word 5's level.
    net = network(0b00101, 0b00110)
    assert sorted(n for n, _ in net.sources) == ["Av0_in", "Av1_in"]
    assert dict(net.sources) == {"Av0_in": True, "Av1_in": False}
    assert dict(net.held) == {"Av2_in": True, "Av3_in": False, "Av4_in": False}
    # The default is the complement: every bit switches, none held.
    full = network(0b00101)
    assert len(full.sources) == 5 and full.held == () and full.previous == 0b11010


@pytest.mark.parametrize("previous", [5, -1, 32])
def test_a_step_needs_two_different_addresses(previous):
    with pytest.raises(ValueError, match="address step"):
        network(5, previous)


def test_held_bits_are_dc_sources_in_the_read_deck(tmp_path: Path):
    c = RomReadConditions(bits=16, word=5, previous_word=6)
    deck = render_rom_read_deck(
        1, 16, models=tmp_path / "m", osdi=tmp_path / "o", conditions=c
    )
    assert sum(x.startswith("Va") for x in deck.splitlines()) == 2
    assert {
        x.split()[1]: x.split()[3] for x in deck.splitlines() if x.startswith("Vh")
    } == {
        "Av2_in": "0.7",
        "Av3_in": "0",
        "Av4_in": "0",
    }


def test_the_decoder_deck_is_the_read_benchs_decoder(tmp_path: Path):
    paths = {"models": tmp_path / "m", "osdi": tmp_path / "o"}
    c = RomReadConditions(bits=16, word=3, previous_word=12)
    deck = render_decoder_deck(1, 16, conditions=c, **paths)
    read = render_rom_read_deck(1, 16, conditions=c, **paths)
    decoder = [
        x
        for x in deck.splitlines()
        if x[:2] in ("Va", "Vh", "NX", "ND", "NB", "RQ", "CC")
    ]
    assert decoder and all(x in read.splitlines() for x in decoder)
    # No bitlines and no wordline network: WL is a lumped load.
    assert "BLP" not in deck and "CWL WL 0 10f" in deck
    assert ".options method=gear rshunt=1e12" in deck
    with pytest.raises(ValueError, match="decoder driver"):
        render_decoder_deck(
            1, 16, conditions=RomReadConditions(driver="ideal"), **paths
        )


def test_the_search_screens_every_word_and_climbs_to_the_slowest_step(
    tmp_path: Path, monkeypatch
):
    # A stub decoder whose delay rises with word 9 and with flipping bits 1
    # and 3 but not bit 0: the climb must drop bit 0 from the complement.
    def fake_simulate(deck_path, text, ngspice):
        word, previous = (int(p[1:]) for p in deck_path.stem.split("_"))
        mask = word ^ previous
        t = 100 + 20 * (word == 9) + 5 * bool(mask & 2) + 5 * bool(mask & 8)
        t -= 3 * bool(mask & 1)
        return f"t_decoded = {t}e-12\nt_pa = 1e-10\nt_pb = 1e-10\n"

    reads = []

    def fake_read(**kwargs):
        reads.append(kwargs["conditions"])
        return [SimpleNamespace(t_sense_s=2e-10)]

    monkeypatch.setattr(search, "_simulate", fake_simulate)
    monkeypatch.setattr(search, "run_rom_read", fake_read)
    monkeypatch.setattr(
        search, "asdict", lambda x: {} if not hasattr(x, "word") else vars(x)
    )
    model = tmp_path / "card.pm"
    model.write_text("")
    monkeypatch.setattr(search, "translate_asap7_model", lambda card, name: "")
    result = search.search_address_steps(
        model_card=model,
        osdi=tmp_path / "o",
        output_dir=tmp_path / "out",
        words=16,
        conditions=RomReadConditions(bits=16),
        top_words=2,
        confirm=1,
        workers=2,
        log=lambda line: None,
    )
    assert len(result.words) == 32  # every word of the 32-word decoder
    worst = result.worst[0]
    assert worst.word == 9 and (worst.word ^ worst.previous) == 0b11110
    assert worst.t_decoded_s == pytest.approx(130e-12)
    # The winner and word 0 from its complement go through the full bench.
    assert [(c.word, c.previous_word) for c in reads] == [(9, 9 ^ 0b11110), (0, 31)]


def test_a_cut_short_search_reuses_its_finished_screens(tmp_path: Path, monkeypatch):
    runs = []

    def fake_simulate(deck_path, text, ngspice):
        runs.append(deck_path.stem)
        deck_path.write_text(text)
        log = "t_decoded = 1e-10\nt_pa = 1e-11\nt_pb = 1e-11\n"
        deck_path.with_suffix(".log").write_text(log)
        return log

    monkeypatch.setattr(search, "_simulate", fake_simulate)
    monkeypatch.setattr(
        search, "run_rom_read", lambda **kw: [SimpleNamespace(t_sense_s=2e-10)]
    )
    monkeypatch.setattr(
        search, "asdict", lambda x: vars(x) if hasattr(x, "word") else {}
    )
    monkeypatch.setattr(search, "translate_asap7_model", lambda card, name: "")
    model = tmp_path / "card.pm"
    model.write_text("")
    kwargs = {
        "model_card": model,
        "osdi": tmp_path / "o",
        "output_dir": tmp_path / "out",
        "words": 16,
        "conditions": RomReadConditions(bits=16),
        "top_words": 1,
        "confirm": 1,
        "workers": 1,
        "log": lambda line: None,
    }
    search.search_address_steps(**kwargs)
    first = len(runs)
    search.search_address_steps(**kwargs)
    assert first > 32 and len(runs) == first  # the second search ran nothing
