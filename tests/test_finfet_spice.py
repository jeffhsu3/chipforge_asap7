"""Unit tests for the generated single-FinFET SPICE fixtures."""

from pathlib import Path

import pytest

from chipforge_asap7.devices import FinFETSpec
from chipforge_asap7.devices.spice import (
    render_characterization_deck,
    render_switch_deck,
    translate_asap7_model,
)

MODEL_CARD = """\
* synthetic two-model excerpt
.model nmos_rvt nmos level = 72
+version = 107 bulkmod = 1 phig = 4.372
+l = 2.1e-8 nbody = 1e22
.model pmos_rvt pmos level = 72
+version = 107 bulkmod = 1 phig = 4.810
"""


def test_translates_only_the_requested_level_72_model():
    translated = translate_asap7_model(MODEL_CARD, "nmos_rvt")

    assert ".model nmos_rvt bsimcmg DEVTYPE=1" in translated
    assert "bulkmod = 1" in translated
    assert "phig = 4.372" in translated
    assert "version" not in translated.lower()
    assert "level" not in translated.lower()
    assert "l =" not in translated.lower()
    assert "pmos_rvt" not in translated


def test_pmos_translation_uses_bsimcmg107_polarity_value():
    translated = translate_asap7_model(MODEL_CARD, "pmos_rvt")
    assert ".model pmos_rvt bsimcmg DEVTYPE=0" in translated


def test_missing_model_is_an_actionable_error():
    with pytest.raises(ValueError, match="nmos_lvt"):
        translate_asap7_model(MODEL_CARD, "nmos_lvt")


def test_characterization_deck_has_one_dut_and_two_dc_sweeps(tmp_path: Path):
    spec = FinFETSpec(fins=2, fingers=3)
    deck = render_characterization_deck(
        spec,
        model_card=tmp_path / "nmos_rvt.pm",
        osdi=tmp_path / "model.osdi",
    )

    assert deck.count("Ndut0 ") == 1
    assert "NFIN=2 L=20n NF=3" in deck
    assert "dc Vgate 0 0.7 0.01" in deck
    assert "dc Vdrain 0 0.7 0.01" in deck
    # Every node of each sweep is kept for a waveform viewer.
    assert "write transfer.raw" in deck
    assert "write output.raw" in deck
    assert "let drain_current_a = -i(Vdrain)" in deck
    assert "not GDS extraction/LVS" in deck


def test_multipliers_are_exact_parallel_instances(tmp_path: Path):
    spec = FinFETSpec(fins=2, fingers=3, multipliers=4)
    deck = render_characterization_deck(
        spec,
        model_card=tmp_path / "nmos_rvt.pm",
        osdi=tmp_path / "model.osdi",
    )

    for index in range(4):
        assert deck.count(f"Ndut{index} ") == 1
    assert " m=" not in deck.lower()


def test_switch_fixture_is_one_transistor_plus_a_load(tmp_path: Path):
    deck = render_switch_deck(
        FinFETSpec(),
        model_card=tmp_path / "nmos_rvt.pm",
        osdi=tmp_path / "model.osdi",
    )

    assert deck.count("Ndut0 ") == 1
    assert "Rload vdd d 100000" in deck
    assert "wrdata switch.dat v(d) supply_current_a" in deck
    assert "write switch.raw" in deck


def test_pmos_decks_reverse_the_biases(tmp_path: Path):
    spec = FinFETSpec(flavor="p")
    characterization = render_characterization_deck(
        spec,
        model_card=tmp_path / "pmos_rvt.pm",
        osdi=tmp_path / "model.osdi",
    )
    switch = render_switch_deck(
        spec,
        model_card=tmp_path / "pmos_rvt.pm",
        osdi=tmp_path / "model.osdi",
    )

    assert "Vsource s 0 0.7" in characterization
    assert "dc Vgate 0.7 0 -0.01" in characterization
    assert "let drain_current_a = i(Vdrain)" in characterization
    assert "Rload d 0 100000" in switch
    assert "Ndut0 d g vdd vdd pmos_rvt" in switch


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vdd": 0},
        {"step": 0},
        {"vdd": 0.7, "step": 0.8},
        {"load_ohms": 0},
    ],
)
def test_switch_rejects_invalid_sweep_values(tmp_path: Path, kwargs):
    with pytest.raises(ValueError):
        render_switch_deck(
            FinFETSpec(),
            model_card=tmp_path / "nmos_rvt.pm",
            osdi=tmp_path / "model.osdi",
            **kwargs,
        )
