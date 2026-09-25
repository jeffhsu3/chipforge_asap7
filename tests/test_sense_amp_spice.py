"""Generated OpenFinRAM sense-amplifier BSIM-CMG fixture."""

from pathlib import Path

import pytest

from chipforge_asap7.devices.sense_amp_spice import (
    render_openfinram_sense_amp,
    render_sense_amp_deck,
)


def test_subcircuit_matches_openfinram_device_count_and_fin_sizes():
    subckt = render_openfinram_sense_amp()
    devices = [line for line in subckt.splitlines() if line.startswith("N")]
    pmos = [line for line in devices if " pmos_rvt " in line]
    nmos = [line for line in devices if " nmos_rvt " in line]

    assert len(devices) == 16
    assert len(pmos) == 8
    assert len(nmos) == 8
    assert all("NFIN=4" in line for line in pmos)
    assert all("NFIN=12" in line for line in nmos)
    assert all("L=20n" in line for line in devices)
    assert all(" W=" not in line.upper() for line in devices)


def test_sa_high_deck_has_correct_stimulus_and_timing(tmp_path: Path):
    deck = render_sense_amp_deck(
        sa_high=True,
        model_card=tmp_path / "models.pm",
        osdi=tmp_path / "model.osdi",
    )

    assert "Vsa sa 0 0.7" in deck
    assert "Vsan san 0 0.6" in deck
    assert "0.495n 0 0.5n 0.7" in deck
    assert "0.515n 0 0.52n 0.7" in deck
    assert "wrdata sa_high.dat" in deck
    assert "write sa_high.raw" in deck
    assert "SA > SAN -> QA high" in deck


def test_san_high_deck_reverses_only_the_input_polarity(tmp_path: Path):
    deck = render_sense_amp_deck(
        sa_high=False,
        model_card=tmp_path / "models.pm",
        osdi=tmp_path / "model.osdi",
    )

    assert "Vsa sa 0 0.6" in deck
    assert "Vsan san 0 0.7" in deck
    assert "wrdata san_high.dat" in deck


def test_custom_fin_counts_and_output_load_are_rendered(tmp_path: Path):
    deck = render_sense_amp_deck(
        sa_high=True,
        model_card=tmp_path / "models.pm",
        osdi=tmp_path / "model.osdi",
        n_fins=6,
        p_fins=2,
        output_load_ff=2.5,
    )

    assert deck.count("NFIN=6") == 8
    assert deck.count("NFIN=2") == 8
    assert "Cqa qa 0 2.5f" in deck
    assert "Cqan qan 0 2.5f" in deck


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vdd": 0},
        {"differential_v": 0},
        {"vdd": 0.7, "differential_v": 0.7},
        {"n_fins": 0},
        {"p_fins": 1.5},
        {"output_load_ff": -1},
        {"precharge_release_ns": 0},
        {"precharge_release_ns": 0.5, "enable_ns": 0.5},
        {"precharge_release_ns": 0.5, "enable_ns": 0.503},
        {"enable_ns": 0.52, "stop_ns": 0.5},
        {"step_ps": 0},
    ],
)
def test_rejects_invalid_bench_parameters(tmp_path: Path, kwargs):
    with pytest.raises(ValueError):
        render_sense_amp_deck(
            sa_high=True,
            model_card=tmp_path / "models.pm",
            osdi=tmp_path / "model.osdi",
            **kwargs,
        )
