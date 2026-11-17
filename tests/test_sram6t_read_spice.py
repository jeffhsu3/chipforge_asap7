"""The 6T SRAM read bench: its decks, and one real run when the tools exist.

The run compiles BSIM-CMG 107 with OpenVAF and simulates it in ngspice; it
skips when either, the ASAP7 model card or the BSIM-CMG source is missing,
and fails instead under ``ASAP7_REQUIRE_TOOLS=1``.
"""

import re
import shutil
from pathlib import Path

import pytest

from chipforge_asap7.devices.spice import (
    compile_bsimcmg107,
    find_default_bsimcmg107_source,
    find_default_model_card,
)
from chipforge_asap7.devices.sram6t_read_spice import (
    Sram6tReadConditions,
    render_access_deck,
    render_read_deck,
    run_sram6t_read,
)

CONDITIONS = Sram6tReadConditions()


def devices(deck):
    return [line.split() for line in deck.splitlines() if line.startswith("N")]


def test_access_deck_is_one_cell_on_the_column_load(tmp_path: Path):
    deck = render_access_deck(
        256, models=tmp_path / "m.pm", osdi=tmp_path / "b.osdi", conditions=CONDITIONS
    )
    cell = devices(deck)
    assert len(cell) == 6
    fins = sorted((d[5], d[-1]) for d in cell)
    assert fins == [("nmos_sram", "NFIN=2")] * 4 + [("pmos_sram", "NFIN=1")] * 2
    assert "Cbl BL 0 15.36f" in deck and "Cbln BLN 0 15.36f" in deck
    assert re.search(r"TARG v\(BL\) VAL=0\.6 FALL=1", deck)
    assert "pre_osdi" in deck and deck.rstrip().endswith(".end")


def test_read_deck_has_the_full_path_and_replica_timed_enable(tmp_path: Path):
    deck = render_read_deck(
        512,
        sae_after_clk_ps=189.2,
        models=tmp_path / "m.pm",
        osdi=tmp_path / "b.osdi",
        conditions=CONDITIONS,
    )
    top = [d for d in devices(deck) if d[0].startswith(("NL", "NH"))]
    assert len(top) == 12  # a stored-0 and a stored-1 cell
    assert deck.count(".subckt") == 4  # mux, latch NAND, buffer, sense amp
    assert "Vsae SAE 0 PULSE(0 0.7 1.1892n 0.04n 0.04n" in deck
    assert "Cq Q 0 46.08f" in deck and "Cbl BL 0 30.72f" in deck
    # The latch starts at 0, so the first rise is a real read.
    assert ".ic v(n49)=0 v(n54)=0.7" in deck
    for name in ("t_cell_rise", "t_cell_fall", "q_high", "q_low"):
        assert f".measure tran {name} " in deck


def test_custom_cell_sizing_is_rendered(tmp_path: Path):
    c = Sram6tReadConditions(pull_down_fins=3, pass_gate_fins=1, pull_up_fins=2)
    deck = render_access_deck(
        16, models=tmp_path / "m", osdi=tmp_path / "o", conditions=c
    )
    sized = {d[0]: d[-1] for d in devices(deck)}
    assert sized["NC1"] == sized["NC2"] == "NFIN=3"
    assert sized["NC0"] == sized["NC3"] == "NFIN=1"
    assert sized["NC4"] == sized["NC5"] == "NFIN=2"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vdd": 0},
        {"sense_margin_v": 0.8},
        {"slew_ps": 0},
        {"output_load_ff": -1},
        {"pull_up_fins": 0},
        {"pass_gate_fins": True},
    ],
)
def test_rejects_bad_conditions(kwargs):
    with pytest.raises(ValueError):
        Sram6tReadConditions(**kwargs)


def test_rejects_bad_depth_and_enable(tmp_path: Path):
    paths = {"models": tmp_path / "m", "osdi": tmp_path / "o", "conditions": CONDITIONS}
    with pytest.raises(ValueError, match="words"):
        render_access_deck(0, **paths)
    with pytest.raises(ValueError, match="SAE"):
        render_read_deck(64, sae_after_clk_ps=2500, **paths)


def test_read_delay_grows_with_column_depth(tmp_path: Path, external_tool):
    card, source = find_default_model_card(), find_default_bsimcmg107_source()
    tools = shutil.which("ngspice") and shutil.which("openvaf")
    external_tool(
        bool(tools and card and source),
        "ngspice, OpenVAF, the ASAP7 model card or the BSIM-CMG source is missing",
    )
    osdi = compile_bsimcmg107(source, tmp_path / "BSIMCMG107.osdi")
    short, tall = run_sram6t_read(
        model_card=card, osdi=osdi, output_dir=tmp_path / "run", words=(128, 512)
    )
    assert short.passed and tall.passed
    assert (tmp_path / "run/results.json").is_file()
    for r in (short, tall):
        assert 0.1e-9 < r.cell_rise_s < 0.5e-9 and 0.1e-9 < r.cell_fall_s < 0.5e-9
        assert r.rise_transition_s > 0 and r.fall_transition_s > 0
    # More cells on the bitline: slower access, and SAE waits for it.
    assert tall.bitline_access_s > 2 * short.bitline_access_s
    assert tall.cell_fall_s > short.cell_fall_s
