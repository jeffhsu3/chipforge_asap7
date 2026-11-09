"""The NOR ROM's read: the reduced netlist, the deck, and judging a simulation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from chipforge_asap7.devices.rom_bitcell import RomArraySpec
from chipforge_asap7.devices.rom_decoder import RomDecoderSpec
from chipforge_asap7.devices.rom_netlist import rom_macro_devices
from chipforge_asap7.devices.rom_timing import (
    CORNERS,
    LOADS_FF,
    OPERATING_POINT_FAILED,
    REPORT_FORMAT,
    CornerTiming,
    ReadPath,
    Stimulus,
    Waveforms,
    circuit_digest,
    load_report,
    read_errors,
    read_path,
    read_sites,
    read_words,
    render_deck,
    search_limit,
    simulate,
    wire_estimates,
)
from chipforge_asap7.devices.rom_views import MEASURED_MARGIN, rom_macro_liberty
from chipforge_asap7.devices.spice import find_default_model_card

OSDI = Path("build/brom_timing/model/BSIMCMG107.osdi")


def _macro(words: int = 16, bits: int = 6) -> RomDecoderSpec:
    return RomDecoderSpec(
        array=RomArraySpec(words, bits),
        sense=True,
        enable=True,
        latch=True,
        sample=True,
        control=True,
        register=True,
        edge_pins=True,
    )


def _measured(max_low: float | None = 18000.0) -> CornerTiming:
    return CornerTiming(
        clk_q_rise=(90.0, 100.0, 130.0),
        clk_q_fall=(80.0, 90.0, 120.0),
        q_rise_slew=(10.0, 20.0, 50.0),
        q_fall_slew=(9.0, 19.0, 49.0),
        setup=-40.0,
        hold=60.0,
        min_low=800.0,
        min_high=200.0,
        droop=1.5,
        max_low=max_low,
    )


def _spice() -> Path:
    if shutil.which("ngspice") is None or not OSDI.is_file():
        pytest.skip("needs ngspice and a compiled BSIM-CMG OSDI library")
    if find_default_model_card() is None:
        pytest.skip("needs the ASAP7 model cards: set ASAP7_MODEL_CARD")
    return OSDI.resolve()


# ── The circuit the race came from ────────────────────────────────────────────
def test_sample_column_drives_one_inverter_per_row():
    """Every latch on the column left SAMPLE too slow at 1024x148 to close the
    register before the sample latch opened, and Q changed before CLK rose."""
    spec = _macro(64, 16)
    lane = spec.sense_lane
    devices = rom_macro_devices(spec, ())
    leaves = [d for d in devices if d.name.startswith("MS_") and d.gate == "SAMPLE"]
    rows = [r for r in range(lane.rows) if lane.row_leaves(r)]
    assert {d.name for d in leaves} == {
        f"MS_{device}_R{r}" for r in rows for device in ("MPS", "MNS")
    }


def test_latches_clock_from_their_rows_buffers():
    spec = _macro(64, 16)
    by_name = {d.name: d for d in rom_macro_devices(spec, ())}
    row = spec.sense_lane.plan[0].row
    # The two latches conduct on opposite levels of the row's own clocks.
    assert by_name["MS_MNT_B0"].gate == f"S_SAMPLEL_R{row}"
    assert by_name["MS_MPT_B0"].gate == f"S_SAMPLEB_R{row}"
    assert by_name["MS_MRNT_B0"].gate == f"S_SAMPLEB_R{row}"
    assert by_name["MS_MRPT_B0"].gate == f"S_SAMPLEL_R{row}"
    assert by_name[f"MS_MNC_R{row}"].gate == f"S_SAMPLEB_R{row}"
    assert by_name[f"MS_MNC_R{row}"].drain == f"S_SAMPLEL_R{row}"


# ── The reduced netlist ───────────────────────────────────────────────────────
def test_reading_far_words_with_the_far_bit_cleared_in_one():
    spec = _macro()
    assert read_words(spec) == (0, 15)
    sites = read_sites(spec)
    assert (0, 0) in sites and (15, 0) not in sites
    assert len(sites) == 16 * 6 - 1


def test_reduction_keeps_the_read_and_every_removed_fin_on_a_kept_gate():
    spec = _macro(64, 16)
    sites = read_sites(spec)
    a, b = read_words(spec)
    path = read_path(spec, sites, (a, b), (0,))
    every = rom_macro_devices(spec, sites)
    assert len(path.devices) + path.removed == len(every)
    names = {d.name for d in path.devices}
    assert {"MS_MNT_B0", "MS_MRNT_B0"} <= names
    assert "MS_MNT_B1" not in names
    kept = {n for d in path.devices for n in (d.drain, d.gate, d.source)} - {
        "VDD",
        "VSS",
    }
    removed = [d for d in every if d.name not in names]
    assert {
        net: sum(x.fins for x in path.loads if x.net == net and x.kind == "gate")
        for net in kept
    } == {net: sum(d.fins for d in removed if d.gate == net) for net in kept}


def test_each_long_net_has_a_wire():
    spec = _macro(64, 16)
    wires = wire_estimates(spec)
    assert all(ohms > 0 and ff > 0 for ohms, ff in wires.values())
    assert any(pattern.startswith("S_SAMPLE[BL]") for pattern in wires)


def test_deck_names_each_device_once_and_drives_every_input():
    spec = _macro()
    path = read_path(spec, read_sites(spec), read_words(spec), (0,))
    stimulus = Stimulus(read_words(spec))
    deck, nodes = render_deck(
        spec, path, stimulus, CORNERS["typical"], Path("x.osdi"), ["Q[0]", "CLK"]
    )
    instances = [line.split()[0] for line in deck.splitlines() if line.startswith("N")]
    assert len(instances) == len(set(instances)) >= len(path.devices)
    for source in ("VCLK", "VRESETN", *(f"VA{i}" for i in range(spec.address_bits))):
        assert any(line.startswith(f"{source} ") for line in deck.splitlines())
    assert len(nodes) == 2
    # A zero load is no capacitor rather than a 0 F one.
    unloaded, _ = render_deck(
        spec, path, Stimulus(read_words(spec), load_ff=0.0), CORNERS["typical"],
        Path("x.osdi"), ["Q[0]"],
    )  # fmt: skip
    assert not any(line.startswith("CQ") for line in unloaded.splitlines())


def test_stimulus_address_can_arrive_after_the_clock_and_turn_after_hold():
    late = Stimulus((0, 1), setup=-30.0, hold=None, slew=4.0)
    assert late.first_fall == 2 * late.high + 100.0
    held = Stimulus((0, 1), setup=100.0, hold=40.0, tail=70.0)
    assert held.stop == held.fall(2) + held.high + 70.0


def test_deck_keeps_a_stop_time_of_any_length():
    """A rounded stop time made searched runs look unfinished, as false failures."""
    spec = _macro()
    path = read_path(spec, read_sites(spec), read_words(spec), (0,))
    stimulus = Stimulus(read_words(spec), low=343.5714285, high=400.0)
    deck, _ = render_deck(
        spec, path, stimulus, CORNERS["typical"], Path("x.osdi"), ["Q[0]"]
    )
    tran = next(line for line in deck.splitlines() if line.startswith("tran "))
    assert float(tran.split()[2].removesuffix("p")) == pytest.approx(
        stimulus.stop, abs=1e-3
    )


def test_deck_can_give_q_a_first_guess_for_the_operating_point():
    """Only with the retry asks for it: the guess must not touch a normal run."""
    spec = _macro()
    path = read_path(spec, read_sites(spec), read_words(spec), (0,))
    stimulus = Stimulus(read_words(spec))
    corner = CORNERS["fast"]
    plain, _ = render_deck(spec, path, stimulus, corner, Path("x.osdi"), ["Q[0]"])
    assert ".nodeset" not in plain
    guessed, _ = render_deck(
        spec, path, stimulus, corner, Path("x.osdi"), ["Q[0]"], q_nodeset=1.0
    )
    nodesets = [line for line in guessed.splitlines() if line.startswith(".nodeset")]
    assert nodesets and all(line.endswith(f"={corner.vdd:g}") for line in nodesets)
    assert OPERATING_POINT_FAILED.search(
        "Error: The operating point could not be simulated successfully."
    )


# ── Judging a read ────────────────────────────────────────────────────────────
def _waves(stimulus: Stimulus, values: list[tuple[float, float]]) -> Waveforms:
    time = np.array([t for t, _ in values])
    return Waveforms(time, {"Q[0]": np.array([v for _, v in values])}, 0.7)


def test_a_read_is_judged_when_the_next_clock_rises():
    s = Stimulus((0, 1, 0))
    right = [
        (0.0, 0.0),
        (s.rise(0) + 50, 0.0),
        (s.rise(0) + 90, 0.7),
        (s.rise(1) + 50, 0.7),
        (s.rise(1) + 90, 0.0),
        (s.rise(2) + 50, 0.0),
        (s.rise(2) + 90, 0.7),
        (s.stop, 0.7),
    ]
    assert read_errors(_waves(s, right), s, (1, 0, 1)) == []


def test_q_changing_before_the_clock_rises_is_an_error():
    """The race this module found: Q takes read 1's word while CLK is still low."""
    s = Stimulus((0, 1, 0))
    early = [
        (0.0, 0.0),
        (s.rise(0) + 50, 0.0),
        (s.rise(0) + 90, 0.7),
        (s.rise(1) - 100, 0.7),
        (s.rise(1) - 50, 0.0),
        (s.rise(2) + 50, 0.0),
        (s.rise(2) + 90, 0.7),
        (s.stop, 0.7),
    ]
    errors = read_errors(_waves(s, early), s, (1, 0, 1))
    assert len(errors) == 1 and errors[0].startswith("read 0:")


def test_q_keeping_the_old_word_is_an_error():
    s = Stimulus((0, 1))
    stuck = [(0.0, 0.7), (s.stop, 0.7)]
    assert [e[:6] for e in read_errors(_waves(s, stuck), s, (1, 0))] == ["read 1"]


# ── Searching for a limit ─────────────────────────────────────────────────────
def test_search_finds_the_limit_within_its_resolution():
    tried = []

    def passes(values):
        tried.append(len(values))
        return [v >= 123.0 for v in values]

    found = search_limit(passes, 0.0, 800.0, points=6, resolution=5.0)
    assert 123.0 <= found <= 128.0
    assert max(tried) == 6


def test_search_stays_above_a_failure_past_a_pass():
    """Passing at 100 but failing at 300 can't report anything under 300."""
    found = search_limit(
        lambda values: [v < 150.0 or v > 300.0 for v in values],
        0.0,
        800.0,
        points=6,
        resolution=5.0,
    )
    assert found > 300.0


# ── The report and its Liberty ────────────────────────────────────────────────
def _report(tmp_path: Path, spec: RomDecoderSpec, **changes) -> Path:
    from dataclasses import asdict

    report = {
        "format": REPORT_FORMAT,
        "macro": spec.macro_name,
        "circuit_sha256": circuit_digest(spec),
        "corners": {"typical": asdict(_measured())},
        "failed": {},
        **changes,
    }
    path = tmp_path / "timing.json"
    path.write_text(json.dumps(report))
    return path


def test_report_is_refused_for_another_shape(tmp_path: Path):
    spec = _macro()
    assert load_report(_report(tmp_path, spec), spec)["typical"] == _measured()
    with pytest.raises(ValueError, match="measured"):
        load_report(_report(tmp_path, spec), _macro(32, 6))
    with pytest.raises(ValueError, match="not a"):
        load_report(_report(tmp_path, spec, format="other"), spec)


def test_report_is_refused_once_the_circuit_has_changed(tmp_path: Path):
    """A shape's name doesn't change with its circuit: the buffer fix kept it."""
    spec = _macro()
    stale = _report(tmp_path, spec, circuit_sha256="0" * 64)
    with pytest.raises(ValueError, match="different circuit"):
        load_report(stale, spec)


def test_digest_tells_shapes_apart_but_not_contents():
    assert circuit_digest(_macro()) == circuit_digest(_macro())
    assert circuit_digest(_macro()) != circuit_digest(_macro(32, 6))


def test_measured_liberty_carries_the_numbers_with_margin():
    spec = _macro()
    text = rom_macro_liberty(spec, "m", "typical", 1.0, None, _measured())
    assert "SIMULATED, NOT SIGN-OFF" in text and "ESTIMATED" not in text
    assert f'index_1 ("{", ".join(f"{x:g}" for x in LOADS_FF)}")' in text
    assert f"{90 * MEASURED_MARGIN:.2f}, {100 * MEASURED_MARGIN:.2f}" in text
    assert f"min_pulse_width_low : {800 * MEASURED_MARGIN:.2f}" in text
    assert "about 18000 ps" in text
    # A negative setup is floored, not made earlier by its margin.
    assert '"10.00"' in text
    assert f'"{60 * MEASURED_MARGIN:.2f}"' in text


def test_estimated_liberty_still_says_so():
    text = rom_macro_liberty(_macro(), "m", "typical", 1.0)
    assert "ESTIMATED, NOT CHARACTERIZED" in text and "rom_q_load" not in text


def test_no_droop_limit_is_said_plainly():
    text = rom_macro_liberty(_macro(), "m", "fast", 1.0, None, _measured(None))
    assert "no limit found" in text


# ── A real simulation ─────────────────────────────────────────────────────────
def test_reduced_read_agrees_with_the_flat_macro(tmp_path: Path):
    """Both read both ways, and the reduction moves no edge by more than 8%."""
    osdi = _spice()
    spec = _macro()
    sites = read_sites(spec)
    a, b = read_words(spec)
    stimulus = Stimulus((a, b, a), low=400.0, high=400.0)
    watch = ["Q[0]", f"WL<{b}>", "READY", "SAMPLE"]
    corner = CORNERS["typical"]
    flat = simulate(
        spec, ReadPath(rom_macro_devices(spec, sites), (), 0), stimulus, corner, osdi,
        watch, tmp_path / "flat", wires=False,
    )  # fmt: skip
    reduced = simulate(
        spec, read_path(spec, sites, (a, b), (0,)), stimulus, corner, osdi, watch,
        tmp_path / "reduced", wires=False,
    )  # fmt: skip
    assert read_errors(flat, stimulus, (1, 0, 1)) == []
    assert read_errors(reduced, stimulus, (1, 0, 1)) == []
    start = stimulus.fall(1)
    for net in watch[1:]:
        ours = reduced.crossing(net, start, stimulus.rise(1), rising=True) - start
        theirs = flat.crossing(net, start, stimulus.rise(1), rising=True) - start
        assert abs(ours - theirs) / theirs < 0.08, net


def test_wire_rc_read_holds_q_until_the_clock_rises(tmp_path: Path):
    osdi = _spice()
    spec = _macro(64, 16)
    a, b = read_words(spec)
    stimulus = Stimulus((a, b, a), low=500.0, high=500.0)
    path = read_path(spec, read_sites(spec), (a, b), (0,))
    w = simulate(
        spec, path, stimulus, CORNERS["typical"], osdi, ["Q[0]"], tmp_path / "rc"
    )
    assert read_errors(w, stimulus, (1, 0, 1)) == []
    assert w.at("Q[0]", stimulus.rise(1)) > 0.8 * w.vdd
