"""The NOR ROM macro's edge pins and its LEF, Verilog and Liberty views.

The Verilog model runs in iverilog against the same stimulus as the switch-level
simulation of the extracted layout; the Liberty is read by OpenSTA and the LEF
by OpenROAD.  Those skip when the tool is missing; ``ASAP7_REQUIRE_TOOLS=1``
makes that a failure.
"""

from __future__ import annotations

import random
import re
import shutil
import subprocess
from pathlib import Path

import gdspy
import pytest
from test_rom_decoder import _extract, _settle

from chipforge_asap7.devices import RomDecoderSpec, build_rom
from chipforge_asap7.devices.rom_bitcell import RomArraySpec
from chipforge_asap7.devices.rom_netlist import (
    rom_macro_cdl,
    rom_macro_devices,
    rom_macro_ports,
)
from chipforge_asap7.devices.rom_pins import TRACK_OFFSET, TRACK_PITCH, plan_macro_pins
from chipforge_asap7.devices.rom_views import (
    MARGIN,
    estimate_timing,
    rom_macro_lef,
    rom_macro_liberty,
    rom_macro_verilog,
    rom_macro_views,
)
from chipforge_asap7.layout.layers import LAYERS, PIN_LAYERS
from chipforge_asap7.verification.lvs import run_lvs
from chipforge_asap7.verification.lvs_report import summarize_lvsdb
from chipforge_asap7.verification.rom_macro_lvs import (
    label_floating_drains,
    rom_macro_pins,
)

TECH_LEF = Path(__file__).resolve().parents[2] / (
    "asap7sc7p5t_28/techlef_misc/asap7_tech_1x_201209.lef"
)
SIZES = [(16, 6), (64, 32), (128, 33)]


def _spec(words=16, bits=6, register=True, edge_pins=True) -> RomDecoderSpec:
    return RomDecoderSpec(
        array=RomArraySpec(words, bits),
        sense=True,
        enable=True,
        latch=True,
        sample=True,
        register=register,
        control=True,
        edge_pins=edge_pins,
    )


def _sites(spec: RomDecoderSpec) -> set[tuple[int, int]]:
    rng = random.Random(spec.array.words * spec.array.bits)
    words, bits = spec.array.words, spec.array.bits
    return {(w, b) for w in range(words) for b in range(bits) if rng.random() < 0.5}


def _build(spec: RomDecoderSpec, sites=()):
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library, build_rom(spec, sites, lib=library)


# ── The pins ──────────────────────────────────────────────────────────────────
def test_edge_pins_need_the_controller():
    with pytest.raises(ValueError, match="edge_pins needs control"):
        RomDecoderSpec(array=RomArraySpec(16, 6), edge_pins=True)


def test_edge_pins_name_the_cell_apart():
    assert "_pins" in _spec().cell_name
    assert "_pins" not in _spec(edge_pins=False).cell_name


@pytest.mark.parametrize(("words", "bits"), [*SIZES, (256, 148), (1024, 148)])
def test_each_pin_is_a_column_on_an_m5_track_down_to_the_edge(words, bits):
    spec = _spec(words, bits)
    _, top = _build(spec)
    bottom = top.get_bounding_box()[0][1]
    labels = {
        (label.text, tuple(label.position))
        for label in top.get_labels(depth=0)
        if (label.layer, label.texttype) == PIN_LAYERS["M5"]
    }
    assert {text for text, _ in labels} == set(rom_macro_ports(spec))
    spec_m5 = (LAYERS["M5"]["layer"], LAYERS["M5"]["datatype"])
    columns = top.get_polygons(by_spec=spec_m5)
    for text, (x, y) in labels:
        assert (x - TRACK_OFFSET) % TRACK_PITCH == 0, text
        under = [
            p
            for p in columns
            if p[:, 0].min() <= x <= p[:, 0].max()
            and p[:, 1].min() <= y <= p[:, 1].max()
        ]
        assert len(under) == 1, text
        assert under[0][:, 1].min() == bottom, text
    assert len({x for _, (x, _) in labels}) == len(labels)


def test_the_interface_keeps_one_name_per_net():
    """``A<i>`` and ``Q<b>`` would be second names for the pins' nets."""
    spec = _spec()
    _, top = _build(spec)
    names = {label.text for label in top.get_labels()}
    assert not {n for n in names if re.fullmatch(r"(A|Q)<\d+>", n)}
    assert names == set(rom_macro_pins(spec))


def test_a_plan_leaves_the_macro_unchanged():
    spec = _spec()
    _, plain = _build(_spec(edge_pins=False))
    routes = plan_macro_pins(spec, plain)
    assert {r.port for r in routes} == set(rom_macro_ports(spec))


@pytest.mark.parametrize("register", [True, False], ids=["registered", "latched"])
@pytest.mark.parametrize(("words", "bits"), SIZES)
def test_edge_pins_are_drc_clean(words, bits, register, asap7_drc):
    library, top = _build(_spec(words, bits, register))
    assert set(asap7_drc(library, top, tag=top.name)) == set()


def _lvs(spec, sites, tmp_path, cdl=None, swap_labels=None):
    library, top = _build(spec, sites)
    label_floating_drains(spec, sites, top)
    if swap_labels:
        for label in top.labels:
            if label.text in swap_labels:
                label.text = swap_labels[label.text]
    gds = tmp_path / "rom.gds"
    library.write_gds(str(gds))
    schematic = tmp_path / "rom.cdl"
    schematic.write_text(cdl or rom_macro_cdl(spec, sites, name=top.name))
    return run_lvs(
        gds,
        schematic,
        tmp_path / "lvs",
        cell_name=top.name,
        asap7_standard_cell=True,
        tie_bodies=False,
        merge_fin_diffusion=True,
        flat=True,
    )


def _renamed(result) -> list:
    return [
        pair
        for circuit in summarize_lvsdb(result.report)["circuits"]
        for pair in circuit.get("renamed_pins", [])
    ]


@pytest.mark.parametrize("register", [True, False], ids=["registered", "latched"])
@pytest.mark.parametrize(("words", "bits"), SIZES)
def test_the_pins_connect_their_nets(words, bits, register, tmp_path, require_klayout):
    spec = _spec(words, bits, register)
    sites = _sites(spec)
    result = _lvs(spec, sites, tmp_path)
    assert result.matched
    assert _renamed(result) == []
    fins = sum(d.fins for d in rom_macro_devices(spec, sites))
    assert result.extracted_netlist.read_text().count("M$") == fins


def test_two_pins_labelled_crosswise_fail_by_name(tmp_path, require_klayout):
    spec = _spec()
    swap = {"Q[0]": "Q[1]", "Q[1]": "Q[0]"}
    result = _lvs(spec, _sites(spec), tmp_path, swap_labels=swap)
    assert sorted(_renamed(result)) == [["Q[0]", "Q[1]"], ["Q[1]", "Q[0]"]]


def test_every_pin_column_is_on_its_own_net(tmp_path):
    """LVS can't say this: it reads a label only as a name, so a VDD column
    labelled VSS still matches.  The extractor holds each name to one net, and
    no two names to the same one, across every M1 rail and M5 column."""
    spec = _spec()
    _, net = _extract(tmp_path, spec, _sites(spec))
    assert set(rom_macro_ports(spec)) <= net.keys()


def test_a_supply_column_labelled_the_other_supply_is_caught(tmp_path, monkeypatch):
    spec = _spec()

    def mislabelled(*args, **kwargs):
        top = build_rom(*args, **kwargs)
        column = next(
            label
            for label in top.labels
            if label.text == "VDD" and (label.layer, label.texttype) == PIN_LAYERS["M5"]
        )
        column.text = "VSS"
        return top

    monkeypatch.setattr("test_rom_decoder.build_rom", mislabelled)
    with pytest.raises(AssertionError):
        _extract(tmp_path, spec, _sites(spec))


# ── LEF ───────────────────────────────────────────────────────────────────────
def _lef_pins(lef: str) -> dict[str, list[tuple[float, ...]]]:
    pins = {}
    for name, body in re.findall(r"PIN (\S+)\n(.*?)  END \1\n", lef, re.DOTALL):
        pins[name] = [
            tuple(float(v) for v in rect)
            for rect in re.findall(r"RECT (\S+) (\S+) (\S+) (\S+) ;", body)
        ]
    return pins


def test_the_lef_pins_are_the_gds_columns_moved_to_its_origin():
    spec = _spec()
    _, top = _build(spec)
    lef = rom_macro_lef(spec, top)
    (x0, y0), (x1, y1) = top.get_bounding_box()
    assert f"SIZE {(x1 - x0) / 1000:.3f} BY {(y1 - y0) / 1000:.3f} ;" in lef
    assert f"FOREIGN {top.name} {x0 / 1000:.3f} {y0 / 1000:.3f} ;" in lef
    pins = _lef_pins(lef)
    assert list(pins) == list(rom_macro_ports(spec))
    routes = plan_macro_pins(spec, _build(_spec(edge_pins=False))[1])
    for route in routes:
        rx0, ry0, rx1, ry1 = route.column
        expected = (
            round((rx0 - x0) / 1000, 3),
            0.0,
            round((rx1 - x0) / 1000, 3),
            round((ry1 - y0) / 1000, 3),
        )
        assert ry0 == y0
        assert expected in pins[route.port], route.port


def test_the_lef_blocks_the_low_metal_and_only_what_is_drawn_above():
    spec = _spec()
    _, top = _build(spec)
    lef = rom_macro_lef(spec, top)
    obs = lef[lef.index("  OBS") :]
    (x0, y0), (x1, y1) = top.get_bounding_box()
    whole = f"RECT 0.000 0.000 {(x1 - x0) / 1000:.3f} {(y1 - y0) / 1000:.3f} ;"
    for metal in ("M1", "M2", "M3"):
        assert f"LAYER {metal} ;\n      {whole}" in obs
    assert whole not in obs.split("LAYER M4 ;")[1]
    # The replica's riser is M5 no pin owns; the pins' columns are not blocked.
    m5_obs = re.findall(r"RECT .*", obs.split("LAYER M5 ;")[1])
    pin_rects = [r for rects in _lef_pins(lef).values() for r in rects]
    assert m5_obs
    for rect in m5_obs:
        values = tuple(float(v) for v in rect.split()[1:5])
        assert values not in pin_rects


def test_the_views_need_edge_pins():
    spec = _spec(edge_pins=False)
    _, top = _build(spec)
    with pytest.raises(ValueError, match="need edge_pins"):
        rom_macro_lef(spec, top)
    with pytest.raises(ValueError, match="need edge_pins"):
        rom_macro_verilog(spec, (), "m")


def test_openroad_reads_the_lef(tmp_path, external_tool):
    openroad = shutil.which("openroad")
    external_tool(openroad is not None, "openroad is not installed")
    external_tool(TECH_LEF.exists(), f"no ASAP7 tech LEF at {TECH_LEF}")
    spec = _spec()
    _, top = _build(spec)
    lef = tmp_path / "macro.lef"
    lef.write_text(rom_macro_lef(spec, top))
    script = tmp_path / "read.tcl"
    script.write_text(
        f"read_lef {TECH_LEF}\nread_lef {lef}\n"
        "foreach lib [[ord::get_db] getLibs] { foreach m [$lib getMasters] {\n"
        '  puts "MASTER [$m getType] [llength [$m getMTerms]]" } }\nexit\n'
    )
    out = subprocess.run(
        [openroad, "-no_init", "-no_splash", "-exit", str(script)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert f"MASTER BLOCK {len(rom_macro_ports(spec))}" in out
    assert "ERROR" not in out


# ── Verilog ───────────────────────────────────────────────────────────────────
def _simulate(tmp_path: Path, model: str, name: str, steps, spec) -> list[str]:
    """Run ``(address, clk, resetn)`` steps through the model; Q after each."""
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    ab, bits = spec.address_bits, spec.array.bits
    stimulus = "\n".join(
        f"    A = {ab}'d{a}; CLK = {clk}; RESETN = {rn}; #10;\n"
        '    $display("Q=%b", Q);'
        for a, clk, rn in steps
    )
    (tmp_path / "tb.v").write_text(
        "`timescale 1ps/1ps\nmodule tb;\n"
        f"  reg [{ab - 1}:0] A; reg CLK; reg RESETN; wire [{bits - 1}:0] Q;\n"
        f"  {name} dut (.A(A), .CLK(CLK), .RESETN(RESETN), .Q(Q));\n"
        f"  initial begin\n{stimulus}\n    $finish;\n  end\nendmodule\n"
    )
    (tmp_path / "model.v").write_text(model)
    run = tmp_path / "tb.vvp"
    subprocess.run(
        [
            iverilog,
            "-g2005",
            "-o",
            str(run),
            str(tmp_path / "tb.v"),
            str(tmp_path / "model.v"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    out = subprocess.run([vvp, str(run)], check=True, capture_output=True, text=True)
    return re.findall(r"Q=([01xz]+)", out.stdout)


def _word(sites, address, bits) -> str:
    return "".join(str(int((address, b) in sites)) for b in reversed(range(bits)))


def test_the_model_registers_q_on_the_rising_clock(tmp_path, external_tool):
    external_tool(shutil.which("iverilog") is not None, "iverilog is not installed")
    spec = _spec()
    sites = _sites(spec)
    words = [a for a in range(16)]
    steps = [(0, 1, 1)]
    for address in words:
        steps += [(address, 0, 1), (address ^ 5, 1, 1)]
    q = _simulate(tmp_path, rom_macro_verilog(spec, sites, "m"), "m", steps, spec)
    for i, address in enumerate(words):
        assert q[1 + 2 * i + 1] == _word(sites, address, 6), address
        if i:  # while CLK is low Q still shows the read before
            assert q[1 + 2 * i] == _word(sites, words[i - 1], 6), address


def test_the_model_reads_nothing_in_reset(tmp_path, external_tool):
    external_tool(shutil.which("iverilog") is not None, "iverilog is not installed")
    spec = _spec()
    sites = {(3, b) for b in range(6)}
    steps = [(0, 1, 1), (3, 0, 1), (3, 1, 1), (9, 0, 0), (9, 1, 0), (9, 0, 0)]
    q = _simulate(tmp_path, rom_macro_verilog(spec, sites, "m"), "m", steps, spec)
    assert q[2:] == ["111111"] * 4


def test_a_latched_models_q_opens_while_clk_is_low(tmp_path, external_tool):
    external_tool(shutil.which("iverilog") is not None, "iverilog is not installed")
    spec = _spec(register=False)
    sites = _sites(spec)
    steps = [(0, 1, 1), (5, 0, 1), (5 ^ 5, 1, 1)]
    q = _simulate(tmp_path, rom_macro_verilog(spec, sites, "m"), "m", steps, spec)
    assert q[1] == q[2] == _word(sites, 5, 6)


@pytest.mark.parametrize("register", [True, False], ids=["registered", "latched"])
def test_the_model_matches_the_extracted_layout(register, tmp_path, external_tool):
    """Both see one random stimulus, resets and all, and agree on every Q."""
    external_tool(shutil.which("iverilog") is not None, "iverilog is not installed")
    spec = _spec(register=register, edge_pins=False)
    named = _spec(register=register)
    sites = _sites(spec)
    rng = random.Random(7)
    steps = [(0, 1, 1)]
    for _ in range(40):
        address = rng.randrange(16)
        resetn = int(rng.random() > 0.1)
        steps += [(address, 0, resetn), (rng.randrange(16), 1, resetn)]
    devices, net = _extract(tmp_path, spec, sites)
    supplies = {net["VDD"]: 1, net["VSS"]: 0}
    value = {}
    switch = []
    for address, clk, resetn in steps:
        pins = {
            **supplies,
            **{net[f"A<{i}>"]: (address >> i) & 1 for i in range(spec.address_bits)},
            net["CLK"]: clk,
            net["RESETN"]: resetn,
        }
        value = _settle(devices, net, pins, value)
        switch.append(
            "".join(
                "x" if value[net[f"Q<{b}>"]] is None else str(value[net[f"Q<{b}>"]])
                for b in reversed(range(6))
            )
        )
    model = rom_macro_verilog(named, sites, "m")
    q = _simulate(tmp_path, model, "m", steps, named)
    # Before its first read the model's Q is x, where the switch level holds
    # whatever its power-up left.
    known = [(s, m) for s, m in zip(switch, q, strict=True) if "x" not in s + m]
    assert len(known) > 60
    assert [m for _, m in known] == [s for s, _ in known]


def test_synthesis_sees_a_black_box():
    model = rom_macro_verilog(_spec(), {(0, 0)}, "m")
    head, body = model.split("`ifndef SYNTHESIS")
    assert "module m (A, CLK, RESETN, Q);" in head
    assert "output [5:0] Q;" in head
    assert body.rstrip().endswith("`endif\nendmodule")


# ── Liberty ───────────────────────────────────────────────────────────────────
def test_the_estimate_grows_with_the_macro():
    small, large = estimate_timing(_spec()), estimate_timing(_spec(256, 64))
    assert large.min_low > small.min_low
    assert large.min_high > small.min_high
    assert large.clk_q > small.clk_q
    assert large.clk_cap == small.clk_cap


def test_the_estimate_scales_by_corner_but_not_its_capacitance():
    t = estimate_timing(_spec())
    fast, slow = t.at("fast"), t.at("slow")
    assert fast.clk_q < t.clk_q < slow.clk_q
    assert fast.min_low < t.min_low < slow.min_low
    assert fast.clk_cap == t.clk_cap == slow.clk_cap
    assert t.setup >= 10 and t.hold >= 10 and MARGIN > 1


def test_a_column_loads_its_pin():
    bare = estimate_timing(_spec())
    loaded = estimate_timing(_spec(), {"CLK": 10.0, "Q[0]": 10.0})
    assert loaded.clk_cap == pytest.approx(bare.clk_cap + 2.0)
    assert loaded.clk_q > bare.clk_q


def test_the_liberty_says_it_is_an_estimate_and_needs_the_register():
    text = rom_macro_liberty(_spec(), "m", "typical", 1.0)
    assert "ESTIMATED, NOT CHARACTERIZED" in text
    assert "library(m_tt)" in text
    with pytest.raises(ValueError, match="needs register"):
        rom_macro_liberty(_spec(register=False), "m", "typical", 1.0)


def test_the_views_are_named_by_file():
    spec = _spec()
    _, top = _build(spec)
    assert set(rom_macro_views(spec, (), top)) == {
        "macro.lef",
        "macro.v",
        "macro_tt.lib",
        "macro_ff.lib",
        "macro_ss.lib",
    }
    latched = _spec(register=False)
    _, top = _build(latched)
    assert set(rom_macro_views(latched, (), top)) == {"macro.lef", "macro.v"}


def test_opensta_reads_the_liberty_and_checks_the_read(tmp_path, external_tool):
    sta = shutil.which("sta")
    external_tool(sta is not None, "OpenSTA is not installed")
    spec = _spec()
    _, top = _build(spec)
    views = rom_macro_views(spec, (), top)
    (tmp_path / "macro.lib").write_text(views["macro_tt.lib"])
    (tmp_path / "top.v").write_text(
        "module top(input clk, input rn, input [3:0] a, output [5:0] q);\n"
        f"  {top.name} u (.A(a), .CLK(clk), .RESETN(rn), .Q(q));\nendmodule\n"
    )
    script = tmp_path / "s.tcl"
    script.write_text(
        f"read_liberty {tmp_path / 'macro.lib'}\n"
        f"read_verilog {tmp_path / 'top.v'}\n"
        "link_design top\n"
        "create_clock -name c -period 1000 [get_ports clk]\n"
        "set_input_delay 20 -clock c -clock_fall [get_ports a]\n"
        "set_output_delay 20 -clock c [get_ports q]\n"
        "report_checks -to [get_ports q[0]] -format short\n"
        "report_checks -from [get_ports a[0]] -format short\n"
        "report_check_types -min_pulse_width -verbose\n"
    )
    out = subprocess.run(
        [sta, "-no_init", "-exit", str(script)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "Warning" not in out and "Error" not in out
    assert "Endpoint: q[0]" in out
    assert "Startpoint: a[0]" in out
    assert "required pulse width (low)" in out
