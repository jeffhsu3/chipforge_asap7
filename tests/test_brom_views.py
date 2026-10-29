"""LEF abstract and Verilog model of the BROM macro, checked against its build.

The Verilog half runs the model under Icarus Verilog and compares it, step by
step, with the transistor switch-level simulation in `test_brom`; it skips
when iverilog is missing.
"""

import json
import random
import re
import shutil
import subprocess

import gdspy
import pytest
from test_brom import settle

from chipforge_asap7.brom import (
    BromSpec,
    build_macro,
    clocked_macro_circuit,
    macro_circuit,
    self_timed_macro_circuit,
)
from chipforge_asap7.brom.layout import macro_periphery, macro_plan
from chipforge_asap7.brom.periphery import bus_positions
from chipforge_asap7.brom.views import macro_lef, macro_verilog

SHAPES = {
    "phases": macro_circuit,
    "clocked": clocked_macro_circuit,
    "self_timed": self_timed_macro_circuit,
}


def words_for(spec):
    return [(i * 37 + 5) % (1 << spec.bits) for i in range(spec.depth)]


def lef_pins(text):
    pins = {}
    for name, body in re.findall(r"PIN (\S+)\n(.*?)END \1\n", text, re.DOTALL):
        rects = [
            tuple(round(float(v) * 1000) for v in r)
            for r in re.findall(r"RECT (\S+) (\S+) (\S+) (\S+) ;", body)
        ]
        pins[name] = (re.search(r"DIRECTION (\w+)", body)[1], rects)
    return pins


@pytest.mark.parametrize("mode", SHAPES)
def test_lef_pins_reach_the_left_edge_under_their_labels(mode):
    spec = BromSpec(12, 3)
    c = SHAPES[mode](spec, words_for(spec))
    top = build_macro(c, lib=gdspy.GdsLibrary(unit=1e-9, precision=1e-10))
    text = macro_lef(c, top)
    (x0, y0), (x1, y1) = top.get_bounding_box()
    size = re.search(r"SIZE (\S+) BY (\S+) ;", text)
    assert (float(size[1]), float(size[2])) == ((x1 - x0) / 1000, (y1 - y0) / 1000)
    pins = lef_pins(text)
    expected = {re.sub(r"<(\d+)>", r"[\1]", p) for p in c.pins}
    assert set(pins) == expected
    buses = bus_positions(macro_periphery(c))
    bank_y = macro_plan(c).bank_y
    for pin in c.pins:
        direction, rects = pins[re.sub(r"<(\d+)>", r"[\1]", pin)]
        assert direction == (
            "OUTPUT"
            if pin.startswith("Q")
            else "INOUT"
            if pin[1:] == "SS" or pin == "VDD"
            else "INPUT"
        )
        lx, ly = 24 - x0, bank_y + buses[pin] - y0
        assert any(r[0] <= lx <= r[2] and r[1] <= ly <= r[3] for r in rects)
        assert any(r[0] == 0 for r in rects), f"{pin} does not reach the edge"
    # No pin shape is also reported as an obstruction.
    obs = text[text.index("  OBS") :]
    for _, rects in pins.values():
        for r in rects:
            assert (
                "RECT {:.3f} {:.3f} {:.3f} {:.3f} ;".format(*(v / 1000 for v in r))
                not in obs
            )


def test_klayout_reads_the_lef(tmp_path):
    kdb = pytest.importorskip("klayout.db")
    spec = BromSpec(16, 2, ways=4)
    c = macro_circuit(spec, words_for(spec))
    top = build_macro(c, lib=gdspy.GdsLibrary(unit=1e-9, precision=1e-10))
    path = tmp_path / "macro.lef"
    path.write_text(macro_lef(c, top))
    layout = kdb.Layout()
    layout.read(str(path))
    cell = layout.cell(c.name)
    (x0, y0), (x1, y1) = top.get_bounding_box()
    assert (cell.bbox().width(), cell.bbox().height()) == (x1 - x0, y1 - y0)
    layers = {layout.get_info(i).name for i in layout.layer_indexes()}
    assert {"M4.PIN", "M1.OBS", "M4.OBS", "M5.OBS"} <= layers


def test_verilog_rejects_other_words():
    spec = BromSpec(8, 2)
    c = macro_circuit(spec, [0] * 8)
    with pytest.raises(ValueError, match="do not match"):
        macro_verilog(c, spec, [1] * 8)


def acknowledgments(state):
    """What each completion input stands for, read off the settled circuit."""
    candidates = any(v for n, v in state.items() if n.startswith("C<"))
    return {
        "PRECHN_FAR": state.get("PRECHN", 0),
        "READ_READY": state.get("EVAL", 0),
        "SAMPLE_CLOSED": 1 - state.get("SAMPLE", 0),
        "CANDIDATE_BUSY": int(bool(state.get("EVAL") or candidates)),
    }


def drive(c, mode, spec, steps, seed, watched):
    """Random reads through the switch model: vectors and the `watched` nets
    they settle to.

    Addresses are random, out-of-range codes included when depth is not a
    power of two; so are CLK, RESETN and the phases. Only the orderings the
    switch model needs are kept, since it cannot resolve a crowbar or a race:
    evaluation ends before precharge, acknowledgments follow the circuit
    rather than arriving with the clock, and an internally driven PRECHN
    never opens the address latches onto a value they do not hold.
    """
    rng = random.Random(seed)
    ideal = () if mode == "phases" else ("PRECHN", "SAMPLE")
    state, vectors, expected = {}, [], []

    def step(vector):
        nonlocal state
        state = settle(c, vector, state or None, ideal_controls=ideal)
        vectors.append(vector)
        expected.append([state.get(net) for net in watched])

    for _ in range(steps):
        address = rng.randrange(1 << spec.address_bits)
        vector = {f"A<{i}>": address >> i & 1 for i in range(spec.address_bits)}
        if mode == "phases":
            prechn, evaluate, sample = (rng.randrange(2) for _ in range(3))
            if state.get("PRECHN") and state.get("EVAL") and not prechn:
                step({**vectors[-1], "EVAL": 0})
            vector.update(PRECHN=prechn, EVAL=evaluate, SAMPLE=sample)
            step(vector)
            continue
        if vectors:
            # PRECHN is internal here, so the address changes on a step of
            # its own, and closed latches see their own value again before
            # they can reopen.
            step({**vectors[-1], **vector})
            if state["PRECHN"]:
                held = {k: state.get(f"ADDR<{k[2:-1]}>", v) for k, v in vector.items()}
                step({**vectors[-1], **held})
            vector = dict(vectors[-1])
        elif mode == "clocked":
            vector.update(acknowledgments({}))
        vector.update(CLK=rng.randrange(2), RESETN=rng.randrange(2))
        step(vector)
        while mode == "clocked" and acknowledgments(state) != {
            k: vectors[-1][k] for k in acknowledgments(state)
        }:
            step({**vectors[-1], **acknowledgments(state)})
    return vectors, expected


@pytest.mark.parametrize(
    "mode,spec",
    [
        ("phases", BromSpec(12, 3)),
        ("phases", BromSpec(32, 2, ways=4)),
        ("clocked", BromSpec(12, 3)),
        ("clocked", BromSpec(16, 2, ways=4)),
        ("self_timed", BromSpec(8, 2)),
    ],
)
def test_verilog_model_matches_transistors(mode, spec, tmp_path, external_tool):
    iverilog = shutil.which("iverilog")
    external_tool(iverilog is not None and shutil.which("vvp"), "iverilog is missing")
    words = words_for(spec)
    c = SHAPES[mode](spec, words)
    # Q, and the phases too where the macro makes its own.
    watched = [f"Q<{b}>" for b in range(spec.bits)]
    if mode != "phases":
        watched += ["PRECHN", "EVAL", "SAMPLE"]
    seed = spec.depth * 7 + len(mode)
    vectors, expected = drive(c, mode, spec, 300, seed, watched)
    probes = [
        f"Q[{net[2:-1]}]" if net.startswith("Q<") else f"dut.{net}" for net in watched
    ]

    scalars = sorted({k for v in vectors for k in v if not k.startswith("A<")})
    bench = [
        "module tb;",
        f"  reg [{spec.address_bits - 1}:0] A;",
        *(f"  reg {name};" for name in scalars),
        f"  wire [{spec.bits - 1}:0] Q;",
        f"  {c.name} dut (.A(A), "
        + ", ".join(f".{n}({n})" for n in scalars)
        + ", .Q(Q));",
        "  initial begin",
    ]
    for vector in vectors:
        address = sum(vector[f"A<{i}>"] << i for i in range(spec.address_bits))
        bench.append(
            f"    A = {address};"
            + "".join(f" {n} = {vector[n]};" for n in scalars)
            + f' #1 $display("S={" %b" * len(probes)}", {", ".join(probes)});'
        )
    bench += ["    $finish;", "  end", "endmodule", ""]
    (tmp_path / "rom.v").write_text(macro_verilog(c, spec, words))
    (tmp_path / "tb.v").write_text("\n".join(bench))
    sim = tmp_path / "tb.vvp"
    subprocess.run(
        [iverilog, "-o", str(sim), str(tmp_path / "tb.v"), str(tmp_path / "rom.v")],
        check=True,
    )
    lines = [
        line.removeprefix("S=").split()
        for line in subprocess.run(
            ["vvp", "-n", str(sim)], check=True, capture_output=True, text=True
        ).stdout.splitlines()
        if line.startswith("S=")
    ]
    assert len(lines) == len(vectors)
    compared = 0
    for step, (model, transistors) in enumerate(zip(lines, expected, strict=True)):
        for net, m, t in zip(watched, model, transistors, strict=True):
            if t is None:
                continue  # never sampled yet in the switch model
            assert m == str(t), f"step {step} {net}: model {m}, transistors {t}"
            compared += 1
    assert compared > len(vectors) * len(watched) // 2


def test_cli_writes_lef_and_verilog(tmp_path):
    from chipforge_asap7.brom.__main__ import main

    words = tmp_path / "words.json"
    words.write_text("[0, 1, 2, 3]")
    out = tmp_path / "out"
    main([str(words), "--bits", "2", "--out", str(out)])
    record = json.loads((out / "manifest.json").read_text())["artifacts"]["macro"]
    assert (out / record["lef"]).read_text().startswith("VERSION 5.8")
    assert f"module {record['cell']} " in (out / record["verilog"]).read_text()
