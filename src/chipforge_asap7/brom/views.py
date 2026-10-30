"""LEF abstract and Verilog model of a compiled BROM macro.

Both views are derived from what the compiler already built: the LEF from the
macro's flattened GDS and pin labels, the Verilog from the circuit's pins and
the programmed words. Neither carries timing; Liberty waits on characterized,
extracted timing.

The macro's bus pins are already named ``A[3]``, as a gate netlist and DEF
name them, so GDS labels, CDL, LEF and Verilog all agree.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from ..layout.layers import LAYERS
from .compiler import BromSpec, selector_map
from .layout import macro_periphery, macro_plan
from .periphery import bus_positions

__all__ = ["macro_lef", "macro_verilog"]

_BUS = re.compile(r"^(\w+)\[(\d+)\]$")
#: Layers the router must treat as fully blocked over the macro.
COVERED = ("M1", "M2", "M3")
#: Layers whose drawn shapes are emitted exactly, so pins on M4 stay reachable.
EXACT = ("M4", "M5")


def _direction(pin: str) -> tuple[str, str]:
    if pin == "VDD":
        return "INOUT", "POWER"
    if pin == "VSS":
        return "INOUT", "GROUND"
    if pin.startswith("Q["):
        return "OUTPUT", "SIGNAL"
    return "INPUT", "SIGNAL"


def _um(nm: float) -> str:
    return f"{nm / 1000:.3f}"


def _rects(top: Any, layer: str) -> list[tuple[float, float, float, float]]:
    spec = (LAYERS[layer]["layer"], LAYERS[layer]["datatype"])
    polygons = top.get_polygons(by_spec=True).get(spec, [])
    rects = set()
    for points in polygons:
        lo, hi = points.min(axis=0), points.max(axis=0)
        if len(points) != 4:
            raise ValueError(f"non-rectangular {layer} shape in BROM macro")
        rects.add((float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])))
    return sorted(rects)


def macro_lef(circuit: Any, top: Any) -> str:
    """LEF 5.8 abstract of `top`, the cell `build_macro(circuit)` returned.

    Coordinates move so the boundary's lower-left corner is the LEF origin;
    ``FOREIGN`` gives where the GDS origin then sits. Each pin is the M4 bus
    under its label. M1-M3 are blocked over the whole footprint, M4 and M5
    only where they are drawn.
    """
    (x0, y0), (x1, y1) = top.get_bounding_box()
    plan = macro_plan(circuit)
    buses = bus_positions(macro_periphery(circuit))
    labels = {pin: (24, plan.bank_y + buses[pin]) for pin in circuit.pins}

    def rect(r):
        return (
            f"      RECT {_um(r[0] - x0)} {_um(r[1] - y0)} "
            f"{_um(r[2] - x0)} {_um(r[3] - y0)} ;"
        )

    m4 = _rects(top, "M4")
    pins, pin_shapes = [], set()
    for pin in circuit.pins:
        px, py = labels[pin]
        shapes = [r for r in m4 if r[0] <= px <= r[2] and r[1] <= py <= r[3]]
        if not shapes:
            raise ValueError(f"no M4 under pin label {pin}")
        pin_shapes.update(shapes)
        direction, use = _direction(pin)
        pins += [
            f"  PIN {pin}",
            f"    DIRECTION {direction} ;",
            f"    USE {use} ;",
            "    PORT",
            "      LAYER M4 ;",
            *(rect(r) for r in shapes),
            "    END",
            f"  END {pin}",
        ]
    box = (x0, y0, x1, y1)
    obs = ["  OBS"]
    for layer in COVERED:
        obs += [f"    LAYER {layer} ;", rect(box)]
    for layer in EXACT:
        shapes = [r for r in _rects(top, layer) if r not in pin_shapes]
        if shapes:
            obs += [f"    LAYER {layer} ;", *(rect(r) for r in shapes)]
    obs.append("  END")
    return "\n".join(
        [
            "VERSION 5.8 ;",
            'BUSBITCHARS "[]" ;',
            'DIVIDERCHAR "/" ;',
            "",
            f"MACRO {circuit.name}",
            "  CLASS BLOCK ;",
            f"  FOREIGN {circuit.name} {_um(-x0)} {_um(-y0)} ;",
            "  ORIGIN 0 0 ;",
            f"  SIZE {_um(x1 - x0)} BY {_um(y1 - y0)} ;",
            "  SYMMETRY X Y ;",
            *pins,
            *obs,
            f"END {circuit.name}",
            "",
            "END LIBRARY",
            "",
        ]
    )


def _mode(circuit: Any) -> str:
    if circuit.replica is not None:
        return "self_timed"
    return "clocked" if "CLK" in circuit.pins else "phases"


def macro_verilog(circuit: Any, spec: BromSpec, words: Sequence[int]) -> str:
    """Zero-delay behavioral model of the macro, matching its transistors.

    It follows the switch-level behavior, not just the read protocol: address
    latches open while PRECHN is low, bitlines are dynamic (precharged by
    PRECHN low, discharged while EVAL and PRECHN are both high, otherwise
    held), unused address codes discharge nothing, and Q latches are
    transparent while SAMPLE is high, followed by the rising-edge register
    when the macro has one. A bitline keeper only holds a level the model
    already holds, so it adds nothing here. The clocked variants carry the
    controller's equations; the self-timed one replaces its replica with
    ideal acknowledgments. Under ``SYNTHESIS`` the module is a black box.
    """
    mapping = selector_map(spec, words)
    if circuit.array is None or circuit.array.mapping != mapping:
        raise ValueError("words do not match the circuit's programmed array")
    mode = _mode(circuit)
    ab, bits, ways = spec.address_bits, spec.bits, spec.ways
    cols = spec.columns
    rows = spec.depth // ways

    ports, decls = [], []
    for pin in circuit.pins:
        if pin in ("VDD", "VSS"):
            continue
        match = _BUS.match(pin)
        name = match[1] if match else pin
        if name in ports:
            continue
        ports.append(name)
        width = {"A": ab, "Q": bits}.get(name)
        kind = "output" if name == "Q" else "input"
        decls.append(
            f"  {kind} [{width - 1}:0] {name};" if width else f"  {kind} {name};"
        )

    lines = [
        (
            f"// Behavioral model of {circuit.name}: {spec.depth} x {bits}, "
            f"{ways}-way, {mode.replace('_', '-')} interface. Zero delay; no timing."
        ),
        f"module {circuit.name} ({', '.join(ports)});",
        *decls,
        "`ifndef SYNTHESIS",
        f"  reg [{bits - 1}:0] mem [0:{spec.depth - 1}];",
        "  initial begin",
        *(f"    mem[{i}] = {bits}'h{w:x};" for i, w in enumerate(words)),
        "  end",
    ]
    if mode != "phases":
        lines.append("  wire PRECHN, EVAL, SAMPLE;")
    if mode == "self_timed":
        # The replica's acknowledgments, with its delays taken as zero.
        lines += [
            "  wire PRECHN_FAR = PRECHN;",
            "  wire READ_READY = EVAL;",
            "  wire SAMPLE_CLOSED = ~SAMPLE;",
            "  wire CANDIDATE_BUSY = EVAL;",
        ]
    if mode != "phases":
        lines += [
            "  wire request = RESETN & ~CLK;",
            "  wire sampling = SAMPLE | ~SAMPLE_CLOSED;",
            "  assign EVAL = PRECHN_FAR & (request | sampling);",
            "  assign SAMPLE = request & EVAL & READ_READY;",
            "  assign PRECHN = request | EVAL | sampling | CANDIDATE_BUSY;",
        ]
    lines += [
        f"  reg [{ab - 1}:0] addr;",
        f"  reg [{cols - 1}:0] bl_low;",
        f"  reg [{bits - 1}:0] q;",
        "  integer bit, way;",
        "  // Address latches are transparent during precharge.",
        "  always @(PRECHN or A) if (!PRECHN) addr = A;",
        "  // Dynamic bitlines: precharge, discharge, or hold.",
        "  always @(PRECHN or EVAL or addr)",
        f"    if (!PRECHN) bl_low = {cols}'b0;",
        f"    else if (EVAL && (addr >> {spec.column_bits}) < {rows})",
        f"      for (bit = 0; bit < {bits}; bit = bit + 1)",
        f"        for (way = 0; way < {ways}; way = way + 1)",
        f"          if (mem[(addr >> {spec.column_bits}) * {ways} + way][bit])",
        f"            bl_low[bit * {ways} + way] = 1'b1;",
        f"  reg [{bits - 1}:0] data;",
        "  always @* begin",
        f"    for (bit = 0; bit < {bits}; bit = bit + 1)",
        (
            "      data[bit] = bl_low[bit];"
            if ways == 1
            else f"      data[bit] = bl_low[bit * {ways} + addr[{spec.column_bits - 1}:0]];"
        ),
        "  end",
        "  // Output latches are transparent while SAMPLE is high.",
        "  always @(SAMPLE or data) if (SAMPLE) q = data;",
    ]
    if circuit.output_register is not None:
        # The rising-edge bank: a master latch open while CLK is low, then a
        # slave open while it is high, between the sampling latches and Q.
        lines += [
            f"  reg [{bits - 1}:0] master, registered;",
            "  always @(CLK or q) if (!CLK) master = q;",
            "  always @(CLK or master) if (CLK) registered = master;",
            "  assign Q = registered;",
        ]
    else:
        lines.append("  assign Q = q;")
    lines += [
        "`endif",
        "endmodule",
        "",
    ]
    return "\n".join(lines)
