"""LEF, Verilog and Liberty views of a placeable NOR ROM macro.

The macro is ``build_rom(spec, ...)`` with ``spec.edge_pins``: its interface is
``A[i]``, ``CLK``, ``RESETN``, ``Q[b]``, ``VDD`` and ``VSS``, on M5 columns
reaching the bottom edge, so GDS labels, CDL, LEF, Verilog and Liberty all name
the same pins, the way the B-ROM's do.

Its read takes one cycle, in one of two ways.  With ``register`` it is
two-phase: ``CLK`` high precharges, ``CLK`` falling closes the address latches
and starts the read, and ``Q`` takes the word as ``CLK`` rises.  A read has to
finish while ``CLK`` is low; if it does not, ``SAMPLE`` never rises and ``Q``
keeps the old word.  With ``flow_through``, ``CLK`` rising closes the address
latches and starts a read that ends itself, and ``Q`` takes the word before
the next rise, holding it until the read after.  Its limit is the period, not
either phase.

**Without a timing report, the Liberty is an estimate.**  `estimate_timing` walks
the macro's actual transistors with the logical-effort numbers
`chipforge_asap7.devices.sizing.LogicalEffortModel` holds, a rough wire
capacitance, no wire resistance and a flat margin.  It tracks the macro's size
but no simulation has checked it; the arcs exist so that a design can be placed
and constrained until the read is simulated.  A corner measured by
`chipforge_asap7.devices.rom_timing` takes its delays and constraints from the
measurement instead, plus `MEASURED_MARGIN`; pin capacitances stay estimates.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from ..layout.layers import LAYERS, PIN_LAYERS
from .rom_decoder import RomDecoderSpec
from .rom_netlist import rom_macro_devices, rom_macro_ports
from .sizing import LogicalEffortModel

if TYPE_CHECKING:
    from .rom_timing import CornerTiming

__all__ = [
    "CORNERS",
    "RomTimingEstimate",
    "estimate_timing",
    "rom_macro_lef",
    "rom_macro_liberty",
    "rom_macro_verilog",
    "rom_macro_views",
]

#: Layers a router must treat as fully blocked over the macro.
COVERED = ("M1", "M2", "M3")
#: Layers whose drawn shapes are emitted exactly, so the M5 pins stay reachable.
EXACT = ("M4", "M5", "M6")

#: Liberty corner -> (file suffix, delay scale), the ratios the NOR ROM_FULL
#: libraries use.
CORNERS = {"typical": ("tt", 1.0), "fast": ("ff", 0.75), "slow": ("ss", 1.44)}
#: Multiplies every estimated delay: nothing here has been simulated.
MARGIN = 1.5
#: The loads a measured ``Q`` table is indexed by, in fF, as
#: `chipforge_asap7.devices.rom_timing` measures them.
LIBERTY_LOADS_FF = (0.0, 1.0, 4.0)
#: Multiplies a measured delay or constraint: its wires are estimates, it has no
#: local wire capacitance and it is one bit of one programming.
MEASURED_MARGIN = 1.25
#: A pin's load when nothing else is known: as the B-ROM's estimate assumes.
OUTPUT_LOAD_FF = 1.0
#: Diffusion capacitance per fin, relative to a fin's gate capacitance.
DIFFUSION_RATIO = 0.59
#: No constraint below this: a process-corner safety floor, in ps.
MIN_CONSTRAINT_PS = 10.0


def _need(spec: RomDecoderSpec) -> None:
    if not spec.edge_pins:
        raise ValueError("the views describe a placeable macro: they need edge_pins")


# ── LEF ───────────────────────────────────────────────────────────────────────
def _um(nm: float) -> str:
    return f"{nm / 1000:.3f}"


def _rects(top: Any, layer: str) -> list[tuple[float, float, float, float]]:
    spec = (LAYERS[layer]["layer"], LAYERS[layer]["datatype"])
    rects = set()
    for points in top.get_polygons(by_spec=spec):
        if len(points) != 4:
            raise ValueError(f"non-rectangular {layer} shape in the ROM macro")
        lo, hi = points.min(axis=0), points.max(axis=0)
        rects.add((float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])))
    return sorted(rects)


def _direction(pin: str) -> tuple[str, str]:
    if pin == "VDD":
        return "INOUT", "POWER"
    if pin == "VSS":
        return "INOUT", "GROUND"
    return ("OUTPUT" if pin.startswith("Q[") else "INPUT"), "SIGNAL"


def _pin_columns(top: Any) -> dict[str, list[tuple[float, float, float, float]]]:
    """Each pin's M5 columns: the shape under each of its labels."""
    m5 = _rects(top, "M5")
    layer, texttype = PIN_LAYERS["M5"]
    columns: dict[str, list] = {}
    for label in top.get_labels(depth=0):
        if (label.layer, label.texttype) != (layer, texttype):
            continue
        px, py = label.position
        under = [r for r in m5 if r[0] <= px <= r[2] and r[1] <= py <= r[3]]
        if len(under) != 1:
            raise ValueError(f"{len(under)} M5 shapes under pin label {label.text}")
        columns.setdefault(label.text, []).append(under[0])
    return columns


def rom_macro_lef(spec: RomDecoderSpec, top: Any, name: str | None = None) -> str:
    """LEF 5.8 abstract of `top`, the cell ``build_rom(spec, ...)`` returned.

    Coordinates move so the bounding box's lower-left corner is the LEF origin,
    and ``FOREIGN``'s point is, per the LEF spec, that origin in the GDS cell's
    coordinates.  Each pin is the M5 column under its label, one ``PORT`` per
    column.  M1-M3 are blocked over the whole footprint, M4-M6 only where they
    are drawn.  `name` overrides the macro's name.
    """
    _need(spec)
    name = name or top.name
    (x0, y0), (x1, y1) = top.get_bounding_box()

    def rect(r: tuple[float, float, float, float]) -> str:
        return (
            f"      RECT {_um(r[0] - x0)} {_um(r[1] - y0)} "
            f"{_um(r[2] - x0)} {_um(r[3] - y0)} ;"
        )

    drawn = {layer: _rects(top, layer) for layer in EXACT}
    columns = _pin_columns(top)
    pins, taken = [], set()
    for pin in rom_macro_ports(spec):
        if pin not in columns:
            raise ValueError(f"the macro has no M5 pin {pin}")
        direction, use = _direction(pin)
        pins += [f"  PIN {pin}", f"    DIRECTION {direction} ;", f"    USE {use} ;"]
        for column in sorted(columns[pin]):
            taken.add(column)
            pins += ["    PORT", "      LAYER M5 ;", rect(column), "    END"]
        pins.append(f"  END {pin}")

    box = (x0, y0, x1, y1)
    obs = ["  OBS"]
    for metal in COVERED:
        obs += [f"    LAYER {metal} ;", rect(box)]
    for metal in EXACT:
        shapes = [r for r in drawn[metal] if metal != "M5" or r not in taken]
        if shapes:
            obs += [f"    LAYER {metal} ;", *(rect(r) for r in shapes)]
    obs.append("  END")
    return "\n".join(
        [
            "VERSION 5.8 ;",
            'BUSBITCHARS "[]" ;',
            'DIVIDERCHAR "/" ;',
            "",
            f"MACRO {name}",
            "  CLASS BLOCK ;",
            f"  FOREIGN {name} {_um(x0)} {_um(y0)} ;",
            "  ORIGIN 0 0 ;",
            f"  SIZE {_um(x1 - x0)} BY {_um(y1 - y0)} ;",
            "  SYMMETRY X Y ;",
            *pins,
            *obs,
            f"END {name}",
            "",
            "END LIBRARY",
            "",
        ]
    )


# ── Verilog ───────────────────────────────────────────────────────────────────
def rom_macro_verilog(
    spec: RomDecoderSpec, programmed: Iterable[tuple[int, int]], name: str
) -> str:
    """Zero-delay behavioral model of the macro, matching its transistors.

    It follows the switch-level behavior: the controller's equations
    (`chipforge_asap7.devices.rom_control`), address latches open while
    ``PRECHN`` is low, read leaves that fall back to 0 in precharge and rise
    once ``EVAL`` selects a programmed bit, the replica as such a leaf with a
    bit in every word, sample latches open while ``SAMPLE`` is high and, with
    ``register``, register latches open while it is low.  Under ``SYNTHESIS``
    the module is a black box.
    """
    _need(spec)
    words, bits, ab = spec.array.words, spec.array.bits, spec.address_bits
    memory = [0] * words
    for word, bit in programmed:
        memory[word] |= 1 << bit
    if spec.flow_through:
        return _flow_through_verilog(spec, memory, name)
    lines = [
        (
            f"// Behavioral model of {name}: {words} x {bits} NOR ROM, self-timed, "
            f"{'registered' if spec.register else 'latched'} output. "
            "Zero delay; no timing."
        ),
        f"module {name} (A, CLK, RESETN, Q);",
        f"  input [{ab - 1}:0] A;",
        "  input CLK;",
        "  input RESETN;",
        f"  output [{bits - 1}:0] Q;",
        "`ifndef SYNTHESIS",
        f"  reg [{bits - 1}:0] mem [0:{words - 1}];",
        "  initial begin",
        *(f"    mem[{i}] = {bits}'h{w:x};" for i, w in enumerate(memory)),
        "  end",
        "  wire PRECHN, EVAL, SAMPLE;",
        "  reg READY;",
        "  wire REQ = RESETN & ~CLK;",
        "  assign PRECHN = REQ | EVAL;",
        "  assign EVAL = (REQ | SAMPLE) & PRECHN;",
        "  assign SAMPLE = REQ & EVAL & READY;",
        f"  reg [{ab - 1}:0] addr;",
        f"  reg [{bits - 1}:0] data, sampled;",
        "  // The address latches are transparent during precharge.",
        "  always @* if (!PRECHN) addr = A;",
        "  // The read leaves: precharge, a selected bit, or hold.",
        "  always @* begin",
        "    if (!PRECHN) begin",
        f"      data = {bits}'b0;",
        "      READY = 1'b0;",
        "    end else if (EVAL) begin",
        "      data = data | mem[addr];",
        "      READY = 1'b1;",
        "    end",
        "  end",
        "  // The sample latches are transparent while SAMPLE is high.",
        "  always @* if (SAMPLE) sampled = data;",
    ]
    if spec.register:
        lines += [
            "  // The register latches take the word as SAMPLE falls.",
            f"  reg [{bits - 1}:0] registered;",
            "  always @* if (!SAMPLE) registered = sampled;",
            "  assign Q = registered;",
        ]
    else:
        lines.append("  assign Q = sampled;")
    lines += ["`endif", "endmodule", ""]
    return "\n".join(lines)


def _flow_through_verilog(spec: RomDecoderSpec, memory: list[int], name: str) -> str:
    """A flow-through macro's model, at the clock edge rather than the switch
    level: the controller's read ends itself, which a zero-delay model of its
    gates would turn into a race.  ``CLK`` rising with ``RESETN`` high latches
    ``A``, and ``Q`` holds that word until the next read's."""
    words, bits, ab = spec.array.words, spec.array.bits, spec.address_bits
    lines = [
        (
            f"// Behavioral model of {name}: {words} x {bits} NOR ROM, self-timed, "
            "flow-through output. Zero delay; no timing."
        ),
        f"module {name} (A, CLK, RESETN, Q);",
        f"  input [{ab - 1}:0] A;",
        "  input CLK;",
        "  input RESETN;",
        f"  output [{bits - 1}:0] Q;",
        "`ifndef SYNTHESIS",
        f"  reg [{bits - 1}:0] mem [0:{words - 1}];",
        "  initial begin",
        *(f"    mem[{i}] = {bits}'h{w:x};" for i, w in enumerate(memory)),
        "  end",
        "  // CLK rising starts a read of A; the sample latches hold its word",
        "  // until the next read's arrives. Q is unknown until the first read.",
        f"  reg [{bits - 1}:0] sampled;",
        "  always @(posedge CLK) if (RESETN) sampled <= mem[A];",
        "  assign Q = sampled;",
        "`endif",
        "endmodule",
        "",
    ]
    return "\n".join(lines)


# ── Liberty ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RomTimingEstimate:
    """Typical-corner numbers in ps, margin included, and input pins' fF."""

    clk_q: float
    slew: float
    setup: float
    hold: float
    min_low: float
    min_high: float
    min_period: float
    clk_cap: float
    resetn_cap: float
    address_cap: float

    def at(self, corner: str) -> RomTimingEstimate:
        """The delays at `corner`; a capacitance doesn't scale."""
        _, scale = CORNERS[corner]
        return RomTimingEstimate(
            **{
                f.name: getattr(self, f.name)
                * (1 if f.name.endswith("_cap") else scale)
                for f in fields(self)
            }
        )


def estimate_timing(
    spec: RomDecoderSpec,
    column_um: dict[str, float] | None = None,
    model: LogicalEffortModel | None = None,
) -> RomTimingEstimate:
    """Walk the read's paths through the macro's own transistors.

    Each stage is ``tau * (load / drive + p_inv)``: `load` is the gate and
    diffusion fins on the net and a guess at its wire, `drive` the input
    capacitance of an inverter with the net's largest driver.  A pin's M5
    column length comes from `column_um`, in um, when known.  The bitline is
    every word of a half programmed, discharged by one selected bit.

    A flow-through read starts on ``CLK`` rising through the edge latch
    (``CLKB``, ``SET``, ``REQB``, ``REQ``), and ``Q`` follows the sample latch.
    Its ``min_period`` is the read, the latch's open time and the precharge;
    its pulses need only arm and set the latch.
    """
    _need(spec)
    model = model or LogicalEffortModel()
    columns = column_um or {}
    c_fin, tau = model.c_fin_fF, model.tau_ps
    gates, diffusion = Counter(), Counter()
    n_side, p_side = Counter(), Counter()
    for d in rom_macro_devices(spec, ()):
        gates[d.gate] += d.fins
        for net in (d.drain, d.source):
            diffusion[net] += d.fins
            (n_side if d.model.startswith("n") else p_side)[net] += d.fins

    lane = spec.sense_lane
    across_um = (lane.height + spec.right_x) / 1000
    row_um = spec.right_x / 1000

    def load(net: str, wire_um: float = 0.0, extra_fF: float = 0.0) -> float:
        cap = gates[net] * c_fin + DIFFUSION_RATIO * diffusion[net] * c_fin
        return cap + wire_um * model.wire_cap_fF_per_um + extra_fF

    def stage(net: str, wire_um: float = 0.0, extra_fF: float = 0.0) -> float:
        drive = max(n_side[net], p_side[net], 1)
        effort = load(net, wire_um, extra_fF) / (2 * drive * c_fin)
        return tau * (effort + model.p_inv)

    q_um = max((um for pin, um in columns.items() if pin.startswith("Q[")), default=0)
    request = stage("C_REQ")
    # CLK rises: REQ falls, SAMPLE falls, each row's SAMPLEB rises and its copy
    # of SAMPLE falls, the register latch opens, and its two inverters drive Q.
    clk_q = (
        request
        + stage("C_SN")
        + stage("SAMPLE", across_um)
        + stage("S_SAMPLEB_R0", row_um)
        + stage("S_SAMPLEL_R0", row_um)
        + stage("S_RX0")
        + stage("S_RY0")
        + stage("Q[0]", q_um, OUTPUT_LOAD_FF)
    )
    slew = 2 * stage("Q[0]", q_um, OUTPUT_LOAD_FF)
    # CLK falls: REQ rises, PRECHN rises, the latches' clock closes.
    closing = (
        request + stage("C_EN") + stage("PRECHN", across_um) + stage("LA_CK", row_um)
    )
    hold = closing
    # The address has to get through its latch and the decoder's lines, which
    # EVAL starts on after the latch closes.
    line = stage("AL<0>", across_um) + stage("LA_Y0") + stage("LA_X0")
    evaluate = request + stage("C_EV0") + stage("C_EVB") + stage("EVAL", across_um)
    setup = line - evaluate

    # The read: EVAL through the decoder to a wordline, a bit discharges the
    # bitline, READY rises and SAMPLE opens.
    half = spec.half.words
    fins = spec.array.bitcell.fins
    bitline_fF = (
        half * fins * DIFFUSION_RATIO * c_fin + row_um * model.wire_cap_fF_per_um
    )
    discharge = tau * bitline_fF / (2 * fins * c_fin)
    decode = 2 * stage("DW0_N") + stage("WL<0>", spec.cells.height / 1000)
    capture = stage("READY", across_um) + stage("C_SN") + stage("SAMPLE", across_um)
    read = request + evaluate + decode + discharge + capture
    # CLK rises: SAMPLE falls, EVAL falls, PRECHN falls and the bitlines recover
    # through both precharge devices' 4 fins.
    recover = tau * bitline_fF / (2 * 4 * c_fin)
    precharge = (
        request + stage("C_SN") + stage("SAMPLE", across_um) + evaluate + recover
    )

    def pin_cap(net: str, port: str) -> float:
        return load(net, columns.get(port, 0.0))

    min_low, min_high, min_period = read, precharge, read + precharge
    if spec.flow_through:
        # CLK rises: CLKB falls, SET rises, REQB falls and REQ rises; then the
        # read as above.  The sample latch opens with SAMPLE and its two
        # inverters drive Q.
        edge = stage("C_CLKB") + stage("C_SET") + stage("C_REQB") + request
        through = stage("C_EV0") + stage("C_EVB") + stage("EVAL", across_um)
        ft_read = edge + through + decode + discharge + capture
        clk_q = (
            ft_read
            + stage("S_SAMPLEL_R0", row_um)
            + stage("S_LX0")
            + stage("S_LY0")
            + stage("Q[0]", q_um, OUTPUT_LOAD_FF)
        )
        hold = (
            edge + stage("C_EN") + stage("PRECHN", across_um) + stage("LA_CK", row_um)
        )
        setup = line - (edge + through)
        done = spec.done_delay * stage("C_DL0") + request
        min_period = ft_read + done + precharge
        # CLK low arms the latch (CLKB, ARMB, ARMED); CLK high sets it.
        min_low = stage("C_CLKB") + stage("C_ARMB") + stage("C_ARMED")
        min_high = edge

    return RomTimingEstimate(
        clk_q=MARGIN * clk_q,
        slew=MARGIN * slew,
        setup=max(MARGIN * setup, MIN_CONSTRAINT_PS),
        hold=max(MARGIN * hold, MIN_CONSTRAINT_PS),
        min_low=MARGIN * min_low,
        min_high=MARGIN * min_high,
        min_period=MARGIN * min_period,
        clk_cap=pin_cap("CLK", "CLK"),
        resetn_cap=pin_cap("RESETN", "RESETN"),
        address_cap=max(
            pin_cap(f"A[{i}]", f"A[{i}]") for i in range(spec.address_bits)
        ),
    )


def _bus_type(name: str, width: int) -> str:
    return (
        f"  type ({name}) {{\n    base_type : array;\n    data_type : bit;\n"
        f"    bit_width : {width};\n    bit_from : {width - 1};\n"
        "    bit_to : 0;\n    downto : true;\n  }\n"
    )


def _scalar(kind: str, value: float) -> str:
    return f'          {kind} (scalar) {{ values ("{value:.2f}"); }}\n'


def _table(kind: str, values: Iterable[float]) -> str:
    row = ", ".join(f"{value:.2f}" for value in values)
    return f'          {kind} (rom_q_load) {{ values ("{row}"); }}\n'


def _guarded(value: float) -> float:
    """A measured constraint with margin, however early it is, and its floor."""
    return max(value + abs(value) * (MEASURED_MARGIN - 1), MIN_CONSTRAINT_PS)


def _constraint(kind: str, value: float) -> str:
    return (
        '        timing() {\n          related_pin : "CLK";\n'
        f"          timing_type : {kind};\n"
        + _scalar("rise_constraint", value)
        + _scalar("fall_constraint", value)
        + "        }\n"
    )


def rom_macro_liberty(
    spec: RomDecoderSpec,
    name: str,
    corner: str,
    area_um2: float,
    estimate: RomTimingEstimate | None = None,
    measured: CornerTiming | None = None,
) -> str:
    """Liberty of one corner, from `measured` if given and otherwise estimated.
    ``Q`` launches at ``CLK`` rising, so this needs ``register`` or
    ``flow_through``.

    With ``register``, ``A`` latches at ``CLK`` falling and ``CLK``'s minimum
    pulse widths carry the read itself: low for a read to finish, high for the
    bitlines to recover.  With ``flow_through``, ``A`` latches at ``CLK`` rising,
    ``Q``'s arc is the whole read, and ``min_period`` carries the read and the
    precharge after it.  ``RESETN`` carries no arc: it is released once, after
    reset, while the clock runs.
    """
    _need(spec)
    if not (spec.register or spec.flow_through):
        raise ValueError(
            "the Liberty's Q arc launches at CLK rising: needs register or flow_through"
        )
    suffix, _ = CORNERS[corner]
    t = (estimate or estimate_timing(spec)).at(corner)
    ab, bits = spec.address_bits, spec.array.bits
    if measured is None:
        setup, hold = t.setup, t.hold
        min_low, min_high, min_period = t.min_low, t.min_high, t.min_period
        q_arcs = (
            _scalar("cell_rise", t.clk_q)
            + _scalar("cell_fall", t.clk_q)
            + _scalar("rise_transition", t.slew)
            + _scalar("fall_transition", t.slew)
        )
        templates = ""
        status = "ESTIMATED, NOT CHARACTERIZED\n * (chipforge_asap7.devices.rom_views)."
    else:
        setup, hold = _guarded(measured.setup), _guarded(measured.hold)
        min_low = MEASURED_MARGIN * measured.min_low
        min_high = MEASURED_MARGIN * measured.min_high
        min_period = MEASURED_MARGIN * (
            measured.min_period
            if measured.min_period is not None
            else measured.min_low + measured.min_high
        )
        scaled = lambda values: [MEASURED_MARGIN * value for value in values]
        q_arcs = (
            _table("cell_rise", scaled(measured.clk_q_rise))
            + _table("cell_fall", scaled(measured.clk_q_fall))
            + _table("rise_transition", scaled(measured.q_rise_slew))
            + _table("fall_transition", scaled(measured.q_fall_slew))
        )
        templates = (
            "  lu_table_template (rom_q_load) {\n"
            "    variable_1 : total_output_net_capacitance;\n"
            f'    index_1 ("{", ".join(f"{load:g}" for load in LIBERTY_LOADS_FF)}");\n'
            "  }\n"
        )
        limit = (
            "no limit found"
            if measured.max_low is None
            else f"about {measured.max_low:g} ps"
        )
        status = (
            f"SIMULATED, NOT SIGN-OFF: ngspice on a reduced netlist with\n"
            f" * estimated wire RC, margin {MEASURED_MARGIN:g}, nothing extracted\n"
            " * (chipforge_asap7.devices.rom_timing).  Pin capacitances are estimates.\n"
            f" * CLK may stay low for {limit} before a bitline droops too far."
        )
    edge = "rising" if spec.flow_through else "falling"
    a_pins = "".join(
        f"      pin(A[{i}]) {{\n        direction : input;\n"
        f"        capacitance : {t.address_cap:.3f};\n"
        + _constraint(f"setup_{edge}", setup)
        + _constraint(f"hold_{edge}", hold)
        + "      }\n"
        for i in range(ab)
    )
    q_pins = "".join(
        f"      pin(Q[{b}]) {{\n        direction : output;\n"
        '        timing() {\n          related_pin : "CLK";\n'
        "          timing_type : rising_edge;\n" + q_arcs + "        }\n      }\n"
        for b in range(bits)
    )
    return (
        f"/* {name}: ASAP7 NOR ROM, {corner} corner. {status} */\n"
        f"library({name}_{suffix}) {{\n"
        "  technology (cmos);\n  delay_model : table_lookup;\n"
        '  time_unit : "1ps";\n  voltage_unit : "1V";\n  current_unit : "1mA";\n'
        '  pulling_resistance_unit : "1kohm";\n  capacitive_load_unit (1, ff);\n'
        "  slew_upper_threshold_pct_fall : 80;\n"
        "  slew_lower_threshold_pct_fall : 20;\n"
        "  slew_upper_threshold_pct_rise : 80;\n"
        "  slew_lower_threshold_pct_rise : 20;\n"
        "  input_threshold_pct_fall : 50;\n  input_threshold_pct_rise : 50;\n"
        "  output_threshold_pct_fall : 50;\n  output_threshold_pct_rise : 50;\n"
        + templates
        + _bus_type(f"ROM_A_{ab}", ab)
        + _bus_type(f"ROM_Q_{bits}", bits)
        + f"  cell({name}) {{\n    area : {area_um2:.4f};\n    dont_touch : true;\n"
        "    dont_use : true;\n    interface_timing : true;\n"
        "    pin(CLK) {\n      direction : input;\n      clock : true;\n"
        f"      capacitance : {t.clk_cap:.3f};\n"
        f"      min_pulse_width_low : {min_low:.2f};\n"
        f"      min_pulse_width_high : {min_high:.2f};\n"
        f"      min_period : {min_period:.2f};\n    }}\n"
        "    pin(RESETN) {\n      direction : input;\n"
        f"      capacitance : {t.resetn_cap:.3f};\n    }}\n"
        f"    bus(A) {{\n      bus_type : ROM_A_{ab};\n"
        f"      direction : input;\n{a_pins}    }}\n"
        f"    bus(Q) {{\n      bus_type : ROM_Q_{bits};\n"
        f"      direction : output;\n{q_pins}    }}\n"
        "  }\n}\n"
    )


# ── All of them ───────────────────────────────────────────────────────────────
def rom_macro_views(
    spec: RomDecoderSpec,
    programmed: Iterable[tuple[int, int]],
    top: Any,
    timing: dict[str, CornerTiming] | None = None,
) -> dict[str, str]:
    """The macro's views by file name: ``macro.lef``, ``macro.v`` and, with
    ``register`` or ``flow_through``, ``macro_{tt,ff,ss}.lib``, from `timing`
    for a corner it has."""
    views = {
        "macro.lef": rom_macro_lef(spec, top),
        "macro.v": rom_macro_verilog(spec, programmed, top.name),
    }
    if spec.register or spec.flow_through:
        columns = {
            pin: max(r[3] - r[1] for r in rects) / 1000
            for pin, rects in _pin_columns(top).items()
        }
        estimate = estimate_timing(spec, columns)
        (x0, y0), (x1, y1) = top.get_bounding_box()
        area = (x1 - x0) * (y1 - y0) / 1e6
        for corner, (suffix, _) in CORNERS.items():
            views[f"macro_{suffix}.lib"] = rom_macro_liberty(
                spec, top.name, corner, area, estimate, (timing or {}).get(corner)
            )
    return views
