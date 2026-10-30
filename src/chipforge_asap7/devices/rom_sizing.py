"""Load-based sizing for the transposed NOR-ROM's actual PA/PB network.

Four cell roles are independently sized: the wordline inverter, its NAND,
the predecoder NANDs and the predecoder output/address-complement inverters.
Each predecoder role uses its worst load, so cells remain interchangeable in
the existing placement. Parallel placement copies count as extra input loads;
we conservatively give each copy enough drive for the entire output load.

Without supplied parasitics, capacitance is an estimate: bitcell gate fins,
poly length, and routed M3/M4 centreline lengths. Metal capacitance comes from
LogicalEffortModel; the poly coefficient is an explicit assumption. Coupling,
vias and local interconnect require extraction. Supplied wordline load is the
TOTAL (wire plus devices), while supplied line loads are WIRE ONLY. Neither
capacitance sizing nor the logical-effort delay is a transient simulation.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace

from .finfet import MAX_FINS
from .inverter import InverterSpec
from .nand import NandSpec
from .rom_decoder import RomDecoderSpec, _stub_span
from .rom_predecoder import network
from .sizing import LogicalEffortModel, Stage

__all__ = ["RomDecoderSizing", "size_rom_decoder"]


def _number(name: str, value: float, *, positive: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )


def _cell_size(drive: float, inputs: int) -> tuple[tuple[int, int], int]:
    """Round UP to legal fins/fingers; NAND n:p is 2:1, inverter 1:1.

    Fins grow first, then fold into more even fingers. The minima preserve
    the NAND contact space and the driver's M2 access in the ROM slots.
    """
    minimum = 2 if inputs == 2 else 3
    _number("required drive", drive, positive=True)
    fingers = 2 * max(1, math.ceil(drive / (2 * (MAX_FINS // inputs))))
    p = max(minimum, math.ceil(drive / fingers))
    return (inputs * p, p), fingers


def _input_cap(cell: NandSpec | InverterSpec, model: LogicalEffortModel) -> float:
    return sum(cell.rows[0]) * cell.fingers * model.c_fin_fF


def _wire_lengths(spec: RomDecoderSpec) -> dict[str, float]:
    """Routed M3/M4 length per predecode/address net, in micrometres.

    Union co-linear stubs so the two postdecode B pins' shared stub is
    counted once. M4 segment endpoints already include their end-caps.
    """
    lengths: dict[str, float] = defaultdict(float)
    vertical: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for (_, line), (_, x0, x1) in spec.segments.items():
        lengths[line] += x1 - x0
    for line, x in spec.joins.items():
        ys = [spec.rail_y(pair, line) for pair in spec.presence[line]]
        vertical[line, x].append((min(ys), max(ys)))
    for word in range(spec.array.words):
        row = spec.row_of(word)
        for pin, line in zip(("A", "B"), spec.lines(word)):
            x = spec.slot_x(word) + spec.stubs(word % 2)[pin]
            vertical[line, x].append(
                _stub_span(spec.pin_y(row, pin), spec.rail_y(row // 2, line))
            )
    for placed in spec.predecoder:
        row = 2 * placed.pair + placed.row
        for pin, line in spec.pre_pins(placed.gate):
            x = spec.pre_x(placed.column) + spec.pre_tracks[placed.row][pin]
            vertical[line, x].append(
                _stub_span(
                    spec.pin_y(row, pin, predecode=True), spec.rail_y(placed.pair, line)
                )
            )
    for (line, _), spans in vertical.items():
        end = -math.inf
        for lo, hi in sorted(spans):
            lengths[line] += max(0, hi - max(lo, end))
            end = max(end, hi)
    return {line: length / 1000 for line, length in lengths.items()}


def _loads(
    spec: RomDecoderSpec, model: LogicalEffortModel, overrides: Mapping[str, float]
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    wire = {
        line: length * model.wire_cap_fF_per_um
        for line, length in _wire_lengths(spec).items()
    }
    # The private NAND->inverter M2 link is absent from the named rails.
    slots = {"post_nand": spec.word_slot}
    if spec.predecode:
        slots["pre_nand"] = spec.pre_slot
    for name, slot in slots.items():
        length = abs(slot.inverter_x + slot.inverter.input_gate_x - slot.nand_y_x)
        wire[name] = length / 1000 * model.wire_cap_fF_per_um
    unknown = overrides.keys() - wire.keys()
    if unknown:
        raise ValueError(f"unknown ROM wire-load nets: {sorted(unknown)}")
    wire.update(overrides)
    load = dict(wire)
    fanout: Counter[str] = Counter()
    for word in range(spec.array.words):
        for line in spec.lines(word):
            load[line] += _input_cap(spec.nand, model)
            fanout[line] += 1
    for placed in spec.predecoder:
        gate = placed.gate
        cell = spec.pre_slot.nand if gate.kind == "and" else spec.pre_slot.inverter
        # Keep repeated inputs: AND(x, x) presents TWO NAND input loads.
        for line in gate.ins:
            load[line] += _input_cap(cell, model)
            fanout[line] += 1
    load["post_nand"] += _input_cap(spec.inverter, model)
    if spec.predecode:
        load["pre_nand"] += _input_cap(spec.pre_slot.inverter, model)
    return load, wire, dict(fanout)


def _stage(
    name: str,
    cell: NandSpec | InverterSpec,
    inputs: int,
    load: float,
    model: LogicalEffortModel,
) -> Stage:
    return Stage(name, inputs, cell.rows[0][1] * cell.fingers, load, model)


@dataclass(frozen=True)
class RomDecoderSizing:
    """A buildable spec, its actual quantized sizes, and auditable loads.

    ``stages`` reports worst loads for the four roles. ``line_load_fF`` and
    ``fanout`` include actual placement copies and tied inputs. Predecoder
    cells share a size per role, not with the wordline cells. ``gate_delay_ps``
    excludes distributed wire resistance. ``wl_wire_delay_ps`` may carry a
    separately extracted/modelled contribution; it is never invented here.
    """

    spec: RomDecoderSpec
    stages: dict[str, Stage]
    line_load_fF: dict[str, float]
    wire_load_fF: dict[str, float]
    fanout: dict[str, int]
    wl_gate_load_fF: float
    wl_load_fF: float
    supplied_wire_nets: tuple[str, ...]
    supplied_wl_load: bool
    wl_wire_delay_ps: float | None
    gate_delay_ps: float
    iterations: int

    @property
    def address_input_cap_fF(self) -> dict[str, float]:
        return {
            line: load
            for line, load in self.line_load_fF.items()
            if line.startswith("A<")
        }

    @property
    def delay_ps(self) -> float:
        return self.gate_delay_ps + (self.wl_wire_delay_ps or 0.0)

    def summary(self) -> str:
        source = "supplied total" if self.supplied_wl_load else "estimated gates + wire"
        lines = [
            f"ROM sizing: WL {self.wl_load_fF:.3f} fF ({source}); {self.iterations} iterations"
        ]
        for name, stage in self.stages.items():
            lines.append(
                f"  {name:12s} n={stage.n_fins:3d} p={stage.p_fins:3d} total fins/input; "
                f"load {stage.load_fF:.3f} fF, effort {stage.effort:.2f}"
            )
        lines.append(
            f"  gate-delay estimate {self.gate_delay_ps:.2f} ps; "
            + (
                "distributed wire delay not included"
                if self.wl_wire_delay_ps is None
                else f"supplied WL wire delay {self.wl_wire_delay_ps:.2f} ps"
            )
        )
        return "\n".join(lines)


def size_rom_decoder(
    spec: RomDecoderSpec,
    *,
    wl_load_fF: float | None = None,
    wire_load_fF: Mapping[str, float] | None = None,
    wl_wire_delay_ps: float | None = None,
    stage_effort: float = 4.0,
    model: LogicalEffortModel | None = None,
    poly_cap_fF_per_um: float = 0.2,
) -> RomDecoderSizing:
    """Size this ROM's four roles, preserving its address split and interleave.

    Args:
        spec: array and placement to size. Explicit role sizes are replaced;
            ``rows`` remains the minimum physical row stack.
        wl_load_fF: supplied worst wordline TOTAL capacitance, including
            transistor gates, e.g. from extracted parasitics plus device caps.
        wire_load_fF: supplied WIRE-only capacitance by named predecode/address
            net. Partial overrides are allowed; missing nets are estimated.
            ``post_nand``/``pre_nand`` override the private NAND-output wires.
        wl_wire_delay_ps: optional separately measured/modelled WL wire delay.
        stage_effort: maximum target logical effort; quantization rounds up.
        model: technology calibration, default ASAP7 TT/RVT at 0.7 V. Supply
            a calibrated model explicitly for another VT or corner.
        poly_cap_fF_per_um: estimated poly wire capacitance, excluding gates;
            an assumption, not an extracted process parameter.

    Iterate sizing and routed wire estimates until sizes stop growing. A
    role's strongest load sets every instance of that role. If fingers no
    longer fit the caller's interleave, construction fails rather than
    silently changing address mapping or truncating drive strength.
    """
    if model is None and spec.array.bitcell.vt != "rvt":
        raise ValueError("non-RVT ROM sizing needs an explicit calibrated model")
    model = model or LogicalEffortModel()
    _number("stage_effort", stage_effort, positive=True)
    if stage_effort <= 1:
        raise ValueError("stage_effort must exceed 1")
    for name in ("tau_ps", "gate_cap_aF_per_fin"):
        _number(name, getattr(model, name), positive=True)
    for name in ("p_inv", "wire_cap_fF_per_um"):
        _number(name, getattr(model, name))
    _number("poly_cap_fF_per_um", poly_cap_fF_per_um)
    if wl_load_fF is not None:
        _number("wl_load_fF", wl_load_fF, positive=True)
    if wl_wire_delay_ps is not None:
        _number("wl_wire_delay_ps", wl_wire_delay_ps)
    overrides = dict(wire_load_fF or {})
    for name, value in overrides.items():
        _number(f"wire_load_fF[{name!r}]", value)

    current = replace(
        spec,
        driver_fins=(3, 3),
        nand_fins=(4, 2),
        predecoder_driver_fins=(3, 3),
        predecoder_nand_fins=(4, 2),
        driver_fingers=2,
        nand_fingers=2,
        predecoder_driver_fingers=2,
        predecoder_nand_fingers=2,
    )
    gate_load = spec.array.bits * spec.array.bitcell.fins * model.c_fin_fF
    roles = (
        ("driver", 1),
        ("nand", 2),
        ("predecoder_driver", 1),
        ("predecoder_nand", 2),
    )
    for iteration in range(1, 65):
        load, wire, fanout = _loads(current, model, overrides)
        longest = max(current.pin_y(row, "Y") for row in range(current.interleave))
        wl_total = (
            wl_load_fF
            if wl_load_fF is not None
            else (
                gate_load
                + current.array.height / 1000 * poly_cap_fF_per_um
                + longest / 1000 * model.wire_cap_fF_per_um
            )
        )
        outputs = {p.gate.out for p in current.predecoder}
        loads = (
            wl_total,
            load["post_nand"],
            max((load[line] for line in outputs), default=0),
            load.get("pre_nand", 0),
        )
        changed = {}
        for (role, inputs), out_load in zip(roles, loads):
            if role.startswith("predecoder") and not current.predecode:
                continue
            old_fins = getattr(current, f"{role}_fins")
            old_fingers = getattr(current, f"{role}_fingers")
            required = (
                model.g(inputs)
                * out_load
                / (stage_effort * (inputs + 1) * model.c_fin_fF)
            )
            fins, fingers = _cell_size(max(required, old_fins[1] * old_fingers), inputs)
            changed[f"{role}_fins"] = fins
            changed[f"{role}_fingers"] = fingers
        if all(getattr(current, name) == value for name, value in changed.items()):
            break
        current = replace(current, **changed)
    else:
        raise ValueError(
            "ROM sizing did not converge in 64 iterations; reduce fanout or bank the ROM"
        )

    stages = {
        "driver": _stage("driver", current.inverter, 1, wl_total, model),
        "post_nand": _stage("post_nand", current.nand, 2, load["post_nand"], model),
    }
    arrival: dict[str, float] = {}
    if current.predecode:
        stages["pre_nand"] = _stage(
            "pre_nand", current.pre_slot.nand, 2, load["pre_nand"], model
        )
        stages["pre_driver"] = _stage(
            "pre_driver", current.pre_slot.inverter, 1, loads[2], model
        )
        gates = network(current.address_bits, current.pa_bits, current.row_bits)

        def time(line: str) -> float:
            if line not in arrival:
                gate = gates.get(line)
                if gate is None:
                    arrival[line] = 0.0
                else:
                    delay = _stage(
                        line, current.pre_slot.inverter, 1, load[line], model
                    ).delay_ps
                    if gate.kind == "and":
                        delay += stages["pre_nand"].delay_ps
                    arrival[line] = max(time(i) for i in gate.ins) + delay
            return arrival[line]

        for word in range(current.array.words):
            for line in current.lines(word):
                time(line)
    delay = (
        max(arrival.values(), default=0)
        + stages["post_nand"].delay_ps
        + stages["driver"].delay_ps
    )
    return RomDecoderSizing(
        current,
        stages,
        load,
        wire,
        fanout,
        gate_load,
        wl_total,
        tuple(sorted(overrides)),
        wl_load_fF is not None,
        wl_wire_delay_ps,
        delay,
        iteration,
    )
