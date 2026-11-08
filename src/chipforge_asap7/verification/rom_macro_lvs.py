"""A whole NOR ROM macro as one flat LVS reference.

`build_rom` places abutting cells -- array tiles, decoder slots, lane rows,
latch and controller rows -- whose diffusion and gate poly merge across their
edges, so the extractor cannot keep any of them apart as subcircuits.  The
reference is therefore flat: every block's devices, one MOS per FIN x GATE
channel, on the macro's own net names.

Nets the macro labels keep their names (``WL<w>``, ``BLL<b>``, ``PA<i>``,
``A<i>``, ``Q<b>``, ``CLK``, ...).  Each block's internal nets are prefixed
so two blocks' local names cannot meet.  An uncontacted series node is one
net per fin, as the extractor sees it without ``merge_fin_diffusion``.

Run it with ``run_lvs(..., flat=True)``: hierarchical extraction gains
nothing against a flat reference and was seen to stall on a 64x32 macro.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from chipforge_asap7.devices.inverter import InverterSpec
from chipforge_asap7.devices.rom_bitcell import ROM_TILE_WORDS, RomArraySpec
from chipforge_asap7.devices.rom_decoder import RomDecoderSpec
from chipforge_asap7.layout.grid import FIN_WIDTH, GATE_WIDTH
from chipforge_asap7.layout.layers import PIN_LAYERS, require_gdspy
from chipforge_asap7.verification.lvs import _inverter_units, _nand_units


@dataclass(frozen=True)
class MacroReference:
    """The reference's device lines and the nets the macro labels."""

    lines: tuple[str, ...]
    pins: tuple[str, ...]

    def render(self, name: str) -> str:
        return "\n".join(
            [
                "* ASAP7 NOR ROM macro LVS reference: one MOS per FIN x GATE channel",
                f".SUBCKT {name} {' '.join(self.pins)}",
                *self.lines,
                f".ENDS {name}",
                ".END",
                "",
            ]
        )

    @property
    def fins(self) -> int:
        return len(self.lines)


def _array(
    spec: RomArraySpec,
    sites: set[tuple[int, int]],
    bitline: str,
    first_word: int,
    tag: str,
) -> list[str]:
    """One half's bit transistors and ISO transistors, unit fin.

    A word's drain is on its bitline when that bit is programmed, else a
    floating net of its own.  Each tile's ISO joins its word 1 drain to the
    next tile's word 0 drain; the last one ends in uncontacted diffusion, one
    net per fin.
    """
    bitcell = spec.bitcell
    model = f"nmos_{bitcell.vt}"
    size = f"L={GATE_WIDTH}n W={FIN_WIDTH}n"

    def drain(word: int, bit: int) -> str:
        if (word, bit) in sites:
            return f"{bitline}<{bit}>"
        return f"{tag}F_W{word}_B{bit}"

    lines = []
    for bit in range(spec.bits):
        for word in range(spec.words):
            for fin in range(bitcell.fins):
                lines.append(
                    f"M{tag}W{word}_B{bit}_f{fin} {drain(word, bit)} "
                    f"WL<{first_word + word}> VSS VSS {model} {size}"
                )
        for tile in range(spec.tiles):
            left = drain(ROM_TILE_WORDS * tile + 1, bit)
            right_word = ROM_TILE_WORDS * tile + 2
            for fin in range(bitcell.fins):
                right = (
                    drain(right_word, bit)
                    if right_word < spec.words
                    else f"{tag}F_END_B{bit}_f{fin}"
                )
                lines.append(
                    f"M{tag}ISO_T{tile}_B{bit}_f{fin} {left} VSS {right} VSS "
                    f"{model} {size}"
                )
    return lines


def _rows(
    devices: Iterable[tuple[str, str, str, str, str, int]],
    bands: dict,
    prefix: str,
    rename: dict[str, str],
    local: set[str],
    series: set[str],
) -> list[str]:
    """A row cell's ``(name, drain, gate, source, flavor, fins)`` as unit fins.

    `rename` maps the cell's ports to macro nets; nets in `local` are the
    cell's own and take `prefix`; nets in `series` are uncontacted and split
    per fin.
    """

    def net(name: str, fin: int) -> str:
        if name in rename:
            return rename[name]
        if name in series:
            return f"{prefix}{name}_f{fin}"
        if name in local:
            return f"{prefix}{name}"
        return name

    lines = []
    for device, drain, gate, source, flavor, fins in devices:
        band = bands[flavor].spec
        bulk = "VSS" if flavor == "n" else "VDD"
        for fin in range(fins):
            lines.append(
                f"M{prefix}{device}_f{fin} {net(drain, fin)} {net(gate, fin)} "
                f"{net(source, fin)} {bulk} {band.model} "
                f"L={band.gate_length}n W={FIN_WIDTH}n"
            )
    return lines


def _edge(inverter: InverterSpec, tag: str) -> list[str]:
    """A slot's driver is one unmirrored inverter: its extra edge gate has no
    neighbour to claim it, so it is one finger on floating poly against an
    uncontacted diffusion stub per fin (as `render_inverter_row_lvs_schematic`
    models an odd row)."""
    return [
        f"M{tag}_EDGE_b{index}_f{fin} {tag}_EDGE_D{index}_f{fin} {tag}_EDGE_G "
        f"{band.rail_net} {band.rail_net} {band.spec.model} "
        f"L={band.spec.gate_length}n W={FIN_WIDTH}n"
        for index, band in enumerate(inverter.bands)
        for fin in range(band.fins)
    ]


def _decoder(spec: RomDecoderSpec) -> list[str]:
    """Each word's NAND and driver, then each placed predecoder gate's."""
    lines: list[str] = []
    slot = spec.word_slot
    for w in range(spec.array.words):
        a, b = spec.lines(w)
        lines += _nand_units(
            slot.nand, a=a, b=b, y=f"DW{w}_N", tag=f"_W{w}", prefix=f"MDW{w}_"
        )
        lines += _inverter_units(
            slot.inverter, gate=f"DW{w}_N", drain=f"WL<{w}>", prefix=f"MDW{w}_D"
        )
        lines += _edge(slot.inverter, f"DW{w}")
    pre = spec.pre_slot
    for p in spec.predecoder:
        tag = f"P{p.pair}_{p.row}_{p.column}"
        gate = p.gate
        if gate.kind == "and":
            a, b = gate.ins
            lines += _nand_units(
                pre.nand, a=a, b=b, y=f"D{tag}_N", tag=f"_{tag}", prefix=f"MD{tag}_"
            )
            driven = f"D{tag}_N"
        else:
            driven = gate.ins[0]
        lines += _inverter_units(
            pre.inverter, gate=driven, drain=gate.out, prefix=f"MD{tag}_D"
        )
        lines += _edge(pre.inverter, f"D{tag}")
    return lines


def _lane(spec: RomDecoderSpec) -> list[str]:
    lane = spec.sense_lane
    rs = lane.row_spec(1)
    devices = lane.devices()
    nets = {n for d in devices for n in d[1:4]}
    series = {n for n in nets if re.fullmatch(r"X\d+", n)}
    local = {n for n in nets if re.fullmatch(r"(LX|LY|LQ|RX|RY)\d+|SAMPLEB_R\d+", n)}
    return _rows(devices, rs.bands, "S_", {}, local, series)


def _latches(spec: RomDecoderSpec) -> list[str]:
    rs = spec.latch_row
    rename = {}
    for k in range(rs.latches):
        rename[f"D<{k}>"] = f"A<{k}>"
        rename[f"Q<{k}>"] = f"AL<{k}>"
    nets = {n for d in rs.devices for n in d[1:4]}
    local = {n for n in nets if n == "CK" or re.fullmatch(r"[XY]\d+", n)}
    return _rows(rs.devices, rs.bands, "LA_", rename, local, set())


def _controller(spec: RomDecoderSpec) -> list[str]:
    cs = spec.control_row
    ports = {"CLK", "RESETN", "READY", "PRECHN", "EVAL", "SAMPLE", "VDD", "VSS"}
    nets = {n for d in cs.devices for n in d[1:4]}
    return _rows(cs.devices, cs.bands, "C_", {}, nets - ports, set(cs.series_nodes))


def _halves(spec: RomDecoderSpec, programmed: Iterable[tuple[int, int]]):
    """``(array, sites, bitline, first_word, tag, x)`` for each array cell
    `build_rom` places, its sites in that cell's own word numbering."""
    sites = set(programmed)
    if spec.control:
        sites |= {(w, spec.replica_bit) for w in range(spec.array.words)}
    if not spec.lane:
        return [(spec.cells, sites, "BL", 0, "", 0)]
    n = spec.half.words
    return [
        (
            spec.half,
            {(w - side * n, b) for w, b in sites if (w >= n) == side},
            label,
            side * n,
            f"A{label[2]}_",
            (0, spec.right_x)[side],
        )
        for side, label in enumerate(("BLL", "BLR"))
    ]


def floating_drains(
    spec: RomDecoderSpec, programmed: Iterable[tuple[int, int]] = ()
) -> list[tuple[str, tuple[float, float]]]:
    """Each unprogrammed drain's reference net and its M1 stub's centre.

    The site a programming V1 would take is on that stub, so an M1 label
    there names the drain.
    """
    out = []
    for array, sites, _, _, tag, x in _halves(spec, programmed):
        for bit in range(array.bits):
            for word in range(array.words):
                if (word, bit) not in sites:
                    sx, sy = array.site(word, bit)
                    out.append((f"{tag}F_W{word}_B{bit}", (x + sx, sy)))
    return out


def label_floating_drains(
    spec: RomDecoderSpec, programmed: Iterable[tuple[int, int]], cell: Any
) -> int:
    """Label every unprogrammed drain of ``build_rom``'s `cell`, for LVS only.

    An unprogrammed drain is a floating net, and a run of them along one bit
    row looks the same as the run in any other: the comparer has nothing but
    a search to tell them apart, and at 128x33 that search is most of the
    compare.  Net names settle such ties, and only ties: a label on the wrong
    drain, or on a drain a V1 joined to its bitline, still fails.  Pair it
    with ``rom_macro_reference(..., name_floating=True)``.  Returns the number
    of labels added.
    """
    gdspy = require_gdspy()
    layer, texttype = PIN_LAYERS["M1"]
    drains = floating_drains(spec, programmed)
    for name, origin in drains:
        cell.add(gdspy.Label(name, origin, layer=layer, texttype=texttype))
    return len(drains)


def rom_macro_reference(
    spec: RomDecoderSpec,
    programmed: Iterable[tuple[int, int]] = (),
    *,
    name_floating: bool = False,
) -> MacroReference:
    """The flat unit-fin reference of ``build_rom(spec, programmed)``.

    With `name_floating` the unprogrammed drains are pins too, to match a
    layout `label_floating_drains` has labelled.
    """
    halves = _halves(spec, programmed)
    lines: list[str] = []
    for array, sites, bitline, first_word, tag, _ in halves:
        lines += _array(array, sites, bitline, first_word, tag)
    lines += _decoder(spec)
    if spec.sense:
        lines += _lane(spec)
    if spec.latch:
        lines += _latches(spec)
    if spec.control:
        lines += _controller(spec)
    pins = rom_macro_pins(spec)
    if name_floating:
        floating = tuple(name for name, _ in floating_drains(spec, programmed))
        pins = (*pins[:-2], *floating, *pins[-2:])
    return MacroReference(tuple(lines), pins)


def rom_macro_pins(spec: RomDecoderSpec) -> tuple[str, ...]:
    """The nets `build_rom` labels, in a fixed order."""
    words, bits = range(spec.array.words), range(spec.cells.bits)
    pins = [f"WL<{w}>" for w in words]
    if spec.lane:
        pins += [f"BLL<{b}>" for b in bits] + [f"BLR<{b}>" for b in bits]
    else:
        pins += [f"BL<{b}>" for b in bits]
    lines = sorted(
        {line for w in words for line in spec.lines(w)}
        | {p.gate.out for p in spec.predecoder},
        key=_natural,
    )
    pins += [ln for ln in lines if ln[:2] in ("PA", "PB")]
    if not spec.latch:
        pins += [f"A<{i}>" for i in range(spec.address_bits)]
    if spec.enable:
        pins.append("EVAL")
    if spec.sense:
        lane = spec.sense_lane
        pins += [p for p in lane.pins if p not in pins and p not in ("VDD", "VSS")]
    if spec.latch:
        pins += [f"A<{i}>" for i in range(spec.address_bits)]
    if spec.control:
        pins += ["CLK", "RESETN"]
    return (*dict.fromkeys(pins), "VDD", "VSS")


def _natural(name: str) -> tuple:
    head, _, tail = name.partition("<")
    return head, int(tail.rstrip(">") or 0)
