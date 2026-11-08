"""The NOR ROM macro's transistors, and its CDL netlist.

`build_rom` places abutting cells -- array tiles, decoder slots, lane rows,
latch and controller rows -- whose diffusion and gate poly merge across their
edges, so no block is a subcircuit of its own: the netlist is flat.  Each
`Transistor` is one physical device (one finger of a gate), sized by fins,
on the macro's own net names.

Two views come from the one list:

* the CDL (`rom_macro_cdl`): one BSIM-CMG instance per transistor, sized by
  ``nfin``, as the B-ROM writes its own and as released ASAP7 CDL is.  Its
  pins are the macro's interface (`rom_macro_ports`);
* the unit-fin LVS reference (`chipforge_asap7.verification.rom_macro_lvs`):
  one MOS per FIN x GATE channel.

They differ only in the uncontacted diffusion a series stack's inner node,
a driver's floating edge finger or the array's last ISO gate ends in.  The
CDL makes it one net, as a schematic does; the unit-fin view splits it per
fin, as the extractor sees it without ``merge_fin_diffusion``.  Such nets are
the transistor's `split` set.

Nets the macro labels keep their names (``WL<w>``, ``BLL<b>``, ``PA<i>``,
``A<i>``, ``Q<b>``, ``CLK``, ...).  Each block's internal nets are prefixed
so that two blocks' local names cannot meet.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..layout.grid import FIN_WIDTH, GATE_WIDTH
from .inverter import InverterSpec
from .nand import NandSpec
from .rom_bitcell import ROM_TILE_WORDS, RomArraySpec
from .rom_decoder import RomDecoderSpec

__all__ = [
    "Transistor",
    "rom_macro_cdl",
    "rom_macro_devices",
    "rom_macro_halves",
    "rom_macro_pins",
    "rom_macro_ports",
]


@dataclass(frozen=True)
class Transistor:
    """One physical transistor: a gate finger over `fins` fins."""

    name: str
    drain: str
    gate: str
    source: str
    bulk: str
    model: str
    length: int
    fins: int
    #: Terminal nets that are uncontacted diffusion: one net per fin to an
    #: extractor that does not merge fin diffusion.
    split: frozenset[str] = field(default=frozenset())

    def cdl(self) -> str:
        """The CDL instance line, sized by ``nfin``."""
        return (
            f"{self.name} {self.drain} {self.gate} {self.source} {self.bulk} "
            f"{self.model} nfin={self.fins} l={self.length}n nf=1 m=1"
        )

    def unit_fins(self) -> list[str]:
        """One unit MOS line per fin, the `split` nets suffixed by fin."""

        def net(name: str, fin: int) -> str:
            return f"{name}_f{fin}" if name in self.split else name

        return [
            f"{self.name}_f{fin} {net(self.drain, fin)} {self.gate} "
            f"{net(self.source, fin)} {self.bulk} {self.model} "
            f"L={self.length}n W={FIN_WIDTH}n"
            for fin in range(self.fins)
        ]


# ── Blocks ────────────────────────────────────────────────────────────────────
def _array(
    spec: RomArraySpec,
    sites: set[tuple[int, int]],
    bitline: str,
    first_word: int,
    tag: str,
) -> list[Transistor]:
    """One half's bit transistors and ISO transistors.

    A word's drain is on its bitline when that bit is programmed, else a
    floating net of its own (its M1 stub contacts every fin).  Each tile's
    ISO joins its word 1 drain to the next tile's word 0 drain; the last one
    ends in uncontacted diffusion.
    """
    bitcell = spec.bitcell
    model = f"nmos_{bitcell.vt}"

    def drain(word: int, bit: int) -> str:
        if (word, bit) in sites:
            return f"{bitline}<{bit}>"
        return f"{tag}F_W{word}_B{bit}"

    out = []
    for bit in range(spec.bits):
        for word in range(spec.words):
            out.append(
                Transistor(
                    f"M{tag}W{word}_B{bit}",
                    drain(word, bit),
                    f"WL<{first_word + word}>",
                    "VSS",
                    "VSS",
                    model,
                    GATE_WIDTH,
                    bitcell.fins,
                )
            )
        for tile in range(spec.tiles):
            left = drain(ROM_TILE_WORDS * tile + 1, bit)
            right_word = ROM_TILE_WORDS * tile + 2
            end = right_word >= spec.words
            right = f"{tag}F_END_B{bit}" if end else drain(right_word, bit)
            out.append(
                Transistor(
                    f"M{tag}ISO_T{tile}_B{bit}",
                    left,
                    "VSS",
                    right,
                    "VSS",
                    model,
                    GATE_WIDTH,
                    bitcell.fins,
                    frozenset({right}) if end else frozenset(),
                )
            )
    return out


def _rows(
    devices: Iterable[tuple[str, str, str, str, str, int]],
    bands: dict,
    prefix: str,
    rename: dict[str, str],
    local: set[str],
    series: set[str],
) -> list[Transistor]:
    """A row cell's ``(name, drain, gate, source, flavor, fins)`` on macro nets.

    `rename` maps the cell's ports to macro nets; nets in `local` are the
    cell's own and take `prefix`; nets in `series` are uncontacted.
    """

    def net(name: str) -> str:
        if name in rename:
            return rename[name]
        if name in local or name in series:
            return f"{prefix}{name}"
        return name

    split = frozenset(net(n) for n in series)
    out = []
    for device, drain, gate, source, flavor, fins in devices:
        band = bands[flavor].spec
        bulk = "VSS" if flavor == "n" else "VDD"
        out.append(
            Transistor(
                f"M{prefix}{device}",
                net(drain),
                net(gate),
                net(source),
                bulk,
                band.model,
                band.gate_length,
                fins,
                split & {net(drain), net(source)},
            )
        )
    return out


def _nand(spec: NandSpec, a: str, b: str, y: str, prefix: str) -> list[Transistor]:
    """A NAND2 of `spec.fingers` stacks: each an n series pair whose inner
    node is uncontacted, and two parallel p pull-ups."""
    n, p = spec.n_band.spec, spec.p_band.spec
    out = []
    for stack in range(spec.fingers):
        node = f"{prefix}N{stack}"
        series = frozenset({node})
        out += [
            Transistor(f"M{prefix}NA{stack}", y, a, node, "VSS", n.model,
                       n.gate_length, n.fins, series),
            Transistor(f"M{prefix}NB{stack}", node, b, "VSS", "VSS", n.model,
                       n.gate_length, n.fins, series),
            Transistor(f"M{prefix}PA{stack}", y, a, "VDD", "VDD", p.model,
                       p.gate_length, p.fins),
            Transistor(f"M{prefix}PB{stack}", y, b, "VDD", "VDD", p.model,
                       p.gate_length, p.fins),
        ]  # fmt: skip
    return out


def _driver(spec: InverterSpec, a: str, y: str, prefix: str) -> list[Transistor]:
    """A slot's driver, one finger per transistor, and its edge finger.

    The driver is one unmirrored inverter: its extra edge gate has no
    neighbour to claim it, so it is one more finger per band, on floating
    poly, against uncontacted diffusion (as
    `chipforge_asap7.verification.lvs.render_inverter_row_lvs_schematic`
    models an odd row).
    """
    out = []
    for index, band in enumerate(spec.bands):
        rail, model = band.rail_net, band.spec
        for finger in range(spec.fingers):
            out.append(
                Transistor(
                    f"M{prefix}D{index}_{finger}",
                    y,
                    a,
                    rail,
                    rail,
                    model.model,
                    model.gate_length,
                    band.fins,
                )
            )
        stub = f"{prefix}EDGE_D{index}"
        out.append(
            Transistor(
                f"M{prefix}EDGE{index}",
                stub,
                f"{prefix}EDGE_G",
                rail,
                rail,
                model.model,
                model.gate_length,
                band.fins,
                frozenset({stub}),
            )
        )
    return out


def _decoder(spec: RomDecoderSpec) -> list[Transistor]:
    """Each word's NAND and driver, then each placed predecoder gate's."""
    out: list[Transistor] = []
    slot = spec.word_slot
    for w in range(spec.array.words):
        a, b = spec.lines(w)
        out += _nand(slot.nand, a, b, f"DW{w}_N", f"DW{w}_")
        out += _driver(slot.inverter, f"DW{w}_N", f"WL<{w}>", f"DW{w}_")
    pre = spec.pre_slot
    for p in spec.predecoder:
        tag = f"DP{p.pair}_{p.row}_{p.column}_"
        gate = p.gate
        if gate.kind == "and":
            a, b = gate.ins
            out += _nand(pre.nand, a, b, f"{tag}N", tag)
            driven = f"{tag}N"
        else:
            driven = gate.ins[0]
        out += _driver(pre.inverter, driven, gate.out, tag)
    return out


def _lane(spec: RomDecoderSpec) -> list[Transistor]:
    lane = spec.sense_lane
    devices = lane.devices()
    nets = {n for d in devices for n in d[1:4]}
    series = {n for n in nets if re.fullmatch(r"X\d+", n)}
    local = {n for n in nets if re.fullmatch(r"(LX|LY|LQ|RX|RY)\d+|SAMPLEB_R\d+", n)}
    return _rows(devices, lane.row_spec(1).bands, "S_", {}, local, series)


def _latches(spec: RomDecoderSpec) -> list[Transistor]:
    rs = spec.latch_row
    rename = {}
    for k in range(rs.latches):
        rename[f"D<{k}>"] = f"A<{k}>"
        rename[f"Q<{k}>"] = f"AL<{k}>"
    nets = {n for d in rs.devices for n in d[1:4]}
    local = {n for n in nets if n == "CK" or re.fullmatch(r"[XY]\d+", n)}
    return _rows(rs.devices, rs.bands, "LA_", rename, local, set())


def _controller(spec: RomDecoderSpec) -> list[Transistor]:
    cs = spec.control_row
    ports = {"CLK", "RESETN", "READY", "PRECHN", "EVAL", "SAMPLE", "VDD", "VSS"}
    nets = {n for d in cs.devices for n in d[1:4]}
    return _rows(cs.devices, cs.bands, "C_", {}, nets - ports, set(cs.series_nodes))


# ── The macro ─────────────────────────────────────────────────────────────────
def rom_macro_halves(spec: RomDecoderSpec, programmed: Iterable[tuple[int, int]]):
    """``(array, sites, bitline, first_word, tag, x)`` for each array cell
    `build_rom` places, its sites in that cell's own word numbering; with
    ``control`` the replica bit is programmed in every word."""
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


def rom_macro_devices(
    spec: RomDecoderSpec, programmed: Iterable[tuple[int, int]] = ()
) -> tuple[Transistor, ...]:
    """Every transistor of ``build_rom(spec, programmed)``."""
    out: list[Transistor] = []
    for array, sites, bitline, first_word, tag, _ in rom_macro_halves(spec, programmed):
        out += _array(array, sites, bitline, first_word, tag)
    out += _decoder(spec)
    if spec.sense:
        out += _lane(spec)
    if spec.latch:
        out += _latches(spec)
    if spec.control:
        out += _controller(spec)
    return tuple(out)


def rom_macro_pins(spec: RomDecoderSpec) -> tuple[str, ...]:
    """Every net `build_rom` labels, in a fixed order: the interface and the
    internal nets it names for LVS."""
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


def rom_macro_ports(spec: RomDecoderSpec) -> tuple[str, ...]:
    """The macro's interface: what a design that places it connects.

    The address, then the timing the macro does not make itself -- ``CLK``
    and ``RESETN`` with ``control``, otherwise ``PRECHN``, ``EVAL`` and
    ``SAMPLE`` as far as it has them -- then the data: ``Q<b>`` with
    ``sample``, the leaves' ``D<b>`` with only ``sense``, else the bitlines.
    The replica bit is not data.
    """
    ports = [f"A<{i}>" for i in range(spec.address_bits)]
    if spec.control:
        ports += ["CLK", "RESETN"]
    else:
        ports += ["PRECHN"] if spec.sense else []
        ports += ["EVAL"] if spec.enable else []
        ports += ["SAMPLE"] if spec.sample else []
    bits = range(spec.array.bits)
    if spec.sample:
        ports += [f"Q<{b}>" for b in bits]
    elif spec.sense:
        ports += [f"D<{b}>" for b in bits]
    elif spec.lane:
        ports += [f"BLL<{b}>" for b in bits] + [f"BLR<{b}>" for b in bits]
    else:
        ports += [f"BL<{b}>" for b in bits]
    return (*ports, "VDD", "VSS")


def rom_macro_cdl(
    spec: RomDecoderSpec,
    programmed: Iterable[tuple[int, int]] = (),
    *,
    name: str,
) -> str:
    """The macro's flat CDL: one instance per transistor, sized by ``nfin``.

    `name` is the subcircuit's, normally the `build_rom` cell's.
    """
    devices = rom_macro_devices(spec, programmed)
    fins = sum(d.fins for d in devices)
    return "\n".join(
        [
            f"* ASAP7 NOR ROM macro {name}: {len(devices)} transistors, {fins} fins",
            f".subckt {name} {' '.join(rom_macro_ports(spec))}",
            *(d.cdl() for d in devices),
            f".ends {name}",
            "",
        ]
    )


def _natural(name: str) -> tuple:
    head, _, tail = name.partition("<")
    return head, int(tail.rstrip(">") or 0)
