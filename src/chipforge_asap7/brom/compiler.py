"""Four-word blocked ROM with factored CGEN and explicit read phases.

Encoding and address order follow chipforge_gt2n's BROM compiler: column
selection uses the lowest address bits, followed by the two word-offset
bits, then the block index. The external phase interface permits measured
ASAP7 timing instead of importing GT2N's unqualified replica delays.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

from ..devices.finfet import FinFETSpec
from .circuit import Circuit, SelectorInstance
from .selector import default_selector_plan

BLOCK_SIZE = 4
CANDIDATES = 16


def _validate_keeper(fins, stack):
    if type(fins) is not int or type(stack) is not int:
        raise TypeError("keeper fins and stack must be integers")
    if fins not in (0, 1, 2, 3):
        raise ValueError("keeper_fins must be 0 (disabled), 1, 2 or 3")
    if stack not in (1, 2, 3, 4):
        raise ValueError("keeper_stack must be between 1 and 4")


@dataclass(frozen=True)
class BromSpec:
    depth: int = 16
    bits: int = 8
    ways: int = 1
    fins: int = 2
    vt: str = "rvt"
    keeper_fins: int = 0
    keeper_stack: int = 2

    def __post_init__(self) -> None:
        for field in ("depth", "bits", "ways", "fins", "keeper_fins", "keeper_stack"):
            if type(getattr(self, field)) is not int:
                raise TypeError(f"{field} must be an integer")
        _validate_keeper(self.keeper_fins, self.keeper_stack)
        if self.ways not in (1, 4):
            raise ValueError("ways must be 1 or 4")
        if self.depth < 4 * self.ways or self.depth % (4 * self.ways):
            raise ValueError("depth must be a positive multiple of 4 * ways")
        if self.bits < 1:
            raise ValueError("bits must be positive")
        FinFETSpec(fins=self.fins, vt=self.vt)
        plan = default_selector_plan(self.fins, self.vt)
        if self.blocks * plan.height + 81 > 30_000:
            raise ValueError("array exceeds the single tap row latch-up reach")

    @property
    def blocks(self) -> int:
        return self.depth // (4 * self.ways)

    @property
    def columns(self) -> int:
        return self.bits * self.ways

    @property
    def address_bits(self) -> int:
        return (self.depth - 1).bit_length()

    @property
    def column_bits(self) -> int:
        return self.ways.bit_length() - 1

    @property
    def pins(self) -> tuple[str, ...]:
        return (
            *(f"A<{i}>" for i in range(self.address_bits)),
            "PRECHN",
            "EVAL",
            "SAMPLE",
            *(f"Q<{i}>" for i in range(self.bits)),
            "VDD",
            "VSS",
        )

    def decode(self, address: int) -> tuple[int, int, int]:
        if type(address) is not int or not 0 <= address < self.depth:
            raise ValueError(f"address must be in 0..{self.depth - 1}")
        row, way = divmod(address, self.ways)
        block, offset = divmod(row, 4)
        return block, offset, way


def selector_map(spec: BromSpec, words: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    if len(words) != spec.depth:
        raise ValueError(f"expected {spec.depth} words")
    if any(type(w) is not int or not 0 <= w < (1 << spec.bits) for w in words):
        raise ValueError(f"words must be unsigned {spec.bits}-bit integers")
    return tuple(
        tuple(
            sum(
                ((words[(4 * block + offset) * spec.ways + way] >> bit) & 1) << offset
                for offset in range(4)
            )
            for bit in range(spec.bits)
            for way in range(spec.ways)
        )
        for block in range(spec.blocks)
    )


def _decode(c: Circuit, literals: Sequence[str], prefix: str, count: int) -> None:
    """Shared binary prefix tree; absent addresses have no decoded output."""
    complements = []
    for a in literals:
        an = c.node()
        c.inv(a, an)
        complements.append(an)
    cache: dict[tuple[int, int], str] = {}

    def term(width: int, value: int) -> str:
        if width == 0:
            return "VDD"
        key = width, value
        if key not in cache:
            literal = (
                literals[width - 1]
                if value >> (width - 1) & 1
                else complements[width - 1]
            )
            if width == 1:
                cache[key] = literal
            else:
                cache[key] = c.node()
                c.and2(
                    term(width - 1, value & ((1 << (width - 1)) - 1)),
                    literal,
                    cache[key],
                )
        return cache[key]

    for value in range(count):
        c.buffer(term(len(literals), value), f"{prefix}<{value}>")


def add_g_generator(c: Circuit, a0: str, a1: str) -> None:
    """GT2N's sixteen two-input functions, implemented with ASAP7 devices."""
    n0, n1 = c.node(), c.node()
    c.inv(a0, n0)
    c.inv(a1, n1)
    for k, source in ((0, "VSS"), (3, n1), (5, n0), (10, a0), (12, a1), (15, "VDD")):
        c.buffer(source, f"G<{k}>")
    for k, a, b in ((1, a1, a0), (2, a1, n0), (4, n1, a0)):
        c.nor(a, b, f"G<{k}>")
    for k, a, b in ((7, a1, a0), (11, a1, n0), (13, n1, a0)):
        c.nand(a, b, f"G<{k}>")
    c.and2(a1, a0, "G<8>")
    c.or2(a1, a0, "G<14>")
    c.xor(a1, a0, "G<6>")
    c.inv("G<6>", "G<9>")


def cgen_circuit(fins: int = 2, vt: str = "rvt") -> Circuit:
    pins = (
        "SEL",
        *(f"G<{k}>" for k in range(16)),
        *(f"C<{k}>" for k in range(16)),
        "VDD",
        "VSS",
    )
    c = Circuit(f"brom_cgen_f{fins}_{vt}", pins, fins, vt)
    for k in range(16):
        # SEL is on the ground side of the NAND stack, as in GT2N.
        start = len(c.devices)
        c.and2(f"G<{k}>", "SEL", f"C<{k}>")
        c.group_since(f"candidate_{k}", start)
    return c


def decoder_circuit(spec: BromSpec) -> Circuit:
    pins = (
        *(f"A<{i}>" for i in range(spec.address_bits)),
        "EVAL",
        "PRECHN",
        *(f"G<{k}>" for k in range(16)),
        *(f"SEL<{b}>" for b in range(spec.blocks)),
        *(f"CSEL<{w}>" for w in range(spec.ways)),
        "VDD",
        "VSS",
    )
    c = Circuit(
        f"brom_decode_{spec.depth}_m{spec.ways}_f{spec.fins}_{spec.vt}",
        pins,
        spec.fins,
        spec.vt,
    )
    _add_decoder(c, spec)
    return c


def _add_decoder(c: Circuit, spec: BromSpec, address: str = "A") -> None:
    shift = spec.column_bits
    add_g_generator(c, f"{address}<{shift}>", f"{address}<{shift + 1}>")
    _decode(c, [f"{address}<{i}>" for i in range(shift)], "CSEL", spec.ways)
    _decode(
        c,
        [f"{address}<{i}>" for i in range(shift + 2, spec.address_bits)],
        "BLOCK",
        spec.blocks,
    )
    c.and2("EVAL", "PRECHN", "evaluate")
    for block in range(spec.blocks):
        c.and2(f"BLOCK<{block}>", "evaluate", f"SEL<{block}>")


def add_readout(c, bitline, output, *, keeper_fins=0, keeper_stack=2):
    """Precharge, inverter, and optional weak positive-feedback PMOS path.

    All keeper gates use the inverted bitline. Series devices allow weaker
    drive than the minimum single-fin transistor; every bulk stays at VDD.
    """
    c.fet(True, bitline, "PRECHN", "VDD")
    c.inv(bitline, output)
    source = "VDD"
    for stage in range(keeper_stack if keeper_fins else 0):
        drain = bitline if stage == keeper_stack - 1 else c.node()
        c.fet(True, drain, output, source, fins=keeper_fins)
        source = drain


def output_circuit(
    fins: int = 2, vt: str = "rvt", *, keeper_fins=0, keeper_stack=2
) -> Circuit:
    _validate_keeper(keeper_fins, keeper_stack)
    name = f"brom_output_f{fins}_{vt}"
    if keeper_fins:
        name += f"_k{keeper_fins}s{keeper_stack}"
    c = Circuit(name, ("BL", "PRECHN", "DOUT", "VDD", "VSS"), fins, vt)
    add_readout(c, "BL", "DOUT", keeper_fins=keeper_fins, keeper_stack=keeper_stack)
    return c


def macro_circuit(spec: BromSpec, words: Sequence[int]) -> Circuit:
    mapping = selector_map(spec, words)
    digest = sha256(repr(mapping).encode()).hexdigest()[:12]
    c = Circuit(
        f"brom_compact_{spec.depth}x{spec.bits}_m{spec.ways}_f{spec.fins}_{spec.vt}_{digest}",
        spec.pins,
        spec.fins,
        spec.vt,
    )
    if spec.keeper_fins:
        c.name += f"_k{spec.keeper_fins}s{spec.keeper_stack}"
    address_open = c.node()
    c.inv("PRECHN", address_open)
    for i in range(spec.address_bits):
        start = len(c.devices)
        c.latch(f"A<{i}>", f"ADDR<{i}>", address_open, "PRECHN")
        c.group_since(f"address_{i}", start)
    _add_decoder(c, spec, "ADDR")
    array_devices = set()
    for block, row in enumerate(mapping):
        for k in range(16):
            start = len(c.devices)
            c.and2(f"G<{k}>", f"SEL<{block}>", f"C<{block},{k}>")
            c.group_since(f"cgen_{block}_{k}", start)
        for column, k in enumerate(row):
            c.fet(False, f"BL<{column}>", f"C<{block},{k}>", "VSS")
            array_devices.add(c.devices[-1].name)
            c.fet(False, f"BL<{column}>", "VSS", "VSS")
            array_devices.add(c.devices[-1].name)
    c.array = SelectorInstance(
        default_selector_plan(spec.fins, spec.vt), mapping, frozenset(array_devices)
    )
    for column in range(spec.columns):
        start = len(c.devices)
        add_readout(
            c,
            f"BL<{column}>",
            f"DOUT<{column}>",
            keeper_fins=spec.keeper_fins,
            keeper_stack=spec.keeper_stack,
        )
        c.group_since(f"readout_{column}", start)
    sampleb = c.node()
    c.inv("SAMPLE", sampleb)
    for bit in range(spec.bits):
        if spec.ways == 1:
            data = f"DOUT<{bit}>"
        else:
            terms = []
            for way in range(spec.ways):
                term = c.node()
                c.and2(f"DOUT<{bit * spec.ways + way}>", f"CSEL<{way}>", term)
                terms.append(term)
            data = terms[0]
            for term in terms[1:]:
                out = c.node()
                c.or2(data, term, out)
                data = out
        start = len(c.devices)
        c.latch(data, f"Q<{bit}>", "SAMPLE", sampleb)
        c.group_since(f"output_latch_{bit}", start)
    return c
