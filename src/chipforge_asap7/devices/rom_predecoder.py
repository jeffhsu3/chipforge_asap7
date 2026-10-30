"""The ROM decoder's predecoders: address bits to the ``PA`` and ``PB`` lines.

The wordline gate is an AND2, ``WL<w> = PA<w mod 2^a> . PB<w >> a>``, so the
predecoders turn the address into those two one-hot groups.  This module is
the logic and its placement; `chipforge_asap7.devices.rom_decoder` draws it,
in the same rows as the wordline gates and from the same NAND + driver slot.

Every gate is an AND2 slot, or an inverter slot for an address bit's
complement ``AN<i>``.  A group of bits is decoded as a tree: its low half and
its high half, each decoded the same way, meet in one AND2 per line.  ``PA``
splits at the row bits instead::

    PA<r + k*m> = AND(R<r>, M<m>)     r: the low log2(k) bits, k = interleave

``R<r>`` names a decoder row and is only ever needed in that row's pair, so
each pair builds the ``PA`` lines it uses from its own ``R`` lines and the
shared ``M`` lines.  A one-bit group, which has no tree, is a buffer: an AND2
with both inputs on the one literal.

`place` puts each gate in a pair of rows.  A ``PA`` gate goes to the pair
whose rails carry its line; any other gate goes where most of the gates it
drives are, or else to the pair with the fewest gates.  Within a pair the
gates fill columns of two, lower row first, ANDs before inverters, so that the
two rows of a column are mostly the same kind of slot.  A pair with an odd
number of gates gets a second copy of one, so that no gate stands across from
a filler.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

__all__ = ["Gate", "Placement", "network", "place", "validate_placement"]


@dataclass(frozen=True)
class Gate:
    """One predecoder gate: ``out = AND(*ins)``, or ``NOT(ins[0])``."""

    out: str
    ins: tuple[str, ...]

    @property
    def kind(self) -> Literal["and", "inv"]:
        return "inv" if len(self.ins) == 1 else "and"


@dataclass(frozen=True)
class Placement:
    """Where a gate stands: its pair, its row in the pair (0 lower), its column."""

    gate: Gate
    pair: int
    row: int
    column: int


def validate_placement(
    gates: dict[str, Gate], interleave: int, placements: tuple[Placement, ...]
) -> None:
    """Check a complete placement, including parallel copies and PA locality."""
    occupied: dict[int, set[tuple[int, int]]] = {}
    seen = set()
    for p in placements:
        if any(
            isinstance(v, bool) or not isinstance(v, int)
            for v in (p.pair, p.row, p.column)
        ):
            raise ValueError("predecoder placement coordinates must be integers")
        if not 0 <= p.pair < interleave // 2 or p.row not in (0, 1) or p.column < 0:
            raise ValueError("predecoder placement is outside its row pairs")
        if gates.get(p.gate.out) != p.gate:
            raise ValueError(f"predecoder placement has an unknown gate: {p.gate.out}")
        if (
            p.gate.out.startswith("PA<")
            and p.pair != int(p.gate.out[3:-1]) % interleave // 2
        ):
            raise ValueError(f"{p.gate.out} must stay with its wordline rails")
        sites = occupied.setdefault(p.pair, set())
        if (p.row, p.column) in sites:
            raise ValueError("predecoder placement has overlapping cells")
        sites.add((p.row, p.column))
        seen.add(p.gate.out)
    if seen != gates.keys():
        raise ValueError("predecoder placement must contain every logical gate")
    for sites in occupied.values():
        columns = max(column for _, column in sites) + 1
        if sites != {(row, column) for row in (0, 1) for column in range(columns)}:
            raise ValueError("predecoder columns must be contiguous and fill both rows")


def network(address_bits: int, pa_bits: int, row_bits: int) -> dict[str, Gate]:
    """Every predecoder gate by the line it drives, inputs ``A<i>`` first.

    `pa_bits` low bits make ``PA``, the rest ``PB``; neither may be empty.
    """
    if not 0 < pa_bits < address_bits:
        raise ValueError(
            f"PA needs 1..{address_bits - 1} of the {address_bits} address bits; "
            f"got {pa_bits}"
        )
    gates: dict[str, Gate] = {}

    def literal(bit: int, value: int) -> str:
        if value:
            return f"A<{bit}>"
        name = f"AN<{bit}>"
        gates.setdefault(name, Gate(name, (f"A<{bit}>",)))
        return name

    def decode(lo: int, hi: int, value: int, name: str = "", split: int = 0) -> str:
        """The line for bits ``lo..hi-1`` equal to `value`."""
        if hi - lo == 1:
            if not name:
                return literal(lo, value)
            ins: tuple[str, ...] = (literal(lo, value),) * 2
        else:
            mid = split or lo + (hi - lo) // 2
            low = decode(lo, mid, value & ((1 << (mid - lo)) - 1))
            high = decode(mid, hi, value >> (mid - lo))
            ins = (low, high)
        name = name or f"D{lo}_{hi - 1}<{value}>"
        gates.setdefault(name, Gate(name, ins))
        return name

    split = row_bits if 0 < row_bits < pa_bits else 0
    for value in range(2**pa_bits):
        decode(0, pa_bits, value, f"PA<{value}>", split)
    for value in range(2 ** (address_bits - pa_bits)):
        decode(pa_bits, address_bits, value, f"PB<{value}>")
    return gates


def place(gates: dict[str, Gate], interleave: int) -> list[Placement]:
    """Pairs, rows and columns for `gates`, see the module docstring."""
    pairs = interleave // 2
    consumers: dict[str, list[str]] = {name: [] for name in gates}
    for gate in gates.values():
        for line in set(gate.ins):
            if line in consumers:
                consumers[line].append(gate.out)

    pair_of: dict[str, int] = {}
    load = [0] * pairs

    def put(name: str, pair: int) -> None:
        pair_of[name] = pair
        load[pair] += 1

    # Consumers before producers: a gate's pair depends on its consumers'.
    for name in gates:
        if name.startswith("PA<"):
            put(name, (int(name[3:-1]) % interleave) // 2)
    pending = [name for name in gates if name not in pair_of]
    while pending:
        ready = [n for n in pending if all(c in pair_of for c in consumers[n])]
        for name in ready:
            votes = Counter(pair_of[c] for c in consumers[name]).most_common()
            best = [p for p, v in votes if v == votes[0][1]] if votes else range(pairs)
            put(name, min(best, key=lambda p: (load[p], p)))
        pending = [n for n in pending if n not in pair_of]

    placements = []
    for pair in range(pairs):
        mine = [g for g in gates.values() if pair_of[g.out] == pair]
        mine.sort(key=lambda g: g.kind != "and")
        if len(mine) % 2:
            # A filler across from a gate would land its rail contacts on the
            # gate's source stubs (V0.M1.AUX.3): fill the spot with a second
            # copy, in parallel, of the gate with the most loads.
            mine.append(max(mine, key=lambda g: len(consumers[g.out])))
        for i, gate in enumerate(mine):
            placements.append(Placement(gate, pair, i % 2, i // 2))
    return placements
