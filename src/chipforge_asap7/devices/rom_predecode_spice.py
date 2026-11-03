"""The NOR-ROM decoder's predecode path to one word, as a SPICE netlist.

`predecode_network` turns a `RomDecoderSpec` into the transistors and wiring
between the address pins and one word's postdecode NAND:

* the predecoder gates whose outputs that word's ``PA`` and ``PB`` lines
  depend on (:func:`~chipforge_asap7.devices.rom_predecoder.network`): every
  AND2 slot as its NAND and inverter, every ``AN<i>`` as its inverter, each
  placed copy separately at its own column, all at the spec's drawn sizes;
* every net on those trees as the routed RC tree the decoder draws: its M4
  segment over each pair of rows it is in, the M3 column joining them, and
  each pin's M3 stub with its V2 and V3;
* every gate input on those nets that is not on the path -- the other words'
  NAND inputs and the other predecoder gates -- as a BSIM-CMG gate load at
  its own stub.

Address pins are ideal edges at the M4 rail end where the decoder labels
them.  The postdecode NAND, the wordline driver and everything after it are
:mod:`~chipforge_asap7.devices.rom_read_spice`'s.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from ..layout.rules import CONTACT_SIZE, M4_WIDTH
from ..verification.stack import STACK_CONDUCTORS

__all__ = ["PredecodeNetwork", "predecode_network"]

_STACK = {s.name: s for s in STACK_CONDUCTORS}
M3_OHM_PER_UM = _STACK["M3"].sheet_resistance_ohm / (CONTACT_SIZE / 1000)
M4_OHM_PER_UM = _STACK["M4"].sheet_resistance_ohm / (M4_WIDTH / 1000)
V2_OHM = _STACK["M2"].via_resistance_ohm
V3_OHM = _STACK["M3"].via_resistance_ohm


@dataclass(frozen=True)
class PredecodeNetwork:
    """The netlist, its address edges, and where the word's NAND inputs land."""

    lines: tuple[str, ...]
    #: ``(node, rises)`` of each address pin's ideal edge.
    sources: tuple[tuple[str, bool], ...]
    #: The selected word's postdecode NAND A (``PA``) and B (``PB``) pins.
    pa_pin: str
    pb_pin: str
    word: int
    #: Every net simulated: the address pins and the gates on the path.
    tree: tuple[str, ...]
    gates: int
    loads: int


def _name(line: str) -> str:
    """A SPICE-safe net name: ``PA<3>`` -> ``PAv3``, ``D1_2<0>`` -> ``D1_2v0``."""
    return line.replace("<", "v").replace(">", "")


def _fins(cell: Any) -> tuple[int, int]:
    """Total (n, p) fins of one input of a NAND/inverter cell."""
    n, p = cell.rows[0]
    return n * cell.fingers, p * cell.fingers


def predecode_network(
    spec: Any, word: int, *, nmos: str, pmos: str, wire_ff_per_um: float
) -> PredecodeNetwork:
    """The address-to-NAND path of `word` in `spec`, see the module docstring.

    The address starts at `word`'s complement, so every bit switches: a bit of
    `word` that is 0 falls and its ``AN`` inverter is on the path.
    """
    if not spec.predecode:
        raise ValueError("the decoder spec has no predecoders")
    pa, pb = spec.lines(word)
    by_line = defaultdict(list)
    for index, placed in enumerate(spec.predecoder):
        by_line[placed.gate.out].append(index)
    tree: set[str] = set()

    def visit(line: str) -> None:
        if line in tree:
            return
        tree.add(line)
        for index in by_line.get(line, ()):
            for source in spec.predecoder[index].gate.ins:
                visit(source)

    visit(pa)
    visit(pb)
    on_path = {i for line in tree for i in by_line.get(line, ())}

    # Every pin on a tree net: (pair, x, pin y, what is there).
    pins: dict[str, list[tuple[int, int, int, tuple]]] = defaultdict(list)
    for w in range(spec.array.words):
        row = spec.row_of(w)
        for pin, line in zip(("A", "B"), spec.lines(w)):
            if line in tree:
                x = spec.slot_x(w) + spec.stubs(w % 2)[pin]
                pins[line].append((row // 2, x, spec.pin_y(row, pin), ("post", w, pin)))
    for index, placed in enumerate(spec.predecoder):
        row = 2 * placed.pair + placed.row
        for pin, line in spec.pre_pins(placed.gate):
            if line in tree:
                x = spec.pre_x(placed.column) + spec.pre_tracks[placed.row][pin]
                y = spec.pin_y(row, pin, predecode=True)
                pins[line].append((placed.pair, x, y, ("pre", index, pin)))

    lines: list[str] = []
    counter = iter(range(10**9))

    def resistor(a: str, b: str, ohm: float) -> None:
        lines.append(f"RQ{next(counter)} {a} {b} {ohm:g}")

    caps: dict[str, float] = defaultdict(float)

    def wire_cap(node: str, length_nm: float) -> None:
        caps[node] += length_nm / 1000 * wire_ff_per_um

    pin_node: dict[tuple, str] = {}
    sources = []
    for line in sorted(tree):
        net = _name(line)
        pairs = sorted(p for p, name in spec.segments if name == line)
        join = spec.joins.get(line)

        def rail(pair: int, x: int, net: str = net) -> str:
            return f"{net}_p{pair}_x{x}"

        for pair in pairs:
            y, x0, x1 = spec.segments[pair, line]
            xs = {x0, x1} | {x for p, x, _, _ in pins[line] if p == pair}
            if join is not None:
                xs.add(join)
            xs = sorted(xs)
            for a, b in pairwise(xs):
                resistor(rail(pair, a), rail(pair, b), (b - a) / 1000 * M4_OHM_PER_UM)
                wire_cap(rail(pair, a), (b - a) / 2)
                wire_cap(rail(pair, b), (b - a) / 2)
            for p, x, pin_y, what in pins[line]:
                if p != pair:
                    continue
                # The M3 stub from the M4 rail down to the pin's V2.
                stub = abs(pin_y - y)
                node = f"{net}_{what[0]}{what[1]}{what[2]}"
                resistor(
                    rail(pair, x), node, V3_OHM + stub / 1000 * M3_OHM_PER_UM + V2_OHM
                )
                wire_cap(node, stub)
                pin_node[what] = node
        if join is not None and len(pairs) > 1:
            stops = sorted((spec.segments[p, line][0], p) for p in pairs)
            for (ya, a), (yb, b) in pairwise(stops):
                resistor(
                    rail(a, join),
                    rail(b, join),
                    (yb - ya) / 1000 * M3_OHM_PER_UM + 2 * V3_OHM,
                )
                wire_cap(rail(a, join), (yb - ya) / 2)
                wire_cap(rail(b, join), (yb - ya) / 2)
        if line not in by_line:
            # An address pin: its edge enters where the decoder labels it.
            first = pairs[0]
            sources.append((rail(first, spec.segments[first, line][1]), line))

    loads = 0
    post_n, post_p = _fins(spec.nand)
    pre_nand_n, pre_nand_p = _fins(spec.pre_slot.nand)
    inv_n, inv_p = _fins(spec.pre_slot.inverter)

    def load(node: str, n: int, p: int) -> None:
        nonlocal loads
        lines.append(f"NL{loads}n 0 {node} 0 0 {nmos} L=20n NFIN={n}")
        lines.append(f"NL{loads}p VDD {node} VDD VDD {pmos} L=20n NFIN={p}")
        loads += 1

    for what, node in pin_node.items():
        kind, index, pin = what
        if kind == "post" and index != word:
            load(node, post_n, post_p)
        elif kind == "pre" and pin != "Y" and index not in on_path:
            gate = spec.predecoder[index].gate
            load(
                node,
                *((pre_nand_n, pre_nand_p) if gate.kind == "and" else (inv_n, inv_p)),
            )

    for index in sorted(on_path):
        gate = spec.predecoder[index].gate
        out = pin_node["pre", index, "Y"]
        if gate.kind == "inv":
            a = pin_node["pre", index, "I"]
            lines += [
                f"NX{index}n {out} {a} 0 0 {nmos} L=20n NFIN={inv_n}",
                f"NX{index}p {out} {a} VDD VDD {pmos} L=20n NFIN={inv_p}",
            ]
            continue
        a, b = pin_node["pre", index, "A"], pin_node["pre", index, "B"]
        mid, nand = f"X{index}mid", f"X{index}nand"
        lines += [
            f"NX{index}a {nand} {a} {mid} 0 {nmos} L=20n NFIN={pre_nand_n}",
            f"NX{index}b {mid} {b} 0 0 {nmos} L=20n NFIN={pre_nand_n}",
            f"NX{index}c {nand} {a} VDD VDD {pmos} L=20n NFIN={pre_nand_p}",
            f"NX{index}d {nand} {b} VDD VDD {pmos} L=20n NFIN={pre_nand_p}",
            f"NX{index}e {out} {nand} 0 0 {nmos} L=20n NFIN={inv_n}",
            f"NX{index}f {out} {nand} VDD VDD {pmos} L=20n NFIN={inv_p}",
        ]
    lines += [f"CQ_{node} {node} 0 {ff:g}f" for node, ff in sorted(caps.items()) if ff]
    edges = tuple(
        (node, bool(word >> int(line[2:-1]) & 1)) for node, line in sorted(sources)
    )
    return PredecodeNetwork(
        lines=tuple(lines),
        sources=edges,
        pa_pin=pin_node["post", word, "A"],
        pb_pin=pin_node["post", word, "B"],
        word=word,
        tree=tuple(sorted(tree)),
        gates=len(on_path),
        loads=loads,
    )
