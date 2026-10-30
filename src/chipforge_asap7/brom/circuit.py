"""Technology-specific transistor circuits shared by BROM CDL and layout.

The topology follows GT2N's factored candidate generator. Device dimensions
are ASAP7 fin counts, never GT2N sheet widths or scaled GT2N geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..devices.finfet import FinFETSpec
from .selector import SelectorPlan


@dataclass(frozen=True)
class Device:
    name: str
    spec: FinFETSpec
    nodes: tuple[str, str, str, str]


@dataclass(frozen=True)
class SelectorInstance:
    plan: SelectorPlan
    mapping: tuple[tuple[int, ...], ...]
    device_names: frozenset[str]


@dataclass(frozen=True)
class ReplicaInstance:
    selector: SelectorInstance
    port_nets: dict[str, str]
    guard_pairs: int
    extra_blocks: int


class Circuit:
    def __init__(
        self, name: str, pins: tuple[str, ...], fins: int = 2, vt: str = "rvt"
    ):
        self.name, self.pins = name, pins
        self.n = FinFETSpec(flavor="n", fins=fins, vt=vt)
        self.p = FinFETSpec(flavor="p", fins=fins, vt=vt)
        self.devices: list[Device] = []
        self._serial = 0
        self.array: SelectorInstance | None = None
        self.replica: ReplicaInstance | None = None
        self.placement_groups: list[tuple[str, tuple[str, ...]]] = []
        self.array_width: int | None = None
        self.distribution: dict | None = None
        self.output_register: dict | None = None

    def group_since(self, name: str, start: int) -> None:
        """Keep a functional group together during peripheral placement."""
        self.placement_groups.append(
            (name, tuple(d.name for d in self.devices[start:]))
        )

    def node(self) -> str:
        self._serial += 1
        return f"n{self._serial}"

    def fet(
        self, pmos: bool, d: str, g: str, s: str, *, fins: int | None = None
    ) -> None:
        self.devices.append(
            Device(
                f"M{len(self.devices)}",
                (self.p if pmos else self.n)
                if fins is None
                else replace(self.p if pmos else self.n, fins=fins),
                (d, g, s, "VDD" if pmos else "VSS"),
            )
        )

    def inv(self, a: str, y: str) -> None:
        self.fet(False, y, a, "VSS")
        self.fet(True, y, a, "VDD")

    def nand(self, a: str, b: str, y: str) -> None:
        middle = self.node()
        self.fet(False, y, a, middle)
        self.fet(False, middle, b, "VSS")
        self.fet(True, y, a, "VDD")
        self.fet(True, y, b, "VDD")

    def and2(self, a: str, b: str, y: str) -> None:
        inverted = self.node()
        self.nand(a, b, inverted)
        self.inv(inverted, y)

    def or2(self, a: str, b: str, y: str) -> None:
        inverted = self.node()
        self.nor(a, b, inverted)
        self.inv(inverted, y)

    def nor(self, a: str, b: str, y: str) -> None:
        middle = self.node()
        self.fet(True, y, a, middle)
        self.fet(True, middle, b, "VDD")
        self.fet(False, y, a, "VSS")
        self.fet(False, y, b, "VSS")

    def xor(self, a: str, b: str, y: str) -> None:
        common, left, right = self.node(), self.node(), self.node()
        self.nand(a, b, common)
        self.nand(a, common, left)
        self.nand(b, common, right)
        self.nand(left, right, y)

    def buffer(self, a: str, y: str) -> None:
        mid = self.node()
        self.inv(a, mid)
        self.inv(mid, y)

    def tg(self, a: str, b: str, en: str, enb: str) -> None:
        self.fet(False, b, en, a)
        self.fet(True, b, enb, a)

    def latch(self, d: str, q: str, en: str, enb: str) -> None:
        """Static noninverting transmission-gate latch, transparent on en."""
        storage, qb = self.node(), self.node()
        self.tg(d, storage, en, enb)
        self.inv(storage, qb)
        self.inv(qb, q)
        self.tg(q, storage, enb, en)

    def netlist(self) -> str:
        return "\n".join(
            [
                f".subckt {self.name} {' '.join(self.pins)}",
                *(d.spec.netlist(d.name, d.nodes) for d in self.devices),
                f".ends {self.name}",
                "",
            ]
        )

    def lvs_reference(self) -> str:
        """Unit-fin reference for the packaged ASAP7 KLayout extractor."""
        lines = [f".subckt {self.name} {' '.join(self.pins)}"]
        for d in self.devices:
            for fin in range(d.spec.total_fins):
                lines.append(
                    f"{d.name}_f{fin} {' '.join(d.nodes)} {d.spec.model} L=20n W=7n"
                )
        return "\n".join([*lines, f".ends {self.name}", ".end", ""])
