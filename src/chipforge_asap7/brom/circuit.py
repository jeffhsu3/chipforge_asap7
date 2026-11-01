"""Technology-specific transistor circuits shared by BROM CDL and layout.

The topology follows GT2N's factored candidate generator. Device dimensions
are ASAP7 fin counts, never GT2N sheet widths or scaled GT2N geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import wraps

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


def logic_unit(fn):
    """Keep semantic cells independent of subsequent physical placement."""

    @wraps(fn)
    def build(self, *args, **kwargs):
        outer = self._unit_depth == 0
        start = len(self.devices)
        self._unit_depth += 1
        try:
            result = fn(self, *args, **kwargs)
        finally:
            self._unit_depth -= 1
        if outer:
            self.logic_units.append(
                (fn.__name__, tuple(d.name for d in self.devices[start:]))
            )
        return result

    return build


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
        self.sample_distribution: dict | None = None
        self.compact_periphery = False
        self.logic_units = []
        self._unit_depth = 0

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

    @logic_unit
    def inv(self, a: str, y: str) -> None:
        self.fet(False, y, a, "VSS")
        self.fet(True, y, a, "VDD")

    @logic_unit
    def nand(self, a: str, b: str, y: str) -> None:
        middle = self.node()
        self.fet(False, y, a, middle)
        self.fet(False, middle, b, "VSS")
        self.fet(True, y, a, "VDD")
        self.fet(True, y, b, "VDD")

    @logic_unit
    def and2(self, a: str, b: str, y: str) -> None:
        inverted = self.node()
        self.nand(a, b, inverted)
        self.inv(inverted, y)

    @logic_unit
    def or2(self, a: str, b: str, y: str) -> None:
        inverted = self.node()
        self.nor(a, b, inverted)
        self.inv(inverted, y)

    @logic_unit
    def nor(self, a: str, b: str, y: str) -> None:
        middle = self.node()
        self.fet(True, y, a, middle)
        self.fet(True, middle, b, "VDD")
        self.fet(False, y, a, "VSS")
        self.fet(False, y, b, "VSS")

    @logic_unit
    def aoi22(self, a, b, c, d, y):
        left, right, middle = self.node(), self.node(), self.node()
        self.fet(False, y, a, left)
        self.fet(False, left, b, "VSS")
        self.fet(False, y, c, right)
        self.fet(False, right, d, "VSS")
        self.fet(True, middle, a, "VDD")
        self.fet(True, middle, b, "VDD")
        self.fet(True, y, c, middle)
        self.fet(True, y, d, middle)

    def mux4(self, data, select, output):
        """20T sum of four products, including non-one-hot selects."""
        a, b = self.node(), self.node()
        self.aoi22(data[0], select[0], data[1], select[1], a)
        self.aoi22(data[2], select[2], data[3], select[3], b)
        self.nand(a, b, output)

    def xor(self, a: str, b: str, y: str) -> None:
        common, left, right = self.node(), self.node(), self.node()
        self.nand(a, b, common)
        self.nand(a, common, left)
        self.nand(b, common, right)
        self.nand(left, right, y)

    @logic_unit
    def buffer(self, a: str, y: str) -> None:
        mid = self.node()
        self.inv(a, mid)
        self.inv(mid, y)

    def tg(self, a: str, b: str, en: str, enb: str) -> None:
        self.fet(False, b, en, a)
        self.fet(True, b, enb, a)

    @logic_unit
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
