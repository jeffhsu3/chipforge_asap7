"""Read extracted parasitics back as numbers a decision can be made on.

`run_open_pex` leaves a post-layout SPICE netlist holding everything the two
extractors found: ``Rext`` segments of the distributed resistance network, in
which a pin is the bare net name (``A``) and a device terminal is a named node
on it (``A.P3.27``), and ``Cext`` lumped couplings between whole nets.  This
module turns that into two things:

* capacitance per net and per pair, and a delay-flavoured *cost* in which
  coupling between two signal nets is weighted up, because a wire that couples
  an input to the output it drives is a Miller capacitance;
* for every pin, the effective resistance from the pin to each device terminal
  on its net -- worst and mean -- by solving the resistor network.  This is what
  a redundant contact or a metal strap buys, and LVS cannot see it.  A net
  with no pin (a NAND's output inside a driver slice) is measured from the
  diffusions that drive it to the gates it drives.  The extractor numbers such
  nets as it finds them, so they are named here by the pins their devices
  touch -- ``int[B0,SEL,WL0]`` -- which survives an edit to the layout.

Resistance comes from KPEX square counting and is deterministic; capacitance
comes from a field solver and carries its mesh tolerance, so compare costs
with a margin and resistances almost exactly.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Parasitics", "effective_resistance", "pairwise_resistance"]

_SUFFIX = {
    "meg": 1e6,
    "a": 1e-18,
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "m": 1e-3,
    "k": 1e3,
    "g": 1e9,
    "t": 1e12,
}
_NUMBER = re.compile(r"^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)([a-zA-Z]*)$")


def _value(text: str) -> float:
    """A SPICE number: ``133.003a``, ``4.11``, ``1e-3``, ``2meg``."""
    match = _NUMBER.match(text.strip())
    if not match:
        raise ValueError(f"not a SPICE value: {text!r}")
    number, suffix = float(match.group(1)), match.group(2).lower()
    for key, scale in _SUFFIX.items():  # "meg" first, so it is not read as milli
        if suffix.startswith(key):
            return number * scale
    return number


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


#: Below this a net is one floored zero-ohm element (a series stack's shared
#: diffusion, per fin), and there is nothing on it to protect.
_NEGLIGIBLE_OHM = 0.01
_TERMINAL = re.compile(r"^(.*)\.[A-Za-z$]+\d+\.\d+$")


def _net_of(node: str) -> str:
    """``A.P3.27`` (a device terminal) or ``A.$4.29`` (a junction) -> ``A``."""
    match = _TERMINAL.match(node)
    return match.group(1) if match else node


def pairwise_resistance(
    edges: list[tuple[str, str, float]], sources: set[str], sinks: set[str]
) -> list[float]:
    """Effective resistance of every connected (source, sink) pair of distinct nodes."""
    import numpy as np  # as in `effective_resistance`

    nodes = sorted({n for a, b, _ in edges for n in (a, b)})
    index = {node: i for i, node in enumerate(nodes)}
    laplacian = np.zeros((len(nodes), len(nodes)))
    for a, b, ohms in edges:
        conductance = 1.0 / max(ohms, 1e-6)
        i, j = index[a], index[b]
        laplacian[i, i] += conductance
        laplacian[j, j] += conductance
        laplacian[i, j] -= conductance
        laplacian[j, i] -= conductance
    inverse = np.linalg.pinv(laplacian)
    # Same island <=> reachable; a pseudo-inverse gives finite nonsense across islands.
    island: dict[str, int] = {}
    for start in nodes:
        if start in island:
            continue
        island[start], frontier = len(island), [start]
        while frontier:
            here = frontier.pop()
            for a, b, _ in edges:
                for this, other in ((a, b), (b, a)):
                    if this == here and other not in island:
                        island[other] = island[start]
                        frontier.append(other)
    found = []
    for source in sorted(sources & index.keys()):
        for sink in sorted(sinks & index.keys()):
            if source != sink and island[source] == island[sink]:
                i, j = index[source], index[sink]
                found.append(float(inverse[i, i] + inverse[j, j] - 2 * inverse[i, j]))
    return found


def effective_resistance(
    edges: list[tuple[str, str, float]], source: str, targets: set[str]
) -> dict[str, float]:
    """Effective resistance from `source` to each reachable node of `targets`.

    Grounds `source`, inverts the reduced Laplacian of its connected component
    and reads the diagonal: ``R_eff(source, t) = (L^-1)[t, t]``.
    """
    import numpy as np  # not a dependency of the package; comes with the gds and pex extras

    neighbours: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for a, b, ohms in edges:
        conductance = 1.0 / max(ohms, 1e-6)
        neighbours[a].append((b, conductance))
        neighbours[b].append((a, conductance))
    if source not in neighbours:
        return {}
    component, frontier = {source}, [source]
    while frontier:
        for other, _ in neighbours[frontier.pop()]:
            if other not in component:
                component.add(other)
                frontier.append(other)
    nodes = sorted(component - {source})
    wanted = [n for n in nodes if n in targets]
    if not wanted:
        return {}
    index = {node: i for i, node in enumerate(nodes)}
    laplacian = np.zeros((len(nodes), len(nodes)))
    for node in nodes:
        for other, conductance in neighbours[node]:
            laplacian[index[node], index[node]] += conductance
            if other != source:
                laplacian[index[node], index[other]] -= conductance
    columns = np.zeros((len(nodes), len(wanted)))
    for column, node in enumerate(wanted):
        columns[index[node], column] = 1.0
    solved = np.linalg.solve(laplacian, columns)
    return {
        node: float(solved[index[node], column]) for column, node in enumerate(wanted)
    }


@dataclass(frozen=True)
class Parasitics:
    """Capacitance between nets (fF) and pin-to-device resistance per pin net (ohm)."""

    pins: tuple[str, ...]
    coupling_fF: dict[tuple[str, str], float] = field(default_factory=dict)
    #: net -> (worst, mean) effective resistance: from the pin to each device
    #: terminal, or on a net without a pin, from each driving diffusion to each gate
    resistance_ohm: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def from_post_layout(
        cls, netlist: str | Path, supplies: tuple[str, ...] = ("VDD", "VSS")
    ) -> Parasitics:
        lines: list[str] = []
        for raw in Path(netlist).read_text().splitlines():
            if raw.startswith("+") and lines:
                lines[-1] += " " + raw[1:].strip()
            else:
                lines.append(raw.strip())

        pins: tuple[str, ...] = ()
        resistors: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        coupling: dict[tuple[str, str], float] = defaultdict(float)
        devices: list[tuple[str, str, str]] = []  # drain, gate, source nodes
        for line in lines:
            fields = line.replace("\\", "").split()
            if not fields or line.startswith("*"):
                continue
            head = fields[0].upper()
            if head == ".SUBCKT" and not pins:
                pins = tuple(fields[2:])
            elif head.startswith("R") and len(fields) >= 4:
                resistors[_net_of(fields[1])].append(
                    (fields[1], fields[2], _value(fields[3]))
                )
            elif head.startswith("C") and len(fields) >= 4:
                coupling[_pair(fields[1], fields[2])] += _value(fields[3]) * 1e15
            elif head[0] in "MN" and len(fields) >= 6:
                devices.append((fields[1], fields[2], fields[3]))

        gates: dict[str, set[str]] = defaultdict(set)
        diffusions: dict[str, set[str]] = defaultdict(set)
        touches: dict[str, set[str]] = defaultdict(
            set
        )  # net -> pins its devices also touch
        for drain, gate, source in devices:
            nets = {_net_of(node) for node in (drain, gate, source)}
            for node, group in (
                (drain, diffusions),
                (gate, gates),
                (source, diffusions),
            ):
                group[_net_of(node)].add(node)
                touches[_net_of(node)] |= {
                    n for n in nets if n in pins and n not in supplies
                }

        resistance: dict[str, tuple[float, float]] = {}
        internal: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for net, edges in resistors.items():
            if net in pins:
                values = list(
                    effective_resistance(
                        edges, net, (gates[net] | diffusions[net]) - {net}
                    ).values()
                )
            else:
                everything = gates[net] | diffusions[net]
                values = pairwise_resistance(
                    edges, diffusions[net] or everything, gates[net] or everything
                )
            if not values or max(values) < _NEGLIGIBLE_OHM:
                continue
            measured = (max(values), sum(values) / len(values))
            if net in pins:
                resistance[net] = measured
            else:
                internal[f"int[{','.join(sorted(touches[net] - {net}))}]"].append(
                    measured
                )
        for (
            label,
            found,
        ) in internal.items():  # nets the pins cannot tell apart: worst first
            for rank, measured in enumerate(sorted(found, reverse=True)):
                resistance[label if len(found) == 1 else f"{label}#{rank}"] = measured
        return cls(pins=pins, coupling_fF=dict(coupling), resistance_ohm=resistance)

    def capacitance_fF(self, net: str) -> float:
        """Everything `net` couples to."""
        return sum(c for pair, c in self.coupling_fF.items() if net in pair)

    def between_fF(self, a: str, b: str) -> float:
        return self.coupling_fF.get(_pair(a, b), 0.0)

    def signals(self, supplies: tuple[str, ...] = ("VDD", "VSS")) -> tuple[str, ...]:
        return tuple(p for p in self.pins if p not in supplies)

    def cost_fF(
        self, supplies: tuple[str, ...] = ("VDD", "VSS"), miller: float = 2.0
    ) -> float:
        """Switched-capacitance cost: each signal's load, signal-to-signal coupling x `miller`.

        A coupling between two signals is seen from both ends, so it enters
        twice, each time weighted by `miller`; the default of two treats it as
        the full Miller capacitance of an inverting stage.
        """
        signals = set(self.signals(supplies))
        total = 0.0
        for (a, b), c in self.coupling_fF.items():
            for this, other in ((a, b), (b, a)):
                if this in signals:
                    total += c * (miller if other in signals else 1.0)
        return total

    def summary(self, supplies: tuple[str, ...] = ("VDD", "VSS")) -> dict:
        signals = self.signals(supplies)
        return {
            "cost_fF": round(self.cost_fF(supplies), 5),
            "capacitance_fF": {
                net: round(self.capacitance_fF(net), 5) for net in signals
            },
            "signal_coupling_fF": {
                f"{a}-{b}": round(c, 5)
                for (a, b), c in sorted(self.coupling_fF.items())
                if a in signals and b in signals
            },
            "pin_to_device_ohm": {
                net: {"worst": round(w, 3), "mean": round(m, 3)}
                for net, (w, m) in sorted(self.resistance_ohm.items())
            },
        }
