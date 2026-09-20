"""Layout reduction: delete and shrink what a cell turns out not to need.

Generators inherit geometry.  `InverterSpec` copied two tall M1 input straps
and a contact per row from the released decoder inverter, where they are pin
access for a parent that straps them together; in a cell whose parent uses one
landing, most of that metal is dead weight running 27 nm from the output.
Nothing in a generator says which shapes are load-bearing.  This pass finds
out, the way test-case reducers (delta debugging, C-Reduce) shrink a failing
program: try removing a piece, keep the removal if the result is still right.

"Still right" is an oracle in three parts, cheapest first:

1. LVS still matches and the extracted cell still has every pin of the
   reference -- the shape was not needed for connectivity.  The pin check is
   separate because KLayout pairs nets by topology: a pin label that has lost
   its metal still "matches".  The shape under each top-level pin label is
   pinned as well: never deleted, and only shrunk to a landing around the label.
2. No DRC category gets worse than it was at the start.
3. Extraction agrees: no pin's worst pin-to-device resistance rises by more
   than `r_tolerance` (so a redundant contact or a strap that lowers
   resistance survives, which LVS alone would happily delete), and the
   switched-capacitance cost does not go up (so a shield does too).
   Resistance extraction alone takes seconds and is exact, so it gates every
   batch and assigns blame shape by shape; the field solve behind the
   capacitance takes minutes on anything larger than a leaf, so it runs once
   before, once after the deletions and once after the shrinks.

Deletions run over vias, then metals, then local interconnect, and repeat to a
fixed point, because deleting a via is what makes the metal above it
deletable.  Single deletions are probed in parallel against the current
layout, their union is verified, and only on disagreement does it fall back to
one at a time.  Then each surviving wire has its ends pulled in by a parallel
k-section search.  Shapes that touch their cell's BOUNDARY are interface
geometry -- rails, abutting stubs, a wordline leaving the top edge -- and are
never edited.

What it reports is the point: a list of edits per cell and layer.  A reduced
GDS is not a generator; the findings are meant to be promoted into spec knobs.
Reduce a cell *in the context that uses it*: a leaf reduced alone will give up
pin landings its parent needs, since nothing tells it they are wanted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..layout.layers import LAYERS, require_gdspy
from .drc import drc_counts, run_drc
from .lvs import run_lvs
from .parasitics import Parasitics

__all__ = [
    "Edit",
    "ReductionConfig",
    "ReductionResult",
    "cost_objection",
    "parasitic_objection",
    "reduce_layout",
    "resistance_objection",
]

Box = tuple[int, int, int, int]
Key = tuple[str, int]  # (cell name, polygon index)
State = dict[Key, "Box | None"]  # absent = as drawn, None = deleted, Box = resized

#: Deletion order within a pass: a via first frees the metal that only it used.
DELETE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("V2", "V1", "V0"),
    ("M3", "M2", "M1"),
    ("LIG",),
)
_LAYER_NAME = {}
for _name, _spec in LAYERS.items():
    _LAYER_NAME.setdefault((_spec["layer"], _spec["datatype"]), _name)
_BOUNDARY = (LAYERS["BOUNDARY"]["layer"], LAYERS["BOUNDARY"]["datatype"])


@dataclass(frozen=True)
class ReductionConfig:
    """What may be edited and what the result must still satisfy."""

    layers: tuple[str, ...] = ("V2", "V1", "V0", "M3", "M2", "M1", "LIG")
    shrink_layers: tuple[str, ...] = ("M3", "M2", "M1")
    cells: tuple[str, ...] | None = None  # None: every cell under the top
    shrink: bool = True
    min_length: int = 18  # never shrink a wire below this along its length
    min_gain: int = 9  # do not bother searching for less than this
    pin_landing: int = 18  # metal kept either side of a pin label, along the wire
    max_passes: int = 4
    tie_bodies: bool = True
    drc: bool = True
    pex: bool = True
    capacitance: bool = (
        True  # False: hold resistance only and never run the field solver
    )
    supplies: tuple[str, ...] = ("VDD", "VSS")
    substrate_net: str = "VSS"
    r_tolerance: float = 0.05
    cost_tolerance: float = 0.03  # layout-to-layout noise of a converged solve
    miller: float = 2.0
    #: FasterCap refines its own mesh until two passes agree to `pex_tolerance`, so
    #: the input mesh can stay coarse but the tolerance cannot: at 0.08-0.15 the
    #: cost of one inverter moved by -56 % to +6 % for edits worth -4 % and -9 %,
    #: while 0.02 tracks a hundredfold finer input mesh to within a point.
    pex_mesh_area_um2: float = 4.0
    pex_tolerance: float = 0.02
    pex_timeout: float = 7200
    jobs: int = field(default_factory=lambda: min(8, os.cpu_count() or 1))
    keep_probes: bool = False


@dataclass(frozen=True)
class Edit:
    kind: str  # "delete" or "shrink"
    cell: str
    layer: str
    before: Box
    after: Box | None
    covered: bool = (
        False  # what was removed lay under other shapes of its layer: a duplicate
    )

    @property
    def removed_nm2(self) -> int:
        def area(b: Box | None) -> int:
            return 0 if b is None else (b[2] - b[0]) * (b[3] - b[1])

        return area(self.before) - area(self.after)


@dataclass
class ReductionResult:
    top: str
    gds: Path
    report: Path
    edits: list[Edit]
    kept_for_parasitics: list[dict]
    before: Parasitics | None
    after: Parasitics | None
    probes: int
    seconds: float

    def summary(self, supplies: tuple[str, ...] = ("VDD", "VSS")) -> str:
        lines = [
            f"{self.top}: {len(self.edits)} edit(s) from {self.probes} probes in {self.seconds:.0f} s"
        ]
        per_layer: dict[str, list[Edit]] = {}
        for edit in self.edits:
            per_layer.setdefault(edit.layer, []).append(edit)
        for layer, edits in per_layer.items():
            deleted = sum(1 for e in edits if e.after is None)
            duplicates = sum(1 for e in edits if e.covered)
            lines.append(
                f"  {layer:4s} {deleted} deleted, {len(edits) - deleted} shrunk, "
                f"{sum(e.removed_nm2 for e in edits)} nm2 removed"
                + (
                    f" ({duplicates} were duplicates under other {layer})"
                    if duplicates
                    else ""
                )
            )
        for kept in self.kept_for_parasitics:
            lines.append(
                f"  kept {kept['layer']} {kept['box']} in {kept['cell']}: {kept['reason']}"
            )
        if self.before and self.after:
            b, a = self.before.cost_fF(supplies), self.after.cost_fF(supplies)
            if b > 0:
                lines.append(
                    f"  switched-capacitance cost {b:.4f} -> {a:.4f} fF ({(a / b - 1) * 100:+.1f} %)"
                )
                signals = self.before.signals(supplies)
                for pair in sorted(self.before.coupling_fF):
                    if all(net in signals for net in pair):
                        lines.append(
                            f"  {pair[0]}-{pair[1]} coupling {self.before.between_fF(*pair):.4f} -> "
                            f"{self.after.between_fF(*pair):.4f} fF"
                        )
            for net, (worst, _) in sorted(self.after.resistance_ohm.items()):
                was = self.before.resistance_ohm.get(net, (float("nan"), 0))[0]
                lines.append(
                    f"  {net}: worst {_span(net)} R {was:.1f} -> {worst:.1f} ohm"
                )
        return "\n".join(lines)


# ── Layout model ─────────────────────────────────────────────────────────────
class LayoutModel:
    """A GDS held as editable rectangles plus everything else verbatim."""

    def __init__(self, gds: str | Path, top: str) -> None:
        gdspy = require_gdspy()
        library = gdspy.GdsLibrary(
            infile=str(gds), unit=1e-9, precision=1e-10, units="convert"
        )
        if top not in library.cells:
            raise ValueError(f"{top!r} is not in {gds}; it has {sorted(library.cells)}")
        self.top = top
        self.order: list[str] = []  # dependencies first
        self.polys: dict[str, list[tuple[int, int, Any, Box | None]]] = {}
        self.labels: dict[str, list[Any]] = {}
        self.refs: dict[str, list[Any]] = {}
        self.boundary: dict[str, Box | None] = {}

        def visit(cell: Any) -> None:
            if cell.name in self.polys:
                return
            if cell.paths:
                raise NotImplementedError(
                    f"{cell.name} holds GDS paths; only polygons are supported"
                )
            for ref in cell.references:
                if not hasattr(ref, "origin") or hasattr(ref, "columns"):
                    raise NotImplementedError(f"{cell.name} holds a cell array")
                visit(ref.ref_cell)
            polys = []
            for pset in cell.polygons:
                for points, layer, datatype in zip(
                    pset.polygons, pset.layers, pset.datatypes
                ):
                    polys.append(
                        (int(layer), int(datatype), points, self._rect(points))
                    )
            self.polys[cell.name] = polys
            self.labels[cell.name] = list(cell.labels)
            self.refs[cell.name] = list(cell.references)
            frames = [p[3] for p in polys if (p[0], p[1]) == _BOUNDARY and p[3]]
            self.boundary[cell.name] = (
                (
                    min(f[0] for f in frames),
                    min(f[1] for f in frames),
                    max(f[2] for f in frames),
                    max(f[3] for f in frames),
                )
                if frames
                else None
            )
            self.order.append(cell.name)

        visit(library.cells[top])
        #: shape -> pin-label points it carries, in the shape's own cell frame
        self.pinned: dict[Key, list[tuple[float, float]]] = {}
        for label in self.labels[top]:
            self._pin(
                top,
                int(label.layer),
                (float(label.position[0]), float(label.position[1])),
            )

    def _pin(self, cell: str, gds_layer: int, point: tuple[float, float]) -> None:
        """Pin whatever drawn rectangle of `gds_layer` lies under `point`, at any depth."""
        for index, (layer, _, _, box) in enumerate(self.polys[cell]):
            if (
                layer == gds_layer
                and box is not None
                and box[0] <= point[0] <= box[2]
                and box[1] <= point[1] <= box[3]
            ):
                self.pinned.setdefault((cell, index), []).append(point)
        for ref in self.refs[cell]:
            self._pin(ref.ref_cell.name, gds_layer, _into_reference(ref, point))

    @staticmethod
    def _rect(points: Any) -> Box | None:
        if len(points) != 4:
            return None
        xs, ys = (
            sorted({round(float(p[0])) for p in points}),
            sorted({round(float(p[1])) for p in points}),
        )
        return (xs[0], ys[0], xs[1], ys[1]) if len(xs) == 2 and len(ys) == 2 else None

    def layer(self, key: Key) -> str:
        layer, datatype, _, _ = self.polys[key[0]][key[1]]
        return _LAYER_NAME.get((layer, datatype), f"{layer}/{datatype}")

    def box(self, key: Key, state: State) -> Box | None:
        return state[key] if key in state else self.polys[key[0]][key[1]][3]

    def covered(self, key: Key, region: Box, state: State) -> bool:
        """True when other rectangles of the same layer and cell cover all of `region`."""
        layer, datatype = self.polys[key[0]][key[1]][:2]
        pieces = [region]
        for index, (other_layer, other_datatype, _, _) in enumerate(self.polys[key[0]]):
            other = (key[0], index)
            if other == key or (other_layer, other_datatype) != (layer, datatype):
                continue
            box = self.box(other, state)
            if box is None:
                continue
            remaining = []
            for piece in pieces:
                remaining += _subtract(piece, box)
            pieces = remaining
            if not pieces:
                return True
        return False

    def is_interface(self, key: Key) -> bool:
        """True when the shape reaches its cell's BOUNDARY: abutment geometry."""
        frame, box = self.boundary[key[0]], self.polys[key[0]][key[1]][3]
        if frame is None or box is None:
            return False
        return (
            box[0] <= frame[0]
            or box[1] <= frame[1]
            or box[2] >= frame[2]
            or box[3] >= frame[3]
        )

    def candidates(
        self,
        layers: tuple[str, ...],
        cells: tuple[str, ...] | None,
        state: State,
        *,
        pinned: bool = False,
    ) -> list[Key]:
        """Editable shapes; those under a pin label only when `pinned` (they may shrink)."""
        keys = []
        for cell in self.order:
            if cells is not None and cell not in cells:
                continue
            for index, (_, _, _, box) in enumerate(self.polys[cell]):
                key = (cell, index)
                if box is None or state.get(key, box) is None:
                    continue
                if key in self.pinned and not pinned:
                    continue
                if self.layer(key) in layers and not self.is_interface(key):
                    keys.append(key)
        return keys

    def write(self, path: str | Path, state: State) -> None:
        gdspy = require_gdspy()
        library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
        built: dict[str, Any] = {}
        for name in self.order:
            cell = gdspy.Cell(name, exclude_from_current=True)
            for index, (layer, datatype, points, box) in enumerate(self.polys[name]):
                key = (name, index)
                if key in state:
                    box = state[key]
                    if box is None:
                        continue
                    cell.add(
                        gdspy.Rectangle(
                            box[:2], box[2:], layer=layer, datatype=datatype
                        )
                    )
                else:
                    cell.add(gdspy.Polygon(points, layer=layer, datatype=datatype))
            for label in self.labels[name]:
                cell.add(
                    gdspy.Label(
                        label.text,
                        label.position,
                        layer=label.layer,
                        texttype=label.texttype,
                    )
                )
            for ref in self.refs[name]:
                cell.add(
                    gdspy.CellReference(
                        built[ref.ref_cell.name],
                        origin=ref.origin,
                        rotation=ref.rotation,
                        magnification=ref.magnification,
                        x_reflection=ref.x_reflection,
                    )
                )
            built[name] = cell
            library.add(cell, include_dependencies=False)
        library.write_gds(str(path))


def _into_reference(ref: Any, point: tuple[float, float]) -> tuple[float, float]:
    """Map a point of the parent into the frame of the referenced cell."""
    x, y = point[0] - ref.origin[0], point[1] - ref.origin[1]
    angle = -math.radians(ref.rotation or 0.0)
    x, y = (
        x * math.cos(angle) - y * math.sin(angle),
        x * math.sin(angle) + y * math.cos(angle),
    )
    scale = ref.magnification or 1.0
    return x / scale, (-y if ref.x_reflection else y) / scale


def _subckt_pins(netlist: Path, name: str) -> set[str]:
    """Named pins of ``.SUBCKT name``.

    Reads a plain SPICE header, continuation lines included, and KLayout's
    extracted form, which numbers the nets and names the pins in ``* pin``
    comments between ``* cell name`` and the header.
    """
    header: list[str] | None = None
    commented: list[str] = []
    in_cell = False
    for raw in netlist.read_text().splitlines():
        fields = raw.split()
        if header is not None:
            if not raw.startswith("+"):
                break
            header += raw[1:].split()
        elif fields[:2] == ["*", "cell"]:
            in_cell = len(fields) > 2 and fields[2].upper() == name.upper()
        elif in_cell and fields[:2] == ["*", "pin"] and len(fields) > 2:
            commented.append(fields[2])
        elif (
            len(fields) >= 2
            and fields[0].upper() == ".SUBCKT"
            and fields[1].upper() == name.upper()
        ):
            header = fields[2:]
    return {pin.upper() for pin in (commented or header or [])}


def _subtract(a: Box, b: Box) -> list[Box]:
    """The parts of rectangle `a` outside rectangle `b`."""
    if b[0] >= a[2] or b[2] <= a[0] or b[1] >= a[3] or b[3] <= a[1]:
        return [a]
    parts: list[Box] = []
    if b[1] > a[1]:
        parts.append((a[0], a[1], a[2], b[1]))
    if b[3] < a[3]:
        parts.append((a[0], b[3], a[2], a[3]))
    low, high = max(a[1], b[1]), min(a[3], b[3])
    if b[0] > a[0]:
        parts.append((a[0], low, b[0], high))
    if b[2] < a[2]:
        parts.append((b[2], low, a[2], high))
    return parts


def _with_edge(box: Box, edge: int, value: int) -> Box:
    resized = list(box)
    resized[edge] = value
    return (resized[0], resized[1], resized[2], resized[3])


def _span(net: str) -> str:
    """What a net's resistance was measured across (see `Parasitics.resistance_ohm`)."""
    return "driver-to-gate" if net.startswith("int[") else "pin-to-device"


def resistance_objection(
    cfg: ReductionConfig, base: Parasitics, new: Parasitics
) -> str:
    """Why `new` is rejected for resistance, or an empty string.

    Held against `base`, the unedited layout, so the tolerance does not
    compound over passes.
    """
    for net, (worst, _) in base.resistance_ohm.items():
        now = new.resistance_ohm.get(net)
        if now is None:
            return f"{net} no longer reaches its devices"
        if now[0] > worst * (1 + cfg.r_tolerance) + 0.05:
            return (
                f"{net} worst {_span(net)} resistance {worst:.1f} -> {now[0]:.1f} ohm"
            )
    return ""


def cost_objection(cfg: ReductionConfig, last_cost: float, new: Parasitics) -> str:
    """Why `new` is rejected for capacitance, against the best cost seen so far."""
    cost = new.cost_fF(cfg.supplies, cfg.miller)
    if cost > last_cost * (1 + cfg.cost_tolerance):
        return f"switched-capacitance cost {last_cost:.4f} -> {cost:.4f} fF"
    return ""


def parasitic_objection(
    cfg: ReductionConfig, base: Parasitics, last_cost: float, new: Parasitics
) -> str:
    """Both floors: resistance first, then cost."""
    return resistance_objection(cfg, base, new) or cost_objection(cfg, last_cost, new)


# ── Reducer ──────────────────────────────────────────────────────────────────
_PEX_WORKER = """
import json, sys
from chipforge_asap7.verification.pex import run_open_pex
result = run_open_pex(**json.loads(sys.argv[1]))
print("POST_LAYOUT=" + str(result.post_layout_netlist))
"""


class _Reducer:
    def __init__(
        self, model: LayoutModel, schematic: Path, out: Path, cfg: ReductionConfig
    ) -> None:
        self.model, self.schematic, self.out, self.cfg = model, schematic, out, cfg
        self.top = model.top
        self.cache: dict[tuple, tuple[bool, str]] = {}
        self.lock = threading.Lock()
        self.probes = 0
        self.pex_runs = 0
        self.solves = 0
        self.solved: dict[tuple, Parasitics] = {}
        self.baseline_drc: dict[str, int] | None = None
        self.pins = _subckt_pins(schematic, model.top)
        self.pool = ThreadPoolExecutor(max_workers=max(1, cfg.jobs))

    # Oracle, parts 1 and 2 ───────────────────────────────────────────────────
    def check(self, state: State) -> tuple[bool, str]:
        signature = tuple(sorted(state.items(), key=lambda item: item[0]))
        with self.lock:
            if signature in self.cache:
                return self.cache[signature]
            self.probes += 1
            work = self.out / "probes" / f"p{self.probes:05d}"
        work.mkdir(parents=True, exist_ok=True)
        gds = work / "layout.gds"
        self.model.write(gds, state)
        verdict = (True, "")
        try:
            lvs = run_lvs(
                gds,
                self.schematic,
                work / "lvs",
                cell_name=self.top,
                tie_bodies=self.cfg.tie_bodies,
            )
            matched, extracted = (
                lvs.matched,
                _subckt_pins(lvs.extracted_netlist, self.top),
            )
        except (RuntimeError, OSError):
            matched, extracted = False, set()
        if not matched:
            verdict = (False, "lvs")
        elif self.pins - extracted:
            verdict = (False, "pins:" + ",".join(sorted(self.pins - extracted)))
        elif self.cfg.drc:
            try:
                counts = drc_counts(run_drc(gds, work / "drc", cell_name=self.top))
            except (RuntimeError, subprocess.TimeoutExpired):
                verdict = (False, "drc:did not run")
            else:
                if self.baseline_drc is None:  # the first probe is the unedited layout
                    self.baseline_drc = dict(counts)
                worse = sorted(
                    c for c, n in counts.items() if n > self.baseline_drc.get(c, 0)
                )
                if worse:
                    verdict = (False, "drc:" + ",".join(worse))
        if not self.cfg.keep_probes:
            shutil.rmtree(work, ignore_errors=True)
        with self.lock:
            self.cache[signature] = verdict
        return verdict

    def check_many(self, states: list[State]) -> list[tuple[bool, str]]:
        return list(self.pool.map(self.check, states))

    # Oracle, part 3 ──────────────────────────────────────────────────────────
    def extract(
        self, state: State, tag: str = "", *, capacitance: bool = True
    ) -> Parasitics:
        """Extract `state`.  Without `capacitance` it is resistance only: seconds, and exact."""
        with self.lock:
            self.pex_runs += 1
            self.solves += capacitance
            work = self.out / (
                f"pex_{tag}" if tag else f"pex_trials/t{self.pex_runs:05d}"
            )
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        gds = work / "layout.gds"
        self.model.write(gds, state)
        arguments = {
            "gds": str(gds),
            "schematic": str(self.schematic),
            "output_dir": str(work / "pex"),
            "cell_name": self.top,
            "tie_bodies": self.cfg.tie_bodies,
            "substrate_net": self.cfg.substrate_net,
            "capacitance": capacitance,
            "capacitance_mesh_area_um2": self.cfg.pex_mesh_area_um2,
            "capacitance_tolerance": self.cfg.pex_tolerance,
            "timeout": self.cfg.pex_timeout,
        }
        completed = subprocess.run(
            [sys.executable, "-c", _PEX_WORKER, json.dumps(arguments)],
            check=False,
            capture_output=True,
            text=True,
            timeout=self.cfg.pex_timeout + 300,
        )
        marker = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("POST_LAYOUT=")
        ]
        if completed.returncode != 0 or not marker:
            raise RuntimeError(
                f"PEX failed in {work}:\n{(completed.stdout + completed.stderr)[-2000:]}"
            )
        found = Parasitics.from_post_layout(marker[-1].split("=", 1)[1])
        if not tag and not self.cfg.keep_probes:
            shutil.rmtree(work, ignore_errors=True)
        return found

    def solve(self, state: State, tag: str = "") -> Parasitics:
        """Full extraction, field solve included; remembered, as nothing here costs more."""
        signature = tuple(sorted(state.items(), key=lambda item: item[0]))
        if signature not in self.solved:
            self.solved[signature] = self.extract(
                state, tag, capacitance=self.cfg.capacitance
            )
        return self.solved[signature]

    def resistance_reason(self, base: Parasitics, state: State) -> str:
        try:
            return resistance_objection(
                self.cfg, base, self.extract(state, capacitance=False)
            )
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            return f"extraction failed: {str(error).strip().splitlines()[-1][:160]}"

    @staticmethod
    def largest_subset(state: State, keys: list[Key], passes: Any) -> list[Key]:
        """Keys that can be deleted together: all of them if that passes, else greedily."""
        if not keys or passes({**state, **dict.fromkeys(keys)}):
            return keys
        trial, accepted = dict(state), []
        for key in keys:
            if passes({**trial, key: None}):
                trial[key] = None
                accepted.append(key)
        return accepted

    # Search ──────────────────────────────────────────────────────────────────
    def shrink_end(self, state: State, key: Key, axis: int, low_end: bool) -> State:
        """Pull one end of a wire in as far as the oracle allows (k-section, parallel).

        Assumes passing is monotone along the wire: once an end position fails,
        every shorter one does.  The longest passing prefix of each round is
        taken, so a violation of that only costs reduction, never correctness.
        """
        box = self.model.box(key, state)
        assert box is not None
        edge = axis if low_end else axis + 2
        toward = 1 if low_end else -1  # direction the edge moves as the wire shrinks
        good = box[edge]
        target = (
            (box[axis + 2] - self.cfg.min_length)
            if low_end
            else (box[axis] + self.cfg.min_length)
        )
        # Keep a landing around each pin label.
        for point in self.model.pinned.get(key, ()):
            stop = (
                math.floor(point[axis]) - self.cfg.pin_landing
                if low_end
                else math.ceil(point[axis]) + self.cfg.pin_landing
            )
            target = min(target, stop) if low_end else max(target, stop)
        if (target - good) * toward < self.cfg.min_gain:
            return state
        tip = _with_edge(
            _with_edge(box, axis, min(good, good + toward * self.cfg.min_gain)),
            axis + 2,
            max(good, good + toward * self.cfg.min_gain),
        )
        # An end that lies under other metal is a junction, not a stub.
        if self.model.covered(key, tip, state):
            return state
        while (target - good) * toward >= 1:
            span = target - good
            steps = sorted(
                {
                    good + round(span * j / self.cfg.jobs)
                    for j in range(1, self.cfg.jobs + 1)
                }
                - {good},
                reverse=toward < 0,
            )
            verdicts = self.check_many(
                [{**state, key: _with_edge(box, edge, at)} for at in steps]
            )
            passed = 0
            while passed < len(steps) and verdicts[passed][0]:
                passed += 1
            if passed:
                good = steps[passed - 1]
            if passed == len(steps):
                break
            target = steps[passed] - toward
        if (good - box[edge]) * toward < self.cfg.min_gain:
            return state
        slab = _with_edge(
            _with_edge(box, axis, min(box[edge], good)), axis + 2, max(box[edge], good)
        )
        if self.model.covered(
            key, slab, state
        ):  # it only gave up overlap: the metal is still there
            return state
        return {**state, key: _with_edge(box, edge, good)}

    def run(self) -> ReductionResult:
        started = time.time()
        cfg, model = self.cfg, self.model
        state: State = {}
        ok, why = self.check(state)
        if not ok:
            raise RuntimeError(
                f"the layout does not pass its own oracle before any edit ({why})"
            )
        base = self.solve(state, "before") if cfg.pex else None
        last_cost = base.cost_fF(cfg.supplies, cfg.miller) if base else 0.0
        kept: list[dict] = []
        refused: set[Key] = set()
        batches: list[list[Key]] = []

        def blame(candidate: State) -> str:
            assert base is not None
            return self.resistance_reason(base, candidate)

        def refuse(key: Key, reason: str) -> None:
            refused.add(key)
            kept.append(
                {
                    "cell": key[0],
                    "layer": model.layer(key),
                    "box": list(model.box(key, state) or ()),
                    "reason": reason,
                }
            )

        for _ in range(cfg.max_passes):
            progressed = False
            for group in DELETE_GROUPS:
                layers = tuple(layer for layer in group if layer in cfg.layers)
                keys = [
                    k
                    for k in model.candidates(layers, cfg.cells, state)
                    if k not in refused
                ]
                verdicts = self.check_many([{**state, k: None} for k in keys])
                passing = [k for k, (good, _) in zip(keys, verdicts) if good]
                passing = self.largest_subset(
                    state, passing, lambda s: self.check(s)[0]
                )
                # Resistance: cheap, so blame is assigned shape by shape, in parallel.
                if (
                    base is not None
                    and passing
                    and self.resistance_reason(
                        base, {**state, **dict.fromkeys(passing)}
                    )
                ):
                    singles = [{**state, k: None} for k in passing]
                    for k, reason in zip(passing, self.pool.map(blame, singles)):
                        if reason:
                            refuse(k, reason)
                    passing = self.largest_subset(
                        state,
                        [k for k in passing if k not in refused],
                        lambda s: not blame(s),
                    )
                if passing:
                    state, progressed = {**state, **dict.fromkeys(passing)}, True
                    batches.append(passing)
            if not progressed:
                break

        # Capacitance: a field solve is minutes on anything but a leaf, and deleting
        # metal all but never raises the cost, so it is checked once for the whole
        # deletion phase.  Only if it objects are the batches replayed to find out why.
        if base is not None and batches:
            solved = self.solve(state)
            if cost_objection(cfg, last_cost, solved):
                state = {}
                for batch in batches:
                    trial = {**state, **dict.fromkeys(batch)}
                    solved = self.solve(trial)
                    if cost_objection(cfg, last_cost, solved):
                        trial = dict(state)
                        for k in batch:
                            solved = self.solve({**trial, k: None})
                            reason = cost_objection(cfg, last_cost, solved)
                            if reason:
                                refuse(k, reason)
                            else:
                                trial[k] = None
                                last_cost = min(
                                    last_cost, solved.cost_fF(cfg.supplies, cfg.miller)
                                )
                    else:
                        last_cost = min(
                            last_cost, solved.cost_fF(cfg.supplies, cfg.miller)
                        )
                    state = trial
                ok, why = self.check(
                    state
                )  # a partial replay can strand what a later batch freed
                if not ok:
                    raise RuntimeError(
                        f"replaying deletions after a cost objection left {why}; "
                        "re-run with a larger cost_tolerance"
                    )
            else:
                last_cost = min(last_cost, solved.cost_fF(cfg.supplies, cfg.miller))

        if cfg.shrink:
            before_shrink = dict(state)
            for key in model.candidates(
                cfg.shrink_layers, cfg.cells, state, pinned=True
            ):
                box = model.box(key, state)
                assert box is not None
                axis = 0 if box[2] - box[0] >= box[3] - box[1] else 1
                if box[axis + 2] - box[axis] < cfg.min_length + cfg.min_gain:
                    continue
                shrunk = state
                for low_end in (True, False):
                    shrunk = self.shrink_end(shrunk, key, axis, low_end)
                if shrunk is not state and base is not None:
                    reason = self.resistance_reason(base, shrunk)
                    if reason:
                        refuse(
                            key, f"not shrunk to {list(shrunk[key] or ())}: {reason}"
                        )
                        continue
                state = shrunk
            if base is not None and state != before_shrink:
                reason = cost_objection(cfg, last_cost, self.solve(state))
                if reason:
                    kept.append(
                        {
                            "cell": "*",
                            "layer": "*",
                            "box": [],
                            "reason": f"all shrinks reverted: {reason}",
                        }
                    )
                    state = before_shrink

        ok, why = self.check(state)
        assert ok, f"reduced layout fails its oracle: {why}"
        after = self.solve(state, "after") if cfg.pex else None
        self.pool.shutdown()

        edits = []
        for key, new_box in sorted(state.items()):
            old = model.polys[key[0]][key[1]][3]
            assert old is not None
            if new_box != old:
                others = {k: v for k, v in state.items() if k != key}
                edits.append(
                    Edit(
                        "delete" if new_box is None else "shrink",
                        key[0],
                        model.layer(key),
                        old,
                        new_box,
                        covered=new_box is None and model.covered(key, old, others),
                    )
                )
        gds = self.out / f"{self.top}_reduced.gds"
        model.write(gds, state)
        report = self.out / "reduction.json"
        result = ReductionResult(
            self.top,
            gds,
            report,
            edits,
            kept,
            base,
            after,
            self.probes,
            time.time() - started,
        )
        report.write_text(
            json.dumps(
                {
                    "top": self.top,
                    "probes": self.probes,
                    "extractions": self.pex_runs,
                    "field_solves": self.solves,
                    "seconds": round(result.seconds, 1),
                    "baseline_drc": self.baseline_drc,
                    "edits": [
                        {
                            "kind": e.kind,
                            "cell": e.cell,
                            "layer": e.layer,
                            "before": list(e.before),
                            "after": list(e.after) if e.after else None,
                            "removed_nm2": e.removed_nm2,
                            "covered": e.covered,
                        }
                        for e in edits
                    ],
                    "kept_for_parasitics": kept,
                    "before": base.summary(cfg.supplies) if base else None,
                    "after": after.summary(cfg.supplies) if after else None,
                },
                indent=1,
            )
            + "\n"
        )
        return result


def reduce_layout(
    gds: str | Path,
    schematic: str | Path,
    output_dir: str | Path,
    *,
    cell_name: str,
    config: ReductionConfig | None = None,
) -> ReductionResult:
    """Delete and shrink what `cell_name` does not need; see the module docstring.

    `schematic` is the LVS reference the layout must keep matching (unit-fin,
    as the `render_*_lvs_schematic` helpers produce).  Writes
    ``<cell>_reduced.gds`` and ``reduction.json`` into `output_dir`.
    """
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = LayoutModel(gds, cell_name)
    return _Reducer(
        model, Path(schematic).expanduser().resolve(), out, config or ReductionConfig()
    ).run()


def main(argv: list[str] | None = None) -> int:
    """``asap7-reduce --gds cell.gds --schematic ref.sp --cell NAME --out DIR``."""
    parser = argparse.ArgumentParser(
        description="Delete and shrink the geometry a cell does not need.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gds", type=Path, required=True)
    parser.add_argument(
        "--schematic", type=Path, required=True, help="Unit-fin LVS reference."
    )
    parser.add_argument("--cell", required=True, help="Top cell to reduce.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", default=",".join(ReductionConfig.layers))
    parser.add_argument(
        "--cells",
        default=None,
        help="Comma-separated cells whose shapes may be edited.",
    )
    parser.add_argument("--no-shrink", action="store_true")
    parser.add_argument("--no-drc", action="store_true")
    parser.add_argument(
        "--no-pex",
        action="store_true",
        help="Skip the extraction floors (LVS and DRC only).",
    )
    parser.add_argument(
        "--no-capacitance",
        action="store_true",
        help="Hold resistance only; never run the field solver (for cells too big to solve).",
    )
    parser.add_argument(
        "--pex-tolerance", type=float, default=ReductionConfig.pex_tolerance
    )
    parser.add_argument(
        "--r-tolerance", type=float, default=ReductionConfig.r_tolerance
    )
    parser.add_argument(
        "--cost-tolerance", type=float, default=ReductionConfig.cost_tolerance
    )
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--keep-probes", action="store_true")
    args = parser.parse_args(argv)
    config = ReductionConfig(
        layers=tuple(args.layers.split(",")),
        cells=tuple(args.cells.split(",")) if args.cells else None,
        shrink=not args.no_shrink,
        drc=not args.no_drc,
        pex=not args.no_pex,
        capacitance=not args.no_capacitance,
        pex_tolerance=args.pex_tolerance,
        r_tolerance=args.r_tolerance,
        cost_tolerance=args.cost_tolerance,
        jobs=args.jobs,
        keep_probes=args.keep_probes,
    )
    result = reduce_layout(
        args.gds, args.schematic, args.out, cell_name=args.cell, config=config
    )
    print(result.summary(config.supplies))
    print(f"wrote {result.gds}\n      {result.report}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
