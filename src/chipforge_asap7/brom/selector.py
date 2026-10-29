"""Periodic two-gate BROM selector core with M2/M4 candidate programming.

One live selector and one grounded isolation gate occupy each 108 nm
period. Diffusion is continuous across columns; every source is grounded.
Blocks have independent gate cuts and M3 stems. Taps and rail closure live
outside the periodic core and are not charged to its area.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Any

from ..devices.finfet import VT_LAYERS
from ..devices.rowcell import label, square
from ..layout.layers import box, require_gdspy


@dataclass(frozen=True)
class SelectorPlan:
    fins: int
    width: int
    height: int
    tracks: tuple[tuple[str, float], ...]
    vt: str = "rvt"

    def __post_init__(self) -> None:
        if type(self.fins) is not int or not 1 <= self.fins <= 3:
            raise ValueError("selector witness supports 1..3 fins")
        if (
            self.width != 108
            or type(self.height) is not int
            or self.height < 432
            or self.height % 432
        ):
            raise ValueError(
                "this witness uses 108 nm columns and a 432 nm row lattice"
            )
        if len(self.tracks) != 16 or any(m not in ("M2", "M4") for m, _ in self.tracks):
            raise ValueError("sixteen M2/M4 candidate tracks are required")
        if any(
            type(y) not in (int, float) or not isfinite(y) or not 0 < y < self.height
            for _, y in self.tracks
        ):
            raise ValueError("candidate positions must be finite and inside the block")
        object.__setattr__(self, "tracks", tuple(tuple(t) for t in self.tracks))
        if self.vt not in VT_LAYERS:
            raise ValueError("invalid VT")

    @property
    def gate_y(self) -> int:
        return 27 * (self.fins + 3)

    @property
    def drain_y(self) -> float:
        return 27 + 13.5 * self.fins

    @property
    def stem_span(self) -> tuple[float, float]:
        ys = [self.gate_y, *(y for _, y in self.tracks)]
        return min(ys) - 17, max(ys) + 17


def build_selector_array(
    plan: SelectorPlan, mapping: tuple[tuple[int, ...], ...], *, lib: Any = None
) -> Any:
    """Draw a finite closed array; all sixteen choices use one programming via."""
    if not mapping or not mapping[0] or any(len(r) != len(mapping[0]) for r in mapping):
        raise ValueError("mapping must be a nonempty rectangle")
    if any(type(k) is not int or not 0 <= k < 16 for row in mapping for k in row):
        raise ValueError("candidate indices must be integers in 0..15")
    if lib is not None and lib.unit != 1e-9:
        raise ValueError("GDS coordinates are nanometres")
    gdspy = require_gdspy()
    digest = sha256(repr((plan, mapping)).encode()).hexdigest()[:12]
    top = gdspy.Cell(f"brom_selector_{digest}", exclude_from_current=True)
    if lib is not None:
        lib.add(top)
    blocks, columns = len(mapping), len(mapping[0])
    w, h = columns * plan.width, blocks * plan.height
    if h + 81 > 30_000:
        raise ValueError("array exceeds the single tap row latch-up reach")
    # The outer source columns close continuous diffusion at both ends.
    box(top, "NSELECT", -54, -27, w + 54, h)
    box(top, "PSELECT", -54, h, w + 54, h + 108)
    if vt := VT_LAYERS[plan.vt]:
        box(top, vt, -54, -27, w + 54, h + 108)
    for y in range(10, h + 81, 27):
        box(top, "FIN", -54, y, w + 54, y + 7)
    for x in range(-27, w + 28, 54):
        box(top, "GATE", x - 10, -5, x + 10, h + 86)
    for y in [*(b * plan.height for b in range(blocks + 1)), h + 81]:
        box(top, "GATE_CUT", -54, y - 22, w + 54, y + 22)
    for col in range(columns + 1):
        x = col * plan.width
        box(top, "M1", x - 9, -14, x + 9, h + 95)
        # Substrate tap, above the core, at each source column.
        box(top, "ACTIVE", x - 8, h + 27, x + 8, h + 81)
        for metal in ("SDT", "LISD"):
            box(top, metal, x - 12, h + 27, x + 12, h + 81)
        square(top, "V0", x, h + 54)
        square(top, "V1", x, h + 81)
    box(top, "M2", -24, h + 72, w + 24, h + 90)
    label(top, "VSS", "M2", (0, h + 81))
    for col in range(columns):
        x = col * plan.width + 81
        box(top, "M3", x - 9, 0, x + 9, h)
        label(top, f"BL<{col}>", "M3", (x, 12))
    for b, row in enumerate(mapping):
        oy = b * plan.height
        alo, ahi = oy + 27, oy + 27 * (plan.fins + 1)
        box(top, "ACTIVE", -8, alo, w + 8, ahi)
        for x in range(0, w + 1, 54):
            for metal in ("SDT", "LISD"):
                box(top, metal, x - 12, alo, x + 12, ahi)
            square(top, "V0", x, oy + plan.drain_y)
        for k, (metal, y) in enumerate(plan.tracks):
            half = 9 if metal == "M2" else 12
            box(top, metal, -48, oy + y - half, w + 48, oy + y + half)
            label(top, f"C<{b},{k}>", metal, (-24, oy + y))
        for col, k in enumerate(row):
            ox = col * plan.width
            dy, gy = oy + plan.drain_y, oy + plan.gate_y
            box(top, "M1", ox + 45, min(alo, dy - 14), ox + 63, max(ahi, dy + 14))
            square(top, "V1", ox + 54, dy)
            box(top, "M2", ox + 37, dy - 9, ox + 98, dy + 9)
            square(top, "V2", ox + 81, dy)
            # Live gate access ends 17 nm short of the isolation poly.
            box(top, "LIG", ox + 16, gy - 11, ox + 54, gy + 11)
            square(top, "V0", ox + 45, gy)
            box(top, "M1", ox + 36, gy - 14, ox + 54, gy + 14)
            square(top, "V1", ox + 45, gy)
            box(top, "M2", ox + 10, gy - 9, ox + 62, gy + 9)
            square(top, "V2", ox + 27, gy)
            lo, hi = plan.stem_span
            box(top, "M3", ox + 18, oy + lo, ox + 36, oy + hi)
            # Grounded isolation gate reaches the next source's M1 column.
            iy = gy + 54
            box(top, "LIG", ox + 70, iy - 11, ox + 108, iy + 11)
            square(top, "V0", ox + 99, iy)
            box(top, "M1", ox + 85, iy - 9, ox + 117, iy + 9)
            metal, y = plan.tracks[k]
            if metal == "M2":
                square(top, "V2", ox + 27, oy + y)
            else:
                box(top, "V3", ox + 18, oy + y - 12, ox + 36, oy + y + 12)
    box(top, "BOUNDARY", -54, -27, w + 54, h + 108)
    return top


def selector_lvs_reference(
    plan: SelectorPlan, mapping: tuple[tuple[int, ...], ...], name: str
) -> str:
    pins = [
        *(f"C<{b},{k}>" for b in range(len(mapping)) for k in range(16)),
        *(f"BL<{c}>" for c in range(len(mapping[0]))),
        "VSS",
    ]
    lines = [f".subckt {name} {' '.join(pins)}"]
    for b, row in enumerate(mapping):
        for col, k in enumerate(row):
            for fin in range(plan.fins):
                for tag, gate in (("sel", f"C<{b},{k}>"), ("iso", "VSS")):
                    lines.append(
                        f"M{tag}_{b}_{col}_{fin} BL<{col}> {gate} VSS VSS nmos_{plan.vt} L=20n W=7n"
                    )
    return "\n".join([*lines, f".ends {name}", ".end", ""])
