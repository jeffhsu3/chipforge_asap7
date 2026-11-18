"""The NOR ROM macro's pins: M5 columns down to its bottom edge.

A self-timed macro (`RomDecoderSpec(control=True)`) names its interface
on M3, deep inside: the address latches' inputs, the controller's
``CLK`` and ``RESETN``, the sample or register latches' outputs, and the
supply straps.  `RomDecoderSpec(edge_pins=True)` carries each one to the
bottom edge on M5, as the B-ROM's pins are: a V3 off the M3 onto a short
M4 pad on a free track, a V4 at an M5 track, and the M5 down to the
edge, where the pin is labelled.  ``VDD`` lands straight on the M4 stripe
over the array with a V4 beside each VDD strap; ``VSS`` escapes from each
VSS strap like a signal.  The bus pins are named as the B-ROM's are,
``A[i]`` and ``Q[b]``, which is how a gate netlist and DEF name them.

The search reads the drawn M4 and M5 and keeps away from them by rules
stricter than the deck's:

- A new M4 pad or M5 column stays 48 nm tip to tip from anything on its
  own track and 24 nm from anything wider beside it (S.1, S.2).
- On an adjacent track it either stays 48 nm clear of the other wire's
  corners (S.3), or runs alongside for at least 48 nm with each pair of
  tips aligned or at least 48 nm apart (S.4, S.5).
- A V4 keeps V4.S.2's 33 nm from every other V4.

Up to 25 lane rows stack their outputs at one x, more than a slot has M5
tracks, so a column may have to stand far from its pin; and in some rows the
M3 crosses a single free M4 track, which its neighbour's pad must not take.
So a pin with one free track reserves its shortest pad there before anything
routes, each pin's reachable M5 span is read around those reservations, and
`_assign` gives every pin its own track within its span, moving none further
than it must.  Nothing else on M5 runs low in the macro: only the replica's
``READY`` riser, which the search avoids like any other M5.

Pads share a row's few free tracks and columns fill a slot's M5, so on a wide
macro an early pin can still shut a later one out.  That pin then goes first and
the signals route again, up to `MAX_ROUTING_PASSES` times.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..layout.layers import LAYERS, PIN_LAYERS, box, require_gdspy
from ..layout.rules import M4_WIDTH, M4_X_GRID

if TYPE_CHECKING:
    from .rom_decoder import RomDecoderSpec

__all__ = ["PinEscape", "draw_macro_pins", "plan_macro_pins"]

TRACK_PITCH = 48  # M4 and M5: M4.AUX.2 and M5.AUX.2
TRACK_OFFSET = 12  # their tracks' centres: 12 + 48k
M5_WIDTH = 24
HALF = 9  # half a V3's side across the M3
V3_M3_CAP = M4_WIDTH // 2 + 5  # M3 past a V3, along the M3 (V3.M3.EN.1)
V3_M4_CAP = HALF + 11  # M4 past a V3, along the M4 (V3.M4.EN.2)
V4_M4_CAP = M5_WIDTH // 2 + 11  # M4 past a V4, along the M4 (V4.M4.EN.1)
V4_M5_CAP = M5_WIDTH // 2 + 12  # M5 past a V4, along the M5 (V4.M5.EN.2)
TIP_SPACE = 48  # M4 tip to tip on one track, and tips on the next track
SIDE_SPACE = 24  # M4.S.1
ADJACENT = 3 * M4_WIDTH  # wall to wall: under this, wires are on adjacent tracks
V4_SPACE = 33  # V4.S.2
REACH = 128  # M5 tracks either side of its M3 that a pin may reach
MAX_ROUTING_PASSES = 40  # tries at the signals, each with a shut-out pin first

Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class PinEscape:
    """One pin's route to the bottom edge.

    ``via3_x`` is the M3 it leaves, or ``None`` when it lands on drawn M4
    (the VDD stripe); ``pad`` the M4 pad's x span then is ``None`` too.
    """

    port: str
    x: int
    track: int
    via3_x: int | None
    pad: tuple[int, int] | None
    bottom: float

    @property
    def column(self) -> Rect:
        h = M5_WIDTH // 2
        return (self.x - h, self.bottom, self.x + h, self.track + V4_M5_CAP)


def _rects(cell: Any, layer: str) -> list[Rect]:
    spec = (LAYERS[layer]["layer"], LAYERS[layer]["datatype"])
    return [
        (
            float(p[:, 0].min()),
            float(p[:, 1].min()),
            float(p[:, 0].max()),
            float(p[:, 1].max()),
        )
        for p in cell.get_polygons(by_spec=spec)
    ]


def _tracks(lo: float, hi: float) -> range:
    """Track centres ``c`` with ``lo <= c <= hi``."""
    start = math.ceil(lo)
    first = start + (TRACK_OFFSET - start) % TRACK_PITCH
    return range(first, int(hi) + 1, TRACK_PITCH)


def _gap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(b0 - a1, a0 - b1)


def _beside(a0: float, a1: float, b0: float, b1: float) -> bool:
    """Whether two wires on adjacent tracks may stand at these spans: alongside
    by S.5's run with each pair of tips aligned or S.4's stagger apart, or
    clear of each other's corners (S.3)."""
    if _gap(a0, a1, b0, b1) >= 0:
        return _gap(a0, a1, b0, b1) >= TIP_SPACE
    run = min(a1, b1) - max(a0, b0)
    return (
        run >= TIP_SPACE
        and (a0 == b0 or abs(a0 - b0) >= TIP_SPACE)
        and (a1 == b1 or abs(a1 - b1) >= TIP_SPACE)
    )


def _m4_clear(pad: tuple[int, int], track: int, m4: Iterable[Rect]) -> bool:
    x0, x1 = pad
    y0, y1 = track - M4_WIDTH // 2, track + M4_WIDTH // 2
    for r in m4:
        gy = _gap(y0, y1, r[1], r[3])
        if gy >= ADJACENT:
            continue
        if gy < SIDE_SPACE:
            if _gap(x0, x1, r[0], r[2]) < (TIP_SPACE if gy < 0 else SIDE_SPACE):
                return False
        elif not _beside(x0, x1, r[0], r[2]):
            return False
    return True


def _m5_clear(column: Rect, m5: Iterable[Rect]) -> bool:
    for r in m5:
        gx = _gap(column[0], column[2], r[0], r[2])
        if gx >= ADJACENT:
            continue
        if gx < SIDE_SPACE:
            if _gap(column[1], column[3], r[1], r[3]) < (
                TIP_SPACE if gx < 0 else SIDE_SPACE
            ):
                return False
        elif not _beside(column[1], column[3], r[1], r[3]):
            return False
    return True


def _v4_clear(x: int, track: int, v4: Iterable[Rect]) -> bool:
    """V4.S.2: 33 nm to any other V4, corners by distance."""
    h = M5_WIDTH // 2
    for r in v4:
        gx = max(_gap(x - h, x + h, r[0], r[2]), 0)
        gy = max(_gap(track - h, track + h, r[1], r[3]), 0)
        if math.hypot(gx, gy) < V4_SPACE:
            return False
    return True


def _pad(px: int, cx: int) -> tuple[int, int]:
    x0 = min(px - V3_M4_CAP, cx - V4_M4_CAP)
    x1 = max(px + V3_M4_CAP, cx + V4_M4_CAP)
    return x0 - x0 % M4_X_GRID, x1 + -x1 % M4_X_GRID


def _near(m4: Iterable[Rect], track: int) -> list[Rect]:
    """The M4 a wire on `track` has to mind."""
    h = M4_WIDTH // 2
    return [r for r in m4 if _gap(track - h, track + h, r[1], r[3]) < ADJACENT]


def _bar(pad: tuple[int, int], track: int) -> Rect:
    h = M4_WIDTH // 2
    return (pad[0], track - h, pad[1], track + h)


@dataclass
class _Pin:
    """A signal pin as the search sees it: its M3 and where it can reach."""

    port: str
    x: int
    tracks: list[int]
    lo: int = 0
    hi: int = 0
    target: int = 0


def _reach(pin: _Pin, m4: list[Rect]) -> None:
    """Read the span of M5 tracks `pin`'s pads can get to, on any of its tracks."""
    reached = []
    around = _tracks(pin.x - REACH * TRACK_PITCH, pin.x + REACH * TRACK_PITCH)
    for track in pin.tracks:
        near = _near(m4, track)
        reached += [c for c in around if _m4_clear(_pad(pin.x, c), track, near)]
    if not reached:
        raise ValueError(f"no free M4 track for {pin.port}")
    pin.lo, pin.hi = min(reached), max(reached)


def _assign(pins: list[_Pin]) -> None:
    """Give each pin its own M5 track in its span, by the smallest displacement
    that fits them all: earliest-deadline-first matching, which finds an
    assignment of tracks to intervals whenever there is one."""
    for reach in range(0, (REACH + 1) * TRACK_PITCH, TRACK_PITCH):
        spans = sorted(
            (
                max(p.lo, _first_track(p.x - reach)),
                min(p.hi, p.x + reach + TRACK_PITCH),
                i,
            )
            for i, p in enumerate(pins)
        )
        if _match(spans, pins):
            return
    raise ValueError("the pins' M5 spans hold fewer tracks than there are pins")


def _first_track(x: int) -> int:
    return x + (TRACK_OFFSET - x) % TRACK_PITCH


def _match(spans: list[tuple[int, int, int]], pins: list[_Pin]) -> bool:
    waiting: list[tuple[int, int]] = []
    next_span, placed = 0, {}
    track = spans[0][0]
    while next_span < len(spans) or waiting:
        if not waiting and spans[next_span][0] > track:
            track = spans[next_span][0]
        while next_span < len(spans) and spans[next_span][0] <= track:
            _, hi, i = spans[next_span]
            heapq.heappush(waiting, (hi, i))
            next_span += 1
        hi, i = heapq.heappop(waiting)
        if track > hi:
            return False
        placed[i] = track
        track += TRACK_PITCH
    for i, target in placed.items():
        pins[i].target = target
    return True


def _escape(
    port: str,
    px: int,
    target: int,
    tracks: Iterable[int],
    bottom: float,
    m4: list[Rect],
    m5: list[Rect],
    v4: list[Rect],
) -> PinEscape | None:
    """A pad off the M3 at `px` on one of `tracks`, to the free M5 track
    nearest `target`."""
    near = {track: _near(m4, track) for track in tracks}
    span = _tracks(target - REACH * TRACK_PITCH, target + REACH * TRACK_PITCH)
    for cx in sorted(span, key=lambda c: (abs(c - target), c)):
        for track, around in near.items():
            pad = _pad(px, cx)
            route = PinEscape(port, cx, track, px, pad, bottom)
            if (
                _m4_clear(pad, track, around)
                and _m5_clear(route.column, m5)
                and _v4_clear(cx, track, v4)
            ):
                return route
    return None


def _land(
    port: str, near: int, stripe: Rect, bottom: float, m5: list[Rect], v4: list[Rect]
) -> PinEscape | None:
    """A V4 on the drawn M4 `stripe`, at the free M5 track nearest `near`."""
    track = round((stripe[1] + stripe[3]) / 2)
    lo, hi = stripe[0] + V4_M4_CAP, stripe[2] - V4_M4_CAP
    span = _tracks(
        max(lo, near - REACH * TRACK_PITCH), min(hi, near + REACH * TRACK_PITCH)
    )
    for cx in sorted(span, key=lambda c: (abs(c - near), c)):
        route = PinEscape(port, cx, track, None, None, bottom)
        if _m5_clear(route.column, m5) and _v4_clear(cx, track, v4):
            return route
    return None


def _route_signals(
    order: list[_Pin],
    reserved: dict[str, Rect],
    bottom: float,
    m4: list[Rect],
    m5: list[Rect],
    v4: list[Rect],
) -> tuple[list[PinEscape], list[Rect], list[Rect], list[Rect]]:
    """Route each pin in turn; a pin that finds no way out goes first next time.

    Pads share a row's few free tracks and columns fill a slot's M5, so an
    early pin can shut a later one out.  Returns the routes and the M4, M5 and
    V4 with them added.
    """
    order = list(order)
    for _ in range(MAX_ROUTING_PASSES):
        passed = _route_in_order(order, reserved, bottom, m4, m5, v4)
        if isinstance(passed, _Pin):
            order.remove(passed)
            order.insert(0, passed)
            continue
        return passed
    raise ValueError(
        f"no free M4 track and M5 column for {passed.port} "
        f"({len(passed.tracks)} free tracks) in {MAX_ROUTING_PASSES} passes"
    )


def _route_in_order(
    order: list[_Pin],
    reserved: dict[str, Rect],
    bottom: float,
    m4: list[Rect],
    m5: list[Rect],
    v4: list[Rect],
) -> tuple[list[PinEscape], list[Rect], list[Rect], list[Rect]] | _Pin:
    """Route `order` on copies of the drawn metal, or return the pin that failed."""
    m4, m5, v4, waiting = [*m4], [*m5], [*v4], dict(reserved)
    routes = []
    for pin in order:
        others = [*m4, *(r for q, r in waiting.items() if q != pin.port)]
        route = _escape(pin.port, pin.x, pin.target, pin.tracks, bottom, others, m5, v4)
        if route is None:
            return pin
        waiting.pop(pin.port, None)
        routes.append(route)
        _add_route(route, m4, m5, v4)
    return routes, m4, m5, v4


def _add_route(
    route: PinEscape, m4: list[Rect], m5: list[Rect], v4: list[Rect]
) -> None:
    m5.append(route.column)
    h = M5_WIDTH // 2
    v4.append((route.x - h, route.track - h, route.x + h, route.track + h))
    if route.pad is not None:
        m4.append(_bar(route.pad, route.track))


def _m3_at(m3: list[Rect], x: float, y: float) -> Rect | None:
    """The M3 column centred on `x` that holds ``(x, y)``."""
    under = [
        r for r in m3 if r[0] <= x - HALF and x + HALF <= r[2] and r[1] <= y <= r[3]
    ]
    return max(under, key=lambda r: r[3] - r[1], default=None)


def _crossing(shape: Rect) -> range:
    """The M4 tracks a V3 can land on along the M3 `shape`."""
    return _tracks(shape[1] + V3_M3_CAP, shape[3] - V3_M3_CAP)


def plan_macro_pins(spec: RomDecoderSpec, cell: Any) -> list[PinEscape]:
    """Every pin's escape from `cell`, the macro `build_rom` drew for `spec`.

    Signals go first, those with the fewest free M4 tracks ahead of the rest,
    then one ``VSS`` per VSS strap and one ``VDD`` beside each VDD strap.  A
    signal with no way out raises; a supply strap with none is skipped, but
    each supply needs one.
    """
    m3, m4, m5, v4 = (_rects(cell, layer) for layer in ("M3", "M4", "M5", "V4"))
    bottom = float(cell.get_bounding_box()[0][1])
    routes: list[PinEscape] = []

    def take(route: PinEscape) -> None:
        routes.append(route)
        _add_route(route, m4, m5, v4)

    pins = []
    for port, (px, py) in spec.interface_points.items():
        shape = _m3_at(m3, px, py)
        if shape is None:
            raise ValueError(f"no M3 under {port} at {(px, py)}")
        x = int(px)
        tracks = [t for t in _crossing(shape) if _m4_clear(_pad(x, x), t, _near(m4, t))]
        if not tracks:
            raise ValueError(f"no free M4 track crosses {port}'s M3")
        pins.append(_Pin(port, x, tracks))
    reserved = {
        p.port: _bar(_pad(p.x, p.x), p.tracks[0]) for p in pins if len(p.tracks) == 1
    }
    for pin in pins:
        _reach(pin, [*m4, *(r for q, r in reserved.items() if q != pin.port)])
    _assign(pins)
    order = sorted(pins, key=lambda p: (len(p.tracks), abs(p.target - p.x), p.port))
    routed, m4, m5, v4 = _route_signals(order, reserved, bottom, m4, m5, v4)
    routes += routed

    for x, y in spec.vss_strap_points:
        shape = _m3_at(m3, x, y)
        route = shape and _escape("VSS", x, x, _crossing(shape), bottom, m4, m5, v4)
        if route:
            take(route)
    y = spec.vdd_stripe_y
    stripe = next(r for r in m4 if r[1] < y < r[3] and r[2] - r[0] > M4_WIDTH * 100)
    for x in spec.vdd_strap_xs:
        route = _land("VDD", x, stripe, bottom, m5, v4)
        if route:
            take(route)
    for net in ("VSS", "VDD"):
        if not any(r.port == net for r in routes):
            raise ValueError(f"no free M5 column for {net}")
    return routes


def draw_macro_pins(spec: RomDecoderSpec, cell: Any) -> list[PinEscape]:
    """Draw `plan_macro_pins` into `cell` and label each column at the edge."""
    gdspy = require_gdspy()
    routes = plan_macro_pins(spec, cell)
    h4, h5 = M4_WIDTH // 2, M5_WIDTH // 2
    layer, texttype = PIN_LAYERS["M5"]
    for r in routes:
        if r.via3_x is not None:
            assert r.pad is not None
            box(
                cell,
                "V3",
                r.via3_x - HALF,
                r.track - h4,
                r.via3_x + HALF,
                r.track + h4,
            )
            box(cell, "M4", r.pad[0], r.track - h4, r.pad[1], r.track + h4)
        box(cell, "V4", r.x - h5, r.track - h5, r.x + h5, r.track + h5)
        box(cell, "M5", *r.column)
        cell.add(
            gdspy.Label(r.port, (r.x, r.bottom + h5), layer=layer, texttype=texttype)
        )
    return routes
