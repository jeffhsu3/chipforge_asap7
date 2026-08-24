"""The standard-cell row stack every abutting ASAP7 cell in this package sits on.

A logic cell, a body tap, a filler and a decoupling capacitor all have to agree
on the same thing: where the transistor bands are, how tall each one is, which
polarity it carries, and which power rail it faces.  Get that wrong by a fin
pitch and the cells stop abutting -- diffusion misses, implant edges cross, and
the row's rails do not line up.

`RowStack` owns exactly that geometry and nothing else.  Rows alternate
orientation the way standard-cell rows do: row 0 puts its nFET at the bottom
and every later row flips, which is what makes each rail lie between two bands
of one polarity so it can carry one supply::

    y=1890 ────────── vss rail ──────────
            13-fin nFET   ┐ row 1, flipped
    y=1485                │
            13-fin pFET   ┘
    y=1080 ────────── vdd rail ──────────
            18-fin pFET   ┐ row 0
    y= 540                │
            18-fin nFET   ┘
    y=   0 ────────── vss rail ──────────

`chipforge_asap7.devices.finfet` still owns the device arithmetic underneath:
each band is a `FinFETSpec`, so band heights, fin positions and ACTIVE spans
come from the same code a standalone device uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..layout.grid import FIN_PITCH
from .finfet import ACTIVE_ENC, CONTACT_SIZE, FinFETSpec

__all__ = ["RowBand", "RowStack"]


@dataclass(frozen=True)
class RowBand:
    """One transistor band of a stack, and where it sits in the cell."""

    spec: FinFETSpec
    row: int
    y0: int
    rail_y: int

    @property
    def flavor(self) -> str:
        return self.spec.flavor

    @property
    def fins(self) -> int:
        return self.spec.fins

    @property
    def height(self) -> int:
        return self.spec.height_per_row

    @property
    def y1(self) -> int:
        """Top Y of the band's select strip."""
        return self.y0 + self.height

    @property
    def active_span(self) -> tuple[float, float]:
        """Bottom/top Y of this band's ACTIVE, in cell coordinates."""
        lo, hi = self.spec.active_span()
        return lo + self.y0, hi + self.y0

    @property
    def rail_below(self) -> bool:
        """True when the band's own power rail runs along its bottom edge."""
        return self.rail_y <= self.y0

    @property
    def contact_y(self) -> float:
        """Center Y of the band's source/drain via row.

        Every column contacts on the ACTIVE edge facing the band's own power
        rail, which is what lets a source reach the rail with a stub instead
        of a conductor running the length of the channel.
        """
        lo, hi = self.active_span
        return lo + CONTACT_SIZE / 2 if self.rail_below else hi - CONTACT_SIZE / 2

    @property
    def rail_net(self) -> str:
        return "VSS" if self.flavor == "n" else "VDD"

    @property
    def implant(self) -> str:
        """Implant a *device* in this band needs."""
        return "NSELECT" if self.flavor == "n" else "PSELECT"

    @property
    def tap_implant(self) -> str:
        """Implant a *body tap* in this band needs -- the opposite one.

        A tap ties the well or substrate this band's devices sit in, so it is
        doped the other way round: an n-band's substrate tie is p-type, and a
        p-band's well tie is n-type.  This one inversion is the whole
        difference between the released TAPCELL and FILLER implant layers.
        """
        return "PSELECT" if self.flavor == "n" else "NSELECT"

    @property
    def in_nwell(self) -> bool:
        return self.flavor == "p"


@dataclass(frozen=True)
class RowStack:
    """A stack of standard-cell rows, alternately flipped.

    Args:
        rows: ``(n_fins, p_fins)`` per row, bottom to top.  More rows is how a
            cell gets taller than one legal fin height without giving up the
            standard-cell rail structure.
        vt: threshold-voltage model shared by every band.
    """

    rows: tuple[tuple[int, int], ...] = ((4, 6),)
    vt: Literal["rvt", "lvt", "slvt", "sram"] = "rvt"

    def __post_init__(self) -> None:
        try:
            rows = tuple((int(n), int(p)) for n, p in self.rows)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rows must be a sequence of (n_fins, p_fins) integer pairs, "
                f"got {self.rows!r}"
            ) from exc
        if not rows:
            raise ValueError("a row stack needs at least one row")
        object.__setattr__(self, "rows", rows)
        # Build the stack once now.  FinFETSpec owns the legal ASAP7 device
        # grid, so this is what rejects an illegal fin count -- with the same
        # message a standalone device would give -- at construction rather
        # than at draw time.  (`assert self.bands` would silently pass: bands
        # is a method, and a bound method is always truthy.)
        self.bands()

    @property
    def code(self) -> str:
        """Per-row fin code, e.g. ``"18n18p_13n13p"``.

        Two stacks can share a fin total and still be different geometry, so
        the code carries the rows rather than the sum.
        """
        return "_".join(f"{n}n{p}p" for n, p in self.rows)

    def band_spec(self, flavor: Literal["n", "p"], fins: int, **kwargs) -> FinFETSpec:
        """The `FinFETSpec` for one band, on this stack's threshold flavor."""
        return FinFETSpec(flavor=flavor, fins=fins, vt=self.vt, **kwargs)

    def bands(self, fingers: int = 1) -> tuple[RowBand, ...]:
        """Every band, bottom to top, sized for `fingers` gates per band."""
        stack: list[RowBand] = []
        y = 0
        for row, (n_fins, p_fins) in enumerate(self.rows):
            pair = (
                (("p", p_fins), ("n", n_fins))
                if row % 2
                else (("n", n_fins), ("p", p_fins))
            )
            row_y0 = y
            row_height = sum(
                self.band_spec(flavor, fins).default_height_per_row
                for flavor, fins in pair
            )
            for position, (flavor, fins) in enumerate(pair):
                spec = self.band_spec(flavor, fins, fingers=fingers)
                stack.append(
                    RowBand(
                        spec=spec,
                        row=row,
                        y0=y,
                        rail_y=row_y0 if position == 0 else row_y0 + row_height,
                    )
                )
                y += spec.height_per_row
        return tuple(stack)

    def row_bands(self, row: int, fingers: int = 1) -> tuple[RowBand, ...]:
        """The two bands of `row`, bottom to top."""
        return tuple(band for band in self.bands(fingers) if band.row == row)

    @property
    def row_ys(self) -> tuple[int, ...]:
        """Y of every row boundary, which is also where a rail runs."""
        bands = self.bands()
        edges = [0]
        for row in range(len(self.rows)):
            edges.append(edges[-1] + sum(b.height for b in bands if b.row == row))
        return tuple(edges)

    @property
    def rails(self) -> tuple[tuple[int, str], ...]:
        """``(y, net)`` of every power rail, bottom to top.

        Flipping alternate rows is what makes this well defined: each interior
        rail lies between two bands of one polarity, so it carries one supply.
        """
        bands = self.bands()
        return tuple(
            (y, (bands[0] if index == 0 else bands[2 * index - 1]).rail_net)
            for index, y in enumerate(self.row_ys)
        )

    def seam_y(self, row: int) -> int:
        """Y of the n/p boundary inside `row`.

        The only Y in a row with no source/drain bar crossing it, which is why
        both the gate contact and any horizontal local-interconnect strap go
        there.
        """
        return self.row_bands(row)[1].y0

    @property
    def seam_ys(self) -> tuple[int, ...]:
        return tuple(self.seam_y(row) for row in range(len(self.rows)))

    @property
    def height(self) -> int:
        return self.row_ys[-1]

    @property
    def fin_grid_ys(self) -> list[int]:
        """Bottom Y of the full manufacturing fin grid across the stack."""
        return [ACTIVE_ENC + i * FIN_PITCH for i in range(self.height // FIN_PITCH)]
