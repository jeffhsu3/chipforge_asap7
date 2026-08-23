"""chipforge-asap7 — ASAP7 PDK collateral for the chipforge hardware ML compiler.

ASAP7 is chipforge core's primary PDK target, so the compiler hook points stay
in core (see chipforge's CLAUDE.md).  This package holds the ASAP7 *layout*
collateral that core's array compilers keep re-deriving: the GDS layer map and
the fin / gate substrate grids under `chipforge_asap7.layout`, plus the
parametric devices built on them under `chipforge_asap7.devices`.
"""

from . import devices, layout

__all__ = ["devices", "layout"]
__version__ = "0.1.0"
