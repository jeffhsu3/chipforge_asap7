"""ASAP7 GDS layer map used by the chipforge array compilers.

Superset of the maps duplicated in chipforge's `scripts/brom_compiler.py`,
`scripts/rom_compiler.py` and `scripts/roma_brom_compiler.py` — the entries are
byte-identical, so a compiler can swap its local `LAYERS` dict for this one
without moving any geometry.

Every value is a ``{"layer": int, "datatype": int}`` mapping so it can be
splatted straight into a gdspy constructor::

    cell.add(gdspy.Rectangle((x0, y0), (x1, y1), **LAYERS["FIN"]))

gdspy is imported lazily, and only by `box()`: importing this package, reading
the layer map and doing grid arithmetic all work without a GDS library
installed.  Install the ``gds`` extra to draw.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from types import ModuleType

__all__ = ["LAYERS", "PIN_LAYERS", "box", "have_gdspy", "layer", "require_gdspy"]

LAYERS: dict[str, dict[str, int]] = {
    "NWELL": {"layer": 1, "datatype": 0},
    "FIN": {"layer": 2, "datatype": 0},
    "GATE": {"layer": 7, "datatype": 0},
    "GATE_CUT": {"layer": 10, "datatype": 0},
    "ACTIVE": {"layer": 11, "datatype": 0},
    "NSELECT": {"layer": 12, "datatype": 0},
    "PSELECT": {"layer": 13, "datatype": 0},
    "SDT": {"layer": 88, "datatype": 0},
    "LIG": {"layer": 16, "datatype": 0},
    # Compatibility alias used by the original chipforge scripts.  In the
    # canonical ASAP7 map layer 16 is LIG, not a cut between LI and M1.
    "CA": {"layer": 16, "datatype": 0},
    "LISD": {"layer": 17, "datatype": 0},
    "V0": {"layer": 18, "datatype": 0},
    "M1": {"layer": 19, "datatype": 0},
    "M1_PIN": {"layer": 19, "datatype": 251},
    "M2": {"layer": 20, "datatype": 0},
    "M2_PIN": {"layer": 20, "datatype": 251},
    "V1": {"layer": 21, "datatype": 0},
    "V2": {"layer": 25, "datatype": 0},
    "M3": {"layer": 30, "datatype": 0},
    "M3_PIN": {"layer": 30, "datatype": 251},
    "V3": {"layer": 35, "datatype": 0},
    "M4": {"layer": 40, "datatype": 0},
    "M4_PIN": {"layer": 40, "datatype": 251},
    "V4": {"layer": 45, "datatype": 0},
    "M5": {"layer": 50, "datatype": 0},
    "M5_PIN": {"layer": 50, "datatype": 251},
    "V5": {"layer": 55, "datatype": 0},
    "M6": {"layer": 60, "datatype": 0},
    "M6_PIN": {"layer": 60, "datatype": 251},
    "V6": {"layer": 65, "datatype": 0},
    "M7": {"layer": 70, "datatype": 0},
    "M7_PIN": {"layer": 70, "datatype": 251},
    "V7": {"layer": 75, "datatype": 0},
    "M8": {"layer": 80, "datatype": 0},
    "M8_PIN": {"layer": 80, "datatype": 251},
    "V8": {"layer": 85, "datatype": 0},
    "M9": {"layer": 90, "datatype": 0},
    "M9_PIN": {"layer": 90, "datatype": 251},
    "V9": {"layer": 95, "datatype": 0},
    "PAD": {"layer": 96, "datatype": 0},
    "SLVT": {"layer": 97, "datatype": 0},
    "LVT": {"layer": 98, "datatype": 0},
    "SRAMDRC": {"layer": 99, "datatype": 0},
    "BOUNDARY": {"layer": 100, "datatype": 0},
    "SRAMVT": {"layer": 110, "datatype": 0},
    "DIEAREA": {"layer": 235, "datatype": 5},
}

#: Metal -> the ``(layer, texttype)`` its pin labels go on.
PIN_LAYERS: dict[str, tuple[int, int]] = {
    metal: (LAYERS[f"{metal}_PIN"]["layer"], LAYERS[f"{metal}_PIN"]["datatype"])
    for metal in ("M1", "M2", "M3", "M4")
}

_gdspy_module: ModuleType | None = None


def require_gdspy() -> ModuleType:
    """Return the gdspy module, importing it on first use.

    Callers that build their own cells (see `chipforge_asap7.devices`) go
    through this rather than importing gdspy directly, so a missing optional
    dependency reports itself the same way everywhere.

    Raises:
        ImportError: with install instructions, when gdspy is not available.
    """
    global _gdspy_module
    if _gdspy_module is None:
        try:
            import gdspy
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "drawing helpers require gdspy; install it with "
                "'pip install chipforge-asap7[gds]' (the layer map, grid "
                "constants and pure-geometry helpers work without it)"
            ) from exc
        _gdspy_module = gdspy
    return _gdspy_module


_gdspy = require_gdspy  # internal alias


def have_gdspy() -> bool:
    """True when the drawing helpers are usable in this environment."""
    try:
        require_gdspy()
    except ImportError:
        return False
    return True


def layer(name: str) -> dict[str, int]:
    """Return the ``{"layer", "datatype"}`` mapping for `name`."""
    try:
        return LAYERS[name]
    except KeyError:
        raise KeyError(
            f"unknown ASAP7 layer {name!r}; known: {sorted(LAYERS)}"
        ) from None


def box(cell: Any, name: str, x0: float, y0: float, x1: float, y1: float) -> None:
    """Add an axis-aligned rectangle on layer `name` to `cell`.

    Corners may be given in any order; they are normalised so callers can pass
    e.g. a rail's two endpoints directly.
    """
    # Resolve gdspy and the layer before touching `cell`, so a missing
    # dependency or a bad layer name always reports itself as such.
    rectangle = _gdspy().Rectangle(
        (min(x0, x1), min(y0, y1)),
        (max(x0, x1), max(y0, y1)),
        **layer(name),
    )
    cell.add(rectangle)
