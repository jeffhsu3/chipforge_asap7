"""The optional chipforge dependency.

chipforge, and migen, which ``chipforge.memory`` imports, are installed by
the ``chipforge`` extra (``uv sync --extra chipforge``). The layout generators
never need them. The ROM macro backends
(`chipforge_asap7.chipforge_backend`) do, and call `require` before importing
it so a missing install names the extra instead of a bare module. The same
arrangement as chipforge_gt2n's ``_chipforge``.
"""

from __future__ import annotations

from importlib.util import find_spec

EXTRA = "chipforge"
MODULES = ("chipforge", "migen")


def available() -> bool:
    """Whether chipforge and migen are importable."""
    return all(find_spec(name) is not None for name in MODULES)


def rom_interface_available() -> bool:
    """Whether chipforge has the ROM macro interface the backends register in.

    Looking it up imports ``chipforge.memory``; a chipforge whose own
    dependencies are missing counts as absent, so importing this package never
    fails for want of it.
    """
    if not available():
        return False
    try:
        return find_spec("chipforge.memory.rom_interface") is not None
    except ImportError:
        return False


def require(feature: str) -> None:
    """Raise a ModuleNotFoundError that names the extra if a module is missing."""
    for name in MODULES:
        if find_spec(name) is None:
            raise ModuleNotFoundError(
                f"{feature} needs {name}; install the {EXTRA!r} extra "
                f"(uv sync --extra {EXTRA})",
                name=name,
            )
