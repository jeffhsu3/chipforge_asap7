"""chipforge-asap7 — ASAP7 PDK collateral for the chipforge hardware ML compiler.

This package holds the ASAP7 *layout* collateral that core's array compilers
keep re-deriving: the GDS layer map and the fin / gate substrate grids under
`chipforge_asap7.layout`, plus the parametric devices built on them under
`chipforge_asap7.devices`, and the ROM and SRAM compilers (``asap7`` command).

Like chipforge_gt2n, it owns its ROM macro backends. With chipforge installed
(the ``chipforge`` extra), importing this package registers
``rom_macro_backend="asap7_brom"`` and ``"asap7_nor_rom"``
(`chipforge_asap7.chipforge_backend`). Without chipforge it imports as before.
"""

from . import devices, layout
from ._chipforge import rom_interface_available

if rom_interface_available():
    from .chipforge_backend import (
        register_asap7_brom_backend,
        register_asap7_nor_rom_backend,
    )

    register_asap7_brom_backend()
    register_asap7_nor_rom_backend()

__all__ = ["devices", "layout"]
__version__ = "0.1.0"
