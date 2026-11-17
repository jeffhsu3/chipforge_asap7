"""ChipForge's view of the ASAP7 ROMs: the ``asap7_brom`` and ``asap7_nor_rom``
``rom_macro_backend`` names, their port contracts, collateral and bank helpers.

The same arrangement as chipforge_gt2n's ``rom/chipforge_backend.py``: the PDK
library owns its backends and registers them in chipforge's
``chipforge.memory.rom_interface`` registry. Importing ``chipforge_asap7``
registers both with their default (256-row) bank; call
`register_asap7_brom_backend` / `register_asap7_nor_rom_backend` for another.
Core's ``chipforge.memory.asap7_brom`` and ``asap7_nor_rom`` re-export
`brom` and `nor_rom`. Needs the ``chipforge`` extra.
"""

from .._chipforge import require

require("chipforge_asap7.chipforge_backend")

from .brom import ASAP7_BROM_BACKEND, register_asap7_brom_backend  # noqa: E402
from .nor_rom import ASAP7_NOR_ROM_BACKEND, register_asap7_nor_rom_backend  # noqa: E402
from .paths import rom_macros_dir  # noqa: E402

__all__ = [
    "ASAP7_BROM_BACKEND",
    "ASAP7_NOR_ROM_BACKEND",
    "register_asap7_brom_backend",
    "register_asap7_nor_rom_backend",
    "rom_macros_dir",
]
