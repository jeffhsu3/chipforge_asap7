"""ASAP7 NOR ROM (``chipforge_asap7.devices.rom_*``) as the weight ROM of one L-Unit.

``rom_macro_backend="asap7_nor_rom"``: registered when ``chipforge_asap7`` is
imported with chipforge installed, as chipforge_gt2n registers ``gt2n_nor``.
Core's ``chipforge.memory.asap7_nor_rom`` re-exports this module. The built
banks stay in core's ``macros/asap7_nor_rom`` (`rom_macros_dir`).

This is the same arrangement as `chipforge_asap7.chipforge_backend.brom`, with a different
compiler: one L-Unit macro serves every unit, hardened against a bank of
stable-named abstracts (``scripts/asap7_nor_rom_bank.py``), and each unit's
programmed macros replace those cells after P&R (``scripts/program_brom_gds_swap.py``,
which reads the bank's ``compiler``). Programming only adds V1s at the array's bit
sites. Unlike the B-ROM's V2/V3, V1 is also the periphery's via, so the compiler's
``frozen_geometry_sha256`` is what catches a periphery V1 changing.

The pins match the B-ROM's, but the read is the edge flow-through read of
docs/rom_macro_contract.md (``asap7-rom --flow-through``): CLK rising latches A and
starts a read that times itself from a replica, Q takes the word before the next
rise and holds it until the read after, and the bitlines precharge in between,
whatever CLK does. Its limit is the period, not either phase: at typical, bare,
1024x148 reads at 1078 ps, against 1496 ps for the two-phase read at 50% duty.
RESETN must be released before the first read. The compiler's decoder needs a
power-of-two word count, so depth comes from `nor_rom_depth`, not the 1-1.5-2
catalog.

The Liberty is whatever the compiler wrote: estimated, or simulated if the bank was
given a timing report (``asap7-rom --characterize``). Neither is sign-off.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

from chipforge.memory.rom_interface import (
    ROMCollateral,
    ROMMacroInterface,
    register_rom_macro_interface,
)

from .brom import CORNERS, bank_dir
from .paths import rom_macros_dir

ASAP7_NOR_ROM_BACKEND = "asap7_nor_rom"
ASAP7_NOR_ROM_ROOT = rom_macros_dir() / "asap7_nor_rom"
#: The depths the compiler builds a clocked macro at.
NOR_ROM_MIN_WORDS = 16
NOR_ROM_MAX_WORDS = 1024
#: Widest macro a word is split into. The compiler builds every clocked width
#: from 2 to 200 bits at those depths.
NOR_ROM_MAX_SLICE_BITS = 200


def nor_rom_depth(depth: int) -> int:
    """Words in the NOR ROM bank holding ``depth`` words: the next power of two."""
    words = max(int(depth), NOR_ROM_MIN_WORDS)
    if words > NOR_ROM_MAX_WORDS:
        raise ValueError(
            f"{depth} words do not fit one NOR ROM macro of at most "
            f"{NOR_ROM_MAX_WORDS} words"
        )
    return 1 << (words - 1).bit_length()


def nor_rom_lunit_name(
    depth: int, input_size: int, weight_width: int = 4, lanes: int = 1
) -> str:
    """The name of the L-Unit macro hardened on a NOR ROM bank."""
    name = f"LUnitNORROM_asap7_Macro_d{depth}_in{input_size}"
    if weight_width != 4:
        name += f"_w{weight_width}"
    return name + (f"_x{lanes}" if lanes > 1 else "")


def nor_rom_collateral(
    name: str, root: Path | None = None, lanes: int = 1
) -> ROMCollateral:
    """``ROMCollateral`` of a built bank (paths may not exist yet)."""
    d = bank_dir(name, root or ASAP7_NOR_ROM_ROOT, lanes)
    meta = d / "bank.json"
    cells = tuple(json.loads(meta.read_text())["cells"]) if meta.exists() else ()
    return ROMCollateral(
        lef=d / "macro.lef",
        libs={c: d / f"macro_{s}.lib" for c, s in CORNERS.items()},
        gds=d / "macro.gds",
        build_command="uv run python scripts/asap7_nor_rom_bank.py",
        cells=cells,
        rtl=(d / "wrapper.v", d / "blackbox.v"),
    )


def asap7_nor_rom_interface(
    rows_per_bank: int, root: Path | None = None, lanes: int = 1
) -> ROMMacroInterface:
    """Interface for one NOR ROM bank of ``rows_per_bank`` words (one bank per lane
    of a ``lanes``-lane L-Unit, each with its own cells)."""
    return ROMMacroInterface(
        # CLK rising latches A and starts the read.
        read="edge_flow_through",
        addr_hold_edge="rise",
        reset=True,
        in_macro_mux=True,
        rows_per_bank=rows_per_bank,
        collateral=functools.partial(nor_rom_collateral, root=root, lanes=lanes),
        lane_cells=lanes > 1,
    )


def bank_liberty(text: str, image_cell: str, cells: tuple[str, ...]) -> str:
    """The compiler's Liberty of `image_cell`, with one identical cell per name."""
    match = re.search(
        rf"^  cell\({re.escape(image_cell)}\) \{{\n.*?^  \}}\n",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"no cell {image_cell} in this Liberty")
    copies = "".join(
        match[0].replace(f"cell({image_cell})", f"cell({cell})", 1) for cell in cells
    )
    head = text[: match.start()].replace(
        f"library({image_cell}_", f"library({cells[0]}_"
    )
    return head + copies + text[match.end() :]


def register_asap7_nor_rom_backend(
    rows_per_bank: int = 256, root: Path | None = None, lanes: int = 1
) -> ROMMacroInterface:
    """Register (or re-register) ``rom_macro_backend="asap7_nor_rom"``.

    The default is one 256-word bank; a flow that builds another passes its own.
    """
    interface = asap7_nor_rom_interface(rows_per_bank, root, lanes)
    register_rom_macro_interface(ASAP7_NOR_ROM_BACKEND, interface)
    return interface
