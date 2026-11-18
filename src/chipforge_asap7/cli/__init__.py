"""The ``asap7`` command: cells, lanes, SRAM and ROM macros, verification.

Laid out like chipforge_gt2n's ``gt2n`` command, by group and then generator:

    asap7 cell finfet --fins 2 --fingers 2
    asap7 lane sense-amp-row ...
    asap7 rom brom --words-json words.json --word-bits 8 --output-dir out
    asap7 rom nor --array.words 64 --array.bits 8
    asap7 sram macro-bench --wordlines 32 --word-bits 2
    asap7 verify lvs --gds a.gds --schematic a.spice --cell TOP --out lvs

Each generator keeps its own options (``asap7 rom brom -h``); the command only
routes to it. The single-purpose ``asap7-*`` scripts still work and run the
same code.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from contextlib import contextmanager

#: group -> (help, {generator: (module, help)}). A group whose table is a
#: module string has no generators of its own: its arguments go to that module.
GROUPS: dict[str, tuple[str, dict[str, tuple[str, str]] | str]] = {
    "cell": (
        "devices, cells and bare arrays",
        {
            "finfet": ("chipforge_asap7.devices.finfet", "parametric FinFET"),
            "inverter": ("chipforge_asap7.devices.inverter", "inverter"),
            "nand": ("chipforge_asap7.devices.nand", "NAND gate"),
            "output-latch": ("chipforge_asap7.devices.output_latch", "output latch"),
            "write-driver": ("chipforge_asap7.devices.write_driver", "write driver"),
            "row-support": ("chipforge_asap7.devices.row_support", "row taps and fillers"),
            "rom-bitcell": ("chipforge_asap7.devices.rom_bitcell", "NOR ROM bitcell and array"),
            "bitline-mux": ("chipforge_asap7.devices.bitline_mux", "bitline column mux"),
        },
    ),
    "lane": (
        "peripheral lanes beside an array",
        {
            "sense-amp-row": ("chipforge_asap7.devices.sense_amp_row", "sense amplifier row"),
            "wl-slice": ("chipforge_asap7.devices.driver_slice", "wordline driver slice"),
            "io-column": ("chipforge_asap7.devices.io_column", "IO column"),
        },
    ),
    "sram": (
        "SRAM benches: the 8T 1R1W macro and the 6T read",
        {
            "6t-read": ("chipforge_asap7.devices.sram6t_read_spice", "6T SRAM read transient bench"),
            "column-bench": ("chipforge_asap7.sram1r1w.column_bench", "1R1W column transient bench"),
            "macro-bench": ("chipforge_asap7.sram1r1w.macro_bench", "1R1W macro transient bench"),
        },
    ),
    "rom": (
        "ROM macros: the NOR ROM and the B-ROM",
        {
            "nor": ("chipforge_asap7.devices.rom_decoder", "NOR ROM array with its decoder"),
            "brom": ("chipforge_asap7.brom.__main__", "programmed B-ROM macro and views"),
            "brom-timing": ("chipforge_asap7.brom.timing", "B-ROM transistor-level timing"),
            "nor-read": ("chipforge_asap7.devices.rom_read_spice", "NOR ROM read transient bench"),
            "nor-address-search": (
                "chipforge_asap7.devices.rom_address_search",
                "search the NOR ROM decoder for its slowest address step",
            ),
        },
    ),
    "verify": ("LVS, PEX and model validation", "chipforge_asap7.verification.__main__"),
    "reduce": ("shrink a GDS to a minimal DRC reproducer", "chipforge_asap7.verification.reduce"),
}


def _usage(prefix: str, table: dict[str, tuple[str, str]], summary: str) -> str:
    width = max(map(len, table))
    rows = "\n".join(f"  {name:<{width}}  {item[1]}" for name, item in table.items())
    return f"usage: {prefix} {{{','.join(table)}}} ...\n\n{summary}\n\n{rows}\n"


@contextmanager
def _prog(name: str):
    """Show ``name`` as the program in the generator's own usage and errors."""
    saved = sys.argv[0]
    sys.argv[0] = name
    try:
        yield
    finally:
        sys.argv[0] = saved


def _run(module: str, prog: str, argv: list[str]) -> int:
    with _prog(prog):
        status = importlib.import_module(module).main(argv)
    return 0 if status is None else int(status)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    top = {name: (name, help) for name, (help, _) in GROUPS.items()}
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage("asap7", top, __doc__.split("\n\n")[0]), end="")
        return 0 if argv else 2
    group, rest = argv[0], argv[1:]
    if group not in GROUPS:
        print(_usage("asap7", top, f"asap7: unknown group {group!r}"), end="", file=sys.stderr)
        return 2
    summary, table = GROUPS[group]
    if isinstance(table, str):
        return _run(table, f"asap7 {group}", rest)
    if not rest or rest[0] in ("-h", "--help"):
        print(_usage(f"asap7 {group}", table, summary), end="")
        return 0 if rest else 2
    name, rest = rest[0], rest[1:]
    if name not in table:
        message = f"asap7 {group}: unknown generator {name!r}"
        print(_usage(f"asap7 {group}", table, message), end="", file=sys.stderr)
        return 2
    return _run(table[name][0], f"asap7 {group} {name}", rest)
