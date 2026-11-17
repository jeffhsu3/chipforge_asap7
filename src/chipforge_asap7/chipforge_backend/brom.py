"""ASAP7 B-ROM (``chipforge_asap7.brom``) as the weight ROM of one L-Unit.

``rom_macro_backend="asap7_brom"``: registered when ``chipforge_asap7`` is
imported with chipforge installed, as chipforge_gt2n registers ``gt2n_brom``.
Core's ``chipforge.memory.asap7_brom`` re-exports this module. The built banks
stay in core's ``macros/asap7_brom`` (`rom_macros_dir`).

Programming only moves V2/V3 cuts, so every image of one B-ROM shape has the same
footprint, pins and other layers. As with the NOR ``ROM_FULL``, one L-Unit macro
serves every unit: it is hardened against a bank of stable-named abstracts
(``brom_selftimed_{depth}x{bits}_m{ways}_..._s{i}``, built by
``workflows/lunit_rom_macro.smk --config rom=brom``), and each unit's programmed
macros replace those cells after P&R (``scripts/program_brom_gds_swap.py``). Their
cells are named by their contents, so programmed units merge into one chip GDS.

A multi-lane L-Unit (``lanes`` > 1: the dense / paired companion lanes) gets
one bank per lane, all of one shape. Each lane instantiates its own wrapper
(``ROM_FULL_*`` for lane 0, ``ROM_FULL_*_L{k}`` for lane k) over its own
abstract masters (``..._s{i}``, ``..._l{k}_s{i}``), so the swap programs each
lane with its own words. The lanes' collateral is one bank directory
(``rom_full_*_x{lanes}``).

The bank's depth is a catalog depth (``brom_depth``: 4-word blocks in the
1-1.5-2 series, 16 words up), so models with different word counts share one
bank shape and one hardened L-Unit; the images zero-pad their words to it.

A word wider than `BROM_MAX_SLICE_BITS` is split across several macros of equal
width (`word_slices`); a 148-bit word is one macro. The compiler routes a
1024x148 4-way bank as a single macro (DRC and LVS clean, hardened into a
110x80 um L-Unit). All slices share A, CLK and RESETN, and each times its own
read.

Interface: ``--self-timed --registered-output`` (``--clocked`` needs four
external completion acknowledgments). CLK high precharges, CLK falling latches
A, Q updates on the next rising edge: a one-cycle registered read, i.e.
``ROMMacroInterface(clocked=True, registered_output=True)``. The in-macro 4:1
mux takes the column bits, so ``addr`` is the logical word index. RESETN must
be released with CLK high before the first read, hence ``reset=True``.

``ROM_FULL_*`` stays the name the L-Unit instantiates; it is a soft wrapper
(``wrapper.v``) around the abstract B-ROM cell, which is the hard macro.
Torch-free, like ``rom_interface``.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

from chipforge.memory.rom_depth_catalog import catalog_depth
from chipforge.memory.rom_interface import (
    ROMCollateral,
    ROMMacroInterface,
    register_rom_macro_interface,
)

from .paths import rom_macros_dir

ASAP7_BROM_BACKEND = "asap7_brom"
ASAP7_BROM_ROOT = rom_macros_dir() / "asap7_brom"
BROM_WAYS = (1, 4)
#: Widest macro a word is split into: the L-Unit's whole 148-bit word.
BROM_MAX_SLICE_BITS = 148
#: Shallowest catalog bank, in 4-word blocks (16 words at column_mux 1).
BROM_CATALOG_MIN_BLOCKS = 4

# Liberty corner -> file suffix (matches the NOR ROM_FULL corners).
CORNERS = {"fast": "ff", "typical": "tt", "slow": "ss"}

# NOT characterized: chipforge_asap7 has no B-ROM Liberty (timing_qualified is
# false). CLK->Q is the output register's TT/1 fF simulation (~57/63 ps,
# chipforge_asap7 docs/brom_timing_estimates.md) with margin; the address
# setup to the CLK fall is a guess. Corners scale by the NOR libs' ff/ss ratio.
_ESTIMATE_PS = {
    "typical": {"clk_q": 80.0, "slew": 25.0, "setup": 40.0, "hold": 10.0},
    "fast": {"clk_q": 60.0, "slew": 18.0, "setup": 30.0, "hold": 10.0},
    "slow": {"clk_q": 115.0, "slew": 35.0, "setup": 60.0, "hold": 20.0},
}


def brom_depth(depth: int, column_mux: int, exact: bool = False) -> int:
    """Words in the B-ROM bank holding ``depth`` words.

    The catalog depth (``chipforge.memory.rom_depth_catalog``): 4-word blocks
    in the 1-1.5-2 series from ``BROM_CATALOG_MIN_BLOCKS`` up, so models of
    different sizes share one bank shape and one hardened L-Unit (116 words
    take the 128-word bank). ``exact`` rounds up to whole blocks only, the
    pre-catalog depth of a bank compiled for exactly these words.
    """
    if column_mux not in BROM_WAYS:
        raise ValueError(f"B-ROM column mux must be 1 or 4, got {column_mux}")
    block = 4 * column_mux
    if exact:
        return max(block, -(-int(depth) // block) * block)
    return catalog_depth(depth, block, BROM_CATALOG_MIN_BLOCKS)


def word_slices(word_bits: int, max_bits: int = BROM_MAX_SLICE_BITS) -> tuple[int, int]:
    """(count, width) of the equal macros holding a ``word_bits``-bit word."""
    if word_bits < 1 or max_bits < 1:
        raise ValueError(f"cannot slice {word_bits} bits into {max_bits}-bit macros")
    count = -(-word_bits // max_bits)
    return count, -(-word_bits // count)


def slice_cells(cell: str, count: int, lane: int = 0) -> tuple[str, ...]:
    """Each slice's stable master name; the swap tells slices (and lanes)
    apart by it. Lane 0 keeps the single-lane names."""
    tag = f"_l{lane}" if lane else ""
    return tuple(f"{cell}{tag}_s{i}" for i in range(count))


def lane_rom_full(name: str, lane: int) -> str:
    """The ``ROM_FULL_*`` wrapper lane ``lane`` instantiates (``lane_interface``)."""
    return f"{name}_L{lane}" if lane else name


def brom_lunit_name(
    depth: int,
    input_size: int,
    weight_width: int = 4,
    column_mux: int = 1,
    lanes: int = 1,
) -> str:
    """The B-ROM L-Unit macro's name (``lunit_rom_macro.smk --config rom=brom``)."""
    name = f"LUnitBROM_asap7_Macro_d{depth}_in{input_size}"
    if weight_width != 4:
        name += f"_w{weight_width}"
    if column_mux > 1:
        name += f"_mux{column_mux}"
    return name + (f"_x{lanes}" if lanes > 1 else "")


def split_words(words: list[int], count: int, width: int) -> list[list[int]]:
    """Per slice, its bits of each word: slice 0 holds the low ``width`` bits."""
    mask = (1 << width) - 1
    return [[(w >> (i * width)) & mask for w in words] for i in range(count)]


def bank_dir(name: str, root: Path | None = None, lanes: int = 1) -> Path:
    """Directory holding one ``ROM_FULL_*`` wrapper's B-ROM bank collateral
    (for ``lanes`` > 1, every lane's wrapper; ``name`` may be any lane's)."""
    base = re.sub(r"_L\d+$", "", name).lower()
    return Path(root or ASAP7_BROM_ROOT) / (base + (f"_x{lanes}" if lanes > 1 else ""))


def brom_collateral(
    name: str, root: Path | None = None, lanes: int = 1
) -> ROMCollateral:
    """``ROMCollateral`` of a built bank (paths may not exist yet)."""
    d = bank_dir(name, root, lanes)
    meta = d / "bank.json"
    cells = tuple(json.loads(meta.read_text())["cells"]) if meta.exists() else ()
    return ROMCollateral(
        lef=d / "macro.lef",
        libs={c: d / f"macro_{s}.lib" for c, s in CORNERS.items()},
        gds=d / "macro.gds",
        build_command=(
            "uv run snakemake -s workflows/lunit_rom_macro.smk --config rom=brom"
        ),
        cells=cells,
        rtl=(d / "wrapper.v", d / "blackbox.v"),
    )


def lef_foreign(lef: Path, cell: str) -> tuple[int, int]:
    """``cell``'s FOREIGN point in nm: per the LEF spec, the LEF origin in the
    GDS cell's coordinates, so a placed macro's GDS origin is its LEF origin
    minus this (as KLayout's DEF reader assembles it)."""
    match = re.search(
        rf"^MACRO {re.escape(cell)}\s.*?^\s*FOREIGN \S+ (\S+) (\S+)",
        lef.read_text(),
        re.S | re.M,
    )
    if match is None:
        raise ValueError(f"no FOREIGN for {cell} in {lef}")
    return round(float(match[1]) * 1000), round(float(match[2]) * 1000)


def asap7_brom_interface(
    rows_per_bank: int, root: Path | None = None, lanes: int = 1
) -> ROMMacroInterface:
    """Interface for one B-ROM bank of ``rows_per_bank`` physical rows
    (one bank per lane of a ``lanes``-lane L-Unit, each with its own cells)."""
    return ROMMacroInterface(
        # CLK falling latches A, so A need only be held to the fall.
        read="two_phase",
        addr_hold_edge="fall",
        reset=True,
        in_macro_mux=True,
        rows_per_bank=rows_per_bank,
        collateral=functools.partial(brom_collateral, root=root, lanes=lanes),
        lane_cells=lanes > 1,
    )


def brom_words(words: list[int], depth: int, word_bits: int) -> list[int]:
    """The bank's words in address order: ``words`` zero-padded to ``depth``."""
    words = [int(w) for w in words]
    if len(words) > depth:
        raise ValueError(f"{len(words)} words do not fit a {depth}-word bank")
    wide = [i for i, w in enumerate(words) if w < 0 or w >> word_bits]
    if wide:
        raise ValueError(f"word {wide[0]} is not a {word_bits}-bit unsigned value")
    return words + [0] * (depth - len(words))


def wrapper_verilog(
    name: str, cells: tuple[str, ...], addr_bits: int, data_bits: int, slice_bits: int
) -> str:
    """``ROM_FULL_*`` port contract (clk/rst/addr/read_data) -> B-ROM slices."""
    banks = "".join(
        f"  {cell} bank{i} (.A(addr), .CLK(clk), .RESETN(resetn), "
        f".Q(q[{(i + 1) * slice_bits - 1}:{i * slice_bits}]));\n"
        for i, cell in enumerate(cells)
    )
    return (
        f"// ROM_FULL contract -> {len(cells)} ASAP7 B-ROM slices "
        "(chipforge.memory.asap7_brom)\n"
        f"module {name} (\n"
        "    input clk,\n"
        "    input rst,\n"
        f"    input  [{addr_bits - 1}:0] addr,\n"
        f"    output [{data_bits - 1}:0] read_data\n"
        ");\n"
        "  wire resetn = ~rst;\n"
        f"  wire [{len(cells) * slice_bits - 1}:0] q;\n"
        f"{banks}"
        f"  assign read_data = q[{data_bits - 1}:0];\n"
        "endmodule\n"
    )


def blackbox_verilog(cells: tuple[str, ...], addr_bits: int, data_bits: int) -> str:
    """Yosys blackboxes of the B-ROM cells (VDD/VSS are physical-only pins)."""
    return "".join(
        "(* blackbox *)\n"
        f"module {cell} (\n"
        f"    input  [{addr_bits - 1}:0] A,\n"
        "    input CLK,\n"
        "    input RESETN,\n"
        f"    output [{data_bits - 1}:0] Q\n"
        ");\nendmodule\n"
        for cell in cells
    )


def _bus_type(name: str, width: int) -> str:
    return (
        f"  type ({name}) {{\n    base_type : array;\n    data_type : bit;\n"
        f"    bit_width : {width};\n    bit_from : {width - 1};\n"
        "    bit_to : 0;\n    downto : true;\n  }\n"
    )


def liberty_text(
    cells: tuple[str, ...], addr_bits: int, data_bits: int, corner: str, area: float
) -> str:
    """Estimated (uncharacterized) Liberty for one corner of the B-ROM cells.

    A latches at the CLK fall (setup/hold_falling); Q launches at the CLK
    rise from the output register. RESETN carries no arc: it is released
    once, after reset, while the clock runs.
    """
    t = _ESTIMATE_PS[corner]
    a_bits = [f"A[{i}]" for i in range(addr_bits)]
    q_bits = [f"Q[{i}]" for i in range(data_bits)]

    def scalar(kind: str, value: float) -> str:
        return f'        {kind} (scalar) {{ values ("{value}"); }}\n'

    a_pins = "".join(
        f"      pin({p}) {{\n        direction : input;\n        capacitance : 1.0;\n"
        '        timing() {\n          related_pin : "CLK";\n'
        "          timing_type : setup_falling;\n"
        + scalar("  rise_constraint", t["setup"])
        + scalar("  fall_constraint", t["setup"])
        + '        }\n        timing() {\n          related_pin : "CLK";\n'
        "          timing_type : hold_falling;\n"
        + scalar("  rise_constraint", t["hold"])
        + scalar("  fall_constraint", t["hold"])
        + "        }\n      }\n"
        for p in a_bits
    )
    q_pins = "".join(
        f"      pin({p}) {{\n        direction : output;\n"
        '        timing() {\n          related_pin : "CLK";\n'
        "          timing_type : rising_edge;\n"
        + scalar("  cell_rise", t["clk_q"])
        + scalar("  cell_fall", t["clk_q"])
        + scalar("  rise_transition", t["slew"])
        + scalar("  fall_transition", t["slew"])
        + "        }\n      }\n"
        for p in q_bits
    )
    body = "".join(
        f"  cell({cell}) {{\n    area : {area:.4f};\n    dont_touch : true;\n"
        "    dont_use : true;\n    interface_timing : true;\n"
        "    pin(CLK) {\n      direction : input;\n      clock : true;\n"
        "      capacitance : 2.0;\n    }\n"
        "    pin(RESETN) {\n      direction : input;\n      capacitance : 1.0;\n    }\n"
        f"    bus(A) {{\n      bus_type : BROM_A_{addr_bits};\n"
        f"      direction : input;\n{a_pins}    }}\n"
        f"    bus(Q) {{\n      bus_type : BROM_Q_{data_bits};\n"
        f"      direction : output;\n{q_pins}    }}\n"
        "  }\n"
        for cell in cells
    )
    return (
        f"/* {cells[0]}: ASAP7 B-ROM, {corner} corner. ESTIMATED, NOT CHARACTERIZED\n"
        " * (chipforge.memory.asap7_brom). */\n"
        f"library({cells[0]}_{CORNERS[corner]}) {{\n"
        "  technology (cmos);\n  delay_model : table_lookup;\n"
        '  time_unit : "1ps";\n  voltage_unit : "1V";\n  current_unit : "1mA";\n'
        '  pulling_resistance_unit : "1kohm";\n  capacitive_load_unit (1, ff);\n'
        "  slew_upper_threshold_pct_fall : 80;\n  slew_lower_threshold_pct_fall : 20;\n"
        "  slew_upper_threshold_pct_rise : 80;\n  slew_lower_threshold_pct_rise : 20;\n"
        "  input_threshold_pct_fall : 50;\n  input_threshold_pct_rise : 50;\n"
        "  output_threshold_pct_fall : 50;\n  output_threshold_pct_rise : 50;\n"
        + _bus_type(f"BROM_A_{addr_bits}", addr_bits)
        + _bus_type(f"BROM_Q_{data_bits}", data_bits)
        + body
        + "}\n"
    )


def register_asap7_brom_backend(
    rows_per_bank: int = 256, root: Path | None = None, lanes: int = 1
) -> ROMMacroInterface:
    """Register (or re-register) ``rom_macro_backend="asap7_brom"``.

    The default is one 256-row bank, the 1024-word 4:1 shape; a flow that
    builds another passes its own, as ``register_gt2n_brom_backend`` takes its
    collateral options.
    """
    interface = asap7_brom_interface(rows_per_bank, root, lanes)
    register_rom_macro_interface(ASAP7_BROM_BACKEND, interface)
    return interface
