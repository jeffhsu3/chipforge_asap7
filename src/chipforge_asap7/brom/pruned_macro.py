"""Data-pruned dual-slot CGEN, decoupled-bitline, and multi-Vt BROM compiler.

Unlike ``macro_circuit`` / ``build_dense_macro`` (which preserve a fixed base
abstract so word reprogramming only modifies V2 and V3 cuts), this flow targets
custom-mask hard macros whose ROM contents are fixed before tapeout:

1. **Co-occurring dual-``NOR2`` ``324 nm`` / singleton ``216 nm`` CGEN**:
   pairs active ``k in 1..14`` candidates by block co-occurrence into shared-
   ``SELB`` ``324 nm`` dual-``NOR2`` slots (``216 nm`` for unpaired singletons
   and ``k = 15`` ``INV``), instantiating zero dummy transistors in blocks where
   only zero or one member of a pair is used.
2. **Decoupled ``k = 0`` bitcells & 2D-trimmed array wires**: grounds ``k = 0``
   selector gates and middle diffusions locally to ``VSS`` ``M1/M2`` without
   bridging onto ``BL<col>``, trims horizontal ``M2/M4`` candidate tracks to the
   active column span in each block, and trims vertical ``M3`` bitlines right
   above each column's highest programmed block.
3. **Active-block-only 2-stack ``EVAL -> SELB<b>`` path**: pre-combines
   ``PRECHN`` with active-low block decode during precharge and emits ``SELB<b>``
   drivers and left-channel risers only for non-zero blocks, leaving a 2-NMOS
   ``NAND2(EVAL, BPRE<b>)`` + 1-stage ``NOR2(GB<k>, SELB<b>)`` on the ``EVAL``
   rising edge while cutting ``PRECHN`` decode gate load to 2 fins.
4. **Energy-sized readout + optional multi-Vt low-leakage co-design**: fuses
   1-fin precharge, hi-skewed sense, and 4:1 column selection beside an
   isolated-keeper Q latch, with optional ``periphery_vt`` / ``crit_vt`` split
   (e.g. ``1-fin sram`` selector array + off-critical logic paired with ``rvt``
   or ``lvt`` on the critical ``EVAL -> Q`` read path).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
from typing import Any

from ..devices.finfet import VT_LAYERS
from ..devices.row import RowStack
from ..devices.row_support import RowSupportSpec, build_row_support
from ..devices.rowcell import box, draw_frame, draw_rails, label, square
from ..layout.layers import require_gdspy
from .circuit import Circuit, Device, SelectorInstance
from .compiler import BromSpec, selector_map
from .dense_bank import build_dense_bank, ceil_grid
from .dense_cell import build_dense_cell, wall_gap
from .dense_cgen import lift6, v3, v4
from .dense_units import dense_units
from .selector import SelectorPlan, default_selector_plan

__all__ = [
    "CgenSlotPlan",
    "build_pruned_macro",
    "build_pruned_selector_array",
    "pruned_macro_circuit",
    "pruned_macro_layout",
]


@dataclass(frozen=True)
class CgenSlotPlan:
    """Co-occurring candidate slot layout and per-block track map."""

    slots: tuple[tuple[int, int | None], ...]
    slot_widths: tuple[int, ...]
    slot_offsets: tuple[int, ...]
    total_width: int
    gb_offsets: tuple[tuple[int, int], ...]
    active_k: tuple[int, ...]
    per_block_k2t: tuple[dict[int, int], ...]
    max_upper: int
    lane_vt: str
    cgen_pfins: int
    cgen_nfins: int


def _fet(
    c: Circuit, pmos: bool, d: str, g: str, s: str, fins: int, vt: str
) -> None:
    base = c.p if pmos else c.n
    spec = (
        base
        if (base.fins == fins and base.vt == vt)
        else replace(base, fins=fins, vt=vt)
    )
    c.devices.append(
        Device(f"M{len(c.devices)}", spec, (d, g, s, "VDD" if pmos else "VSS"))
    )


def _record_unit(c: Circuit, name: str, start: int) -> None:
    members = tuple(d.name for d in c.devices[start:])
    c.placement_groups.append((name, members))
    c.logic_units.append((name, members))


def _add_sized_inv(
    c: Circuit,
    name: str,
    a: str,
    y: str,
    pfins: int = 2,
    nfins: int = 1,
    *,
    fingers: int = 1,
    vt: str | None = None,
) -> None:
    use_vt = c.n.vt if vt is None else vt
    start = len(c.devices)
    for _ in range(fingers):
        _fet(c, True, y, a, "VDD", pfins, use_vt)
        _fet(c, False, y, a, "VSS", nfins, use_vt)
    _record_unit(c, name, start)


def _add_nand2(
    c: Circuit,
    name: str,
    a: str,
    b: str,
    y: str,
    pfins: int = 1,
    nfins: int = 2,
    *,
    vt: str | None = None,
) -> None:
    use_vt = c.n.vt if vt is None else vt
    start = len(c.devices)
    mid = c.node()
    _fet(c, True, y, a, "VDD", pfins, use_vt)
    _fet(c, True, y, b, "VDD", pfins, use_vt)
    _fet(c, False, y, a, mid, nfins, use_vt)
    _fet(c, False, mid, b, "VSS", nfins, use_vt)
    _record_unit(c, name, start)


def _add_nor2(
    c: Circuit,
    name: str,
    a: str,
    b: str,
    y: str,
    pfins: int = 2,
    nfins: int = 1,
    *,
    vt: str | None = None,
) -> None:
    use_vt = c.n.vt if vt is None else vt
    start = len(c.devices)
    mid = c.node()
    _fet(c, True, mid, a, "VDD", pfins, use_vt)
    _fet(c, True, y, b, mid, pfins, use_vt)
    _fet(c, False, y, a, "VSS", nfins, use_vt)
    _fet(c, False, y, b, "VSS", nfins, use_vt)
    _record_unit(c, name, start)


def _add_aoi22(
    c: Circuit,
    name: str,
    a0: str,
    a1: str,
    b0: str,
    b1: str,
    y: str,
    fins: int = 1,
    *,
    vt: str | None = None,
) -> None:
    use_vt = c.n.vt if vt is None else vt
    start = len(c.devices)
    pa = c.node()
    na = c.node()
    nb = c.node()
    _fet(c, True, pa, a0, "VDD", fins, use_vt)
    _fet(c, True, pa, a1, "VDD", fins, use_vt)
    _fet(c, True, y, b0, pa, fins, use_vt)
    _fet(c, True, y, b1, pa, fins, use_vt)
    _fet(c, False, y, a0, na, fins, use_vt)
    _fet(c, False, na, a1, "VSS", fins, use_vt)
    _fet(c, False, y, b0, nb, fins, use_vt)
    _fet(c, False, nb, b1, "VSS", fins, use_vt)
    _record_unit(c, name, start)


def _add_addr_latch(
    c: Circuit,
    idx: int,
    d_pin: str,
    prechn: str,
    prechn_open: str,
    q_out: str,
    fins: int = 1,
    *,
    vt: str | None = None,
) -> None:
    """Compact active-low input address latch (settles during precharge)."""
    use_vt = c.n.vt if vt is None else vt
    start = len(c.devices)
    st = c.node()
    qb = c.node()
    _fet(c, False, st, prechn_open, d_pin, fins, use_vt)
    _fet(c, True, st, prechn, d_pin, fins, use_vt)
    _fet(c, True, qb, st, "VDD", fins, use_vt)
    _fet(c, False, qb, st, "VSS", fins, use_vt)
    _fet(c, True, q_out, qb, "VDD", 2, use_vt)
    _fet(c, False, q_out, qb, "VSS", 1, use_vt)
    _fet(c, False, st, prechn, q_out, fins, use_vt)
    _fet(c, True, st, prechn_open, q_out, fins, use_vt)
    _record_unit(c, f"address_latch_{idx}", start)


def _add_sense_m1(
    c: Circuit,
    bit: int,
    bl: str,
    prechn: str,
    dout: str,
    *,
    pre_fins: int = 1,
    keeper_fins: int = 0,
    keeper_stack: int = 2,
    vt: str | None = None,
) -> None:
    """Precharge PMOS + 2p/1n hi-skewed sense inverter (+ optional keeper) for ways=1."""
    use_vt = c.n.vt if vt is None else vt
    start = len(c.devices)
    _fet(c, True, bl, prechn, "VDD", pre_fins, use_vt)
    _fet(c, True, dout, bl, "VDD", 2, use_vt)
    _fet(c, False, dout, bl, "VSS", 1, use_vt)
    source = "VDD"
    for stage in range(keeper_stack if keeper_fins else 0):
        drain = bl if stage == keeper_stack - 1 else c.node()
        _fet(c, True, drain, dout, source, keeper_fins, use_vt)
        source = drain
    # Bank placers find a column's receiver by ``readout_<column>``.
    _record_unit(c, f"readout_{bit}", start)


def _add_sense_way(
    c: Circuit,
    bit: int,
    way: int,
    bl: str,
    prechn: str,
    cs: str,
    csb: str,
    dout: str,
    *,
    pre_fins: int = 1,
    sense_pfins: int = 3,
    vt: str | None = None,
) -> None:
    """1-stage hi-skewed gated tristate sense + precharge for ways=4."""
    use_vt = c.n.vt if vt is None else vt
    start = len(c.devices)
    mp = c.node()
    mn = c.node()
    _fet(c, True, bl, prechn, "VDD", pre_fins, use_vt)
    _fet(c, True, mp, bl, "VDD", sense_pfins, use_vt)
    _fet(c, True, dout, csb, mp, sense_pfins, use_vt)
    _fet(c, False, mn, bl, "VSS", 1, use_vt)
    _fet(c, False, dout, cs, mn, 1, use_vt)
    # Bank placers find a column's receiver by ``readout_<column>``.
    _record_unit(c, f"readout_{bit * 4 + way}", start)


def _add_fast_out_latch(
    c: Circuit,
    bit: int,
    dout: str,
    sample: str,
    sampleb: str,
    q_pin: str,
    *,
    vt: str | None = None,
) -> None:
    """Isolated-keeper tristate sampling latch + 4-fin push-pull output driver."""
    use_vt = c.n.vt if vt is None else vt
    start = len(c.devices)
    mp_d = c.node()
    mn_d = c.node()
    sb = c.node()
    qk = c.node()
    mp_k = c.node()
    mn_k = c.node()
    _fet(c, True, mp_d, dout, "VDD", 2, use_vt)
    _fet(c, True, sb, sampleb, mp_d, 2, use_vt)
    _fet(c, False, mn_d, dout, "VSS", 2, use_vt)
    _fet(c, False, sb, sample, mn_d, 2, use_vt)
    _fet(c, True, qk, sb, "VDD", 1, use_vt)
    _fet(c, False, qk, sb, "VSS", 1, use_vt)
    _fet(c, True, mp_k, qk, "VDD", 1, use_vt)
    _fet(c, True, sb, sample, mp_k, 1, use_vt)
    _fet(c, False, mn_k, qk, "VSS", 1, use_vt)
    _fet(c, False, sb, sampleb, mn_k, 1, use_vt)
    _fet(c, True, q_pin, sb, "VDD", 2, use_vt)
    _fet(c, True, q_pin, sb, "VDD", 2, use_vt)
    _fet(c, False, q_pin, sb, "VSS", 2, use_vt)
    _fet(c, False, q_pin, sb, "VSS", 2, use_vt)
    # ``output_latch_<bit>`` lets wide words use bit tiles (``SLICE_FIRST_BITS``).
    _record_unit(c, f"output_latch_{bit}", start)


def _decode_active_high(
    c: Circuit,
    bits: list[str],
    prefix: str,
    needed: set[int],
    *,
    vt: str,
) -> None:
    """Factored active-high precharge-window decoder pruned to ``needed`` outputs."""
    n = len(bits)
    if not needed:
        return
    if n == 1:
        if 0 in needed:
            _add_sized_inv(c, f"{prefix}_inv0", bits[0], f"{prefix}<0>", 2, 1, vt=vt)
        if 1 in needed:
            b0 = c.node()
            _add_sized_inv(c, f"{prefix}_buf1a", bits[0], b0, 1, 1, vt=vt)
            _add_sized_inv(c, f"{prefix}_buf1b", b0, f"{prefix}<1>", 2, 1, vt=vt)
        return
    if n == 2:
        n0 = c.node() if (needed & {1}) else ""
        n1 = c.node() if (needed & {2}) else ""
        if n0:
            _add_sized_inv(c, f"{prefix}_in0", bits[0], n0, 1, 1, vt=vt)
        if n1:
            _add_sized_inv(c, f"{prefix}_in1", bits[1], n1, 1, 1, vt=vt)
        if 0 in needed:
            _add_nor2(c, f"{prefix}_dec_0", bits[1], bits[0], f"{prefix}<0>", 1, 1, vt=vt)
        if 1 in needed:
            _add_nor2(c, f"{prefix}_dec_1", bits[1], n0, f"{prefix}<1>", 1, 1, vt=vt)
        if 2 in needed:
            _add_nor2(c, f"{prefix}_dec_2", n1, bits[0], f"{prefix}<2>", 1, 1, vt=vt)
        if 3 in needed:
            nb3 = c.node()
            _add_nand2(c, f"{prefix}_dec_3b", bits[1], bits[0], nb3, 1, 1, vt=vt)
            _add_sized_inv(c, f"{prefix}_dec_3", nb3, f"{prefix}<3>", 2, 1, vt=vt)
        return
    half = n // 2
    lo_mask = (1 << half) - 1
    lo_pref, hi_pref = f"{prefix}_L", f"{prefix}_H"
    _decode_active_high(
        c, bits[:half], lo_pref, {i & lo_mask for i in needed}, vt=vt
    )
    _decode_active_high(
        c, bits[half:], hi_pref, {i >> half for i in needed}, vt=vt
    )
    for i in sorted(needed):
        nb = c.node()
        _add_nand2(
            c,
            f"{prefix}_nd_{i}",
            f"{lo_pref}<{i & lo_mask}>",
            f"{hi_pref}<{i >> half}>",
            nb,
            1,
            1,
            vt=vt,
        )
        _add_sized_inv(c, f"{prefix}_iv_{i}", nb, f"{prefix}<{i}>", 2, 1, vt=vt)


def _decode_block_selb(
    c: Circuit,
    row_addr: list[str],
    built_blocks: Sequence[int],
    eval_sig: str,
    *,
    off_vt: str,
    fast_vt: str,
) -> None:
    """Emit ``SELB<b> = ~(EVAL & BPRE<b>)``, ``BPRE<b> = PRECHN & row_decode(b)``.

    Only ``built_blocks`` get a chain. Each block's chain (its block-decode
    NAND, ``BPRE<b>`` NOR2 and ``SELB<b>`` NAND2) is emitted back to back:
    the bank's fallback placement keeps device order, and emitting each stage
    as its own run across all blocks makes every block's two internal nets
    cross rows, each needing an M5 column (256x30 m1: 381 vs 533 um2).
    """
    prechn_b = c.node()
    _add_sized_inv(c, "bpre_prechn_inv", "PRECHN", prechn_b, 1, 1, vt=off_vt)
    built_set = set(built_blocks)
    n = len(row_addr)
    half = n // 2
    lo_mask = (1 << half) - 1
    blockb: dict[int, str] = {}
    if n == 1:
        if 0 in built_set:
            blockb[0] = row_addr[0]
        if 1 in built_set:
            b1 = c.node()
            _add_sized_inv(c, "blk_inv1", row_addr[0], b1, 1, 1, vt=off_vt)
            blockb[1] = b1
    elif n == 2:
        n0 = c.node() if (built_set & {0, 2, 3}) else ""
        n1 = c.node() if (built_set & {0, 1, 3}) else ""
        if n0:
            _add_sized_inv(c, "blk_in0", row_addr[0], n0, 1, 1, vt=off_vt)
        if n1:
            _add_sized_inv(c, "blk_in1", row_addr[1], n1, 1, 1, vt=off_vt)
        if 0 in built_set:
            b0 = c.node()
            _add_nand2(c, "blkb_0", n1, n0, b0, 1, 1, vt=off_vt)
            blockb[0] = b0
        if 1 in built_set:
            b1 = c.node()
            _add_nand2(c, "blkb_1", n1, row_addr[0], b1, 1, 1, vt=off_vt)
            blockb[1] = b1
        if 2 in built_set:
            b2 = c.node()
            _add_nand2(c, "blkb_2", row_addr[1], n0, b2, 1, 1, vt=off_vt)
            blockb[2] = b2
        if 3 in built_set:
            p3 = c.node()
            b3 = c.node()
            _add_nor2(c, "blkb_3a", n1, n0, p3, 1, 1, vt=off_vt)
            _add_sized_inv(c, "blkb_3", p3, b3, 1, 1, vt=off_vt)
            blockb[3] = b3
    else:
        _decode_active_high(
            c, row_addr[:half], "BLK_L", {b & lo_mask for b in built_set}, vt=off_vt
        )
        _decode_active_high(
            c, row_addr[half:], "BLK_H", {b >> half for b in built_set}, vt=off_vt
        )

    for b in built_blocks:
        if n > 2:
            blockb[b] = c.node()
            _add_nand2(
                c,
                f"blkb_{b}",
                f"BLK_L<{b & lo_mask}>",
                f"BLK_H<{b >> half}>",
                blockb[b],
                1,
                1,
                vt=off_vt,
            )
        bpre = f"BPRE<{b}>"
        _add_nor2(c, f"bpre_{b}", prechn_b, blockb[b], bpre, 2, 1, vt=off_vt)
        _add_nand2(c, f"selb_{b}", eval_sig, bpre, f"SELB<{b}>", 2, 3, vt=fast_vt)


def _add_gb_generator(
    c: Circuit, a0: str, a1: str, needed_gb: Sequence[int], *, vt: str
) -> None:
    """1-stage active-low ``GB<k> = ~G<k>`` generator only for active ``k in 1..14``."""
    n0 = c.node()
    n1 = c.node()
    _add_sized_inv(c, "gb_inv_a0", a0, n0, 2, 1, vt=vt)
    _add_sized_inv(c, "gb_inv_a1", a1, n1, 2, 1, vt=vt)
    if not needed_gb:
        anc = c.node()
        _add_nand2(c, "gb_anchor", n1, a0, anc, 1, 1, vt=vt)
        return

    needed_set = set(needed_gb)
    if 6 in needed_set and 9 in needed_set:
        ordered = [6, *[k for k in needed_gb if k not in (6, 9)], 9]
    else:
        ordered = list(needed_gb)

    for k in ordered:
        out = f"GB<{k}>"
        if k == 10:
            _add_sized_inv(c, f"gb_{k}", a0, out, 2, 1, vt=vt)
        elif k == 5:
            _add_sized_inv(c, f"gb_{k}", n0, out, 2, 1, vt=vt)
        elif k == 12:
            _add_sized_inv(c, f"gb_{k}", a1, out, 2, 1, vt=vt)
        elif k == 3:
            _add_sized_inv(c, f"gb_{k}", n1, out, 2, 1, vt=vt)
        elif k == 1:
            _add_nand2(c, f"gb_{k}", n1, n0, out, 1, 1, vt=vt)
        elif k == 2:
            _add_nand2(c, f"gb_{k}", n1, a0, out, 1, 1, vt=vt)
        elif k == 4:
            _add_nand2(c, f"gb_{k}", a1, n0, out, 1, 1, vt=vt)
        elif k == 8:
            g8 = c.node()
            _add_nor2(c, "gb_8_nor", n1, n0, g8, 1, 1, vt=vt)
            _add_sized_inv(c, "gb_8", g8, out, 2, 1, vt=vt)
        elif k == 14:
            g14 = c.node()
            _add_nand2(c, "gb_14_nand", n1, n0, g14, 1, 1, vt=vt)
            _add_sized_inv(c, "gb_14", g14, out, 2, 1, vt=vt)
        elif k == 13:
            _add_nor2(c, f"gb_{k}", a1, n0, out, 1, 1, vt=vt)
        elif k == 11:
            _add_nor2(c, f"gb_{k}", n1, a0, out, 1, 1, vt=vt)
        elif k == 7:
            _add_nor2(c, f"gb_{k}", n1, n0, out, 1, 1, vt=vt)
        elif k == 6:
            _add_aoi22(c, f"gb_{k}", a1, n0, n1, a0, out, 1, vt=vt)
        elif k == 9:
            if 6 in needed_set:
                _add_sized_inv(c, f"gb_{k}", "GB<6>", out, 2, 1, vt=vt)
            else:
                _add_aoi22(c, f"gb_{k}", a1, a0, n1, n0, out, 1, vt=vt)


def _m2_track_roles(fins: int) -> tuple[tuple[int, int], tuple[int, ...]]:
    """``(direct, upper)`` selector ``M2`` tracks for a fin count.

    Direct tracks are driven straight from the left CGEN; upper tracks are
    lifted over the array on ``M6`` and dropped from the right, so they span
    from their first active column to the array's right edge.  With three
    fins, track 0 (``y = 29``) sits 11 nm from the ``k = 0`` cells' ``VSS``
    ``M2`` strap at the block boundary, so it is neither: an upper track
    would cross those straps.  The five remaining upper tracks still cover a
    block's worst case of ``15 - 10`` lifted candidates.
    """
    direct = (0 if fins < 3 else 1, 6)
    upper = tuple(t for t in range(8) if t not in direct and (fins < 3 or t != 0))
    return direct, upper


@lru_cache(maxsize=64)
def _build_cgen_templates(
    fins: int, vt: str, pfins: int = 3, nfins: int = 1
) -> dict[Any, Any]:
    """Build reusable ``324 nm`` dual-``NOR2`` and ``216 nm`` singleton CGEN cells."""
    gdspy = require_gdspy()
    tracks = default_selector_plan(fins, vt).tracks
    direct = _m2_track_roles(fins)[0]
    stack = RowStack(((fins, fins),), vt=vt, band_height=216)

    spacer = gdspy.Cell(
        f"brom_sp54_{fins}_{vt}_p{pfins}n{nfins}", exclude_from_current=True
    )
    draw_frame(spacer, stack, 54, vt)
    draw_rails(spacer, stack, 54)
    filler = build_row_support(RowSupportSpec(stack, "filler"), draw_pin_labels=False)
    filler.name += f"_pruned_cgen_p{pfins}n{nfins}"
    tap = build_row_support(RowSupportSpec(stack, "tap"), draw_pin_labels=False)
    tap.name += f"_pruned_cgen_p{pfins}n{nfins}"

    pair_devs = (
        ("p", pfins, "m1", "GL", "VDD"),
        ("p", pfins, "CL", "SELB", "m1"),
        ("n", nfins, "CL", "SELB", "VSS"),
        ("n", nfins, "CL", "GL", "VSS"),
        ("p", pfins, "m2", "GR", "VDD"),
        ("p", pfins, "CR", "SELB", "m2"),
        ("n", nfins, "CR", "SELB", "VSS"),
        ("n", nfins, "CR", "GR", "VSS"),
    )
    nl_devs = (
        ("p", pfins, "CL", "SELB", "m1"),
        ("p", pfins, "m1", "GL", "VDD"),
        ("n", nfins, "CL", "GL", "VSS"),
        ("n", nfins, "CL", "SELB", "VSS"),
    )
    nr_devs = (
        ("p", pfins, "m2", "GR", "VDD"),
        ("p", pfins, "CR", "SELB", "m2"),
        ("n", nfins, "CR", "SELB", "VSS"),
        ("n", nfins, "CR", "GR", "VSS"),
    )
    inv_devs = (
        ("p", pfins, "C", "SELB", "VDD"),
        ("n", nfins, "C", "SELB", "VSS"),
    )
    out: dict[Any, Any] = {"spacer": spacer, "filler": filler, "tap": tap}
    for flip in (False, True):
        sy = 414 if flip else 18
        res = (sy, *(432 - tracks[k][1] if flip else tracks[k][1] for k in direct))
        specs = (
            ("pair", pair_devs, ("GL", "GR", "SELB", "CL", "CR"), (("GL", 81), ("GR", 189))),
            ("nl", nl_devs, ("GL", "SELB", "CL"), (("GL", 81), ("CL", 135), ("SELB", 27))),
            ("nr", nr_devs, ("GR", "SELB", "CR"), (("GR", 27),)),
            ("inv", inv_devs, ("SELB", "C"), ()),
        )
        suffix = f"_cgen_p{pfins}n{nfins}_{'hi' if flip else 'lo'}"
        for tag, devs, ports, ftrunks in specs:
            u, p = build_dense_cell(
                devs,
                ports,
                vt=vt,
                reserved_m2=res,
                fixed_trunks=ftrunks,
                share=True,
                abut_y=True,
            )
            u.name += f"_{tag}{suffix}"
            u.labels.clear()
            tx = dict(p.trunks)["SELB"]
            cxs = [dict(p.trunks)[cp] for cp in ports if cp.startswith("C")]
            spans = {(tx - 9, tx + 9), *((cx - 9, cx + 9) for cx in cxs)}
            u.remove_polygons(
                lambda poly, l, d, spans=spans: l == 30
                and d == 0
                and (min(poly[:, 0]), max(poly[:, 0])) in spans
            )
            for polygon in u.polygons:
                polygon.polygons = list(polygon.polygons)
            ys = [ty for n, _, _, ty in p.contacts if n == "SELB"]
            box(u, "M3", tx - 9, min(sy, *ys) - 14, tx + 9, max(sy, *ys) + 14)
            for cx in cxs:
                box(u, "M3", cx - 9, 16, cx + 9, 416)
            out[(tag, flip)] = (u, p)
    out["inv_left_pad"] = wall_gap(out[("inv", False)][1].edges[0])
    return out


def _cgen_slot_plan(
    mapping: tuple[tuple[int, ...], ...],
    fins: int,
    *,
    lane_vt: str = "rvt",
    cgen_pfins: int = 3,
    cgen_nfins: int = 1,
) -> CgenSlotPlan:
    """Pair candidates by block co-occurrence and assign collision-free tracks."""
    freq: dict[int, int] = defaultdict(int)
    for row in mapping:
        for k in set(row):
            if k != 0:
                freq[k] += 1
    k_gb = sorted((k for k in freq if k != 15), key=lambda k: (-freq[k], k))
    has15 = 15 in freq

    slots: list[tuple[int, int | None]] = []
    rem = list(k_gb)
    while rem:
        kl = rem.pop(0)
        if not rem:
            slots.append((kl, None))
            break
        kr = max(
            rem,
            key=lambda ck: (
                sum(1 for row in mapping if (kl in row) == (ck in row)),
                freq[ck],
                -ck,
            ),
        )
        rem.remove(kr)
        slots.append((kl, kr))
    if has15:
        slots.append((15, None))

    widths: list[int] = []
    offsets: list[int] = []
    gb_offsets: list[tuple[int, int]] = []
    active_k: list[int] = []
    cursor = 0
    for kl, kr in slots:
        w = 216 if kr is None else 324
        offsets.append(cursor)
        widths.append(w)
        active_k.append(kl)
        if kl != 15:
            gb_offsets.append((kl, cursor + 81))
        if kr is not None:
            active_k.append(kr)
            gb_offsets.append((kr, cursor + 189))
        cursor += w

    cgen_t = _build_cgen_templates(fins, lane_vt, cgen_pfins, cgen_nfins)
    direct_m2, upper_m2 = _m2_track_roles(fins)
    m4_tracks = (8, 9, 10, 11, 12, 13, 14, 15)

    per_block_k2t: list[dict[int, int]] = []
    max_upper = 0
    for b, row in enumerate(mapping):
        row_set = {k for k in row if k != 0}
        flip = bool(b % 2)
        pair_trunks = dict(cgen_t[("pair", flip)][1].trunks)
        cl_first = pair_trunks["CL"] <= pair_trunks["CR"]

        slot_groups: list[list[int]] = []
        for kl, kr in slots:
            hl = kl in row_set
            hr = kr is not None and kr in row_set
            if hl and hr:
                assert kr is not None
                slot_groups.append([kl, kr] if cl_first else [kr, kl])
            elif hl:
                slot_groups.append([kl])
            elif hr:
                assert kr is not None
                slot_groups.append([kr])

        k2t: dict[int, int] = {}
        n_b = sum(len(g) for g in slot_groups)
        if n_b <= 8:
            flat = [k for g in slot_groups for k in g]
            for idx, k in enumerate(flat):
                k2t[k] = m4_tracks[idx]
        elif n_b <= 10:
            flat = [k for g in slot_groups for k in g]
            n_dir = n_b - 8
            for idx, k in enumerate(flat[:-n_dir]):
                k2t[k] = m4_tracks[idx]
            for idx, k in enumerate(flat[-n_dir:]):
                k2t[k] = direct_m2[idx]
        else:
            n_up = n_b - 10
            max_upper = max(max_upper, n_up)
            left_groups = slot_groups[:n_up]
            right_groups = slot_groups[n_up:]
            for idx, grp in enumerate(left_groups):
                k2t[grp[0]] = upper_m2[idx]
            left_seconds = [grp[1] for grp in left_groups if len(grp) == 2]
            right_cands = [k for grp in right_groups for k in grp]
            # Tracks 8 and 9 (y = 60, 108) sit in slots strictly right of all lift6
            # jogs (y4 = 60), so left_seconds land on tracks 10..15 (dy >= 72 nm).
            k2t[right_cands[0]] = m4_tracks[0]
            k2t[right_cands[1]] = m4_tracks[1]
            rest = [*left_seconds, *right_cands[2:]]
            for idx, k in enumerate(rest[:6]):
                k2t[k] = m4_tracks[2 + idx]
            for idx, k in enumerate(rest[6:]):
                k2t[k] = direct_m2[idx]
        per_block_k2t.append(k2t)

    return CgenSlotPlan(
        slots=tuple(slots),
        slot_widths=tuple(widths),
        slot_offsets=tuple(offsets),
        total_width=cursor,
        gb_offsets=tuple(gb_offsets),
        active_k=tuple(active_k),
        per_block_k2t=tuple(per_block_k2t),
        max_upper=max_upper,
        lane_vt=lane_vt,
        cgen_pfins=cgen_pfins,
        cgen_nfins=cgen_nfins,
    )


def _slot_c_trunks(
    cgen_t: dict[Any, Any],
    flip: bool,
    sx_slot: int,
    kl: int,
    kr: int | None,
    hl: bool,
    hr: bool,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Return ``([(k, cx), ...], [sel_tx, ...])`` without adding cells."""
    if not hl and not hr:
        return [], []
    if kr is None:
        if kl == 15:
            pad = cgen_t["inv_left_pad"]
            pi = cgen_t[("inv", flip)][1]
            tr = dict(pi.trunks)
            return [(15, sx_slot + pad + tr["C"])], [sx_slot + pad + tr["SELB"]]
        pr = cgen_t[("nr", flip)][1]
        tr = dict(pr.trunks)
        return [(kl, sx_slot + 54 + tr["CR"])], [sx_slot + 54 + tr["SELB"]]
    if hl and hr:
        pp = cgen_t[("pair", flip)][1]
        tr = dict(pp.trunks)
        return (
            [(kl, sx_slot + tr["CL"]), (kr, sx_slot + tr["CR"])],
            [sx_slot + tr["SELB"]],
        )
    if hl:
        pl = cgen_t[("nl", flip)][1]
        tr = dict(pl.trunks)
        return [(kl, sx_slot + tr["CL"])], [sx_slot + tr["SELB"]]
    pr = cgen_t[("nr", flip)][1]
    tr = dict(pr.trunks)
    return [(kr, sx_slot + 162 + tr["CR"])], [sx_slot + 162 + tr["SELB"]]


def build_pruned_selector_array(
    plan: SelectorPlan,
    mapping: tuple[tuple[int, ...], ...],
    per_block_k2t: Sequence[dict[int, int]] | None = None,
    *,
    lib: Any = None,
) -> Any:
    """Build selector array with ``k = 0`` decoupled from ``BL`` and 2D-trimmed wires."""
    if not mapping or not mapping[0] or any(len(r) != len(mapping[0]) for r in mapping):
        raise ValueError("mapping must be a nonempty rectangle")
    if any(type(k) is not int or not 0 <= k < 16 for row in mapping for k in row):
        raise ValueError("candidate indices must be integers in 0..15")
    if lib is not None and lib.unit != 1e-9:
        raise ValueError("GDS coordinates are nanometres")
    if per_block_k2t is None:
        per_block_k2t = _cgen_slot_plan(
            mapping, plan.fins, lane_vt=plan.vt
        ).per_block_k2t

    gdspy = require_gdspy()
    digest = sha256(repr((plan, mapping, tuple(per_block_k2t))).encode()).hexdigest()[:12]
    top = gdspy.Cell(f"brom_pruned_sel_{digest}", exclude_from_current=True)
    if lib is not None:
        lib.add(top)
    blocks, columns = len(mapping), len(mapping[0])
    w, h = columns * plan.width, blocks * plan.height
    if h + 81 > 30_000:
        raise ValueError("array exceeds the single tap row latch-up reach")

    direct_m2 = set(_m2_track_roles(plan.fins)[0])

    box(top, "NSELECT", -54, -27, w + 54, h)
    box(top, "PSELECT", -54, h, w + 54, h + 108)
    if vt_layer := VT_LAYERS[plan.vt]:
        box(top, vt_layer, -54, -27, w + 54, h + 108)
    box(top, "BOUNDARY", -54, -27, w + 54, h + 108)
    for y in range(10, h + 81, 27):
        box(top, "FIN", -54, y, w + 54, y + 7)
    for x in range(-27, w + 28, 54):
        box(top, "GATE", x - 10, -5, x + 10, h + 86)
    for y in [*(b * plan.height for b in range(blocks + 1)), h + 81]:
        box(top, "GATE_CUT", -54, y - 22, w + 54, y + 22)
    for col in range(columns + 1):
        x = col * plan.width
        box(top, "M1", x - 9, -14, x + 9, h + 95)
        box(top, "ACTIVE", x - 8, h + 27, x + 8, h + 81)
        for metal in ("SDT", "LISD"):
            box(top, metal, x - 12, h + 27, x + 12, h + 81)
        square(top, "V0", x, h + 54)
        square(top, "V1", x, h + 81)
    box(top, "M2", -24, h + 72, w + 24, h + 90)
    for col in range(columns):
        x = col * plan.width + 81
        max_b = max((b for b in range(blocks) if mapping[b][col] != 0), default=-1)
        bl_top = (
            64
            if max_b < 0
            else min(h, max(64, max_b * plan.height + plan.drain_y + 28))
        )
        box(top, "M3", x - 9, 0, x + 9, bl_top)

    for b, row in enumerate(mapping):
        oy = b * plan.height
        alo, ahi = oy + 27, oy + 27 * (plan.fins + 1)
        box(top, "ACTIVE", -8, alo, w + 8, ahi)
        for x in range(0, w + 1, 54):
            for metal in ("SDT", "LISD"):
                box(top, metal, x - 12, alo, x + 12, ahi)
            if x % 108 == 0:
                square(top, "V0", x, oy + plan.drain_y)

        k2t = per_block_k2t[b]
        for k, t in k2t.items():
            cols_k = [col for col, rk in enumerate(row) if rk == k]
            metal, y = plan.tracks[t]
            half = 9 if metal == "M2" else 12
            if metal == "M4" or t in direct_m2:
                x_lo = -48
                x_hi = max(cols_k) * plan.width + 48
            else:
                x_lo = min(cols_k) * plan.width
                x_hi = w + 48
            box(top, metal, x_lo, oy + y - half, x_hi, oy + y + half)

        for col, k in enumerate(row):
            ox = col * plan.width
            dy, gy = oy + plan.drain_y, oy + plan.gate_y
            iy = gy + 54
            box(top, "LIG", ox + 70, iy - 11, ox + 108, iy + 11)
            square(top, "V0", ox + 99, iy)
            box(top, "M1", ox + 85, iy - 9, ox + 117, iy + 9)
            if k == 0:
                # Ground live gate and middle diffusion to VSS M1/M2 without
                # touching BL<col> on M3 (eliminates both BL cap and floating LVS islands).
                box(top, "LIG", ox, gy - 11, ox + 38, gy + 11)
                square(top, "V0", ox + 9, gy)
                box(top, "M1", ox - 9, gy - 9, ox + 23, gy + 9)
                square(top, "V0", ox + 54, dy)
                box(top, "M1", ox + 45, oy - 14, ox + 63, max(ahi, dy + 14))
                square(top, "V1", ox, oy)
                square(top, "V1", ox + 54, oy)
                box(top, "M2", ox - 19, oy - 9, ox + 127, oy + 9)
            else:
                square(top, "V0", ox + 54, dy)
                box(top, "M1", ox + 45, min(alo, dy - 14), ox + 63, max(ahi, dy + 14))
                square(top, "V1", ox + 54, dy)
                box(top, "M2", ox + 37, dy - 9, ox + 98, dy + 9)
                square(top, "V2", ox + 81, dy)
                t = k2t[k]
                metal, y = plan.tracks[t]
                box(top, "LIG", ox + 16, gy - 11, ox + 54, gy + 11)
                square(top, "V0", ox + 45, gy)
                box(top, "M1", ox + 36, gy - 14, ox + 54, gy + 14)
                square(top, "V1", ox + 45, gy)
                box(top, "M2", ox + 10, gy - 9, ox + 62, gy + 9)
                square(top, "V2", ox + 27, gy)
                lo, hi = min(plan.gate_y, y) - 17, max(plan.gate_y, y) + 17
                box(top, "M3", ox + 18, oy + lo, ox + 36, oy + hi)
                if metal == "M2":
                    square(top, "V2", ox + 27, oy + y)
                else:
                    box(top, "V3", ox + 18, oy + y - 12, ox + 36, oy + y + 12)
    return top


def _m4_seg(cell: Any, a: float, b: float, y: float) -> None:
    lo = (min(a, b) - 24) // 48 * 48
    hi = ceil_grid(max(a, b) + 24, 48)
    box(cell, "M4", lo, y - 12, hi, y + 12)


def pruned_macro_circuit(
    spec: BromSpec,
    words: Sequence[int],
    *,
    periphery_vt: str | None = None,
    crit_vt: str | None = None,
) -> Circuit:
    """Build a data-pruned high-PPA BROM ``Circuit`` for fixed word contents.

    ``periphery_vt`` defaults to ``spec.vt`` and sets the threshold flavor for
    precharge-settled off-critical logic and the CGEN lane.  ``crit_vt``
    defaults to ``periphery_vt`` and sets the threshold flavor for the critical
    ``EVAL -> SELB -> BL -> DOUT -> Q`` path (e.g. ``spec.vt="sram"`` with
    ``crit_vt="rvt"`` or ``"lvt"`` for low standby leakage without sacrificing
    read access time).
    """
    if spec.registered_output:
        raise ValueError(
            "pruned_macro_circuit uses external read phases (PRECHN/EVAL/SAMPLE)"
        )
    off_vt = spec.vt if periphery_vt is None else periphery_vt
    fast_vt = off_vt if crit_vt is None else crit_vt
    for vt_opt in (off_vt, fast_vt):
        if vt_opt not in VT_LAYERS:
            raise ValueError(f"unknown vt {vt_opt!r}; expected one of {tuple(VT_LAYERS)}")

    mapping = selector_map(spec, words)
    digest = sha256(repr((spec, off_vt, fast_vt, mapping)).encode()).hexdigest()[:12]
    vt_tag = (
        spec.vt
        if (off_vt, fast_vt) == (spec.vt, spec.vt)
        else f"{spec.vt}_p{off_vt}_c{fast_vt}"
    )
    c = Circuit(
        f"brom_pruned_{spec.depth}x{spec.bits}_m{spec.ways}_f{spec.fins}_{vt_tag}_{digest}",
        spec.pins,
        spec.fins,
        spec.vt,
    )
    if spec.keeper_fins:
        if spec.ways != 1:
            raise ValueError(
                "pruned ways=4 uses gated tristate sense without bitline keeper"
            )
        c.name += f"_k{spec.keeper_fins}s{spec.keeper_stack}"

    cgen_pfins = 2 if off_vt == "sram" else 3
    slot_plan = _cgen_slot_plan(
        mapping, spec.fins, lane_vt=off_vt, cgen_pfins=cgen_pfins, cgen_nfins=1
    )
    needed_gb = [k for k in slot_plan.active_k if k != 15]
    ab = spec.address_bits

    # 1. Input address latches (settle during precharge)
    addr_open = c.node()
    _add_sized_inv(
        c,
        "prechn_inv",
        "PRECHN",
        addr_open,
        2,
        2,
        fingers=1 if ab <= 8 else 2,
        vt=off_vt,
    )
    for i in range(ab):
        _add_addr_latch(
            c, i, f"A[{i}]", "PRECHN", addr_open, f"ADDR<{i}>", fins=1, vt=off_vt
        )

    # 2. Column way decoder (ways=4), active-low GB<k> generator, and block BPRE<b>
    shift = spec.column_bits
    if spec.ways == 4:
        _decode_active_high(
            c, ["ADDR<0>", "ADDR<1>"], "CSEL_RAW", {0, 1, 2, 3}, vt=off_vt
        )
        csel_fingers = max(1, (spec.bits + 11) // 12)
        for w in range(4):
            if csel_fingers == 1:
                _add_sized_inv(
                    c, f"cselb_drv_{w}", f"CSEL_RAW<{w}>", f"CSELB<{w}>", 2, 1, vt=off_vt
                )
                _add_sized_inv(
                    c, f"csel_drv_{w}", f"CSELB<{w}>", f"CSEL<{w}>", 2, 1, vt=off_vt
                )
            else:
                _add_sized_inv(
                    c,
                    f"cselb_drv_{w}",
                    f"CSEL_RAW<{w}>",
                    f"CSELB<{w}>",
                    2,
                    2,
                    fingers=csel_fingers,
                    vt=off_vt,
                )
                _add_sized_inv(
                    c,
                    f"csel_drv_{w}",
                    f"CSELB<{w}>",
                    f"CSEL<{w}>",
                    2,
                    2,
                    fingers=csel_fingers,
                    vt=off_vt,
                )

    _add_gb_generator(c, f"ADDR<{shift}>", f"ADDR<{shift + 1}>", needed_gb, vt=off_vt)

    active_blocks = [b for b in range(spec.blocks) if slot_plan.per_block_k2t[b]]
    built_blocks = active_blocks if active_blocks else [0]

    # 3. 1-stage 2-NMOS SELB<b> drivers on EVAL rising path (only for built_blocks)
    if spec.blocks == 1:
        _add_nand2(c, "selb_0", "EVAL", "PRECHN", "SELB<0>", 2, 3, vt=fast_vt)
    else:
        row_addr = [f"ADDR<{i}>" for i in range(shift + 2, ab)]
        eval_sig = "EVAL"
        if len(built_blocks) > 12:
            ev_b = c.node()
            eval_sig = c.node()
            _add_sized_inv(c, "eval_inv0", "EVAL", ev_b, 3, 3, vt=fast_vt)
            _add_sized_inv(c, "eval_inv1", ev_b, eval_sig, 3, 3, fingers=2, vt=fast_vt)
        _decode_block_selb(
            c, row_addr, built_blocks, eval_sig, off_vt=off_vt, fast_vt=fast_vt
        )

    # 4. Sense + Column Mux + Output Latches (direct SAMPLE for <=32b, chunked above)
    pre_fins = 1 if spec.blocks <= 32 else 2
    way_pfins = 2 if (off_vt == "sram" and fast_vt != "lvt") else 3
    chunk_size = 16
    n_chunks = (spec.bits + chunk_size - 1) // chunk_size
    direct_sample = spec.bits <= 32
    for ch in range(n_chunks):
        sb_net = f"SAMPLEB_{ch}"
        if direct_sample:
            _add_sized_inv(
                c,
                f"samp_inv0_{ch}",
                "SAMPLE",
                sb_net,
                3,
                3 if ch % 2 == 0 else 2,
                vt=fast_vt,
            )
            s_net = "SAMPLE"
        else:
            s_net = f"SAMPLE_EN_{ch}"
            _add_sized_inv(c, f"samp_inv0_{ch}", "SAMPLE", sb_net, 3, 3, vt=fast_vt)
            _add_sized_inv(c, f"samp_inv1_{ch}", sb_net, s_net, 3, 3, vt=fast_vt)
        lo_bit = ch * chunk_size
        hi_bit = min(spec.bits, lo_bit + chunk_size)
        for bit in range(lo_bit, hi_bit):
            dout = f"DOUT<{bit}>"
            if spec.ways == 1:
                _add_sense_m1(
                    c,
                    bit,
                    f"BL<{bit}>",
                    "PRECHN",
                    dout,
                    pre_fins=pre_fins,
                    keeper_fins=spec.keeper_fins,
                    keeper_stack=spec.keeper_stack,
                    vt=fast_vt,
                )
            else:
                for way in range(4):
                    col = bit * 4 + way
                    _add_sense_way(
                        c,
                        bit,
                        way,
                        f"BL<{col}>",
                        "PRECHN",
                        f"CSEL<{way}>",
                        f"CSELB<{way}>",
                        dout,
                        pre_fins=pre_fins,
                        sense_pfins=way_pfins,
                        vt=fast_vt,
                    )
            _add_fast_out_latch(c, bit, dout, s_net, sb_net, f"Q[{bit}]", vt=fast_vt)

    # 5. CGEN lane devices (only active (b, k) pairs; zero dummy transistors)
    cgen_device_names: set[str] = set()
    for b in range(spec.blocks):
        k2t = slot_plan.per_block_k2t[b]
        for kl, kr in slot_plan.slots:
            for k in (kl, kr):
                if k is None or k not in k2t:
                    continue
                start = len(c.devices)
                cname = f"C<{b},{k}>"
                if k == 15:
                    _fet(c, True, cname, f"SELB<{b}>", "VDD", cgen_pfins, off_vt)
                    _fet(c, False, cname, f"SELB<{b}>", "VSS", 1, off_vt)
                else:
                    mid = c.node()
                    _fet(c, True, mid, f"GB<{k}>", "VDD", cgen_pfins, off_vt)
                    _fet(c, True, cname, f"SELB<{b}>", mid, cgen_pfins, off_vt)
                    _fet(c, False, cname, f"SELB<{b}>", "VSS", 1, off_vt)
                    _fet(c, False, cname, f"GB<{k}>", "VSS", 1, off_vt)
                cgen_device_names.update(d.name for d in c.devices[start:])
                c.group_since(f"pruned_cgen_{b}_{k}", start)

    # 6. Selector array devices (spec.fins, spec.vt)
    array_devices: set[str] = set()
    for b, row in enumerate(mapping):
        for col, k in enumerate(row):
            if k == 0:
                _fet(c, False, "VSS", "VSS", "VSS", spec.fins, spec.vt)
                array_devices.add(c.devices[-1].name)
                _fet(c, False, "VSS", "VSS", "VSS", spec.fins, spec.vt)
                array_devices.add(c.devices[-1].name)
            else:
                _fet(c, False, f"BL<{col}>", f"C<{b},{k}>", "VSS", spec.fins, spec.vt)
                array_devices.add(c.devices[-1].name)
                _fet(c, False, f"BL<{col}>", "VSS", "VSS", spec.fins, spec.vt)
                array_devices.add(c.devices[-1].name)

    c.array = SelectorInstance(
        default_selector_plan(spec.fins, spec.vt), mapping, frozenset(array_devices)
    )
    c.pruned_spec = spec  # type: ignore[attr-defined]
    c.pruned_cgen_devices = frozenset(cgen_device_names)  # type: ignore[attr-defined]
    c.pruned_plan = slot_plan  # type: ignore[attr-defined]
    c.pruned_vts = (off_vt, fast_vt)  # type: ignore[attr-defined]
    return c


def _build_pruned_macro(c: Circuit) -> tuple[Any, dict[str, Any]]:
    spec: BromSpec = c.pruned_spec  # type: ignore[attr-defined]
    assert c.array is not None
    mapping = c.array.mapping
    slot_plan: CgenSlotPlan = c.pruned_plan  # type: ignore[attr-defined]
    off_vt, fast_vt = getattr(c, "pruned_vts", (spec.vt, spec.vt))
    per_block_k2t = slot_plan.per_block_k2t
    max_upper = slot_plan.max_upper
    gdspy = require_gdspy()

    gb_dx = dict(slot_plan.gb_offsets)
    needed_gb = [k for k, _ in slot_plan.gb_offsets]
    max_block_of_gb: dict[int, int] = {}
    for b, k2t in enumerate(per_block_k2t):
        for k in k2t:
            if k != 15:
                max_block_of_gb[k] = b

    active_blocks = [b for b in range(spec.blocks) if per_block_k2t[b]]
    sel_x0 = 60
    sel_x_map = {b: sel_x0 + 48 * idx for idx, b in enumerate(active_blocks)}
    sel_chan = ceil_grid(sel_x0 + 48 * max(1, len(active_blocks)) + 36, 108)
    cgen_origin = sel_chan
    cgen_cells_x0 = cgen_origin + 108
    cgen_support_x0 = cgen_cells_x0 + slot_plan.total_width
    lane_end = cgen_support_x0 + 324
    array_x = lane_end + 108
    array_w = spec.columns * 108
    array_end = array_x + array_w
    right_drops_end = (
        array_end + 48
        if max_upper == 0
        else ceil_grid(array_end + 120, 48) + 12 + 96 * (max_upper - 1) + 48
    )
    vdd_out_x = ceil_grid(right_drops_end + 36 - 12, 48) + 12
    vss_out_x = vdd_out_x + 96
    top_min_w = ceil_grid(vss_out_x + 84, 432)

    excluded = set(c.array.device_names) | set(c.pruned_cgen_devices)  # type: ignore[attr-defined]
    bank_devices = [d for d in c.devices if d.name not in excluded]
    used_in_bank = {n for d in bank_devices for n in d.nodes}
    top_section_nets = (
        [f"GB<{k}>" for k in needed_gb if k in max_block_of_gb]
        + [f"SELB<{b}>" for b in active_blocks]
        + [f"BL<{col}>" for col in range(spec.columns)]
    )
    boundary_nets = [n for n in top_section_nets if n in used_in_bank]
    bank_vt = off_vt if off_vt == fast_vt else "rvt"
    bank_c = Circuit(
        c.name + "_bank",
        tuple(dict.fromkeys([*c.pins, *boundary_nets])),
        c.n.fins,
        bank_vt,
    )
    bank_c.devices = bank_devices
    bank_c.logic_units = [
        (kind, members)
        for kind, members in c.logic_units
        if all(m not in excluded for m in members)
    ]
    bank_c.placement_groups = [
        (role, members)
        for role, members in c.placement_groups
        if not role.startswith("pruned_cgen_")
    ]

    extracted = dense_units(bank_c)
    total_unit_w = sum(u.plan.width + 54 for u in extracted)
    min_pin_w = ceil_grid((max(spec.columns, len(c.pins) + 4) + 10) * 48, 432)
    target_aspect_w = ceil_grid(int((total_unit_w * 432 * 1.35) ** 0.5), 432)
    base_macro_w = max(top_min_w, min_pin_w, target_aspect_w, 3456)

    cgen_t = _build_cgen_templates(
        spec.fins, slot_plan.lane_vt, slot_plan.cgen_pfins, slot_plan.cgen_nfins
    )
    direct_tracks, upper_m2 = _m2_track_roles(spec.fins)
    direct_m2 = set(direct_tracks)
    upper_index = {t: idx for idx, t in enumerate(upper_m2)}

    def header_spans(_bw: int) -> dict[str, list[int]]:
        spans: dict[str, list[int]] = defaultdict(list)
        for k in needed_gb:
            if k in max_block_of_gb:
                spans[f"GB<{k}>"].append(cgen_cells_x0 + gb_dx[k])
        for b, sx in sel_x_map.items():
            spans[f"SELB<{b}>"].append(sx)
        for col in range(spec.columns):
            spans[f"BL<{col}>"].append(array_x + 108 * col + 81)
        spans["VDD"].append(vdd_out_x)
        spans["VSS"].append(vss_out_x)
        return spans

    def array_base(header_top: int, height: int) -> int:
        return ceil_grid(max(header_top + 228, height), 432)

    anchors: dict[str, tuple[int, int, str | None]] = {}
    for col in range(spec.columns):
        bx = array_x + 108 * col + 81
        anchors[f"BL<{col}>"] = (bx - 24, bx + 24, None)
    for b, sx in sel_x_map.items():
        anchors[f"SELB<{b}>"] = (sx, sx, None)
    for k in needed_gb:
        if k in max_block_of_gb:
            gx = cgen_cells_x0 + gb_dx[k]
            anchors[f"GB<{k}>"] = (gx - 24, gx + 24, None)

    cgen_m5_cols: set[int] = set()
    if max_upper > 0:
        for b, k2t in enumerate(per_block_k2t):
            flip = bool(b % 2)
            for s, (kl, kr) in enumerate(slot_plan.slots):
                sx_slot = cgen_cells_x0 + slot_plan.slot_offsets[s]
                outs, _ = _slot_c_trunks(
                    cgen_t, flip, sx_slot, kl, kr, kl in k2t, kr is not None and kr in k2t
                )
                for k_out, cx in outs:
                    if k2t[k_out] in upper_index:
                        cgen_m5_cols.add(ceil_grid(cx - 12, 48) + 12)
    drop_m5_cols = [
        ceil_grid(array_end + 120, 48) + 12 + 96 * j for j in range(max_upper)
    ]
    blocked_m5: set[int] = set()
    for mx in (*sorted(cgen_m5_cols), *drop_m5_cols):
        blocked_m5.update((mx - 48, mx, mx + 48))

    bank = None
    macro_w = base_macro_w
    for extra in (0, 432, 864, 1296, 1728, 2592, 3888, 5184):
        try_w = base_macro_w + extra
        try:
            bank = build_dense_bank(
                bank_c,
                units=extracted,
                width=try_w,
                pin_width=try_w,
                header_spans=header_spans,
                external=c.pins,
                cost=array_base,
                anchors=lambda _bw, a=anchors: dict(a),
                bottom=c.pins,
                blocked=blocked_m5,
            )
            macro_w = max(try_w, bank.width)
            break
        except ValueError:
            continue
    if bank is None:
        raise RuntimeError("failed to place/route bottom peripheral bank")

    top = gdspy.Cell(c.name, exclude_from_current=True)
    bank.cell.labels.clear()
    top.add(gdspy.CellReference(bank.cell))

    base = array_base(
        max((y for y, _, _ in bank.header.values()), default=0), bank.height
    )
    row_top = max(
        (
            u["origin_nm"][1] if u["mirror_y"] else u["origin_nm"][1] + 432
            for u in bank.units
        ),
        default=432,
    )
    routes = list(bank.routes)
    checkpoints: list[tuple[str, str, float, float]] = [
        (net, metal, x, y) for net, (metal, x, y) in bank.pins.items()
    ]

    def rise(net: str, x: int, y: int) -> None:
        cx = bank.anchored[net]
        v3(top, x, y)
        _m4_seg(top, x, cx, y)
        v4(top, cx, y)
        box(top, "M5", cx - 12, row_top - 24, cx + 12, y + 24)
        routes.extend(
            [
                (net, "V3", (x, y), (x, y)),
                (net, "M4", (min(x, cx), y), (max(x, cx), y)),
                (net, "V4", (cx, y), (cx, y)),
                (net, "M5", (cx, row_top - 24), (cx, y)),
            ]
        )

    def join(
        net: str, x: int, y: int, *, source: str = "M3", local_x: int | None = None
    ) -> None:
        if net in bank.anchored:
            rise(net, x, y)
            return
        bx = x if local_x is None else local_x
        by, _left_end, _right_end = bank.header[net]
        if bx == x and source == "M3":
            v3(top, bx, by)
            box(top, "M3", bx - 9, min(y, by) - 17, bx + 9, max(y, by) + 17)
            routes.extend(
                [
                    (net, "V3", (bx, by), (bx, by)),
                    (net, "M3", (bx, min(y, by)), (bx, max(y, by))),
                ]
            )
            return
        if source == "M3":
            v3(top, x, y)
            routes.append((net, "V3", (x, y), (x, y)))
        elif source == "M2":
            square(top, "V2", x, y)
            routes.append((net, "V2", (x, y), (x, y)))
        _m4_seg(top, x, bx, y)
        v3(top, bx, y)
        v3(top, bx, by)
        box(top, "M3", bx - 9, min(y, by) - 17, bx + 9, max(y, by) + 17)
        routes.extend(
            [
                (net, "M4", (min(x, bx), y), (max(x, bx), y)),
                (net, "V3", (bx, y), (bx, y)),
                (net, "V3", (bx, by), (bx, by)),
                (net, "M3", (bx, min(y, by)), (bx, max(y, by))),
            ]
        )

    sel_plan = c.array.plan
    sel_cell = build_pruned_selector_array(sel_plan, mapping, per_block_k2t)
    top.add(gdspy.CellReference(sel_cell, (array_x, base)))

    inv_pad = cgen_t["inv_left_pad"]
    for b in range(spec.blocks):
        flip = bool(b % 2)
        oy = base + b * 432
        ry = oy + 432 if flip else oy
        k2t = per_block_k2t[b]

        top.add(
            gdspy.CellReference(cgen_t["filler"], (cgen_origin, ry), x_reflection=flip)
        )

        used_sel_xs: list[int] = []
        for s, (kl, kr) in enumerate(slot_plan.slots):
            sx_slot = cgen_cells_x0 + slot_plan.slot_offsets[s]
            sw = slot_plan.slot_widths[s]
            hl = kl in k2t
            hr = kr is not None and kr in k2t

            if not hl and not hr:
                for dx in range(0, sw, 108):
                    top.add(
                        gdspy.CellReference(
                            cgen_t["filler"], (sx_slot + dx, ry), x_reflection=flip
                        )
                    )
            elif kr is None:
                if kl == 15:
                    ui, _ = cgen_t[("inv", flip)]
                    if inv_pad == 54:
                        top.add(
                            gdspy.CellReference(
                                cgen_t["spacer"], (sx_slot, ry), x_reflection=flip
                            )
                        )
                        top.add(
                            gdspy.CellReference(
                                ui, (sx_slot + 54, ry), x_reflection=flip
                            )
                        )
                    else:
                        top.add(
                            gdspy.CellReference(ui, (sx_slot, ry), x_reflection=flip)
                        )
                        top.add(
                            gdspy.CellReference(
                                cgen_t["spacer"], (sx_slot + 162, ry), x_reflection=flip
                            )
                        )
                else:
                    ur, _ = cgen_t[("nr", flip)]
                    top.add(
                        gdspy.CellReference(
                            cgen_t["spacer"], (sx_slot, ry), x_reflection=flip
                        )
                    )
                    top.add(
                        gdspy.CellReference(ur, (sx_slot + 54, ry), x_reflection=flip)
                    )
            elif hl and hr:
                up_cell, _ = cgen_t[("pair", flip)]
                top.add(
                    gdspy.CellReference(up_cell, (sx_slot, ry), x_reflection=flip)
                )
            elif hl:
                ul, _ = cgen_t[("nl", flip)]
                top.add(gdspy.CellReference(ul, (sx_slot, ry), x_reflection=flip))
                top.add(
                    gdspy.CellReference(
                        cgen_t["filler"], (sx_slot + 216, ry), x_reflection=flip
                    )
                )
            else:
                ur, _ = cgen_t[("nr", flip)]
                top.add(
                    gdspy.CellReference(
                        cgen_t["filler"], (sx_slot, ry), x_reflection=flip
                    )
                )
                top.add(
                    gdspy.CellReference(
                        cgen_t["spacer"], (sx_slot + 108, ry), x_reflection=flip
                    )
                )
                top.add(
                    gdspy.CellReference(ur, (sx_slot + 162, ry), x_reflection=flip)
                )

            outs, sel_txs = _slot_c_trunks(cgen_t, flip, sx_slot, kl, kr, hl, hr)
            for sel_tx in sel_txs:
                used_sel_xs.append(sel_tx)
                square(top, "V2", sel_tx, oy + 18)

            for k_out, cx in outs:
                t = k2t[k_out]
                metal, ty_rel = sel_plan.tracks[t]
                ty = oy + ty_rel
                cname = f"C<{b},{k_out}>"
                if metal == "M4":
                    v3(top, cx, ty)
                    box(top, "M4", cx - 24, ty - 12, array_x, ty + 12)
                    checkpoints.append((cname, "M4", array_x - 12, ty))
                elif t in direct_m2:
                    square(top, "V2", cx, ty)
                    box(top, "M2", cx - 19, ty - 9, array_x, ty + 9)
                    checkpoints.append((cname, "M2", array_x - 12, ty))
                else:
                    j = upper_index[t]
                    y6 = 16 + ceil_grid(oy, 64) + 64 * j
                    y4 = oy + 60
                    sx6 = ceil_grid(cx - 12, 48) + 12
                    rx = ceil_grid(array_end + 120, 48) + 12 + 96 * j
                    lift6(top, cx, y4, sx6, y6)
                    box(top, "M6", sx6 - 24, y6 - 16, rx + 24, y6 + 16)
                    lift6(top, rx, y4, rx, y6)
                    box(top, "M3", rx - 9, min(ty, y4) - 17, rx + 9, max(ty, y4) + 17)
                    square(top, "V2", rx, ty)
                    box(top, "M2", array_end, ty - 9, rx + 19, ty + 9)
                    checkpoints.append((cname, "M2", array_end + 12, ty))

            for k_cand in (kl, kr):
                if (
                    k_cand is not None
                    and k_cand != 15
                    and k_cand in max_block_of_gb
                    and b <= max_block_of_gb[k_cand]
                ):
                    gx = cgen_cells_x0 + gb_dx[k_cand]
                    y_hi = oy + 446 if b < max_block_of_gb[k_cand] else oy + 416
                    box(top, "M3", gx - 9, oy - 14, gx + 9, y_hi)

        for i, supp in enumerate(("filler", "tap", "filler")):
            top.add(
                gdspy.CellReference(
                    cgen_t[supp], (cgen_support_x0 + 108 * i, ry), x_reflection=flip
                )
            )

        if used_sel_xs:
            sy = oy + 18
            sx = sel_x_map[b]
            s_idx = active_blocks.index(b)
            max_sel_x = max(used_sel_xs)
            box(top, "M2", sx - 19, sy - 9, max_sel_x + 19, sy + 9)
            square(top, "V2", sx, sy)
            py = (
                oy + 60
                if (b and f"SELB<{b}>" in bank.anchored)
                else base - 36 - 96 * (s_idx % 3)
            )
            box(top, "M3", sx - 9, min(py, sy) - 17, sx + 9, max(py, sy) + 17)
            join(f"SELB<{b}>", sx, py, local_x=sx)
            checkpoints.append((f"SELB<{b}>", "M2", max_sel_x, sy))

        px = lane_end - 54
        for net, out_x in (("VDD", vdd_out_x), ("VSS", vss_out_x)):
            rail_y = oy + (0 if (net == "VDD") == flip else 432)
            square(top, "V1", px, rail_y)
            box(top, "M2", px - 19, rail_y - 9, px + 19, rail_y + 9)
            square(top, "V2", px, rail_y)
            py = rail_y + 12
            box(top, "M3", px - 9, rail_y - 17, px + 9, py + 17)
            join(net, px, py, local_x=out_x)
            checkpoints.append((net, "M1", px, rail_y))

    active_gb = [
        (k, dx) for k, dx in slot_plan.gb_offsets if k in max_block_of_gb
    ]
    for idx, (k, dx) in enumerate(active_gb):
        gx = cgen_cells_x0 + dx
        py = base - 36 - 96 * (idx % 2)
        box(top, "M3", gx - 9, py - 17, gx + 9, base + 14)
        join(f"GB<{k}>", gx, py)
        checkpoints.append((f"GB<{k}>", "M3", gx, base + 200))

    for col in range(spec.columns):
        bx = array_x + 108 * col + 81
        py = base - 36 - 96 * (col % 2)
        box(top, "M3", bx - 9, py - 17, bx + 9, base + 14)
        join(f"BL<{col}>", bx, py)
        checkpoints.append((f"BL<{col}>", "M3", bx, base + 32))

    sy_tap = base + spec.blocks * 432 + 81
    box(
        top,
        "M2",
        min(array_x, vss_out_x) - 19,
        sy_tap - 9,
        max(array_x, vss_out_x) + 19,
        sy_tap + 9,
    )
    square(top, "V2", vss_out_x, sy_tap)
    py_tap = base + spec.blocks * 432 + 108
    box(
        top,
        "M3",
        vss_out_x - 9,
        min(py_tap, sy_tap) - 17,
        vss_out_x + 9,
        max(py_tap, sy_tap) + 17,
    )
    join("VSS", vss_out_x, py_tap, local_x=vss_out_x)
    checkpoints.append(("VSS", "M2", array_x, sy_tap))

    ports: dict[str, tuple[str, float, float]] = {}
    for pin in c.pins:
        metal, px_pin, py_pin = bank.pins[pin]
        label(top, pin, metal, (px_pin, py_pin))
        ports[pin] = (metal, px_pin, py_pin)

    bounds = top.get_bounding_box()
    top_h = ceil_grid(max(base + spec.blocks * 432 + 160, float(bounds[1, 1])), 64)
    final_w = max(macro_w, top_min_w, ceil_grid(float(bounds[1, 0]), 48))
    x0 = min(0.0, float(bounds[0, 0]))
    y0 = min(0.0, float(bounds[0, 1]))
    box(top, "BOUNDARY", x0, y0, final_w, top_h)

    return top, {
        "certificates": list(bank.certificates),
        "routing_certificates": list(bank.routing_certificates),
        "joint_optimization": bank.joint_optimization,
        "ports": ports,
        "arrays": [
            {
                "origin_nm": (array_x, base),
                "width_nm": array_w,
                "height_nm": spec.blocks * 432,
                "active_candidates": list(slot_plan.active_k),
                "active_blocks": active_blocks,
            }
        ],
        "bottom_units": bank.units,
        "routes": routes,
        "checkpoints": checkpoints,
        "array_y_nm": base,
        "cgen_slots": len(slot_plan.slots),
        "cgen_width_nm": slot_plan.total_width,
        "periphery_vt": off_vt,
        "crit_vt": fast_vt,
        "timing_qualified": False,
    }


def pruned_macro_layout(c: Circuit) -> tuple[Any, dict[str, Any]]:
    """Cached physical build and route-plan metadata for a pruned BROM circuit."""
    from .dense_macro import canonicalize_cells

    if c.array is None or getattr(c, "pruned_spec", None) is None:
        raise ValueError("pruned_macro_layout requires a pruned_macro_circuit")
    key = sha256(
        repr(
            (
                c.devices,
                c.pins,
                c.placement_groups,
                c.pruned_spec,  # type: ignore[attr-defined]
                getattr(c, "pruned_vts", None),
                c.array.plan,
                c.array.mapping,
                sorted(c.array.device_names),
            )
        ).encode()
    ).hexdigest()
    cached = getattr(c, "_dense_layout", None)
    if cached is None or cached[0] != key:
        top, meta = _build_pruned_macro(c)
        canonicalize_cells(top)
        certificates = meta.pop("certificates")
        routing_certificates = meta.pop("routing_certificates")
        joint_optimization = meta.pop("joint_optimization")
        meta["schema"] = "brom_pruned_physical_v2"
        geometry = sha256()
        for layer, polygons in sorted(top.get_polygons(by_spec=True).items()):
            geometry.update(repr(layer).encode())
            for polygon in polygons:
                geometry.update(polygon.tobytes())
        meta["geometry_sha256"] = geometry.hexdigest()
        meta["physical_sha256"] = sha256(repr((key, meta)).encode()).hexdigest()
        meta["placement_certificates"] = certificates
        meta["routing_certificates"] = routing_certificates
        meta["joint_optimization"] = joint_optimization
        c._dense_layout = (key, top, meta)  # type: ignore[attr-defined]
    _, top, meta = c._dense_layout  # type: ignore[attr-defined]
    result = deepcopy(top)
    result.name = c.name
    return result, meta


def build_pruned_macro(circuit: Circuit, *, lib: Any = None) -> Any:
    """Build a pruned BROM GDS cell with hierarchical metal route verification."""
    from .dense_macro import verify_dense_routes

    top, plan = pruned_macro_layout(circuit)
    verify_dense_routes(top, plan["checkpoints"])
    if lib is not None:
        if lib.unit != 1e-9:
            raise ValueError("BROM layout requires lib.unit=1e-9")
        lib.add(top)
    return top
