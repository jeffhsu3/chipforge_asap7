"""Data-pruned single-sided CGEN and decoupled-bitline BROM macro compiler.

Unlike ``macro_circuit`` / ``build_dense_macro`` (which preserve a fixed base
abstract so word reprogramming only modifies V2 and V3 cuts), this flow targets
custom-mask hard macros whose ROM contents are fixed before tapeout:

1. **Data-pruned 1-stage NOR2/INV CGEN**: instantiates candidate drivers only
   for active ``(block, k)`` pairs (``k != 0``) in compact 216 nm single-sided
   slots, assigning block tracks to left-entering ``M4`` (``8..15``) and direct
   ``M2`` (``{0/1, 6}``) before using right-dropped ``M6`` lifts.
2. **Decoupled ``k = 0`` bitcells & trimmed candidate tracks**: grounds ``k = 0``
   selector gates and middle diffusions locally to ``VSS`` ``M1`` without
   bridging ``V1/M2/V2`` onto ``BL<col>``, and trims horizontal ``M2/M4``
   candidate tracks to the active column span in each block.
3. **2-gate critical ``EVAL`` path**: precomputes active-low ``GB<1..14>`` and
   row ``BLOCK<b>`` signals during precharge (with singleton 1-fin off-critical
   sizing for low leakage and fast unit-fin LVS color refinement), leaving only
   ``NAND3(EVAL, PRECHN, BLOCK<b>) -> SELB<b>`` and ``NOR2(GB<k>, SELB<b>)`` on
   the ``EVAL`` rising path.
4. **Merged gated-tristate ``ways = 4`` readout + isolated-keeper Q latch**:
   fuses precharge, hi-skewed sense, and 4:1 column muxing into a 5T tristate
   way receiver beside an isolated-keeper sampling latch and dedicated output
   driver.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from hashlib import sha256
from typing import Any

from ..devices.finfet import VT_LAYERS
from ..devices.row import RowStack
from ..devices.row_support import RowSupportSpec, build_row_support
from ..devices.rowcell import box, draw_frame, draw_rails, label, square
from ..layout.layers import require_gdspy
from .circuit import Circuit, SelectorInstance
from .compiler import BromSpec, selector_map
from .dense_bank import build_dense_bank, ceil_grid
from .dense_cell import build_dense_cell
from .dense_cgen import lift6, v3, v4
from .dense_units import dense_units
from .selector import SelectorPlan, default_selector_plan

__all__ = [
    "build_pruned_macro",
    "build_pruned_selector_array",
    "pruned_macro_circuit",
    "pruned_macro_layout",
]


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
) -> None:
    start = len(c.devices)
    for _ in range(fingers):
        c.fet(True, y, a, "VDD", fins=pfins)
        c.fet(False, y, a, "VSS", fins=nfins)
    _record_unit(c, name, start)


def _add_nand2(
    c: Circuit, name: str, a: str, b: str, y: str, pfins: int = 1, nfins: int = 2
) -> None:
    start = len(c.devices)
    mid = c.node()
    c.fet(True, y, a, "VDD", fins=pfins)
    c.fet(True, y, b, "VDD", fins=pfins)
    c.fet(False, y, a, mid, fins=nfins)
    c.fet(False, mid, b, "VSS", fins=nfins)
    _record_unit(c, name, start)


def _add_nor2(
    c: Circuit, name: str, a: str, b: str, y: str, pfins: int = 2, nfins: int = 1
) -> None:
    start = len(c.devices)
    mid = c.node()
    c.fet(True, mid, a, "VDD", fins=pfins)
    c.fet(True, y, b, mid, fins=pfins)
    c.fet(False, y, a, "VSS", fins=nfins)
    c.fet(False, y, b, "VSS", fins=nfins)
    _record_unit(c, name, start)


def _add_nand3(
    c: Circuit,
    name: str,
    a: str,
    b: str,
    d_in: str,
    y: str,
    pfins: int = 1,
    nfins: int = 3,
) -> None:
    start = len(c.devices)
    m1 = c.node()
    m2 = c.node()
    c.fet(False, y, a, m1, fins=nfins)
    c.fet(False, m1, b, m2, fins=nfins)
    c.fet(False, m2, d_in, "VSS", fins=nfins)
    c.fet(True, y, a, "VDD", fins=pfins)
    c.fet(True, y, b, "VDD", fins=pfins)
    c.fet(True, y, d_in, "VDD", fins=pfins)
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
) -> None:
    start = len(c.devices)
    pa = c.node()
    na = c.node()
    nb = c.node()
    c.fet(True, pa, a0, "VDD", fins=fins)
    c.fet(True, pa, a1, "VDD", fins=fins)
    c.fet(True, y, b0, pa, fins=fins)
    c.fet(True, y, b1, pa, fins=fins)
    c.fet(False, y, a0, na, fins=fins)
    c.fet(False, na, a1, "VSS", fins=fins)
    c.fet(False, y, b0, nb, fins=fins)
    c.fet(False, nb, b1, "VSS", fins=fins)
    _record_unit(c, name, start)


def _add_addr_latch(
    c: Circuit,
    idx: int,
    d_pin: str,
    prechn: str,
    prechn_open: str,
    q_out: str,
    fins: int = 1,
) -> None:
    """Compact active-low input address latch (settles during precharge)."""
    start = len(c.devices)
    st = c.node()
    qb = c.node()
    c.fet(False, st, prechn_open, d_pin, fins=fins)
    c.fet(True, st, prechn, d_pin, fins=fins)
    c.fet(True, qb, st, "VDD", fins=fins)
    c.fet(False, qb, st, "VSS", fins=fins)
    c.fet(True, q_out, qb, "VDD", fins=2)
    c.fet(False, q_out, qb, "VSS", fins=1)
    c.fet(False, st, prechn, q_out, fins=fins)
    c.fet(True, st, prechn_open, q_out, fins=fins)
    _record_unit(c, f"address_latch_{idx}", start)


def _add_sense_m1(
    c: Circuit,
    bit: int,
    bl: str,
    prechn: str,
    dout: str,
    *,
    keeper_fins: int = 0,
    keeper_stack: int = 2,
) -> None:
    """Precharge PMOS + hi-skewed sense inverter (+ optional keeper) for ways=1."""
    start = len(c.devices)
    c.fet(True, bl, prechn, "VDD", fins=2)
    c.fet(True, dout, bl, "VDD", fins=3)
    c.fet(False, dout, bl, "VSS", fins=1)
    source = "VDD"
    for stage in range(keeper_stack if keeper_fins else 0):
        drain = bl if stage == keeper_stack - 1 else c.node()
        c.fet(True, drain, dout, source, fins=keeper_fins)
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
) -> None:
    """1-stage hi-skewed gated tristate sense + precharge for ways=4."""
    start = len(c.devices)
    mp = c.node()
    mn = c.node()
    c.fet(True, bl, prechn, "VDD", fins=2)
    c.fet(True, mp, bl, "VDD", fins=3)
    c.fet(True, dout, csb, mp, fins=3)
    c.fet(False, mn, bl, "VSS", fins=1)
    c.fet(False, dout, cs, mn, fins=1)
    # Bank placers find a column's receiver by ``readout_<column>``.
    _record_unit(c, f"readout_{bit * 4 + way}", start)


def _add_fast_out_latch(
    c: Circuit, bit: int, dout: str, sample: str, sampleb: str, q_pin: str
) -> None:
    """Isolated-keeper tristate sampling latch + 6-fin output driver."""
    start = len(c.devices)
    mp_d = c.node()
    mn_d = c.node()
    sb = c.node()
    qk = c.node()
    mp_k = c.node()
    mn_k = c.node()
    c.fet(True, mp_d, dout, "VDD", fins=3)
    c.fet(True, sb, sampleb, mp_d, fins=3)
    c.fet(False, mn_d, dout, "VSS", fins=3)
    c.fet(False, sb, sample, mn_d, fins=3)
    c.fet(True, qk, sb, "VDD", fins=1)
    c.fet(False, qk, sb, "VSS", fins=1)
    c.fet(True, mp_k, qk, "VDD", fins=1)
    c.fet(True, sb, sample, mp_k, fins=1)
    c.fet(False, mn_k, qk, "VSS", fins=1)
    c.fet(False, sb, sampleb, mn_k, fins=1)
    c.fet(True, q_pin, sb, "VDD", fins=3)
    c.fet(True, q_pin, sb, "VDD", fins=3)
    c.fet(False, q_pin, sb, "VSS", fins=3)
    c.fet(False, q_pin, sb, "VSS", fins=3)
    # ``output_latch_<bit>`` lets wide words use bit tiles (``SLICE_FIRST_BITS``).
    _record_unit(c, f"output_latch_{bit}", start)


def _decode_tree(c: Circuit, bits: list[str], prefix: str, limit: int) -> None:
    """Factored active-high precharge-window decoder with singleton-1-fin sizing."""
    n = len(bits)
    if n == 0:
        _add_sized_inv(c, f"{prefix}_tieb", "VDD", f"{prefix}_zb", 1, 1)
        _add_sized_inv(c, f"{prefix}_tie0", f"{prefix}_zb", f"{prefix}<0>", 1, 1)
        return
    if n == 1:
        _add_sized_inv(c, f"{prefix}_inv0", bits[0], f"{prefix}<0>", 2, 1)
        if limit > 1:
            b0 = c.node()
            _add_sized_inv(c, f"{prefix}_buf1a", bits[0], b0, 1, 1)
            _add_sized_inv(c, f"{prefix}_buf1b", b0, f"{prefix}<1>", 2, 1)
        return
    if n == 2:
        n0, n1 = c.node(), c.node()
        _add_sized_inv(c, f"{prefix}_in0", bits[0], n0, 1, 1)
        _add_sized_inv(c, f"{prefix}_in1", bits[1], n1, 1, 1)
        for i, (x, y_in) in enumerate(((bits[1], bits[0]), (bits[1], n0), (n1, bits[0]))):
            if i < limit:
                _add_nor2(c, f"{prefix}_dec_{i}", x, y_in, f"{prefix}<{i}>", 1, 1)
        if limit > 3:
            nb3 = c.node()
            _add_nand2(c, f"{prefix}_dec_3b", bits[1], bits[0], nb3, 1, 1)
            _add_sized_inv(c, f"{prefix}_dec_3", nb3, f"{prefix}<3>", 2, 1)
        return
    if n == 3:
        n0, n1, n2 = c.node(), c.node(), c.node()
        _add_sized_inv(c, f"{prefix}_in0", bits[0], n0, 1, 1)
        _add_sized_inv(c, f"{prefix}_in1", bits[1], n1, 1, 1)
        _add_sized_inv(c, f"{prefix}_in2", bits[2], n2, 1, 1)
        pre_b = []
        for j, (x, y_in) in enumerate(((n1, n0), (n1, bits[0]), (bits[1], n0))):
            pb = c.node()
            _add_nand2(c, f"{prefix}_pre_{j}", x, y_in, pb, 1, 1)
            pre_b.append(pb)
        pb3_pos = c.node()
        pb3 = c.node()
        _add_nor2(c, f"{prefix}_pre_3a", n1, n0, pb3_pos, 1, 1)
        _add_sized_inv(c, f"{prefix}_pre_3", pb3_pos, pb3, 1, 1)
        pre_b.append(pb3)
        for i in range(limit):
            hi_inv = bits[2] if (i >> 2) == 0 else n2
            _add_nor2(c, f"{prefix}_dec_{i}", pre_b[i & 3], hi_inv, f"{prefix}<{i}>", 1, 1)
        return
    half = n // 2
    lo_lim = 1 << half
    hi_lim = (limit + lo_lim - 1) >> half
    lo_pref, hi_pref = f"{prefix}_L", f"{prefix}_H"
    _decode_tree(c, bits[:half], lo_pref, lo_lim)
    _decode_tree(c, bits[half:], hi_pref, hi_lim)
    for i in range(limit):
        nb = c.node()
        _add_nand2(
            c,
            f"{prefix}_nd_{i}",
            f"{lo_pref}<{i & (lo_lim - 1)}>",
            f"{hi_pref}<{i >> half}>",
            nb,
            1,
            1,
        )
        _add_sized_inv(c, f"{prefix}_iv_{i}", nb, f"{prefix}<{i}>", 2, 1)


def _add_gb_generator(
    c: Circuit, a0: str, a1: str, needed_gb: Sequence[int]
) -> None:
    """1-stage active-low ``GB<k> = ~G<k>`` generator only for active ``k in 1..14``."""
    n0 = c.node()
    n1 = c.node()
    _add_sized_inv(c, "gb_inv_a0", a0, n0, 2, 1)
    _add_sized_inv(c, "gb_inv_a1", a1, n1, 2, 1)
    if not needed_gb:
        anc = c.node()
        _add_nand2(c, "gb_anchor", n1, a0, anc, 1, 1)
        return

    for k in needed_gb:
        out = f"GB<{k}>"
        if k == 10:
            _add_sized_inv(c, f"gb_{k}", a0, out, 2, 1)
        elif k == 5:
            _add_sized_inv(c, f"gb_{k}", n0, out, 2, 1)
        elif k == 12:
            _add_sized_inv(c, f"gb_{k}", a1, out, 2, 1)
        elif k == 3:
            _add_sized_inv(c, f"gb_{k}", n1, out, 2, 1)
        elif k == 1:
            _add_nand2(c, f"gb_{k}", n1, n0, out, 1, 1)
        elif k == 2:
            _add_nand2(c, f"gb_{k}", n1, a0, out, 1, 1)
        elif k == 4:
            _add_nand2(c, f"gb_{k}", a1, n0, out, 1, 1)
        elif k == 8:
            g8 = c.node()
            _add_nor2(c, "gb_8_nor", n1, n0, g8, 1, 1)
            _add_sized_inv(c, "gb_8", g8, out, 2, 1)
        elif k == 14:
            g14 = c.node()
            _add_nand2(c, "gb_14_nand", n1, n0, g14, 1, 1)
            _add_sized_inv(c, "gb_14", g14, out, 2, 1)
        elif k == 13:
            _add_nor2(c, f"gb_{k}", a1, n0, out, 1, 1)
        elif k == 11:
            _add_nor2(c, f"gb_{k}", n1, a0, out, 1, 1)
        elif k == 7:
            _add_nor2(c, f"gb_{k}", n1, n0, out, 1, 1)
        elif k == 9:
            _add_aoi22(c, f"gb_{k}", a1, a0, n1, n0, out, 1)
        elif k == 6:
            _add_aoi22(c, f"gb_{k}", a1, n0, n1, a0, out, 1)


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


def _block_track_plan(
    mapping: tuple[tuple[int, ...], ...], fins: int
) -> tuple[list[int], list[dict[int, int]], int]:
    """Choose global CGEN slot order and per-block track assignment minimizing M6 lifts."""
    freq: dict[int, int] = defaultdict(int)
    for row in mapping:
        for k in set(row):
            if k != 0:
                freq[k] += 1
    active_k = sorted(freq.keys(), key=lambda k: (k == 15, -freq[k], k))

    direct_m2, upper_m2 = _m2_track_roles(fins)
    m4_tracks = (8, 9, 10, 11, 12, 13, 14, 15)

    slot_index = {k: i for i, k in enumerate(active_k)}
    per_block_k2t: list[dict[int, int]] = []
    max_upper = 0
    for row in mapping:
        used_b = sorted((k for k in set(row) if k != 0), key=lambda k: slot_index[k])
        k2t: dict[int, int] = {}
        n_b = len(used_b)
        if n_b <= 8:
            for idx, k in enumerate(used_b):
                k2t[k] = m4_tracks[idx]
        elif n_b <= 10:
            n_dir = n_b - 8
            for idx, k in enumerate(used_b[:-n_dir]):
                k2t[k] = m4_tracks[idx]
            for idx, k in enumerate(used_b[-n_dir:]):
                k2t[k] = direct_m2[idx]
        else:
            n_up = n_b - 10
            max_upper = max(max_upper, n_up)
            for idx, k in enumerate(used_b[:n_up]):
                k2t[k] = upper_m2[idx]
            for idx, k in enumerate(used_b[n_up : n_up + 8]):
                k2t[k] = m4_tracks[idx]
            for idx, k in enumerate(used_b[n_up + 8 :]):
                k2t[k] = direct_m2[idx]
        per_block_k2t.append(k2t)
    return active_k, per_block_k2t, max_upper


def build_pruned_selector_array(
    plan: SelectorPlan,
    mapping: tuple[tuple[int, ...], ...],
    per_block_k2t: Sequence[dict[int, int]] | None = None,
    *,
    lib: Any = None,
) -> Any:
    """Build selector array with ``k = 0`` decoupled from ``BL`` and trimmed candidate tracks."""
    if not mapping or not mapping[0] or any(len(r) != len(mapping[0]) for r in mapping):
        raise ValueError("mapping must be a nonempty rectangle")
    if any(type(k) is not int or not 0 <= k < 16 for row in mapping for k in row):
        raise ValueError("candidate indices must be integers in 0..15")
    if lib is not None and lib.unit != 1e-9:
        raise ValueError("GDS coordinates are nanometres")
    if per_block_k2t is None:
        _, per_block_k2t, _ = _block_track_plan(mapping, plan.fins)

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
        box(top, "M3", x - 9, 0, x + 9, h)

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


def _build_cgen_templates(fins: int, vt: str) -> dict[Any, Any]:
    gdspy = require_gdspy()
    tracks = default_selector_plan(fins, vt).tracks
    direct = _m2_track_roles(fins)[0]
    stack = RowStack(((fins, fins),), vt=vt, band_height=216)

    spacer = gdspy.Cell(f"brom_sp54_{fins}_{vt}", exclude_from_current=True)
    draw_frame(spacer, stack, 54, vt)
    draw_rails(spacer, stack, 54)
    filler = build_row_support(RowSupportSpec(stack, "filler"), draw_pin_labels=False)
    filler.name += "_pruned_cgen"
    tap = build_row_support(RowSupportSpec(stack, "tap"), draw_pin_labels=False)
    tap.name += "_pruned_cgen"

    nor2_devs = (
        ("p", 3, "mid", "GB", "VDD"),
        ("p", 3, "C", "SELB", "mid"),
        ("n", 1, "C", "SELB", "VSS"),
        ("n", 1, "C", "GB", "VSS"),
    )
    inv_devs = (
        ("p", 3, "C", "SELB", "VDD"),
        ("n", 1, "C", "SELB", "VSS"),
    )
    out: dict[Any, Any] = {"spacer": spacer, "filler": filler, "tap": tap}
    for flip in (False, True):
        sy = 414 if flip else 18
        res = (sy, *(432 - tracks[k][1] if flip else tracks[k][1] for k in direct))
        un, pn = build_dense_cell(
            nor2_devs,
            ("GB", "SELB", "C"),
            vt=vt,
            reserved_m2=res,
            fixed_trunks=(("GB", 27),),
            share=True,
            abut_y=True,
        )
        ui, pi = build_dense_cell(
            inv_devs,
            ("SELB", "C"),
            vt=vt,
            reserved_m2=res,
            share=True,
            abut_y=True,
        )
        suffix = "_cgen_hi" if flip else "_cgen_lo"
        for u, p in ((un, pn), (ui, pi)):
            u.name += suffix
            u.labels.clear()
            tx = dict(p.trunks)["SELB"]
            cx = dict(p.trunks)["C"]
            u.remove_polygons(
                lambda poly, l, d, tx=tx, cx=cx: l == 30
                and d == 0
                and (min(poly[:, 0]), max(poly[:, 0])) in ((tx - 9, tx + 9), (cx - 9, cx + 9))
            )
            for polygon in u.polygons:
                polygon.polygons = list(polygon.polygons)
            ys = [ty for n, _, _, ty in p.contacts if n == "SELB"]
            box(u, "M3", tx - 9, min(sy, *ys) - 14, tx + 9, max(sy, *ys) + 14)
            box(u, "M3", cx - 9, 16, cx + 9, 416)
        out[("nor", flip)] = (un, pn)
        out[("inv", flip)] = (ui, pi)
    return out


def pruned_macro_circuit(spec: BromSpec, words: Sequence[int]) -> Circuit:
    """Build a data-pruned high-PPA BROM ``Circuit`` for fixed word contents."""
    if spec.registered_output:
        raise ValueError("pruned_macro_circuit uses external read phases (PRECHN/EVAL/SAMPLE)")
    mapping = selector_map(spec, words)
    digest = sha256(repr((spec, mapping)).encode()).hexdigest()[:12]
    c = Circuit(
        f"brom_pruned_{spec.depth}x{spec.bits}_m{spec.ways}_f{spec.fins}_{spec.vt}_{digest}",
        spec.pins,
        spec.fins,
        spec.vt,
    )
    if spec.keeper_fins:
        if spec.ways != 1:
            raise ValueError("pruned ways=4 uses gated tristate sense without bitline keeper")
        c.name += f"_k{spec.keeper_fins}s{spec.keeper_stack}"

    active_k, per_block_k2t, max_upper = _block_track_plan(mapping, spec.fins)
    needed_gb = [k for k in active_k if k != 15]
    ab = spec.address_bits

    # 1. Input address latches
    addr_open = c.node()
    _add_sized_inv(
        c, "prechn_inv", "PRECHN", addr_open, 2, 2, fingers=1 if ab <= 8 else 2
    )
    for i in range(ab):
        _add_addr_latch(c, i, f"A[{i}]", "PRECHN", addr_open, f"ADDR<{i}>", fins=1)

    # 2. Column way decoder (ways=4), active-low GB<k> generator, and row block decoder
    shift = spec.column_bits
    if spec.ways == 4:
        _decode_tree(c, ["ADDR<0>", "ADDR<1>"], "CSEL_RAW", 4)
        csel_fingers = max(1, (spec.bits + 11) // 12)
        for w in range(4):
            if csel_fingers == 1:
                _add_sized_inv(c, f"cselb_drv_{w}", f"CSEL_RAW<{w}>", f"CSELB<{w}>", 2, 1)
                _add_sized_inv(c, f"csel_drv_{w}", f"CSELB<{w}>", f"CSEL<{w}>", 2, 1)
            else:
                _add_sized_inv(
                    c, f"cselb_drv_{w}", f"CSEL_RAW<{w}>", f"CSELB<{w}>", 2, 2, fingers=csel_fingers
                )
                _add_sized_inv(
                    c, f"csel_drv_{w}", f"CSELB<{w}>", f"CSEL<{w}>", 2, 2, fingers=csel_fingers
                )

    _add_gb_generator(c, f"ADDR<{shift}>", f"ADDR<{shift + 1}>", needed_gb)

    row_addr = [f"ADDR<{i}>" for i in range(shift + 2, ab)]
    if spec.blocks > 1:
        _decode_tree(c, row_addr, "BLOCK", spec.blocks)

    # 3. 1-stage SELB<b> drivers on EVAL rising path
    if spec.blocks == 1:
        _add_nand2(c, "selb_0", "EVAL", "PRECHN", "SELB<0>", 1, 3)
    else:
        eval_sig = "EVAL"
        if spec.blocks > 12:
            ev_b = c.node()
            eval_sig = c.node()
            _add_sized_inv(c, "eval_inv0", "EVAL", ev_b, 3, 3)
            _add_sized_inv(c, "eval_inv1", ev_b, eval_sig, 3, 3, fingers=2)
        for b in range(spec.blocks):
            _add_nand3(
                c, f"selb_{b}", eval_sig, "PRECHN", f"BLOCK<{b}>", f"SELB<{b}>", 1, 3
            )

    # 4. Sense + Column Mux + Output Latches (chunked SAMPLE buffering for wide words)
    chunk_size = 12
    n_chunks = (spec.bits + chunk_size - 1) // chunk_size
    for ch in range(n_chunks):
        sb_net = f"SAMPLEB_{ch}"
        s_net = f"SAMPLE_EN_{ch}"
        _add_sized_inv(c, f"samp_inv0_{ch}", "SAMPLE", sb_net, 3, 3, fingers=2)
        _add_sized_inv(c, f"samp_inv1_{ch}", sb_net, s_net, 3, 3, fingers=2)
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
                    keeper_fins=spec.keeper_fins,
                    keeper_stack=spec.keeper_stack,
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
                    )
            _add_fast_out_latch(c, bit, dout, s_net, sb_net, f"Q[{bit}]")

    cgen_device_names: set[str] = set()
    for b in range(spec.blocks):
        k2t = per_block_k2t[b]
        for k in k2t:
            start = len(c.devices)
            cname = f"C<{b},{k}>"
            if k == 15:
                c.fet(True, cname, f"SELB<{b}>", "VDD", fins=3)
                c.fet(False, cname, f"SELB<{b}>", "VSS", fins=1)
            else:
                mid = c.node()
                c.fet(True, mid, f"GB<{k}>", "VDD", fins=3)
                c.fet(True, cname, f"SELB<{b}>", mid, fins=3)
                c.fet(False, cname, f"SELB<{b}>", "VSS", fins=1)
                c.fet(False, cname, f"GB<{k}>", "VSS", fins=1)
            cgen_device_names.update(d.name for d in c.devices[start:])
            c.group_since(f"pruned_cgen_{b}_{k}", start)

    array_devices: set[str] = set()
    for b, row in enumerate(mapping):
        for col, k in enumerate(row):
            if k == 0:
                c.fet(False, "VSS", "VSS", "VSS", fins=spec.fins)
                array_devices.add(c.devices[-1].name)
                c.fet(False, "VSS", "VSS", "VSS", fins=spec.fins)
                array_devices.add(c.devices[-1].name)
            else:
                c.fet(False, f"BL<{col}>", f"C<{b},{k}>", "VSS", fins=spec.fins)
                array_devices.add(c.devices[-1].name)
                c.fet(False, f"BL<{col}>", "VSS", "VSS", fins=spec.fins)
                array_devices.add(c.devices[-1].name)

    c.array = SelectorInstance(
        default_selector_plan(spec.fins, spec.vt), mapping, frozenset(array_devices)
    )
    c.pruned_spec = spec  # type: ignore[attr-defined]
    c.pruned_cgen_devices = frozenset(cgen_device_names)  # type: ignore[attr-defined]
    c.pruned_plan = (active_k, tuple(per_block_k2t), max_upper)  # type: ignore[attr-defined]
    return c


def _build_pruned_macro(c: Circuit) -> tuple[Any, dict[str, Any]]:
    spec: BromSpec = c.pruned_spec  # type: ignore[attr-defined]
    assert c.array is not None
    mapping = c.array.mapping
    active_k, per_block_k2t, max_upper = c.pruned_plan  # type: ignore[attr-defined]
    needed_gb = [k for k in active_k if k != 15]
    gdspy = require_gdspy()

    n_slots = len(active_k)
    slot_of = {k: s for s, k in enumerate(active_k)}
    max_block_of_gb: dict[int, int] = {}
    for b, k2t in enumerate(per_block_k2t):
        for k in k2t:
            if k != 15:
                max_block_of_gb[k] = b

    sel_x0 = 60
    sel_chan = ceil_grid(sel_x0 + 48 * spec.blocks + 36, 108)
    cgen_origin = sel_chan
    cgen_cells_x0 = cgen_origin + 108
    cgen_support_x0 = cgen_cells_x0 + 216 * n_slots
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
        + [f"SELB<{b}>" for b in range(spec.blocks) if per_block_k2t[b]]
        + [f"BL<{col}>" for col in range(spec.columns)]
    )
    boundary_nets = [n for n in top_section_nets if n in used_in_bank]
    bank_c = Circuit(
        c.name + "_bank",
        tuple(dict.fromkeys([*c.pins, *boundary_nets])),
        c.n.fins,
        c.n.vt,
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

    cgen_t = _build_cgen_templates(spec.fins, spec.vt)
    nor_lo_plan = cgen_t[("nor", False)][1]
    inv_lo_plan = cgen_t[("inv", False)][1]
    nor_gb_dx = 54 + dict(nor_lo_plan.trunks)["GB"]
    nor_c_dx = 54 + dict(nor_lo_plan.trunks)["C"]
    inv_c_dx = dict(inv_lo_plan.trunks)["C"]

    def header_spans(_bw: int) -> dict[str, list[int]]:
        spans: dict[str, list[int]] = defaultdict(list)
        for k in needed_gb:
            if k in max_block_of_gb:
                spans[f"GB<{k}>"].append(cgen_cells_x0 + 216 * slot_of[k] + nor_gb_dx)
        for b in range(spec.blocks):
            if per_block_k2t[b]:
                spans[f"SELB<{b}>"].append(sel_x0 + 48 * b)
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
    for b in range(spec.blocks):
        if per_block_k2t[b]:
            sx = sel_x0 + 48 * b
            anchors[f"SELB<{b}>"] = (sx, sx, None)
    for k in needed_gb:
        if k in max_block_of_gb:
            gx = cgen_cells_x0 + 216 * slot_of[k] + nor_gb_dx
            anchors[f"GB<{k}>"] = (gx - 24, gx + 24, None)

    cgen_m5_cols = [
        ceil_grid(
            cgen_cells_x0
            + 216 * s
            + (inv_c_dx if active_k[s] == 15 else nor_c_dx)
            - 12,
            48,
        )
        + 12
        for s in range(n_slots)
    ]
    drop_m5_cols = [
        ceil_grid(array_end + 120, 48) + 12 + 96 * j for j in range(max_upper)
    ]
    blocked_m5: set[int] = set()
    for mx in (*cgen_m5_cols, *drop_m5_cols):
        blocked_m5.update((mx - 48, mx, mx + 48))

    bank = None
    macro_w = base_macro_w
    for extra in (0, 432, 864, 1296, 1728, 2592, 3888, 5184):
        try_w = base_macro_w + extra
        try:
            bank = build_dense_bank(
                bank_c,
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

    direct_tracks, upper_m2 = _m2_track_roles(spec.fins)
    direct_m2 = set(direct_tracks)
    upper_index = {t: idx for idx, t in enumerate(upper_m2)}

    for b in range(spec.blocks):
        flip = bool(b % 2)
        oy = base + b * 432
        ry = oy + 432 if flip else oy
        k2t = per_block_k2t[b]

        top.add(
            gdspy.CellReference(cgen_t["filler"], (cgen_origin, ry), x_reflection=flip)
        )

        used_sel_xs: list[int] = []
        for s, k in enumerate(active_k):
            sx_cell = cgen_cells_x0 + 216 * s
            if k not in k2t:
                top.add(
                    gdspy.CellReference(cgen_t["filler"], (sx_cell, ry), x_reflection=flip)
                )
                top.add(
                    gdspy.CellReference(
                        cgen_t["filler"], (sx_cell + 108, ry), x_reflection=flip
                    )
                )
            else:
                if k == 15:
                    ui, pi = cgen_t[("inv", flip)]
                    top.add(gdspy.CellReference(ui, (sx_cell, ry), x_reflection=flip))
                    top.add(
                        gdspy.CellReference(
                            cgen_t["spacer"], (sx_cell + 162, ry), x_reflection=flip
                        )
                    )
                    cx = sx_cell + dict(pi.trunks)["C"]
                    sel_tx = sx_cell + dict(pi.trunks)["SELB"]
                else:
                    un, pn = cgen_t[("nor", flip)]
                    top.add(
                        gdspy.CellReference(cgen_t["spacer"], (sx_cell, ry), x_reflection=flip)
                    )
                    top.add(
                        gdspy.CellReference(un, (sx_cell + 54, ry), x_reflection=flip)
                    )
                    cx = sx_cell + 54 + dict(pn.trunks)["C"]
                    sel_tx = sx_cell + 54 + dict(pn.trunks)["SELB"]
                used_sel_xs.append(sel_tx)

                square(top, "V2", sel_tx, oy + 18)
                t = k2t[k]
                metal, ty_rel = sel_plan.tracks[t]
                ty = oy + ty_rel
                cname = f"C<{b},{k}>"
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

            if k != 15 and k in max_block_of_gb and b <= max_block_of_gb[k]:
                gx = sx_cell + nor_gb_dx
                y_hi = oy + 446 if b < max_block_of_gb[k] else oy + 416
                box(top, "M3", gx - 9, oy - 14, gx + 9, y_hi)

        for i, supp in enumerate(("filler", "tap", "filler")):
            top.add(
                gdspy.CellReference(
                    cgen_t[supp], (cgen_support_x0 + 108 * i, ry), x_reflection=flip
                )
            )

        if used_sel_xs:
            sy = oy + 18
            sx = sel_x0 + 48 * b
            max_sel_x = max(used_sel_xs)
            box(top, "M2", sx - 19, sy - 9, max_sel_x + 19, sy + 9)
            square(top, "V2", sx, sy)
            py = (
                oy + 60
                if (b and f"SELB<{b}>" in bank.anchored)
                else base - 36 - 96 * (b % 3)
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

    for k in needed_gb:
        if k in max_block_of_gb:
            gx = cgen_cells_x0 + 216 * slot_of[k] + nor_gb_dx
            py = base - 36
            box(top, "M3", gx - 9, py - 17, gx + 9, base + 14)
            join(f"GB<{k}>", gx, py)
            checkpoints.append((f"GB<{k}>", "M3", gx, base + 200))

    for col in range(spec.columns):
        bx = array_x + 108 * col + 81
        py = base - 36 - 96 * (col % 2)
        box(top, "M3", bx - 9, py - 17, bx + 9, base + 14)
        join(f"BL<{col}>", bx, py)
        checkpoints.append((f"BL<{col}>", "M3", bx, base + 216))

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
    box(top, "M3", vss_out_x - 9, min(py_tap, sy_tap) - 17, vss_out_x + 9, max(py_tap, sy_tap) + 17)
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
                "active_candidates": active_k,
            }
        ],
        "bottom_units": bank.units,
        "routes": routes,
        "checkpoints": checkpoints,
        "array_y_nm": base,
        "cgen_slots": len(active_k),
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
        meta["schema"] = "brom_pruned_physical_v1"
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
