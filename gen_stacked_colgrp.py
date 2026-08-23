"""
stacked_colgrp compiler — tiles the ASAP7 push-rule row/colgrp to form
the `stacked_colgrp_x{WL}x{BITS}x{BANKS}` found in `tech/gds/srambank_32b.gds`
and `tmp/sram_flat_*.sp:.SUBCKT stacked_colgrp_x512x16x1`.

Hierarchy (matches OpenFinRAM C++ `include/main_layout_helpers.hpp`):

  sram_cell_6t_122      (2*CPP 0.108 ×10*FP 0.270, push-rule leaf)
    ↓ y-stack N
  sram_cell_row         (N deep, 1 BL pair)  — scripts/gen_sram_row.py
    ↓ x-stack MUX (typically 4)
  array_sram_6t122      (N deep, 4 BL pairs, 4 columns, + dummy top/bot)
    ↓ + iocolgrp periphery
  colgrp_sram_6t122     (N deep, 4 physical columns muxed to 1 logical bit)
    ↓ x-stack NUM_COLGRP
  stacked_colgrp_x{WL}x{BITS}x1  (WL rows, BITS logical bits, 1 bank)
    Example in repo: stacked_colgrp_x512x16x1  WLT[0:255] WLB[0:255] D[0:15] Q[0:15]
                     16× colgrp_sram_6t122 sharing WL bus, each gives 1 D/Q
                     stacked_colgrp_x1024x32x1 32× colgrp, 1024 WL (512 WLT+512 WLB)
                     stacked_colgrp_x4x2x1     2× colgrp, 4 WL

Naming:  x{W}x{B}x1  =  W wordlines total (WLT+WLB), B logical bits, 1 bank.
"sram_cell_row" in SPICE is a *column* sharing BL/BLN; "colgrp" groups
4 columns with y-mux; "stacked" groups B such colgrps horizontally.

This script builds the hierarchy *without* the analog periphery (sense-amp,
write-driver, precharge, y-mux) — those are handled in the C++ flow via
`iocolgrp_sram_6t122_v2`. The geometry here is the bitcell array portion,
which dominates area and DRC and is the focus for the FinFET/CFET compiler.

Placement is grid-constrained:
  CELL_WIDTH  =0.108, CELL_HEIGHT=0.270, colgrp pitch = MUX*CELL_WIDTH on CPP grid
  stacked_width  = NUM_COLGRP * MUX * CELL_WIDTH
  stacked_height = WORDS * CELL_HEIGHT (+ dummy, omitted unless --with-dummy)
Mirroring (MY) inside each row is inherited; colgrps themselves are not
mirrored (they abut on M2 BL/M1 rails).

Usage:
  python scripts/gen_stacked_colgrp.py                           # 16×256 like x512x16 (demo)
  python scripts/gen_stacked_colgrp.py -o sc_x4x2.gds --bits 2 --words 4 --mux 1 --no-mirror
  python scripts/gen_stacked_colgrp.py -o sc_x1024x32.gds --bits 32 --words 1024 --mux 4
  python scripts/gen_stacked_colgrp.py --from-gds tech/gds/srambank_32b_boundary_2.gds

Feeds the OpenFinRAM FinFET/CFET bank compiler (row → colgrp → stacked).
"""

import argparse
import math
import warnings
import gdstk

# ---------------------------------------------------------------------------
# ASAP7 grids — must match gen_pushrule_parametric.py / gen_sram_row.py
# ---------------------------------------------------------------------------
CPP          = 0.054
FIN_PITCH    = 0.027
CELL_WIDTH   = 2 * CPP         # 0.108
CELL_HEIGHT  = 10 * FIN_PITCH  # 0.270
VIA_SIZE     = 0.018

L_WELL   = (1, 0)
L_FIN    = (2, 0)
L_GATE   = (7, 0)
L_GCUT   = (10, 0)
L_ACTIVE = (11, 0)
L_NSEL   = (12, 0)
L_PSEL   = (13, 0)
L_LIG    = (16, 0)
L_LISD   = (17, 0)
L_V0     = (18, 0)
L_M1     = (19, 0)
L_M2     = (20, 0)
L_V1     = (21, 0)
L_V2     = (25, 0)
L_M3     = (30, 0)
L_SRAMDRC  = (99, 0)
L_BOUNDARY = (100, 0)

Y_FINS = [0.2665, 0.2395, 0.2125, 0.1855, 0.1585, 0.1315, 0.1045, 0.0775, 0.0505, -0.0035, 0.0235]

def _poly(cell, pts, lt):
    cell.add(gdstk.Polygon(pts, lt[0], lt[1]))

def _via(cell, x, y, lt, s=VIA_SIZE):
    _poly(cell, [[x,y],[x+s,y],[x+s,y+s],[x,y+s]], lt)

def _build_bitcell(lib, name="sram_cell_6t_122"):
    cell = lib.new_cell(name)
    _poly(cell, [[-0.008,0.1485],[0.062,0.1485],[0.062,0.1755],[-0.008,0.1755]], L_ACTIVE)
    _poly(cell, [[-0.008,0.0135],[0.116,0.0135],[0.116,0.0675],[-0.008,0.0675]], L_ACTIVE)
    _poly(cell, [[-0.008,0.2025],[0.116,0.2025],[0.116,0.2565],[-0.008,0.2565]], L_ACTIVE)
    _poly(cell, [[0.046,0.0945],[0.116,0.0945],[0.116,0.1215],[0.046,0.1215]], L_ACTIVE)
    _poly(cell, [[-0.027,0.081],[0.135,0.081],[0.135,0.189],[-0.027,0.189]], L_WELL)
    _poly(cell, [[0.0,0.0],[CELL_WIDTH,0.0],[CELL_WIDTH,CELL_HEIGHT],[0.0,CELL_HEIGHT]], L_SRAMDRC)
    _poly(cell, [[0.0,0.0],[CELL_WIDTH,0.0],[CELL_WIDTH,CELL_HEIGHT],[0.0,CELL_HEIGHT]], L_BOUNDARY)
    _poly(cell, [[0.017,-0.019],[0.037,-0.019],[0.037,0.277],[0.017,0.277]], L_GATE)
    _poly(cell, [[0.071,-0.007],[0.091,-0.007],[0.091,0.289],[0.071,0.289]], L_GATE)
    _poly(cell, [[0.054,-0.0085],[0.1085,-0.0085],[0.1085,0.0085],[0.054,0.0085]], L_GCUT)
    _poly(cell, [[0.0,0.2615],[0.054,0.2615],[0.054,0.2785],[0.0,0.2785]], L_GCUT)
    _poly(cell, [[0.0,0.0715],[0.054,0.0715],[0.054,0.0885],[0.0,0.0885]], L_GCUT)
    _poly(cell, [[0.054,0.1815],[0.108,0.1815],[0.108,0.1985],[0.054,0.1985]], L_GCUT)
    for yf in Y_FINS:
        _poly(cell, [[-0.027,yf],[0.135,yf],[0.135,yf+0.007],[-0.027,yf+0.007]], L_FIN)
    _poly(cell, [[0.017,0.0935],[0.054,0.0935],[0.054,0.1095],[0.017,0.1095]], L_LIG)
    _poly(cell, [[0.054,0.1605],[0.091,0.1605],[0.091,0.1765],[0.054,0.1765]], L_LIG)
    _poly(cell, [[0.022,-0.008],[0.054,-0.008],[0.054,0.008],[0.022,0.008]], L_LIG)
    _poly(cell, [[0.054,0.262],[0.086,0.262],[0.086,0.278],[0.054,0.278]], L_LIG)
    _poly(cell, [[-0.012,0.1315],[0.012,0.1315],[0.012,0.1755],[-0.012,0.1755]], L_LISD)
    _poly(cell, [[-0.012,0.0195],[0.012,0.0195],[0.012,0.0675],[-0.012,0.0675]], L_LISD)
    _poly(cell, [[0.042,0.152],[0.066,0.152],[0.066,0.2465],[0.042,0.2465]], L_LISD)
    _poly(cell, [[0.096,0.2025],[0.12,0.2025],[0.12,0.2505],[0.096,0.2505]], L_LISD)
    _poly(cell, [[0.096,0.0945],[0.12,0.0945],[0.12,0.1385],[0.096,0.1385]], L_LISD)
    _poly(cell, [[-0.012,0.2105],[0.012,0.2105],[0.012,0.283],[-0.012,0.283]], L_LISD)
    _poly(cell, [[0.042,0.0235],[0.066,0.0235],[0.066,0.118],[0.042,0.118]], L_LISD)
    _poly(cell, [[0.096,-0.013],[0.12,-0.013],[0.12,0.059],[0.096,0.059]], L_LISD)
    for x,y in [[-0.009,0.1315],[0.099,-0.009],[-0.009,0.261],[0.0595,0.261],[-0.009,0.043],[0.099,0.1205],[0.099,0.209],[0.0305,-0.009]]:
        _via(cell, x,y, L_V0)
    _poly(cell, [[0.099,0.1675],[0.117,0.1675],[0.117,0.2355],[0.099,0.2355]], L_M1)
    _poly(cell, [[0.0305,-0.019],[0.0485,-0.019],[0.0485,0.019],[0.0305,0.019]], L_M1)
    _poly(cell, [[-0.009,0.215],[0.009,0.215],[0.009,0.289],[-0.009,0.289]], L_M1)
    _poly(cell, [[-0.009,0.0345],[0.009,0.0345],[0.009,0.1025],[-0.009,0.1025]], L_M1)
    _poly(cell, [[0.127,0.1385],[0.0725,0.1385],[0.0725,0.1495],[-0.019,0.1495],[-0.019,0.1315],[0.0355,0.1315],[0.0355,0.1205],[0.127,0.1205]], L_M1)
    _poly(cell, [[0.0595,0.251],[0.0775,0.251],[0.0775,0.289],[0.0595,0.289]], L_M1)
    _poly(cell, [[0.099,-0.019],[0.117,-0.019],[0.117,0.055],[0.099,0.055]], L_M1)
    _poly(cell, [[-0.027,0.126],[0.135,0.126],[0.135,0.144],[-0.027,0.144]], L_M2)
    _poly(cell, [[-0.026,0.0745],[0.136,0.0745],[0.136,0.0925],[-0.026,0.0925]], L_M2)
    _poly(cell, [[-0.027,0.225],[0.135,0.225],[0.135,0.243],[-0.027,0.243]], L_M2)
    _poly(cell, [[-0.026,0.1775],[0.136,0.1775],[0.136,0.1955],[-0.026,0.1955]], L_M2)
    _poly(cell, [[0.0215,-0.009],[0.072,-0.009],[0.072,0.009],[0.0215,0.009]], L_M2)
    _poly(cell, [[0.036,0.261],[0.0865,0.261],[0.0865,0.279],[0.036,0.279]], L_M2)
    _poly(cell, [[-0.027,0.027],[0.135,0.027],[0.135,0.045],[-0.027,0.045]], L_M2)
    _poly(cell, [[0.045,-0.04],[0.063,-0.04],[0.063,0.31],[0.045,0.31]], L_M3)
    for x,y in [[-0.009,0.0745],[0.099,0.1775],[0.045,0.126],[-0.009,0.225],[0.0305,-0.009],[0.0595,0.261],[0.099,0.027]]:
        _via(cell, x,y, L_V1)
    _via(cell, 0.045,0.261, L_V2)
    _via(cell, 0.045,-0.009, L_V2)
    _poly(cell, [[-0.027,0.0],[0.135,0.0],[0.135,0.081],[-0.027,0.081]], L_NSEL)
    _poly(cell, [[-0.027,0.189],[0.135,0.189],[0.135,0.27],[-0.027,0.27]], L_NSEL)
    _poly(cell, [[-0.027,0.081],[0.135,0.081],[0.135,0.189],[-0.027,0.189]], L_PSEL)
    return cell

def _build_row(lib, bitcell, num_cells, use_mirror, name):
    """Vertical stack of bitcells sharing BL/BLN — same as gen_sram_row."""
    cell = lib.new_cell(name)
    for i in range(num_cells):
        y0 = i * CELL_HEIGHT
        if use_mirror and (i % 2 == 1):
            ref = gdstk.Reference(bitcell, origin=(0, y0 + CELL_HEIGHT), x_reflection=True)
        else:
            ref = gdstk.Reference(bitcell, origin=(0, y0))
        cell.add(ref)
        cell.add(gdstk.Label(f"WL[{i}]", (0.055, y0 + CELL_HEIGHT/2), layer=30, texttype=251))
    h = num_cells * CELL_HEIGHT
    w = CELL_WIDTH
    cell.add(gdstk.Polygon([[0,0],[w,0],[w,h],[0,h]], *L_BOUNDARY))
    cell.add(gdstk.Polygon([[0,0],[w,0],[w,h],[0,h]], *L_SRAMDRC))
    cell.add(gdstk.Label("BL",  (0.13, h/2), layer=20, texttype=251))
    cell.add(gdstk.Label("BLN", (-0.018, h/2), layer=20, texttype=251))
    return cell

def generate_stacked_colgrp(
    out_file: str,
    num_bits: int = 16,
    words: int = 512,
    mux: int = 4,
    use_mirror: bool = True,
    with_dummy: bool = False,
    bitcell_gds: str | None = None,
    bitcell_name: str = "sram_cell_6t_122",
    colgrp_name: str = "colgrp_sram_6t122",
    stacked_name: str | None = None,
):
    """
    Generate stacked_colgrp GDS.

    words = total WL count (WLT+WLB). In SPICE WLT[0:W/2-1] + WLB[0:W/2-1] map to
    the same physical rows (mirrored WL bus). We implement as `words` physical
    rows per column (e.g. 256 rows → 512 WL with WLT/WLB split).
    For simple `words<=512` the row height is `words*CELL_HEIGHT`.

    mux = physical columns per logical bit (y-mux). 4 is default for ASAP7
    (seen in srambank_32b.gds array_x32x4). 1 gives 1:1 (no mux).

    num_bits = logical bits (number of colgrp copies horizontally).

    with_dummy adds dummy top/bot rows (not DRC-required for demo).
    """
    if stacked_name is None:
        stacked_name = f"stacked_colgrp_x{words}x{num_bits}x1"
    lib = gdstk.Library(unit=1e-6, precision=1e-10)

    # bitcell
    if bitcell_gds:
        src = gdstk.read_gds(bitcell_gds)
        bitcell = next((c for c in src.cells if c.name == bitcell_name), src.cells[0])
        nb = lib.new_cell(bitcell.name)
        for p in bitcell.polygons:
            nb.add(gdstk.Polygon(p.points, p.layer, p.datatype))
        for lbl in bitcell.labels:
            nb.add(gdstk.Label(lbl.text, lbl.origin, layer=lbl.layer, texttype=lbl.texttype))
        bitcell = nb
    else:
        bitcell = _build_bitcell(lib, name=bitcell_name)

    # row
    row = _build_row(lib, bitcell, words, use_mirror, name="sram_cell_row")

    # array (mux columns sharing WLs) — horizontal tiling of rows
    array = lib.new_cell("array_sram_6t122")
    if with_dummy:
        # placeholders for dummy_topbot_v1/v2 — zero-area for DRC demo
        for idx, nm in enumerate(["dummy_topbot_v1","dummy_topbot_v2"]):
            dc = lib.new_cell(nm)
            dc.add(gdstk.Polygon([[0,0],[CELL_WIDTH,0],[CELL_WIDTH,CELL_HEIGHT],[0,CELL_HEIGHT]], *L_BOUNDARY))
            array.add(gdstk.Reference(dc, origin=(0, -CELL_HEIGHT if idx==0 else words*CELL_HEIGHT)))
    for j in range(mux):
        x0 = j * CELL_WIDTH
        # Horizontal bit mirroring: alternate MX (flip X) to share BL contacts
        # Foundry pattern is two mirrored halves (sramcol_x32: 16 normal +16 mirrored);
        # alternating per column is equivalent and DRC-clean, matching GDS.
        if use_mirror and (j % 2 == 1):
            # X-flip = x_reflection (flip Y) + 180deg rot
            array.add(gdstk.Reference(row, origin=(x0 + CELL_WIDTH, 0), rotation=math.pi, x_reflection=True))
        else:
            array.add(gdstk.Reference(row, origin=(x0, 0)))
        array.add(gdstk.Label(f"BL[{j}]",  (x0+0.06, (words*CELL_HEIGHT)/2), layer=20, texttype=251))
        array.add(gdstk.Label(f"BLN[{j}]", (x0+0.06, (words*CELL_HEIGHT)/2+0.05), layer=20, texttype=251))
    aw = mux * CELL_WIDTH
    ah = words * CELL_HEIGHT + (2*CELL_HEIGHT if with_dummy else 0)
    ay0 = -CELL_HEIGHT if with_dummy else 0
    array.add(gdstk.Polygon([[0,ay0],[aw,ay0],[aw,ay0+ah],[0,ay0+ah]], *L_BOUNDARY))
    array.add(gdstk.Polygon([[0,ay0],[aw,ay0],[aw,ay0+ah],[0,ay0+ah]], *L_SRAMDRC))

    # colgrp (array + iocolgrp placeholder)
    colgrp = lib.new_cell(colgrp_name)
    colgrp.add(gdstk.Reference(array, origin=(0,0)))
    # iocolgrp is periphery (sense-amp etc) — place small stub above array
    ioc_h = 2.0  # placeholder height, on CPP grid (approx)
    ioc = lib.new_cell("iocolgrp_stub")
    ioc.add(gdstk.Polygon([[0,0],[aw,0],[aw,ioc_h],[0,ioc_h]], *L_BOUNDARY))
    ioc.add(gdstk.Label("D", (aw/2, ioc_h/2), layer=20, texttype=251))
    ioc.add(gdstk.Label("Q", (aw/2, ioc_h/2+0.2), layer=20, texttype=251))
    colgrp.add(gdstk.Reference(ioc, origin=(0, ah + (ay0 if with_dummy else 0))))
    cw, ch = aw, ah + ioc_h
    colgrp.add(gdstk.Polygon([[0,ay0],[cw,ay0],[cw,ay0+ch],[0,ay0+ch]], *L_BOUNDARY))
    colgrp.add(gdstk.Polygon([[0,ay0],[cw,ay0],[cw,ay0+ch],[0,ay0+ch]], *L_SRAMDRC))
    colgrp.add(gdstk.Label("VDD!", (cw/2, ch/2), layer=20, texttype=251))

    # stacked_colgrp — horizontal tiling of colgrps sharing WL bus
    # Mirrored per-bit (MX) to match foundry: colgrp_x64x4b has array at
    # (-0.005,0) normal + (9.175,0) mirrored (x_reflection True rot pi).
    stacked = lib.new_cell(stacked_name)
    col_pitch = cw  # on CPP grid
    for b in range(num_bits):
        x0 = b * col_pitch
        if use_mirror and (b % 2 == 1):
            # X-flip for odd bits — colgrp mirrored so adjacent bits share
            # power tap / BL edge. Origin shifted by cw to keep abutment.
            stacked.add(gdstk.Reference(colgrp, origin=(x0 + cw, 0), rotation=math.pi, x_reflection=True))
        else:
            stacked.add(gdstk.Reference(colgrp, origin=(x0, 0)))
        stacked.add(gdstk.Label(f"D[{b}]", (x0+cw/2, ch+0.5), layer=20, texttype=251))
        stacked.add(gdstk.Label(f"Q[{b}]", (x0+cw/2, ch+0.8), layer=20, texttype=251))
    sw = num_bits * cw
    sh = ch
    stacked.add(gdstk.Polygon([[0,ay0],[sw,ay0],[sw,ay0+sh],[0,ay0+sh]], *L_BOUNDARY))
    stacked.add(gdstk.Polygon([[0,ay0],[sw,ay0],[sw,ay0+sh],[0,ay0+sh]], *L_SRAMDRC))
    # WL bus labels on M3
    for i in range(min(words, 8)):  # sample first 8 for readability
        stacked.add(gdstk.Label(f"WLT[{i}]", (sw/2, i*CELL_HEIGHT+0.135), layer=30, texttype=251))
    # Also broadcast WLT/WLB split for >256 case (mirrored top/bottom halves)
    # (SPICE uses WLT[0:W/2-1] + WLB[0:W/2-1]; we keep single WL bus for layout demo)

    lib.write_gds(out_file)
    print(f"Generated {stacked_name} bits={num_bits} words={words} mux={mux} "
          f"{'mirrored' if use_mirror else 'unmirrored'} {sw:.3f}x{sh:.3f} um -> {out_file}")
    print(f"  Hierarchy: bitcell({CELL_WIDTH}x{CELL_HEIGHT}) → row({words}) → array(mux={mux}) → colgrp → stacked({num_bits})")
    print(f"  SPICE: .SUBCKT {stacked_name} WLT[0:{words-1}] WLB[0:{words-1}] D[0:{num_bits-1}] Q[0:{num_bits-1}] ...")
    return out_file


def _parse_args():
    p = argparse.ArgumentParser(description="Generate stacked_colgrp (tiles colgrp horizontally, like srambank_32b.gds)")
    p.add_argument("-o","--out", default="tech/gds/stacked_colgrp.gds", help="output GDS")
    p.add_argument("--bits", type=int, default=16, help="logical bits (colgrp copies, default 16 for x512x16)")
    p.add_argument("--words", type=int, default=512, help="wordlines total (default 512 for x512, use 256 for x64x4 demo)")
    p.add_argument("--mux", type=int, default=4, choices=[1,2,4,8], help="physical cols per bit (y-mux, default 4)")
    p.add_argument("--no-mirror", action="store_true", help="disable MY mirror inside rows")
    p.add_argument("--with-dummy", action="store_true", help="add dummy top/bot rows")
    p.add_argument("--bitcell-gds", default=None, help="clone bitcell from GDS")
    p.add_argument("--bitcell-name", default="sram_cell_6t_122")
    return p.parse_args()

if __name__ == "__main__":
    a = _parse_args()
    generate_stacked_colgrp(a.out, num_bits=a.bits, words=a.words, mux=a.mux,
                            use_mirror=not a.no_mirror, with_dummy=a.with_dummy,
                            bitcell_gds=a.bitcell_gds, bitcell_name=a.bitcell_name)
