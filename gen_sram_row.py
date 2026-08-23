"""
sram_cell_row compiler — tiles the ASAP7 push-rule 6T bitcell vertically
to form the `sram_cell_row` found in `tech/gds/srambank_32b.gds`.

In that GDS (and matching SPICE `tmp/sram_flat_*.sp:.SUBCKT sram_cell_row`),
a "row" is actually a *column* of bitcells sharing a common BL/BLN pair:

  .SUBCKT sram_cell_row WL[0] ... WL[255] BLN BL VDD VSS  (*256 deep, 1 BL pair*)
    X0 WL[0] BLN BL VDD VSS sram_cell_6t_122
    X1 WL[1] BLN BL VDD VSS sram_cell_6t_122  ... (256 instances)
  .ENDS

Stacked vertically on the FinFET grid:

  CELL_WIDTH  = 2*CPP = 0.108 um  (fixed)
  CELL_HEIGHT =10*FP  = 0.270 um  (fixed)
  Row height  = N * CELL_HEIGHT
  Row width   =     CELL_WIDTH

Mirroring (MY) every other cell is used in real arrays to share
horizontal M2 VDD/VSS rails (`src/layermap.cpp: M1 19:0, M2 20:0`) and keep
fin/active continuity DRC-clean. The bitcell's M2 rails at y=0.036,
0.126, 0.235 ... abut correctly when mirrored. Fin pitch (27 nm) and CPP
(54 nm) are never scaled.

This script reuses `scripts/gen_pushrule_parametric.py` — the fixed
push-rule leaf (no `sx/sy`). For standalone use it inlines the same
constants/polygons so it can generate the bitcell in-memory without an
intermediate GDS file.

Usage:
  python scripts/gen_sram_row.py                          # 256-deep default, like srambank_32b.gds
  python scripts/gen_sram_row.py -o row_64.gds -n 64      # 64-deep
  python scripts/gen_sram_row.py -o row_512.gds -n 512 --no-mirror
  python scripts/gen_sram_row.py --from-gds tech/gds/srambank_32b.gds  # clone GDS hierarchy

Part of the OpenFinRAM FinFET/CFET compiler: row = tile(bitcell) is the
correct parametric level; bitcell W/H are quantized and not scaled.
"""

import argparse
import warnings
from pathlib import Path
import gdstk

# ---------------------------------------------------------------------------
# ASAP7 grids — must match gen_pushrule_parametric.py and src/layermap.cpp
# ---------------------------------------------------------------------------
CPP          = 0.054
FIN_PITCH    = 0.027
CELL_WIDTH   = 2 * CPP        # 0.108
CELL_HEIGHT  = 10 * FIN_PITCH # 0.270
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
    """
    Build the fixed push-rule bitcell in-memory (same geometry as
    gen_pushrule_parametric.generate_pushrule_bitcell, no scaling).
    Extracted from srambank_32b.gds:sram_cell_6t_122.
    """
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
    # GCUT — half-width complementary islands matching srambank_32b.gds:sram_cell_6t_122
    # Do NOT use full 0→0.109; bottom is 0.054→0.1085, top 0→0.054 (0.0545 with 0.0005 overhang)
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
    # keep bitcell labels minimal; row will add indexed WL[] labels
    return cell


def generate_sram_cell_row(
    out_file: str,
    num_cells: int = 256,
    cell_name: str = "sram_cell_row",
    bitcell_name: str = "sram_cell_6t_122",
    use_mirror: bool = True,
    bitcell_gds: str | None = None,
):
    """
    Compile a row (vertical stack sharing BL/BLN) from the push-rule bitcell.

    Each bitcell `i` gets `WL[i]` on M3 (30:0). All share `BL`/`BLN` on M2,
    `VDD`/`VSS` on M2/pwell. No geometry scaling — placement is on the
    CPP/FP grid: `y = i * CELL_HEIGHT` (or mirrored variant).

    When `use_mirror` is True, odd rows are `MY` mirrored (`y = (i+1)*H`
    with `x_reflection=True`) so adjacent cells share `VDD` with `VDD` and
    `VSS` with `VSS`, matching the foundry push-rule array and the GDS
    `srambank_32b.gds:sram_cell_row` (check `BL`/`BLN` M2 continuity).

    If `bitcell_gds` is given, load that GDS's bitcell instead of building
    in-memory (useful to clone exact PDK cell).
    """
    if num_cells <= 0:
        raise ValueError("num_cells must be >=1")
    lib = gdstk.Library(unit=1e-6, precision=1e-10)

    # 1) obtain bitcell
    if bitcell_gds:
        src = gdstk.read_gds(bitcell_gds)
        # Heuristic: cell with BOUNDARY (100:0) and ~0.108x0.27 bbox
        bitcell = None
        for c in src.cells:
            tags = c.get_layer_tags() if hasattr(c, "get_layer_tags") else []
            # fallback: name match
            if c.name == bitcell_name:
                bitcell = c
                break
            if any(gdstk.get_layer(t)==100 for t in tags):
                bitcell = c
                break
        if bitcell is None:
            bitcell = src.cells[0]
        # copy into new library (gdstk requires same lib for ref)
        # Re-create by copying polygons/labels
        new_bc = lib.new_cell(bitcell.name)
        for p in bitcell.polygons:
            new_bc.add(gdstk.Polygon(p.points, p.layer, p.datatype))
        for lbl in bitcell.labels:
            new_bc.add(gdstk.Label(lbl.text, lbl.origin, layer=lbl.layer, texttype=lbl.texttype, magnification=lbl.magnification, rotation=lbl.rotation))
        bitcell = new_bc
    else:
        bitcell = _build_bitcell(lib, name=bitcell_name)

    row = lib.new_cell(cell_name)
    row_height = num_cells * CELL_HEIGHT
    row_width  = CELL_WIDTH

    # 2) tile bitcells
    wl_x = 0.0555  # center of M3 WL in bitcell coords
    wl_y_rel = -0.028  # label y relative to cell origin
    for i in range(num_cells):
        y0 = i * CELL_HEIGHT
        if use_mirror and (i % 2 == 1):
            # MY mirror: reflect across horizontal axis at top of cell
            # gdstk.Reference with x_reflection=True mirrors in y around origin;
            # placing at y0 + CELL_HEIGHT with mirror gives correct abutment.
            ref = gdstk.Reference(bitcell, origin=(0, y0 + CELL_HEIGHT), x_reflection=True)
            wl_y = y0 + CELL_HEIGHT - wl_y_rel - CELL_HEIGHT  # approx; adjust label below
        else:
            ref = gdstk.Reference(bitcell, origin=(0, y0))
            wl_y = y0 + wl_y_rel
        row.add(ref)
        # Row-level indexed WL label on M3 30:251 (pin layer)
        # Place at M3 WL center for this slice; x ~0.0555, y ~ y0 + offset
        label_y = (y0 + CELL_HEIGHT - 0.028) if (use_mirror and i%2==1) else (y0 - 0.028 + 0.055)
        # Use simple y0 + 0.135 center for robustness (middle of cell)
        label_y = y0 + CELL_HEIGHT/2
        # Keep original offset for familiarity but ensure inside cell
        row.add(gdstk.Label(f"WL[{i}]", (wl_x, y0 + 0.135), layer=30, texttype=251))

    # 3) Row-level aggregates: BOUNDARY / SRAMDRC covering full row
    # Do not draw M2 BL/BLN/VDD/VSS polygons here — they are already in
    # each bitcell and abut; adding extras would DRC. Only add row boundary.
    row.add(gdstk.Polygon([[0,0],[row_width,0],[row_width,row_height],[0,row_height]], *L_BOUNDARY))
    row.add(gdstk.Polygon([[0,0],[row_width,0],[row_width,row_height],[0,row_height]], *L_SRAMDRC))

    # Global BL/BLN/VDD/VSS pin labels (M2 20:251) — at row center/top
    row.add(gdstk.Label("BL",  (0.13, row_height/2), layer=20, texttype=251))
    row.add(gdstk.Label("BLN", (-0.018, row_height/2), layer=20, texttype=251))
    row.add(gdstk.Label("VDD!", (0.0795, row_height/2), layer=20, texttype=251))
    row.add(gdstk.Label("VSS!", (0.074, row_height/2), layer=20, texttype=251))

    # SPICE-compatible .SUBCKT view can be emitted alongside GDS by caller
    lib.write_gds(out_file)
    print(f"Generated {cell_name} N={num_cells} {'mirrored' if use_mirror else 'unmirrored'} "
          f"{row_width:.3f}x{row_height:.3f} um -> {out_file} "
          f"(WL[0:{num_cells-1}] BL/BLN VDD VSS)")
    print(f"  Subcircuit: .SUBCKT {cell_name} WL[0:{num_cells-1}] BLN BL VDD VSS")
    return out_file


def _parse_args():
    p = argparse.ArgumentParser(description="Compile sram_cell_row from ASAP7 push-rule bitcell (like srambank_32b.gds)")
    p.add_argument("-o","--out", default="tech/gds/sram_cell_row.gds", help="output GDS")
    p.add_argument("-n","--num-cells", type=int, default=256, help="number of bitcells (WLs) stacked vertically (default 256, as in srambank_32b)")
    p.add_argument("--bitcell-gds", default=None, help="optional GDS to clone bitcell from (else use in-memory push-rule)")
    p.add_argument("--bitcell-name", default="sram_cell_6t_122", help="bitcell cell name")
    p.add_argument("--cell-name", default="sram_cell_row", help="row cell name")
    p.add_argument("--no-mirror", action="store_true", help="disable alternating MY mirror (not recommended for DRC)")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    generate_sram_cell_row(
        out_file=args.out,
        num_cells=args.num_cells,
        cell_name=args.cell_name,
        bitcell_name=args.bitcell_name,
        use_mirror=not args.no_mirror,
        bitcell_gds=args.bitcell_gds,
    )
