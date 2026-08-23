#!/usr/bin/env python3
"""
Tiled decoder grid using golden handcrafted cells from srambank_32b.gds.
Instances NAND3x1 and custom predecode cells instead of raw polygons.
"""
import gdstk
import sys, os

GOLDEN_GDS = "/home/jeff/iv4/repos/asap7/asap7_sram_0p0/gds/srambank_32b.gds"

CELL_MAP = {
    "nand3": "NAND3x1_ASAP7_75t_R_predecode",
    "predecode": "predecode_sram_6t122_32bit_AND_P1N1",
    "dec_nand": "dec_nand_12f_12f_for_and_size_reduced_post_decode_P1N1",
    "decoder_5to32": "decoder_5to32_sram_6t122_and_halved_P1N1_A_original",
}

def generate(addr_width=4, num_wl=32, mode="nand3",
             col_pitch=0.65, row_pitch=0.35, out_dir="tmp"):
    os.makedirs(out_dir, exist_ok=True)
    # Load golden lib (contains all sub-cells with references resolved)
    golden = gdstk.read_gds(GOLDEN_GDS)
    lib = gdstk.Library(name="predecode_inst")
    cell = lib.new_cell("predecode_grid")

    src_cell_name = CELL_MAP.get(mode, CELL_MAP["nand3"])
    # Find source cell in golden
    src_cell = None
    for c in golden.cells:
        if c.name == src_cell_name:
            src_cell = c
            break
    if src_cell is None:
        raise ValueError(f"Cell {src_cell_name} not found in {GOLDEN_GDS}")

    # Place tiled instances in regular grid
    # Layout: horizontal strip of instances, each representing a pre-decode term
    terms = []
    for i in range(addr_width):
        terms.append(f"A{i}")
        terms.append(f"~A{i}")

    # Place instances in a grid: rows = terms, cols = 1 (strip) or tiled by num_wl groups
    for idx, term in enumerate(terms):
        y = idx * row_pitch
        # Place one instance; for a real decoder you'd tile across columns
        ref = gdstk.Reference(src_cell, (0.1, y))
        cell.add(ref)
        # Label
        lbl = gdstk.Label(
            text=term,
            origin=(0.05, y + 0.05),
            magnification=0.02,
            layer=30,
            texttype=0,
        )
        cell.add(lbl)

    # Optionally place full decoder block at bottom (option 3 hybrid)
    full_dec_name = CELL_MAP["decoder_5to32"]
    full_dec = None
    for c in golden.cells:
        if c.name == full_dec_name:
            full_dec = c
            break
    if full_dec and mode == "decoder_5to32":
        # Place as fixed block at bottom center
        h = len(terms) * row_pitch + 0.2
        ref = gdstk.Reference(full_dec, (0.1, h))
        cell.add(ref)

    out_path = os.path.join(out_dir, "predecode_grid_inst.gds")
    lib.write_gds(out_path)
    print(f"Wrote {out_path} (mode={mode}, terms={len(terms)}, src={src_cell_name})")
    # Print bbox of source for reference
    (xmin,ymin),(xmax,ymax) = src_cell.bounding_box()
    print(f"  Source bbox: w={xmax-xmin:.3f} h={ymax-ymin:.3f}")
    return out_path

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "nand3"
    a = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    generate(addr_width=a, mode=m)
