"""The extracted grid must draw byte-identical geometry to chipforge's compilers.

These helpers were copied out of chipforge's array compilers, which still carry
their own inlined copies — so this guards against the two drifting apart.  Skips
when the chipforge checkout (or the ASAP7 PDK GDS it loads) is unavailable.
"""

import os
import sys
from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.layout import (
    LAYERS,
    centered_fin_ys,
    draw_fin_grid,
    draw_gate_cuts,
    draw_gate_grid,
    row_boundary_ys,
)


def _find_chipforge_scripts() -> Path | None:
    """Locate a chipforge checkout: $CHIPFORGE_ROOT, else a sibling of this repo."""
    if env := os.environ.get("CHIPFORGE_ROOT"):
        candidates = [Path(env).expanduser()]
    else:
        # This repo may sit beside chipforge, or under a parallel workspace
        # (~/iv4/repos/chipforge_asap7 alongside ~/iv3/repos/chipforge).
        roots = list(Path(__file__).resolve().parents[2:5])
        candidates = [r / "chipforge" for r in roots]
        candidates += [r / "repos" / "chipforge" for r in roots]
        candidates += [p for r in roots for p in sorted(r.glob("*/repos/chipforge"))]
    for root in candidates:
        if (root / "scripts" / "rom_compiler.py").exists():
            return root / "scripts"
    return None


SCRIPTS = _find_chipforge_scripts()

pytestmark = pytest.mark.skipif(
    SCRIPTS is None,
    reason="no chipforge checkout found; set CHIPFORGE_ROOT to run parity tests",
)


@pytest.fixture(scope="module")
def scripts():
    """Import the chipforge array compilers as modules."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    mods = {}
    for name in (
        "rom_compiler",
        "brom_compiler",
        "pseudo_nmos_compiler",
        "gen_rom_bitcell",
    ):
        mods[name] = pytest.importorskip(name)
    return mods


def direct_polys(cell, layer_name, y_max=None):
    """Bounding boxes of polygons added directly to `cell` (references excluded)."""
    spec = (LAYERS[layer_name]["layer"], LAYERS[layer_name]["datatype"])
    out = []
    for pset in cell.polygons:
        for pts, lyr, dt in zip(pset.polygons, pset.layers, pset.datatypes):
            if (lyr, dt) != spec:
                continue
            xs, ys = pts[:, 0], pts[:, 1]
            bbox = (
                round(xs.min(), 4),
                round(ys.min(), 4),
                round(xs.max(), 4),
                round(ys.max(), 4),
            )
            if y_max is None or bbox[1] <= y_max:
                out.append(bbox)
    return sorted(out)


def test_matches_rom_compiler(scripts):
    """rom_compiler.generate_rom_full: array substrate under the NOR ROM."""
    rom = scripts["rom_compiler"]
    depth, num_bits = 8, 12
    ref = rom.generate_rom_full(
        depth=depth, num_bits=num_bits, pattern=None, top_cell_name="REF_ROM"
    ).cells["REF_ROM"]

    left = rom._decoder_col_width(depth, num_bits, rom._load_platform_cells("asap7"))
    left = ((left + 107) // 108) * 108
    total_w = left + (num_bits + 1) * rom.PITCH_X
    height = (depth + 1) * rom.WL_PITCH

    new = gdspy.Cell("NEW_ROM", exclude_from_current=True)
    draw_fin_grid(new, left, total_w, 0, height)
    draw_gate_grid(new, left, total_w, -1200.0, height)
    draw_gate_cuts(
        new, [(left - 17, total_w - 17)], row_boundary_ys(depth, rom.WL_PITCH)
    )

    for lyr in ("FIN", "GATE", "GATE_CUT"):
        assert direct_polys(new, lyr) == direct_polys(ref, lyr), lyr


def test_matches_brom_compiler(scripts):
    """brom_compiler.generate_brom_array_slave: 108 nm candidate-row pitch."""
    brom = scripts["brom_compiler"]
    n_cands, n_bits = 16, 4
    ref = brom.generate_brom_array_slave(num_cands=n_cands, num_bits=n_bits).cells[
        "BROM_ARRAY_SLAVE"
    ]
    width = (n_bits + 1) * brom.PITCH_X
    height = (n_cands + 1) * brom.PITCH_Y

    new = gdspy.Cell("NEW_BROM", exclude_from_current=True)
    draw_fin_grid(new, 0, width, 0, height)
    draw_gate_grid(new, 0, width, -4.0, height)
    draw_gate_cuts(new, [(-100, width + 100)], row_boundary_ys(n_cands, brom.PITCH_Y))

    for lyr in ("FIN", "GATE", "GATE_CUT"):
        assert direct_polys(new, lyr) == direct_polys(ref, lyr), lyr


def test_matches_pseudo_nmos_compiler(scripts):
    """pseudo_nmos_compiler.generate_pnmos_full: same substrate, ratioed array."""
    rom, pnm = scripts["rom_compiler"], scripts["pseudo_nmos_compiler"]
    depth, num_bits = 8, 8
    ref = pnm.generate_pnmos_full(
        depth=depth,
        num_bits=num_bits,
        pattern=None,
        plan=pnm.plan_load(),
        top_cell_name="REF_PNM",
    ).cells["REF_PNM"]

    left = rom._decoder_col_width(depth, num_bits, rom._load_platform_cells("asap7"))
    left = ((left + 107) // 108) * 108
    total_w = left + (num_bits + 1) * pnm.PITCH_X
    array_h = (depth + 1) * pnm.WL_PITCH

    new = gdspy.Cell("NEW_PNM", exclude_from_current=True)
    draw_fin_grid(new, left, total_w, 0, array_h)
    draw_gate_grid(new, left, total_w, -1200.0, array_h)
    draw_gate_cuts(
        new, [(left - 17, total_w - 17)], row_boundary_ys(depth, pnm.WL_PITCH)
    )

    for lyr in ("FIN", "GATE", "GATE_CUT"):
        # The load band above the array carries its own well-tap fins; the
        # substrate grid stops at array_h.
        assert direct_polys(new, lyr) == direct_polys(ref, lyr, y_max=array_h), lyr


def test_matches_bitcell_fin_centering(scripts):
    """gen_rom_bitcell centres its active fins the same way."""
    grb = scripts["gen_rom_bitcell"]
    for cell_h, fins in ((81, 2), (108, 2), (135, 3)):
        # Distinct names: gdspy keeps a process-global current library.
        name = f"ROM_BitCell_{cell_h}_{fins}"
        cell = grb.generate_rom_bitcell(
            pdk=grb.ASAP7, cell_h=cell_h, cpp_width=2, fins=fins, cell_name=name
        ).cells[name]
        ref_ys = sorted(y0 for _, y0, _, _ in direct_polys(cell, "FIN"))
        assert ref_ys == [float(y) for y in centered_fin_ys(cell_h, fins)]
