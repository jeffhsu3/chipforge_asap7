"""The 8T bitcell family the 1R1W column is built from, read from OpenFinRAM.

OpenFinRAM generates an ASAP7 true-dual-port 8T cell (``sram_cell_8t``,
108 x 594 nm) from the published ``sram_cell_6t_122`` with a second access
pair, its mirrored-slot variants, the array edge family and the well tap
(OpenFinRAM ``docs/asap7_8t_bitcell.md``).  Its libraries are build output
there (``scripts/tech_gds.py``); git keeps their geometry digests in
``tech/gds/digests.json``.  `load_cell_library` reads them and refuses any
whose geometry is not the recorded one.

The 1R1W column uses the cell's port A only to write and port B only to read:
``WLA`` is the write wordline and ``BLA``/``BLAN`` the write bitlines,
``WLB``, ``BLB``/``BLBN`` the read ones.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..layout.transform import edge_boundary, require_gdstk

__all__ = [
    "BITCELL",
    "CellLibrary",
    "LIBRARIES",
    "default_gds_dir",
    "geometry_digest",
    "load_cell_library",
]

BITCELL = "sram_cell_8t"
#: Mirrored-slot variant: its WLB landings sit on other M4 tracks, so mirrored
#: neighbours never meet landing to landing.
BITCELL_MIRRORED = "sram_cell_8t_b"
#: The untapped row's last mirrored cell, which keeps its bitline vias inside.
BITCELL_MIRRORED_END = "sram_cell_8t_b_end"
TAP = "tapcell_sram_8t"
FILLER = "FILLER_cgedge_8t"
#: The OpenFinRAM libraries read, in ``tech/gds``.
LIBRARIES = ("sram_cell_8t", "sram_cell_8t_edges", "sram_cell_8t_tap")


def default_gds_dir() -> Path:
    """``$OPENFINRAM_ROOT/tech/gds``, or OpenFinRAM beside this repository."""
    root = os.environ.get("OPENFINRAM_ROOT")
    if root:
        return Path(root) / "tech" / "gds"
    return Path(__file__).resolve().parents[3].parent / "OpenFinRAM" / "tech" / "gds"


def geometry_digest(library: Any) -> str:
    """SHA-256 of a gdstk library's geometry, as OpenFinRAM's ``tech_gds.py`` takes it.

    Polygons, labels and references at 0.01 nm (the file in um), sorted, so
    the digest does not depend on record order.
    """

    def q(value: float) -> int:
        return round(value * 1e5)

    cells = []
    for cell in library.cells:
        polygons = sorted(
            (p.layer, p.datatype, [(q(x), q(y)) for x, y in p.points])
            for p in cell.polygons
        )
        labels = sorted(
            (lb.text, lb.layer, lb.texttype, q(lb.origin[0]), q(lb.origin[1]))
            for lb in cell.labels
        )
        references = sorted(
            (
                r.cell.name if hasattr(r.cell, "name") else str(r.cell),
                q(r.origin[0]),
                q(r.origin[1]),
                round(r.rotation or 0, 9),
                r.magnification or 1,
                bool(r.x_reflection),
                [(q(x), q(y)) for x, y in r.repetition.get_offsets()]
                if r.repetition is not None and r.repetition.size
                else [],
            )
            for r in cell.references
        )
        cells.append((cell.name, polygons, labels, references))
    cells.sort(key=lambda cell: cell[0])
    return hashlib.sha256(json.dumps(cells, separators=(",", ":")).encode()).hexdigest()


@dataclass
class CellLibrary:
    """OpenFinRAM's 8T family, every cell in one gdstk library (um)."""

    library: Any
    cells: dict[str, Any]
    digests: dict[str, str]
    source: Path

    @property
    def bitcell(self) -> Any:
        return self.cells[BITCELL]

    @property
    def tap(self) -> Any:
        return self.cells[TAP]

    def boundary(self, name: str = BITCELL) -> tuple[float, float, float, float]:
        return edge_boundary(self.cells[name])

    @property
    def width(self) -> float:
        """One wordline slot, um."""
        x0, _, x1, _ = self.boundary()
        return x1 - x0

    @property
    def height(self) -> float:
        """One bitline row, um."""
        _, y0, _, y1 = self.boundary()
        return y1 - y0


def load_cell_library(
    gds_dir: Path | None = None, *, check_digests: bool = True
) -> CellLibrary:
    """Read `LIBRARIES` from `gds_dir` (`default_gds_dir`) into one library.

    Each file's geometry must match OpenFinRAM's ``digests.json`` beside it:
    a library regenerated with another geometry would silently change what
    the column is built of.
    """
    gdstk = require_gdstk()
    gds_dir = Path(gds_dir) if gds_dir is not None else default_gds_dir()
    if not gds_dir.is_dir():
        raise FileNotFoundError(
            f"no OpenFinRAM cell libraries at {gds_dir}: set OPENFINRAM_ROOT, "
            "or build them there with scripts/tech_gds.py"
        )
    recorded = {}
    if check_digests:
        recorded = json.loads((gds_dir / "digests.json").read_text())
    merged = gdstk.Library("sram1r1w_cells", unit=1e-6, precision=2.5e-10)
    cells: dict[str, Any] = {}
    digests = {}
    for name in LIBRARIES:
        path = gds_dir / f"{name}.gds"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing: build it with scripts/tech_gds.py"
            )
        library = gdstk.read_gds(str(path), unit=1e-6)
        digests[name] = geometry_digest(library)
        if check_digests and digests[name] != recorded.get(name):
            raise ValueError(
                f"{path}'s geometry is not the one OpenFinRAM recorded "
                f"({digests[name][:12]} vs {str(recorded.get(name))[:12]}): regenerate it"
            )
        for cell in library.cells:
            if cell.name in cells:
                continue  # the edge and tap libraries repeat the bitcell
            cells[cell.name] = cell
            merged.add(cell)
    for required in (BITCELL, BITCELL_MIRRORED, BITCELL_MIRRORED_END, TAP, FILLER):
        if required not in cells:
            raise ValueError(f"{gds_dir}: no {required}")
    return CellLibrary(merged, cells, digests, gds_dir)
