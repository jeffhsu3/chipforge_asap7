"""Run the public ASAP7 KLayout DRC runset and read back what it found.

The runset (``drc_ASAP7.lydrc`` from ASAP7_for_KLayout) is the DRC authority
for this package.  It is educational collateral, not foundry sign-off, and it
is not redistributed here: `find_drc_deck` looks for it under
``ASAP7_DRC_DECK`` and the usual checkout location.
"""

from __future__ import annotations

import os
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .lvs import find_klayout

__all__ = ["DRC_DECK_ENV", "DRCViolation", "drc_counts", "find_drc_deck", "run_drc"]

DRC_DECK_ENV = "ASAP7_DRC_DECK"
_DEFAULT_DECK = Path.home() / "iv4/repos/ASAP7_for_KLayout/drc/drc_ASAP7.lydrc"
_POINT = re.compile(r"(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)")

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class DRCViolation:
    """One marker: its rule name and, when the report gives one, where (nm)."""

    category: str
    bbox: Box | None


def find_drc_deck(deck: str | Path | None = None) -> Path:
    """Resolve the runset, or raise `FileNotFoundError` saying where it looked."""
    candidates = [deck, os.environ.get(DRC_DECK_ENV), _DEFAULT_DECK]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().is_file():
            return Path(candidate).expanduser().resolve()
    raise FileNotFoundError(
        f"the public ASAP7 KLayout runset was not found; set {DRC_DECK_ENV} "
        f"(looked at {[str(c) for c in candidates if c]})"
    )


def _bbox_nm(value: str) -> Box | None:
    """Bounding box of a report value such as ``polygon: (0.1,0.2;0.3,0.4)`` (um)."""
    points = [
        (float(x) * 1000, float(y) * 1000) for x, y in _POINT.findall(value or "")
    ]
    if not points:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def parse_report(report: str | Path) -> list[DRCViolation]:
    """Read a KLayout ``.lyrdb`` report database."""
    found = []
    for item in ET.parse(report).getroot().findall("./items/item"):
        # The runset writes categories quoted, sometimes with the description
        # glued on: "'V0.AUX.1-2V0.AUX.1-2 : V0 must ...'".
        category = (item.findtext("category") or "").strip("'\"").split(" ")[0]
        found.append(
            DRCViolation(category, _bbox_nm(item.findtext("./values/value") or ""))
        )
    return found


def run_drc(
    gds: str | Path,
    output_dir: str | Path,
    *,
    cell_name: str,
    deck: str | Path | None = None,
    klayout: str | Path | None = None,
    timeout: float = 600,
) -> list[DRCViolation]:
    """Run the runset on `cell_name` of `gds`; an empty list means clean."""
    gds_path = Path(gds).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = out / f"{cell_name}.lyrdb"
    completed = subprocess.run(
        [
            str(find_klayout(klayout)),
            "-b",
            "-r",
            str(find_drc_deck(deck)),
            "-rd",
            f"input={gds_path}",
            "-rd",
            f"topcell={cell_name}",
            "-rd",
            f"output={report}",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0 or not report.is_file():
        raise RuntimeError(
            f"KLayout DRC failed with exit {completed.returncode}: "
            f"{(completed.stdout + completed.stderr)[-1500:]}"
        )
    return parse_report(report)


def drc_counts(violations: list[DRCViolation]) -> Counter[str]:
    return Counter(v.category for v in violations)
