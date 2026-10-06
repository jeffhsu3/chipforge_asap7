"""Run ASAP7 DRC and read back what it found.

Two engines.  gdscheck (``--process asap7 --suite main``) is the default and
the DRC authority for this package: its rules follow the DRM's text.  The
public KLayout runset (``drc_ASAP7.lydrc`` from ASAP7_for_KLayout) remains
available (``ASAP7_DRC_ENGINE=klayout``), but some of its rules do not read as
written: ACTIVE.W.2 and SDT.W.3 list heights of 1-12 fins and flag any other,
M1.S.2 skips any polygon with an edge of 36 nm or more.  Neither is foundry
sign-off; neither deck is redistributed here.  `find_gdscheck` looks under
``GDSCHECK``, ``PATH`` and the usual checkouts, `find_drc_deck` under
``ASAP7_DRC_DECK`` and the usual checkout location.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .lvs import find_klayout

__all__ = [
    "DRC_DECK_ENV",
    "DRC_ENGINE_ENV",
    "DRCViolation",
    "GDSCHECK_ENV",
    "default_drc_engine",
    "drc_counts",
    "find_drc_deck",
    "find_gdscheck",
    "run_drc",
    "run_gdscheck",
]

DRC_DECK_ENV = "ASAP7_DRC_DECK"
DRC_ENGINE_ENV = "ASAP7_DRC_ENGINE"
GDSCHECK_ENV = "GDSCHECK"
_DEFAULT_DECK = Path.home() / "iv4/repos/ASAP7_for_KLayout/drc/drc_ASAP7.lydrc"
_DEFAULT_GDSCHECKS = (
    Path.home() / "iv4/repos/OpenFinRAM/build/tools/gdscheck/bin/gdscheck",
    Path.home() / "iv4/repos/gdscheck/target/release/gdscheck",
)
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


def default_drc_engine() -> str:
    """``gdscheck`` unless ``ASAP7_DRC_ENGINE`` says ``klayout``."""
    engine = os.environ.get(DRC_ENGINE_ENV, "gdscheck").strip().lower()
    if engine not in ("gdscheck", "klayout"):
        raise ValueError(f"{DRC_ENGINE_ENV} must be gdscheck or klayout, got {engine!r}")
    return engine


def find_gdscheck(binary: str | Path | None = None) -> Path:
    """Resolve a gdscheck with the full ASAP7 process, or raise `FileNotFoundError`."""
    candidates = [binary, os.environ.get(GDSCHECK_ENV), shutil.which("gdscheck"), *_DEFAULT_GDSCHECKS]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().is_file() and os.access(Path(candidate).expanduser(), os.X_OK):
            return Path(candidate).expanduser().resolve()
    raise FileNotFoundError(
        f"gdscheck was not found; set {GDSCHECK_ENV} (looked at {[str(c) for c in candidates if c]})"
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


def run_gdscheck(
    gds: str | Path,
    output_dir: str | Path,
    *,
    cell_name: str,
    binary: str | Path | None = None,
    suite: str = "main",
    timeout: float = 600,
) -> list[DRCViolation]:
    """Run gdscheck's ASAP7 `suite` on `cell_name` of `gds`; an empty list means clean.

    One `DRCViolation` per marker (an item's multiplicity counted out).
    """
    gds_path = Path(gds).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = out / f"{cell_name}.lyrdb"
    report.unlink(missing_ok=True)
    completed = subprocess.run(
        [str(find_gdscheck(binary)), "run", "--input", str(gds_path), "--process", "asap7",
         "--suite", suite, "--topcell", cell_name, "--report", str(report)],
        check=False, capture_output=True, text=True, timeout=timeout,
    )  # fmt: skip
    (out / f"{cell_name}.gdscheck.log").write_text(completed.stdout + completed.stderr)
    # 0 clean, 2 violations; anything else did not finish.
    if completed.returncode not in (0, 2) or not report.is_file():
        raise RuntimeError(
            f"gdscheck failed with exit {completed.returncode}: "
            f"{(completed.stdout + completed.stderr)[-1500:]}"
        )
    found = []
    for item in ET.parse(report).getroot().findall("./items/item"):
        category = (item.findtext("category") or "").strip("'\"").split(" ")[0]
        count = int(item.findtext("multiplicity") or 1)
        found += [DRCViolation(category, _bbox_nm(item.findtext("./values/value") or ""))] * count
    return found


def run_drc(
    gds: str | Path,
    output_dir: str | Path,
    *,
    cell_name: str,
    deck: str | Path | None = None,
    klayout: str | Path | None = None,
    timeout: float = 600,
    engine: str | None = None,
) -> list[DRCViolation]:
    """Run DRC on `cell_name` of `gds` (`engine`, else `default_drc_engine`); empty means clean."""
    if (engine or default_drc_engine()) == "gdscheck":
        return run_gdscheck(gds, output_dir, cell_name=cell_name, timeout=timeout)
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
