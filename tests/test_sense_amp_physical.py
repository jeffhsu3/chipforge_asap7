"""Optional KLayout DRC/LVS regression for the routed sense amplifier."""

from __future__ import annotations

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import SenseAmpSpec, build_sense_amp
from chipforge_asap7.verification import render_sense_amp_lvs_schematic, run_lvs


def _first_existing(candidates: list[Path | None]) -> Path | None:
    return next(
        (path for path in candidates if path is not None and path.is_file()), None
    )


def _verification_tools() -> tuple[Path | None, Path | None]:
    home = Path.home()
    klayout_from_path = shutil.which("klayout")
    klayout = _first_existing(
        [
            Path(os.environ["KLAYOUT_BIN"]) if os.environ.get("KLAYOUT_BIN") else None,
            Path(klayout_from_path) if klayout_from_path else None,
            home / "iv4/repos/OpenRAM/miniconda/bin/klayout",
        ]
    )
    deck = _first_existing(
        [
            Path(os.environ["ASAP7_DRC_DECK"])
            if os.environ.get("ASAP7_DRC_DECK")
            else None,
            home / "iv4/repos/ASAP7_for_KLayout/drc/drc_ASAP7.lydrc",
        ]
    )
    return klayout, deck


KLAYOUT, ASAP7_DRC_DECK = _verification_tools()


def _write_layout(spec: SenseAmpSpec, path: Path) -> None:
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    build_sense_amp(spec, lib=library)
    library.write_gds(str(path))


@pytest.mark.skipif(KLAYOUT is None, reason="KLayout is not installed")
def test_sense_amp_layout_matches_the_16t_unit_fin_reference(tmp_path: Path):
    spec = SenseAmpSpec(n_fins=2, p_fins=2)
    gds = tmp_path / "sense_amp.gds"
    schematic = tmp_path / "sense_amp.sp"
    _write_layout(spec, gds)
    schematic.write_text(render_sense_amp_lvs_schematic(spec))

    result = run_lvs(
        gds,
        schematic,
        tmp_path / "lvs",
        cell_name=spec.cell_name,
        klayout=KLAYOUT,
    )

    assert result.matched, result.log.read_text()[-4000:]


@pytest.mark.skipif(
    KLAYOUT is None or ASAP7_DRC_DECK is None,
    reason="KLayout and the public ASAP7 DRC deck are not installed",
)
@pytest.mark.parametrize(
    "spec",
    [
        SenseAmpSpec(n_fins=2, p_fins=2),
        SenseAmpSpec(),
        SenseAmpSpec(n_fins=6, p_fins=3, vt="lvt"),
    ],
)
def test_sense_amp_is_clean_in_the_public_asap7_drc_deck(
    tmp_path: Path, spec: SenseAmpSpec
):
    gds = tmp_path / f"{spec.cell_name}.gds"
    report = tmp_path / f"{spec.cell_name}.lyrdb"
    _write_layout(spec, gds)

    result = subprocess.run(
        [
            str(KLAYOUT),
            "-b",
            "-r",
            str(ASAP7_DRC_DECK),
            "-rd",
            f"input={gds}",
            "-rd",
            f"topcell={spec.cell_name}",
            "-rd",
            f"output={report}",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]

    items = ET.parse(report).getroot().findall("./items/item")
    violations = [item.findtext("category") for item in items]
    assert violations == []
