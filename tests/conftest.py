"""Fixtures shared by the layout, DRC, LVS and PEX regressions."""

from __future__ import annotations

import os
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import pytest

Box = tuple[float, float, float, float]

#: Set to ``1`` to turn "external tool not installed" skips into failures.
#: Without it a machine that lacks KLayout, the public ASAP7 KLayout runset or
#: the released standard-cell library still reports a green suite while the
#: geometry claims in `chipforge_asap7.devices.finfet` go unverified.
REQUIRE_TOOLS_ENV = "ASAP7_REQUIRE_TOOLS"
#: Overrides the search for the public ASAP7 KLayout rule deck.
DRC_DECK_ENV = "ASAP7_DRC_DECK"


def find_asap7_drc_deck() -> Path | None:
    """Locate the public ASAP7 KLayout runset, if it is installed."""
    override = os.environ.get(DRC_DECK_ENV)
    candidates = [
        Path(override) if override else None,
        Path.home() / "iv4/repos/ASAP7_for_KLayout/drc/drc_ASAP7.lydrc",
    ]
    return next(
        (path for path in candidates if path is not None and path.is_file()), None
    )


def find_klayout_or_none() -> Path | None:
    """The LVS runner's KLayout resolution, without the exception."""
    from chipforge_asap7.verification.lvs import find_klayout

    try:
        return find_klayout()
    except FileNotFoundError:
        return None


@pytest.fixture(autouse=True)
def _restore_gdspy_current_library():
    """Undo any `gdspy.current_library` assignment a test makes.

    Several regressions retarget the process-global library so `write_gds`
    picks up their cells.  `build_finfet` builds with
    ``exclude_from_current=True``, so the leak is harmless today -- but only by
    luck, and only until a test relies on the default library.
    """
    try:
        import gdspy
    except ImportError:  # pragma: no cover - the gds extra is not installed
        yield
        return
    saved = gdspy.current_library
    try:
        yield
    finally:
        gdspy.current_library = saved


@pytest.fixture
def external_tool() -> Callable[[bool, str], None]:
    """Skip on a missing external tool, or fail under `ASAP7_REQUIRE_TOOLS=1`."""

    def _require(available: bool, reason: str) -> None:
        if available:
            return
        if os.environ.get(REQUIRE_TOOLS_ENV) == "1":
            pytest.fail(f"{REQUIRE_TOOLS_ENV}=1 but {reason}")
        pytest.skip(reason)

    return _require


def layer_boxes(cell, layer_name: str) -> list[Box]:
    """Bounding boxes of every polygon on `layer_name`, sorted."""
    from chipforge_asap7.layout import LAYERS

    target = (LAYERS[layer_name]["layer"], LAYERS[layer_name]["datatype"])
    result: list[Box] = []
    for polygon_set in cell.polygons:
        for points, gds_layer, datatype in zip(
            polygon_set.polygons, polygon_set.layers, polygon_set.datatypes
        ):
            if (gds_layer, datatype) == target:
                xs, ys = points[:, 0], points[:, 1]
                result.append(
                    (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
                )
    return sorted(result)


@pytest.fixture
def boxes_on() -> Callable[[object, str], list[Box]]:
    """`layer_boxes` as a fixture, for tests that read drawn geometry."""
    return layer_boxes


@pytest.fixture(scope="session")
def released_library() -> Path:
    """Root of the public ASAP7 7.5-track standard-cell release, if present."""
    return Path.home() / "iv4/repos/asap7/asap7sc7p5t_28"


@pytest.fixture(scope="session")
def released_sram_gds() -> Path:
    """The public ASAP7 SRAM bank GDS, if present.

    `dec_inv_62f_halved_AND` lives in here; it is the cell
    `chipforge_asap7.devices.inverter` is modelled on, and the only place the
    released collateral shows a tapless, stacked-band, shared-diffusion cell.
    """
    return Path.home() / "iv4/repos/asap7/asap7_sram_0p0/gds/srambank_32b.gds"


@pytest.fixture
def require_klayout(external_tool) -> Path:
    """The KLayout binary the verification runners use; skip if it is missing."""
    klayout = find_klayout_or_none()
    external_tool(klayout is not None, "KLayout is not installed (set KLAYOUT_BIN)")
    return klayout


@pytest.fixture
def asap7_drc(external_tool, tmp_path) -> Callable[..., list[str]]:
    """Run ASAP7 DRC over a library, and report what it found.

    The engine is gdscheck (``--suite main``) unless ``ASAP7_DRC_ENGINE=klayout``
    picks the public KLayout runset.  Returns the violation *categories*
    rather than a pass/fail, because a cell can be as clean as a cell can be
    and still trip rules that only a placed design satisfies -- latch-up
    without a tap row, implant enclosure at the end of an abutting row.
    Asserting the exact category set keeps those documented instead of
    waived.
    """
    from chipforge_asap7.verification.drc import default_drc_engine, find_gdscheck, run_gdscheck

    def _run(library, top, tag: str = "drc") -> list[str]:
        gds = tmp_path / f"{tag}.gds"
        if default_drc_engine() == "gdscheck":
            try:
                find_gdscheck()
                found = True
            except FileNotFoundError:
                found = False
            external_tool(found, "gdscheck is not installed (set GDSCHECK)")
            library.write_gds(str(gds))
            return [v.category for v in run_gdscheck(gds, tmp_path / f"{tag}_drc", cell_name=top.name)]
        klayout = find_klayout_or_none()
        deck = find_asap7_drc_deck()
        external_tool(
            klayout is not None and deck is not None,
            "KLayout and the public ASAP7 DRC deck are not installed "
            f"(set KLAYOUT_BIN and {DRC_DECK_ENV})",
        )
        report = tmp_path / f"{tag}.lyrdb"
        library.write_gds(str(gds))
        result = subprocess.run(
            [
                str(klayout),
                "-b",
                "-r",
                str(deck),
                "-rd",
                f"input={gds}",
                "-rd",
                f"topcell={top.name}",
                "-rd",
                f"output={report}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
        items = ET.parse(report).getroot().findall("./items/item")
        # The runset writes categories quoted, e.g. "'SDT.W.3'".
        return [(item.findtext("category") or "").strip("'\"") for item in items]

    return _run
