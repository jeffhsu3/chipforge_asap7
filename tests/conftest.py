"""Fixtures shared by the layout, DRC, LVS and PEX regressions."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

Box = tuple[float, float, float, float]

#: Set to ``1`` to turn "external tool not installed" skips into failures.
#: Without it a machine that lacks KLayout, the public ASAP7 KLayout runset or
#: the released standard-cell library still reports a green suite while the
#: geometry claims in `chipforge_asap7.devices.finfet` go unverified.
REQUIRE_TOOLS_ENV = "ASAP7_REQUIRE_TOOLS"


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
