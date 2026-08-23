"""The package must stay importable and reusable outside this repo.

Guards the properties a consumer depends on: a dependency-free import, a public
API that matches its `__all__`, and a version that agrees with the installed
distribution metadata.
"""

import subprocess
import sys
from importlib import metadata

import pytest

import chipforge_asap7
from chipforge_asap7 import layout


def test_import_does_not_require_gdspy():
    """Importing the package must not pull in a GDS library.

    Callers that only need pitches / alignment math (LEF, Liberty, sizing and
    DRC scripts) depend on this — it is why gdspy is an extra, not a dependency.
    """
    code = (
        "import sys, chipforge_asap7, chipforge_asap7.layout as L;"
        "L.column_gate_track(3); L.centered_fin_ys(81, 2);"
        "assert 'gdspy' not in sys.modules, sorted(m for m in sys.modules if 'gds' in m);"
        "print('clean')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "clean"


def test_public_api_is_importable():
    """Every name in `__all__` resolves — a stale re-export list breaks consumers."""
    missing = [name for name in layout.__all__ if not hasattr(layout, name)]
    assert not missing
    # The two submodules stay reachable for callers that prefer explicit paths.
    from chipforge_asap7.layout import grid, layers

    assert set(grid.__all__) | set(layers.__all__) == set(layout.__all__)


def test_version_matches_distribution_metadata():
    try:
        installed = metadata.version("chipforge-asap7")
    except metadata.PackageNotFoundError:
        pytest.skip("chipforge-asap7 is not installed in this environment")
    assert installed == chipforge_asap7.__version__


def test_py_typed_marker_ships():
    """Type hints are only visible to consumers when PEP 561 marker is present."""
    from pathlib import Path

    assert (Path(chipforge_asap7.__file__).parent / "py.typed").exists()


def test_have_gdspy_reports_this_environment():
    assert layout.have_gdspy() is True  # dev extra installs gdspy
