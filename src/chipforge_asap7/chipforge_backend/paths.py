"""Where the ASAP7 ROM banks live: core's ``macros/``, not this repository.

``CHIPFORGE_ASAP7_MACROS`` when set (as chipforge_gt2n reads
``CHIPFORGE_GT2N_MACROS``), else the ``macros/`` of the chipforge checkout
this environment imports, else ``macros/`` under the current directory, where
core's workflows run.
"""

from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path

ENV = "CHIPFORGE_ASAP7_MACROS"


def rom_macros_dir() -> Path:
    """The directory holding ``asap7_brom/`` and ``asap7_nor_rom/`` banks."""
    value = os.environ.get(ENV)
    if value:
        return Path(value).expanduser()
    spec = find_spec("chipforge")
    if spec is not None and spec.origin:
        # src/chipforge/__init__.py -> the checkout, when installed editable.
        checkout = Path(spec.origin).resolve().parents[2]
        if (checkout / "pyproject.toml").is_file():
            return checkout / "macros"
    return Path("macros")
