"""The part of a BROM macro that does not depend on its ROM contents.

Programming only moves V2 and V3 cuts, so every image of one macro shape has the
same footprint, pins and other layers. A flow can therefore harden against one
stable-named abstract and swap each unit's programmed macro in afterwards. The
fingerprints here let it refuse a macro for which that does not hold.
"""

from __future__ import annotations

import re
from hashlib import sha256
from typing import Any

from ..layout.layers import LAYERS

__all__ = ["PROGRAMMING_LAYERS", "abstract_name", "frozen_geometry_sha256"]

#: The only layers that differ between two images of one macro shape.
PROGRAMMING_LAYERS = ("V2", "V3")

#: The programming digest in a macro name.
_DIGEST = re.compile(r"_[0-9a-f]{12}(?=_|$)")


def abstract_name(name: str) -> str:
    """The macro name `name` without its programming digest."""
    stable, count = _DIGEST.subn("", name)
    if count != 1:
        raise ValueError(f"expected one programming digest in {name!r}, found {count}")
    return stable


def frozen_geometry_sha256(top: Any) -> str:
    """Hash of `top`'s flattened shapes and labels, leaving out V2 and V3.

    Polygons and labels are sorted, since a programmed cell's digest must not
    depend on the order its hierarchy happens to flatten in.
    """
    skipped = {
        (LAYERS[layer]["layer"], LAYERS[layer]["datatype"])
        for layer in PROGRAMMING_LAYERS
    }
    digest = sha256()
    for layer, polygons in sorted(top.get_polygons(by_spec=True).items()):
        if layer in skipped:
            continue
        digest.update(repr(layer).encode())
        # Adding 0.0 turns any -0.0 into 0.0, which differs byte for byte.
        for blob in sorted((p.round(1) + 0.0).tobytes() for p in polygons):
            digest.update(blob)
    labels = (
        (label.text, label.layer, label.texttype, tuple(label.position.round(1) + 0.0))
        for label in top.get_labels()
    )
    digest.update(repr(sorted(labels)).encode())
    return digest.hexdigest()
