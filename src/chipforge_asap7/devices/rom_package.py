"""A programmed NOR ROM's deliverables, and the part that programming leaves alone.

Programming only adds V1s at the array's bit sites, so every image of one macro
shape has the same footprint, pins and other shapes.  A flow can therefore
harden against one stable-named abstract and swap each unit's programmed macro
in afterwards.  The fingerprints here let it refuse a macro for which that does
not hold.

V1 is also the periphery's via, so the fingerprint leaves out only the V1s on
data bit sites: a V1 anywhere else, the replica row's included, is frozen.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..layout.layers import LAYERS
from .cli import write_gds
from .rom_decoder import RomDecoderSpec, build_rom

__all__ = [
    "PROGRAMMING_LAYERS",
    "frozen_geometry_sha256",
    "programming_digest",
    "programming_sites",
    "rom_image_name",
    "sites_from_words",
    "write_rom_package",
]

#: The only layer that differs between two images of one macro shape.
PROGRAMMING_LAYERS = ("V1",)

#: Written into each GDS, so one input gives one file byte for byte.
_TIMESTAMP = datetime(2000, 1, 1, tzinfo=UTC)

#: How long `run_lvs` may take: 1024 x 199 needs most of its default half hour.
_LVS_TIMEOUT_S = 4 * 3600

#: The clock protocol of ``control``, for whoever reads the manifest.
_INTERFACE = (
    "CLK high precharges; CLK falling closes the address latches and starts the "
    "read; {output}; RESETN active low; internal replica feedback"
)


def sites_from_words(words: Any, spec: RomDecoderSpec) -> list[tuple[int, int]]:
    """The ``(word, bit)`` sites holding a 1 in `words`, a list of unsigned ints."""
    array = spec.array
    if not isinstance(words, list):
        raise TypeError("words must be a JSON array of unsigned integers")
    if len(words) != array.words:
        raise ValueError(f"{len(words)} words for a {array.words}-word ROM")
    for word in words:
        # bool is an int to Python, and never what a ROM image meant.
        if not isinstance(word, int) or isinstance(word, bool):
            raise TypeError(f"word {word!r} is not an integer")
        if not 0 <= word < 1 << array.bits:
            raise ValueError(f"word {word:#x} does not fit in {array.bits} bits")
    return [
        (w, b)
        for w, word in enumerate(words)
        for b in range(array.bits)
        if word >> b & 1
    ]


def programming_digest(programmed: Iterable[tuple[int, int]]) -> str:
    """Twelve hex digits naming a set of programmed sites."""
    sites = sorted({(int(w), int(b)) for w, b in programmed})
    return sha256(json.dumps(sites).encode()).hexdigest()[:12]


def rom_image_name(spec: RomDecoderSpec, programmed: Iterable[tuple[int, int]]) -> str:
    """The macro's name when programmed so: its stable name and contents' digest."""
    return f"{spec.macro_name}_p{programming_digest(programmed)}"


def programming_sites(spec: RomDecoderSpec) -> set[tuple[float, float]]:
    """The centre of every data bit's V1 in macro coordinates, programmed or not.

    Not the replica row's: those are programmed in every image.
    """
    half = spec.half
    return {
        (round(x + half.site(word, bit)[0], 1), round(half.site(word, bit)[1], 1))
        for x in spec.half_xs
        for word in range(half.words)
        for bit in range(spec.array.bits)
    }


def frozen_geometry_sha256(spec: RomDecoderSpec, top: Any) -> str:
    """Hash of `top`'s flattened shapes and labels, leaving out programming.

    A V1 centred on a data bit's site is programming, and any other is frozen.
    Polygons and labels are sorted, since an image's digest must not depend on
    the order its hierarchy happens to flatten in.
    """
    via = LAYERS["V1"]["layer"], LAYERS["V1"]["datatype"]
    sites = programming_sites(spec)
    digest = sha256()
    for layer, polygons in sorted(top.get_polygons(by_spec=True).items()):
        digest.update(repr(layer).encode())
        # Adding 0.0 turns any -0.0 into 0.0, which differs byte for byte.
        rounded = [p.round(1) + 0.0 for p in polygons]
        if layer == via:
            rounded = [p for p in rounded if _centre(p) not in sites]
        for blob in sorted(p.tobytes() for p in rounded):
            digest.update(blob)
    labels = (
        (label.text, label.layer, label.texttype, tuple(label.position.round(1) + 0.0))
        for label in top.get_labels()
    )
    digest.update(repr(sorted(labels)).encode())
    return digest.hexdigest()


def _centre(polygon: Any) -> tuple[float, float]:
    x, y = (polygon.min(axis=0) + polygon.max(axis=0)) / 2
    return round(float(x), 1), round(float(y), 1)


def _write_abstract(spec: RomDecoderSpec, top: Any, out: Path) -> dict[str, Any]:
    """Write the stable-named LEF and GDS, and return the manifest's record.

    The fingerprints come from the image `top`, so that two images whose
    geometry parted ways disagree.  The GDS is the macro with no bits set: it
    depends on nothing but the shape, and a unit's image must replace it.
    """
    from .rom_views import rom_macro_lef

    name = spec.macro_name
    lef = rom_macro_lef(spec, top, name=name)
    (out / "abstract.lef").write_text(lef)
    write_gds(
        lambda lib: build_rom(spec, (), name=name, lib=lib),
        out / "abstract.gds",
        timestamp=_TIMESTAMP,
    )
    return {
        "cell": name,
        "lef": "abstract.lef",
        "gds": "abstract.gds",
        "lef_sha256": sha256(lef.encode()).hexdigest(),
        "frozen_geometry_sha256": frozen_geometry_sha256(spec, top),
        "programming_layers": {
            layer: [LAYERS[layer]["layer"], LAYERS[layer]["datatype"]]
            for layer in PROGRAMMING_LAYERS
        },
    }


def _run_drc(gds: Path, cell: str, out: Path, record: dict[str, Any]) -> bool:
    """Record gdscheck's result; True if it found anything."""
    from ..verification.drc import drc_counts, run_gdscheck

    violations = run_gdscheck(gds, out / "drc", cell_name=cell)
    record["drc"] = dict(drc_counts(violations))
    return bool(violations)


def _run_lvs(
    spec: RomDecoderSpec,
    sites: list[tuple[int, int]],
    cell: str,
    out: Path,
    record: dict[str, Any],
) -> bool:
    """Compare the macro to the CDL written beside it; True if they differ.

    On a copy with its unprogrammed drains labelled, since those labels settle
    ties in the compare but do not belong in a delivered GDS.  KLayout compares
    pins by topology, so a match still has to keep every pin's name, and find
    every label the macro should have and no other.
    """
    from ..verification.lvs import run_lvs
    from ..verification.lvs_report import summarize_lvsdb
    from ..verification.rom_macro_lvs import floating_drains, label_floating_drains
    from .rom_netlist import rom_macro_pins, rom_macro_ports

    def labelled(lib: Any) -> Any:
        top = build_rom(spec, sites, name=cell, lib=lib)
        label_floating_drains(spec, sites, top)
        return top

    copy = out / "lvs" / "labelled.gds"
    write_gds(labelled, copy)
    try:
        result = run_lvs(
            copy,
            out / record["cdl"],
            out / "lvs",
            cell_name=cell,
            asap7_standard_cell=True,
            tie_bodies=False,
            merge_fin_diffusion=True,
            flat=True,
            timeout=_LVS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        record["lvs"] = "timed_out"
        return True
    summary = summarize_lvsdb(result.report)
    renamed = [p for c in summary["circuits"] for p in c["renamed_pins"]]
    pins = {
        line.removeprefix("* pin ").strip()
        for line in result.extracted_netlist.read_text().splitlines()
        if line.startswith("* pin ")
    }
    expected = {
        *rom_macro_ports(spec),
        *rom_macro_pins(spec),
        *(net for net, _ in floating_drains(spec, sites)),
    }
    missing, extra = sorted(expected - pins), sorted(pins - expected)
    matched = result.matched and not renamed and not missing and not extra
    record["lvs"] = "passed" if matched else "failed"
    record["renamed_pins"] = renamed
    record["missing_pins"] = missing
    record["extra_pins"] = extra
    return not matched


def write_rom_package(
    spec: RomDecoderSpec,
    programmed: Iterable[tuple[int, int]],
    out: Path,
    *,
    drc: bool = False,
    lvs: bool = False,
) -> bool:
    """Write the macro's package to `out`; True if a check it ran failed.

    ``macro.gds``, ``macro.cdl``, ``macro.lef``, ``macro.v``, with ``register``
    ``macro_{tt,ff,ss}.lib``, the abstract and ``manifest.json``.  The manifest
    is written before each check runs and after it, so a check that dies leaves
    ``not_run`` behind rather than an older run's result.
    """
    from .rom_netlist import rom_macro_cdl, rom_macro_devices, rom_macro_ports
    from .rom_views import rom_macro_views

    sites = sorted(set(programmed))
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    # Invalidate a prior successful report before writing anything it describes.
    manifest.unlink(missing_ok=True)

    name = rom_image_name(spec, sites)
    top, gds = write_gds(
        lambda lib: build_rom(spec, sites, name=name, lib=lib),
        out / "macro.gds",
        timestamp=_TIMESTAMP,
    )
    (out / "macro.cdl").write_text(rom_macro_cdl(spec, sites, name=name))
    views = rom_macro_views(spec, sites, top)
    for filename, text in views.items():
        (out / filename).write_text(text)
    (x0, y0), (x1, y1) = top.get_bounding_box()
    record: dict[str, Any] = {
        "cell": name,
        "transistors": len(rom_macro_devices(spec, sites)),
        "gds": gds.name,
        "cdl": "macro.cdl",
        "lef": "macro.lef",
        "verilog": "macro.v",
        "liberty": {
            filename.removesuffix(".lib").removeprefix("macro_"): filename
            for filename in views
            if filename.endswith(".lib")
        },
        "width_nm": float(x1 - x0),
        "height_nm": float(y1 - y0),
        "drc": "not_run",
        "lvs": "not_run",
    }
    output = (
        "Q updates on CLK rising" if spec.register else "Q follows the sample latches"
    )
    report = {
        "spec": asdict(spec),
        "top_cell": name,
        "programming": {
            "digest": programming_digest(sites),
            "bits_set": len(sites),
            "bits": spec.array.words * spec.array.bits,
        },
        "pins": list(rom_macro_ports(spec)),
        "controller": "replica_feedback",
        "registered_output": spec.register,
        "layout": "nor_center_lane",
        # A narrow array may draw more straps than asked, for the lane's rows.
        "strap_every": spec.cells.strap_every,
        "interface": _INTERFACE.format(output=output),
        "timing_qualified": False,
        "artifacts": {"macro": record},
        "abstract": _write_abstract(spec, top, out),
    }

    def save() -> None:
        manifest.write_text(json.dumps(report, indent=2) + "\n")

    save()
    failed = False
    if drc:
        failed |= _run_drc(gds, name, out, record)
        save()
    if lvs:
        failed |= _run_lvs(spec, sites, name, out, record)
        save()
    return failed
