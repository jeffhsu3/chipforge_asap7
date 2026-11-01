"""Measure dense BROM units and shape matrix; optionally compare archived areas."""

import argparse
import json
from pathlib import Path

from chipforge_asap7.brom import (
    BromSpec,
    build_macro,
    build_periphery,
    cgen_circuit,
    controller_circuit,
    decoder_circuit,
    macro_circuit,
    output_circuit,
    output_register_circuit,
    self_timed_macro_circuit,
)
from chipforge_asap7.brom.dense_macro import dense_macro_layout
from chipforge_asap7.brom.periphery import geometry_digest, periphery_plan


def measure(c):
    top = (build_macro if c.array else build_periphery)(c)
    bounds = top.get_bounding_box()
    width, height = map(float, bounds[1] - bounds[0])
    physical = dense_macro_layout(c)[1] if c.array else None
    certificates = (
        physical["placement_certificates"]
        if physical
        else periphery_plan(c).certificates
    )
    result = {
        "width_nm": width,
        "height_nm": height,
        "area_um2": width * height / 1e6,
        "transistors": len(c.devices),
        "geometry_digest": geometry_digest(top),
        "placement_certificates": certificates,
    }
    if c.array:
        result["physical_sha256"] = dense_macro_layout(c)[1]["physical_sha256"]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path, help="Archived channel-layout area.json")
    p.add_argument("--matrix", action="store_true")
    args = p.parse_args()
    cases = {
        "cgen": cgen_circuit(),
        "decoder_16x4": decoder_circuit(BromSpec(16, 4)),
        "output": output_circuit(),
        "controller": controller_circuit(),
        "register_4": output_register_circuit(4),
        "selftimed_registered_4x4": self_timed_macro_circuit(
            BromSpec(4, 4, registered_output=True), [10, 12, 0, 0]
        ),
    }
    if args.matrix:
        for d, b, w in (
            (4, 1, 1),
            (4, 4, 1),
            (4, 16, 1),
            (16, 9, 1),
            (16, 4, 4),
            (64, 16, 4),
            (1024, 16, 4),
        ):
            words = [
                sum(
                    (((bit * w + a % w) % 16) >> (a // w % 4) & 1) << bit
                    for bit in range(b)
                )
                for a in range(d)
            ]
            cases[f"phases_{d}x{b}_m{w}"] = macro_circuit(BromSpec(d, b, w), words)
    reference = (
        json.loads(args.reference.read_text())["cases"] if args.reference else {}
    )
    results = {}
    for name, c in cases.items():
        r = measure(c)
        if name in reference:
            old = reference[name]["compact"]["area_um2"]
            r.update(archived_channel_area_um2=old, area_ratio=old / r["area_um2"])
        results[name] = r
        print(f"{name}: {r['area_um2']:.3f} um^2", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "scope": "drawn bounding boxes; dense topology and placement; no global minimum or timing claim",
                "reference": str(args.reference) if args.reference else None,
                "cases": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
