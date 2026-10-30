"""Compare emitted BROM peripheral geometry with CP-SAT channel compaction."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from chipforge_asap7.brom import (
    BromSpec,
    build_macro,
    build_periphery,
    cgen_circuit,
    compact_periphery,
    controller_circuit,
    decoder_circuit,
    output_circuit,
    output_register_circuit,
    self_timed_macro_circuit,
)
from chipforge_asap7.brom.layout import macro_periphery
from chipforge_asap7.brom.periphery import geometry_digest, periphery_plan


def measure(c):
    top = (build_macro if c.array else build_periphery)(c)
    bounds = top.get_bounding_box()
    width, height = map(float, bounds[1] - bounds[0])
    bank = periphery_plan(macro_periphery(c) if c.array else c)
    return {
        "width_nm": width,
        "height_nm": height,
        "area_um2": width * height / 1e6,
        "periphery_area_um2": bank.width * (bank.height + 54) / 1e6,
        "circuit_sha256": hashlib.sha256(c.netlist().encode()).hexdigest(),
        "geometry_digest": geometry_digest(top),
        "optimization": bank.optimization,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
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
    results = {}
    for name, c in cases.items():
        baseline = measure(c)
        optimized = measure(compact_periphery(deepcopy(c)))
        results[name] = {
            "baseline": baseline,
            "compact": optimized,
            "area_reduction_percent": 100
            * (1 - optimized["area_um2"] / baseline["area_um2"]),
            "periphery_reduction_percent": 100
            * (1 - optimized["periphery_area_um2"] / baseline["periphery_area_um2"]),
        }
        print(f"{name}: {results[name]['area_reduction_percent']:.2f}% less top area")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "scope": "drawn bounding boxes; same devices; no timing qualification",
                "cases": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
