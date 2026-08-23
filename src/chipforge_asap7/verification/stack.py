"""ASAP7 process stack used by the open KPEX/FasterCap flow.

Only part of the vertical/material stack is public.  Values recovered from the
released xACT3D standard-cell netlists are marked ``released_xact``; geometry
from the ASAP7 process paper is marked ``public_geometry``; the remaining
dielectric and upper-metal values are explicit ``assumption`` or
``extrapolation`` entries.  The generated protobuf JSON intentionally contains
only fields understood by KPEX.  :func:`calibration_manifest` carries the
provenance and uncertainty alongside it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "STACK_CONDUCTORS",
    "Conductor",
    "calibration_manifest",
    "technology_definition",
    "write_kpex_technology",
]


@dataclass(frozen=True)
class Conductor:
    """One routed conductor in the open extraction stack (dimensions in um)."""

    name: str
    lvs_name: str
    gds_layer: int
    z_um: float
    thickness_um: float
    sheet_resistance_ohm: float
    provenance: str
    relative_uncertainty: float
    via_above: str | None = None
    via_gds_layer: int | None = None
    via_width_um: float | None = None
    via_resistance_ohm: float | None = None


# Z locations preserve released thicknesses and the paper's 2:1 metal aspect
# ratio.  Absolute dielectric gaps above M1 are not published; they are a
# monotonic stack suitable for field solving, then calibrated/bounded against
# the released INVxp33 xACT view.
STACK_CONDUCTORS: tuple[Conductor, ...] = (
    Conductor(
        "Gate",
        "gate_con",
        7,
        0.100,
        0.056,
        7.14290,
        "released_xact",
        0.20,
        "gate_lig_link",
        16,
        0.020,
        0.1,
    ),
    Conductor(
        "Local",
        "local_con",
        16,
        0.157,
        0.048,
        8.33353,
        "released_xact",
        0.35,
        "v0_local",
        18,
        0.018,
        19.3796,
    ),
    Conductor(
        "M1",
        "m1",
        19,
        0.223,
        0.036,
        3.03147,
        "released_xact",
        0.18,
        "v1",
        21,
        0.018,
        19.5347,
    ),
    Conductor(
        "M2",
        "m2",
        20,
        0.277,
        0.036,
        3.03147,
        "released_xact",
        0.18,
        "v2",
        25,
        0.018,
        19.5347,
    ),
    Conductor(
        "M3",
        "m3",
        30,
        0.331,
        0.036,
        3.03147,
        "public_geometry_extrapolation",
        0.35,
        "v3",
        35,
        0.018,
        19.5347,
    ),
    Conductor(
        "M4",
        "m4",
        40,
        0.391,
        0.048,
        2.27360,
        "public_geometry_extrapolation",
        0.40,
        "v4",
        45,
        0.024,
        17.0,
    ),
    Conductor(
        "M5",
        "m5",
        50,
        0.463,
        0.048,
        2.27360,
        "public_geometry_extrapolation",
        0.40,
        "v5",
        55,
        0.024,
        17.0,
    ),
    Conductor(
        "M6",
        "m6",
        60,
        0.543,
        0.064,
        1.70520,
        "public_geometry_extrapolation",
        0.45,
        "v6",
        65,
        0.032,
        14.0,
    ),
    Conductor(
        "M7",
        "m7",
        70,
        0.639,
        0.064,
        1.70520,
        "public_geometry_extrapolation",
        0.45,
        "v7",
        75,
        0.032,
        14.0,
    ),
    Conductor(
        "M8",
        "m8",
        80,
        0.751,
        0.080,
        1.36416,
        "public_geometry_extrapolation",
        0.50,
        "v8",
        85,
        0.040,
        12.0,
    ),
    Conductor(
        "M9",
        "m9",
        90,
        0.871,
        0.080,
        1.36416,
        "public_geometry_extrapolation",
        0.50,
    ),
)


def _pair(layer: int, datatype: int = 0) -> dict[str, int]:
    result = {"layer": layer}
    if datatype:
        result["datatype"] = datatype
    return result


def _layer(
    name: str,
    purpose: str,
    pair: tuple[int, int],
    *,
    labels: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "purpose": purpose,
        "name": name,
        "description": f"ASAP7 {name}",
        "drw_gds_pair": _pair(*pair),
    }
    if labels:
        # Public ASAP7 cells place pin text on datatype 251 and do not carry a
        # separate pin polygon.  The containing drawing polygon is the pin.
        result["pin_gds_pair"] = _pair(*pair)
        result["label_gds_pair"] = _pair(pair[0], 251)
    return result


def _computed(
    lvs_name: str,
    canonical: str,
    purpose: str,
    pair: tuple[int, int],
    *,
    kind: str = "KIND_REGULAR",
) -> dict[str, Any]:
    return {
        "kind": kind,
        "layer_info": {
            "purpose": purpose,
            "name": lvs_name,
            "description": f"ASAP7 LVS layer {lvs_name}",
            "drw_gds_pair": _pair(*pair),
        },
        "original_layer_name": canonical,
    }


def technology_definition() -> dict[str, Any]:
    """Build a KPEX ``Technology`` protobuf JSON-compatible dictionary."""

    synthetic = {
        "PWell": (100, 100),
        "nSD": (11, 112),
        "pSD": (11, 113),
        "SDT_N": (88, 112),
        "SDT_P": (88, 113),
        "Gate": (7, 100),
        "GateLIG": (16, 101),
        "Local": (16, 100),
        "V0": (18, 100),
    }

    layers = [
        _layer("PWell", "PURPOSE_PWELL", synthetic["PWell"]),
        _layer("NWell", "PURPOSE_NWELL", (1, 0)),
        _layer("nSD", "PURPOSE_N_IMPLANT", synthetic["nSD"]),
        _layer("pSD", "PURPOSE_P_IMPLANT", synthetic["pSD"]),
        _layer("SDT_N", "PURPOSE_CONTACT", synthetic["SDT_N"]),
        _layer("SDT_P", "PURPOSE_CONTACT", synthetic["SDT_P"]),
        _layer("Gate", "PURPOSE_METAL", synthetic["Gate"]),
        _layer("GATE_LIG_LINK", "PURPOSE_VIA", synthetic["GateLIG"]),
        _layer("Local", "PURPOSE_METAL", synthetic["Local"]),
        _layer("V0", "PURPOSE_VIA", synthetic["V0"]),
    ]
    for conductor in STACK_CONDUCTORS[2:]:
        layers.append(
            _layer(
                conductor.name,
                "PURPOSE_METAL",
                (conductor.gds_layer, 0),
                labels=True,
            )
        )
        if conductor.via_above and conductor.via_gds_layer is not None:
            via_number = conductor.via_above.upper()
            layers.append(
                _layer(via_number, "PURPOSE_VIA", (conductor.via_gds_layer, 0))
            )

    computed = [
        _computed("pwell", "PWell", "PURPOSE_PWELL", synthetic["PWell"]),
        _computed("nwell", "NWell", "PURPOSE_NWELL", (1, 0)),
        _computed("nsd", "nSD", "PURPOSE_N_IMPLANT", synthetic["nSD"]),
        _computed("psd", "pSD", "PURPOSE_P_IMPLANT", synthetic["pSD"]),
        _computed("sdt_n", "SDT_N", "PURPOSE_CONTACT", synthetic["SDT_N"]),
        _computed("sdt_p", "SDT_P", "PURPOSE_CONTACT", synthetic["SDT_P"]),
        _computed("gate_con", "Gate", "PURPOSE_METAL", synthetic["Gate"]),
        _computed(
            "gate_lig_link",
            "GATE_LIG_LINK",
            "PURPOSE_VIA",
            synthetic["GateLIG"],
        ),
        _computed("local_con", "Local", "PURPOSE_METAL", synthetic["Local"]),
        _computed("v0_local", "V0", "PURPOSE_VIA", synthetic["V0"]),
    ]
    for conductor in STACK_CONDUCTORS[2:]:
        computed.append(
            _computed(
                conductor.lvs_name,
                conductor.name,
                "PURPOSE_METAL",
                (conductor.gds_layer, 0),
            )
        )
        computed.append(
            _computed(
                f"{conductor.lvs_name}_text",
                conductor.name,
                "PURPOSE_METAL",
                (conductor.gds_layer, 251),
                kind="KIND_LABEL",
            )
        )
        if conductor.via_above and conductor.via_gds_layer is not None:
            computed.append(
                _computed(
                    conductor.via_above,
                    conductor.via_above.upper(),
                    "PURPOSE_VIA",
                    (conductor.via_gds_layer, 0),
                )
            )

    process_layers: list[dict[str, Any]] = [
        {
            "name": "VSUBS",
            "layer_type": "LAYER_TYPE_SUBSTRATE",
            "substrate_layer": {"height": 0.0, "thickness": 0.1, "reference": "fox"},
        },
        {
            "name": "NWell",
            "layer_type": "LAYER_TYPE_NWELL",
            "nwell_layer": {"reference": "fox"},
        },
        {
            "name": "nSD",
            "layer_type": "LAYER_TYPE_DIFFUSION",
            "diffusion_layer": {
                "reference": "fox",
                "contact_above": {
                    "name": "sdt_n",
                    "layer_below": "nsd",
                    "metal_above": "local_con",
                    "thickness": 0.008,
                    "width": 0.024,
                    "spacing": 0.005,
                },
            },
        },
        {
            "name": "pSD",
            "layer_type": "LAYER_TYPE_DIFFUSION",
            "diffusion_layer": {
                "reference": "fox",
                "contact_above": {
                    "name": "sdt_p",
                    "layer_below": "psd",
                    "metal_above": "local_con",
                    "thickness": 0.008,
                    "width": 0.024,
                    "spacing": 0.005,
                },
            },
        },
        {
            "name": "fox",
            "layer_type": "LAYER_TYPE_FIELD_OXIDE",
            "field_oxide_layer": {"dielectric_k": 3.9},
        },
    ]
    for index, conductor in enumerate(STACK_CONDUCTORS):
        metal: dict[str, Any] = {
            "z": conductor.z_um,
            "thickness": conductor.thickness_um,
        }
        if conductor.via_above:
            next_conductor = STACK_CONDUCTORS[index + 1]
            metal["contact_above"] = {
                "name": conductor.via_above,
                "layer_below": conductor.lvs_name,
                "metal_above": next_conductor.lvs_name,
                "thickness": next_conductor.z_um
                - conductor.z_um
                - conductor.thickness_um,
                "width": conductor.via_width_um,
                "spacing": conductor.via_width_um,
            }
        process_layers.append(
            {
                "name": conductor.name,
                "layer_type": "LAYER_TYPE_METAL",
                "metal_layer": metal,
            }
        )
        process_layers.append(
            {
                "name": f"ild_{conductor.name.lower()}",
                "layer_type": "LAYER_TYPE_SIMPLE_DIELECTRIC",
                "simple_dielectric_layer": {
                    "dielectric_k": 3.9 if conductor.name in {"Gate", "Local"} else 2.7,
                    "reference": "fox" if conductor.name == "Gate" else conductor.name,
                },
            }
        )

    resistance_layers = [
        {
            "layer_name": conductor.name,
            # KPEX protobuf unit is milliohm/square.
            "resistance": conductor.sheet_resistance_ohm * 1000,
        }
        for conductor in STACK_CONDUCTORS
    ]
    via_resistances = [
        {
            "via_name": "V0"
            if conductor.via_above == "v0_local"
            else conductor.via_above.upper(),
            "resistance": conductor.via_resistance_ohm * 1000,
        }
        for conductor in STACK_CONDUCTORS
        if conductor.via_above and conductor.via_resistance_ohm is not None
    ]

    return {
        "name": "asap7-open-calibrated",
        "layers": layers,
        "lvs_computed_layers": computed,
        "process_stack": {"layers": process_layers},
        "process_parasitics": {
            "side_halo": 0.25,
            "resistance": {
                "layers": resistance_layers,
                "contacts": [
                    {
                        "contact_name": "sdt_n",
                        "device_layer_name": "nsd",
                        "layer_above": "local_con",
                        "resistance": 1000.0,
                    },
                    {
                        "contact_name": "sdt_p",
                        "device_layer_name": "psd",
                        "layer_above": "local_con",
                        "resistance": 1000.0,
                    },
                ],
                "vias": via_resistances,
            },
        },
    }


def calibration_manifest() -> dict[str, Any]:
    """Return machine-readable provenance and uncertainty for the stack."""

    return {
        "status": "research-grade_not_signoff",
        "reference_corner": "ASAP7 typical_27, nominal 25 C, circuit 27 C",
        "released_reference_cell": "INVxp33_ASAP7_75t_R",
        "conductors": [asdict(conductor) for conductor in STACK_CONDUCTORS],
        "dielectric": {
            "field_oxide_k": 3.9,
            "local_ild_k": 3.9,
            "beol_ild_k": 2.7,
            "provenance": "assumption_calibrated_to_released_xact",
            "relative_uncertainty": 0.35,
        },
        "limitations": [
            "No public ASAP7 Calibre LVS/xACT rule deck is redistributed.",
            "Local combines GATE, LIG, and LISD into one distributed-R sheet.",
            "Diffusion resistance is represented at the terminal/contact boundary.",
            "M3-M9 resistance and dielectric gaps are extrapolations.",
            "FasterCap models Manhattan 3-D conductors; FinFET device-internal capacitance remains in BSIM-CMG.",
        ],
    }


def write_kpex_technology(path: str | Path) -> Path:
    """Materialize the generated KPEX protobuf JSON and validate when possible."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(technology_definition(), indent=2) + "\n")
    try:
        from klayout_pex.tech_info import TechInfo

        TechInfo.parse_tech_def(str(output))
    except ImportError:
        pass
    return output
