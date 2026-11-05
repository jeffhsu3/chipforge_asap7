"""ASAP7 blocked-ROM compiler with a compact periodic selector array."""

import os

# The BLAS reads these when NumPy first loads, which the imports below trigger.
# Compiles run side by side (one per word slice), and a thread pool in each made
# the placement solve hundreds of times slower. Set them to override.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from .compact import compact_periphery
from .compiler import (
    BromSpec,
    cgen_circuit,
    decoder_circuit,
    macro_circuit,
    output_circuit,
    selector_map,
)
from .controller import clocked_macro_circuit, controller_circuit
from .distribution import DistributionSpec, size_control_distribution
from .layout import build_macro
from .output_register import output_register_circuit
from .periphery import build_periphery
from .replica import ReplicaSpec, self_timed_macro_circuit
from .selector import SelectorPlan, build_selector_array, default_selector_plan
from .wire_rc import WireRCSpec, routed_control_rc, size_wire_rc_distribution

__all__ = [
    "BromSpec",
    "DistributionSpec",
    "ReplicaSpec",
    "SelectorPlan",
    "WireRCSpec",
    "build_macro",
    "build_periphery",
    "build_selector_array",
    "cgen_circuit",
    "clocked_macro_circuit",
    "compact_periphery",
    "controller_circuit",
    "decoder_circuit",
    "default_selector_plan",
    "macro_circuit",
    "output_circuit",
    "output_register_circuit",
    "routed_control_rc",
    "selector_map",
    "self_timed_macro_circuit",
    "size_control_distribution",
    "size_wire_rc_distribution",
]
