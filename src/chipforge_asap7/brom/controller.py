"""Clock-request controller with explicit physical completion acknowledgments.

CLK high requests precharge; CLK low requests a read. The four acknowledgment
inputs may remain external or connect to the experimental selector replica.
Physical completion margins require qualified control distribution.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .circuit import Circuit
from .compiler import BromSpec, macro_circuit
from .output_register import add_registered_output

CONTROLLER_INPUTS = (
    "CLK",
    "RESETN",
    "PRECHN_FAR",
    "READ_READY",
    "SAMPLE_CLOSED",
    "CANDIDATE_BUSY",
)
CONTROLLER_OUTPUTS = ("PRECHN", "EVAL", "SAMPLE")


def add_controller(c: Circuit) -> None:
    """Drive the existing macro phases, using active-high acknowledgments.

    On return to precharge, SAMPLE falls first. EVAL stays high until the
    output latch network acknowledges closure. PRECHN stays high until both
    EVAL and the candidate replica have fallen. RESETN cancels a request via
    this same drain sequence; it does not asynchronously clear the data latches.
    """

    def gate(kind, *nets):
        start = len(c.devices)
        getattr(c, kind)(*nets)
        c.group_since(f"controller_{start}", start)

    clkb, request, sample_open = c.node(), c.node(), c.node()
    evaluate_request, ready_eval = c.node(), c.node()
    pending, busy = c.node(), c.node()
    gate("inv", "CLK", clkb)
    gate("and2", "RESETN", clkb, request)
    gate("inv", "SAMPLE_CLOSED", sample_open)
    # Also hold evaluation while SAMPLE itself is still high: acknowledgment
    # must not be able to cut evaluation before the local sample output falls.
    sampling = c.node()
    gate("or2", "SAMPLE", sample_open, sampling)
    gate("or2", request, sampling, evaluate_request)
    gate("and2", evaluate_request, "PRECHN_FAR", "EVAL")
    gate("and2", "EVAL", "READ_READY", ready_eval)
    gate("and2", request, ready_eval, "SAMPLE")
    gate("or2", request, "EVAL", pending)
    gate("or2", sampling, "CANDIDATE_BUSY", busy)
    gate("or2", pending, busy, "PRECHN")


def controller_circuit(fins: int = 2, vt: str = "rvt") -> Circuit:
    c = Circuit(
        f"brom_controller_f{fins}_{vt}",
        (*CONTROLLER_INPUTS, *CONTROLLER_OUTPUTS, "VDD", "VSS"),
        fins,
        vt,
    )
    add_controller(c)
    return c


def clocked_macro_circuit(spec: BromSpec, words: Sequence[int]) -> Circuit:
    """Compile a controller-integrated macro with external completion inputs.

    The feedback pins are real ports, not tied-off stand-ins for a replica.
    The original externally sequenced macro remains available separately.
    """
    c = macro_circuit(replace(spec, registered_output=False), words)
    c.name = c.name.replace("brom_compact_", "brom_clocked_")
    c.pins = (
        *(f"A[{i}]" for i in range(spec.address_bits)),
        *CONTROLLER_INPUTS,
        *(f"Q[{i}]" for i in range(spec.bits)),
        "VDD",
        "VSS",
    )
    add_controller(c)
    if spec.registered_output:
        add_registered_output(c, spec.bits)
    return c
