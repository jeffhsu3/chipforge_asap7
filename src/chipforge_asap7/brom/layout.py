"""Dense central CGEN, split selector halves and bottom decoder assembly."""

from .dense_macro import _strip, dense_macro_layout, verify_dense_routes


def macro_plan(circuit):
    """Return the actual physical plan shared with GDS, LEF and wire RC."""
    return dense_macro_layout(circuit)[1]


def macro_periphery(circuit):
    """Bottom logic only; the candidate drivers belong to the central spine."""
    if circuit.array is None:
        raise ValueError("requires a macro_circuit")
    excluded = set(circuit.array.device_names)
    if circuit.replica:
        excluded.update(circuit.replica.selector.device_names)
    excluded.update(
        n
        for role, ns in circuit.placement_groups
        if role.startswith(("cgen_", "left_cgen_"))
        for n in ns
    )
    return _strip(circuit, excluded)


def build_macro(circuit, *, lib=None):
    top, plan = dense_macro_layout(circuit)
    verify_dense_routes(top, plan["checkpoints"])
    if lib is not None:
        if lib.unit != 1e-9:
            raise ValueError("BROM layout requires lib.unit=1e-9")
        lib.add(top)
    return top


def verify_macro_routes(circuit, top):
    verify_dense_routes(top, macro_plan(circuit)["checkpoints"])
