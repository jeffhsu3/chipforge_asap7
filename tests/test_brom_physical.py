"""DRC and LVS of every BROM cell the compiler emits, on one shared matrix.

The macro options compose, so a few macros cover them all: each fin count,
both way counts, rvt and lvt, all three interfaces (external phases, clocked,
self-timed), the keeper, the registered output and both sizings. Each case is
named for what it combines, so a failure still points at the options in play.
"""

import gdspy
import pytest

from chipforge_asap7.brom import (
    BromSpec,
    build_macro,
    build_periphery,
    cgen_circuit,
    clocked_macro_circuit,
    controller_circuit,
    decoder_circuit,
    macro_circuit,
    output_circuit,
    self_timed_macro_circuit,
    size_control_distribution,
    size_wire_rc_distribution,
)
from chipforge_asap7.brom.output_register import output_register_circuit


def macro(make, spec, sizing=None):
    def build():
        c = make(spec, [(i * 3 + 1) % (1 << spec.bits) for i in range(spec.depth)])
        return sizing(c) if sizing else c

    return build


CELLS = {
    # Standalone cells the CLI writes beside the macro.
    "output": output_circuit,
    "cgen": cgen_circuit,
    "decoder-12x1": lambda: decoder_circuit(BromSpec(12, 1)),
    "controller-f1": lambda: controller_circuit(fins=1),
    "output-register": lambda: output_register_circuit(bits=2),
    # Macros.
    "phases-4x2": macro(macro_circuit, BromSpec(4, 2)),
    "phases-12x1-f3": macro(macro_circuit, BromSpec(12, 1, fins=3)),
    "phases-16x3-m4-f1-lvt": macro(
        macro_circuit, BromSpec(16, 3, ways=4, fins=1, vt="lvt")
    ),
    "clocked-16x1-m4-f3-registered-sized": macro(
        clocked_macro_circuit,
        BromSpec(16, 1, ways=4, fins=3, registered_output=True),
        size_control_distribution,
    ),
    "selftimed-8x1-f1-keeper4-sized": macro(
        self_timed_macro_circuit,
        BromSpec(8, 1, fins=1, keeper_fins=1, keeper_stack=4),
        size_control_distribution,
    ),
    "selftimed-16x1-m4-f3-keeper-registered-wirerc": macro(
        self_timed_macro_circuit,
        BromSpec(
            16, 1, ways=4, fins=3, keeper_fins=2, keeper_stack=1, registered_output=True
        ),
        size_wire_rc_distribution,
    ),
}


@pytest.mark.parametrize("name", CELLS)
def test_drc_and_lvs(name, tmp_path, asap7_drc, require_klayout):
    from chipforge_asap7.verification.lvs import run_lvs
    from chipforge_asap7.verification.lvs_report import summarize_lvsdb

    c = CELLS[name]()
    if "sized" in name or "wirerc" in name:
        assert c.distribution["nets"], "sizing inserted no buffers"
    lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    top = (build_macro if c.array is not None else build_periphery)(c, lib=lib)
    if c.replica is not None:
        assert len(top.references) == 3  # data array, replica array, periphery
    assert asap7_drc(lib, top, tag=top.name) == []
    gds, ref = tmp_path / "dut.gds", tmp_path / "dut.spice"
    lib.write_gds(str(gds))
    ref.write_text(c.lvs_reference())
    result = run_lvs(gds, ref, tmp_path / "lvs", cell_name=top.name)
    assert result.matched, result.log.read_text()
    assert not any(
        r["renamed_pins"] for r in summarize_lvsdb(result.report)["circuits"]
    )
    pins = {
        line.removeprefix("* pin ").strip()
        for line in result.extracted_netlist.read_text().splitlines()
        if line.startswith("* pin ")
    }
    assert pins == set(c.pins)
