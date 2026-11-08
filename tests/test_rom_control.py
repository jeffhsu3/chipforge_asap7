"""The NOR ROM's read controller: its handshake, DRC and LVS.

The handshake runs on the controller's own transistor netlist at switch
level.  DRC and LVS skip when gdscheck or KLayout is missing;
``ASAP7_REQUIRE_TOOLS=1`` makes that a failure.
"""

from __future__ import annotations

from pathlib import Path

import gdspy
import pytest
from test_rom_decoder import _settle

from chipforge_asap7.devices import RowSupportSpec, build_row_support
from chipforge_asap7.devices.rom_control import (
    RomControlSpec,
    assign_levels,
    build_rom_control_row,
)
from chipforge_asap7.verification.lvs import run_lvs

NO_TAP_IN_CELL = {"ACTIVE.LUP.1"}

SPECS = [
    RomControlSpec(),
    RomControlSpec(fingers=1),
    RomControlSpec(fingers=8, vt="lvt"),
]


def _library():
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    return library


class _Controller:
    """The controller's netlist at switch level, stepped through input changes."""

    NET = {"VDD": "VDD", "VSS": "VSS"}

    def __init__(self, spec: RomControlSpec) -> None:
        self.devices = [
            {"terminals": (d, s), "gate": g, "pmos": flavor == "p"}
            for _, d, g, s, flavor, _ in spec.devices
        ]
        self.value: dict = {}

    def step(self, **inputs: int) -> dict:
        driven = {"VDD": 1, "VSS": 0, **inputs}
        self.value = _settle(self.devices, self.NET, driven, self.value)
        return self.value

    def outputs(self) -> tuple:
        return tuple(self.value.get(n) for n in ("PRECHN", "EVAL", "SAMPLE"))


def test_a_read_cycle_runs_the_phases_in_order():
    c = _Controller(RomControlSpec())
    c.step(CLK=1, RESETN=1, READY=0)
    assert c.outputs() == (0, 0, 0)  # precharging
    c.step(CLK=0, RESETN=1, READY=0)
    assert c.outputs() == (1, 1, 0)  # precharge off, wordline on
    c.step(CLK=0, RESETN=1, READY=1)
    assert c.outputs() == (1, 1, 1)  # the replica has read: sample
    c.step(CLK=1, RESETN=1, READY=1)
    assert c.outputs() == (0, 0, 0)
    c.step(CLK=1, RESETN=1, READY=0)
    assert c.outputs() == (0, 0, 0)


def test_each_phase_waits_for_the_one_before():
    """Held at its old value, each driver output holds the next phase back."""
    spec = RomControlSpec()
    c = _Controller(spec)
    c.step(CLK=1, RESETN=1, READY=0)
    # PRECHN still low (its driver not through): EVAL must not rise.
    c.step(CLK=0, RESETN=1, READY=1, PRECHN=0)
    assert c.value["EVAL"] == 0 and c.value["SAMPLE"] == 0
    c.step(CLK=0, RESETN=1, READY=1)
    assert c.outputs() == (1, 1, 1)
    # CLK rises with SAMPLE still high: EVAL and PRECHN wait for it.
    c.step(CLK=1, RESETN=1, READY=1, SAMPLE=1)
    assert c.value["EVAL"] == 1 and c.value["PRECHN"] == 1
    # SAMPLE down, EVAL still high: PRECHN waits for EVAL.
    c.step(CLK=1, RESETN=1, READY=1, EVAL=1)
    assert c.value["SAMPLE"] == 0 and c.value["PRECHN"] == 1
    c.step(CLK=1, RESETN=1, READY=1)
    assert c.outputs() == (0, 0, 0)


def test_resetn_drains_a_read_like_clk_high():
    c = _Controller(RomControlSpec())
    c.step(CLK=1, RESETN=1, READY=0)
    c.step(CLK=0, RESETN=1, READY=1)
    assert c.outputs() == (1, 1, 1)
    c.step(CLK=0, RESETN=0, READY=1)
    assert c.outputs() == (0, 0, 0)
    c.step(CLK=0, RESETN=0, READY=0)
    assert c.outputs() == (0, 0, 0)  # no read while reset


def test_levels_keep_tracks_on_one_level_apart():
    spans = {"a": (0, 100), "b": (120, 200), "c": (150, 300), "d": (90, 110)}
    # b would be 20 nm past a's end and 10 past d's: it takes a third level,
    # and c, 50 nm past a, joins a.
    assert assign_levels(spans, 3) == {"a": 0, "d": 1, "b": 2, "c": 0}
    assert assign_levels(spans, 3, fixed={"c": 1}) == {"c": 1, "a": 0, "d": 1, "b": 2}
    with pytest.raises(ValueError, match="no M2 level"):
        assign_levels(spans, 1)


def test_isolated_rows_only_trip_the_missing_body_tap(asap7_drc):
    library = _library()
    top = library.new_cell("rom_control_isolated")
    cursor = 0
    for spec in SPECS:
        top.add(
            gdspy.CellReference(build_rom_control_row(spec, lib=library), (cursor, 0))
        )
        cursor += spec.width + 1080
    assert set(asap7_drc(library, top, tag="isolated")) == NO_TAP_IN_CELL


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.cell_name)
def test_terminated_row_is_clean(spec, asap7_drc):
    library = _library()
    row = build_rom_control_row(spec, lib=library)
    support = {
        kind: build_row_support(
            RowSupportSpec(stack=spec.stack, kind=kind), lib=library
        )
        for kind in ("filler", "tap")
    }
    top = library.new_cell("rom_control_terminated")
    cursor = 0
    for item in ("filler", "row", "filler", "tap", "filler"):
        cell = row if item == "row" else support[item]
        top.add(gdspy.CellReference(cell, origin=(cursor, 0)))
        cursor += spec.width if item == "row" else cell.get_bounding_box()[1][0]
    assert asap7_drc(library, top, tag="terminated") == []


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.cell_name)
def test_row_matches_its_unit_fin_reference(spec, tmp_path: Path, require_klayout):
    library = _library()
    build_rom_control_row(spec, lib=library)
    gds = tmp_path / "row.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(spec.lvs_reference())
    result = run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    )
    assert result.matched
    fins = sum(device[5] for device in spec.devices)
    assert result.extracted_netlist.read_text().count("M$") == fins


def test_a_crossed_input_fails(tmp_path: Path, require_klayout):
    """A reference whose SAMPLE NAND takes PRECHN for EVAL must not match."""
    spec = RomControlSpec()
    library = _library()
    build_rom_control_row(spec, lib=library)
    gds = tmp_path / "row.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    text = spec.lvs_reference()
    wrong = text.replace("MP_SN_1_f0 VDD EVAL SN", "MP_SN_1_f0 VDD PRECHN SN")
    assert wrong != text
    reference.write_text(wrong)
    assert not run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=spec.cell_name, tie_bodies=True
    ).matched
