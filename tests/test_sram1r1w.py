"""The 1R1W column: OpenFinRAM's 8T cell, its array, the role-cut IO blocks.

Needs gdstk and OpenFinRAM's generated cell libraries (``OPENFINRAM_ROOT``, or
OpenFinRAM beside this repository); skips without them.  DRC, LVS and the
transient bench also need gdscheck, KLayout and ngspice.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

gdstk = pytest.importorskip("gdstk")

from chipforge_asap7.devices.io_column import (  # noqa: E402
    IoColumnSpec,
    block_netlist,
    build_io_column,
    io_column_pins,
)
from chipforge_asap7.sram1r1w.array import (  # noqa: E402
    MUX_ROWS,
    build_array,
    default_tap_pitch,
    slot_layout,
)
from chipforge_asap7.sram1r1w.cells import (  # noqa: E402
    LIBRARIES,
    default_gds_dir,
    load_cell_library,
)
from chipforge_asap7.sram1r1w.column import (  # noqa: E402
    ColumnSpec,
    build_column,
    column_netlist,
    column_pins,
    io_specs,
)
from chipforge_asap7.sram1r1w.column_bench import (  # noqa: E402
    DEFAULT_SEQUENCE,
    BenchTiming,
    _schedule,
    spice_netlist,
)

pytestmark = pytest.mark.skipif(
    not all((default_gds_dir() / f"{name}.gds").is_file() for name in LIBRARIES),
    reason="OpenFinRAM's 8T cell libraries are not built (set OPENFINRAM_ROOT)",
)


@pytest.fixture(scope="module")
def cells():
    return load_cell_library()


# ── Cells and array ──────────────────────────────────────────────────────────
def test_the_cells_are_the_geometry_openfinram_recorded(cells):
    assert cells.width == pytest.approx(0.108)
    assert cells.height == pytest.approx(0.594)
    recorded = json.loads((cells.source / "digests.json").read_text())
    assert all(cells.digests[name] == recorded[name] for name in LIBRARIES)


def test_a_library_with_other_geometry_is_refused(tmp_path: Path):
    source = default_gds_dir()
    for name in LIBRARIES:
        shutil.copy(source / f"{name}.gds", tmp_path)
    recorded = json.loads((source / "digests.json").read_text())
    recorded[LIBRARIES[0]] = "0" * 64
    (tmp_path / "digests.json").write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match="not the one OpenFinRAM recorded"):
        load_cell_library(tmp_path)


@pytest.mark.parametrize(
    ("wordlines", "pitch", "slots", "taps"),
    [(4, 4, 6, [4]), (32, 16, 36, [16, 34]), (6, 2, 12, [2, 6, 10]), (8, 0, 8, [])],
)
def test_taps_follow_every_pitch_cells(wordlines, pitch, slots, taps):
    total, cell_slots, tap_slots = slot_layout(wordlines, pitch)
    assert (total, tap_slots) == (slots, taps)
    assert len(cell_slots) == wordlines
    assert not set(cell_slots) & {t + d for t in tap_slots for d in (0, 1)}


def test_the_default_tap_pitch_divides_the_row():
    assert [default_tap_pitch(w) for w in (4, 24, 32, 64, 100)] == [4, 8, 16, 16, 4]
    with pytest.raises(ValueError, match="does not divide"):
        slot_layout(12, 8)


def test_the_array_is_four_rows_of_one_bitline_set_each(cells):
    lib = gdstk.Library(unit=1e-6, precision=2.5e-10)
    for cell in cells.library.cells:
        lib.add(cell)
    library = replace(cells, library=lib)
    array = build_array(lib, library, 32)
    texts = {label.text for label in array.labels}
    assert {f"WLA[{i}]" for i in range(32)} <= texts
    assert {f"BLB[{r}]" for r in range(MUX_ROWS)} <= texts
    boundary = [p for p in array.polygons if p.layer == 100][0].bounding_box()
    assert boundary[1][0] == pytest.approx(36 * 0.108)
    assert boundary[1][1] == pytest.approx(MUX_ROWS * 0.594)


# ── IO blocks ────────────────────────────────────────────────────────────────
def test_the_write_block_keeps_only_the_driver_and_the_read_block_the_sensing(cells):
    specs = io_specs(cells)
    assert specs["write"].cells == ("write_driver",)
    assert specs["read"].cells == ("sense_amp", "output_latch")
    write_pins, read_pins = (set(io_column_pins(specs[r])) for r in ("write", "read"))
    assert {"D", "WRENA", "WRENAN"} <= write_pins
    assert not {"SAE", "OE", "OEB", "Q"} & write_pins
    assert {"SAE", "OE", "OEB", "Q"} <= read_pins
    assert not {"D", "WRENA", "WRENAN"} & read_pins
    assert "Msa_" not in block_netlist(specs["write"])
    assert "Mwd_" not in block_netlist(specs["read"])


def test_a_role_cut_block_is_narrower():
    import gdspy

    widths = {}
    for role in ("read_write", "read", "write"):
        library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
        gdspy.current_library = library
        cell = build_io_column(IoColumnSpec(role=role), lib=library)
        (x0, _), (x1, _) = cell.get_bounding_box()
        widths[role] = x1 - x0
    assert widths["write"] < widths["read"] <= widths["read_write"]


def test_a_two_sided_block_must_read_and_write():
    from chipforge_asap7.devices import BitlineMuxSpec

    with pytest.raises(ValueError):
        IoColumnSpec(
            mux=BitlineMuxSpec(rows=2, bitline_layer="M4"), two_sided=True, role="read"
        )


# ── Column ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def column32(cells):
    spec = ColumnSpec(32)
    return spec, build_column(cells, spec), column_netlist(cells, spec)


def test_the_column_exposes_each_port_and_nothing_internal(column32):
    spec, column, _ = column32
    texts = {label.text for label in column.labels}
    assert texts == set(column_pins(spec))


def test_the_column_netlist_has_every_cell_and_both_blocks(column32):
    spec, _, cdl = column32
    assert sum(line.startswith("Xcell_") for line in cdl.splitlines()) == spec.words
    header = next(
        line for line in cdl.splitlines() if line.startswith(f".SUBCKT {spec.name}")
    )
    assert header.split()[2:] == column_pins(spec)
    assert "Xwrite " in cdl and "Xread " in cdl


def test_the_bench_netlist_is_osdi_instances_without_brackets(column32):
    _, _, cdl = column32
    netlist, models = spice_netlist(cdl)
    assert models == {"nmos_sram", "pmos_sram", "nmos_rvt", "pmos_rvt"}
    assert "[" not in netlist
    assert not [line for line in netlist.splitlines() if line.startswith("M")]
    assert "Nwd_NI wd_dn d vss vss nmos_rvt L=20n NFIN=3 NF=1" in netlist


def test_the_schedule_overlaps_nothing_it_should_not(column32):
    spec, _, _ = column32
    steps = _schedule(spec, DEFAULT_SEQUENCE, BenchTiming())
    # Each select and its complement always disagree, and SAE only rises once
    # the read precharge is off.
    for k in range(MUX_ROWS):
        for port in ("w", "r"):
            assert [t for t, _ in steps[f"{port}ysel[{k}]"]] == [
                t for t, _ in steps[f"{port}yseln[{k}]"]
            ]
    assert steps["wprechn"][0] == (0.0, 0)


@pytest.mark.slow
def test_the_column_is_drc_clean_but_for_known_findings(
    column32, cells, tmp_path, external_tool
):
    """Only the IO cells' tip-to-side findings (redraw pending), M4 grid, and
    the taps' gate ends on the open top and bottom, which a macro's dummy end
    rows close."""
    from chipforge_asap7.verification.drc import run_drc

    from .conftest import find_gdscheck_or_none

    external_tool(find_gdscheck_or_none() is not None, "gdscheck is not installed")
    spec, column, _ = column32
    gds = tmp_path / "column.gds"
    cells.library.write_gds(str(gds))
    findings = run_drc(gds, tmp_path / "drc", cell_name=column.name)
    assert {f.category for f in findings} <= {
        "M1.S.2", "M2.S.2", "M4.AUX.1", "M4.AUX.2", "GATE.S.3",
    }  # fmt: skip
    edge = {round(f.bbox[1], 1) for f in findings if f.category == "GATE.S.3"}
    assert edge <= {-13.8, 2389.8}


@pytest.mark.slow
def test_the_column_reads_and_writes_without_disturbing_a_cell(
    tmp_path: Path, external_tool
):
    from chipforge_asap7.devices.rom_timing import CORNERS, default_osdi
    from chipforge_asap7.sram1r1w.column_bench import run_bench

    external_tool(shutil.which("ngspice") is not None, "ngspice is not installed")
    try:
        osdi = default_osdi(tmp_path)
    except FileNotFoundError as error:
        pytest.skip(str(error))
    result = run_bench(ColumnSpec(8), CORNERS["typical"], osdi, tmp_path / "bench")
    assert result.passed, result
    assert result.hold_margin_mv > 300


@pytest.mark.parametrize("role", ["write", "read"])
def test_each_role_block_matches_its_flat_unit_fin_reference(
    role, cells, tmp_path: Path, require_klayout
):
    import gdspy

    from chipforge_asap7.verification.lvs import render_io_column_lvs_schematic, run_lvs

    spec = io_specs(cells)[role]
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    cell = build_io_column(spec, lib=library)
    gds = tmp_path / "block.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(render_io_column_lvs_schematic(spec, cell_name=cell.name))
    result = run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=cell.name, tie_bodies=True, flat=True,
        timeout=1800,
    )  # fmt: skip
    assert result.matched
    fins = spec.mux.selects * sum(f for *_, f in spec.mux.devices) + sum(
        sum(f for *_, f in getattr(spec, part).devices) for part in spec.cells
    )
    assert result.extracted_netlist.read_text().count("M$") == fins


def test_a_column_matches_its_netlist(tmp_path: Path, require_klayout):
    """Four wordlines: KLayout's compare slows steeply with more identical cells."""
    from chipforge_asap7.verification import run_hierarchical_lvs

    cells = load_cell_library()
    spec = ColumnSpec(4)
    column = build_column(cells, spec)
    gds = tmp_path / "column.gds"
    cells.library.write_gds(str(gds))
    cdl = tmp_path / "column.cdl"
    cdl.write_text(column_netlist(cells, spec))
    flatten = ("sram_cell_*", "tapcell_*", "FILLER_*", "dummy_*", "*__row", "sramcol_*",
               "array_*", "col_cap_*", "blmux_*", "sarow_*", "wrdrv_*", "outlatch_*",
               "filler_fin_*", "tap_fin_*")  # fmt: skip
    result = run_hierarchical_lvs(
        gds, cdl, tmp_path / "lvs", cell_name=column.name, flatten_circuits=flatten
    )
    assert result.matched, result.failing
