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

    from conftest import find_gdscheck_or_none

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


# ── Periphery rows ───────────────────────────────────────────────────────────
def _row_name(row) -> str:
    return getattr(row, "cell_name", None) or row.name


def _macro_rows():
    """Every distinct row of the default macro, and the wide select row of a
    32-bit one (12-finger buffers)."""
    from chipforge_asap7.sram1r1w.periphery import MacroSpec

    spec = MacroSpec()
    rows = [
        *spec.controllers.values(),
        *spec.select_rows.values(),
        *spec.rank_rows.values(),
        *spec.latch_rows.values(),
        MacroSpec(bits=32).select_rows["W"],
    ]
    return list({_row_name(row): row for row in rows}.values())


def _build_row(row, library):
    from chipforge_asap7.devices.gate_row import GateRowSpec, build_gate_row
    from chipforge_asap7.devices.rom_latch import build_rom_address_latch_row

    if isinstance(row, GateRowSpec):
        return build_gate_row(row, lib=library)
    return build_rom_address_latch_row(row, lib=library)


@pytest.mark.parametrize("row", _macro_rows(), ids=_row_name)
def test_each_periphery_row_is_drc_clean_terminated(row, asap7_drc):
    import gdspy

    from chipforge_asap7.devices.row_support import RowSupportSpec, build_row_support

    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    cell = _build_row(row, library)
    support = {
        kind: build_row_support(RowSupportSpec(stack=row.stack, kind=kind), lib=library)
        for kind in ("filler", "tap")
    }
    top = library.new_cell("terminated")
    cursor = 0
    for item in ("filler", "row", "filler", "tap", "filler"):
        part = cell if item == "row" else support[item]
        top.add(gdspy.CellReference(part, origin=(cursor, 0)))
        cursor += row.width if item == "row" else part.get_bounding_box()[1][0]
    assert asap7_drc(library, top, tag="terminated") == []


@pytest.mark.parametrize("row", _macro_rows(), ids=_row_name)
def test_each_periphery_row_matches_its_reference(row, tmp_path: Path, require_klayout):
    import gdspy

    from chipforge_asap7.verification.lvs import run_lvs

    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    _build_row(row, library)
    gds = tmp_path / "row.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.spice"
    reference.write_text(row.lvs_reference())
    result = run_lvs(
        gds, reference, tmp_path / "lvs", cell_name=_row_name(row), tie_bodies=True
    )
    assert result.matched
    assert result.extracted_netlist.read_text().count("M$") == sum(
        d[5] for d in row.devices
    )


# ── Wordline driver strips ───────────────────────────────────────────────────
def _stack(wordlines: int = 32):
    from chipforge_asap7.layout.transform import edge_boundary
    from chipforge_asap7.sram1r1w.periphery import MacroSpec
    from chipforge_asap7.sram1r1w.strips import build_strip_pair

    cells = load_cell_library()
    column = build_column(cells, ColumnSpec(wordlines))
    spec = MacroSpec(wordlines).slice
    pair = build_strip_pair(cells, column, spec)
    top = cells.library.new_cell("stack")
    top.add(
        gdstk.Reference(column), gdstk.Reference(pair, (0, -edge_boundary(pair)[3]))
    )
    return cells, column, pair, spec, top


def test_the_strips_drive_every_wordline_from_below():
    _, column, pair, _, _ = _stack()
    ours = {lb.text: lb for lb in pair.labels if lb.text.startswith(("WWL", "RWL"))}
    theirs = {lb.text: lb for lb in column.labels if lb.text.startswith(("WWL", "RWL"))}
    assert set(ours) == set(theirs)
    for name, label in ours.items():
        assert label.layer == theirs[name].layer
        assert float(label.origin[0]) == pytest.approx(float(theirs[name].origin[0]))


def test_the_strip_pair_takes_its_decode_from_its_bottom_and_right_edges():
    """Both ports' SELs come up from the pair's bottom, where the PS rank
    stands, port B's on M3 and port A's on an M5 track; each strip's B lines
    are M4 rails on tracks, out at the pair's right end."""
    from chipforge_asap7.layout.transform import edge_boundary

    _, _, pair, _, _ = _stack()
    x0, _, x1, top = edge_boundary(pair)
    pins = {lb.text: lb for lb in pair.labels}
    for k in range(8):
        a, b = pins[f"SEL_A[{k}]"], pins[f"SEL_B[{k}]"]
        assert (a.layer, b.layer) == (50, 30)
        assert float(a.origin[1]) < 0.01 and float(b.origin[1]) < 0.01
        assert round((float(a.origin[0]) - 0.012) * 1000) % 48 == 0
    for port in "AB":
        for j in range(4):
            rail = pins[f"B_{port}[{j}]"]
            assert rail.layer == 40
            assert float(rail.origin[0]) == pytest.approx(x1 - 0.010)
            assert round((float(rail.origin[1]) - top - 0.012) * 1000) % 48 == 0


def test_the_strip_pair_matches_its_netlist(tmp_path: Path, require_klayout):
    from chipforge_asap7.sram1r1w.strips import strip_pair_netlist
    from chipforge_asap7.verification import run_hierarchical_lvs

    cells, _, pair, spec, _ = _stack()
    gds, cdl = tmp_path / "strips.gds", tmp_path / "strips.cdl"
    cells.library.write_gds(str(gds))
    cdl.write_text(strip_pair_netlist(spec, 32, pair.name))
    flatten = ("wl_slice_*", "wl_strip_*", "wl_via*", "nand_fin*", "inv_fin*", "*__nand",
               "*__inv", "*__drv", "tap_fin*", "filler_fin*")  # fmt: skip
    result = run_hierarchical_lvs(gds, cdl, tmp_path / "lvs", cell_name=pair.name,
                                  flatten_circuits=flatten)  # fmt: skip
    assert result.matched, result.failing


@pytest.mark.slow
def test_the_strips_add_no_drc_finding_under_a_column(tmp_path: Path, external_tool):
    """Abutted, the pair closes the taps' gate ends on the column's bottom and
    adds nothing: what is left is the column's own (its IO cells', and the
    open top edge's gate ends)."""
    from chipforge_asap7.verification.drc import run_drc

    from conftest import find_gdscheck_or_none

    external_tool(find_gdscheck_or_none() is not None, "gdscheck is not installed")
    cells, *_, top = _stack()
    gds = tmp_path / "stack.gds"
    cells.library.write_gds(str(gds))
    findings = run_drc(gds, tmp_path / "drc", cell_name=top.name)
    assert {f.category for f in findings} <= {
        "M1.S.2", "M2.S.2", "M4.AUX.1", "M4.AUX.2", "GATE.S.3",
    }  # fmt: skip
    assert {round(f.bbox[1], 1) for f in findings if f.category == "GATE.S.3"} == {
        2389.8
    }
    assert all(f.bbox[1] >= 0 for f in findings if f.category.startswith("M4"))


# ── Wordline core: both halves' strip pairs and the PS rank ─────────────────
CORE_FLATTEN = ("wl_slice_*", "wl_strip_*", "wl_strips_*", "wl_via*", "nand_fin*",
                "inv_fin*", "*__nand", "*__inv", "*__drv", "tap_fin*", "filler_fin*",
                "sram1r1w_rank_*")  # fmt: skip


def test_the_rank_lines_each_read_ps_under_its_slices_sel():
    from chipforge_asap7.sram1r1w.band import build_wordline_core
    from chipforge_asap7.sram1r1w.periphery import MacroSpec

    core = build_wordline_core(MacroSpec(32, 2), load_cell_library())
    read = core.band.rows["R"]
    sel = sorted(
        float(lb.origin[0]) for lb in core.pair.labels if lb.text.startswith("SEL_B")
    )
    x_left = float(core.pair.bounding_box()[0][0])
    outs = [
        x_left + read.out_x(g) / 1000 for g in read.gates if g.out.startswith("RPS[")
    ]
    assert outs == pytest.approx(sel)


#: A VDD-edged slice and, from 32 bits, a VSS-edged one: rank rows the other
#: way up, 324 nm bands, and their n-wells tapped only by the rows' own taps.
CORE_SIZES = [(32, 2), pytest.param(32, 32, marks=pytest.mark.slow)]


@pytest.mark.parametrize(("wordlines", "bits"), CORE_SIZES)
def test_the_wordline_core_matches_its_netlist(
    tmp_path: Path, require_klayout, wordlines, bits
):
    """Each PS line reaches its slice in both halves: the read's on M3, the
    write's on M5, through the band."""
    from chipforge_asap7.sram1r1w.band import build_wordline_core
    from chipforge_asap7.sram1r1w.periphery import MacroSpec
    from chipforge_asap7.verification import run_hierarchical_lvs

    cells = load_cell_library()
    core = build_wordline_core(MacroSpec(wordlines, bits), cells)
    gds, cdl = tmp_path / "core.gds", tmp_path / "core.cdl"
    cells.library.write_gds(str(gds))
    cdl.write_text(core.netlist)
    result = run_hierarchical_lvs(gds, cdl, tmp_path / "lvs", cell_name=core.cell.name,
                                  flatten_circuits=CORE_FLATTEN)  # fmt: skip
    assert result.matched, result.failing


@pytest.mark.slow
@pytest.mark.parametrize(("wordlines", "bits"), [(32, 2), (32, 32)])
def test_the_wordline_core_adds_no_drc_finding_between_two_columns(
    tmp_path: Path, external_tool, wordlines, bits
):
    """A column on the upper pair and one, mirrored, under the lower: what is
    left is the columns' own (their IO cells', and their open edges' gate
    ends), none in the core."""
    from chipforge_asap7.layout.transform import edge_boundary
    from chipforge_asap7.sram1r1w.band import build_wordline_core
    from chipforge_asap7.sram1r1w.periphery import MacroSpec
    from chipforge_asap7.verification.drc import run_drc

    from conftest import find_gdscheck_or_none

    external_tool(find_gdscheck_or_none() is not None, "gdscheck is not installed")
    cells = load_cell_library()
    core = build_wordline_core(MacroSpec(wordlines, bits), cells)
    column = core.column
    _, bottom, _, top = edge_boundary(core.cell)
    tile = cells.library.new_cell("tile")
    tile.add(
        gdstk.Reference(core.cell),
        gdstk.Reference(column, (0, top)),
        gdstk.Reference(column, (0, bottom), x_reflection=True),
    )
    gds = tmp_path / "tile.gds"
    cells.library.write_gds(str(gds))
    findings = run_drc(gds, tmp_path / "drc", cell_name="tile")
    assert {f.category for f in findings} <= {
        "M1.S.2", "M2.S.2", "M4.AUX.1", "M4.AUX.2", "GATE.S.3",
    }  # fmt: skip
    assert not [f for f in findings if bottom * 1000 - 1 <= f.bbox[1] <= top * 1000 + 1]


def _settle(devices, inputs):
    """Every net static CMOS `devices` drive from `inputs` (net -> 0/1), switch level."""
    values = {"VDD": 1, "VSS": 0, **inputs}
    by_net = {}
    for d in devices:
        for net in (d.drain, d.source):
            by_net.setdefault(net, []).append(d)

    def reaches(net, rail, kind, seen):
        if net == rail:
            return True
        seen.add(net)
        for d in by_net.get(net, ()):
            on = values.get(d.gate) == (1 if kind == "n" else 0)
            other = d.source if d.drain == net else d.drain
            if d.model.startswith(kind + "mos") and on and other not in seen:
                if reaches(other, rail, kind, seen):
                    return True
        return False

    for _ in range(64):
        changed = False
        for net in by_net:
            if net in values:
                continue
            up, down = reaches(net, "VDD", "p", set()), reaches(net, "VSS", "n", set())
            if up != down:
                values[net], changed = int(up), True
        if not changed:
            return values
    raise AssertionError("the logic did not settle")


@pytest.mark.parametrize("wordlines", [4, 8, 16, 32, 128])
@pytest.mark.parametrize("port", ["W", "R"])
def test_each_decoder_selects_one_row_and_one_wordline(port, wordlines):
    """Switch level, from the latched address: one select row, one wordline
    in each half, and none without the enable."""
    from chipforge_asap7.sram1r1w.periphery import MacroSpec, decode_devices

    spec = MacroSpec(wordlines=wordlines)
    select = "wysel" if port == "W" else "rysel"
    wordline = "WWL" if port == "W" else "RWL"
    enable = "WEN" if port == "W" else "REN"
    devices = decode_devices(spec, port)
    step = max(1, spec.words // 64)
    for address in [*range(0, spec.words, step), spec.words - 1]:
        for on in (0, 1):
            latched = {}
            for k in range(spec.address_bits):
                latched[f"{port}AL<{k}>"] = address >> k & 1
                latched[f"{port}ALN<{k}>"] = 1 - (address >> k & 1)
            v = _settle(devices, {enable: on, **latched})
            row, slot = address & 3, address >> 2
            for k in range(4):
                assert v[f"{select}[{k}]"] == (k == row), (address, k)
                assert v[f"{select}n[{k}]"] == (k != row), (address, k)
            for half in ("U", "L"):
                lines = [i for i in range(wordlines) if v[f"{wordline}{half}[{i}]"]]
                assert lines == ([slot] if on else []), (address, on, half)
