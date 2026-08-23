"""Unit and integration regressions for the open RC assembly path."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import gdspy
import pytest

from chipforge_asap7.devices import FinFETSpec, build_finfet
from chipforge_asap7.verification.lvs import (
    render_finfet_lvs_schematic,
    run_lvs,
)
from chipforge_asap7.verification.pex import (
    add_distributed_resistance,
    add_maxwell_capacitance,
    find_fastercap,
    normalize_unit_finfet_netlist,
    reference_subcircuit_pins,
    reorder_subcircuit_pins,
    run_open_pex,
)
from chipforge_asap7.verification.stack import (
    calibration_manifest,
    technology_definition,
    write_kpex_technology,
)

HAVE_KPEX = importlib.util.find_spec("klayout_pex") is not None
FASTERCAP_LIBRARY = os.environ.get("FASTERCAP_LIBRARY_DIR")


def _fastercap() -> Path | None:
    """The field solver `run_open_pex` would use, without the exception."""
    try:
        return find_fastercap()
    except FileNotFoundError:
        return None


def test_unit_finfet_normalization_is_runnable_spice():
    raw = (
        ".SUBCKT F D G S B\n"
        "M$1 D.x G.x S.x B nmos_rvt L=.02U W=.007U AS=.1P\n"
        "Rext_1 D D.x 0 PEX_R\n"
        ".ENDS F\n"
    )
    normalized = normalize_unit_finfet_netlist(raw)
    assert "NPEX_M_1 D.x G.x S.x B nmos_rvt L=.02U NFIN=1 NF=1" in normalized
    assert "Rext_1 D D.x 0.001" in normalized
    assert " W=" not in normalized
    assert "PEX_R" not in normalized


def test_reference_pin_contract_is_restored(tmp_path: Path):
    reference = tmp_path / "ref.sp"
    reference.write_text(".SUBCKT INV A VDD VSS Y\n.ENDS INV\n")
    assert reference_subcircuit_pins(reference, "INV") == ("A", "VDD", "VSS", "Y")
    reordered = reorder_subcircuit_pins(
        ".SUBCKT INV Y A VSS VDD\n.ENDS INV\n",
        "INV",
        ("A", "VDD", "VSS", "Y"),
    )
    assert ".SUBCKT INV A VDD VSS Y" in reordered


def test_generated_technology_records_uncertainty_and_validates(tmp_path: Path):
    technology = technology_definition()
    assert technology["name"] == "asap7-open-calibrated"
    assert technology["process_parasitics"]["resistance"]["layers"]
    assert calibration_manifest()["status"] == "research-grade_not_signoff"
    path = write_kpex_technology(tmp_path / "tech.pb.json")
    assert path.is_file()


@pytest.mark.skipif(not HAVE_KPEX, reason="KPEX is not installed")
def test_maxwell_matrix_is_lumped_without_losing_substrate_capacitance():
    import klayout.db as kdb

    netlist = kdb.Netlist()
    circuit = kdb.Circuit()
    circuit.name = "C"
    netlist.add(circuit)
    for name in ("VSS", "A", "Y"):
        circuit.create_net(name)
    records, warnings = add_maxwell_capacitance(
        netlist,
        "C",
        ["g1_VSUBS", "g2_A", "g3_Y"],
        [
            [3e-15, -1e-15, -2e-15],
            [-1e-15, 4e-15, -1e-15],
            [-2e-15, -1e-15, 5e-15],
        ],
        substrate_net="VSS",
    )
    assert len(records) == 5
    assert warnings == []
    assert sum(item["capacitance_farad"] for item in records) == pytest.approx(8e-15)


@pytest.mark.skipif(not HAVE_KPEX, reason="KPEX is not installed")
def test_kpex_resistance_reconnects_device_terminals(tmp_path: Path, require_klayout):
    import klayout.db as kdb
    from klayout_pex.klayout.lvsdb_extractor import KLayoutExtractionContext
    from klayout_pex.rcx25.r.r_extractor import RExtractor
    from klayout_pex.tech_info import TechInfo
    from klayout_pex_protobuf.kpex.klayout.r_extractor_tech_pb2 import (
        RExtractorTech,
    )

    spec = FinFETSpec()
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    build_finfet(spec, lib=library)
    gds = tmp_path / "device.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.sp"
    reference.write_text(render_finfet_lvs_schematic(spec))
    lvs = run_lvs(gds, reference, tmp_path / "lvs", cell_name=spec.cell_name)
    assert lvs.matched

    tech_path = write_kpex_technology(tmp_path / "tech.pb.json")
    tech = TechInfo.from_json(str(tech_path), dielectric_filter=None)
    lvsdb = kdb.LayoutVsSchematic()
    lvsdb.read(str(lvs.report))
    context = KLayoutExtractionContext.prepare_extraction(
        lvsdb, spec.cell_name, tech, False
    )
    algorithm = RExtractorTech.Algorithm.ALGORITHM_SQUARE_COUNTING
    extractor = RExtractor(context, algorithm, algorithm, 0.5, 0.0, 0.0, True)
    request = extractor.prepare_request()
    result = extractor.extract(request)
    expanded = context.lvsdb.netlist().dup()
    records, reconnects = add_distributed_resistance(
        expanded, spec.cell_name, request, result
    )

    assert len(records) == 19
    assert reconnects == 3
    device = next(expanded.circuit_by_name(spec.cell_name).each_device())
    assert device.net_for_terminal("G").expanded_name().startswith("G.P")
    assert device.net_for_terminal("D").expanded_name().startswith("D.P")
    assert device.net_for_terminal("S").expanded_name().startswith("S.P")
    assert device.net_for_terminal("B").expanded_name() == "B"


@pytest.mark.skipif(not HAVE_KPEX, reason="KPEX is not installed")
def test_complete_open_pex_with_fastercap_when_configured(
    tmp_path: Path, external_tool, require_klayout
):
    fastercap = _fastercap()
    external_tool(
        fastercap is not None,
        "FasterCap is not built (set FASTERCAP_EXE, or run scripts/build_fastercap.sh)",
    )
    spec = FinFETSpec()
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    build_finfet(spec, lib=library)
    gds = tmp_path / "device.gds"
    library.write_gds(str(gds))
    reference = tmp_path / "reference.sp"
    reference.write_text(render_finfet_lvs_schematic(spec))

    result = run_open_pex(
        gds,
        reference,
        tmp_path / "pex",
        cell_name=spec.cell_name,
        fastercap=fastercap,
        fastercap_library_dir=FASTERCAP_LIBRARY,
        substrate_net="B",
        capacitance_mesh_area_um2=4.0,
        capacitance_tolerance=0.15,
    )
    assert result.lvs.matched
    assert "NFIN=1 NF=1" in result.post_layout_netlist.read_text()
    assert result.capacitance_csv.is_file()
    assert result.resistance_csv.read_text().count("\n") == 20
