"""Formal KLayout LVS regressions for unit-fin and released-cell modes.

Skips when KLayout or the public standard-cell release is missing; set
``ASAP7_REQUIRE_TOOLS=1`` to turn those skips into failures.
"""

from __future__ import annotations

from pathlib import Path

import gdspy

from chipforge_asap7.devices import FinFETSpec, build_finfet
from chipforge_asap7.verification.lvs import (
    find_klayout,
    normalize_asap7_cdl_reference,
    render_finfet_lvs_schematic,
    run_lvs,
)


def _klayout() -> Path | None:
    try:
        return find_klayout()
    except FileNotFoundError:
        return None


KLAYOUT = _klayout()


def _write_device(spec: FinFETSpec, path: Path) -> None:
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    gdspy.current_library = library
    build_finfet(spec, lib=library)
    library.write_gds(str(path))


def test_generated_finfet_lvs_matches_and_detects_wrong_fin_count(
    tmp_path: Path, external_tool
):
    external_tool(KLAYOUT is not None, "KLayout is not installed (set KLAYOUT_BIN)")
    spec = FinFETSpec(flavor="p", fins=2, fingers=2, vt="lvt")
    gds = tmp_path / "device.gds"
    _write_device(spec, gds)

    reference = tmp_path / "reference.spice"
    reference.write_text(render_finfet_lvs_schematic(spec))
    matched = run_lvs(gds, reference, tmp_path / "match", cell_name=spec.cell_name)
    assert matched.matched
    assert matched.report.is_file()
    assert matched.extracted_netlist.read_text().count("M$") == spec.total_fins

    wrong = tmp_path / "wrong.spice"
    wrong.write_text(
        render_finfet_lvs_schematic(
            FinFETSpec(flavor="p", fins=1, fingers=2, vt="lvt"),
            cell_name=spec.cell_name,
        )
    )
    mismatch = run_lvs(gds, wrong, tmp_path / "mismatch", cell_name=spec.cell_name)
    assert not mismatch.matched

    # A topology error must fail independently of sizing.  Short every
    # physical D terminal to G in an otherwise correctly sized reference.
    shorted_lines: list[str] = []
    for line in reference.read_text().splitlines():
        fields = line.split()
        if fields and fields[0].startswith("Mch_"):
            fields[1:4] = ["G" if net == "D" else net for net in fields[1:4]]
            line = " ".join(fields)
        shorted_lines.append(line)
    shorted = tmp_path / "shorted.spice"
    shorted.write_text("\n".join(shorted_lines) + "\n")
    short_mismatch = run_lvs(
        gds, shorted, tmp_path / "short_mismatch", cell_name=spec.cell_name
    )
    assert not short_mismatch.matched


def test_released_cdl_nfin_is_expanded_to_unit_fin_devices(tmp_path: Path):
    source = tmp_path / "reference.cdl"
    source.write_text(
        ".SUBCKT INV A VDD VSS Y\n"
        "MM0 Y A VSS VSS nmos_rvt w=54n l=20n nfin=2\n"
        "+ m=2\n"
        ".ENDS INV\n"
    )
    normalized = normalize_asap7_cdl_reference(source, tmp_path / "unit.cdl")
    text = normalized.read_text()
    assert text.count("nmos_rvt") == 4
    assert "W=7n" in text
    assert "NFIN" not in text.upper()


def test_released_invxp33_lvs_when_public_release_is_available(
    tmp_path: Path, external_tool, released_library: Path
):
    external_tool(KLAYOUT is not None, "KLayout is not installed (set KLAYOUT_BIN)")
    gds = released_library / "GDS/asap7sc7p5t_28_R_220121a.gds"
    cdl = released_library / "CDL/LVS/asap7sc7p5t_28_R.cdl"
    external_tool(
        gds.is_file() and cdl.is_file(),
        "the public ASAP7 standard-cell release is not available",
    )
    result = run_lvs(
        gds,
        cdl,
        tmp_path,
        cell_name="INVxp33_ASAP7_75t_R",
        asap7_standard_cell=True,
    )
    assert result.matched
