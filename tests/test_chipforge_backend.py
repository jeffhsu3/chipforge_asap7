"""The ASAP7 ROM macro backends register in chipforge, as chipforge_gt2n's do.

Needs the ``chipforge`` extra; skipped without it.
"""

from __future__ import annotations

import pytest

from chipforge_asap7._chipforge import rom_interface_available

pytestmark = pytest.mark.skipif(
    not rom_interface_available(), reason="needs the chipforge extra"
)


def test_importing_the_package_registers_both_backends():
    import chipforge_asap7  # noqa: F401
    from chipforge.memory.rom_interface import rom_macro_interface

    class Config:
        rom_macro_backend = "asap7_brom"

    brom = rom_macro_interface(Config)
    assert (brom.read, brom.addr_hold_edge, brom.reset) == ("two_phase", "fall", True)
    assert brom.rows_per_bank == 256 and brom.in_macro_mux
    Config.rom_macro_backend = "asap7_nor_rom"
    nor = rom_macro_interface(Config)
    assert (nor.read, nor.addr_hold_edge) == ("edge_flow_through", "rise")


def test_reregistering_changes_the_bank(tmp_path):
    from chipforge.memory.rom_interface import ROM_MACRO_INTERFACES

    from chipforge_asap7.chipforge_backend import register_asap7_brom_backend

    try:
        register_asap7_brom_backend(64, tmp_path, lanes=2)
        interface = ROM_MACRO_INTERFACES["asap7_brom"]
        assert interface.rows_per_bank == 64 and interface.lane_cells
        assert interface.collateral("ROM_FULL_64x148_L1").lef == (
            tmp_path / "rom_full_64x148_x2" / "macro.lef"
        )
    finally:
        register_asap7_brom_backend()


def test_the_banks_live_in_core_unless_redirected(monkeypatch, tmp_path):
    from chipforge_asap7.chipforge_backend.paths import ENV, rom_macros_dir

    monkeypatch.setenv(ENV, str(tmp_path))
    assert rom_macros_dir() == tmp_path
    monkeypatch.delenv(ENV)
    assert (rom_macros_dir().parent / "pyproject.toml").is_file()

