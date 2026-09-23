"""Open-source LVS and parasitic extraction support for ASAP7.

The public ASAP7 release does not contain the proprietary Calibre LVS/xACT
rule decks used to create the released standard-cell views.  This package is
therefore a research-grade, explicitly calibrated flow rather than a foundry
sign-off replacement.  See :mod:`chipforge_asap7.verification.pex` for the
assumptions and machine-readable uncertainty report.
"""

from .drc import DRCViolation, drc_counts, find_drc_deck, run_drc
from .lvs import (
    LVSResult,
    find_klayout,
    normalize_asap7_cdl_reference,
    render_bitline_mux_group_lvs_schematic,
    render_bitline_mux_lvs_schematic,
    render_driver_slice_lvs_schematic,
    render_finfet_lvs_schematic,
    render_io_column_lvs_schematic,
    render_nand_lvs_schematic,
    render_nand_row_lvs_schematic,
    render_output_latch_lvs_schematic,
    render_sense_amp_lvs_schematic,
    render_sense_amp_row_lvs_schematic,
    render_write_driver_lvs_schematic,
    run_lvs,
)
from .lvs_report import HierarchicalLVSResult, run_hierarchical_lvs, summarize_lvsdb
from .parasitics import Parasitics, effective_resistance
from .pex import PEXResult, find_fastercap, run_open_pex
from .reduce import Edit, ReductionConfig, ReductionResult, reduce_layout
from .reference import ValidationResult, validate_released_inverter
from .stack import calibration_manifest, write_kpex_technology

__all__ = [
    "DRCViolation",
    "Edit",
    "HierarchicalLVSResult",
    "LVSResult",
    "PEXResult",
    "Parasitics",
    "ReductionConfig",
    "ReductionResult",
    "ValidationResult",
    "calibration_manifest",
    "drc_counts",
    "effective_resistance",
    "find_drc_deck",
    "find_fastercap",
    "find_klayout",
    "normalize_asap7_cdl_reference",
    "reduce_layout",
    "render_bitline_mux_group_lvs_schematic",
    "render_bitline_mux_lvs_schematic",
    "render_driver_slice_lvs_schematic",
    "render_finfet_lvs_schematic",
    "render_io_column_lvs_schematic",
    "render_nand_lvs_schematic",
    "render_nand_row_lvs_schematic",
    "render_output_latch_lvs_schematic",
    "render_sense_amp_lvs_schematic",
    "render_sense_amp_row_lvs_schematic",
    "render_write_driver_lvs_schematic",
    "run_drc",
    "run_hierarchical_lvs",
    "run_lvs",
    "run_open_pex",
    "summarize_lvsdb",
    "validate_released_inverter",
    "write_kpex_technology",
]
