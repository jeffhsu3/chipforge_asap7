"""LVS of a layout this package did not draw: an assembled macro.

`run_lvs` proves one generated cell against a reference written for it.  A
macro is different in three ways, and `run_hierarchical_lvs` is `run_lvs` with
each of them handled:

* Its reference is the designer's netlist: released CDL with ``NFIN`` devices
  and one net for the inside of a series stack.  So the reference is expanded
  to unit fins and source/drain is extracted the way the CDL assumes
  (``merge_fin_diffusion``).
* It holds an SRAM array, whose bitcell is not a circuit in the layout -- its
  transistors are completed by its neighbours -- so such cells are flattened on
  both sides (``flatten_circuits``).
* It holds released standard cells whose GDS and CDL disagree about the order
  of a series stack.  Calibre's gate recognition lets that pass; KLayout's
  compare is strictly topological.  A first pass finds the library cells that
  fail, or that only match by swapping pins; those the series/parallel check
  proves equivalent are compared by pin name in a second pass, and named in
  the result.  Nothing else is excused.

`summarize_lvsdb` turns a report database into a dictionary.  It runs inside
KLayout's own Python, so it needs the binary and nothing else.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from .lvs import LVSResult, find_klayout, run_lvs

__all__ = ["HierarchicalLVSResult", "run_hierarchical_lvs", "summarize_lvsdb"]

ASAP7_LIBRARY_CELLS = "*_ASAP7_75t_*"


def summarize_lvsdb(
    report: str | Path, *, klayout: str | Path | None = None, timeout: float = 600
) -> dict[str, Any]:
    """The cross-reference of `report` (``.lvsdb`` or ``.lvsdb.gz``) as a dictionary.

    ``circuits`` lists every circuit pair bottom-up with its status, counts of
    matched and unmatched nets, devices, pins and subcircuits, a few examples of
    each that failed, ``renamed_pins`` (pins KLayout paired under different
    names to make the circuit match) and, where a pair differs,
    ``series_order_only``: True when the two are the same series/parallel
    network with the same fins per input, False when they are not, None when
    the circuit has feedback and the question has no answer.  A failing
    circuit also gets ``supply_shorts``: instance pins that sit on a supply net
    in the layout and not in the reference, which is how a short to a rail
    shows up when the compare itself can only say that nothing matches.
    """
    report_path = Path(report).expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"LVS report does not exist: {report_path}")
    out = report_path.with_name(report_path.name.split(".")[0] + ".summary.json")
    completed = subprocess.run(
        [
            str(find_klayout(klayout)), "-b", "-r", str(Path(__file__).with_name("lvs_report_pya.py")),
            "-rd", f"db_path={report_path}", "-rd", f"out_path={out}",
        ],
        check=False, capture_output=True, text=True, timeout=timeout,
    )  # fmt: skip
    if completed.returncode != 0 or not out.is_file():
        raise RuntimeError(
            f"KLayout could not read {report_path}: {(completed.stdout + completed.stderr)[-1500:]}"
        )
    return json.loads(out.read_text())


@dataclass(frozen=True)
class HierarchicalLVSResult:
    """`matched` is the verdict; the rest says what it rests on."""

    matched: bool
    lvs: LVSResult
    summary: dict[str, Any]
    #: library cells compared by pin name, each proven to differ from its CDL
    #: in series order alone
    series_order_cells: tuple[str, ...] = ()
    #: circuits that do not match, bottom-up; a parent of one is only "skipped"
    failing: tuple[str, ...] = field(default=())

    def describe(self) -> str:
        lines = [
            f"LVS {'MATCH' if self.matched else 'MISMATCH'}: {self.summary.get('top')}"
        ]
        if self.series_order_cells:
            lines.append(
                "  compared by pin name (GDS and CDL differ in series order only): "
                + ", ".join(self.series_order_cells)
            )
        for circuit in self.summary["circuits"]:
            if circuit["status"] == "match":
                continue
            lines.append(
                f"  {circuit['layout'] or circuit['reference']}: {circuit['status']}"
            )
            for what, tally in circuit["counts"].items():
                bad = {k: v for k, v in tally.items() if k != "match"}
                if bad:
                    lines.append(f"      {what}: {tally.get('match', 0)} match, {bad}")
                    for first, second in circuit["examples"].get(what, [])[:4]:
                        lines.append(
                            f"          layout {first!s:28s} | reference {second}"
                        )
            shorts = circuit.get("supply_shorts", [])
            if shorts:
                lines.append(
                    f"      {sum(s['count'] for s in shorts)} instance pin(s) on a supply here only:"
                )
                for short in shorts[:8]:
                    lines.append(
                        f"          {short['supply']} <- {short['circuit']}.{short['pin']} x{short['count']}"
                    )
                if len(shorts) > 8:
                    lines.append(f"          ... and {len(shorts) - 8} more pins")
        return "\n".join(lines)


def run_hierarchical_lvs(
    gds: str | Path,
    schematic: str | Path,
    output_dir: str | Path,
    *,
    cell_name: str,
    flatten_circuits: Sequence[str] = (),
    library_cells: str = ASAP7_LIBRARY_CELLS,
    double_implant_is_tap: bool = True,
    klayout: str | Path | None = None,
    timeout: float = 3600,
) -> HierarchicalLVSResult:
    """Transistor-level LVS of an assembled macro; see the module docstring.

    `schematic` is ordinary SPICE or CDL with ``nfin=`` devices.  Bodies are
    tied by declaration, because tapless library cells expose their wells as
    pins otherwise; that the taps exist is DRC's question (``ACTIVE.LUP.1``).
    """
    out = Path(output_dir).expanduser().resolve()

    def one_pass(tag: str, blank: Sequence[str]) -> tuple[LVSResult, dict[str, Any]]:
        lvs = run_lvs(
            gds, schematic, out / tag, cell_name=cell_name, klayout=klayout,
            asap7_standard_cell=True, flatten_circuits=flatten_circuits, blank_circuits=blank,
            double_implant_is_tap=double_implant_is_tap, timeout=timeout,
        )  # fmt: skip
        return lvs, summarize_lvsdb(lvs.report, klayout=klayout, timeout=timeout)

    lvs, summary = one_pass("pass1", ())
    excused = tuple(
        circuit["layout"]
        for circuit in summary["circuits"]
        if circuit["layout"]
        and fnmatchcase(circuit["layout"], library_cells)
        and (circuit["status"] != "match" or circuit["renamed_pins"])
        and circuit.get("series_order_only") is True
    )
    if excused:
        lvs, summary = one_pass("pass2", excused)
    failing = tuple(
        circuit["layout"] or circuit["reference"]
        for circuit in summary["circuits"]
        if circuit["status"] in ("mismatch", "nomatch")
    )
    return HierarchicalLVSResult(
        matched=bool(summary["matched"]) and lvs.matched,
        lvs=lvs,
        summary=summary,
        series_order_cells=excused,
        failing=failing,
    )
