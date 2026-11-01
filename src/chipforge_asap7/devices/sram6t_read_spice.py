"""Transient BSIM-CMG read path of an ASAP7 6T SRAM column, clock to Q.

This is the read bench of OpenFinRAM's ``scripts/characterize_read.py``
(``--mode table3``, its MOST-report Table III reproduction), moved from Xyce
onto ngspice with BSIM-CMG 107 loaded through OSDI:

* one real ``sram_cell_6t_122`` cell per stored value (pull-down and pass
  gate two fins, pull-up one), on ``nmos_sram``/``pmos_sram``;
* the rest of the column as an equivalent bitline capacitance per cell;
* OpenFinRAM's precharge/column mux, the 16-transistor ``sense_amp_sram``
  (:func:`~chipforge_asap7.devices.sense_amp_spice.render_openfinram_sense_amp`),
  the ``io_nand`` latch and the tristate output buffer;
* a replica-timed sense enable: a separate deck measures the wordline to
  sense-margin bitline access, and SAE fires that long after CLK plus a guard.

Two cycles read a stored 1 and then a stored 0, giving the clock-to-Q cell
rise and fall delays and the 10-90 % output transitions. The column is a
lumped capacitance with no wire resistance, and the read is one corner of
one model card: these are compact-model estimates, not Liberty signoff.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .sense_amp_spice import render_openfinram_sense_amp
from .spice import (
    compile_bsimcmg107,
    find_default_bsimcmg107_source,
    find_default_model_card,
    translate_asap7_model,
)

__all__ = [
    "Sram6tReadConditions",
    "Sram6tReadResult",
    "render_access_deck",
    "render_read_deck",
    "run_sram6t_read",
]

#: OpenFinRAM's per-cell bitline load (wire plus junction).
DEFAULT_BITLINE_FF_PER_CELL = 0.06
#: The MOST report's Table III stimulus point.
DEFAULT_SLEW_PS = 40.0
DEFAULT_OUTPUT_LOAD_FF = 46.08
DEFAULT_WORDS = (128, 256, 512, 1024)
MODELS = ("nmos_sram", "pmos_sram", "nmos_rvt", "pmos_rvt")

# OpenFinRAM's precharge/column mux, latch NAND and tristate output buffer,
# as ngspice OSDI devices (NFIN, no planar W).
PERIPHERY = """\
.subckt sram_prech_ymux_6t112_v1 blprechn yseln ysel san sa bln bl VDD VSS
N0 bln ysel san VSS nmos_rvt L=20n NFIN=3
N1 bl ysel sa VSS nmos_rvt L=20n NFIN=3
N2 bln blprechn VDD VDD pmos_rvt L=20n NFIN=3
N3 bl blprechn VDD VDD pmos_rvt L=20n NFIN=3
N4 san yseln bln VDD pmos_rvt L=20n NFIN=3
N5 sa yseln bl VDD pmos_rvt L=20n NFIN=3
.ends sram_prech_ymux_6t112_v1
.subckt io_nand_3f_6f A B Y VDD VSS
N0 VSS A 6 VSS nmos_rvt L=20n NFIN=3
N1 6 A VSS VSS nmos_rvt L=20n NFIN=3
N2 Y B 6 VSS nmos_rvt L=20n NFIN=3
N3 6 B Y VSS nmos_rvt L=20n NFIN=3
N4 VDD A Y VDD pmos_rvt L=20n NFIN=3
N5 Y B VDD VDD pmos_rvt L=20n NFIN=3
.ends io_nand_3f_6f
.subckt TBUF_INV A ENB EN VDD VSS Y
NP1 n1 A VDD VDD pmos_rvt L=20n NFIN=48
NP2 Y ENB n1 VDD pmos_rvt L=20n NFIN=36
NN1 n2 A VSS VSS nmos_rvt L=20n NFIN=48
NN2 Y EN n2 VSS nmos_rvt L=20n NFIN=36
.ends TBUF_INV
"""


@dataclass(frozen=True)
class Sram6tReadConditions:
    """Supply, stimulus and column model for one read characterization."""

    vdd: float = 0.7
    temperature_c: float = 25.0
    slew_ps: float = DEFAULT_SLEW_PS
    output_load_ff: float = DEFAULT_OUTPUT_LOAD_FF
    bitline_ff_per_cell: float = DEFAULT_BITLINE_FF_PER_CELL
    sense_margin_v: float = 0.1
    sae_guard_ps: float = 85.0
    pull_down_fins: int = 2
    pass_gate_fins: int = 2
    pull_up_fins: int = 1
    step_ps: float = 1.0

    def __post_init__(self) -> None:
        if self.vdd <= 0:
            raise ValueError("vdd must be positive")
        if not 0 < self.sense_margin_v < self.vdd:
            raise ValueError("sense_margin_v must be positive and below vdd")
        for name in ("slew_ps", "step_ps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("output_load_ff", "bitline_ff_per_cell", "sae_guard_ps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("pull_down_fins", "pass_gate_fins", "pull_up_fins"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.slew_ps >= 1000:
            raise ValueError("slew_ps must fit inside the 2 ns clock phases")


@dataclass(frozen=True)
class Sram6tReadResult:
    """Measured read of one column depth; delays in seconds."""

    words: int
    bitline_access_s: float | None
    sae_after_clk_s: float | None
    cell_rise_s: float | None
    cell_fall_s: float | None
    rise_transition_s: float | None
    fall_transition_s: float | None
    q_high_v: float | None
    q_low_v: float | None
    passed: bool


def _cell(prefix: str, stored: int, wordline: str, c: Sram6tReadConditions):
    q, qb = f"{prefix}Q", f"{prefix}QB"
    pd, pg, pu = c.pull_down_fins, c.pass_gate_fins, c.pull_up_fins
    lines = [
        f"N{prefix}0 {qb} {wordline} BLN 0 nmos_sram L=20n NFIN={pg}",
        f"N{prefix}1 {q} {qb} 0 0 nmos_sram L=20n NFIN={pd}",
        f"N{prefix}2 0 {q} {qb} 0 nmos_sram L=20n NFIN={pd}",
        f"N{prefix}3 BL {wordline} {q} 0 nmos_sram L=20n NFIN={pg}",
        f"N{prefix}4 {q} {qb} VDD VDD pmos_sram L=20n NFIN={pu}",
        f"N{prefix}5 VDD {q} {qb} VDD pmos_sram L=20n NFIN={pu}",
    ]
    high, low = (0, c.vdd) if stored == 0 else (c.vdd, 0)
    return "\n".join(lines), f".ic v({q})={high:g} v({qb})={low:g}"


def _header(title: str, models: Path, c: Sram6tReadConditions) -> str:
    return f'* {title}\n.include "{models.resolve()}"\n.temp {c.temperature_c:g}\n'


def _control(osdi: Path) -> str:
    path = str(osdi.resolve())
    if any(ch.isspace() for ch in path):
        raise ValueError(f"ngspice pre_osdi paths cannot contain whitespace: {path}")
    return f".control\npre_osdi {path}\nrun\nquit\n.endc\n.end\n"


def _validate_words(words: int) -> None:
    if isinstance(words, bool) or not isinstance(words, int) or words < 1:
        raise ValueError("words must be a positive integer")


def render_access_deck(
    words: int, *, models: Path, osdi: Path, conditions: Sram6tReadConditions
) -> str:
    """Wordline 50 % to the bitline falling by the sense margin, reading a 0."""
    _validate_words(words)
    c = conditions
    cbl = words * c.bitline_ff_per_cell
    cell, ic = _cell("C", 0, "WL", c)
    return (
        _header(f"6T read access, {words} cells on the bitline", models, c)
        + f"""Vsupply VDD 0 {c.vdd:g}
{cell}
Cbl BL 0 {cbl:g}f
Cbln BLN 0 {cbl:g}f
{ic}
.ic v(BL)={c.vdd:g} v(BLN)={c.vdd:g}
Vwl WL 0 PULSE(0 {c.vdd:g} 1n 10p 10p 5n 10n)
.tran {c.step_ps:g}p 3n
.measure tran t_access TRIG v(WL) VAL={c.vdd / 2:g} RISE=1 TARG v(BL) VAL={c.vdd - c.sense_margin_v:g} FALL=1
"""
        + _control(osdi)
    )


def render_read_deck(
    words: int,
    *,
    sae_after_clk_ps: float,
    models: Path,
    osdi: Path,
    conditions: Sram6tReadConditions,
) -> str:
    """Two clock cycles: read a stored 1, then the stored 0, through to Q."""
    _validate_words(words)
    c = conditions
    if not 0 < sae_after_clk_ps < 1800:
        raise ValueError("SAE must fire inside the 2 ns read phase")
    cbl = words * c.bitline_ff_per_cell
    sae = 1.0 + sae_after_clk_ps / 1000.0
    sl, vdd, half = c.slew_ps / 1000.0, c.vdd, c.vdd / 2
    zero, ic_zero = _cell("L", 0, "WLlo", c)
    one, ic_one = _cell("H", 1, "WLhi", c)

    def pulse(v0, v1, delay_ns, width_ns, period_ns):
        return f"PULSE({v0:g} {v1:g} {delay_ns:g}n {sl:g}n {sl:g}n {width_ns:g}n {period_ns:g}n)"

    return (
        _header(f"6T read, clock to Q, {words} cells on the bitline", models, c)
        + PERIPHERY
        + render_openfinram_sense_amp()
        + f"""Vsupply VDD 0 {vdd:g}
{zero}
{one}
Xmux blprechn yseln ysel san sa BLN BL VDD 0 sram_prech_ymux_6t112_v1
Xsa sa san SAE SAPRECHN qa qan VDD 0 sense_amp_sram
Xn0 n49 qa n54 VDD 0 io_nand_3f_6f
Xn1 n54 qan n49 VDD 0 io_nand_3f_6f
Ninv_n n48 n49 0 0 nmos_rvt L=20n NFIN=3
Ninv_p n48 n49 VDD VDD pmos_rvt L=20n NFIN=3
Xbuf n48 oeb oe VDD 0 Q TBUF_INV
Cq Q 0 {c.output_load_ff:g}f
Cbl BL 0 {cbl:g}f
Cbln BLN 0 {cbl:g}f
{ic_zero}
{ic_one}
.ic v(BL)={vdd:g} v(BLN)={vdd:g} v(sa)={vdd:g} v(san)={vdd:g} v(qa)={vdd:g} v(qan)={vdd:g} v(Q)=0
* The output latch starts holding 0 (Q follows n49), so the first cycle's rise
* is a read of the stored 1. Left to the DC solve, ngspice can settle the
* latch at 1, and that rise would time only the buffer enable.
.ic v(n49)=0 v(n54)={vdd:g}
Vclk clk 0 {pulse(0, vdd, 1, 2, 4)}
Vbpn blprechn 0 {pulse(0, vdd, 1, 2, 4)}
Vysel ysel 0 {pulse(0, vdd, 1, 2, 4)}
Vyseln yseln 0 {pulse(vdd, 0, 1, 2, 4)}
Voe oe 0 {pulse(0, vdd, 1, 2, 4)}
Voeb oeb 0 {pulse(vdd, 0, 1, 2, 4)}
Vwlhi WLhi 0 {pulse(0, vdd, 1, 1.8, 100)}
Vwllo WLlo 0 {pulse(0, vdd, 5, 1.8, 100)}
Vsaprechn SAPRECHN 0 {pulse(0, vdd, sae - 0.02, 1.85, 4)}
Vsae SAE 0 {pulse(0, vdd, sae, 1.8, 4)}
.tran {c.step_ps:g}p 8n
.measure tran t_cell_rise TRIG v(clk) VAL={half:g} RISE=1 TARG v(Q) VAL={half:g} RISE=1
.measure tran t_cell_fall TRIG v(clk) VAL={half:g} RISE=2 TARG v(Q) VAL={half:g} FALL=1
.measure tran t_q_rise_10 WHEN v(Q)={0.1 * vdd:g} RISE=1
.measure tran t_q_rise_90 WHEN v(Q)={0.9 * vdd:g} RISE=1
.measure tran t_q_fall_90 WHEN v(Q)={0.9 * vdd:g} FALL=1
.measure tran t_q_fall_10 WHEN v(Q)={0.1 * vdd:g} FALL=1
.measure tran q_high FIND v(Q) AT=4.5n
.measure tran q_low FIND v(Q) AT=7.5n
"""
        + _control(osdi)
    )


def _measure(log: str, name: str) -> float | None:
    match = re.search(rf"^\s*{name}\s*=\s*([-+0-9.eE]+)", log, re.MULTILINE)
    return float(match.group(1)) if match else None


def _simulate(deck: Path, text: str, ngspice: str) -> str:
    simulator = shutil.which(ngspice) if not Path(ngspice).is_file() else ngspice
    if simulator is None:
        raise FileNotFoundError(f"ngspice was not found: {ngspice!r}")
    deck.write_text(text)
    completed = subprocess.run(
        [str(simulator), "-b", deck.name],
        cwd=deck.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    log = completed.stdout + completed.stderr
    deck.with_suffix(".log").write_text(log)
    if completed.returncode or "simulation(s) aborted" in log:
        raise RuntimeError(
            f"ngspice failed for {deck.name} (exit {completed.returncode}); "
            f"see {deck.with_suffix('.log')}"
        )
    return log


def _difference(log: str, later: str, earlier: str) -> float | None:
    a, b = _measure(log, later), _measure(log, earlier)
    return a - b if a is not None and b is not None else None


def run_sram6t_read(
    *,
    model_card: Path,
    osdi: Path,
    output_dir: Path,
    words: Sequence[int] = DEFAULT_WORDS,
    conditions: Sram6tReadConditions | None = None,
    ngspice: str = "ngspice",
) -> list[Sram6tReadResult]:
    """Characterize each column depth; write decks, logs and results.json."""
    c = conditions or Sram6tReadConditions()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models = output_dir / "asap7_sram_rvt_bsimcmg107.pm"
    card = model_card.read_text()
    models.write_text("\n".join(translate_asap7_model(card, m) for m in MODELS))
    results = []
    for depth in words:
        _validate_words(depth)
        access_log = _simulate(
            output_dir / f"access_{depth}.sp",
            render_access_deck(depth, models=models, osdi=osdi, conditions=c),
            ngspice,
        )
        access = _measure(access_log, "t_access")
        if access is None:
            # The bitline never developed the margin: no read to time.
            results.append(
                Sram6tReadResult(
                    depth, None, None, None, None, None, None, None, None, False
                )
            )
            continue
        sae_ps = access * 1e12 + c.sae_guard_ps
        log = _simulate(
            output_dir / f"read_{depth}.sp",
            render_read_deck(
                depth, sae_after_clk_ps=sae_ps, models=models, osdi=osdi, conditions=c
            ),
            ngspice,
        )
        rise, fall = _measure(log, "t_cell_rise"), _measure(log, "t_cell_fall")
        high, low = _measure(log, "q_high"), _measure(log, "q_low")
        results.append(
            Sram6tReadResult(
                words=depth,
                bitline_access_s=access,
                sae_after_clk_s=sae_ps * 1e-12,
                cell_rise_s=rise,
                cell_fall_s=fall,
                rise_transition_s=_difference(log, "t_q_rise_90", "t_q_rise_10"),
                fall_transition_s=_difference(log, "t_q_fall_10", "t_q_fall_90"),
                q_high_v=high,
                q_low_v=low,
                passed=rise is not None
                and fall is not None
                and high is not None
                and high > 0.9 * c.vdd
                and low is not None
                and low < 0.1 * c.vdd,
            )
        )
    (output_dir / "results.json").write_text(
        json.dumps(
            {
                "conditions": asdict(c),
                "model_card": str(model_card),
                "results": [asdict(r) for r in results],
            },
            indent=2,
        )
        + "\n"
    )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``asap7-sram6t-read``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--words", type=int, nargs="+", default=list(DEFAULT_WORDS))
    parser.add_argument("--vdd", type=float, default=0.7)
    parser.add_argument("--temperature-c", type=float, default=25.0)
    parser.add_argument("--slew-ps", type=float, default=DEFAULT_SLEW_PS)
    parser.add_argument("--output-load-ff", type=float, default=DEFAULT_OUTPUT_LOAD_FF)
    parser.add_argument(
        "--bitline-ff-per-cell", type=float, default=DEFAULT_BITLINE_FF_PER_CELL
    )
    parser.add_argument("--sae-guard-ps", type=float, default=85.0)
    parser.add_argument("--step-ps", type=float, default=1.0)
    parser.add_argument("--model-card", type=Path)
    parser.add_argument("--osdi", type=Path)
    parser.add_argument("--bsimcmg-source", type=Path)
    parser.add_argument("--openvaf", default="openvaf")
    parser.add_argument("--ngspice", default="ngspice")
    parser.add_argument("--out", type=Path, default=Path("build/sram6t_read"))
    args = parser.parse_args(argv)

    model_card = args.model_card or find_default_model_card()
    if model_card is None or not model_card.is_file():
        parser.error(
            "ASAP7 model card not found; pass --model-card or ASAP7_MODEL_CARD"
        )
    try:
        conditions = Sram6tReadConditions(
            vdd=args.vdd,
            temperature_c=args.temperature_c,
            slew_ps=args.slew_ps,
            output_load_ff=args.output_load_ff,
            bitline_ff_per_cell=args.bitline_ff_per_cell,
            sae_guard_ps=args.sae_guard_ps,
            step_ps=args.step_ps,
        )
    except ValueError as exc:
        parser.error(str(exc))
    output_dir = args.out.resolve()
    osdi = args.osdi
    if osdi is None:
        source = args.bsimcmg_source or find_default_bsimcmg107_source()
        if source is None:
            parser.error(
                "BSIM-CMG 107 source not found; pass --bsimcmg-source or --osdi"
            )
        osdi = compile_bsimcmg107(
            source, output_dir / "BSIMCMG107.osdi", openvaf=args.openvaf
        )
    elif not osdi.is_file():
        parser.error(f"OSDI library does not exist: {osdi}")

    results = run_sram6t_read(
        model_card=model_card,
        osdi=osdi,
        output_dir=output_dir,
        words=args.words,
        conditions=conditions,
        ngspice=args.ngspice,
    )

    def ns(value):
        return f"{value * 1e9:.3f}" if value is not None else "  -  "

    print(
        f"{'words':>6} {'access':>7} {'rise':>7} {'fall':>7} {'rise tr':>8} {'fall tr':>8}  (ns)"
    )
    for r in results:
        print(
            f"{r.words:>6} {ns(r.bitline_access_s):>7} {ns(r.cell_rise_s):>7} "
            f"{ns(r.cell_fall_s):>7} {ns(r.rise_transition_s):>8} "
            f"{ns(r.fall_transition_s):>8}  {'ok' if r.passed else 'FAILED'}"
        )
    print(f"artifacts: {output_dir}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
