"""Transient BSIM-CMG read of one NOR-ROM bitline: programmed against floating.

A NOR-ROM bit row (:mod:`chipforge_asap7.devices.rom_bitcell`) is one M2
bitline running along the words.  A programmed word ties its drain to the
bitline through a V1, so the word transistor discharges the precharged line
when its poly wordline rises; an unprogrammed word's drain floats.

The bench runs two bitlines side by side through one precharge and one
wordline pulse, both in the worst case for their outcome:

* ``BLP`` reads a **programmed** bit.  The other ``words - 1`` words are
  programmed too: every one is an off transistor on the line, so the line
  carries the most drain capacitance the selected device must discharge.
* ``BLU`` reads an **unprogrammed** bit next to the same ``words - 1``
  programmed words.  Nothing pulls it down but their off-state leakage, which
  is the most that can droop a line that should read high.

Every word device is a real BSIM-CMG 107 transistor with the tile's fin count,
source on VSS.  The M2 bitline is a distributed RC ladder, one segment per
word (half a 162 nm tile): the stack's M2 sheet resistance over the 18 nm
track, and an assumed capacitance per length.  The sense end is word 0, and
the selected word is the far one, read through the whole line.  The V1/M1
stubs, the grounded isolation gates and the poly wordline's RC are not
modelled.  A 3-fin precharge PMOS at the sense end is off during the read and
leaks toward VDD, as in the column.

Measured from the wordline's 50 % point, on ``BLP``:

* ``t_sense`` -- the line has fallen by the sense margin from its precharged
  level, which the precharge release kicks slightly above VDD (a
  sense-amplifier read);
* ``t_half``  -- the line has crossed VDD/2 (an inverter read, no amplifier).

``BLU`` is read at both instants: its droop is the margin the leakage takes
away.  A depth passes when the amplifier read keeps at least
``min_sense_split`` of the margin between the lines and the inverter read
keeps ``BLU`` above VDD/2 at ``t_half`` with the same fraction of VDD/2 to
spare.  This is one model card at one corner: a compact-model feasibility
estimate, not signoff.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from ..layout.rules import CONTACT_SIZE
from ..verification.stack import STACK_CONDUCTORS
from .rom_bitcell import RomBitcellSpec
from .spice import (
    compile_bsimcmg107,
    find_default_bsimcmg107_source,
    find_default_model_card,
    translate_asap7_model,
)
from .sram6t_read_spice import _control, _measure, _simulate

__all__ = [
    "RomReadConditions",
    "RomReadResult",
    "render_rom_read_deck",
    "run_rom_read",
]

DEFAULT_WORDS = (16, 64, 256, 1024)
DEFAULT_FINS = (1, 2)
#: Same explicit wire assumption as the BROM routed-RC estimate.
DEFAULT_WIRE_FF_PER_UM = 0.2
#: The bitline is a minimum-width M2 track: the stack's sheet resistance
#: over its width, which the tile draws flush with the 18 nm V1s.
DEFAULT_WIRE_OHM_PER_UM = next(
    s.sheet_resistance_ohm for s in STACK_CONDUCTORS if s.name == "M2"
) / (CONTACT_SIZE / 1000)
VTS = ("rvt", "lvt", "slvt")


@dataclass(frozen=True)
class RomReadConditions:
    """Supply, stimulus and bitline model for one read characterization."""

    vdd: float = 0.7
    temperature_c: float = 25.0
    vt: str = "rvt"
    slew_ps: float = 40.0
    wire_ff_per_um: float = DEFAULT_WIRE_FF_PER_UM
    wire_ohm_per_um: float = DEFAULT_WIRE_OHM_PER_UM
    sense_margin_v: float = 0.1
    #: Fraction of each margin the leaking line must keep (see module doc).
    min_sense_split: float = 0.5
    precharge_fins: int = 3
    step_ps: float = 1.0
    window_ns: float = 4.0

    def __post_init__(self) -> None:
        if self.vdd <= 0:
            raise ValueError("vdd must be positive")
        if self.vt not in VTS:
            raise ValueError(f"vt must be one of {VTS}")
        if not 0 < self.sense_margin_v < self.vdd / 2:
            raise ValueError("sense_margin_v must be positive and below vdd/2")
        if not 0 < self.min_sense_split <= 1:
            raise ValueError("min_sense_split must be in (0, 1]")
        for name in ("slew_ps", "step_ps", "window_ns"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("wire_ff_per_um", "wire_ohm_per_um"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        fins = self.precharge_fins
        if isinstance(fins, bool) or not isinstance(fins, int) or fins < 1:
            raise ValueError("precharge_fins must be a positive integer")


@dataclass(frozen=True)
class RomReadResult:
    """One bitline depth at one fin count; times in seconds, levels in volts."""

    fins: int
    words: int
    wire_ff: float
    wire_ohm: float
    blp_precharged_v: float | None
    blu_precharged_v: float | None
    t_sense_s: float | None
    t_half_s: float | None
    blu_at_sense_v: float | None
    blu_at_half_v: float | None
    blu_end_v: float | None
    sense_split_v: float | None
    passed_sense: bool
    passed_inverter: bool


def _word_pitch_um() -> float:
    """M2 along the row per word: one tile of two words per bitcell width."""
    return RomBitcellSpec().width / 2 / 1000


def _wire_ff(words: int, conditions: RomReadConditions) -> float:
    return words * _word_pitch_um() * conditions.wire_ff_per_um


def _wire_ohm(words: int, conditions: RomReadConditions) -> float:
    """Sense end to the far word: ``words - 1`` segments."""
    return (words - 1) * _word_pitch_um() * conditions.wire_ohm_per_um


def _validate(fins: int, words: int) -> None:
    for name, value in (("fins", fins), ("words", words)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    RomBitcellSpec(fins=fins)  # the tile's own fin limits


def render_rom_read_deck(
    fins: int,
    words: int,
    *,
    models: Path,
    osdi: Path,
    conditions: RomReadConditions,
    precharged_v: float | None = None,
) -> str:
    """Precharge both lines, release, raise one wordline; time BLP, watch BLU.

    Releasing the precharge PMOS kicks both floating lines above VDD through
    its gate overlap.  The sense margin is counted from ``precharged_v``,
    BLP's level as the wordline starts to rise (VDD if not yet known): a first run
    measures it, a second times the read from it.
    """
    _validate(fins, words)
    c = conditions
    vdd, half = c.vdd, c.vdd / 2
    sense = (vdd if precharged_v is None else precharged_v) - c.sense_margin_v
    sl = c.slew_ps / 1000.0
    nmos, pmos = f"nmos_{c.vt}", f"pmos_{c.vt}"
    seg_um = _word_pitch_um()
    seg_ohm, seg_ff = seg_um * c.wire_ohm_per_um, seg_um * c.wire_ff_per_um

    def node(line: str, k: int) -> str:
        # Word 0 sits at the sense end; a resistance-free line is one node.
        return f"BL{line}" if k == 0 or seg_ohm == 0 else f"BL{line}_{k}"

    ladder = []
    for line in ("P", "U"):
        for k in range(words):
            if k and seg_ohm:
                ladder.append(
                    f"R{line}{k} {node(line, k - 1)} {node(line, k)} {seg_ohm:g}"
                )
            ladder.append(f"C{line}{k} {node(line, k)} 0 {seg_ff:g}f")
            if k < words - 1:
                ladder.append(
                    f"N{line}{k} {node(line, k)} 0 0 0 {nmos} L=20n NFIN={fins}"
                )
    far = words - 1
    end = 1.1 + c.window_ns
    return (
        f"* NOR-ROM read, {fins}-fin words, {words} words on each bitline\n"
        f'.include "{models.resolve()}"\n.temp {c.temperature_c:g}\n'
        f"Vsupply VDD 0 {vdd:g}\n"
        # Precharge low until 1 ns, the wordline up 0.1 ns after it releases.
        f"Vprechn PRECHN 0 PWL(0 0 1n 0 {1 + sl:g}n {vdd:g})\n"
        f"Vwl WL 0 PWL(0 0 1.1n 0 {1.1 + sl:g}n {vdd:g})\n"
        f"NPP BLP PRECHN VDD VDD {pmos} L=20n NFIN={c.precharge_fins}\n"
        f"NPU BLU PRECHN VDD VDD {pmos} L=20n NFIN={c.precharge_fins}\n"
        # The selected word is the far one, behind the whole line: programmed
        # on BLP; on BLU its drain floats, so it is not on the line at all.
        f"NPsel {node('P', far)} WL 0 0 {nmos} L=20n NFIN={fins}\n"
        + "\n".join(ladder)
        + "\n"
        f".tran {c.step_ps:g}p {end:g}n\n"
        f".measure tran blp_pre FIND v(BLP) AT=1.1n\n"
        f".measure tran blu_pre FIND v(BLU) AT=1.1n\n"
        f".measure tran t_sense TRIG v(WL) VAL={half:g} RISE=1 "
        f"TARG v(BLP) VAL={sense:g} FALL=1\n"
        f".measure tran t_half TRIG v(WL) VAL={half:g} RISE=1 "
        f"TARG v(BLP) VAL={half:g} FALL=1\n"
        f".measure tran blu_sense FIND v(BLU) WHEN v(BLP)={sense:g} FALL=1\n"
        f".measure tran blu_half FIND v(BLU) WHEN v(BLP)={half:g} FALL=1\n"
        f".measure tran blu_end FIND v(BLU) AT={end:g}n\n" + _control(osdi)
    )


def run_rom_read(
    *,
    model_card: Path,
    osdi: Path,
    output_dir: Path,
    fins: Sequence[int] = DEFAULT_FINS,
    words: Sequence[int] = DEFAULT_WORDS,
    conditions: RomReadConditions | None = None,
    ngspice: str = "ngspice",
) -> list[RomReadResult]:
    """Characterize every fin count and depth; write decks, logs, results.json."""
    c = conditions or RomReadConditions()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models = output_dir / f"asap7_{c.vt}_bsimcmg107.pm"
    card = model_card.read_text()
    models.write_text(
        "\n".join(translate_asap7_model(card, f"{t}_{c.vt}") for t in ("nmos", "pmos"))
    )
    results = []
    for f in fins:
        for depth in words:
            first = _simulate(
                output_dir / f"precharge_f{f}_{depth}.sp",
                render_rom_read_deck(f, depth, models=models, osdi=osdi, conditions=c),
                ngspice,
            )
            pre = _measure(first, "blp_pre")
            log = _simulate(
                output_dir / f"read_f{f}_{depth}.sp",
                render_rom_read_deck(
                    f, depth, models=models, osdi=osdi, conditions=c, precharged_v=pre
                ),
                ngspice,
            )
            t_sense, t_half = _measure(log, "t_sense"), _measure(log, "t_half")
            blu_sense, blu_half = _measure(log, "blu_sense"), _measure(log, "blu_half")
            split = (
                blu_sense - (pre - c.sense_margin_v)
                if blu_sense is not None and pre is not None and t_sense is not None
                else None
            )
            half = c.vdd / 2
            results.append(
                RomReadResult(
                    fins=f,
                    words=depth,
                    wire_ff=_wire_ff(depth, c),
                    wire_ohm=_wire_ohm(depth, c),
                    blp_precharged_v=pre,
                    blu_precharged_v=_measure(log, "blu_pre"),
                    t_sense_s=t_sense,
                    t_half_s=t_half,
                    blu_at_sense_v=blu_sense,
                    blu_at_half_v=blu_half,
                    blu_end_v=_measure(log, "blu_end"),
                    sense_split_v=split,
                    passed_sense=split is not None
                    and split >= c.min_sense_split * c.sense_margin_v,
                    passed_inverter=blu_half is not None
                    and blu_half - half >= c.min_sense_split * half,
                )
            )
    (output_dir / "results.json").write_text(
        json.dumps(
            {
                "conditions": asdict(c),
                "model_card": str(model_card),
                "scope": "distributed M2 RC, selected word at the far end; no poly wordline RC; one model card and corner",
                "results": [asdict(r) for r in results],
            },
            indent=2,
        )
        + "\n"
    )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``asap7-rom-read``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fins", type=int, nargs="+", default=list(DEFAULT_FINS))
    parser.add_argument("--words", type=int, nargs="+", default=list(DEFAULT_WORDS))
    parser.add_argument("--vdd", type=float, default=0.7)
    parser.add_argument("--temperature-c", type=float, default=25.0)
    parser.add_argument("--vt", choices=VTS, default="rvt")
    parser.add_argument("--wire-ff-per-um", type=float, default=DEFAULT_WIRE_FF_PER_UM)
    parser.add_argument(
        "--wire-ohm-per-um", type=float, default=DEFAULT_WIRE_OHM_PER_UM
    )
    parser.add_argument("--sense-margin-v", type=float, default=0.1)
    parser.add_argument("--min-sense-split", type=float, default=0.5)
    parser.add_argument("--step-ps", type=float, default=1.0)
    parser.add_argument("--window-ns", type=float, default=4.0)
    parser.add_argument("--model-card", type=Path)
    parser.add_argument("--osdi", type=Path)
    parser.add_argument("--bsimcmg-source", type=Path)
    parser.add_argument("--openvaf", default="openvaf")
    parser.add_argument("--ngspice", default="ngspice")
    parser.add_argument("--out", type=Path, default=Path("build/rom_read"))
    args = parser.parse_args(argv)

    model_card = args.model_card or find_default_model_card()
    if model_card is None or not model_card.is_file():
        parser.error(
            "ASAP7 model card not found; pass --model-card or ASAP7_MODEL_CARD"
        )
    try:
        conditions = RomReadConditions(
            vdd=args.vdd,
            temperature_c=args.temperature_c,
            vt=args.vt,
            wire_ff_per_um=args.wire_ff_per_um,
            wire_ohm_per_um=args.wire_ohm_per_um,
            sense_margin_v=args.sense_margin_v,
            min_sense_split=args.min_sense_split,
            step_ps=args.step_ps,
            window_ns=args.window_ns,
        )
        for f in args.fins:
            for depth in args.words:
                _validate(f, depth)
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

    results = run_rom_read(
        model_card=model_card,
        osdi=osdi,
        output_dir=output_dir,
        fins=args.fins,
        words=args.words,
        conditions=conditions,
        ngspice=args.ngspice,
    )

    def ps(value):
        return f"{value * 1e12:.1f}" if value is not None else "-"

    def mv(value):
        return f"{value * 1e3:.0f}" if value is not None else "-"

    print(
        f"{'fins':>4} {'words':>6} {'t_sense':>8} {'t_half':>8} "
        f"{'BLU@sense':>10} {'BLU@half':>9} {'split':>6}  sense / inverter"
    )
    for r in results:
        print(
            f"{r.fins:>4} {r.words:>6} {ps(r.t_sense_s):>8} {ps(r.t_half_s):>8} "
            f"{mv(r.blu_at_sense_v):>10} {mv(r.blu_at_half_v):>9} "
            f"{mv(r.sense_split_v):>6}  "
            f"{'ok' if r.passed_sense else 'FAIL'} / "
            f"{'ok' if r.passed_inverter else 'FAIL'}   (ps, mV)"
        )
    print(f"artifacts: {output_dir}")
    return 0 if all(r.passed_sense for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
