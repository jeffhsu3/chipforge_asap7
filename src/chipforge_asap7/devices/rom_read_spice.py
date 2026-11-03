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
the selected word is the far one, read through the whole line.  A 3-fin
precharge PMOS at the sense end is off during the read and leaks toward VDD,
as in the column.

The wordline (:func:`wordline_network`) is the drawn one for a ``bits``-row
array: an M3 strap from the decoder at the array's top edge, a V2/V1/V0/LIG
stack onto the continuous gate poly at every strap row, and every row's word
transistor gate on the poly as a BSIM-CMG load.  The read uses the slowest
row.  The wordline's driver (:func:`decoder_driver`) is the decoder's last two
stages as BSIM-CMG transistors: the postdecode NAND, its PA input rising with
the edge and PB already high, and the inverter onto the top of the M3 strap,
sized as the drawn decoder (the default), as `size_rom_decoder` sizes them,
or ideal.  With a decoder driver the predecoders are simulated too
(:func:`~chipforge_asap7.devices.rom_predecode_spice.predecode_network`):
every gate between the address pins and the word's PA and PB lines, and
those nets' routed RC and gate loads, in a ``2 * words``-word decoder whose
word 0 ends its lane half's bitline.  The V1/M1 stubs and the grounded
isolation gates are not modelled.

Each address pin is driven by a released ASAP7 buffer (``BUFx4`` by
default) or an ideal edge.  Measured from the first address edge's 50 %
point -- the buffer input's -- (PA's without the predecoders; the decoder
output's for the ideal driver), on ``BLP``:

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
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

from ..layout.grid import GATE_WIDTH
from ..layout.rules import CONTACT_SIZE
from ..verification.stack import STACK_CONDUCTORS
from .rom_bitcell import (
    HALF_LIG,
    STRAP_WL_XS,
    STRAP_WL_Y,
    RomArraySpec,
    RomBitcellSpec,
)
from .spice import (
    compile_bsimcmg107,
    find_default_bsimcmg107_source,
    find_default_model_card,
    translate_asap7_model,
)
from .sram6t_read_spice import _control, _measure, _simulate

__all__ = [
    "DRIVERS",
    "RomReadConditions",
    "RomReadResult",
    "WordlineNetwork",
    "decoder_driver",
    "decoder_spec",
    "render_rom_read_deck",
    "run_rom_read",
    "strap_stack_ohm",
    "wordline_network",
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
_STACK = {s.name: s for s in STACK_CONDUCTORS}
#: Every WL is one M3 strap the width of a via, and gate poly at GATE.W.1.
M3_OHM_PER_UM = _STACK["M3"].sheet_resistance_ohm / (CONTACT_SIZE / 1000)
POLY_OHM_PER_UM = _STACK["Gate"].sheet_resistance_ohm / (GATE_WIDTH / 1000)
#: Only to rank the rows by wordline delay: the BROM model's gate load.
GATE_FF_PER_FIN_ESTIMATE = 0.0417
DEFAULT_BITS = 64
DRIVERS = ("ideal", "default", "sized")


def strap_stack_ohm(fins: int = 2) -> float:
    """M3 to poly at a strap row: V2, V1, V0, the LIG run and the gate link.

    The V0/V1/V2 stack is drawn at one X, so M1 and M2 add only their via
    landings; the LIG runs ``HALF_LIG * 2`` wide from the gate's centre to the
    V0's (:func:`~chipforge_asap7.devices.rom_bitcell.build_rom_strap`).
    """
    run_nm = abs(RomBitcellSpec(fins=fins).gate_xs["WLa"] - STRAP_WL_XS[0])
    lig = _STACK["Local"].sheet_resistance_ohm * run_nm / (2 * HALF_LIG)
    return (
        _STACK["Gate"].via_resistance_ohm  # gate to LIG link
        + lig
        + _STACK["Local"].via_resistance_ohm  # V0
        + _STACK["M1"].via_resistance_ohm  # V1
        + _STACK["M2"].via_resistance_ohm  # V2
    )


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
    #: Bit rows the wordline crosses, and the strap interval; ``wordline_rc``
    #: False drives the selected gate directly with the ideal edge.
    bits: int = DEFAULT_BITS
    strap_every: int = 16
    wordline_rc: bool = True
    #: What drives the wordline: ``"ideal"`` (the edge itself), ``"default"``
    #: (the drawn decoder's NAND and inverter), ``"sized"`` (as
    #: `size_rom_decoder` sizes them for this array), or explicit total fins
    #: ``(nand_n, nand_p, inverter_n, inverter_p)``.
    driver: str | tuple[int, int, int, int] = "default"
    #: With a decoder driver, also simulate the predecoders from the address
    #: pins (`predecode_network`), sized with the same decoder; else the PA
    #: line is an ideal edge and PB is already high.
    predecode: bool = True
    #: What drives each address pin, with the predecoders: ``"ideal"`` (the
    #: edge itself) or a released buffer from `ADDRESS_BUFFERS`.
    address_driver: str = "BUFx4"
    #: Couple each PA/PB/address M4 rail to its neighbouring tracks, and
    #: simulate the previous address's lines as the opposing aggressors.
    rail_coupling: bool = True

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
        for name in ("precharge_fins", "bits", "strap_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        from .rom_predecode_spice import ADDRESS_BUFFERS

        if (
            self.address_driver != "ideal"
            and self.address_driver not in ADDRESS_BUFFERS
        ):
            raise ValueError(
                f"address_driver must be 'ideal' or one of {sorted(ADDRESS_BUFFERS)}"
            )
        for name in ("wordline_rc", "predecode", "rail_coupling"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if isinstance(self.driver, str):
            if self.driver not in DRIVERS:
                raise ValueError(f"driver must be one of {DRIVERS} or four fin counts")
        elif len(self.driver) != 4 or any(
            isinstance(f, bool) or not isinstance(f, int) or f < 1 for f in self.driver
        ):
            raise ValueError("an explicit driver is four positive total fin counts")
        else:
            object.__setattr__(self, "driver", tuple(self.driver))


@dataclass(frozen=True)
class RomReadResult:
    """One bitline depth at one fin count; times in seconds, levels in volts."""

    fins: int
    words: int
    bits: int
    wordline_row: int | None
    driver_fins: tuple[int, int, int, int] | None
    #: Address pins to the word's PA and PB NAND inputs, and the later one.
    t_pa_s: float | None
    t_pb_s: float | None
    t_predecode_s: float | None
    #: The later NAND input to the decoder output.
    t_driver_s: float | None
    t_wordline_s: float | None
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


@dataclass(frozen=True)
class WordlineNetwork:
    """The drawn wordline as SPICE: M3 strap, strap stacks, poly, gate loads."""

    lines: tuple[str, ...]
    selected_row: int
    selected_gate: str
    rows: int
    straps: int


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Dense Gaussian elimination with partial pivoting (no numpy needed)."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            if factor:
                for k in range(col, n + 1):
                    a[r][k] -= factor * a[col][k]
    x = [0.0] * n
    for r in reversed(range(n)):
        x[r] = (a[r][n] - sum(a[r][k] * x[k] for k in range(r + 1, n))) / a[r][r]
    return x


def wordline_network(fins: int, c: RomReadConditions, nmos: str) -> WordlineNetwork:
    """One wordline of a ``c.bits``-row array, driven from the decoder above.

    Geometry is `RomArraySpec`'s: the M3 strap runs the array's height from
    the decoder at the top edge; each strap row drops a via stack onto the
    continuous gate poly; every bit row hangs its word transistor's gate on
    the poly at the row's centre.  Wire capacitance uses ``wire_ff_per_um``.

    The read uses the row with the largest first moment of the mesh's step
    response (Elmore delay generalised to a mesh), ranked with an estimated
    gate load; the transient then uses the real BSIM-CMG gates.
    """
    spec = RomArraySpec(
        words=2,
        bits=c.bits,
        bitcell=RomBitcellSpec(fins=fins),
        strap_every=c.strap_every,
    )
    half_row = spec.bitcell.height / 2
    taps = [y + STRAP_WL_Y for y in spec.strap_ys]
    poly = sorted(
        [(y, f"WT{k}") for k, y in enumerate(taps)]
        + [(spec.row_y(r) + half_row, f"WR{r}") for r in range(c.bits)]
    )
    resistors: list[tuple[str, str, float]] = []
    caps: dict[str, float] = {}

    def wire(a: str, b: str, length_nm: float, ohm_per_um: float) -> None:
        um = length_nm / 1000
        resistors.append((a, b, um * ohm_per_um))
        for n in (a, b):
            caps[n] = caps.get(n, 0.0) + um * c.wire_ff_per_um / 2

    # M3: decoder (WL, the source) at the top edge, down past every strap row.
    m3 = [("WL", spec.height)] + [
        (f"WM{k}", y) for k, y in reversed(list(enumerate(taps)))
    ]
    for (a, ya), (b, yb) in pairwise(m3):
        wire(a, b, ya - yb, M3_OHM_PER_UM)
    stack = strap_stack_ohm(fins)
    for k in range(len(taps)):
        resistors.append((f"WM{k}", f"WT{k}", stack))
    for (ya, a), (yb, b) in pairwise(poly):
        wire(a, b, yb - ya, POLY_OHM_PER_UM)

    # Rank the rows: first moments m = G^-1 C 1, the source node grounded.
    nodes = sorted({n for r in resistors for n in r[:2]} - {"WL"})
    index = {n: i for i, n in enumerate(nodes)}
    g = [[0.0] * len(nodes) for _ in nodes]
    for a, b, ohm in resistors:
        for x, y in ((a, b), (b, a)):
            if x in index:
                g[index[x]][index[x]] += 1 / ohm
                if y in index:
                    g[index[x]][index[y]] -= 1 / ohm
    load = [
        caps.get(n, 0.0)
        + (fins * GATE_FF_PER_FIN_ESTIMATE if n.startswith("WR") else 0)
        for n in nodes
    ]
    moment = _solve(g, load)
    selected = max(range(c.bits), key=lambda r: (moment[index[f"WR{r}"]], -r))

    lines = [f"RW{i} {a} {b} {ohm:g}" for i, (a, b, ohm) in enumerate(resistors)]
    lines += [f"CW_{n} {n} 0 {ff:g}f" for n, ff in sorted(caps.items()) if ff]
    # Every other row's word transistor on this wordline loads the poly.
    lines += [
        f"NG{r} 0 WR{r} 0 0 {nmos} L=20n NFIN={fins}"
        for r in range(c.bits)
        if r != selected
    ]
    return WordlineNetwork(tuple(lines), selected, f"WR{selected}", c.bits, len(taps))


@lru_cache(maxsize=64)
def decoder_spec(fins: int, words: int, c: RomReadConditions) -> Any:
    """The decoder of a ROM whose lane halves are `words`-word bitlines.

    A ``2 * words``-word, ``c.bits``-row array with the center lane: word 0
    is the far end of its half's bitline and the furthest word from the
    predecoders in the lane.  ``"sized"`` is `size_rom_decoder`'s, widening
    the interleave when its cells do not fit; otherwise the drawn defaults.
    """
    from .rom_decoder import RomDecoderSpec
    from .rom_sizing import size_rom_decoder

    if words < 8 or words & (words - 1):
        raise ValueError("the predecoded bench needs a power-of-two words >= 8")
    array = RomArraySpec(
        words=2 * words,
        bits=c.bits,
        bitcell=RomBitcellSpec(fins=fins, vt=c.vt),
        strap_every=c.strap_every,
    )
    if c.driver != "sized":
        return RomDecoderSpec(array=array)
    errors = []
    for interleave in (8, 16, 32):
        if 2 * words < 2 * interleave:
            break
        try:
            return size_rom_decoder(
                RomDecoderSpec(array=array, interleave=interleave)
            ).spec
        except ValueError as exc:
            errors.append(f"interleave {interleave}: {exc}")
    raise ValueError("no interleave fits the sized decoder: " + "; ".join(errors))


def _predecoded(c: RomReadConditions) -> bool:
    return c.predecode and c.driver != "ideal"


def decoder_driver(
    fins: int, words: int, c: RomReadConditions
) -> tuple[int, int, int, int] | None:
    """Total fins of the wordline's NAND and inverter, or None for ideal.

    ``"default"`` is `RomDecoderSpec`'s drawn decoder; ``"sized"`` asks
    `size_rom_decoder` for a ``c.bits``-row array, widening the interleave
    when the sized cells do not fit its slots.  Totals are fins per finger times
    fingers: each finger is a parallel device.
    """
    if c.driver == "ideal":
        return None
    if not isinstance(c.driver, str):
        return c.driver
    from .rom_decoder import RomDecoderSpec
    from .rom_sizing import size_rom_decoder

    def totals(spec) -> tuple[int, int, int, int]:
        (nn, np_), (dn, dp) = spec.cell_fins[:2]
        return (
            nn * spec.nand_fingers,
            np_ * spec.nand_fingers,
            dn * spec.driver_fingers,
            dp * spec.driver_fingers,
        )

    if _predecoded(c):
        return totals(decoder_spec(fins, words, c))
    bitcell = RomBitcellSpec(fins=fins, vt=c.vt)
    if c.driver == "default":
        return totals(RomDecoderSpec(array=RomArraySpec(bitcell=bitcell)))
    errors = []
    for interleave in (8, 16, 32):
        # The wordline driver's load is the bits and their fins, not the
        # word count, so the smallest decoder this interleave takes sizes it
        # (and sizing a wide one re-routes it on every iteration).
        array = RomArraySpec(
            words=2 * interleave,
            bits=c.bits,
            bitcell=bitcell,
            strap_every=c.strap_every,
        )
        try:
            spec = RomDecoderSpec(array=array, interleave=interleave)
            return totals(size_rom_decoder(spec).spec)
        except ValueError as exc:
            errors.append(f"interleave {interleave}: {exc}")
    raise ValueError("no interleave fits the sized driver: " + "; ".join(errors))


def _driver_lines(
    totals: tuple[int, int, int, int],
    nmos: str,
    pmos: str,
    pa: str = "PA",
    pb: str = "PB",
) -> list[str]:
    """The word's postdecode NAND on `pa` and `pb`, feeding the inverter."""
    nn, np_, dn, dp = totals
    return [
        f"ND1 nandout {pa} nmid 0 {nmos} L=20n NFIN={nn}",
        f"ND2 nmid {pb} 0 0 {nmos} L=20n NFIN={nn}",
        f"ND3 nandout {pa} VDD VDD {pmos} L=20n NFIN={np_}",
        f"ND4 nandout {pb} VDD VDD {pmos} L=20n NFIN={np_}",
        f"ND5 WL nandout 0 0 {nmos} L=20n NFIN={dn}",
        f"ND6 WL nandout VDD VDD {pmos} L=20n NFIN={dp}",
    ]


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
    totals = decoder_driver(fins, words, c)
    edge = "RISE"
    rise = f"PWL(0 0 1.1n 0 {1.1 + sl:g}n {vdd:g})"
    fall = f"PWL(0 {vdd:g} 1.1n {vdd:g} {1.1 + sl:g}n 0)"
    pa_pin, pb_pin = "PA", "PB"
    if totals is None:
        start = "WL"
        source = f"Vwl WL 0 {rise}\n"
    elif _predecoded(c):
        from .rom_predecode_spice import predecode_network

        pre = predecode_network(
            decoder_spec(fins, words, c),
            0,
            nmos=nmos,
            pmos=pmos,
            wire_ff_per_um=c.wire_ff_per_um,
            address_buffer=None if c.address_driver == "ideal" else c.address_driver,
            coupling=c.rail_coupling,
        )
        pa_pin, pb_pin = pre.pa_pin, pre.pb_pin
        # Every address bit switches; times start at the first pin's edge.
        start, rises = pre.sources[0]
        edge = "RISE" if rises else "FALL"
        source = (
            "".join(
                f"Va{i} {node} 0 {rise if up else fall}\n"
                for i, (node, up) in enumerate(pre.sources)
            )
            + "\n".join(pre.lines)
            + "\n"
            + "\n".join(_driver_lines(totals, nmos, pmos, pa_pin, pb_pin))
            + "\n"
        )
    else:
        # The predecoded PA line rises with the edge; PB is already high.
        start = "PA"
        source = (
            f"Vpa PA 0 PWL(0 0 1.1n 0 {1.1 + sl:g}n {vdd:g})\n"
            f"Vpb PB 0 {vdd:g}\n" + "\n".join(_driver_lines(totals, nmos, pmos)) + "\n"
        )
    if c.wordline_rc:
        network = wordline_network(fins, c, nmos)
        gate, wordline = network.selected_gate, "\n".join(network.lines) + "\n"
    else:
        gate, wordline = "WL", ""
    return (
        f"* NOR-ROM read, {fins}-fin words, {words} words on each bitline\n"
        f'.include "{models.resolve()}"\n.temp {c.temperature_c:g}\n'
        f"Vsupply VDD 0 {vdd:g}\n"
        # Precharge low until 1 ns, the wordline up 0.1 ns after it releases.
        f"Vprechn PRECHN 0 PWL(0 0 1n 0 {1 + sl:g}n {vdd:g})\n"
        + source
        + f"NPP BLP PRECHN VDD VDD {pmos} L=20n NFIN={c.precharge_fins}\n"
        f"NPU BLU PRECHN VDD VDD {pmos} L=20n NFIN={c.precharge_fins}\n"
        # The selected word is the far one, behind the whole line: programmed
        # on BLP; on BLU its drain floats, so it is not on the line at all.
        f"NPsel {node('P', far)} {gate} 0 0 {nmos} L=20n NFIN={fins}\n"
        + "\n".join(ladder)
        + "\n"
        + wordline
        # Gear: the decoder's femtosecond stub RCs on strong drivers make the
        # trapezoidal rule ring, and ngspice then crawls (a BUFx4 deck ran an
        # hour); Gear agrees with it within 0.1 ps where both finish.
        + ".options method=gear\n"
        + f".tran {c.step_ps:g}p {end:g}n\n"
        f".measure tran blp_pre FIND v(BLP) AT=1.1n\n"
        f".measure tran blu_pre FIND v(BLU) AT=1.1n\n"
        f".measure tran t_sense TRIG v({start}) VAL={half:g} {edge}=1 "
        f"TARG v(BLP) VAL={sense:g} FALL=1\n"
        f".measure tran t_half TRIG v({start}) VAL={half:g} {edge}=1 "
        f"TARG v(BLP) VAL={half:g} FALL=1\n"
        f".measure tran blu_sense FIND v(BLU) WHEN v(BLP)={sense:g} FALL=1\n"
        f".measure tran blu_half FIND v(BLU) WHEN v(BLP)={half:g} FALL=1\n"
        f".measure tran blu_end FIND v(BLU) AT={end:g}n\n"
        f".measure tran t_decoded TRIG v({start}) VAL={half:g} {edge}=1 "
        f"TARG v(WL) VAL={half:g} RISE=1\n"
        f".measure tran t_pa TRIG v({start}) VAL={half:g} {edge}=1 "
        f"TARG v({pa_pin}) VAL={half:g} RISE=1\n"
        f".measure tran t_pb TRIG v({start}) VAL={half:g} {edge}=1 "
        f"TARG v({pb_pin}) VAL={half:g} RISE=1\n"
        f".measure tran t_wordline TRIG v(WL) VAL={half:g} RISE=1 "
        f"TARG v({gate}) VAL={half:g} RISE=1\n" + _control(osdi)
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
            t_pre = t_driver = None
            arrivals = [None, None]
            if c.driver != "ideal":
                decoded = _measure(log, "t_decoded")
                arrivals = [_measure(log, n) for n in ("t_pa", "t_pb")]
                if _predecoded(c):
                    if None not in arrivals:
                        t_pre = max(arrivals)
                elif arrivals[0] is not None:
                    t_pre = 0.0  # PA is the ideal edge itself
                if decoded is not None and t_pre is not None:
                    t_driver = decoded - t_pre
            results.append(
                RomReadResult(
                    fins=f,
                    words=depth,
                    bits=c.bits,
                    wordline_row=(
                        wordline_network(f, c, f"nmos_{c.vt}").selected_row
                        if c.wordline_rc
                        else None
                    ),
                    driver_fins=decoder_driver(f, depth, c),
                    t_pa_s=arrivals[0] if _predecoded(c) else None,
                    t_pb_s=arrivals[1] if _predecoded(c) else None,
                    t_predecode_s=t_pre,
                    t_driver_s=t_driver,
                    t_wordline_s=_measure(log, "t_wordline") if c.wordline_rc else None,
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
                "scope": "distributed M2 bitline RC with the selected word at its far end; M3/strap/poly wordline RC with BSIM gate loads at the slowest row; one model card and corner",
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
    parser.add_argument("--bits", type=int, default=DEFAULT_BITS)
    parser.add_argument("--strap-every", type=int, default=16)
    parser.add_argument(
        "--no-wordline-rc",
        action="store_true",
        help="drive the selected gate directly with the ideal edge",
    )
    parser.add_argument(
        "--driver",
        choices=DRIVERS,
        default="default",
        help="wordline driver: ideal edge, drawn decoder, or size_rom_decoder",
    )
    parser.add_argument(
        "--no-predecode",
        action="store_true",
        help="drive PA with the ideal edge instead of the predecoders",
    )
    parser.add_argument(
        "--address-driver",
        default="BUFx4",
        help="what drives each address pin: 'ideal' or a released BUFx* cell",
    )
    parser.add_argument(
        "--no-rail-coupling",
        action="store_true",
        help="ground the predecode rails' capacitance instead of coupling them",
    )
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
            bits=args.bits,
            strap_every=args.strap_every,
            wordline_rc=not args.no_wordline_rc,
            driver=args.driver,
            predecode=not args.no_predecode,
            address_driver=args.address_driver,
            rail_coupling=not args.no_rail_coupling,
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
