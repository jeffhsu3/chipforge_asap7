"""The 1R1W macro's clocking, control and decode, and its netlist.

One ``CLK``.  Each port starts a self-timed access on CLK's rising edge if its
enable was low there, and a read and a write may share a cycle.

The macro is two mirrored half words, as GT2N's rd8t macro is: bits
``[0, h)`` above a centre band and ``[h, bits)`` below it, each half with its
own wordline drivers (`chipforge_asap7.sram1r1w.strips`) on its own
wordlines, ``WWLU``/``RWLU`` and ``WWLL``/``RWLL``.  Everything else is in the
centre band, once for both halves.

Per port, the band holds four rows:

* **enable latch** (`RomAddressLatchRowSpec`, one latch on ``CLK``): ``WEB``
  or ``REB``, held while CLK is high.  Its complement node is the port's
  active-high enable ``WE`` or ``RE``.
* **controller** (`control_row`): the NOR ROM's edge-triggered controller
  (`chipforge_asap7.devices.rom_control`), as a `GateRowSpec`.  CLK's rise
  latches a request; three phases rise in order and fall in reverse once
  ``DONE`` drops it:

  ====================  ===================  =========================
  controller output     write port           read port
  ====================  ===================  =========================
  ``PRECHN``            ``wprechn``          ``rprechn``
  ``EVAL``              ``wrena``            ``REN``: wordline enable
  ``SAMPLE``            ``WEN``: wordline    ``sae``, once ``READY``
  ``DONE`` delay        `write_delay`        `read_delay`
  ====================  ===================  =========================

* **address latches** (`RomAddressLatchRowSpec` on the port's ``PRECHN``),
  and the write port's ``DIN`` latches.  Each latch's ``Y`` node is its
  address bit's complement.
* **select row** (`select_row`): the column select and the ``B`` lines.

and the two ports share one **rank** of ``PS`` gates (`rank_row`), one per
wordline slice, between the two halves' slices.

The read's ``READY`` is a replica bitline: `replica_cells` cells hold 0 on one
bitline loaded like a real one, on a replica wordline driven like a half's
from ``REN``.  It discharges `replica_cells` times faster than a real bitline,
so ``sae`` fires once a real one has fallen about ``VDD / (2 * replica_cells)``.
The read IO's output latch takes the word then.

Decode, from the latched address ``AL<k>`` and its complement ``ALN<k>``:

* ``ysel[k]`` (and ``yseln[k]``): bits 0 and 1, ``NAND2 -> INV -> INV``;
* ``B[j] = EN . a2 . a3``, ``NAND3 -> INV``, with ``EN`` the port's
  ``WEN`` or ``REN``;
* ``PS[s]``: the remaining bits, one AND per slice; past three bits the rank
  ANDs predecoded groups of two or three bits, made in the select row;
* each half's slice: ``WL[4s + j] = PS[s] . B[j]``.

The enable is on the four ``B`` lines rather than on every slice's select, so
the ``PS`` lines are static through an access and only ``B`` carries its edge.

Compared with one controller per port fed through a gated clock, these are
fused:

* ``CLKB = NAND(CLK, WE)``: the clock gate is the controller's first gate;
* ``REQ = NOR(DONE, REQB)``: there is no reset;
* ``wrenan`` is the write controller's ``EVB`` node, and ``READY`` the read
  controller's own inverter;
* the write controller's ``SN`` is a NAND2, its ``READY`` being high.

The netlist is flat outside the columns: one `Transistor` per device on the
macro's nets.  Each bit's column is an instance of `column_netlist`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property

from ..devices.driver_slice import DriverSliceSpec
from ..devices.gate_row import Gate, GateRowSpec
from ..devices.inverter import InverterSpec
from ..devices.nand import NandSpec
from ..devices.rom_latch import RomAddressLatchRowSpec
from ..devices.rom_netlist import Transistor, _rows
from ..devices.sizing import LogicalEffortModel, size_decoder
from .array import MUX_ROWS
from .cells import CellLibrary
from .column import ColumnSpec, column_netlist, column_pins

__all__ = [
    "MacroSpec",
    "control_row",
    "decode_devices",
    "macro_cdl",
    "macro_devices",
    "macro_pins",
    "rank_row",
    "select_row",
]

#: One column's height, um: a wordline crosses this much of each bit.
COLUMN_UM = MUX_ROWS * 0.594
#: What one 8T cell presents to its wordline, with its 0.594 um of wire: the
#: figure OpenFinRAM sizes its slices by (``generate_asap7_8t_wl_slices.py``).
WORDLINE_FF_PER_CELL = 0.181
#: Fins of the gate on each column's select, per bit: the two mux pass gates
#: of a 6-fin transmission gate pair.
SELECT_FINS_PER_BIT = 12
#: A gate-row finger's input: one 4-fin nFET and one 4-fin pFET.
FINGER_FINS = 8
#: Fanout a buffer finger drives.
FANOUT = 4
#: Upper address bits a rank gate takes directly; past it, predecoded groups.
RANK_BITS = 3
PORTS = {"W": "write", "R": "read"}


def _fingers(load_fins: float) -> int:
    return max(1, math.ceil(load_fins / (FANOUT * FINGER_FINS)))


def _taper(fingers: int) -> tuple[int, ...]:
    """An odd chain of inverters ending at `fingers`, each at most `FANOUT`
    times the one before, from one finger."""
    stages = 1
    while fingers ** (1 / stages) > FANOUT:
        stages += 2
    return tuple(max(1, round(fingers ** ((k + 1) / stages))) for k in range(stages))


def _groups(bits: int) -> tuple[int, ...]:
    """Bits per predecoded group: two or three each, as few groups as can be."""
    count = math.ceil(bits / 3)
    base, extra = divmod(bits, count)
    return tuple(base + (g < extra) for g in range(count))


@dataclass(frozen=True)
class MacroSpec:
    """A ``wordlines * 4`` word by `bits` macro, in two half words.

    Args:
        wordlines: wordline slots; a power of two, 4 to 1024.
        bits: data bits, one column each; the upper half takes ``ceil(bits / 2)``.
        write_delay: inverters (odd) from the write wordline's rise to its
            controller's ``DONE``: the write pulse.
        read_delay: inverters (odd) that ``sae`` stays high.
        replica_cells: cells discharging the replica bitline.
        tap_pitch: the columns' (`ColumnSpec`).
        control_fingers: fingers of each controller output driver.
    """

    wordlines: int = 32
    bits: int = 2
    write_delay: int = 21
    read_delay: int = 7
    replica_cells: int = 2
    tap_pitch: int | None = None
    control_fingers: int = 4
    model: LogicalEffortModel = field(default_factory=LogicalEffortModel, repr=False)

    def __post_init__(self) -> None:
        w = self.wordlines
        if not 4 <= w <= 1024 or w & (w - 1):
            raise ValueError(f"wordlines must be a power of two, 4 to 1024, got {w}")
        if self.bits < 2:
            raise ValueError(f"bits must be at least 2, one per half, got {self.bits}")
        for name in ("write_delay", "read_delay"):
            value = getattr(self, name)
            if value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be odd and positive, got {value}")
        if not 1 <= self.replica_cells <= w:
            raise ValueError(f"replica_cells must be 1..{w}, got {self.replica_cells}")

    @property
    def words(self) -> int:
        return self.wordlines * MUX_ROWS

    @property
    def address_bits(self) -> int:
        return self.words.bit_length() - 1

    @property
    def upper_bits(self) -> int:
        """Address bits the ``PS`` lines decode: all but the mux's and ``B``'s."""
        return self.address_bits - 4

    @property
    def slices(self) -> int:
        return self.wordlines // 4

    @property
    def halves(self) -> dict[str, range]:
        """Each half's bits: ``"U"`` above the centre band, ``"L"`` below it."""
        h = (self.bits + 1) // 2
        return {"U": range(h), "L": range(h, self.bits)}

    def half_of(self, bit: int) -> str:
        return "U" if bit in self.halves["U"] else "L"

    @property
    def column(self) -> ColumnSpec:
        return ColumnSpec(self.wordlines, self.tap_pitch)

    @property
    def name(self) -> str:
        return f"sram1r1w_{self.words}x{self.bits}"

    @property
    def wordline_load_ff(self) -> float:
        """What one half's wordline driver sees: every cell along it, with its wire."""
        return WORDLINE_FF_PER_CELL * len(self.halves["U"]) * MUX_ROWS

    @cached_property
    def slice(self) -> DriverSliceSpec:
        """Both halves' slice, sized for the larger half."""
        return DriverSliceSpec.from_sizing(
            size_decoder(self.wordline_load_ff, self.wordlines), max_active_fins=12
        )

    # ── Sizing ────────────────────────────────────────────────────────────────
    @property
    def _slice_input_fins(self) -> int:
        """One input of one slice NAND."""
        nand = self.slice.nand
        return nand.fingers * (nand.n_band.spec.fins + nand.p_band.spec.fins)

    @property
    def select_fingers(self) -> int:
        """Each ``ysel``/``yseln`` buffer: one select of every bit."""
        return _fingers(SELECT_FINS_PER_BIT * self.bits)

    @property
    def b_fingers(self) -> tuple[int, ...]:
        """Each ``B`` line's inverter chain: one NAND input in every slice of
        both halves.  ``B`` carries the access's edge, so it is tapered."""
        return _taper(_fingers(2 * self.slices * self._slice_input_fins))

    @property
    def rank_fingers(self) -> int:
        """Each ``PS`` inverter: one finger, all a slice's pitch leaves beside
        its NAND.  It drives four slice NAND inputs in each half, but its line
        follows the address, not the clock: it has to settle before the
        ``B`` lines rise, so a slow one adds to the address's setup time,
        not to the access."""
        return 1

    # ── Rows ──────────────────────────────────────────────────────────────────
    @property
    def controllers(self) -> dict[str, GateRowSpec]:
        return {
            port: control_row(port, delay, self.control_fingers)
            for port, delay in (("W", self.write_delay), ("R", self.read_delay))
        }

    @property
    def latch_rows(self) -> dict[str, RomAddressLatchRowSpec]:
        """Each port's enable latch, and its address latches (and the write ``DIN``)."""
        a = self.address_bits
        return {
            "W_enable": RomAddressLatchRowSpec(latches=1),
            "R_enable": RomAddressLatchRowSpec(latches=1),
            "W": RomAddressLatchRowSpec(latches=a + self.bits),
            "R": RomAddressLatchRowSpec(latches=a),
        }

    @property
    def select_rows(self) -> dict[str, GateRowSpec]:
        return {port: select_row(self, port) for port in PORTS}

    @property
    def rank_rows(self) -> dict[str, GateRowSpec]:
        return {port: rank_row(self, port) for port in PORTS if self.upper_bits}


def macro_pins(spec: MacroSpec) -> tuple[str, ...]:
    a, b = spec.address_bits, spec.bits
    return (
        "CLK",
        "WEB",
        *(f"WADDR<{k}>" for k in range(a)),
        *(f"DIN<{i}>" for i in range(b)),
        "REB",
        *(f"RADDR<{k}>" for k in range(a)),
        *(f"DOUT<{i}>" for i in range(b)),
        "VDD",
        "VSS",
    )


# ── Rows ──────────────────────────────────────────────────────────────────────
def control_row(port: str, delay: int, fingers: int) -> GateRowSpec:
    """A port's controller: the NOR ROM's edge controller, fused for this macro.

    Inputs ``CLK``, the enable ``EB`` and, reading, the replica bitline
    ``RBL``.  The write controller's ``EVB`` is ``wrenan``: a driver too."""
    read = port == "R"
    chain = [f"DL{i}" for i in range(delay - 1)] + ["DONE"]
    gates = [
        Gate("CLKB", "nand2", ("CLK", "EB")),
        Gate("ARMED", "nor2", ("REQ", "ARMB")),
        Gate("ARMB", "nor2", ("CLKB", "ARMED")),
        Gate("SET", "nor2", ("CLKB", "ARMB")),
        Gate("REQB", "nor2", ("SET", "REQ")),
        Gate("REQ", "nor2", ("DONE", "REQB")),
        Gate("EN", "nor2", ("REQ", "EVAL")),
        Gate("PRECHN", "inv", ("EN",), fingers),
        Gate("PN", "inv", ("PRECHN",)),
        Gate("ERN", "nor2", ("REQ", "SAMPLE")),
        Gate("EV0", "nor2", ("ERN", "PN")),
        Gate("EVB", "inv", ("EV0",), 1 if read else fingers),
        Gate("EVAL", "inv", ("EVB",), fingers),
    ]
    if read:
        gates += [
            Gate("READY", "inv", ("RBL",)),
            Gate("SN", "nand3", ("REQ", "EVAL", "READY")),
        ]
    else:
        gates.append(Gate("SN", "nand2", ("REQ", "EVAL")))
    gates.append(Gate("SAMPLE", "inv", ("SN",), fingers))
    gates += [
        Gate(out, "inv", (src,))
        for src, out in zip(["SN", *chain], chain, strict=False)
    ]
    return GateRowSpec(
        tuple(gates),
        inputs=("CLK", "EB", "RBL") if read else ("CLK", "EB"),
        outputs=("PRECHN", "EVAL", "SAMPLE")
        if read
        else ("PRECHN", "EVAL", "SAMPLE", "EVB"),
        name=f"sram1r1w_control_{PORTS[port]}_d{delay}_f{fingers}",
    )


def _lit(port: str, bit: int, value: int) -> str:
    """Latched address bit `bit` (``value`` 1) or its complement (0)."""
    return f"{port}AL<{bit}>" if value else f"{port}ALN<{bit}>"


def _enable(port: str) -> str:
    return "WEN" if port == "W" else "REN"


def _group_line(port: str, group: int, value: int) -> str:
    return f"{port}G{group}[{value}]"


def select_row(spec: MacroSpec, port: str) -> GateRowSpec:
    """A port's column select, ``B`` lines and, past `RANK_BITS`, the rank's groups.

    The read port's row also drives the replica's ``B`` line, ``RB``, through
    the same gates as a real ``B`` line."""
    select = "wysel" if port == "W" else "rysel"
    gates: list[Gate] = []
    for k in range(MUX_ROWS):
        yb = f"{port}YB{k}"
        gates += [
            Gate(yb, "nand2", (_lit(port, 0, k & 1), _lit(port, 1, k >> 1 & 1))),
            Gate(f"{select}[{k}]", "inv", (yb,), spec.select_fingers),
            Gate(f"{select}n[{k}]", "inv", (f"{select}[{k}]",), spec.select_fingers),
        ]
    en = _enable(port)
    for j in range(4):
        gates.append(
            Gate(
                f"{port}BN{j}",
                "nand3",
                (en, _lit(port, 2, j & 1), _lit(port, 3, j >> 1 & 1)),
            )
        )
        gates += _chain(f"{port}BN{j}", f"{port}B[{j}]", spec.b_fingers)
    outputs = [f"{select}{n}[{k}]" for k in range(MUX_ROWS) for n in ("", "n")]
    outputs += [f"{port}B[{j}]" for j in range(4)]
    if spec.upper_bits > RANK_BITS:
        loads = spec.slices // 2  # rank gates on each group line, at least
        start = 4
        for g, width in enumerate(_groups(spec.upper_bits)):
            for v in range(1 << width):
                lits = tuple(_lit(port, start + b, v >> b & 1) for b in range(width))
                line = _group_line(port, g, v)
                gates += [
                    Gate(f"{line}N", f"nand{width}", lits),
                    Gate(line, "inv", (f"{line}N",), _fingers(loads * FINGER_FINS)),
                ]
                outputs.append(line)
            start += width
    if port == "R":
        gates.append(Gate("RBN", "nand3", ("REN", "REN", "REN")))
        gates += _chain("RBN", "RB", spec.b_fingers)
        outputs.append("RB")
    used = {n for g in gates for n in g.ins} - {g.out for g in gates}
    return GateRowSpec(
        tuple(gates),
        inputs=tuple(sorted(used)),
        outputs=tuple(outputs),
        name=f"sram1r1w_select_{PORTS[port]}_{spec.words}x{spec.bits}",
    )


def _chain(src: str, out: str, fingers: tuple[int, ...]) -> list[Gate]:
    """Inverters from `src` to `out`, one per entry of `fingers`."""
    nets = [src, *(f"{out}_T{k}" for k in range(len(fingers) - 1)), out]
    return [Gate(nets[k + 1], "inv", (nets[k],), f) for k, f in enumerate(fingers)]


def rank_row(spec: MacroSpec, port: str) -> GateRowSpec:
    """A port's ``PS`` gates, one per slice, in slice order."""
    u = spec.upper_bits
    if u == 0:
        raise ValueError("a macro of 4 wordlines has no PS lines")
    gates: list[Gate] = []
    for s in range(spec.slices):
        out = f"{port}PS[{s}]"
        if u == 1:
            gates.append(Gate(out, "inv", (_lit(port, 4, 1 - s),), spec.rank_fingers))
            continue
        if u <= RANK_BITS:
            ins = tuple(_lit(port, 4 + b, s >> b & 1) for b in range(u))
        else:
            ins, shift = [], 0
            for g, width in enumerate(_groups(u)):
                ins.append(_group_line(port, g, s >> shift & ((1 << width) - 1)))
                shift += width
            ins = tuple(ins)
        gates += [
            Gate(f"{port}PSN{s}", f"nand{len(ins)}", ins),
            Gate(out, "inv", (f"{port}PSN{s}",), spec.rank_fingers),
        ]
    used = {n for g in gates for n in g.ins} - {g.out for g in gates}
    return GateRowSpec(
        tuple(gates),
        inputs=tuple(sorted(used)),
        outputs=tuple(f"{port}PS[{s}]" for s in range(spec.slices)),
        name=f"sram1r1w_rank_{PORTS[port]}_{spec.words}x{spec.bits}",
    )


# ── Devices ───────────────────────────────────────────────────────────────────
def _t(name: str, d: str, g: str, s: str, flavor: str, fins: int,
       series: Iterable[str] = ()) -> Transistor:  # fmt: skip
    bulk = "VSS" if flavor == "n" else "VDD"
    model = "nmos_rvt" if flavor == "n" else "pmos_rvt"
    return Transistor(name, d, g, s, bulk, model, 20, fins, frozenset(series))


def _gate_row(
    row: GateRowSpec, tag: str, rename: dict[str, str] | None = None
) -> list[Transistor]:
    rename = rename or {}
    local = set(row.local_nets) - set(rename)
    return _rows(row.devices, row.bands, tag, rename, local, set(row.series_nodes))


def _latch_row(
    row: RomAddressLatchRowSpec, clock: str, rename: dict[str, str], tag: str
) -> list[Transistor]:
    nets = {n for d in row.devices for n in d[1:4]}
    local = {n for n in nets if n == "CK" or n[0] in "XY" and n[1:].isdigit()} - set(
        rename
    )
    return _rows(row.devices, row.bands, tag, {"PRECHN": clock, **rename}, local, set())


def _slice_nand(spec: NandSpec, a: str, b: str, y: str, tag: str) -> list[Transistor]:
    n, p = spec.n_band.spec, spec.p_band.spec
    out = []
    for k in range(spec.fingers):
        x = f"{tag}X{k}"
        out += [
            Transistor(f"M{tag}NA{k}", y, a, x, "VSS", n.model, n.gate_length, n.fins, frozenset({x})),
            Transistor(f"M{tag}NB{k}", x, b, "VSS", "VSS", n.model, n.gate_length, n.fins, frozenset({x})),
            Transistor(f"M{tag}PA{k}", y, a, "VDD", "VDD", p.model, p.gate_length, p.fins),
            Transistor(f"M{tag}PB{k}", y, b, "VDD", "VDD", p.model, p.gate_length, p.fins),
        ]  # fmt: skip
    return out


def _slice_driver(spec: InverterSpec, a: str, y: str, tag: str) -> list[Transistor]:
    out = []
    for index, band in enumerate(spec.bands):
        rail, model = band.rail_net, band.spec
        for k in range(spec.fingers):
            out.append(Transistor(f"M{tag}D{index}_{k}", y, a, rail, rail, model.model,
                                  model.gate_length, band.fins))  # fmt: skip
    return out


def _ps(spec: MacroSpec, port: str, s: int) -> str:
    return f"{port}PS[{s}]" if spec.upper_bits else "VDD"


def _slices(spec: MacroSpec, port: str) -> list[Transistor]:
    """Both halves' slices of a port: ``{WL}{H}[4s + j] = PS[s] . B[j]``."""
    wordline = "WWL" if port == "W" else "RWL"
    out = []
    for half, bits in spec.halves.items():
        if not bits:
            continue
        for s in range(spec.slices):
            for j in range(4):
                i, node = 4 * s + j, f"{port}WN{half}{4 * s + j}"
                out += _slice_nand(spec.slice.nand, _ps(spec, port, s), f"{port}B[{j}]", node,
                                   f"{port}WS{half}{i}_")  # fmt: skip
                out += _slice_driver(
                    spec.slice.inverter,
                    node,
                    f"{wordline}{half}[{i}]",
                    f"{port}WD{half}{i}_",
                )
    return out


def decode_devices(spec: MacroSpec, port: str) -> list[Transistor]:
    """A port's select row, rank and slices: from ``{port}AL<k>``, its
    complements and the enable to the selects and both halves' wordlines."""
    out = _gate_row(spec.select_rows[port], f"{port}S_")
    if spec.upper_bits:
        out += _gate_row(spec.rank_rows[port], f"{port}K_")
    return out + _slices(spec, port)


def _replica(spec: MacroSpec) -> list[Transistor]:
    """A read bitline loaded as a real one, `replica_cells` cells reading 0 onto
    it, on a wordline loaded as one half's and driven from ``RB``."""
    slice_ = spec.slice
    out = _slice_nand(slice_.nand, "RB", "RB", "RPWN", "RPS_")
    out += _slice_driver(slice_.inverter, "RPWN", "RWLR", "RPD_")
    # RB carries one slice NAND; a B line carries one per slice in each half.
    nand = slice_.nand
    for c in range(2 * spec.slices - 1):
        for k in range(nand.fingers):
            out += [
                _t(f"MRPBN{c}_{k}", "VSS", "RB", "VSS", "n", nand.n_band.spec.fins),
                _t(f"MRPBP{c}_{k}", "VDD", "RB", "VDD", "p", nand.p_band.spec.fins),
            ]
    cells = len(spec.halves["U"]) * MUX_ROWS
    for c in range(cells * 2 - spec.replica_cells):
        out.append(
            Transistor(f"MRPL{c}", "VSS", "RWLR", "VSS", "VSS", "nmos_sram", 20, 2)
        )
    for c in range(spec.replica_cells):
        q = f"RPQ{c}"
        out += [
            Transistor(f"MRPA{c}", "RBLR", "RWLR", q, "VSS", "nmos_sram", 20, 2),
            Transistor(f"MRPD{c}", q, "VDD", "VSS", "VSS", "nmos_sram", 20, 2),
        ]
    for c in range(spec.wordlines - spec.replica_cells):
        out.append(
            Transistor(f"MRPO{c}", "RBLR", "VSS", "VDD", "VSS", "nmos_sram", 20, 2)
        )
    out += [
        _t("MRPPRE", "RBLR", "rprechn", "VDD", "p", 6),
        _t("MRPMN", "RBLR", "VSS", "VDD", "n", 6),
        _t("MRPMP", "VDD", "VDD", "RBLR", "p", 6),
    ]
    return out


def macro_devices(spec: MacroSpec) -> list[Transistor]:
    """Every transistor outside the columns."""
    a, b = spec.address_bits, spec.bits
    rows = spec.latch_rows
    out: list[Transistor] = []
    for port, enable in (("W", "WE"), ("R", "RE")):
        out += _latch_row(rows[f"{port}_enable"], "CLK",
                          {"D<0>": f"{port}EB", "Q<0>": f"{port}EBL", "Y0": enable}, f"{port}E_")  # fmt: skip
    out += _gate_row(spec.controllers["W"], "CW_", {
        "CLK": "CLK", "EB": "WE", "PRECHN": "wprechn", "EVAL": "wrena", "SAMPLE": "WEN", "EVB": "wrenan"})  # fmt: skip
    out += _gate_row(spec.controllers["R"], "CR_", {
        "CLK": "CLK", "EB": "RE", "RBL": "RBLR", "PRECHN": "rprechn", "EVAL": "REN", "SAMPLE": "sae"})  # fmt: skip
    write = {f"D<{k}>": f"WADDR<{k}>" for k in range(a)}
    write |= {f"Q<{k}>": f"WAL<{k}>" for k in range(a)} | {
        f"Y{k}": f"WALN<{k}>" for k in range(a)
    }
    write |= {f"D<{a + i}>": f"DIN<{i}>" for i in range(b)}
    write |= {f"Q<{a + i}>": f"DL<{i}>" for i in range(b)}
    out += _latch_row(rows["W"], "wprechn", write, "WL_")
    read = {f"D<{k}>": f"RADDR<{k}>" for k in range(a)}
    read |= {f"Q<{k}>": f"RAL<{k}>" for k in range(a)} | {
        f"Y{k}": f"RALN<{k}>" for k in range(a)
    }
    out += _latch_row(rows["R"], "rprechn", read, "RL_")
    out += decode_devices(spec, "W") + decode_devices(spec, "R")
    return out + _replica(spec)


def macro_cdl(spec: MacroSpec, cells: CellLibrary) -> str:
    """The macro: the column subcircuit, then the periphery flat and one column per bit."""
    column = spec.column
    lines = [column_netlist(cells, column).replace("\n.END\n", "\n")]
    devices = macro_devices(spec)
    lines.append(f".SUBCKT {spec.name} {' '.join(macro_pins(spec))}")
    lines += [d.cdl() for d in devices]
    shared = {"oe": "VDD", "oeb": "VSS"}
    for i in range(spec.bits):
        half = spec.half_of(i)
        nets = {**shared, "D": f"DL<{i}>", "Q": f"DOUT<{i}>"}
        for w in range(spec.wordlines):
            nets[f"WWL[{w}]"] = f"WWL{half}[{w}]"
            nets[f"RWL[{w}]"] = f"RWL{half}[{w}]"
        lines.append(
            f"Xbit{i} {' '.join(nets.get(p, p) for p in column_pins(column))} {column.name}"
        )
    lines += [f".ENDS {spec.name}", ".END", ""]
    return "\n".join(lines)
