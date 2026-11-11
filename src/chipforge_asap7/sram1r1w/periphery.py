"""The 1R1W macro's clocking, control and decode, and its netlist.

One ``CLK``.  Each port starts a self-timed cycle on CLK's rising edge if its
enable was low there, and a read and a write may share a cycle::

    WEB, REB   latched while CLK is low; held while it is high
    CLKW       = CLK . !WEBL     CLKR = CLK . !REBL
    write      RomControlSpec(edge) on CLKW, READY tied high
    read       RomControlSpec(edge) on CLKR, READY from a replica bitline

Each controller is the NOR ROM's edge-triggered one
(`chipforge_asap7.devices.rom_control`).  CLK's rise latches a request, and
three phases rise in order and fall in reverse once ``DONE`` drops it:

====================  ==========================  ==============================
controller output     write port                  read port
====================  ==========================  ==============================
``PRECHN``            ``wprechn``: precharge off  ``rprechn``: precharge off
``EVAL``              ``wrena``: driver on        ``REN``: read wordline enable
``SAMPLE``            ``WEN``: wordline enable    ``sae``: sense, once ``READY``
====================  ==========================  ==============================

So a write turns its wordline off before its driver, and its driver off
before the precharge.  The write's ``READY`` is tied high: its wordline is
open for its ``done_delay`` inverters (`MacroSpec.write_delay`), then a few
gates more.  The read's ``READY`` is the replica bitline: `replica_cells`
cells hold 0 on one bitline loaded like a real one, on a replica wordline
driven like a real one from ``REN``.  It discharges `replica_cells` times
faster than a real bitline, so ``sae`` fires once a real bitline has fallen
about ``VDD / (2 * replica_cells)``.  ``sae`` stays high for
`MacroSpec.read_delay` inverters.  The read IO's output latch takes the word
then and holds it until the next read's ``sae``.

Each port's address latches, and the write port's ``DIN`` latches, are the
NOR ROM's (`chipforge_asap7.devices.rom_latch`): open while the port's
``PRECHN`` is low, closed from just after CLK's rise to the cycle's end.
Under them, each port has:

* a column select, ``ysel[k]`` with complement ``yseln[k]``, from the low two
  address bits;
* a wordline decoder of `DriverSliceSpec` slices sized by `size_decoder`,
  giving ``WL[4s + j] = SEL[s] . B[j]``:
  * ``B[j]`` decodes wordline-index bits 0 and 1;
  * ``SEL[s] = EN . PS[s]``, with ``PS[s]`` the one-hot decode of the
    remaining bits;
  * ``EN`` is ``WEN`` or ``REN``.

The netlist is flat outside the columns: one `Transistor` per device on the
macro's nets.  Each bit's column is an instance of `column_netlist`.

Address bit 0 and 1 choose the mux row; the rest choose the wordline slot.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property

from ..devices.driver_slice import DriverSliceSpec
from ..devices.inverter import InverterSpec
from ..devices.nand import NandSpec
from ..devices.rom_control import RomControlSpec
from ..devices.rom_latch import RomAddressLatchRowSpec
from ..devices.rom_netlist import Transistor, _rows
from ..devices.sizing import LogicalEffortModel, size_decoder
from .array import MUX_ROWS
from .cells import CellLibrary
from .column import ColumnSpec, column_netlist, column_pins

__all__ = ["MacroSpec", "macro_cdl", "macro_devices", "macro_pins"]

#: One column's height, um: a wordline crosses this much of each bit.
COLUMN_UM = MUX_ROWS * 0.594
#: Small logic: every gate that is not a wordline slice.
LOGIC_FINS = (4, 4)
#: Column-select and SEL buffers.
BUFFER_FINS = (8, 8)
#: What one 8T cell presents to its wordline, with its 0.594 um of wire: the
#: figure OpenFinRAM sizes its slices by (``generate_asap7_8t_wl_slices.py``).
WORDLINE_FF_PER_CELL = 0.181


@dataclass(frozen=True)
class MacroSpec:
    """A ``wordlines * 4`` word by `bits` macro.

    Args:
        wordlines: wordline slots; a power of two, at least 4.
        bits: data bits, one column each.
        write_delay: inverters (odd) from the write wordline's rise to its
            controller's ``DONE``: the write pulse.
        read_delay: inverters (odd) that ``sae`` stays high.
        replica_cells: cells discharging the replica bitline.
        tap_pitch: the columns' (`ColumnSpec`).
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
        if w < 4 or w & (w - 1):
            raise ValueError(f"wordlines must be a power of two of at least 4, got {w}")
        if self.bits < 1:
            raise ValueError(f"bits must be positive, got {self.bits}")
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
    def slices(self) -> int:
        return self.wordlines // 4

    @property
    def column(self) -> ColumnSpec:
        return ColumnSpec(self.wordlines, self.tap_pitch)

    @property
    def name(self) -> str:
        return f"sram1r1w_{self.words}x{self.bits}"

    @property
    def wordline_load_ff(self) -> float:
        """What a wordline driver sees: every cell along it, each with its wire."""
        return WORDLINE_FF_PER_CELL * self.bits * MUX_ROWS

    @property
    def controllers(self) -> dict[str, RomControlSpec]:
        """Each port's controller row: the NOR ROM's, edge-triggered."""
        return {
            port: RomControlSpec(
                fingers=self.control_fingers, edge=True, done_delay=delay
            )
            for port, delay in (("write", self.write_delay), ("read", self.read_delay))
        }

    @property
    def latch_rows(self) -> dict[str, RomAddressLatchRowSpec]:
        """The enables' row (on CLK), the write port's (address and DIN), the read port's."""
        a = self.address_bits
        return {
            name: RomAddressLatchRowSpec(latches=n)
            for name, n in (("enable", 2), ("write", a + self.bits), ("read", a))
        }

    @cached_property
    def slice(self) -> DriverSliceSpec:
        return DriverSliceSpec.from_sizing(
            size_decoder(self.wordline_load_ff, self.wordlines), max_active_fins=12
        )


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


# ── Gates ─────────────────────────────────────────────────────────────────────
def _t(
    name: str,
    d: str,
    g: str,
    s: str,
    flavor: str,
    fins: int,
    series: Iterable[str] = (),
) -> Transistor:
    bulk = "VSS" if flavor == "n" else "VDD"
    model = "nmos_rvt" if flavor == "n" else "pmos_rvt"
    return Transistor(name, d, g, s, bulk, model, 20, fins, frozenset(series))


def _inv(a: str, y: str, tag: str, fins=LOGIC_FINS) -> list[Transistor]:
    n, p = fins
    return [_t(f"M{tag}N", y, a, "VSS", "n", n), _t(f"M{tag}P", y, a, "VDD", "p", p)]


def _nand2(a: str, b: str, y: str, tag: str, fins=LOGIC_FINS) -> list[Transistor]:
    n, p = fins
    x = f"{tag}X"
    return [
        _t(f"M{tag}NA", y, a, x, "n", n, {x}),
        _t(f"M{tag}NB", x, b, "VSS", "n", n, {x}),
        _t(f"M{tag}PA", y, a, "VDD", "p", p),
        _t(f"M{tag}PB", y, b, "VDD", "p", p),
    ]


def _nor2(a: str, b: str, y: str, tag: str, fins=LOGIC_FINS) -> list[Transistor]:
    n, p = fins
    x = f"{tag}X"
    return [
        _t(f"M{tag}PA", x, a, "VDD", "p", p, {x}),
        _t(f"M{tag}PB", y, b, x, "p", p, {x}),
        _t(f"M{tag}NA", y, a, "VSS", "n", n),
        _t(f"M{tag}NB", y, b, "VSS", "n", n),
    ]


def _and2(a: str, b: str, y: str, tag: str, fins=LOGIC_FINS) -> list[Transistor]:
    return _nand2(a, b, f"{tag}N", f"{tag}A", LOGIC_FINS) + _inv(
        f"{tag}N", y, f"{tag}I", fins
    )


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
            out.append(
                Transistor(
                    f"M{tag}D{index}_{k}",
                    y,
                    a,
                    rail,
                    rail,
                    model.model,
                    model.gate_length,
                    band.fins,
                )  # fmt: skip
            )
    return out


def _onehot(
    bits: list[tuple[str, str]], tag: str
) -> tuple[list[str], list[Transistor]]:
    """One line per value of `bits` (``(true, complement)`` literals, LSB first)."""
    if len(bits) == 1:
        return [bits[0][1], bits[0][0]], []
    if len(bits) == 2:
        lines, devices = [], []
        for v in range(4):
            a = bits[0][0 if v & 1 else 1]
            b = bits[1][0 if v & 2 else 1]
            out = f"{tag}{v}"
            devices += _and2(a, b, out, f"{tag}{v}_")
            lines.append(out)
        return lines, devices
    half = len(bits) // 2
    low, dl = _onehot(bits[:half], f"{tag}L")
    high, dh = _onehot(bits[half:], f"{tag}H")
    lines, devices = [], dl + dh
    for h, hi in enumerate(high):
        for lo_i, lo in enumerate(low):
            out = f"{tag}{h * len(low) + lo_i}"
            devices += _and2(lo, hi, out, f"{tag}{h}_{lo_i}_")
            lines.append(out)
    return lines, devices


# ── Blocks ────────────────────────────────────────────────────────────────────
def _latch_row(
    row: RomAddressLatchRowSpec, clock: str, rename: dict[str, str], tag: str
) -> list[Transistor]:
    nets = {n for d in row.devices for n in d[1:4]}
    local = {n for n in nets if n == "CK" or n[0] in "XY" and n[1:].isdigit()}
    return _rows(row.devices, row.bands, tag, {"PRECHN": clock, **rename}, local, set())


def _controller(spec: MacroSpec, port: str) -> list[Transistor]:
    read = port == "R"
    row = spec.controllers["read" if read else "write"]
    rename = {
        "CLK": f"CLK{port}",
        "RESETN": "VDD",
        "READY": "RREADY" if read else "VDD",
        "PRECHN": "rprechn" if read else "wprechn",
        "EVAL": "REN" if read else "wrena",
        "SAMPLE": "sae" if read else "WEN",
    }
    nets = {n for d in row.devices for n in d[1:4]}
    local = nets - set(rename) - {"VDD", "VSS"}
    return _rows(
        row.devices, row.bands, f"C{port}_", rename, local, set(row.series_nodes)
    )


def _decoder(
    spec: MacroSpec, port: str, enable: str, wordline: str, select: str
) -> list[Transistor]:
    """Column select and wordlines of one port, from its latched address ``{port}AL<k>``."""
    lits = [(f"{port}AL<{k}>", f"{port}ALN<{k}>") for k in range(spec.address_bits)]
    out: list[Transistor] = []
    for k, (true, comp) in enumerate(lits):
        out += _inv(true, comp, f"{port}AI{k}_")
    # Column select: one-hot of bits 0 and 1, buffered, and its complement.
    rows, devices = _onehot(lits[:2], f"{port}YS")
    out += devices
    for k, line in enumerate(rows):
        # YB = !line; ysel = !YB = line; YT = !YB; yseln = !YT = !line.
        out += _inv(line, f"{port}YB{k}", f"{port}YB{k}_", BUFFER_FINS)
        out += _inv(f"{port}YB{k}", f"{select}[{k}]", f"{port}YP{k}_", BUFFER_FINS)
        out += _inv(f"{port}YB{k}", f"{port}YT{k}", f"{port}YT{k}_", BUFFER_FINS)
        out += _inv(f"{port}YT{k}", f"{select}n[{k}]", f"{port}YN{k}_", BUFFER_FINS)
    # Wordlines: B[j] from index bits 0-1, SEL[s] = EN . PS[s] from the rest.
    index = lits[2:]
    b_lines, devices = _onehot(index[:2], f"{port}PB")
    out += devices
    upper = index[2:]
    if upper:
        ps, devices = _onehot(upper, f"{port}PS")
        out += devices
    else:
        ps = ["VDD"]
    slice_ = spec.slice
    for s, line in enumerate(ps):
        sel = f"{port}SEL{s}"
        if line == "VDD":
            out += _inv(enable, f"{sel}B", f"{sel}B_") + _inv(
                f"{sel}B", sel, f"{sel}_", BUFFER_FINS
            )
        else:
            out += _and2(enable, line, sel, f"{sel}_", BUFFER_FINS)
        for j, b in enumerate(b_lines):
            i = 4 * s + j
            node = f"{port}WN{i}"
            out += _slice_nand(slice_.nand, sel, b, node, f"{port}WS{i}_")
            out += _slice_driver(
                slice_.inverter, node, f"{wordline}[{i}]", f"{port}WD{i}_"
            )
    return out


def _replica(spec: MacroSpec) -> list[Transistor]:
    """A read bitline loaded as a real one, `replica_cells` cells reading 0 onto it."""
    slice_ = spec.slice
    out = _slice_nand(slice_.nand, "REN", "VDD", "RPWN", "RPS_")
    out += _slice_driver(slice_.inverter, "RPWN", "RWLR", "RPD_")
    cells = spec.bits * MUX_ROWS
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
        _t("MRPMN", "RBLR", "VSS", "RPMX", "n", 6),
        _t("MRPMP", "RPMX", "VDD", "RBLR", "p", 6),
    ]
    out += _inv("RBLR", "RREADY", "RPR_")
    return out


def macro_devices(spec: MacroSpec) -> list[Transistor]:
    """Every transistor outside the columns."""
    rows = spec.latch_rows
    out = _latch_row(
        rows["enable"],
        "CLK",
        {"D<0>": "WEB", "D<1>": "REB", "Q<0>": "WEBL", "Q<1>": "REBL"},
        "EB_",
    )
    out += _inv("CLK", "CLKBG", "CKI_")
    out += _nor2("CLKBG", "WEBL", "CLKW", "CKW_") + _nor2(
        "CLKBG", "REBL", "CLKR", "CKR_"
    )
    out += _controller(spec, "W") + _controller(spec, "R")
    out += _inv("wrena", "wrenan", "WRN_", BUFFER_FINS)
    a, b = spec.address_bits, spec.bits
    write_latches = {f"D<{k}>": f"WADDR<{k}>" for k in range(a)}
    write_latches |= {f"Q<{k}>": f"WAL<{k}>" for k in range(a)}
    write_latches |= {f"D<{a + i}>": f"DIN<{i}>" for i in range(b)}
    write_latches |= {f"Q<{a + i}>": f"DL<{i}>" for i in range(b)}
    out += _latch_row(rows["write"], "wprechn", write_latches, "WL_")
    read_latches = {f"D<{k}>": f"RADDR<{k}>" for k in range(a)}
    read_latches |= {f"Q<{k}>": f"RAL<{k}>" for k in range(a)}
    out += _latch_row(rows["read"], "rprechn", read_latches, "RL_")
    out += _decoder(spec, "W", "WEN", "WWL", "wysel")
    out += _decoder(spec, "R", "REN", "RWL", "rysel")
    out += _replica(spec)
    return out


def macro_cdl(spec: MacroSpec, cells: CellLibrary) -> str:
    """The macro: the column subcircuit, then the periphery flat and one column per bit."""
    column = spec.column
    lines = [column_netlist(cells, column).replace("\n.END\n", "\n")]
    devices = macro_devices(spec)
    lines.append(f".SUBCKT {spec.name} {' '.join(macro_pins(spec))}")
    lines += [d.cdl() for d in devices]
    shared = {"oe": "VDD", "oeb": "VSS"}
    for i in range(spec.bits):
        nets = {**shared, "D": f"DL<{i}>", "Q": f"DOUT<{i}>"}
        lines.append(
            f"Xbit{i} {' '.join(nets.get(p, p) for p in column_pins(column))} {column.name}"
        )
    lines += [f".ENDS {spec.name}", ".END", ""]
    return "\n".join(lines)
