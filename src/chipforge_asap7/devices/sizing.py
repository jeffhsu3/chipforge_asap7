"""Logical-effort sizing of the wordline decoder, in fins.

The released decoder is a logical-effort ladder walked back from the
wordline, and nothing about it needs searching.  Per wordline the path is::

    address -> [NANDb -> INV] -> predecode line -> [NAND2 -> INV] -> [NAND2 -> INV62] -> WL
               predecode        PA / PB / PC       slice AND          post-decode AND

with ``WL<i> = (PA . WLENA) . (PB . PC<i>)`` inside each four-wordline slice:
the select ``PA.WLENA`` is shared by the slice's four post-decode NANDs, the
row term ``PB.PC<i>`` is private to one.  Sizing each stage for a target
stage effort ``f`` from its load,

    C_in = g * C_out / f        g = 1 (INV), (m + 1) / 2 (NANDm, equal n/p strength)

and quantising to whole fins reproduces the released cells: a 22.8 fF
wordline at f = 4.43 gives the 62-fin driver and the 14/7-fin NAND of
`dec_inv_62f_halved_AND` and `dec_nand_12f_12f_..._P1N1` exactly.

Everything here is arithmetic; no GDS or SPICE library is needed.  The
technology numbers in `LogicalEffortModel` were measured with Xyce on the
ASAP7 TT BSIM-CMG card at 0.7 V, RVT, 20 nm gates (tapered inverter chains at
h = 1..8, a lumped-capacitor equivalence, and a NAND2 input-load match):

    tau = 1.76 ps, p_inv = 0.59, FO4 = 8.1 ps, 41.7 aF of gate per fin,
    NAND2 g = 1.44 measured against 1.50 ideal (rise and fall within 4 %)

They are defaults, not constants: pass your own model for another corner.
The wire capacitance is an assumption (0.2 fF/um, typical of thin lower
metal) and the wordline's own RC is not included -- add it to `delay_ps`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "SLICE_WIDTH_UM",
    "WORDLINES_PER_SLICE",
    "DecoderSizing",
    "LogicalEffortModel",
    "Stage",
    "size_decoder",
    "split_fins_into_rows",
]

#: A slice is four wordlines at the 108 nm bitcell pitch.
WORDLINES_PER_SLICE = 4
SLICE_WIDTH_UM = 0.432
#: The tallest device `FinFETSpec` draws; a wider one is folded into more rows.
_MAX_FINS_PER_BAND = 18


@dataclass(frozen=True)
class LogicalEffortModel:
    """Technology numbers for logical-effort sizing (ASAP7 TT, 0.7 V, RVT)."""

    tau_ps: float = 1.76
    p_inv: float = 0.59
    gate_cap_aF_per_fin: float = 41.7
    wire_cap_fF_per_um: float = 0.2

    @property
    def c_fin_fF(self) -> float:
        return self.gate_cap_aF_per_fin / 1000.0

    def g(self, inputs: int) -> float:
        """Logical effort of an `inputs`-input NAND (1 is an inverter).

        With equal n and p strength per fin (ASAP7's P1N1) a NAND sized for
        unit drive has ``inputs`` series nFET fins against one pFET fin per
        input, so ``g = (inputs + 1) / 2``.
        """
        return (inputs + 1) / 2.0

    def p(self, inputs: int) -> float:
        return inputs * self.p_inv


@dataclass(frozen=True)
class Stage:
    """One sized gate: ``drive`` is the fin count of the equal-drive inverter.

    An ``inputs``-input NAND of drive ``k`` has ``k`` pFET fins and
    ``inputs * k`` fins in each series nFET, per input.
    """

    name: str
    inputs: int
    drive: int
    load_fF: float
    model: LogicalEffortModel = field(repr=False, default_factory=LogicalEffortModel)

    @property
    def n_fins(self) -> int:
        return self.inputs * self.drive

    @property
    def p_fins(self) -> int:
        return self.drive

    @property
    def c_in_fF(self) -> float:
        """Capacitance one input presents to whatever drives it."""
        return (self.n_fins + self.p_fins) * self.model.c_fin_fF

    @property
    def effort(self) -> float:
        """Stage effort actually achieved after rounding to whole fins."""
        return self.model.g(self.inputs) * self.load_fF / self.c_in_fF

    @property
    def delay_ps(self) -> float:
        return self.model.tau_ps * (self.effort + self.model.p(self.inputs))


@dataclass(frozen=True)
class DecoderSizing:
    """The sized decoder.  `stages` is keyed by role; `paths` by predecode line."""

    wl_load_fF: float
    depth: int
    stage_effort: float
    address_bits: int
    slices: int
    groups: dict[str, int]
    stages: dict[str, Stage]
    line_load_fF: dict[str, float]
    paths: dict[str, tuple[str, ...]]
    model: LogicalEffortModel = field(repr=False, default_factory=LogicalEffortModel)

    def path_delay_ps(self, line: str) -> float:
        return sum(self.stages[name].delay_ps for name in self.paths[line])

    @property
    def delay_ps(self) -> float:
        """Address to wordline-driver output along the slowest predecode line.

        Gate delay only: the distributed RC of the wordline itself is the
        array's and is added by whoever knows its length.
        """
        return max(self.path_delay_ps(line) for line in self.paths)

    @property
    def critical_line(self) -> str:
        return max(self.paths, key=self.path_delay_ps)

    @property
    def address_input_cap_fF(self) -> float:
        """Largest load any one address bit sees at the predecoder."""
        firsts = [self.stages[path[0]] for path in self.paths.values()]
        return max(stage.c_in_fF for stage in firsts)

    # ── What the layout generators need ───────────────────────────────────────
    @property
    def driver_rows(self) -> tuple[tuple[int, int], ...]:
        """`InverterSpec.rows` for the two-finger wordline driver."""
        per_finger = self.stages["driver"].drive // 2
        return tuple((fins, fins) for fins in split_fins_into_rows(per_finger))

    @property
    def nand_rows(self) -> tuple[tuple[int, int], ...]:
        """`NandSpec.rows` for the two-finger post-decode NAND."""
        nand = self.stages["post_nand"]
        return ((nand.n_fins // 2, nand.p_fins // 2),)

    def summary(self) -> str:
        lines = [
            (
                f"decoder for {self.depth} wordlines of {self.wl_load_fF:.1f} fF, "
                f"f = {self.stage_effort:.2f}: {self.address_bits} address bits, "
                f"{self.slices} slice(s), groups {self.groups}"
            ),
        ]
        for stage in self.stages.values():
            kind = "INV" if stage.inputs == 1 else f"NAND{stage.inputs}"
            lines.append(
                f"  {stage.name:12s} {kind:5s} n={stage.n_fins:3d} p={stage.p_fins:3d} fins  "
                f"load {stage.load_fF:7.2f} fF  effort {stage.effort:4.2f}  {stage.delay_ps:5.1f} ps"
            )
        lines.append(
            f"  slowest line {self.critical_line}: {self.delay_ps:.1f} ps gate delay, "
            f"address input {self.address_input_cap_fF:.2f} fF"
        )
        return "\n".join(lines)


def split_fins_into_rows(
    fins: int, *, limit: int = _MAX_FINS_PER_BAND
) -> tuple[int, ...]:
    """Fold `fins` into bands of at most `limit`, tallest first.

    31 fins per finger becomes ``(18, 13)`` -- the released driver's two rows.
    """
    if fins < 1:
        raise ValueError(f"fins must be >= 1, got {fins}")
    rows = [limit] * (fins // limit)
    if fins % limit:
        rows.append(fins % limit)
    return tuple(rows)


def _size(
    name: str,
    inputs: int,
    load_fF: float,
    effort: float,
    model: LogicalEffortModel,
    quantum: int = 1,
) -> Stage:
    """Size one gate for `load_fF`, rounding its drive to a multiple of `quantum`."""
    c_in = model.g(inputs) * load_fF / effort
    ideal = c_in / ((inputs + 1) * model.c_fin_fF)
    drive = max(quantum, quantum * round(ideal / quantum))
    return Stage(name=name, inputs=inputs, drive=drive, load_fF=load_fF, model=model)


def size_decoder(
    wl_load_fF: float,
    depth: int,
    *,
    stage_effort: float = 4.0,
    model: LogicalEffortModel | None = None,
) -> DecoderSizing:
    """Size every gate between an address bit and the wordline, in fins.

    Args:
        wl_load_fF: what one wordline presents to its driver -- the access
            gates of every cell on it plus its wire.  It is set by the cells
            *along* a wordline (bits x column-mux ratio), not by `depth`.
        depth: number of wordlines.  It decides how the address is split into
            predecode groups and how many gates share each predecode line,
            not how big the wordline driver is.
        stage_effort: target effort per stage.  Four is the classic optimum;
            the released decoder sits at about 4.4.
        model: technology numbers; defaults to the measured ASAP7 TT set.

    The two bits decoded inside a slice are group ``PC``; the rest are split
    into ``PB`` (the larger half, ANDed with PC per wordline) and ``PA``
    (ANDed with WLENA per slice), which is the released 5-to-32 arrangement
    (PA 1 bit, PB 2, PC 2).  The driver and the post-decode NAND are rounded
    to even drives because both are drawn with two fingers.
    """
    if wl_load_fF <= 0:
        raise ValueError(f"wl_load_fF must be positive, got {wl_load_fF}")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError(f"depth must be a positive integer, got {depth!r}")
    if stage_effort <= 1:
        raise ValueError(f"stage_effort must exceed 1, got {stage_effort}")
    model = model or LogicalEffortModel()

    slices = math.ceil(depth / WORDLINES_PER_SLICE)
    address_bits = max(2, math.ceil(math.log2(depth))) if depth > 1 else 2
    slice_bits = address_bits - 2
    groups = {"PC": 2, "PB": slice_bits - slice_bits // 2, "PA": slice_bits // 2}
    if max(groups.values()) > 4:
        raise ValueError(
            f"{depth} wordlines need a {max(groups.values())}-bit predecode group; "
            "split the array into banks or add a predecode level"
        )

    f = stage_effort
    stages: dict[str, Stage] = {}
    stages["driver"] = _size("driver", 1, wl_load_fF, f, model, quantum=2)
    stages["post_nand"] = _size(
        "post_nand", 2, stages["driver"].c_in_fF, f, model, quantum=2
    )
    nand_in = stages["post_nand"].c_in_fF
    stages["row_inv"] = _size("row_inv", 1, nand_in, f, model)
    stages["select_inv"] = _size(
        "select_inv", 1, WORDLINES_PER_SLICE * nand_in, f, model
    )
    stages["row_nand"] = _size("row_nand", 2, stages["row_inv"].c_in_fF, f, model)
    stages["select_nand"] = _size(
        "select_nand", 2, stages["select_inv"].c_in_fF, f, model
    )

    # Predecode lines run across every slice; each is loaded by the slice
    # gates that listen to it plus its own wire.
    wire = slices * SLICE_WIDTH_UM * model.wire_cap_fF_per_um
    row_in, select_in = stages["row_nand"].c_in_fF, stages["select_nand"].c_in_fF
    listeners = {
        "PC": slices * row_in,
        "PB": math.ceil(slices / 2 ** groups["PB"]) * WORDLINES_PER_SLICE * row_in,
        "PA": math.ceil(slices / 2 ** groups["PA"]) * select_in,
    }
    line_load = {line: gates + wire for line, gates in listeners.items()}
    line_load["WLENA"] = slices * select_in + wire

    paths: dict[str, tuple[str, ...]] = {}
    for line, bits in groups.items():
        if bits == 0:  # the line is tied high; nothing drives it
            line_load.pop(line)
            continue
        inv, gate = f"{line.lower()}_inv", f"{line.lower()}_nand"
        stages[inv] = _size(inv, 1, line_load[line], f, model)
        stages[gate] = _size(gate, max(bits, 1), stages[inv].c_in_fF, f, model)
        tail = (
            ("select_nand", "select_inv") if line == "PA" else ("row_nand", "row_inv")
        )
        paths[line] = (gate, inv, *tail, "post_nand", "driver")
    if not paths:  # a single slice: the address is the PC group alone
        raise AssertionError("PC is always decoded")  # pragma: no cover

    return DecoderSizing(
        wl_load_fF=wl_load_fF,
        depth=depth,
        stage_effort=f,
        address_bits=address_bits,
        slices=slices,
        groups=groups,
        stages=stages,
        line_load_fF=line_load,
        paths=paths,
        model=model,
    )
