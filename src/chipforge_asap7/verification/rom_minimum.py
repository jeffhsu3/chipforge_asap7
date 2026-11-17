"""A scoped minimum-area proof for the existing periodic NOR-ROM topology.

The lower-bound model uses only necessary FEOL inequalities, with continuous
coordinates. UNSAT in this relaxed model excludes smaller tiles in the stated
topology. A routing model adds the tile's contacts and wiring -- LISD, V0, the
M1 VSS column and drain stubs, programming V1s and the M2 bitline -- under the
pinned M1/M2/V0/V1/LISD rules, including the edge-length spacing classes. The
drawn routing must satisfy it at the bound. A separate drawn witness must
attain the bound, pass the pinned gdscheck deck, and have the expected
extracted connectivity. This is not a search over other NOR-ROM architectures.

Run with ``python -m chipforge_asap7.verification.rom_minimum --help``.
Z3 is invoked through SMT-LIB; no Python solver bindings are required.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..devices.rom_bitcell import (
    RomArraySpec,
    RomBitcellSpec,
    build_rom_array,
    build_rom_bitcell,
)
from ..layout.layers import LAYERS, require_gdspy
from .drc import drc_counts, find_gdscheck, run_gdscheck
from .minimum_proof import dimension_queries, run_query, smt_number
from .minimum_proof import smt_assert as _assert
from .minimum_proof import tool_info as _tool_info
from .minimum_proof import validate_rule_source as validate_pinned_source

ASSUMPTIONS = (
    "Periodic rectangular two-word, one-bit tile with continuous diffusion along X.",
    "Gate order is WLa, WLb, ISO, then the next tile's WLa; ISO is held at VSS.",
    "Two NMOS words share a grounded source and have independently V1-programmed drains.",
    "Each transistor has the requested number of distinct parallel fins.",
    "Different bit rows have separate ACTIVE strips and bitlines.",
    "Ordinary ASAP7 rules apply: no SRAMDRC marker or SRAM rule exceptions.",
)
ROUTING_ASSUMPTIONS = (
    "Each source/drain column has one rectangular LISD bar spanning its row's ACTIVE, clear of both gates.",
    "Each LISD bar carries one V0; V0s and V1s are 18 nm squares.",
    "The shared source reaches VSS through one straight full-height M1 column, continuous across rows.",
    "Each drain lands on its own rectangular M1 stub, horizontal or vertical, inside its tile row.",
    "Each programming V1 sits on its drain's stub; both meet one straight full-width M2 bitline per row.",
    "Identical rows repeat in Y; the wordline straps, taps and edges lie outside the tile.",
)
LIMITS = (
    "Area is the periodic core area per bit; straps, taps, edges and periphery are excluded.",
    "Alternative topologies, programming layers and diffusion-sharing schemes are outside the proof.",
    "Switches are ideal; leakage, read current, timing, noise margins and PVT are not proved.",
    "The SMT model is a necessary-rule relaxation, not an encoding of the full DRC deck.",
    "The routing model omits corner-to-corner (M1.S.6) spacing and conditional V0/V1 spacings.",
    "Rule transcription and the topology-to-constraint mapping are explicit trusted assumptions.",
    "Z3 proof terms are saved; they are not independently checked by a second proof checker.",
    "DRC checks the finite witness; it does not certify every array size or programmed image.",
)


def rule_manifest() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("rom_minimum_rules.json").read_text())


def validate_rule_source(pdk: Path) -> dict[str, Any]:
    return validate_pinned_source(pdk, rule_manifest())


def _facts() -> dict[str, int]:
    return {name: fact["value"] for name, fact in rule_manifest()["facts"].items()}


def minimum_dimensions(fins: int) -> tuple[int, int]:
    """Lower bounds on tile width and row pitch, independently of the generator."""
    if isinstance(fins, bool) or not isinstance(fins, int) or fins < 1:
        raise ValueError("fins must be a positive integer")
    r = _facts()
    width = 3 * (r["gate_width_nm"] + r["gate_space_nm"])
    height = (
        (fins - 1) * r["fin_pitch_nm"]
        + r["fin_width_nm"]
        + 2 * r["active_fin_enclosure_nm"]
        + r["active_space_nm"]
    )
    return width, height


def layout_constraints(fins: int) -> str:
    """Necessary linear inequalities; neither minimum dimension is asserted.

    The wraparound gate spacing and inter-row ACTIVE spacing encode the two
    periodic boundaries. Fin grid membership is relaxed to minimum spacing
    between distinct fins, making the modeled space larger, not smaller.
    """
    minimum_dimensions(fins)  # validate the count
    r = _facts()
    pitch = r["gate_width_nm"] + r["gate_space_nm"]
    names = ["width", "height", "ga", "gb", "gi", "alo", "ahi"]
    names += [f"fin{i}" for i in range(fins)]
    lines = [
        "; Units: nm. See rom_minimum_rules.json for the audited rule facts.",
        "; Three gates per period is a topology assumption, not a DRC rule.",
        "(set-option :produce-proofs true)",
        "(set-logic QF_LRA)",
        *(f"(declare-const {name} Real)" for name in names),
        _assert("positive_width", "(> width 0)"),
        _assert("positive_height", "(> height 0)"),
        _assert("gate_ab", f"(>= (- gb ga) {pitch})"),
        _assert("gate_bi", f"(>= (- gi gb) {pitch})"),
        _assert("gate_wrap", f"(>= (- (+ ga width) gi) {pitch})"),
        _assert("active_lower", f"(>= (- fin0 alo) {r['active_fin_enclosure_nm']})"),
        _assert(
            "active_upper",
            f"(>= (- ahi fin{fins - 1}) {r['fin_width_nm'] + r['active_fin_enclosure_nm']})",
        ),
        _assert("active_wrap", f"(>= (- (+ alo height) ahi) {r['active_space_nm']})"),
    ]
    for i in range(1, fins):
        lines.append(
            _assert(
                f"fin_spacing_{i}", f"(>= (- fin{i} fin{i - 1}) {r['fin_pitch_nm']})"
            )
        )
    return "\n".join(lines) + "\n"


#: Routed rectangles in the tile, in cyclic X order: the columns' LISD bars,
#: their V0s, the M1 VSS column and drain stubs, and the two programming V1s.
COLUMNS = ("da", "s", "db")
ROUTED = (
    *(f"lisd_{c}" for c in COLUMNS),
    *(f"v0_{c}" for c in COLUMNS),
    "m1_s",
    "m1_da",
    "m1_db",
    "v1_da",
    "v1_db",
)


def _rect(name: str) -> tuple[str, str, str, str]:
    return tuple(f"{name}_{edge}" for edge in ("x0", "y0", "x1", "y1"))


def _edge_class_space(
    facts: dict[str, Any], prefix: str, la: str, lb: str, tiny: bool
) -> str:
    """Spacing two facing edges of lengths `la` and `lb` need, as an SMT term.

    Long edges are at least `edge_long_min_nm`; a short edge is `mid` from
    `edge_mid_min_nm` and `tiny` below it (pdk.yml's virtual layers). Long to
    long is S.1, short to long S.2, mid to mid S.3; a tiny edge facing any
    short edge is S.4/S.5 where the layer has them (`tiny`); LISD has none, so
    there the pair relaxes to S.1, which every pair satisfies.
    """
    long_ = facts["edge_long_min_nm"]
    mid = facts["edge_mid_min_nm"]
    is_long = lambda l: f"(>= {l} {smt_number(long_)})"
    is_tiny = lambda l: f"(< {l} {mid})"
    both_long = f"(and {is_long(la)} {is_long(lb)})"
    one_long = f"(or {is_long(la)} {is_long(lb)})"
    any_tiny = f"(or {is_tiny(la)} {is_tiny(lb)})"
    tiny_space = (
        facts[f"{prefix}_space_tiny_nm"] if tiny else facts[f"{prefix}_space_long_nm"]
    )
    return (
        f"(ite {both_long} {facts[f'{prefix}_space_long_nm']} "
        f"(ite {one_long} {facts[f'{prefix}_space_short_long_nm']} "
        f"(ite {any_tiny} {tiny_space} {facts[f'{prefix}_space_mid_nm']})))"
    )


def routing_constraints(fins: int) -> str:
    """The tile's contacts and wiring under the pinned routing rules.

    Appended to `layout_constraints`. Every assertion is a necessary condition
    on the stated routing topology (`ROUTING_ASSUMPTIONS`): a layout drawn in
    it must satisfy them all, and the drawn tile is checked to. The X order of
    the columns is fixed by the topology, with the next tile's drain stub at
    `+ width`; the next row's copy of every shape is at `+ height`.
    """
    minimum_dimensions(fins)
    r = _facts()
    via = r["via_size_nm"]
    half_gate = r["gate_width_nm"] / 2
    lines = [
        "; Routing: LISD, V0, M1 column and stubs, V1 and the M2 bitline.",
        *(f"(declare-const {v} Real)" for n in ROUTED for v in _rect(n)),
        "(declare-const m2_y0 Real)",
        "(declare-const m2_y1 Real)",
    ]
    add = lambda name, expr: lines.append(_assert(name, expr))

    def inside(inner: str, outer: str, margin: float = 0) -> str:
        ix0, iy0, ix1, iy1 = _rect(inner)
        ox0, oy0, ox1, oy1 = _rect(outer)
        m = smt_number(margin)
        return (
            f"(and (>= (- {ix0} {ox0}) {m}) (>= (- {ox1} {ix1}) {m}) "
            f"(>= (- {iy0} {oy0}) {m}) (>= (- {oy1} {iy1}) {m}))"
        )

    def pair_enclosure(inner: str, outer: str, near: float, far: float) -> str:
        """Some opposite pair encloses by `near` on one side and `far` on the other."""
        ix0, iy0, ix1, iy1 = _rect(inner)
        ox0, oy0, ox1, oy1 = _rect(outer)
        terms = []
        for lo, hi in (((ix0, ox0), (ox1, ix1)), ((iy0, oy0), (oy1, iy1))):
            a, b = f"(- {lo[0]} {lo[1]})", f"(- {hi[0]} {hi[1]})"
            terms.append(f"(and (>= {a} {near}) (>= {b} {far}))")
            terms.append(f"(and (>= {a} {far}) (>= {b} {near}))")
        return f"(or {' '.join(terms)})"

    def flush_pair(inner: str, outer: str) -> str:
        """Max enclosure 0 on an opposite pair: the outer shape is flush there."""
        ix0, iy0, ix1, iy1 = _rect(inner)
        ox0, oy0, ox1, oy1 = _rect(outer)
        return f"(or (and (= {ix0} {ox0}) (= {ix1} {ox1})) (and (= {iy0} {oy0}) (= {iy1} {oy1})))"

    def x_space(
        left: str, right: str, prefix: str, tiny: bool, shift: str = "0"
    ) -> str:
        """Facing across X: where the Y spans overlap, the class spacing applies."""
        _, ly0, lx1, ly1 = _rect(left)
        rx0, ry0, _, ry1 = _rect(right)
        gap = f"(- (+ {rx0} {shift}) {lx1})"
        facing = f"(and (< {ly0} {ry1}) (< {ry0} {ly1}))"
        need = _edge_class_space(
            r, prefix, f"(- {ly1} {ly0})", f"(- {ry1} {ry0})", tiny
        )
        return f"(and (>= {gap} 0) (=> {facing} (>= {gap} {need})))"

    def y_space(shape: str, prefix: str, tiny: bool) -> str:
        """A shape against its copy one row up: facing across Y, edges its width."""
        x0, y0, x1, y1 = _rect(shape)
        width = f"(- {x1} {x0})"
        need = _edge_class_space(r, prefix, width, width, tiny)
        return f"(>= (- (+ {y0} height) {y1}) {need})"

    gates = {"da": ("(- gi width)", "ga"), "s": ("ga", "gb"), "db": ("gb", "gi")}
    for column in COLUMNS:
        lisd, v0 = f"lisd_{column}", f"v0_{column}"
        lx0, ly0, lx1, ly1 = _rect(lisd)
        vx0, vy0, vx1, vy1 = _rect(v0)
        left, right = gates[column]
        add(f"lisd_width_{column}", f"(>= (- {lx1} {lx0}) {r['lisd_width_nm']})")
        add(f"lisd_spans_active_{column}", f"(and (<= {ly0} alo) (>= {ly1} ahi))")
        add(
            f"lisd_clear_of_gates_{column}",
            f"(and (>= {lx0} (+ {left} {smt_number(half_gate)})) (<= {lx1} (- {right} {smt_number(half_gate)})))",
        )
        add(f"lisd_row_space_{column}", y_space(lisd, "lisd", False))
        add(
            f"v0_size_{column}",
            f"(and (= (- {vx1} {vx0}) {via}) (= (- {vy1} {vy0}) {via}))",
        )
        add(f"v0_on_lisd_{column}", inside(v0, lisd))
        add(
            f"v0_lisd_enclosure_{column}",
            pair_enclosure(
                v0, lisd, r["v0_lisd_enclosure_nm"], r["v0_lisd_enclosure_nm"]
            ),
        )
        add(
            f"v0_row_space_{column}",
            f"(>= (- (+ {vy0} height) {vy1}) {r['v0_space_nm']})",
        )
    for a, b, shift in (("da", "s", "0"), ("s", "db", "0"), ("db", "da", "width")):
        add(
            f"lisd_space_{a}_{b}",
            x_space(f"lisd_{a}", f"lisd_{b}", "lisd", False, shift),
        )
        add(
            f"v0_space_{a}_{b}",
            _v0_x_space(f"v0_{a}", f"v0_{b}", r["v0_space_nm"], shift),
        )

    # The VSS column: full height and continuous across rows, so its sides are
    # long; it covers the source V0, flush with it across X (V0.M1.AUX.3).
    sx0, sy0, sx1, sy1 = _rect("m1_s")
    vx0, vy0, vx1, vy1 = _rect("v0_s")
    add("m1_column_full_height", f"(and (= {sy0} 0) (= {sy1} height))")
    add(
        "m1_column_on_v0",
        f"(and (= {sx0} {vx0}) (= {sx1} {vx1}) (<= {sy0} {vy0}) (>= {sy1} {vy1}))",
    )
    add("m1_column_width", f"(>= (- {sx1} {sx0}) {r['m1_width_nm']})")
    add(
        "m1_column_endcap",
        f"(or (>= (- {vy0} {sy0}) {r['v0_m1_endcap_nm']}) (>= (- {sy1} {vy1}) {r['v0_m1_endcap_nm']}))",
    )

    for drain in ("da", "db"):
        stub, v0, v1 = f"m1_{drain}", f"v0_{drain}", f"v1_{drain}"
        mx0, my0, mx1, my1 = _rect(stub)
        ux0, uy0, ux1, uy1 = _rect(v1)
        add(f"m1_stub_on_v0_{drain}", inside(v0, stub))
        add(f"m1_stub_flush_{drain}", flush_pair(v0, stub))
        add(
            f"m1_stub_endcap_{drain}", pair_enclosure(v0, stub, r["v0_m1_endcap_nm"], 0)
        )
        # M1.A.1 on a stub one via wide across its flush pair: linear in the length.
        add(
            f"m1_stub_area_{drain}",
            f"(or (>= (* {via} (- {mx1} {mx0})) {r['m1_area_nm2']}) (>= (* {via} (- {my1} {my0})) {r['m1_area_nm2']}))",
        )
        add(f"m1_stub_in_row_{drain}", f"(and (>= {my0} 0) (<= {my1} height))")
        add(f"m1_stub_row_space_{drain}", y_space(stub, "m1", True))
        add(
            f"v1_size_{drain}",
            f"(and (= (- {ux1} {ux0}) {via}) (= (- {uy1} {uy0}) {via}))",
        )
        add(f"v1_on_stub_{drain}", inside(v1, stub))
        add(
            f"v1_m1_enclosure_{drain}",
            pair_enclosure(
                v1, stub, r["v1_m1_enclosure_nm"], r["v1_m1_opposite_enclosure_nm"]
            ),
        )
        # The bitline runs the full width, so V1.M2.EN.2's end-cap is along X
        # and V1.M2.AUX.2's flush pair can only be across Y.
        add(f"v1_m2_flush_{drain}", f"(and (= {uy0} m2_y0) (= {uy1} m2_y1))")
        add(
            f"v1_row_space_{drain}",
            f"(>= (- (+ {uy0} height) {uy1}) {r['v1_space_nm']})",
        )
    add("m1_space_da_s", x_space("m1_da", "m1_s", "m1", True))
    add("m1_space_s_db", x_space("m1_s", "m1_db", "m1", True))
    add("m1_space_db_da", x_space("m1_db", "m1_da", "m1", True, "width"))
    add("v1_space_da_db", f"(>= (- v1_db_x0 v1_da_x1) {r['v1_space_nm']})")
    add("v1_space_db_da", f"(>= (- (+ v1_da_x0 width) v1_db_x1) {r['v1_space_nm']})")
    add("m2_width", f"(>= (- m2_y1 m2_y0) {r['m2_width_nm']})")
    add("m2_row_space", f"(>= (- (+ m2_y0 height) m2_y1) {r['m2_space_long_nm']})")
    return "\n".join(lines) + "\n"


def _v0_x_space(left: str, right: str, space: int, shift: str) -> str:
    _, ly0, lx1, ly1 = _rect(left)
    rx0, ry0, _, ry1 = _rect(right)
    gap = f"(- (+ {rx0} {shift}) {lx1})"
    facing = f"(and (< {ly0} {ry1}) (< {ry0} {ly1}))"
    return f"(and (>= {gap} 0) (=> {facing} (>= {gap} {space})))"


def _rectangle_reader(cell: Any):
    polygons = cell.get_polygons(by_spec=True)

    def rectangles(layer: str) -> list[tuple[float, float, float, float]]:
        key = (LAYERS[layer]["layer"], LAYERS[layer]["datatype"])
        boxes = []
        for points in polygons.get(key, []):
            x0, y0 = map(float, points.min(axis=0))
            x1, y1 = map(float, points.max(axis=0))
            if len(points) != 4 or set(map(tuple, points)) != {
                (x0, y0),
                (x0, y1),
                (x1, y0),
                (x1, y1),
            }:
                raise ValueError(f"proof witness requires rectangular {layer}")
            boxes.append((x0, y0, x1, y1))
        return sorted(boxes)

    return rectangles


def measure_routing(
    tile: Any, array: Any, spec: RomArraySpec, bit: int
) -> dict[str, float]:
    """Read the drawn routing of one tile, with V1s from a programmed array row.

    The tile has no V1s: `build_rom_array` drops them on the stubs. `bit` must
    be a row of `array` whose first tile has both words programmed; its V1s are
    read back into tile coordinates.
    """
    rectangles = _rectangle_reader(tile)
    lisd, v0, m1, m2 = (rectangles(name) for name in ("LISD", "V0", "M1", "M2"))
    height = spec.bitcell.height
    if len(lisd) != 3 or len(v0) != 3 or len(m1) != 3 or len(m2) != 1:
        raise ValueError("proof witness needs three LISD/V0/M1 shapes and one M2")
    column = [box for box in m1 if box[1] == 0 and box[3] == height]
    stubs = sorted(box for box in m1 if box not in column)
    if len(column) != 1 or len(stubs) != 2:
        raise ValueError("proof witness needs one full-height M1 column and two stubs")
    row = spec.row_y(bit)
    v1 = sorted(
        (x0, y0 - row, x1, y1 - row)
        for x0, y0, x1, y1 in _rectangle_reader(array)("V1")
        if 0 <= (x0 + x1) / 2 < spec.bitcell.width
        and row <= (y0 + y1) / 2 < row + height
    )
    if len(v1) != 2:
        raise ValueError("the witness row must program both words of its first tile")
    if m2[0][0] > 0 or m2[0][2] < spec.bitcell.width:
        raise ValueError("proof witness bitline does not span the tile")
    shapes = {
        **{f"lisd_{c}": box for c, box in zip(COLUMNS, lisd)},
        **{f"v0_{c}": box for c, box in zip(COLUMNS, v0)},
        "m1_s": column[0],
        "m1_da": stubs[0],
        "m1_db": stubs[1],
        "v1_da": v1[0],
        "v1_db": v1[1],
    }
    witness = {
        var: value
        for name, box in shapes.items()
        for var, value in zip(_rect(name), box)
    }
    witness["m2_y0"], witness["m2_y1"] = m2[0][1], m2[0][3]
    return witness


def measure_tile(cell: Any, fins: int) -> dict[str, float]:
    """Read the actual polygons, not the spec's width/height properties."""
    rectangles = _rectangle_reader(cell)
    if rectangles("SRAMDRC") or rectangles("GATE_CUT"):
        raise ValueError("proof witness excludes SRAMDRC and gate cuts in the tile")
    boundaries, active = rectangles("BOUNDARY"), rectangles("ACTIVE")
    if len(boundaries) != 1 or len(active) != 1:
        raise ValueError(
            "proof witness needs one boundary and one continuous ACTIVE strip"
        )
    x0, y0, x1, y1 = boundaries[0]
    ax0, alo, ax1, ahi = active[0]
    if (x0, y0) != (0, 0) or (ax0, ax1) != (x0, x1):
        raise ValueError(
            "proof witness must start at zero and span the tile with ACTIVE"
        )
    r = _facts()
    gates = rectangles("GATE")
    if len(gates) != 3 or any(
        b - a != r["gate_width_nm"] or (lo, hi) != (y0, y1) for a, lo, b, hi in gates
    ):
        raise ValueError("proof witness requires three full-height, rule-width gates")
    active_fins = [box for box in rectangles("FIN") if box[1] < ahi and box[3] > alo]
    active_fins.sort(key=lambda box: box[1])
    if len(active_fins) != fins or any(
        (a, b) != (x0, x1) or hi - lo != r["fin_width_nm"]
        for a, lo, b, hi in active_fins
    ):
        raise ValueError("proof witness has the wrong fin count, width or span")
    return {
        "width": x1 - x0,
        "height": y1 - y0,
        **{
            name: (gate[0] + gate[2]) / 2
            for name, gate in zip(("ga", "gb", "gi"), gates)
        },
        "alo": alo,
        "ahi": ahi,
        **{f"fin{i}": fin[1] for i, fin in enumerate(active_fins)},
    }


def proof_queries(
    fins: int, witness: dict[str, float], routing: dict[str, float] | None = None
) -> dict[str, tuple[str, str]]:
    """The FEOL bound and drawn witness, plus the routed witness when given.

    The bound stays on the FEOL model: its UNSAT needs no routing assumption.
    `routed_drawn_witness` fixes every drawn routing coordinate as well and
    must be SAT under the routing rules too, so the drawn wiring fits the bound.
    """
    width, height = minimum_dimensions(fins)
    queries = dimension_queries(layout_constraints(fins), width, height, witness)
    if routing is not None:
        source, _ = dimension_queries(
            layout_constraints(fins) + routing_constraints(fins),
            width,
            height,
            {**witness, **routing},
        )["drawn_witness"]
        queries["routed_drawn_witness"] = (source, "sat")
    return queries


def topology_check() -> dict[str, Any]:
    """Exhaust all programming and wordline states of two periodic tiles.

    BL and VSS are nodes 0 and 1; the four distinct drains are nodes 2..5.
    Omitting ISO merges drains across each tile boundary and must produce a
    counterexample. This is an ideal-switch model, not extracted analog LVS.
    """

    def conducts(program: int, selected: int, isolated: bool) -> bool:
        parent = list(range(6))

        def find(node: int) -> int:
            while parent[node] != node:
                node = parent[node]
            return node

        edges = []
        for word in range(4):
            if program & (1 << word):
                edges.append((0, word + 2))
            if selected & (1 << word):
                edges.append((1, word + 2))
        if not isolated:
            edges.extend(((3, 4), (5, 2)))
        for a, b in edges:
            parent[find(a)] = find(b)
        return find(0) == find(1)

    counterexample = None
    for program in range(16):
        for selected in range(16):
            expected = bool(program & selected)
            if conducts(program, selected, True) != expected:
                raise ValueError(
                    "the stated topology does not implement independent NOR bits"
                )
            if (
                counterexample is None
                and conducts(program, selected, False) != expected
            ):
                counterexample = {
                    "program_mask": program,
                    "selected_wordlines": selected,
                    "expected_discharge": expected,
                    "actual_discharge": not expected,
                }
    if counterexample is None:
        raise ValueError(
            "removing isolation did not produce the required counterexample"
        )
    return {
        "words": 4,
        "programming_patterns": 16,
        "wordline_states": 16,
        "cases_passed": 256,
        "without_isolation_counterexample": counterexample,
    }


def extract_witness(gds: Path, cell_name: str, spec: RomArraySpec) -> dict[str, int]:
    """Check the drawn four-word array containing all 16 programming patterns."""
    import gdstk

    from . import connectivity as cv

    top = next(
        c for c in gdstk.read_gds(str(gds), unit=1e-6).cells if c.name == cell_name
    )
    flat = top.copy("flat").flatten()
    merged = gdstk.Cell("merged")
    for layer, datatype in {(p.layer, p.datatype) for p in flat.polygons}:
        same = [p for p in flat.polygons if (p.layer, p.datatype) == (layer, datatype)]
        merged.add(*gdstk.boolean(same, [], "or", layer=layer, datatype=datatype))
    merged.add(*flat.labels)
    devices, labels = cv.extract_connectivity(merged, expected_channels=96)
    if any(len(ids) != 1 for ids in labels.values()):
        raise ValueError("a witness pin is split across nets")
    nets = {name: next(iter(ids)) for name, ids in labels.items()}
    if len(set(nets.values())) != len(nets):
        raise ValueError("witness pins are shorted")
    word_gates = {nets[f"WL<{word}>"]: word for word in range(4)}
    seen: Counter[tuple[int, int]] = Counter()
    floating = set()
    isolation = 0
    for device in devices:
        if device["pmos"] or device["nfin"] != spec.bitcell.fins:
            raise ValueError("witness has an unexpected device type or fin count")
        if device["gate"] not in word_gates:
            if device["gate"] != nets["VSS"]:
                raise ValueError("isolation gate is not grounded")
            isolation += 1
            continue
        word = word_gates[device["gate"]]
        y = device["center"][1] * 1000
        rows = [
            bit
            for bit in range(16)
            if spec.row_y(bit) <= y < spec.row_y(bit) + spec.bitcell.height
        ]
        if len(rows) != 1:
            raise ValueError("word device is outside its bit row")
        bit = rows[0]
        seen[bit, word] += 1
        terminals = set(device["terminals"])
        if nets["VSS"] not in terminals or len(terminals) != 2:
            raise ValueError(
                "word device lacks its grounded source or has a shorted drain"
            )
        (drain,) = terminals - {nets["VSS"]}
        if bit & (1 << word):
            if drain != nets[f"BL<{bit}>"]:
                raise ValueError("programmed drain does not reach its own bitline")
        else:
            if drain in nets.values() or drain in floating:
                raise ValueError("unprogrammed drain is not independently floating")
            floating.add(drain)
    if (
        seen != Counter({(bit, word): 1 for bit in range(16) for word in range(4)})
        or isolation != 32
    ):
        raise ValueError(
            "witness transistor inventory differs from the stated topology"
        )
    return {
        "programming_patterns": 16,
        "word_devices": sum(seen.values()),
        "grounded_isolation_devices": isolation,
    }


def _draw_witness(
    fins: int, gds: Path
) -> tuple[dict[str, float], dict[str, float], RomArraySpec, str]:
    gdspy = require_gdspy()
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    previous = gdspy.current_library
    gdspy.current_library = library
    try:
        bitcell = RomBitcellSpec(fins=fins)
        tile = build_rom_bitcell(bitcell, lib=library)
        witness = measure_tile(tile, fins)
        # Distinct bitlines carry all four-word programming patterns. Both
        # within-pair and between-pair neighbours see every bit combination.
        array = RomArraySpec(words=4, bits=16, bitcell=bitcell, strap_every=16)
        sites = [
            (word, bit) for bit in range(16) for word in range(4) if bit & (1 << word)
        ]
        top = build_rom_array(array, sites, lib=library)
        library.write_gds(str(gds))
        # Row 3 programs both words of every tile.
        routing = measure_routing(tile, top, array, bit=3)
        return witness, routing, array, top.name
    finally:
        gdspy.current_library = previous


def prove_minimum(
    fins: int, out: Path, *, pdk: Path, z3: Path, gdscheck: Path
) -> dict[str, Any]:
    """Produce the bound, solver logs and fully checked constructive witness.

    Failure leaves a failed report; a previous successful report cannot survive
    a failed rerun. Every success requires source hashes, SAT/UNSAT, extraction
    and DRC. No missing verification tool is treated as a pass.
    """
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "format": "chipforge_asap7.rom_minimum.v2",
        "status": "incomplete",
        "fins": fins,
        "assumptions": ASSUMPTIONS,
        "limits": LIMITS,
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        width, height = minimum_dimensions(fins)
        report["rule_source"] = validate_rule_source(pdk)
        package = Path(__file__).resolve().parents[1]
        report["implementation_sha256"] = {
            str(path.relative_to(package)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(package.rglob("*.py"))
        }
        report["tools"] = {
            "z3": _tool_info(z3, "-version"),
            "gdscheck": _tool_info(gdscheck, "--version"),
        }
        report["topology"] = topology_check()
        gds = out / "witness.gds"
        witness, routing, array, cell_name = _draw_witness(fins, gds)
        if (witness["width"], witness["height"]) != (width, height):
            raise ValueError("drawn tile does not attain the independent lower bound")
        report["drawn_tile_nm"] = witness
        report["drawn_routing_nm"] = routing
        report["routing_assumptions"] = ROUTING_ASSUMPTIONS
        report["queries"] = {
            name: run_query(z3, source, out / f"{name}.smt2", expected)
            for name, (source, expected) in proof_queries(
                fins, witness, routing
            ).items()
        }
        report["gds"] = {
            "path": gds.name,
            "cell": cell_name,
            "sha256": sha256(gds.read_bytes()).hexdigest(),
        }
        violations = run_gdscheck(
            gds,
            out / "drc",
            cell_name=cell_name,
            binary=gdscheck,
            pdk_file=pdk / "pdk.yml",
        )
        report["drc"] = {"suite": "main", "violations": dict(drc_counts(violations))}
        if violations:
            raise ValueError("the constructive witness fails gdscheck")
        report["extracted_connectivity"] = extract_witness(gds, cell_name, array)
        report["minimum"] = {
            "tile_width_nm": width,
            "row_pitch_nm": height,
            "bits_per_tile": 2,
            "core_area_per_bit_nm2": width * height // 2,
        }
        report["area_argument"] = (
            "Positive width >= W and height >= H imply width*height/2 >= W*H/2. The drawn witness attains both bounds."
        )
        report["status"] = "proved_for_stated_topology"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fins", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--gdscheck-root",
        type=Path,
        required=True,
        help="checkout containing the pinned pdks/asap7 sources",
    )
    parser.add_argument("--gdscheck", type=Path)
    parser.add_argument("--z3", type=Path, default=shutil.which("z3"))
    parser.add_argument("--out", type=Path, default=Path("build/rom_minimum"))
    args = parser.parse_args(argv)
    if args.z3 is None:
        parser.error("Z3 is required; install its executable or supply --z3")
    try:
        binary = find_gdscheck(args.gdscheck)
        for fins in dict.fromkeys(args.fins):
            destination = args.out / f"f{fins}"
            print(f"Checking {fins}-fin ROM minimum...", flush=True)
            result = prove_minimum(
                fins,
                destination,
                pdk=args.gdscheck_root / "pdks/asap7",
                z3=Path(args.z3),
                gdscheck=binary,
            )
            print(
                f"  {result['status']}: {result['minimum']['core_area_per_bit_nm2']} nm²/bit -> {destination / 'report.json'}",
                flush=True,
            )
    except (
        ValueError,
        RuntimeError,
        OSError,
        subprocess.SubprocessError,
        ImportError,
    ) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    main()
