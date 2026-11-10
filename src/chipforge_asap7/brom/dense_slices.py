"""Local bit datapaths with control placement in the remaining row space.

Each tile holds one bit's sampling receiver, latch and output register per
row. Tile width follows the actual cell widths, rather than the word width.
Mux4 readouts occupy four interleaved rows aligned to the physical bitlines.
The control and completion trees are placed against the actual x AND y pin
positions of those datapaths, so their wires need not all rise to the top.
"""

import re
from collections import defaultdict

import numpy as np

from .dense_cell import wall_gap
from .dense_place import _blocks, _legalize, _segments, unit_gap

_LATCH = re.compile(r"output_latch_(\d+)$")
_READOUT = re.compile(r"readout_(\d+)$")
_BIT_UNIT = re.compile(
    r"(?:output_latch|output_mux|register_master|register_slave|sample_receiver_\w+)_(\d+)$"
)


def _shape(units):
    """A legal, ordered run of cells, including its edge spacers."""
    placed, x = [], 0
    for unit in units:
        if placed:
            x += unit_gap(placed[-1][0], unit)[0]
        placed.append((unit, x))
        x += unit.plan.width
    return placed, x


def _datapaths(units, anchors):
    bits = {int(m[1]) for u in units if (m := _LATCH.fullmatch(u.name))}
    columns = {int(m[1]) for u in units if (m := _READOUT.fullmatch(u.name))}
    if not bits or bits != set(range(len(bits))):
        return None
    ways = len(columns) // len(bits)
    if ways not in (1, 4) or columns != set(range(len(bits) * ways)):
        return None
    if any(len(anchors.get(f"BL<{c}>", ())) != 1 for c in columns):
        return None
    chains, controls, readouts = defaultdict(list), [], []
    for unit in units:
        match = _BIT_UNIT.fullmatch(unit.name)
        if match and int(match[1]) in bits:
            chains[int(match[1])].append(unit)
        elif match := _READOUT.fullmatch(unit.name):
            if ways == 4:
                readouts.append(unit)
            else:
                chains[int(match[1])].append(unit)
        else:
            controls.append(unit)

    def rank(unit):
        if unit.name.startswith("readout_"):
            return 0
        if unit.name.startswith("output_mux_"):
            return 1
        if unit.name.startswith("register_master_"):
            return 3
        if unit.name.startswith("register_slave_"):
            return 4
        return 2

    # Stable sorting preserves each sample receiver/latch/observer run and
    # the internal order of the three mux4 gates.
    shapes = {bit: _shape(sorted(chain, key=rank)) for bit, chain in chains.items()}
    return ways, shapes, controls, readouts


def _readout_rows(readouts, anchors, packed, first, width, interleave=4):
    """Align BL access trunks without crossing a tap gap or another cell.

    Column ``c`` prefers row ``first + c % interleave``.
    """
    segments = _segments(1, width)
    ordered = sorted(
        readouts,
        key=lambda u: anchors[f"BL<{int(_READOUT.fullmatch(u.name)[1])}>"][0],
    )
    for unit in ordered:
        column = int(_READOUT.fullmatch(unit.name)[1])
        net = f"BL<{column}>"
        mapping = dict(unit.nets)
        pin_x = next(x for p, x in unit.plan.trunks if mapping[p] == net)
        target = round((anchors[net][0] - pin_x) / 54) * 54
        best = None
        for row in range(first, len(packed)):
            cells = packed[row]
            for j in range(len(cells) + 1):
                lo = (
                    cells[j - 1][1]
                    + cells[j - 1][0].plan.width
                    + unit_gap(cells[j - 1][0], unit)[1]
                    if j
                    else 108
                )
                hi = (
                    cells[j][1] - unit_gap(unit, cells[j][0])[1]
                    if j < len(cells)
                    else width
                ) - unit.plan.width
                for _, a, b in segments:
                    low, high = max(lo, a), min(hi, b - unit.plan.width)
                    if low > high:
                        continue
                    x = min(max(target, -(-low // 54) * 54), high // 54 * 54)
                    if x < low:
                        continue
                    score = (abs(x - target), row - first != column % interleave, row, x)
                    if best is None or score < best:
                        best = score
        if best is None:
            return False
        _, _, row, x = best
        packed[row].append((unit, x))
        packed[row].sort(key=lambda p: p[1])
    return True


def _free_segments(packed, width):
    """Space for control cells, keeping legal breaks beside fixed datapaths."""
    free = []
    for row, lo, hi in _segments(len(packed), width):
        start = lo
        for unit, x in packed[row]:
            if x + unit.plan.width <= lo or x >= hi:
                continue
            end = (x - 108) // 108 * 108
            if end > start:
                free.append((row, start, end))
            start = max(start, -(-(x + unit.plan.width + 108) // 108) * 108)
        if start < hi:
            free.append((row, start, hi))
    return free


def _place_controls(units, anchors, packed, width, sinks, spread_decoder=False):
    """Place fixed cell runs using pin offsets and two-dimensional anchors."""
    if not units:
        return packed
    fixed = {
        net: [(x, (len(packed) - 1) * 432) for x in xs if x is not None]
        for net, xs in anchors.items()
    }
    for row, cells in enumerate(packed):
        for unit, x in cells:
            mapping = dict(unit.nets)
            for port, dx in unit.plan.trunks:
                if port in unit.ports:
                    fixed.setdefault(mapping[port], []).append((x + dx, row * 432))

    blocks = _blocks(units)
    widths, inner, pins = [], [], defaultdict(list)
    for k, block in enumerate(blocks):
        placed, width_ = _shape([units[i] for i in block])
        widths.append(width_)
        inner.append([x for _, x in placed])
        for unit, x in placed:
            mapping = dict(unit.nets)
            for port, dx in unit.plan.trunks:
                if port in unit.ports and mapping[port] not in ("VDD", "VSS"):
                    pins[mapping[port]].append((k, x + dx))

    count = len(blocks)
    matrix = np.zeros((count, count))
    rhs = np.zeros((count, 2))
    for net, accesses in pins.items():
        offsets = defaultdict(list)
        for k, dx in accesses:
            offsets[k].append(dx)
        owners = [(k, float(np.mean(xs))) for k, xs in offsets.items()]
        boundary = fixed.get(net, ())
        degree = len(owners) + len(boundary)
        if degree < 2:
            continue
        # Constant total weight per net: a shared clock must not overwhelm
        # the local, two-terminal completion and data connections.
        weight = 2 / (degree * (degree - 1))
        for i, dx in owners:
            matrix[i, i] += weight * (degree - 1)
            for j, other_x in owners:
                if i != j:
                    matrix[i, j] -= weight
                    rhs[i, 0] += weight * (other_x - dx)
            for x, y in boundary:
                rhs[i, 0] += weight * (x - dx)
                rhs[i, 1] += weight * y
    vertical = matrix.copy()
    for net in set(sinks) - {"VDD", "VSS"}:
        for i in {k for k, _ in pins.get(net, ())}:
            vertical[i, i] += 1
    total, used, spring = sum(widths), 0, 1e-4
    for i, width_ in enumerate(widths):
        order = (used + width_ / 2) * width / total - width_ / 2
        used += width_
        matrix[i, i] += spring
        vertical[i, i] += spring
        rhs[i, 0] += spring * order
        rhs[i, 1] += spring * (len(packed) - 1) * 216
    targets = np.linalg.solve(matrix, rhs[:, 0])
    row_targets = np.linalg.solve(vertical, rhs[:, 1]) / 432
    if spread_decoder:
        # The decoder's many outputs share the narrow central strip. Give
        # its stages vertical room before the joint router refines x and
        # columns; a quadratic target alone piles them into the top rows.
        decoder = [
            i
            for i, block in enumerate(blocks)
            if all(
                re.fullmatch(
                    r"(?:inv|address|buffer|nor|nand|and2|or2)_\d+", units[j].name
                )
                for j in block
            )
        ]
        for rank, i in enumerate(
            sorted(decoder, key=lambda i: (row_targets[i], targets[i], i))
        ):
            row_targets[i] = rank * (len(packed) - 1) / max(len(decoder) - 1, 1)
    segments = _free_segments(packed, width)
    if not segments:
        return None
    placed = _legalize(
        blocks,
        widths,
        targets,
        segments,
        [0] * count,
        lambda j, k: unit_gap(units[blocks[j][-1]], units[blocks[k][0]]),
        lambda k: wall_gap(units[blocks[k][0]].plan.edges[0]),
        row_targets=row_targets,
    )
    if placed is None:
        return None
    for k, block in enumerate(blocks):
        row, x = placed[k]
        for i, dx in zip(block, inner[k]):
            packed[row].append((units[i], x + dx))
    for row in packed:
        row.sort(key=lambda p: p[1])
    return packed


def sliced_rows(units, anchors, width, rows, *, sinks=(), spread_decoder=False):
    """Keep bit datapaths local, with all cells legal and body taps reserved.

    Return None for an unsupported circuit or insufficient space. ``rows``
    is a lower bound: tall tiles and the mux4 readout strip can add rows.
    Routing is still required before accepting this placement.
    """
    width = width // 108 * 108
    paths = _datapaths(units, anchors)
    if paths is None:
        return None
    ways, shapes, controls, readouts = paths
    # Leave space at both tile edges for fillers and completion-tree cells.
    minimum = max(w for _, w in shapes.values()) + 432
    tiles = []
    for _, lo, hi in _segments(1, width):
        count = max(1, (hi - lo) // minimum)
        for k in range(count):
            a = lo + ((hi - lo) * k // count) // 108 * 108
            b = lo + ((hi - lo) * (k + 1) // count) // 108 * 108
            if b - a >= minimum:
                tiles.append((a, b))
    if not tiles:
        return None
    assigned = defaultdict(list)
    for bit in shapes:
        x = sum(anchors[f"BL<{bit * ways + i}>"][0] for i in range(ways)) / ways
        tile = min(range(len(tiles)), key=lambda t: abs(x - sum(tiles[t]) / 2))
        assigned[tile].append((x, bit))
    data_rows = max(rows, max(map(len, assigned.values())))
    packed = [[] for _ in range(data_rows + (4 if readouts else 0))]
    for tile, entries in assigned.items():
        lo, hi = tiles[tile]
        for j, (x, bit) in enumerate(sorted(entries)):
            row = round(j * (data_rows - 1) / max(len(entries) - 1, 1))
            cells, width_ = shapes[bit]
            origin = max(
                lo + 108,
                min(round((x - width_ / 2) / 108) * 108, hi - width_ - 108),
            )
            packed[row].extend((unit, origin + dx) for unit, dx in cells)
    for row in packed:
        row.sort(key=lambda p: p[1])
    if not _readout_rows(readouts, anchors, packed, data_rows, width):
        return None
    return _place_controls(controls, anchors, packed, width, sinks, spread_decoder)


#: Rows beyond ``spare`` that `banded_rows` adds for control logic.
SPARE_LIMIT = 4
_MUX = re.compile(r"output_mux_(\d+)$")
_MASTER = re.compile(r"register_master_(\d+)$")
_SLAVE = re.compile(r"register_slave_(\d+)$")


def _bands(units, anchors):
    """Group a datapath into function bands of per-bit blocks.

    Return (readouts, bands, controls, centers) or None. ``bands`` lists,
    nearest the array first, each band's blocks as (bit, units); ``centers``
    maps a bit to the mean x of its bitlines.
    """
    bits = {int(m[1]) for u in units if (m := _LATCH.fullmatch(u.name))}
    columns = {int(m[1]) for u in units if (m := _READOUT.fullmatch(u.name))}
    if not bits or bits != set(range(len(bits))):
        return None
    ways = len(columns) // len(bits)
    if ways not in (1, 4) or columns != set(range(len(bits) * ways)):
        return None
    if any(len(anchors.get(f"BL<{c}>", ())) != 1 for c in columns):
        return None
    centers = {
        b: sum(anchors[f"BL<{b * ways + i}>"][0] for i in range(ways)) / ways
        for b in bits
    }
    readouts, controls = [], []
    mux, sample, master, slave = (defaultdict(list) for _ in range(4))
    # A SAMPLE receiver block runs from its first receiver cell to its last,
    # with its latches inside; keep it whole, in electrical order.
    for block in _blocks(units):
        members = [units[i] for i in block]
        first = members[0].name
        if (m := _BIT_UNIT.fullmatch(first)) and first.startswith("sample_receiver_"):
            latched = [
                int(k[1]) for u in members if (k := _LATCH.fullmatch(u.name))
            ]
            if latched:
                sample[min(latched)].extend(members)
                continue
        for unit in members:
            name = unit.name
            if _READOUT.fullmatch(name):
                readouts.append(unit)
            elif m := _MUX.fullmatch(name):
                mux[int(m[1])].append(unit)
            elif (m := _LATCH.fullmatch(name)) and int(m[1]) in bits:
                sample[int(m[1])].append(unit)
            elif m := _MASTER.fullmatch(name):
                master[int(m[1])].append(unit)
            elif m := _SLAVE.fullmatch(name):
                slave[int(m[1])].append(unit)
            else:
                controls.append(unit)
    register = {
        b: master.get(b, []) + slave.get(b, []) for b in set(master) | set(slave)
    }
    bands = [
        sorted(band.items())
        for band in (mux, sample, register)
        if band
    ]
    return readouts, bands, controls, centers


#: A bitline-free x range at least this wide (nm) between the arrays is
#: left to control logic: the decoder's SEL and G drops land there.
CONTROL_GAP = 4320


def _control_gap(anchors):
    """The widest bitline-free x range between the arrays, or None."""
    xs = sorted(
        x
        for net, points in anchors.items()
        if net.startswith("BL<")
        for x in points
        if x is not None
    )
    gaps = [(b - a, a, b) for a, b in zip(xs, xs[1:])]
    if not gaps or max(gaps)[0] < CONTROL_GAP:
        return None
    _, lo, hi = max(gaps)
    return lo, hi


def _band_segments(rows, width, gap):
    """Row segments for band cells, less the control gap and a spacer."""
    segments = _segments(rows, width)
    if gap is None:
        return segments
    lo = (gap[0] + 216) // 108 * 108
    hi = -(-(gap[1] - 216) // 108) * 108
    result = []
    for row, a, b in segments:
        for p, q in ((a, min(b, lo)), (max(a, hi), b)):
            if q - p >= 432:
                result.append((row, p, q))
    return result


def _band_rows(blocks, centers, width, rows, gap=None):
    """Abacus-pack one band's blocks into ``rows`` rows under their bits."""
    shapes = [_shape(members) for _, members in blocks]
    widths = [w for _, w in shapes]
    targets = [centers[bit] - w / 2 for (bit, _), w in zip(blocks, widths)]
    placed = _legalize(
        blocks,
        widths,
        targets,
        _band_segments(rows, width, gap),
        [0] * len(blocks),
        lambda j, k: unit_gap(shapes[j][0][-1][0], shapes[k][0][0][0]),
        lambda k: wall_gap(shapes[k][0][0][0].plan.edges[0]),
    )
    if placed is None:
        return None
    packed = [[] for _ in range(rows)]
    for k, (cells, _) in enumerate(shapes):
        row, x = placed[k]
        packed[row].extend((unit, x + dx) for unit, dx in cells)
    return packed


def banded_rows(
    units, anchors, width, *, fill=0.85, spare=1, sinks=(), spread_decoder=False
):
    """Pitch-matched function bands: each bit's cells sit under its bitlines.

    Nearest the array: the readouts, then a band of mux cells, a band of
    SAMPLE receiver and output latch blocks, and a band of register latches.
    Bands stay out of a bitline-free gap between the arrays, where the
    decoder's drops land; control logic fills it.
    Each band has the fewest rows its cells fill to ``fill``; ``spare``
    empty rows below take what control logic the bands' free space does
    not. Return None for an unsupported circuit or insufficient space.
    Routing is still required before accepting this placement.
    """
    width = width // 108 * 108
    grouped = _bands(units, anchors)
    if grouped is None:
        return None
    readouts, bands, controls, centers = grouped
    gap = _control_gap(anchors)
    capacity = sum(hi - lo for _, lo, hi in _band_segments(1, width, gap))
    stacked = []
    for blocks in bands:
        need = sum(_shape(members)[1] for _, members in blocks)
        rows = max(1, -(-need // int(capacity * fill)))
        while (packed := _band_rows(blocks, centers, width, rows, gap)) is None:
            rows += 1
            if rows > len(blocks):
                return None
        stacked.append(packed)
    need = sum(u.plan.width for u in readouts)
    readout_rows = max(1, -(-need // int(capacity * fill))) if readouts else 0
    # Rows are numbered from the bottom: the band nearest the array is last.
    # Control logic takes the bands' free space, then up to SPARE_LIMIT
    # further rows below them.
    for extra in range(spare, spare + SPARE_LIMIT + 1):
        packed = [[] for _ in range(extra)]
        for band in reversed(stacked):
            packed.extend([list(row) for row in band])
        first = len(packed)
        packed.extend([] for _ in range(readout_rows))
        for row in packed:
            row.sort(key=lambda p: p[1])
        if readouts and not _readout_rows(
            readouts, anchors, packed, first, width, readout_rows
        ):
            return None
        placed = _place_controls(
            list(controls), anchors, packed, width, sinks, spread_decoder
        )
        if placed is not None:
            return [row for row in placed if row]
    return None
