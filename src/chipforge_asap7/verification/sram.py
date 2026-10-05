"""ASAP7 8T bitcell, edge-cell, and tap/strap verification and topology auditing.

Audits dual-port bitcell topology, cross-coupled storage nodes, forced-state dummies,
caps, array abutment, and well/substrate tap rules.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
from pathlib import Path
from typing import Any

from ..layout.layers import LAYERS
from ..layout.transform import (
    bbox,
    direct_label,
    dummy_array_name,
    edge_boundary,
    indexed_labels,
    is_box,
    oriented_cell,
    oriented_name,
    placed_def_instances,
    require_gdstk,
    topbot_name,
)
from .connectivity import (
    METALS,
    MetalGraph,
    _assert,
    extract_connectivity,
    filtered_polygon_fingerprint,
    hierarchical_fingerprint,
    layer_polygons,
    polygon_fingerprint,
    process_fingerprint,
    verify_rules,
    verify_tap_rules,
)

__all__ = [
    "ACCESS_NFIN",
    "CELL_RATIO",
    "CONCURRENT_CELL_RATIO",
    "CORE_EDGE_CELL_NAMES",
    "EDGE_CELL_NAMES",
    "ORIENTATIONS",
    "PULLDOWN_NFIN",
    "PULLUP_NFIN",
    "PULLUP_RATIO",
    "RECOMMENDED_MIN_CELL_RATIO",
    "assert_close",
    "bitline_entries",
    "colgrp_name",
    "fin_grid_offset",
    "group_tie_bands",
    "io_block_specs",
    "iocol_pins",
    "is_bitline_m4_rail",
    "is_full_m2_rail",
    "load_cells",
    "port_io_name",
    "port_nodes",
    "sizing_report",
    "supply_rails",
    "tie_bands",
    "verify_cap_topology",
    "verify_decoder_physical",
    "verify_dummy_topology",
    "verify_edge_gds",
    "verify_gds",
    "verify_iocolumn_gds",
    "verify_strap_topology",
    "verify_tap_gds",
    "verify_tap_topology",
    "verify_topology",
    "wlb_landing_tracks",
]


PULLDOWN_NFIN = 2
ACCESS_NFIN = 2
PULLUP_NFIN = 1
CELL_RATIO = PULLDOWN_NFIN / ACCESS_NFIN
CONCURRENT_CELL_RATIO = PULLDOWN_NFIN / (2 * ACCESS_NFIN)
PULLUP_RATIO = PULLUP_NFIN / ACCESS_NFIN
RECOMMENDED_MIN_CELL_RATIO = 1.5


def sizing_report() -> list[str]:
    """The cell's strength ratios, derived from the sizing constants.

    Every device shares one gate pitch, so fin counts alone fix these ratios;
    verify_topology has already asserted the extracted GDS matches the
    constants, so reporting from the constants reports the layout.
    """
    lines = [
        f"pull-down nfin={PULLDOWN_NFIN}, access nfin={ACCESS_NFIN}, "
        f"pull-up nfin={PULLUP_NFIN} (shared L, one gate pitch)",
        f"cell ratio (pull-down/access)      = {CELL_RATIO:.2f}",
        f"  with both ports selected         = {CONCURRENT_CELL_RATIO:.2f}",
        f"pull-up ratio (pull-up/access)     = {PULLUP_RATIO:.2f}",
    ]
    if CELL_RATIO < RECOMMENDED_MIN_CELL_RATIO:
        lines.append(
            f"WARNING: cell ratio {CELL_RATIO:.2f} is below the "
            f"{RECOMMENDED_MIN_CELL_RATIO:.2f} usually recommended for a "
            f"single-port read, and both ports selected halves it to "
            f"{CONCURRENT_CELL_RATIO:.2f}.  docs/asap7_8t_bitcell.md declares "
            f"same-address read/read legal, so this is the sizing that "
            f"contract rests on.  Measure it: "
            f"ctest -R asap7_8t_stability_check"
        )
    return lines


CELL_NAME = "sram_cell_8t"
VARIANT_B_NAME = CELL_NAME + "_b"
VARIANT_B_END_NAME = CELL_NAME + "_b_end"
TAP_CELL_NAME = "tapcell_sram_8t"
STRAP_CELL_NAME = "strapcell_sram_8t"

CORE_EDGE_CELL_NAMES = (
    "dummy_cell_8t",
    "sram_cell_8t_col_cap",
    "sram_cell_8t_row_cap",
    "sram_cell_8t_corner",
)
ORIENTATIONS = ((False, False), (True, False), (False, True), (True, True))

EDGE_CELL_NAMES = tuple(dict.fromkeys(
    CORE_EDGE_CELL_NAMES
    + tuple(
        oriented_name(base, mx, my)
        for base in ("dummy_vertical_8t", "sram_cell_8t_corner")
        for mx, my in ORIENTATIONS
    )
    + tuple(topbot_name(mx, my) for mx, my in ORIENTATIONS)
    + ("FILLER_BLANK_8t", "FILLER_cgedge_8t")
))

WLB_LANDINGS = {False: (-0.132, 0.396), True: (0.012, 0.300)}
MARKER = (0.000, -0.162, 0.108, 0.432)
TAP_SLOTS = 2
TAP_MARKER = (0.000, -0.162, TAP_SLOTS * 0.108, 0.432)
TAP_IMPLANT_INSET = 0.027
TAP_TIE_OFFSET = (TAP_SLOTS - 1) * 0.108 / 2
TAP_BAND_INSET = 0.0135
STRAP_SPINE_X = {"vss!": 0.027 + TAP_TIE_OFFSET, "vdd!": 0.081 + TAP_TIE_OFFSET}
STRAP_M5_HALF = 0.012

WLA_M3 = (0.045, -0.162, 0.063, 0.432)
WLB_M5 = (0.048, -0.162, 0.072, 0.432)

WELL = LAYERS["NWELL"]["layer"]
FIN = LAYERS["FIN"]["layer"]
GATE = LAYERS["GATE"]["layer"]
GCUT = LAYERS["GATE_CUT"]["layer"]
ACTIVE = LAYERS["ACTIVE"]["layer"]
NSELECT = LAYERS["NSELECT"]["layer"]
PSELECT = LAYERS["PSELECT"]["layer"]
LIG = LAYERS["LIG"]["layer"]
LISD = LAYERS["LISD"]["layer"]
V0 = LAYERS["V0"]["layer"]
M1 = LAYERS["M1"]["layer"]
M2 = LAYERS["M2"]["layer"]
V1 = LAYERS["V1"]["layer"]
V2 = LAYERS["V2"]["layer"]
M3 = LAYERS["M3"]["layer"]
V3 = LAYERS["V3"]["layer"]
M4 = LAYERS["M4"]["layer"]
V4 = LAYERS["V4"]["layer"]
M5 = LAYERS["M5"]["layer"]
SDT = LAYERS["SDT"]["layer"]
SRAMDRC = LAYERS["SRAMDRC"]["layer"]
BOUNDARY = LAYERS["BOUNDARY"]["layer"]
SRAMVT = LAYERS["SRAMVT"]["layer"]


def _contains(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    """True when box `inner` sits inside box `outer`."""
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def is_full_m2_rail(poly: Any) -> bool:
    if poly.layer != M2 or poly.datatype != 0:
        return False
    x0, _y0, x1, _y1 = bbox(poly)
    return x1 - x0 >= 0.160


def is_bitline_m4_rail(poly: Any) -> bool:
    if poly.layer != M4 or poly.datatype != 0:
        return False
    x0, y0, x1, y1 = bbox(poly)
    return x0 in (-0.027, -0.038) and x1 - x0 >= 0.160 and (y0, y1) in {(-0.048, -0.024), (0.336, 0.360)}


def supply_rails(bitcell: Any) -> dict[str, list[tuple[float, float]]]:
    """Map vdd!/vss! to the y-extent of the full-width M2 rails they pin."""
    rails = [bbox(poly) for poly in bitcell.polygons if is_full_m2_rail(poly)]
    nets: dict[str, list[tuple[float, float]]] = {}
    for label in bitcell.labels:
        if label.layer != M2 or label.text not in ("vdd!", "vss!"):
            continue
        x, y = float(label.origin[0]), float(label.origin[1])
        hits = [r for r in rails if r[0] <= x <= r[2] and r[1] <= y <= r[3]]
        _assert(len(hits) == 1, f"{label.text} pin is not on exactly one M2 rail")
        nets.setdefault(label.text, []).append((hits[0][1], hits[0][3]))
    _assert(set(nets) == {"vdd!", "vss!"}, "bitcell is missing an M2 supply pin")
    return nets


def tie_bands(bitcell: Any) -> list[tuple[float, float, str]]:
    """Regions the tap must tie, as (y0, y1, net), ordered bottom to top."""
    well = bbox(next(p for p in bitcell.polygons if p.layer == WELL))
    bands = [
        (bbox(p)[1], bbox(p)[3], "vss!")
        for p in bitcell.polygons if p.layer == NSELECT
    ]
    bands.append((well[1], well[3], "vdd!"))
    return sorted(bands)


def group_tie_bands(
    bands: list[tuple[float, float, str]],
) -> list[tuple[str, list[tuple[float, float, str]]]]:
    """Collapse adjacent same-net bands so each takes one shared M1 strap."""
    groups: list[tuple[str, list[tuple[float, float, str]]]] = []
    for band in bands:
        if groups and groups[-1][0] == band[2]:
            groups[-1][1].append(band)
        else:
            groups.append((band[2], [band]))
    return groups


def wlb_landing_tracks(cell: Any) -> set[float]:
    """y of every M4 shape that is not a full-width bitline: the WLB landings."""
    return {
        round((bbox(p)[1] + bbox(p)[3]) / 2, 4)
        for p in layer_polygons(cell, M4)
        if not is_bitline_m4_rail(p)
    }


def sizing_report(
    pulldown_nfin: int = 2,
    access_nfin: int = 2,
    pullup_nfin: int = 1,
    recommended_min_cell_ratio: float = 1.5,
) -> list[str]:
    """The cell's strength ratios, derived from fin count sizing constants."""
    cell_ratio = pulldown_nfin / access_nfin
    concurrent_cell_ratio = pulldown_nfin / (2 * access_nfin)
    pullup_ratio = pullup_nfin / access_nfin
    lines = [
        f"pull-down nfin={pulldown_nfin}, access nfin={access_nfin}, "
        f"pull-up nfin={pullup_nfin} (shared L, one gate pitch)",
        f"cell ratio (pull-down/access)      = {cell_ratio:.2f}",
        f"  with both ports selected         = {concurrent_cell_ratio:.2f}",
        f"pull-up ratio (pull-up/access)     = {pullup_ratio:.2f}",
    ]
    if cell_ratio < recommended_min_cell_ratio:
        lines.append(
            f"WARNING: cell ratio {cell_ratio:.2f} is below the "
            f"{recommended_min_cell_ratio:.2f} usually recommended for a "
            f"single-port read, and both ports selected halves it to "
            f"{concurrent_cell_ratio:.2f}.  docs/asap7_8t_bitcell.md declares "
            f"same-address read/read legal, so this is the sizing that "
            f"contract rests on.  Measure it: "
            f"ctest -R asap7_8t_stability_check"
        )
    return lines


def verify_topology(
    cell: Any,
    access_nfin: int = 2,
    pulldown_nfin: int = 2,
    pullup_nfin: int = 1,
) -> None:
    """Audit the complete electrical graph and device sizing of an 8T bitcell."""
    devices, labels = extract_connectivity(cell, expected_channels=8)
    expected_pins = {"WLA", "WLB", "BLA", "BLAN", "BLB", "BLBN", "vdd!", "vss!"}
    _assert(
        expected_pins <= set(labels),
        f"missing pins: {sorted(expected_pins - set(labels))}",
    )
    for pin in ("WLA", "WLB", "BLA", "BLAN", "BLB", "BLBN"):
        _assert(len(labels[pin]) == 1, f"{pin} must label exactly one electrical net")
    signal_roots = {
        next(iter(labels[p]))
        for p in ("WLA", "WLB", "BLA", "BLAN", "BLB", "BLBN")
    }
    _assert(len(signal_roots) == 6, "one or more independent signal pins are shorted")
    _assert(
        not (signal_roots & (labels["vdd!"] | labels["vss!"])),
        "a signal pin is shorted to a supply rail",
    )

    def access_map(wordline: str, bitlines: tuple[str, str]) -> dict[str, int]:
        wl = next(iter(labels[wordline]))
        selected = [device for device in devices if device["gate"] == wl]
        _assert(len(selected) == 2, f"{wordline} must drive exactly two access devices")
        result: dict[str, int] = {}
        for device in selected:
            terms = set(device["terminals"])
            _assert(not device["pmos"], f"{wordline} access device is not NMOS")
            _assert(
                device["nfin"] == access_nfin,
                f"{wordline} access device is {device['nfin']}-fin, not nfin={access_nfin}",
            )
            matches = [pin for pin in bitlines if next(iter(labels[pin])) in terms]
            _assert(
                len(matches) == 1,
                f"{wordline} access device is not tied to one {bitlines} bitline",
            )
            bitline_root = next(iter(labels[matches[0]]))
            storage = terms - {bitline_root}
            _assert(len(storage) == 1, "access device has ambiguous storage terminal")
            result[matches[0]] = storage.pop()
        _assert(set(result) == set(bitlines), f"{wordline} does not cover both bitlines")
        return result

    port_a = access_map("WLA", ("BLA", "BLAN"))
    port_b = access_map("WLB", ("BLB", "BLBN"))
    _assert(port_a["BLA"] == port_b["BLB"], "BLA and BLB do not access the same node")
    _assert(
        port_a["BLAN"] == port_b["BLBN"],
        "BLAN and BLBN do not access the same node",
    )
    _assert(port_a["BLA"] != port_a["BLAN"], "Q and QB are shorted")

    q, qb = port_a["BLA"], port_a["BLAN"]
    supply_vdd = labels["vdd!"]
    supply_vss = labels["vss!"]
    for gate_root, drain_root in ((q, qb), (qb, q)):
        inverter = [device for device in devices if device["gate"] == gate_root]
        _assert(len(inverter) == 2, "storage node does not drive one CMOS inverter")
        _assert(
            {device["pmos"] for device in inverter} == {False, True},
            "storage inverter is not one PMOS plus one NMOS",
        )
        for device in inverter:
            terms = set(device["terminals"])
            expected_fins = pullup_nfin if device["pmos"] else pulldown_nfin
            _assert(
                device["nfin"] == expected_fins,
                f"inverter {'pull-up' if device['pmos'] else 'pull-down'} is "
                f"{device['nfin']}-fin, not nfin={expected_fins}",
            )
            _assert(drain_root in terms, "cross-coupled inverter drain is disconnected")
            other = terms - {drain_root}
            expected_supply = supply_vdd if device["pmos"] else supply_vss
            _assert(
                len(other) == 1 and next(iter(other)) in expected_supply,
                "inverter source is not connected to its supply",
            )

    _assert(sum(device["pmos"] for device in devices) == 2, "expected 2 PMOS devices")
    _assert(sum(not device["pmos"] for device in devices) == 6, "expected 6 NMOS devices")


def verify_dummy_topology(cell: Any) -> None:
    """Verify the forced-state dummy cell topology."""
    devices, labels = extract_connectivity(cell, expected_channels=8)
    bitlines = ("BLA", "BLAN", "BLB", "BLBN")
    _assert(set(bitlines) <= set(labels), "dummy is missing bitline pins")
    _assert(
        "WLA" not in labels and "WLB" not in labels,
        "dummy must not expose wordline pins",
    )
    _assert(
        "vdd!" in labels and "vss!" in labels,
        "dummy is missing forced storage/supply labels",
    )

    pin_roots = {pin: next(iter(labels[pin])) for pin in bitlines}
    _assert(len(set(pin_roots.values())) == 4, "dummy bitlines are shorted")
    expected_storage = {
        "BLA": labels["vss!"],
        "BLB": labels["vss!"],
        "BLAN": labels["vdd!"],
        "BLBN": labels["vdd!"],
    }
    for pin, bitline_root in pin_roots.items():
        access = [device for device in devices if bitline_root in device["terminals"]]
        _assert(
            len(access) == 1 and not access[0]["pmos"],
            f"{pin} is not attached to exactly one access NMOS",
        )
        device = access[0]
        _assert(
            device["gate"] in labels["vss!"],
            f"{pin} access gate is not forced inactive",
        )
        other = set(device["terminals"]) - {bitline_root}
        _assert(
            len(other) == 1 and next(iter(other)) in expected_storage[pin],
            f"{pin} is attached to the wrong forced storage node",
        )

    _assert(sum(device["pmos"] for device in devices) == 2, "dummy expected 2 PMOS devices")
    _assert(sum(not device["pmos"] for device in devices) == 6, "dummy expected 6 NMOS devices")
    for device in devices:
        expected_fins = 1 if device["pmos"] else 2
        _assert(
            device["nfin"] == expected_fins,
            f"dummy {'PMOS' if device['pmos'] else 'NMOS'} is {device['nfin']}-fin, not nfin={expected_fins}",
        )


def verify_cap_topology(cell: Any, cap_kind: str) -> None:
    """Verify that a cap cell is device-free and maintains correct pin continuity."""
    devices, labels = extract_connectivity(cell, expected_channels=0)
    _assert(not devices, f"{cell.name} must be electrically device-free")
    _assert(not layer_polygons(cell, ACTIVE), f"{cell.name} must not contain ACTIVE")
    bitlines = {"BLA", "BLAN", "BLB", "BLBN"}

    if cap_kind == "col":
        _assert(bitlines <= set(labels), "column cap is missing bitline pins")
        _assert("WLA" not in labels and "WLB" not in labels, "column cap unexpectedly exposes wordlines")
        roots = [next(iter(labels[pin])) for pin in sorted(bitlines)]
        _assert(len(set(roots)) == 4, "column-cap bitlines are shorted")
    elif cap_kind == "row":
        _assert({"WLA", "WLB"} <= set(labels), "row cap is missing wordline pins")
        _assert(not (bitlines & set(labels)), "row cap unexpectedly exposes bitlines")
        _assert(next(iter(labels["WLA"])) != next(iter(labels["WLB"])), "row-cap wordlines are shorted")
    elif cap_kind == "corner":
        _assert(not ({"WLA", "WLB"} | bitlines) & set(labels), "corner cap unexpectedly exposes signal pins")
    else:
        raise ValueError(f"unknown cap kind {cap_kind}")

    _assert("vss!" in labels, f"{cell.name} is missing ground continuity")


def verify_tap_topology(cell: Any, bitcell: Any) -> None:
    """The tap ties both polarities, holds no device, and passes bitlines."""
    devices, labels = extract_connectivity(cell, expected_channels=0)
    _assert(not devices, "tap cell must be device-free")

    well = [bbox(p) for p in layer_polygons(cell, WELL)]
    nsel = [bbox(p) for p in layer_polygons(cell, NSELECT)]
    psel = sorted(bbox(p) for p in layer_polygons(cell, PSELECT))
    _assert(len(well) == 1, "tap must have exactly one n-well band")
    _assert(
        nsel == [(
            well[0][0] + TAP_IMPLANT_INSET,
            well[0][1],
            well[0][2] - TAP_IMPLANT_INSET,
            well[0][3],
        )],
        "tap n+ implant must span the n-well band, 27 nm in from each seam",
    )
    for box in psel:
        _assert(
            abs(box[0] - (TAP_MARKER[0] + TAP_IMPLANT_INSET)) < 1e-9
            and abs(box[2] - (TAP_MARKER[2] - TAP_IMPLANT_INSET)) < 1e-9,
            "tap p+ implant must stop 27 nm short of each seam",
        )
        _assert(box[2] - box[0] >= 0.108 - 1e-9, "tap p+ implant is under the 108 nm minimum width")
    _assert(bool(psel), "tap has no p+ substrate tie")
    for box in psel:
        _assert(
            box[3] <= well[0][1] or well[0][3] <= box[1],
            "tap p+ implant overlaps the n-well",
        )

    implants = nsel + psel
    actives = sorted(bbox(p) for p in layer_polygons(cell, ACTIVE))
    _assert(len(actives) == len(implants), "tap does not place one tie diffusion per implant band")
    gates = [bbox(p) for p in layer_polygons(cell, GATE)]
    contacts = [bbox(p) for p in layer_polygons(cell, LISD)]
    vias = [bbox(p) for p in layer_polygons(cell, V0)]
    straps = [bbox(p) for p in layer_polygons(cell, M1)]
    for active in actives:
        host = [box for box in implants if _contains(box, active)]
        _assert(len(host) == 1, "tie diffusion is not inside one implant band")
        for gate in gates:
            _assert(active[2] <= gate[0] or gate[2] <= active[0], "tie diffusion crosses a gate column")
        contact = [box for box in contacts if _contains(box, active)]
        _assert(len(contact) == 1, "tie diffusion has no local-interconnect tab")
        via = [box for box in vias if _contains(contact[0], box)]
        _assert(len(via) == 1, "tie contact has no V0")
        _assert(any(_contains(strap, via[0]) for strap in straps), "tie V0 is not covered by an M1 strap")

    rails = supply_rails(cell)
    v1_boxes = [bbox(p) for p in layer_polygons(cell, V1)]
    for net, group in group_tie_bands(tie_bands(bitcell)):
        bottom = group[0][0] + TAP_BAND_INSET
        top = group[-1][1] - TAP_BAND_INSET
        strap = [m for m in straps if abs(m[1] - bottom) < 1e-9 and abs(m[3] - top) < 1e-9]
        _assert(len(strap) == 1, f"tap has no single {net} strap")
        landed = [v for v in v1_boxes if _contains(strap[0], v)]
        _assert(bool(landed), f"tap {net} strap has no V1 to a supply rail")
        for via in landed:
            _assert(
                any(rail[0] <= via[1] and via[3] <= rail[1] for rail in rails[net]),
                f"tap {net} V1 does not land on a {net} rail",
            )

    _assert(labels["vdd!"].isdisjoint(labels["vss!"]), "tap shorts VDD to VSS")

    def rail_predicate(poly: Any) -> bool:
        return is_full_m2_rail(poly) or is_bitline_m4_rail(poly)

    stretch = (TAP_SLOTS - 1) * (MARKER[2] - MARKER[0])
    expected_rails = sorted(
        (p.layer, round(b[0], 6), round(b[1], 6), round(b[2] + stretch, 6), round(b[3], 6))
        for p in bitcell.polygons if rail_predicate(p) for b in [bbox(p)]
    )
    drawn_rails = sorted(
        (p.layer, round(b[0], 6), round(b[1], 6), round(b[2], 6), round(b[3], 6))
        for p in cell.polygons if rail_predicate(p) for b in [bbox(p)]
    )
    _assert(drawn_rails == expected_rails, "tap rails do not abut the bitcell's bitlines and supplies")


def verify_strap_topology(strap: Any, tap: Any) -> None:
    """The strap is the tap plus a working M2-to-M5 climb per supply."""
    tap_shapes = {(p.layer, p.datatype, bbox(p)) for p in tap.polygons}
    strap_shapes = {(p.layer, p.datatype, bbox(p)) for p in strap.polygons}
    missing = tap_shapes - strap_shapes
    _assert(not missing, f"strap drops {len(missing)} tap shapes, e.g. {sorted(missing)[:2]}")

    x0, y0, x1, y1 = TAP_MARKER
    spines = sorted(bbox(p) for p in layer_polygons(strap, M5))
    _assert(len(spines) == len(STRAP_SPINE_X), f"expected {len(STRAP_SPINE_X)} M5 spines, found {len(spines)}")
    for spine, (net, centre) in zip(spines, sorted(STRAP_SPINE_X.items(), key=lambda kv: kv[1])):
        _assert(
            abs(spine[0] - (centre - STRAP_M5_HALF)) < 1e-9
            and abs(spine[2] - (centre + STRAP_M5_HALF)) < 1e-9,
            f"{net} M5 spine is not centred on {centre}",
        )
        _assert(
            abs(spine[1] - y0) < 1e-9 and abs(spine[3] - y1) < 1e-9,
            f"{net} M5 spine does not span the full row pitch",
        )
        _assert(x0 <= spine[0] and spine[2] <= x1, f"{net} M5 spine leaves the tap slot")
    _assert(spines[1][0] - spines[0][2] >= 0.018 - 1e-9, "the two M5 spines are closer than 18 nm")

    for poly in layer_polygons(strap, M3):
        box = bbox(poly)
        _assert(
            box[0] >= TAP_MARKER[0] + 0.027 - 1e-9 and box[2] <= TAP_MARKER[2] - 0.027 + 1e-9,
            f"strap M3 at {box} intrudes on the neighbour overhang",
        )

    bitline_y = [
        (bbox(p)[1], bbox(p)[3])
        for p in layer_polygons(strap, M4) if is_bitline_m4_rail(p)
    ]
    _assert(len(bitline_y) == 2, "strap lost a port-B bitline M4 rail")
    for poly in layer_polygons(strap, M4):
        if is_bitline_m4_rail(poly):
            continue
        box = bbox(poly)
        for low, high in bitline_y:
            _assert(box[3] <= low or high <= box[1], f"strap M4 climb at {box} overlaps a bitline rail")

    _devices, labels = extract_connectivity(strap, expected_channels=0)
    _assert(len(labels["vdd!"]) == 1, "strap VDD rail and M5 spine are not connected")
    _assert(len(labels["vss!"]) == 2, "strap VSS rail and M5 spine are not connected")
    _assert(labels["vdd!"].isdisjoint(labels["vss!"]), "strap shorts VDD to VSS")


def verify_gds(path: Path) -> str:
    """Verify bitcell GDS and return combined SHA-256 fingerprint."""
    gdstk = require_gdstk()
    lib = gdstk.read_gds(str(path))
    cells = {candidate.name: candidate for candidate in lib.cells}
    _assert(
        set(cells) == {CELL_NAME, VARIANT_B_NAME, VARIANT_B_END_NAME},
        f"{path} must contain all three bitcell masters, found {sorted(cells)}",
    )
    for cell in cells.values():
        verify_rules(cell)
        verify_topology(cell)
    tracks = {name: wlb_landing_tracks(cell) for name, cell in cells.items()}
    _assert(
        tracks[CELL_NAME] == set(WLB_LANDINGS[False])
        and tracks[VARIANT_B_NAME] == tracks[VARIANT_B_END_NAME] == set(WLB_LANDINGS[True]),
        f"WLB landings are not on their tracks: {tracks}",
    )
    _assert(not tracks[CELL_NAME] & tracks[VARIANT_B_NAME], "the variants' WLB landings share a track")
    return hashlib.sha256("".join(polygon_fingerprint(cells[n]) for n in sorted(cells)).encode()).hexdigest()


def verify_tap_gds(path: Path, bitcell_path: Path) -> dict[str, str]:
    """Verify tap and strap cells against the bitcell."""
    gdstk = require_gdstk()
    lib = gdstk.read_gds(str(path))
    cells = {cell.name: cell for cell in lib.cells}
    expected = {TAP_CELL_NAME, STRAP_CELL_NAME}
    _assert(set(cells) == expected, f"tap library cells are {sorted(cells)}, expected {sorted(expected)}")
    bit_lib = gdstk.read_gds(str(bitcell_path))
    bitcell = next((c for c in bit_lib.cells if c.name == CELL_NAME), None)
    if bitcell is None:
        raise ValueError(f"{CELL_NAME!r} not found in {bitcell_path}")
    for name in sorted(expected):
        verify_tap_rules(cells[name])
        verify_tap_topology(cells[name], bitcell)
    verify_strap_topology(cells[STRAP_CELL_NAME], cells[TAP_CELL_NAME])
    return {name: polygon_fingerprint(cells[name]) for name in sorted(expected)}


def verify_edge_gds(path: Path, row_counts: tuple[int, ...] = (64,)) -> dict[str, str]:
    """Verify all 8T edge master cells and parameterized dummy arrays."""
    gdstk = require_gdstk()
    lib = gdstk.read_gds(str(path))
    cells = {cell.name: cell for cell in lib.cells}
    expected = set(EDGE_CELL_NAMES) | {
        dummy_array_name(count, mirror_x=mx, mirror_y=my)
        for count in row_counts for mx, my in ORIENTATIONS
    }
    _assert(set(cells) == expected, f"edge library cells are {sorted(cells)}, expected {sorted(expected)}")

    for name in CORE_EDGE_CELL_NAMES:
        verify_rules(cells[name])
    verify_dummy_topology(cells["dummy_cell_8t"])
    verify_cap_topology(cells["sram_cell_8t_col_cap"], "col")
    verify_cap_topology(cells["sram_cell_8t_row_cap"], "row")
    verify_cap_topology(cells["sram_cell_8t_corner"], "corner")

    frames = {name: process_fingerprint(cells[name]) for name in CORE_EDGE_CELL_NAMES}
    _assert(len(set(frames.values())) == 1, "dummy/cap process frames do not match at abutment boundaries")

    col_cap = cells["sram_cell_8t_col_cap"]
    row_cap = cells["sram_cell_8t_row_cap"]
    corner = cells["sram_cell_8t_corner"]

    def rail_predicate(poly: Any) -> bool:
        return is_full_m2_rail(poly) or is_bitline_m4_rail(poly)

    _assert(
        filtered_polygon_fingerprint(col_cap, rail_predicate)
        == filtered_polygon_fingerprint(corner, rail_predicate),
        "column and corner cap rails do not abut identically",
    )

    conductor_layers = {LIG, LISD, V0, M1, V1, M2, V2, M3, V3, M4, V4, M5}

    def wordline_predicate(poly: Any) -> bool:
        return poly.layer in conductor_layers and not rail_predicate(poly)

    _assert(
        filtered_polygon_fingerprint(row_cap, wordline_predicate)
        == filtered_polygon_fingerprint(corner, wordline_predicate),
        "row and corner cap wordline routes do not abut identically",
    )

    col_rail_boxes = {bbox(poly) for poly in col_cap.polygons if rail_predicate(poly)}
    expected_rails = {
        (-0.027, 0.027, 0.135, 0.045),
        (-0.026, 0.0745, 0.136, 0.0925),
        (-0.027, 0.126, 0.135, 0.144),
        (-0.026, 0.1775, 0.136, 0.1955),
        (-0.027, 0.225, 0.135, 0.243),
        (-0.038, -0.048, 0.135, -0.024),
        (-0.027, 0.336, 0.135, 0.360),
    }
    _assert(col_rail_boxes == expected_rails, "column-cap rails do not span the complete mirrored cell pitch")
    _assert(any(is_box(poly, WLA_M3) for poly in layer_polygons(row_cap, M3)), "row cap is missing full-height WLA M3 trunk")
    _assert(any(is_box(poly, WLB_M5) for poly in layer_polygons(row_cap, M5)), "row cap is missing full-height WLB M5 trunk")

    row_cap_b = oriented_cell(cells[oriented_name("dummy_vertical_8t", True, False)], row_cap.name, True, False)
    corner_b = oriented_cell(cells[oriented_name("sram_cell_8t_corner", True, False)], corner.name, True, False)
    _assert(
        filtered_polygon_fingerprint(row_cap_b, wordline_predicate)
        == filtered_polygon_fingerprint(corner_b, wordline_predicate),
        "variant row and corner cap wordline routes do not abut identically",
    )
    _assert(
        wlb_landing_tracks(row_cap) == set(WLB_LANDINGS[False])
        and wlb_landing_tracks(row_cap_b) == set(WLB_LANDINGS[True]),
        "the x-mirrored edge masters do not take variant B's WLB landings",
    )
    for mx, my in ORIENTATIONS:
        for name, kind, canonical in (
            (oriented_name("dummy_vertical_8t", mx, my), "row", row_cap_b if mx else row_cap),
            (oriented_name("sram_cell_8t_corner", mx, my), "corner", corner_b if mx else corner),
            (topbot_name(mx, my), "col", col_cap),
        ):
            cell = cells[name]
            _assert(not cell.references, f"{name}: oriented master must be materialized")
            restored = oriented_cell(cell, canonical.name, mx, my)
            _assert(
                polygon_fingerprint(restored) == polygon_fingerprint(canonical),
                f"{name}: geometry/pins do not match the canonical orientation",
            )
            verify_rules(restored)
            verify_cap_topology(cell, kind)

    blank = cells["FILLER_BLANK_8t"]
    width, height = MARKER[2] - MARKER[0], MARKER[3] - MARKER[1]
    _assert(edge_boundary(blank) == (0, MARKER[1], width / 2, MARKER[3]), "blank filler is not a half-width 8T tile")
    _assert(
        {p.layer for p in blank.polygons} == {FIN, GATE, BOUNDARY}
        and not blank.labels
        and not blank.references,
        "blank filler contains unexpected devices, conductors, or hierarchy",
    )
    _assert(len(layer_polygons(blank, GATE)) == 1, "blank filler needs one gate column")
    _assert(
        [bbox(p)[1::2] for p in layer_polygons(blank, FIN)]
        == [bbox(p)[1::2] for p in layer_polygons(col_cap, FIN)],
        "blank filler FIN grid does not match 8T",
    )
    filler = cells["FILLER_cgedge_8t"]
    _assert(edge_boundary(filler) == (0, 0, width, round(4 * height, 7)), "column-group filler is not four 8T rows high")
    _assert(len(filler.references) == 8, "column-group filler needs eight half-width tiles")
    for index, ref in enumerate(filler.references):
        row, col = divmod(index, 2)
        expected_y = (row + 1) * height + MARKER[1] if row % 2 else row * height - MARKER[1]
        _assert(
            ref.cell_name == blank.name
            and not ref.rotation
            and bool(ref.x_reflection) == bool(row % 2)
            and all(abs(a - b) < 1e-7 for a, b in zip(ref.origin, (col * width / 2, expected_y))),
            "column-group filler has a misplaced blank tile",
        )
    for count in row_counts:
        for mx, my in ORIENTATIONS:
            row = cells[dummy_array_name(count, mirror_x=mx, mirror_y=my)]
            _assert(edge_boundary(row) == (0, 0, round(count * width, 7), round(height, 7)), f"{row.name}: wrong row-count boundary")
            _assert(len(row.references) == count, f"{row.name}: wrong dummy count")
            for index, ref in enumerate(row.references):
                slot = count - 1 - index if mx else index
                _assert(
                    ref.cell_name == oriented_name("dummy_vertical_8t", bool(index % 2) != mx, my)
                    and not ref.rotation
                    and not ref.x_reflection
                    and all(abs(a - b) < 1e-7 for a, b in zip(ref.origin, (slot * width - MARKER[0], -MARKER[1]))),
                    f"{row.name}: wrong dummy slot/orientation",
                )
    return {
        name: polygon_fingerprint(cell.copy(name + "_flat").flatten())
        for name, cell in sorted(cells.items())
    }


def load_cells(path: Path) -> tuple[Any, dict[str, Any]]:
    """Read a GDS library and return (library, {name: cell})."""
    gdstk = require_gdstk()
    library = gdstk.read_gds(str(path))
    return library, {cell.name: cell for cell in library.cells}


def assert_close(actual: float, expected: float, message: str) -> None:
    if abs(actual - expected) > 1e-6:
        raise RuntimeError(f"{message}: {actual} != {expected}")


def port_io_name(port: str) -> str:
    """``A``, ``B``, or ``B2``: port B's two-sided block, shared by a pair of banks."""
    return f"iocol_sram_8t_{port.lower()}"


def colgrp_name(wordlines: int, kind: str = "") -> str:
    """``colgrp_x{N}x4_sram_8t``; `kind` ``half`` or ``pair`` for the shared-port-B tiles."""
    return f"colgrp{'_' + kind if kind else ''}_x{wordlines}x4_sram_8t"


def iocol_pins(port: str, two_sided: bool = False, mux_rows: int = 4) -> list[str]:
    """Pin order of ``iocol_sram_8t_{a,b,b2}``, as SpiceGenerator instantiates it."""
    leaves = 2 * mux_rows if two_sided else mux_rows
    return [
        *(f"BL_{port}[{i}]" for i in range(leaves)),
        *(f"BLN_{port}[{i}]" for i in range(leaves)),
        *(f"ysel_{port}[{i}]" for i in range(leaves)),
        *(f"yseln_{port}[{i}]" for i in range(leaves)),
        f"blprechn_{port}", *([f"blprechn_{port}R"] if two_sided else []),
        f"sae_{port}", f"wrena_{port}", f"wrenan_{port}",
        f"oe_out_{port}", f"oeb_out_{port}", f"D{port}", f"Q{port}", "VDD", "VSS",
    ]


def bitline_entries(bitcell: Any) -> dict[str, tuple[float, float]]:
    """Where the bitcell puts each port's bitline pair, in nm from the bottom of its boundary.

    The pair's metal is the bar under its label: port A's on M2, port B's on
    M4. This is what the block's leaves are built to.
    """
    gdstk = require_gdstk()
    _, y0, _, _ = edge_boundary(bitcell)
    entries = {}
    for port, (true, comp, layer) in {"A": ("BLA", "BLAN", M2), "B": ("BLB", "BLBN", M4)}.items():
        ys = []
        for name in (true, comp):
            label = direct_label(bitcell, name)
            x, y = map(float, label.origin)
            bars = [
                poly for poly in bitcell.polygons
                if poly.layer == layer and gdstk.inside([(x, y)], [poly])[0]
            ]
            if len(bars) != 1:
                raise RuntimeError(f"{bitcell.name}: {name} does not sit on one {layer} bar")
            by0, by1 = bbox(bars[0])[1], bbox(bars[0])[3]
            ys.append(round(((float(by0) + float(by1)) / 2 - y0) * 1000, 1))
        entries[port] = (ys[0], ys[1])
    return entries


def fin_grid_offset(
    bitcell: Any, fin_pitch: float = 27.0, leaf_fin_phase: float = 13.5
) -> float:
    """How far up the bitcell's row a chipforge block sits for its fins to land on the bitcell's grid, nm."""
    _, y0, _, _ = edge_boundary(bitcell)
    phases = {
        round(((bbox(poly)[1] + bbox(poly)[3]) / 2 - y0) * 1000 % fin_pitch, 1) % fin_pitch
        for poly in bitcell.polygons if poly.layer == FIN
    }
    if len(phases) != 1:
        raise RuntimeError(f"{bitcell.name}: fins are not on one {fin_pitch} nm grid: {sorted(phases)}")
    return (leaf_fin_phase - phases.pop()) % fin_pitch


def io_block_specs(bitcell: Any, mux_rows: int = 4) -> dict[str, Any]:
    """One `IoColumnSpec` per port: the block at the bitcell's bitline heights and on its fin grid, one sense phase."""
    from ..devices import BitlineMuxSpec, IoColumnSpec

    _, y0, _, y1 = edge_boundary(bitcell)
    entries = bitline_entries(bitcell)
    offset = fin_grid_offset(bitcell)
    specs = {}
    for port, layer in (("A", "M2"), ("B", "M4")):
        mux = BitlineMuxSpec(
            rows=2, selects=mux_rows, bitline_entry=entries[port], bitline_layer=layer,
            grid_offset=offset,
        )
        if abs(mux.height / 1000 - (y1 - y0)) > 1e-6:
            raise RuntimeError("the block's two rows do not add up to the bitcell's row")
        specs[port] = IoColumnSpec(mux=mux, one_sense_phase=True)
    specs["B2"] = IoColumnSpec(mux=specs["B"].mux, one_sense_phase=True, two_sided=True)
    return specs


def verify_iocolumn_gds(
    path: Path, wordline_counts: list[int], spice: Path | None = None
) -> dict[str, str]:
    """Verify dual-port 8T IO columns and return SHA-256 fingerprint per cell."""
    mux_rows = 4
    seam_tie_inset = 0.012
    via_half = 0.009
    m2_end_cap = 0.006

    library, cells = load_cells(path)
    bitcell = cells.get("sram_cell_8t")
    if bitcell is None:
        raise RuntimeError(f"{path}: missing sram_cell_8t")
    specs = io_block_specs(bitcell)
    expected = {
        "sram_cell_8t", "sram_cell_8t_b", "sram_cell_8t_b_end", "FILLER_BLANK_8t", "FILLER_cgedge_8t",
        port_io_name("A"), port_io_name("B"), port_io_name("B2"), "col_cap_x4_sram_8t",
    }
    expected.update(topbot_name(False, my) for my in (False, True))
    for count in wordline_counts:
        expected.update({
            f"sramcol_x{count}_sram_8t",
            f"array_x{count}x4_sram_8t",
            colgrp_name(count), colgrp_name(count, "half"), colgrp_name(count, "pair"),
        })
    blocks = {spec.cell_name for spec in specs.values()}
    block_cells = {
        name for name in cells
        if name.startswith(("iocol_x", "blmux_", "sarow_", "wrdrv_", "outlatch_", "filler_fin", "tap_fin"))
    }
    if not blocks <= block_cells:
        raise RuntimeError(f"{path}: missing the IO blocks {sorted(blocks - block_cells)}")
    if set(cells) != expected | block_cells:
        raise RuntimeError(
            f"{path}: cell set mismatch; missing={sorted(expected - set(cells))}, "
            f"extra={sorted(set(cells) - expected - block_cells)}"
        )

    io_boxes = {}
    for port, layer in (("A", M2), ("B", M4)):
        io = cells[port_io_name(port)]
        spec = specs[port]
        if [reference.cell_name for reference in io.references] != [spec.cell_name]:
            raise RuntimeError(f"{io.name}: expected exactly its own block, {spec.cell_name}")
        (reference,) = io.references
        mirrored = port == "A"
        if bool(reference.x_reflection) != mirrored or bool(reference.rotation) != mirrored:
            raise RuntimeError(f"{io.name}: port A's block is mirrored in x, port B's as drawn")
        assert_close(float(reference.origin[1]), spec.mux.grid_offset / 1000, f"{io.name}: block fin-grid offset")
        io_boxes[port] = edge_boundary(io)
        assert_close(io_boxes[port][3] - io_boxes[port][1], spec.height / 1000, f"{io.name} height")
        assert_close(io_boxes[port][2] - io_boxes[port][0], spec.width / 1000, f"{io.name} width")
        face_x = io_boxes[port][2] if mirrored else io_boxes[port][0]
        entry = spec.mux.bitline_entry
        for prefix, y_leaf in (("BL", entry[0]), ("BLN", entry[1])):
            labels = indexed_labels(io, f"{prefix}_{port}")
            if set(labels) != set(range(mux_rows)):
                raise RuntimeError(f"{io.name}: incomplete {prefix}_{port} bus")
            for index, label in labels.items():
                if label.layer != layer:
                    raise RuntimeError(f"{io.name}: {label.text} is on the wrong layer")
                if abs(float(label.origin[0]) - face_x) > 0.0061:
                    raise RuntimeError(f"{io.name}: {label.text} is not at the array face")
                row = spec.mux.height / 1000
                want = (index + 1) * row - y_leaf / 1000 if index % 2 else index * row + y_leaf / 1000
                assert_close(float(label.origin[1]), want, f"{io.name}: {label.text} height")
        own = {poly.layer for poly in io.polygons} - {BOUNDARY, M1, V1, M2, V2, M3}
        if own:
            raise RuntimeError(f"{io.name}: unexpected routing layers {sorted(own)}")
        straps = [bbox(q) for q in io.polygons if q.layer == M3]
        if len(straps) != 2 or any(
            abs((y1 - y0) - (io_boxes[port][3] - io_boxes[port][1])) > 1e-6
            for _, y0, _, y1 in straps
        ):
            raise RuntimeError(f"{io.name}: expected a full-height VDD and VSS strap")
        names = [label.text for label in io.labels]
        required = set(iocol_pins(port))
        if not required <= set(names):
            raise RuntimeError(f"{io.name}: missing IO pins {sorted(required - set(names))}")
        import re
        stale = [n for n in names if re.match(r"(YSEL|BLPRECH|SAPRECHN|WRENA|D_|Q_|yselt|yselb|blprecht|blprechb)", n)]
        if stale:
            raise RuntimeError(f"{io.name}: wrapper-era pins leaked: {stale}")
        if any(n == "VDD" and label.layer != M1 for n, label in zip(names, io.labels)):
            raise RuntimeError(f"{io.name}: a supply label is not on an M1 rail")

    io_b2 = cells[port_io_name("B2")]
    b2_box = edge_boundary(io_b2)
    assert_close(b2_box[2] - b2_box[0], specs["B2"].width / 1000, f"{io_b2.name} width")
    for prefix in ("BL_B", "BLN_B"):
        labels = indexed_labels(io_b2, prefix)
        if set(labels) != set(range(2 * mux_rows)):
            raise RuntimeError(f"{io_b2.name}: incomplete {prefix} bus")
        for index in range(mux_rows):
            left, right = labels[index], labels[index + mux_rows]
            assert_close(float(left.origin[0]), b2_box[0] + 0.006, f"{left.text} at the left face")
            assert_close(float(right.origin[0]), b2_box[2] - 0.006, f"{right.text} at the right face")
            assert_close(float(right.origin[1]), float(left.origin[1]), f"{right.text} height")
    required = set(iocol_pins("B", two_sided=True))
    if not required <= {label.text for label in io_b2.labels}:
        raise RuntimeError(f"{io_b2.name}: missing IO pins {sorted(required - {l.text for l in io_b2.labels})}")

    if spice is not None:
        text = spice.read_text()
        header = f".SUBCKT {port_io_name('B2')} {' '.join(iocol_pins('B', two_sided=True))}"
        if header not in text or ".SUBCKT iocol_block_b2 " not in text:
            raise RuntimeError(f"{spice}: {port_io_name('B2')} is missing or its pins are not {header}")
        for port in ("A", "B"):
            header = f".SUBCKT {port_io_name(port)} {' '.join(iocol_pins(port))}"
            if header not in text:
                raise RuntimeError(f"{spice}: {port_io_name(port)} is missing or its pins are not {header}")
            if f".SUBCKT iocol_block_{port.lower()} " not in text:
                raise RuntimeError(f"{spice}: missing the block netlist of port {port}")

    cap_array = cells["col_cap_x4_sram_8t"]
    expected_caps = ["FILLER_cgedge_8t"] + [topbot_name(False, bool(row % 2)) for row in range(mux_rows)]
    if [ref.cell_name for ref in cap_array.references] != expected_caps:
        raise RuntimeError(f"{cap_array.name}: expected filler and four oriented cap masters")
    if any(ref.rotation or ref.x_reflection for ref in cap_array.references):
        raise RuntimeError(f"{cap_array.name}: edge masters must only be translated")
    cap_box = edge_boundary(cap_array)
    for count in wordline_counts:
        array = cells[f"array_x{count}x4_sram_8t"]
        colgrp = cells[colgrp_name(count)]
        array_box = edge_boundary(array)
        colgrp_box = edge_boundary(colgrp)
        expected_width = (
            (array_box[2] - array_box[0]) + (cap_box[2] - cap_box[0])
            + sum(box[2] - box[0] for box in io_boxes.values())
        )
        assert_close(colgrp_box[2] - colgrp_box[0], expected_width, f"{colgrp.name}: width")
        assert_close(colgrp_box[3] - colgrp_box[1], 2.376, f"{colgrp.name}: height")
        expected_refs = [
            port_io_name("A"), "col_cap_x4_sram_8t",
            f"array_x{count}x4_sram_8t", port_io_name("B"),
        ]
        if [reference.cell_name for reference in colgrp.references] != expected_refs:
            raise RuntimeError(f"{colgrp.name}: expected iocol A / cap / array / iocol B")
        if any(ref.rotation or ref.x_reflection for ref in colgrp.references):
            raise RuntimeError(f"{colgrp.name}: one unsplit array, nothing mirrored")
        xs = [float(reference.origin[0]) for reference in colgrp.references]
        if xs != sorted(xs):
            raise RuntimeError(f"{colgrp.name}: port A must be left of the array, port B right")
        for prefix, layer in (("WLA", M3), ("WLB", M5)):
            labels = indexed_labels(colgrp, prefix)
            if set(labels) != set(range(count)):
                raise RuntimeError(f"{colgrp.name}: incomplete {prefix} bus")
            if any(label.layer != layer for label in labels.values()):
                raise RuntimeError(f"{colgrp.name}: {prefix} is on wrong layer")
        seam_x = edge_boundary(cells[port_io_name("A")])[2]
        tie_x0 = seam_x - seam_tie_inset - via_half - m2_end_cap
        m2 = [bbox(poly) for poly in colgrp.polygons if poly.layer == M2]
        straps = [box for box in m2 if abs(box[0] - tie_x0) > 1e-6]
        if len(straps) != 2 * mux_rows:
            raise RuntimeError(f"{colgrp.name}: expected eight port-A bitline straps")
        port_a_ties = [box for box in (bbox(poly) for poly in colgrp.polygons if poly.layer == V1) if box[2] < seam_x]
        if len(m2) - len(straps) != 3 * mux_rows or len(port_a_ties) != 3 * mux_rows:
            raise RuntimeError(f"{colgrp.name}: expected a port-A seam tie per supply bar")
        required_colgrp = {
            "DA", "QA", "DB", "QB",
            "wrenaA", "wrenanA", "wrenaB", "wrenanB",
            "oeb_outA", "oe_outA", "oeb_outB", "oe_outB",
            "blprechnA", "blprechnB", "sae_A", "sae_B", "VDD", "VSS",
        }
        required_colgrp.update(
            f"{prefix}{port}[{index}]"
            for prefix in ("yseln", "ysel")
            for port in ("A", "B") for index in range(mux_rows)
        )
        names = {label.text for label in colgrp.labels}
        if not required_colgrp.issubset(names):
            raise RuntimeError(
                f"{colgrp.name}: missing pins {sorted(required_colgrp - names)}"
            )
        import re
        split_era = sorted(name for name in names
                           if re.match(r"(WLT|WLBA|WLBB|yselt|yselb|blprecht|blprechb)", name))
        if split_era:
            raise RuntimeError(f"{colgrp.name}: split-era pins remain: {split_era}")

        half, pair = cells[colgrp_name(count, "half")], cells[colgrp_name(count, "pair")]
        half_box, pair_box = edge_boundary(half), edge_boundary(pair)
        assert_close(
            half_box[2] - half_box[0],
            expected_width - (io_boxes["B"][2] - io_boxes["B"][0]),
            f"{half.name}: width",
        )
        assert_close(
            pair_box[2] - pair_box[0],
            2 * (half_box[2] - half_box[0]) + b2_box[2] - b2_box[0],
            f"{pair.name}: width",
        )
        if [ref.cell_name for ref in pair.references] != [half.name, io_b2.name, half.name]:
            raise RuntimeError(f"{pair.name}: expected half / iocol B2 / half")
        if not pair.references[2].x_reflection:
            raise RuntimeError(f"{pair.name}: the second half is mirrored")
        for prefix, layer in (("WLA", M3), ("WLB", M5)):
            if set(indexed_labels(pair, prefix)) != set(range(2 * count)):
                raise RuntimeError(f"{pair.name}: incomplete {prefix} bus")
        pair_names = {label.text for label in pair.labels}
        required_pair = {
            "DA", "QA", "DB", "QB", "wrenaB", "wrenanB", "oeb_outB", "oe_outB", "sae_B",
            "blprechnB", "blprechnBR", "VDD", "VSS",
        }
        for suffix in ("", "R"):
            required_pair.update(f"{c}A{suffix}" for c in ("wrena", "wrenan", "oeb_out", "oe_out", "blprechn"))
            required_pair.add(f"sae_A{suffix}")
        required_pair.update(
            f"{prefix}{port}[{index}]"
            for prefix in ("yseln", "ysel")
            for port in "AB" for index in range(2 * mux_rows)
        )
        if not required_pair <= pair_names:
            raise RuntimeError(f"{pair.name}: missing pins {sorted(required_pair - pair_names)}")

    digests: dict[str, str] = {}
    for name in sorted(cells):
        if not name.startswith(("iocol_sram", "colgrp_", "col_cap_")):
            continue
        digests[name] = hierarchical_fingerprint(cells[name])
    return digests


def port_nodes(module: dict[str, Any]) -> list[tuple[str, Any]]:
    """Expand ports, including one-bit vectors preserved as Yosys attributes."""
    result: list[tuple[str, Any]] = []
    for name, info in module["ports"].items():
        bits = info["bits"]
        attributes = module.get("netnames", {}).get(name, {}).get("attributes", {})
        vector = (
            len(bits) > 1
            or "offset" in info
            or "upto" in info
            or bool(int(attributes.get("single_bit_vector", "0"), 2))
        )
        for i, bit in enumerate(bits):
            index = info.get("offset", 0) + (
                len(bits) - 1 - i if info.get("upto") else i
            )
            result.append((f"{name}[{index}]" if vector else name, bit))
    return result


def verify_decoder_physical(
    path: Path,
    plan: dict[str, Any],
    design: dict[str, Any] | None = None,
    master_sizes: dict[str, tuple[float, float]] | Callable[[], dict[str, tuple[float, float]]] | None = None,
    top_name: str = "sram_decoder_2rw",
) -> dict[str, Any]:
    """Verify physical DRC/LVS contracts of synthesized and routed decoder GDS."""
    gdstk = require_gdstk()
    lib = gdstk.read_gds(str(path))
    cell = next(c for c in lib.cells if c.name == top_name)
    # DEF pin shapes are streamed as datatype 251. Include them as conductors
    # for connectivity, exactly as the macro compiler's controller import.
    for poly in list(cell.polygons):
        if poly.datatype == 251 and poly.layer in METALS:
            copy = poly.copy()
            copy.datatype = 0
            cell.add(copy)
    graph = MetalGraph(cell)
    roots: set[int] = set()
    for output in plan["outputs"]:
        labels = [label for label in cell.labels if label.text == output["name"]]
        if len(labels) != 1:
            raise RuntimeError(f"missing/ambiguous output {output['name']}")
        label = labels[0]
        if label.layer != output["layer"] or abs(label.origin[0] - output["x"]) > 1e-7:
            raise RuntimeError(f"{label.text} is not aligned with its bitcell pin")
        root = graph.label_root(label)
        if root in roots:
            raise RuntimeError("decoder wordlines are shorted")
        roots.add(root)
        # Every output route must reach standard-cell M1, not just a pin stub.
        if not any(
            p.layer == 19 and graph.root(i) == root
            for i, p in enumerate(graph.polygons)
        ):
            raise RuntimeError(f"{label.text} is an isolated output stub")
    net_roots: dict[str, set[int]] = {}
    aliases: dict[str, str] = {}

    def connect(net: str, label: Any) -> None:
        actual_net = aliases.get(net, net)
        net_roots.setdefault(actual_net, set()).add(graph.label_root(label))

    if design is not None:
        if callable(master_sizes):
            resolved_sizes = master_sizes()
        elif master_sizes is not None:
            resolved_sizes = master_sizes
        else:
            resolved_sizes = {}
            for c in lib.cells:
                (x0, y0), (x1, y1) = c.bounding_box()
                resolved_sizes[c.name] = (round(float(x1 - x0), 7), round(float(y1 - y0), 7))

        module = design["modules"][top_name]
        aliases = {
            port["bits"][0]: name
            for name, port in module["ports"].items()
            if name in ("VDD", "VSS")
        }
        placements = placed_def_instances(
            path.with_suffix(".def"), {c.name: c for c in lib.cells}, resolved_sizes
        )
        top_labels = {label.text: label for label in cell.labels}
        for pin, bit in port_nodes(module):
            if pin not in top_labels:
                raise RuntimeError(f"missing physical port {pin}")
            connect(bit, top_labels[pin])
        for name, instance in module["cells"].items():
            if name not in placements:
                raise RuntimeError(f"missing physical instance {name}")
            reference = placements[name]
            if reference.cell_name != instance["type"]:
                raise RuntimeError(f"wrong physical master at {name}")
            labels = reference.get_labels()
            for pin, bits in instance["connections"].items():
                if len(bits) != 1:
                    raise RuntimeError(f"non-scalar standard-cell pin {name}/{pin}")
                hits = [
                    label
                    for label in labels
                    if label.text.upper() == pin.upper() and label.layer in METALS
                ]
                if not hits:
                    raise RuntimeError(f"missing GDS label {name}/{pin}")
                for label in hits:
                    connect(bits[0], label)
        # Trace all supply pins, including filler/tap instances, back to the
        # PDN. The metal graph excludes transistor channels.
        for label in cell.get_labels():
            if (
                label.text.upper().rstrip("!") in ("VDD", "VSS")
                and label.layer in METALS
            ):
                connect(label.text.upper().rstrip("!"), label)
        if not {"VDD", "VSS"} <= net_roots.keys():
            raise RuntimeError("decoder is missing supply pins")
        owner: dict[int, str] = {}
        for net, components in net_roots.items():
            if len(components) != 1:
                raise RuntimeError(
                    f"decoder net {net} is disconnected ({len(components)} components)"
                )
            r = next(iter(components))
            if r in owner and owner[r] != net:
                raise RuntimeError(f"decoder short between {owner[r]} and {net}")
            owner[r] = net
    return {
        "wordline_pins": len(roots),
        "aligned": True,
        "isolated": True,
        "connected_nets": len(net_roots),
    }
