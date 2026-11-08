"""A whole NOR ROM macro as one flat LVS reference.

The reference is `chipforge_asap7.devices.rom_netlist`'s transistors, one
MOS per FIN x GATE channel; the macro's CDL is the same list sized by
``nfin``.  An uncontacted node is one net per fin, as the extractor sees it
without ``merge_fin_diffusion``.

Run it with ``run_lvs(..., flat=True)``: hierarchical extraction gains
nothing against a flat reference and was seen to stall on a 64x32 macro.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from chipforge_asap7.devices.rom_decoder import RomDecoderSpec
from chipforge_asap7.devices.rom_netlist import (
    rom_macro_devices,
    rom_macro_halves,
    rom_macro_pins,
)
from chipforge_asap7.layout.layers import PIN_LAYERS, require_gdspy

__all__ = [
    "MacroReference",
    "floating_drains",
    "label_floating_drains",
    "rom_macro_pins",
    "rom_macro_reference",
]


@dataclass(frozen=True)
class MacroReference:
    """The reference's device lines and the nets the macro labels."""

    lines: tuple[str, ...]
    pins: tuple[str, ...]

    def render(self, name: str) -> str:
        return "\n".join(
            [
                "* ASAP7 NOR ROM macro LVS reference: one MOS per FIN x GATE channel",
                f".SUBCKT {name} {' '.join(self.pins)}",
                *self.lines,
                f".ENDS {name}",
                ".END",
                "",
            ]
        )

    @property
    def fins(self) -> int:
        return len(self.lines)


def floating_drains(
    spec: RomDecoderSpec, programmed: Iterable[tuple[int, int]] = ()
) -> list[tuple[str, tuple[float, float]]]:
    """Each unprogrammed drain's reference net and its M1 stub's centre.

    The site a programming V1 would take is on that stub, so an M1 label
    there names the drain.
    """
    out = []
    for array, sites, _, _, tag, x in rom_macro_halves(spec, programmed):
        for bit in range(array.bits):
            for word in range(array.words):
                if (word, bit) not in sites:
                    sx, sy = array.site(word, bit)
                    out.append((f"{tag}F_W{word}_B{bit}", (x + sx, sy)))
    return out


def label_floating_drains(
    spec: RomDecoderSpec, programmed: Iterable[tuple[int, int]], cell: Any
) -> int:
    """Label every unprogrammed drain of ``build_rom``'s `cell`, for LVS only.

    An unprogrammed drain is a floating net, and a run of them along one bit
    row looks the same as the run in any other: the comparer has nothing but
    a search to tell them apart, and at 128x33 that search is most of the
    compare.  Net names settle such ties, and only ties: a label on the wrong
    drain, or on a drain a V1 joined to its bitline, still fails.  Pair it
    with ``rom_macro_reference(..., name_floating=True)``.  Returns the number
    of labels added.
    """
    gdspy = require_gdspy()
    layer, texttype = PIN_LAYERS["M1"]
    drains = floating_drains(spec, programmed)
    for name, origin in drains:
        cell.add(gdspy.Label(name, origin, layer=layer, texttype=texttype))
    return len(drains)


def rom_macro_reference(
    spec: RomDecoderSpec,
    programmed: Iterable[tuple[int, int]] = (),
    *,
    name_floating: bool = False,
) -> MacroReference:
    """The flat unit-fin reference of ``build_rom(spec, programmed)``.

    With `name_floating` the unprogrammed drains are pins too, to match a
    layout `label_floating_drains` has labelled.
    """
    lines = [
        line for d in rom_macro_devices(spec, programmed) for line in d.unit_fins()
    ]
    pins = rom_macro_pins(spec)
    if name_floating:
        floating = tuple(name for name, _ in floating_drains(spec, programmed))
        pins = (*pins[:-2], *floating, *pins[-2:])
    return MacroReference(tuple(lines), pins)
