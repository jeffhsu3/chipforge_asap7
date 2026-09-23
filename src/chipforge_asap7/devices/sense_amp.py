"""OpenFinRAM's 16-transistor ``sense_amp_sram``: the reference topology.

A symmetric regenerative latch with differential ``SA/SAN`` inputs,
active-high ``SAE`` evaluation, active-low ``SAPRECHN`` precharge, and
complementary ``QA/QAN`` outputs.  This module holds the circuit only: the
SPICE bench in :mod:`chipforge_asap7.devices.sense_amp_spice` simulates it,
and :mod:`chipforge_asap7.devices.sense_amp_row` draws it as an IO cell,
without the four output dummies (``N12``, ``N13``, ``P14``, ``P15``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "DEFAULT_N_FINS",
    "DEFAULT_P_FINS",
    "SENSE_AMP_PINS",
    "SenseAmpTransistor",
    "sense_amp_transistors",
]

# OpenFinRAM's characterization sizing (scripts/characterize_read.py).
DEFAULT_N_FINS = 12
DEFAULT_P_FINS = 4

SENSE_AMP_PINS = (
    "SA",
    "SAN",
    "SAE",
    "SAPRECHN",
    "QA",
    "QAN",
    "VDD",
    "VSS",
)


@dataclass(frozen=True)
class SenseAmpTransistor:
    """One logical transistor in the sense-amplifier topology."""

    name: str
    flavor: Literal["n", "p"]
    drain: str
    gate: str
    source: str
    bulk: str

    @property
    def terminals(self) -> dict[str, str]:
        """Map leaf FinFET terminal names to circuit nets."""

        return {
            "D": self.drain,
            "G": self.gate,
            "S": self.source,
            "B": self.bulk,
        }


_TRANSISTORS = (
    SenseAmpTransistor("P0", "p", "QAN", "SAPRECHN", "VDD", "VDD"),
    SenseAmpTransistor("P1", "p", "N59", "SAPRECHN", "QAN", "VDD"),
    SenseAmpTransistor("P2", "p", "QA", "SAPRECHN", "N59", "VDD"),
    SenseAmpTransistor("P3", "p", "VDD", "SAPRECHN", "QA", "VDD"),
    SenseAmpTransistor("P4", "p", "VDD", "QA", "QAN", "VDD"),
    SenseAmpTransistor("P5", "p", "QA", "QAN", "VDD", "VDD"),
    SenseAmpTransistor("N6", "n", "N52", "SA", "N57", "VSS"),
    SenseAmpTransistor("N7", "n", "N58", "SAN", "N52", "VSS"),
    SenseAmpTransistor("N8", "n", "N57", "QA", "QAN", "VSS"),
    SenseAmpTransistor("N9", "n", "QA", "QAN", "N58", "VSS"),
    SenseAmpTransistor("N10", "n", "VSS", "SAE", "N52", "VSS"),
    SenseAmpTransistor("N11", "n", "N52", "SAE", "VSS", "VSS"),
    SenseAmpTransistor("N12", "n", "QAN", "VSS", "VSS", "VSS"),
    SenseAmpTransistor("N13", "n", "VSS", "VSS", "QA", "VSS"),
    SenseAmpTransistor("P14", "p", "QAN", "VDD", "VDD", "VDD"),
    SenseAmpTransistor("P15", "p", "VDD", "VDD", "QA", "VDD"),
)


def sense_amp_transistors() -> tuple[SenseAmpTransistor, ...]:
    """Return the canonical OpenFinRAM 16-transistor topology."""

    return _TRANSISTORS
