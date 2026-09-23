"""OpenFinRAM's sense-amplifier topology, and the IO cell that draws it."""

from __future__ import annotations

from chipforge_asap7.devices import (
    SENSE_AMP_PINS,
    SenseAmpRowSpec,
    sense_amp_transistors,
)


def test_topology_is_the_symmetric_openfinram_16t_latch():
    devices = sense_amp_transistors()

    assert len(devices) == 16
    assert sum(device.flavor == "n" for device in devices) == 8
    assert sum(device.flavor == "p" for device in devices) == 8
    assert {device.name for device in devices} == {
        *(f"P{index}" for index in (*range(6), 14, 15)),
        *(f"N{index}" for index in range(6, 14)),
    }
    assert {net for device in devices for net in device.terminals.values()} == {
        *SENSE_AMP_PINS,
        "N52",
        "N57",
        "N58",
        "N59",
    }


def test_the_row_cell_draws_the_topology_without_its_output_dummies():
    """The IO cell is this circuit, less the four devices gated off by a supply."""
    dummies = {
        device.name
        for device in sense_amp_transistors()
        if device.gate in ("VDD", "VSS")
    }
    assert dummies == {"N12", "N13", "P14", "P15"}

    drawn = {(gate, flavor) for _, _, gate, _, flavor, _ in SenseAmpRowSpec().devices}
    assert drawn == {
        (device.gate, device.flavor)
        for device in sense_amp_transistors()
        if device.name not in dummies
    }
