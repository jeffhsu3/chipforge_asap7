"""Static rising-edge output bank with interlocked master/slave clock phases."""

from dataclasses import replace

from .circuit import Circuit


def add_register_bank(c, connections):
    """Two 8T latches per bit; complementary phase feedback prevents overlap.

    The receiving master data still has a finite setup/hold aperture around
    CLK rising. This is not a synchronizer or an asynchronous-ready handshake.
    """
    if c.output_register is not None:
        raise ValueError("output register already present")

    def gate(kind, name, *nets):
        start = len(c.devices)
        getattr(c, kind)(*nets)
        c.group_since(name, start)

    gate("inv", "register_clock_inverter", "CLK", "REG_CLKB")
    gate("and2", "register_master_enable", "REG_CLKB", "REG_SB", "REG_M")
    gate("inv", "register_master_complement", "REG_M", "REG_MB")
    gate("and2", "register_slave_enable", "CLK", "REG_MB", "REG_S")
    gate("inv", "register_slave_complement", "REG_S", "REG_SB")
    for bit, (data, output) in enumerate(connections):
        master = f"REG_DATA<{bit}>"
        gate("latch", f"register_master_{bit}", data, master, "REG_M", "REG_MB")
        gate("latch", f"register_slave_{bit}", master, output, "REG_S", "REG_SB")
    c.output_register = {
        "edge": "rising",
        "macro_read_latency_cycles": 1,
        "sample_ready_before_capture_required": True,
        "connections": [list(pair) for pair in connections],
        "master_enable": "REG_M",
        "master_complement": "REG_MB",
        "slave_enable": "REG_S",
        "slave_complement": "REG_SB",
        "setup_hold_qualified": False,
        "reset_clears_q": False,
    }


def add_registered_output(c, bits):
    """Keep the replica-controlled sampling latches ahead of the edge register."""
    if c.output_register is not None or "CLK" not in c.pins:
        raise ValueError("registered output requires an unregistered clocked macro")
    rename = {f"Q[{i}]": f"SAMPLED<{i}>" for i in range(bits)}
    c.devices = [
        replace(d, nodes=tuple(rename.get(n, n) for n in d.nodes)) for d in c.devices
    ]
    add_register_bank(c, [(f"SAMPLED<{i}>", f"Q[{i}]") for i in range(bits)])
    c.name += "_posedge"


def output_register_circuit(bits=1, fins=2, vt="rvt"):
    if type(bits) is not int or bits < 1:
        raise ValueError("bits must be a positive integer")
    c = Circuit(
        f"brom_output_register_{bits}_f{fins}_{vt}",
        (
            "CLK",
            *(f"D<{i}>" for i in range(bits)),
            *(f"Q<{i}>" for i in range(bits)),
            "VDD",
            "VSS",
        ),
        fins,
        vt,
    )
    add_register_bank(c, [(f"D<{i}>", f"Q<{i}>") for i in range(bits)])
    return c
