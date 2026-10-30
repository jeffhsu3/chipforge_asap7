"""Receiving-gate sampling closure, return topology, and checker coverage."""

from dataclasses import replace

import numpy as np
import pytest
from test_brom import settle
from test_brom_replica import synthetic_replica_waveform

from chipforge_asap7.brom import BromSpec, self_timed_macro_circuit
from chipforge_asap7.brom.replica_timing import analyze_replica


def test_each_latch_has_local_controls_and_all_return_to_closure():
    c = self_timed_macro_circuit(BromSpec(4, 4, registered_output=True), [0, 1, 2, 3])
    banks = c.sample_distribution["banks"]
    assert len(banks) == 5 and banks[-1]["latch_group"] == "replica_sample_latch"
    assert len({b["enable"] for b in banks}) == 5
    assert len({b["complement"] for b in banks}) == 5
    by_name = {d.name: d for d in c.devices}
    assert len(by_name) == len(c.devices)
    assert max(int(d.name[1:]) for d in c.devices) == len(c.devices) - 1
    groups = dict(c.placement_groups)
    for b in banks:
        members = groups[b["latch_group"]]
        assert [
            by_name[m].nodes[1]
            for m in (members[0], members[1], members[6], members[7])
        ] == [b["enable"], b["complement"], b["complement"], b["enable"]]
        # Receiving controls never drive another latch bank.
        for other in banks:
            if other is not b:
                assert not {b["enable"], b["complement"]} & {
                    by_name[n].nodes[1] for n in groups[other["latch_group"]]
                }


@pytest.mark.parametrize("bank_index", [0, 1, 2])
@pytest.mark.parametrize("pending", ["enable", "complement", "command_inverted"])
def test_any_unclosed_or_inflight_receiver_withholds_ack(bank_index, pending):
    c = self_timed_macro_circuit(BromSpec(4, 2), [0, 1, 2, 3])
    banks = c.sample_distribution["banks"]
    groups = dict(c.placement_groups)
    detectors = {
        n
        for name, names in groups.items()
        if name.startswith(
            (
                "sample_receiver_off_",
                "sample_receiver_quiet_",
                "sample_receiver_closed_",
            )
        )
        for n in names
    }
    sense = {
        b[key] for b in banks for key in ("enable", "complement", "command_inverted")
    }
    # Inject a receiving-wire stall only at completion-sensor inputs, without
    # shorting the real driver. Exercise the real AND return tree/controller.
    c.devices = [
        replace(d, nodes=(d.nodes[0], "RETURN_" + d.nodes[1], *d.nodes[2:]))
        if d.name in detectors and d.nodes[1] in sense
        else d
        for d in c.devices
    ]
    inputs = {"CLK": 1, "RESETN": 0, "A[0]": 0, "A[1]": 0}
    for b in banks:
        inputs.update(
            {
                "RETURN_" + b["enable"]: 0,
                "RETURN_" + b["complement"]: 1,
                "RETURN_" + b["command_inverted"]: 1,
            }
        )
    inputs["RETURN_" + banks[bank_index][pending]] = 1 if pending == "enable" else 0
    ideal = ("SAMPLE", "PRECHN", *(b["enable"] for b in banks))
    state = settle(c, inputs, ideal_controls=ideal)
    assert state["SAMPLE_CLOSED"] == 0
    inputs["RETURN_" + banks[bank_index][pending]] = 0 if pending == "enable" else 1
    released = settle(c, inputs, state, ideal_controls=ideal)
    assert released["SAMPLE_CLOSED"] == 1
    assert released["EVAL"] == 0 and released["PRECHN"] == 0


def test_checker_uses_every_receiving_latch_including_replica():
    t, traces, spec, words = synthetic_replica_waveform()
    for i in range(spec.bits + 1):
        traces[f"SAMPLE_RX<{i}>"] = traces["SAMPLE"].copy()
        traces[f"SAMPLEB_RX<{i}>"] = traces["SAMPLEB"].copy()
    traces["SAMPLE_CLOSED"] = np.ones_like(t)
    for address in range(spec.depth):
        traces["SAMPLE_CLOSED"][address * 1000 + 440 : address * 1000 + 820] = 0
    good = analyze_replica(t, traces, spec, words, 1)
    assert good["passed"], good["failures"]
    assert good["minimum_margins_ps"]["sample_closed_margin_ps"] > 0
    # The command and data bank close normally; only the dummy receiving P-gate
    # lags. Both acknowledgment and evaluation must wait for that receiver.
    traces["SAMPLEB_RX<1>"][810:900] = 0
    bad = analyze_replica(t, traces, spec, words, 1)
    assert not bad["passed"]
    assert any("SAMPLE_CLOSED precedes" in f for f in bad["failures"])
    assert any("evaluation ended before" in f for f in bad["failures"])


@pytest.mark.parametrize("bits", [2, 4])
def test_idle_nodesets_are_consistent_without_forcing_transient_state(bits, tmp_path):
    from chipforge_asap7.brom import ReplicaSpec, size_wire_rc_distribution
    from chipforge_asap7.brom.dc_initialization import idle_logic_state
    from chipforge_asap7.brom.replica_timing import render_replica_bench
    from chipforge_asap7.brom.timing import TimingConditions

    spec = BromSpec(4, bits, registered_output=True)
    words = [0, 1, 2, 3]
    c = size_wire_rc_distribution(self_timed_macro_circuit(spec, words))
    state = idle_logic_state(c, spec.address_bits)
    assert all(state[n] == 0 for n in ("PRECHN", "EVAL", "SAMPLE"))
    assert state["SAMPLE_CLOSED"] == 1
    for bank in c.sample_distribution["banks"]:
        assert state[bank["enable"]] == 0
        assert state[bank["complement"]] == state[bank["closed"]] == 1
    deck, signals = render_replica_bench(
        spec,
        words,
        ReplicaSpec(),
        TimingConditions(),
        tmp_path / "model.pm",
        tmp_path / "model.osdi",
        wire_rc_controls=True,
    )
    assert ".nodeset V(rcdrv_" in deck
    assert " uic" not in deck and ".ic " not in deck
    assert len([s for s in signals if s.startswith("SAMPLE_RX<")]) == bits + 1
