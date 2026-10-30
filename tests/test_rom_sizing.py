"""NOR-ROM sizing against known loads, fanout, and its emitted cell sizes."""

from dataclasses import replace

import pytest

from chipforge_asap7.devices import (
    LogicalEffortModel,
    RomArraySpec,
    RomBitcellSpec,
    RomDecoderSpec,
    size_rom_decoder,
)

NO_WIRE = LogicalEffortModel(wire_cap_fF_per_um=0)


def spec(words=16, bits=8, fins=2, **kwargs):
    return RomDecoderSpec(RomArraySpec(words, bits, RomBitcellSpec(fins)), **kwargs)


def size(base=None, **kwargs):
    return size_rom_decoder(
        base or spec(), model=NO_WIRE, poly_cap_fF_per_um=0, **kwargs
    )


def test_wordline_load_counts_every_bitcell_gate_even_for_clear_bits():
    light = size(spec(bits=16))
    wide = size(spec(bits=64))
    more_fins = size(spec(bits=64, fins=3))
    assert light.wl_load_fF == pytest.approx(16 * 2 * 0.0417)
    assert wide.wl_load_fF == pytest.approx(64 * 2 * 0.0417)
    assert more_fins.wl_load_fF == pytest.approx(64 * 3 * 0.0417)
    assert light.spec.inverter.pull_up_fins < wide.spec.inverter.pull_up_fins
    assert wide.spec.inverter.pull_up_fins < more_fins.spec.inverter.pull_up_fins


def test_supplied_wordline_load_replaces_the_total_and_sizes_real_devices():
    result = size(wl_load_fF=8, wl_wire_delay_ps=3.5)
    # C_in = 8/4 fF; each balanced inverter fin-pair contributes 0.0834 fF.
    assert result.spec.inverter.pull_down_fins == 24
    assert result.spec.inverter.pull_up_fins == 24
    assert result.wl_load_fF == 8
    assert result.supplied_wl_load
    assert result.delay_ps == result.gate_delay_ps + 3.5
    assert result.spec.nand.rows != result.spec.inverter.rows
    for name, stage in result.stages.items():
        assert stage.effort <= 4 + 1e-12, name
    assert "supplied total" in result.summary()


def test_depth_changes_predecoder_fanout_independently_of_wordline_drive():
    shallow = size(spec(16), wl_load_fF=4)
    deep = size(spec(256), wl_load_fF=4)
    for name in ("driver", "post_nand"):
        assert shallow.stages[name].drive == deep.stages[name].drive
    assert deep.stages["pre_driver"].drive > shallow.stages["pre_driver"].drive
    assert deep.fanout["PB<0>"] == 32
    assert shallow.fanout["PB<0>"] == 8


def test_tied_buffer_inputs_and_parallel_copies_are_real_input_loads():
    result = size(wl_load_fF=2)
    nand, inv = result.spec.pre_slot.nand, result.spec.pre_slot.inverter
    nand_cap = sum(nand.rows[0]) * nand.fingers * NO_WIRE.c_fin_fF
    inv_cap = inv.total_fins * NO_WIRE.c_fin_fF
    # In a 3+1 decode, A3 drives PB1's tied NAND inputs plus AN3's inverter.
    assert result.fanout["A<3>"] == 3
    assert result.address_input_cap_fF["A<3>"] == pytest.approx(2 * nand_cap + inv_cap)
    # AN0 is duplicated for placement; A0 also drives four PA NAND inputs.
    assert result.fanout["A<0>"] == 6
    assert result.address_input_cap_fF["A<0>"] == pytest.approx(
        4 * nand_cap + 2 * inv_cap
    )


def test_supplied_line_wire_capacitance_is_added_to_fanout_not_to_the_estimate():
    light = size(wl_load_fF=2)
    heavy = size(wl_load_fF=2, wire_load_fF={"PB<0>": 10})
    assert heavy.wire_load_fF["PB<0>"] == 10
    nand = heavy.spec.nand
    gates = 8 * sum(nand.rows[0]) * nand.fingers * NO_WIRE.c_fin_fF
    assert heavy.line_load_fF["PB<0>"] == pytest.approx(10 + gates)
    assert heavy.supplied_wire_nets == ("PB<0>",)
    assert heavy.stages["pre_driver"].drive > light.stages["pre_driver"].drive
    assert heavy.stages["driver"].drive == light.stages["driver"].drive


def test_estimated_wires_include_array_poly_and_decoder_route_length():
    result = size_rom_decoder(spec(64))
    s = result.spec
    metal_length = max(s.pin_y(row, "Y") for row in range(s.interleave)) / 1000
    expected = result.wl_gate_load_fF + 0.2 * (s.array.height / 1000 + metal_length)
    assert result.wl_load_fF == pytest.approx(expected)
    assert result.wire_load_fF["PB<0>"] > 0
    assert not result.supplied_wl_load
    assert result.wl_wire_delay_ps is None
    assert "distributed wire delay not included" in result.summary()
    assert all(s.effort <= 4 + 1e-12 for s in result.stages.values())


def test_large_load_folds_driver_without_changing_the_address_mapping():
    base = spec()
    result = size(base, wl_load_fF=16)
    assert result.spec.driver_fingers == 4
    assert result.spec.inverter.pull_up_fins == 48
    assert result.spec.interleave == base.interleave
    assert result.spec.pa_bits == base.pa_bits
    assert all(result.spec.lines(w) == base.lines(w) for w in range(16))


def test_no_predecode_keeps_external_rail_loads_and_only_two_stages():
    result = size(spec(8, predecode=False, lane=False), wl_load_fF=4)
    assert set(result.stages) == {"driver", "post_nand"}
    assert not result.address_input_cap_fF
    assert "pre_nand" not in result.line_load_fF
    assert result.fanout["PB<0>"] == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wl_load_fF": 0},
        {"wl_load_fF": float("nan")},
        {"wl_load_fF": float("inf")},
        {"stage_effort": 1},
        {"stage_effort": float("nan")},
        {"poly_cap_fF_per_um": -1},
        {"wire_load_fF": {"PB<0>": -1}},
        {"wire_load_fF": {"typo": 1}},
        {"wl_wire_delay_ps": -1},
    ],
)
def test_rejects_invalid_loads(kwargs):
    with pytest.raises(ValueError):
        size_rom_decoder(spec(), **kwargs)


def test_non_rvt_needs_an_explicit_model():
    base = spec()
    base = replace(base, array=replace(base.array, bitcell=RomBitcellSpec(vt="lvt")))
    with pytest.raises(ValueError, match="calibrated model"):
        size_rom_decoder(base)
    assert size_rom_decoder(base, model=NO_WIRE).spec.inverter.vt == "lvt"


def test_unbuildable_drive_does_not_silently_clip_or_change_interleave():
    with pytest.raises(ValueError, match="interleave"):
        size(wl_load_fF=100)
