"""Logical-effort decoder sizing: it has to land on the released cells."""

from __future__ import annotations

import pytest

from chipforge_asap7.devices import (
    LogicalEffortModel,
    size_decoder,
    split_fins_into_rows,
)

#: The wordline load and stage effort the released 32-wordline decoder implies.
RELEASED_WL_FF = 22.8
RELEASED_EFFORT = 4.43


def test_reproduces_the_released_driver_and_post_decode_nand():
    """62-fin `dec_inv_62f_halved_AND` and the 14/7-fin post-decode NAND."""
    sizing = size_decoder(RELEASED_WL_FF, 32, stage_effort=RELEASED_EFFORT)
    driver, nand = sizing.stages["driver"], sizing.stages["post_nand"]
    assert (driver.n_fins, driver.p_fins) == (62, 62)
    assert (nand.n_fins, nand.p_fins) == (28, 14)  # per input; two fingers of 14 and 7
    assert sizing.driver_rows == ((18, 18), (13, 13))
    assert sizing.nand_rows == ((14, 7),)
    assert sizing.groups == {"PC": 2, "PB": 2, "PA": 1}  # the released 5-to-32 split
    assert (sizing.address_bits, sizing.slices) == (5, 8)


def test_every_stage_lands_near_the_target_effort():
    sizing = size_decoder(RELEASED_WL_FF, 32, stage_effort=4.0)
    for stage in sizing.stages.values():
        if stage.drive > 2:  # a one- or two-fin gate is quantised, not sized
            assert 3.0 < stage.effort < 5.2, stage
    assert sizing.stages["driver"].effort == pytest.approx(4.0, rel=0.05)


def test_the_wordline_load_sizes_the_driver_and_depth_does_not():
    shallow, deep = size_decoder(20.0, 16), size_decoder(20.0, 256)
    assert shallow.stages["driver"].drive == deep.stages["driver"].drive
    assert shallow.stages["post_nand"].drive == deep.stages["post_nand"].drive
    light, heavy = size_decoder(10.0, 64), size_decoder(40.0, 64)
    assert heavy.stages["driver"].drive > 3 * light.stages["driver"].drive
    # Depth shows up as fan-out on the predecode lines instead.
    assert deep.line_load_fF["PC"] > 4 * shallow.line_load_fF["PC"]
    assert deep.stages["pc_inv"].drive > shallow.stages["pc_inv"].drive
    # Holding the stage effort holds the delay: what a deeper decoder costs is
    # input capacitance, which the address drivers upstream have to supply.
    assert deep.delay_ps == pytest.approx(shallow.delay_ps, rel=0.15)
    assert deep.address_input_cap_fF > 5 * shallow.address_input_cap_fF


def test_predecode_groups_follow_the_depth():
    assert size_decoder(10.0, 4).groups == {"PC": 2, "PB": 0, "PA": 0}
    assert set(size_decoder(10.0, 4).paths) == {"PC"}
    assert size_decoder(10.0, 8).groups == {"PC": 2, "PB": 1, "PA": 0}
    assert size_decoder(10.0, 64).groups == {"PC": 2, "PB": 2, "PA": 2}
    assert size_decoder(10.0, 1024).groups == {"PC": 2, "PB": 4, "PA": 4}
    assert size_decoder(10.0, 100).slices == 25  # rounds up to whole slices
    with pytest.raises(ValueError, match="predecode group"):
        size_decoder(10.0, 4096)


def test_delay_is_the_slowest_predecode_path():
    sizing = size_decoder(RELEASED_WL_FF, 32)
    assert sizing.delay_ps == max(sizing.path_delay_ps(line) for line in sizing.paths)
    assert sizing.paths["PA"][-4:] == (
        "select_nand",
        "select_inv",
        "post_nand",
        "driver",
    )
    assert sizing.paths["PC"][-4:] == ("row_nand", "row_inv", "post_nand", "driver")
    # Six stages at an effort of four: about six FO4-ish delays of gate delay.
    assert 40 < sizing.delay_ps < 70
    assert "slowest line" in sizing.summary()


def test_model_numbers_are_the_measured_asap7_tt_set():
    model = LogicalEffortModel()
    assert model.tau_ps * (4 + model.p_inv) == pytest.approx(8.08, abs=0.05)  # FO4
    assert (model.g(1), model.g(2), model.g(3)) == (1.0, 1.5, 2.0)
    slow = LogicalEffortModel(tau_ps=3.0)
    assert size_decoder(20.0, 32, model=slow).delay_ps > size_decoder(20.0, 32).delay_ps


def test_split_fins_into_rows():
    assert split_fins_into_rows(31) == (18, 13)
    assert split_fins_into_rows(18) == (18,)
    assert split_fins_into_rows(40) == (18, 18, 4)
    assert split_fins_into_rows(5, limit=2) == (2, 2, 1)


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((0.0, 32), {}),
        ((10.0, 0), {}),
        ((10.0, 2.5), {}),
        ((10.0, 32), {"stage_effort": 1.0}),
    ],
)
def test_rejects_nonsense(args, kwargs):
    with pytest.raises(ValueError):
        size_decoder(*args, **kwargs)
