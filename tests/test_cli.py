"""The generator command lines, built from the specs by tyro."""

from __future__ import annotations

import pytest

pytest.importorskip("tyro")
gdspy = pytest.importorskip("gdspy")

from chipforge_asap7.devices import (
    BitlineMuxSpec,
    FinFETSpec,
    InverterSpec,
    IoColumnSpec,
)
from chipforge_asap7.devices.bitline_mux import main as bitline_mux_main
from chipforge_asap7.devices.cli import Option, parse_spec


def test_spec_fields_are_the_options_and_tuples_are_space_separated():
    args = parse_spec(
        InverterSpec,
        ["--rows", "18", "18", "13", "13", "--no-abut", "--input-reach", "14", "14"],
        description="",
    )

    assert args.spec == InverterSpec(
        rows=((18, 18), (13, 13)), abut=False, input_reach=(14, 14)
    )
    assert args.out is None


def test_a_nested_spec_is_prefixed_and_keeps_its_parents_default():
    args = parse_spec(
        IoColumnSpec,
        [
            "--mux.selects",
            "8",
            "--mux.bitline-entry",
            "510",
            "78",
            "--mux.bitline-layer",
            "M4",
            "--no-tap",
        ],
        description="",
    )

    assert args.spec.mux.selects == 8
    assert args.spec.mux.bitline_entry == (510, 78)
    assert args.spec.mux.rows == IoColumnSpec().mux.rows == 2
    assert not args.spec.tap


def test_hidden_fields_are_not_options_and_the_spec_is_the_real_class():
    args = parse_spec(
        FinFETSpec, ["--fins", "3"], description="", hide=("gate_length",)
    )
    assert type(args.spec) is FinFETSpec
    assert args.spec == FinFETSpec(fins=3)

    with pytest.raises(SystemExit):
        parse_spec(
            FinFETSpec, ["--gate-length", "30"], description="", hide=("gate_length",)
        )


def test_a_value_the_spec_rejects_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse_spec(FinFETSpec, ["--fins", "0"], description="")

    assert exit_info.value.code == 2
    assert "fins must be >= 1" in capsys.readouterr().err


def test_options_that_are_not_spec_fields():
    group = Option("group", bool, False, "")

    assert (
        parse_spec(BitlineMuxSpec, [], description="", options=(group,)).group is False
    )
    assert parse_spec(
        BitlineMuxSpec, ["--group"], description="", options=(group,)
    ).group


def test_writing_a_gds_leaves_gdspys_current_library_alone(tmp_path):
    before = gdspy.current_library
    out = tmp_path / "mux.gds"

    bitline_mux_main(["--selects", "2", "--group", "--out", str(out)])

    assert gdspy.current_library is before
    # The group's leaves are drawn into the file too, not left behind.
    assert set(gdspy.GdsLibrary(infile=str(out)).cells) == {
        "blmux_3n3p_h135x162_group2",
        "blmux_3n3p_h135x162_group2__leaf0",
        "blmux_3n3p_h135x162_group2__leaf1",
    }
