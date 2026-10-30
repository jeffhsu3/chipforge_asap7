"""ROM shape selection and routing-aware placement, including drawn geometry."""

import json
from collections import Counter
from dataclasses import replace

import pytest
from test_rom_decoder import _extract, _library, _switch_level

from chipforge_asap7.devices import (
    RomArraySpec,
    RomDecoderSpec,
    build_rom,
    optimize_rom_packing,
    rom_metrics,
    search_rom_shapes,
    size_rom_decoder,
)
from chipforge_asap7.devices.rom_decoder import main
from chipforge_asap7.devices.rom_predecoder import network


def _spec(words=64, bits=4):
    return RomDecoderSpec(RomArraySpec(words=words, bits=bits))


@pytest.fixture(scope="module")
def packed():
    return optimize_rom_packing(_spec(), evaluations=64, seed=0)


def test_packing_is_reproducible_and_preserves_logic_and_row_pairs(packed):
    spec = packed.spec
    assert packed.selected.metrics.cost <= packed.initial.metrics.cost
    assert packed.selected.metrics.area_um2 < packed.initial.metrics.area_um2
    assert optimize_rom_packing(_spec(), evaluations=64, seed=0).spec == spec
    assert spec.predecoder_placement is not None
    assert spec.cell_name != _spec().cell_name
    assert spec.lines(33) == _spec().lines(33)
    assert set(Counter((p.pair, p.column) for p in spec.predecoder).values()) == {2}
    gates = network(spec.address_bits, spec.pa_bits, spec.row_bits)
    assert {p.gate.out for p in spec.predecoder} == set(gates)
    for p in spec.predecoder:
        assert p.gate == gates[p.gate.out]
        if p.gate.out.startswith("PA<"):
            assert p.pair == int(p.gate.out[3:-1]) % spec.interleave // 2
    for pair in range(spec.pairs):
        segments = [v for (p, _), v in spec.segments.items() if p == pair]
        for i, (y, x0, x1) in enumerate(segments):
            for y2, a, b in segments[i + 1 :]:
                if y2 == y:
                    assert x1 + 48 <= a or b + 48 <= x0
                elif abs(y2 - y) == 48:
                    assert x0 != b and x1 != a


def test_one_trial_and_external_predecode_keep_the_existing_layout(packed):
    assert optimize_rom_packing(packed.spec, evaluations=1).spec == packed.spec
    spec = RomDecoderSpec(RomArraySpec(words=8, bits=4), lane=False, predecode=False)
    assert optimize_rom_packing(spec).spec == spec


def test_invalid_explicit_placements_are_rejected(packed):
    spec = packed.spec
    places = spec.predecoder_placement
    with pytest.raises(ValueError, match="every logical gate|both rows"):
        replace(spec, predecoder_placement=places[:-1])
    with pytest.raises(ValueError, match="overlapping"):
        replace(spec, predecoder_placement=(*places, places[0]))
    pa = next(i for i, p in enumerate(places) if p.gate.out.startswith("PA<"))
    wrong = list(places)
    wrong[pa] = replace(wrong[pa], pair=(wrong[pa].pair + 1) % spec.pairs)
    with pytest.raises(ValueError, match="wordline rails"):
        replace(spec, predecoder_placement=tuple(wrong))


def test_shape_search_selects_by_routed_area_and_records_rejected_shapes():
    spec = _spec(64, 32)
    result = search_rom_shapes(
        spec, interleaves=(4, 8), splits=(3, 4), packing_evaluations=1
    )
    assert result.best.spec.pa_bits == 3  # the fewest-rails heuristic chooses 4
    assert result.best.metrics.area_um2 < rom_metrics(spec).area_um2
    assert len(result.candidates) == 2 and len(result.rejected) == 1
    assert all(c.spec.array == spec.array for c in result.candidates)
    report = json.loads(json.dumps(result.report(), allow_nan=False))
    assert report["selected"] == result.best.spec.cell_name
    assert report["pareto"] and report["rejected"][0]["reason"]
    assert all(c["gate_delay_estimate_ps"] is None for c in report["candidates"])


def test_strap_search_and_explicit_split_are_respected():
    spec = replace(_spec(), split=3)
    result = search_rom_shapes(
        spec, interleaves=(8,), strap_intervals=(4, 8), packing_evaluations=1
    )
    assert {c.spec.array.strap_every for c in result.candidates} == {4, 8}
    assert {c.spec.pa_bits for c in result.candidates} == {3}
    assert all(
        c.spec.array.words == 64 and c.spec.array.bits == 4 for c in result.candidates
    )


def test_sized_trials_recompute_loads_after_placement():
    result = optimize_rom_packing(_spec(16, 2), evaluations=6, sizing={"wl_load_fF": 4})
    independently_sized = size_rom_decoder(result.spec, wl_load_fF=4)
    assert result.selected.sizing.line_load_fF == independently_sized.line_load_fF
    assert result.selected.sizing.gate_delay_ps == independently_sized.gate_delay_ps
    assert result.selected.metrics == rom_metrics(independently_sized.spec)


@pytest.mark.parametrize(
    "kwargs", [{"interleaves": ()}, {"splits": (2.5,)}, {"packing_evaluations": 0}]
)
def test_bad_search_controls_fail(kwargs):
    with pytest.raises(ValueError):
        search_rom_shapes(_spec(), **kwargs)


def test_all_rejected_search_has_no_selected_shape():
    result = search_rom_shapes(_spec(), interleaves=(4,), packing_evaluations=1)
    assert not result.candidates and result.report()["selected"] is None
    with pytest.raises(ValueError, match="no feasible ROM shapes"):
        _ = result.best


def test_impossible_routing_has_a_finite_bound(monkeypatch):
    calls = []

    def impossible(spec, height):
        calls.append(height)

    monkeypatch.setattr(RomDecoderSpec, "_pack", impossible)
    with pytest.raises(ValueError, match="routing bound"):
        _ = _spec(16).pair_height
    assert 1 < len(calls) < 100


def test_search_explores_both_default_interleaves_without_stalling():
    result = search_rom_shapes(_spec(64, 32), packing_evaluations=64)
    assert {c.spec.interleave for c in result.candidates} == {8, 16}
    assert result.best.metrics.cost <= rom_metrics(_spec(64, 32)).cost


def test_cli_writes_selected_gds_and_machine_readable_report(tmp_path):
    out = tmp_path / "selected.gds"
    main(
        [
            "--array.words",
            "16",
            "--array.bits",
            "2",
            "--search",
            "--search-interleaves",
            "8",
            "--packing-evaluations",
            "6",
            "--out",
            str(out),
        ]
    )
    report = json.loads(out.with_suffix(".search.json").read_text())
    assert out.stat().st_size > 0
    assert report["selected"] in report["pareto"]
    assert report["gds"]["path"] == str(out)


@pytest.mark.parametrize(
    ("words", "interleave"), [(16, 8), (32, 16), (64, 8), (256, 8)]
)
def test_optimized_shapes_are_drc_clean(words, interleave, asap7_drc):
    result = search_rom_shapes(
        _spec(words, 2), interleaves=(interleave,), packing_evaluations=32
    )
    spec = result.best.spec
    library = _library()
    cell = build_rom(spec, [(w, w % 2) for w in range(words)], lib=library)
    assert asap7_drc(library, cell, tag="optimized_rom") == []


@pytest.mark.parametrize("words", [16, 64])
def test_searched_gds_decodes_every_address_and_preserves_programmed_bits(
    tmp_path, words
):
    spec = search_rom_shapes(
        _spec(words, 2), interleaves=(8,), packing_evaluations=32
    ).best.spec
    assert spec.predecoder_placement is not None
    devices, nets = _extract(tmp_path, spec, [(w, w % 2) for w in range(words)])
    for address in range(words):
        driven = {nets["VDD"]: 1, nets["VSS"]: 0}
        driven.update(
            {nets[f"A<{b}>"]: (address >> b) & 1 for b in range(spec.address_bits)}
        )
        value = _switch_level(devices, nets, driven)
        assert [value.get(nets[f"WL<{w}>"]) for w in range(words)] == [
            int(w == address) for w in range(words)
        ]
        for side, prefix in enumerate(("BLL", "BLR")):
            for bit in range(2):
                expected = (
                    0 if (side, bit) == (address // (words // 2), address % 2) else None
                )
                assert value.get(nets[f"{prefix}<{bit}>"]) == expected
