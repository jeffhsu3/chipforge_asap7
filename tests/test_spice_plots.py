"""Plots of the SPICE benches, drawn from synthetic ngspice ``wrdata`` output."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from chipforge_asap7.devices.spice_plots import (
    _sense_amp_timing,
    main,
    plot_finfet_dc,
    plot_run,
    plot_sense_amp,
    read_wrdata,
)

VDD = 0.7
PNG_MAGIC = b"\x89PNG"


def _write(path: Path, names: list[str], rows: list[tuple[float, ...]]) -> None:
    lines = [" " + " ".join(f"{name:<15}" for name in names)]
    lines += [" " + " ".join(f"{value: .8e}" for value in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")


def _finfet_run(run: Path, *, flavor: str) -> Path:
    run.mkdir()
    sweep = [i * 0.01 for i in range(71)]
    # Turn-on as the gate rises for an nFET and as it falls for a pFET.
    gate_drive = sweep if flavor == "n" else [VDD - v for v in sweep]
    current = [1e-11 * math.exp(20 * v) for v in gate_drive]
    _write(
        run / "transfer.dat", ["v-sweep", "drain_current_a"], list(zip(sweep, current))
    )
    _write(
        run / "output.dat",
        ["v-sweep", "drain_current_a"],
        [(v, 1e-5 * v) for v in sweep],
    )
    vout = [VDD / (1 + math.exp(30 * (g - 0.35))) for g in gate_drive]
    if flavor == "p":
        vout = [VDD - v for v in vout]
    _write(
        run / "switch.dat",
        ["v-sweep", "v(d)", "supply_current_a"],
        [(v, o, 0.0) for v, o in zip(sweep, vout)],
    )
    (run / "results.json").write_text(
        json.dumps(
            {
                "model": f"{flavor}mos_rvt",
                "cell_name": f"{flavor}mos_fin_111",
                "on_off_ratio": current[-1] / current[0],
                "passed": True,
            }
        )
    )
    return run


def _sense_amp_run(run: Path, *, control_edges: bool = True) -> Path:
    run.mkdir()
    names = [
        "time",
        "v(sa)",
        "v(san)",
        "v(SAPRECHN)",
        "v(SAE)",
        "v(qa)",
        "v(qan)",
        "supply_current_a",
    ]
    result = {"topology": "sense_amp_sram", "nmos_model": "nmos_rvt", "pmos_model": "pmos_rvt",
              "vdd_v": VDD, "n_fins": 12, "p_fins": 4, "passed": False}  # fmt: skip
    for case, sa_high in (("sa_high", True), ("san_high", False)):
        rows = []
        for step in range(301):
            t = step * 5e-12
            prech = VDD if control_edges and t >= 0.5e-9 else 0.0
            sae = VDD if control_edges and t >= 0.52e-9 else 0.0
            fall = VDD * math.exp(-max(0.0, t - 0.52e-9) / 5e-12)
            high, low = VDD, fall
            qa, qan = (high, low) if sa_high else (low, high)
            # t = 0 carries the initial-condition spike the plot must ignore.
            current = 1e-2 if step == 0 else 5e-5 * (fall / VDD)
            rows.append(
                (
                    t,
                    0.7 if sa_high else 0.6,
                    0.6 if sa_high else 0.7,
                    prech,
                    sae,
                    qa,
                    qan,
                    current,
                )
            )
        _write(run / f"{case}.dat", names, rows)
        result[case] = {"sa_v": rows[-1][1], "san_v": rows[-1][2], "passed": sa_high,
                        "resolution_delay_s": 12e-12 if sa_high else None,
                        "evaluation_energy_j": 5e-16}  # fmt: skip
    (run / "results.json").write_text(json.dumps(result))
    return run


def test_wrdata_columns_are_read_by_lowercased_name(tmp_path: Path):
    path = tmp_path / "x.dat"
    _write(path, ["time", "v(SAE)"], [(0.0, 0.1), (1e-12, 0.2)])

    assert read_wrdata(path) == {"time": [0.0, 1e-12], "v(sae)": [0.1, 0.2]}


@pytest.mark.parametrize("flavor", ["n", "p"])
def test_finfet_run_is_drawn_beside_its_results(tmp_path: Path, flavor: str):
    run = _finfet_run(tmp_path / "run", flavor=flavor)

    png = plot_finfet_dc(run)

    assert png == run / "dc.png"
    assert png.read_bytes().startswith(PNG_MAGIC)


def test_sense_amp_run_is_drawn_even_when_a_case_fails(tmp_path: Path):
    """The failing case has no resolution delay; that must not stop the plot."""
    run = _sense_amp_run(tmp_path / "run")

    png = plot_sense_amp(run, out=tmp_path / "elsewhere" / "sa.png")

    assert png.read_bytes().startswith(PNG_MAGIC)


def test_missing_control_edges_use_the_recorded_time_range(tmp_path: Path):
    run = _sense_amp_run(tmp_path / "run", control_edges=False)
    data = read_wrdata(run / "sa_high.dat")
    verdict = json.loads((run / "results.json").read_text())["sa_high"]

    timing = _sense_amp_timing(data, verdict, VDD)

    assert timing.enable_ns is None
    assert timing.resolution_ns is None
    assert timing.window_ns == (data["time"][0] * 1e9, data["time"][-1] * 1e9)
    assert plot_sense_amp(run).read_bytes().startswith(PNG_MAGIC)


def test_plot_run_tells_the_benches_apart(tmp_path: Path, capsys):
    finfet = _finfet_run(tmp_path / "finfet", flavor="n")
    sense_amp = _sense_amp_run(tmp_path / "sense_amp")

    assert plot_run(finfet).name == "dc.png"
    main([str(sense_amp)])
    assert "transient.png" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):
        plot_run(tmp_path)
