"""Two images of one macro shape share an abstract and differ only in V2/V3."""

import json

import pytest

from chipforge_asap7.brom.__main__ import main
from chipforge_asap7.brom.abstract import abstract_name
from chipforge_asap7.layout.layers import LAYERS

FLAGS = ("--bits", "2", "--self-timed", "--registered-output")


def compile_image(tmp_path, name, words, *flags):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(words))
    out = tmp_path / name
    main([str(path), *FLAGS, *flags, "--out", str(out)])
    return out, json.loads((out / "manifest.json").read_text())


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("brom_abstract")
    return (
        compile_image(tmp_path, "zero", [0, 0, 0, 0]),
        compile_image(tmp_path, "image", [0, 1, 2, 3]),
        tmp_path,
    )


def test_abstract_name_drops_only_the_digest():
    name = "brom_selftimed_64x32_m1_f2_rvt_36dde0f4dae5_dense_posedge_g4_e1"
    assert abstract_name(name) == "brom_selftimed_64x32_m1_f2_rvt_dense_posedge_g4_e1"
    with pytest.raises(ValueError, match="programming digest"):
        abstract_name("brom_selftimed_64x32_m1_f2_rvt_dense")


def test_images_share_an_abstract(images):
    (zero_dir, zero), (image_dir, image), _ = images
    assert zero["top_cell"] != image["top_cell"]
    assert zero["abstract"] == image["abstract"]
    assert (zero_dir / "abstract.lef").read_text() == (
        image_dir / "abstract.lef"
    ).read_text()
    assert zero["abstract"]["programming_layers"] == {"V2": [25, 0], "V3": [35, 0]}


def test_abstract_views_carry_the_stable_name(images):
    (zero_dir, zero), _, _ = images
    kdb = pytest.importorskip("klayout.db")
    cell = zero["abstract"]["cell"]
    assert f"MACRO {cell}\n" in (zero_dir / "abstract.lef").read_text()
    layout = kdb.Layout()
    layout.read(str(zero_dir / "abstract.gds"))
    assert layout.top_cell().name == cell
    # The content-named view keeps its own name.
    assert f"MACRO {zero['top_cell']}\n" in (zero_dir / "macro.lef").read_text()


def test_images_differ_only_in_the_programming_layers(images):
    """Independent of the fingerprint: an XOR of the written GDS files."""
    (zero_dir, _), (image_dir, _), _ = images
    kdb = pytest.importorskip("klayout.db")

    def load(path):
        layout = kdb.Layout()
        layout.read(str(path))
        return layout

    a, b = load(zero_dir / "macro.gds"), load(image_dir / "macro.gds")
    programming = {
        (LAYERS[layer]["layer"], LAYERS[layer]["datatype"]) for layer in ("V2", "V3")
    }
    differing = set()
    for info in {*a.layer_infos(), *b.layer_infos()}:
        key = (info.layer, info.datatype)
        region = [
            kdb.Region(layout.top_cell().begin_shapes_rec(layout.layer(*key)))
            for layout in (a, b)
        ]
        if not (region[0] ^ region[1]).is_empty():
            differing.add(key)
    assert differing and differing <= programming


def test_a_different_shape_has_a_different_abstract(images):
    (_, zero), _, tmp_path = images
    _, other = compile_image(tmp_path, "fins", [0, 0, 0, 0], "--fins", "1")
    assert other["abstract"]["cell"] != zero["abstract"]["cell"]
    assert (
        other["abstract"]["frozen_geometry_sha256"]
        != zero["abstract"]["frozen_geometry_sha256"]
    )
