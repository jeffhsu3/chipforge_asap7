"""Tests for in-memory connectivity extraction, DSU, and layout transform utilities."""

import pytest

from chipforge_asap7.layout.transform import (
    dummy_array_name,
    have_gdstk,
    oriented_name,
    topbot_name,
)
from chipforge_asap7.verification.connectivity import DisjointSetUnion


def test_dsu_basic_operations():
    dsu = DisjointSetUnion(5)
    for i in range(5):
        assert dsu.find(i) == i
    dsu.union(0, 1)
    dsu.union(1, 2)
    assert dsu.find(0) == dsu.find(2)
    assert dsu.find(3) != dsu.find(0)
    dsu.union(3, 4)
    assert dsu.find(3) == dsu.find(4)
    dsu.union(2, 4)
    assert dsu.find(0) == dsu.find(3)


def test_naming_helpers():
    assert oriented_name("dummy", False, False) == "dummy"
    assert oriented_name("dummy", True, False) == "dummy_lr"
    assert oriented_name("dummy", False, True) == "dummy_v2"
    assert oriented_name("dummy", True, True) == "dummy_v2_lr"

    assert topbot_name(False, False) == "dummy_topbot_8t_v1"
    assert topbot_name(True, False) == "dummy_topbot_8t_v1_lr"
    assert topbot_name(False, True) == "dummy_topbot_8t_v2"
    assert topbot_name(True, True) == "dummy_topbot_8t_v2_lr"

    assert dummy_array_name(64) == "dummy_vertical_array_X64_8t"
    assert dummy_array_name(64, tap_pitch=16, mirror_x=True) == "dummy_vertical_array_X64_tap16_8t_lr"


def test_gdstk_geometry_helpers():
    gdstk = pytest.importorskip("gdstk")
    from chipforge_asap7.layout.transform import bbox, edge_boundary, is_box, rect

    cell = gdstk.Cell("test_cell")
    rect(cell, (0.0, -0.1, 0.108, 0.27), layer=100)
    assert len(cell.polygons) == 1
    poly = cell.polygons[0]
    assert bbox(poly) == (0.0, -0.1, 0.108, 0.27)
    assert is_box(poly, (0.0, -0.1, 0.108, 0.27))
    assert not is_box(poly, (0.0, 0.0, 0.108, 0.27))
    assert edge_boundary(cell) == (0.0, -0.1, 0.108, 0.27)
