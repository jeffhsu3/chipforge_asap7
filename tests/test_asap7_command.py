"""The ``asap7`` command routes to the generators, as ``gt2n`` does."""

from __future__ import annotations

import importlib

import pytest

from chipforge_asap7 import cli


def test_every_generator_has_a_main_that_takes_argv():
    for _, table in cli.GROUPS.values():
        modules = [table] if isinstance(table, str) else [m for m, _ in table.values()]
        for module in modules:
            main = importlib.import_module(module).main
            assert "argv" in main.__code__.co_varnames[: main.__code__.co_argcount]


def test_help_lists_the_groups_and_a_groups_generators(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert all(group in out for group in cli.GROUPS)
    assert cli.main(["rom"]) == 2
    out = capsys.readouterr().out
    assert "usage: asap7 rom {nor,brom,brom-timing,nor-read,nor-address-search}" in out


@pytest.mark.parametrize(
    "argv, message",
    [(["nope"], "unknown group 'nope'"), (["rom", "nope"], "unknown generator 'nope'")],
)
def test_unknown_names_are_usage_errors(capsys, argv, message):
    assert cli.main(argv) == 2
    assert message in capsys.readouterr().err


def test_a_generator_gets_the_rest_of_the_line_under_its_own_name(monkeypatch):
    seen = {}

    def fake_main(argv):
        import sys

        seen["argv"], seen["prog"] = argv, sys.argv[0]
        return 3

    monkeypatch.setattr("chipforge_asap7.brom.timing.main", fake_main)
    assert cli.main(["rom", "brom-timing", "--word-bits", "8"]) == 3
    assert seen == {"argv": ["--word-bits", "8"], "prog": "asap7 rom brom-timing"}
