"""Command lines for the cell generators, built from their specs by tyro.

Every generator's options are its spec's fields: tyro reads the names, types,
defaults and the ``Args:`` section of the spec's docstring, so a new field is a
new flag with its help already written.  Tuples are space-separated
(``--rows 18 18 13 13``), ``Literal`` fields are choices, a spec nested in
another is prefixed with its field (``--mux.selects 8``), and a boolean that
defaults on is turned off with ``--no-<name>``.

tyro comes with the ``gds`` extra, alongside gdspy, and is imported only here,
so the specs themselves stay free of it.
"""

from __future__ import annotations

import dataclasses
import sys
import typing
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any

from ..layout.layers import require_gdspy

__all__ = ["Option", "parse_spec", "write_gds"]


@dataclasses.dataclass(frozen=True)
class Option:
    """A command-line option that is not a spec field, such as ``--group``."""

    name: str
    type: Any
    default: Any
    help: str


def _require_tyro() -> Any:
    try:
        import tyro
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "the generator command lines require tyro; install it with "
            "'pip install chipforge-asap7[gds]'"
        ) from exc
    return tyro


def _with_hidden(tyro: Any, spec_type: type, hide: Sequence[str]) -> type:
    """`spec_type` with the `hide` fields dropped from the command line.

    Hiding is a tyro annotation, and the spec modules must not import tyro, so
    the annotation goes on a subclass that exists only while parsing.
    """
    hints = typing.get_type_hints(spec_type)
    fields = []
    for field in dataclasses.fields(spec_type):
        if field.name in hide:
            fields.append(
                (
                    field.name,
                    Annotated[hints[field.name], tyro.conf.Suppress],
                    dataclasses.field(default=field.default),
                )
            )
    unknown = set(hide) - {name for name, *_ in fields}
    if unknown:
        raise ValueError(f"{spec_type.__name__} has no fields {sorted(unknown)}")
    return dataclasses.make_dataclass(
        spec_type.__name__, fields, bases=(spec_type,), frozen=True
    )


def parse_spec(
    spec_type: type | None,
    argv: Sequence[str] | None,
    *,
    description: str,
    hide: Sequence[str] = (),
    options: Sequence[Option] = (),
) -> Any:
    """Parse `argv` into ``.spec`` (a `spec_type`), ``.out`` and each of `options`.

    A spec that rejects the values given (its ``__post_init__`` raises) is
    reported as a usage error rather than a traceback.  `spec_type` may be
    ``None`` for a generator whose options are not a spec's fields.
    """
    tyro = _require_tyro()
    fields: list[tuple[str, Any, Any]] = []
    if spec_type is not None:
        parsed_type = _with_hidden(tyro, spec_type, hide) if hide else spec_type
        fields.append(
            (
                "spec",
                Annotated[parsed_type, tyro.conf.arg(name="")],
                dataclasses.field(default_factory=parsed_type),
            )
        )
    for option in options:
        fields.append(
            (
                option.name,
                Annotated[option.type, tyro.conf.arg(help=option.help)],
                dataclasses.field(default=option.default),
            )
        )
    fields.append(
        (
            "out",
            Annotated[
                Path | None,
                tyro.conf.arg(help="Output GDS path; the cell name when omitted."),
            ],
            dataclasses.field(default=None),
        )
    )
    arguments = dataclasses.make_dataclass("Arguments", fields)
    try:
        args = tyro.cli(
            arguments,
            args=argv,
            description=description,
            config=(tyro.conf.FlagCreatePairsOff,),
        )
        if spec_type is not None and hide:
            args.spec = spec_type(
                **{
                    f.name: getattr(args.spec, f.name)
                    for f in dataclasses.fields(spec_type)
                }
            )
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    return args


def write_gds(build: Callable[[Any], Any], out: Path | None) -> tuple[Any, Path]:
    """Build a cell into a fresh nanometre library and write it; return ``(cell, path)``.

    `build` takes the library and returns the top cell.  Without `out` the file
    is named after that cell, in the current directory.
    """
    gdspy = require_gdspy()
    library = gdspy.GdsLibrary(unit=1e-9, precision=1e-10)
    # A sub-cell drawn without an explicit library lands in gdspy's current
    # one; point that at this file while building, and put it back after.
    previous, gdspy.current_library = gdspy.current_library, library
    try:
        cell = build(library)
    finally:
        gdspy.current_library = previous
    path = out or Path(f"{cell.name}.gds")
    path.parent.mkdir(parents=True, exist_ok=True)
    library.write_gds(str(path))
    return cell, path
