"""Small shared helpers for auditable, scoped bitcell minimum-area proofs."""

from __future__ import annotations

import subprocess
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import Any


def validate_rule_source(pdk: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on a changed, missing or additional PDK source file."""
    actual = {
        str(path.relative_to(pdk)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(pdk.rglob("*.yml"))
    }
    if actual != manifest["sha256"]:
        changed = sorted(
            name
            for name in actual.keys() | manifest["sha256"].keys()
            if actual.get(name) != manifest["sha256"].get(name)
        )
        raise ValueError(
            f"gdscheck rule source differs from the proof's pin: {changed}"
        )
    return manifest


def smt_assert(name: str, expression: str) -> str:
    return f"(assert (! {expression} :named {name}))"


def smt_number(value: float) -> str:
    number = Fraction(str(value))
    sign = number < 0
    number = abs(number)
    text = (
        str(number.numerator)
        if number.denominator == 1
        else f"(/ {number.numerator} {number.denominator})"
    )
    return f"(- {text})" if sign else text


def dimension_queries(
    base: str, width: int, height: int, witness: dict[str, float]
) -> dict[str, tuple[str, str]]:
    bounds = {
        "smaller_width": f"(< width {width})",
        "smaller_height": f"(< height {height})",
        "smaller_either_dimension": f"(or (< width {width}) (< height {height}))",
    }
    queries = {
        name: (base + smt_assert(name, bound) + "\n(check-sat)\n(get-proof)\n", "unsat")
        for name, bound in bounds.items()
    }
    equalities = [
        smt_assert(f"drawn_{name}", f"(= {name} {smt_number(value)})")
        for name, value in witness.items()
    ]
    queries["drawn_witness"] = (
        base + "\n".join(equalities) + "\n(check-sat)\n(get-model)\n",
        "sat",
    )
    return queries


def run_query(z3: Path, source: str, path: Path, expected: str) -> dict[str, Any]:
    path.write_text(source)
    result = subprocess.run(
        [str(z3), "-smt2", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    log = path.with_suffix(".z3.txt")
    log.write_text(result.stdout + result.stderr)
    status = result.stdout.splitlines()[0] if result.stdout else "no result"
    if result.returncode or status != expected or "(error" in result.stdout:
        raise RuntimeError(f"{path.name}: expected {expected}, got {status}; see {log}")
    return {
        "status": status,
        "query": path.name,
        "query_sha256": sha256(source.encode()).hexdigest(),
        "output": log.name,
    }


def tool_info(binary: Path, version_flag: str) -> dict[str, str]:
    result = subprocess.run(
        [str(binary), version_flag],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return {
        "path": str(binary),
        "version": result.stdout.strip(),
        "sha256": sha256(binary.read_bytes()).hexdigest(),
    }
