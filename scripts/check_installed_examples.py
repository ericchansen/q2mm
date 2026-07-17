#!/usr/bin/env python3
"""Run source-only examples against an installed q2mm wheel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

SMALL_EXAMPLES = (
    ("ch3f", 1),
    ("ch3f-sn2", 1),
)
PUBLICATION_EXAMPLES = (
    ("rh-enamide", 9),
    ("heck-relay", 23),
    ("pd-allyl", 21),
    ("pd-conjugate", 10),
    ("rh-conjugate", 10),
    ("ferrocene", 7),
)


class InstalledExampleError(RuntimeError):
    """An installed-wheel example violated its executable contract."""


def _run_json(command: list[str], *, cwd: Path, environment: dict[str, str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=True,
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as exc:
        raise InstalledExampleError(
            f"Example failed: {command}\nstdout:\n{exc.stdout or ''}\nstderr:\n{exc.stderr or ''}"
        ) from exc
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise InstalledExampleError(f"Example did not print one JSON document:\n{completed.stdout}") from exc
    if not isinstance(value, dict):
        raise InstalledExampleError("Example result must be a JSON object.")
    return value


def _assert_output(result: dict[str, Any], output_root: Path, *, expected_cases: int) -> None:
    if result.get("case_count") != expected_cases:
        raise InstalledExampleError(f"Example case count {result.get('case_count')!r}; expected {expected_cases}.")
    if result.get("bounded_ci") is not True:
        raise InstalledExampleError("Installed example did not report bounded_ci=true.")
    optimization = result.get("optimization")
    if not isinstance(optimization, dict) or optimization.get("iterations") != 1:
        raise InstalledExampleError("Installed example did not enter its one-iteration bounded optimizer.")
    if optimization.get("convergence_claim") is not False:
        raise InstalledExampleError("Bounded installed example made a convergence claim.")
    saved = result.get("saved")
    if not isinstance(saved, dict):
        raise InstalledExampleError("Installed example did not report saved paths.")
    root = output_root.resolve()
    for key in ("force_field", "manifest"):
        value = saved.get(key)
        if not isinstance(value, str):
            raise InstalledExampleError(f"Installed example did not report saved {key}.")
        path = Path(value).resolve()
        if root not in path.parents or not path.is_file():
            raise InstalledExampleError(f"Installed example wrote {key} outside its output root: {path}")


def _write_minimal_fchk(path: Path) -> None:
    dimension = 6
    lower = []
    for row in range(dimension):
        for column in range(row + 1):
            lower.append(0.1 if row == column else 0.0)
    values = "\n".join(
        " ".join(f"{value: .8E}" for value in lower[index : index + 5]) for index in range(0, len(lower), 5)
    )
    path.write_text(
        "Generated H2 installed-example input\n"
        "Freq      RHF                           STO-3G\n"
        "Number of atoms                            I                2\n"
        "Charge                                     I                0\n"
        "Multiplicity                               I                1\n"
        "Atomic numbers                             I   N=           2\n"
        "           1           1\n"
        "Current cartesian coordinates              R   N=           6\n"
        "  0.00000000E+00  0.00000000E+00  0.00000000E+00"
        "  1.40000000E+00  0.00000000E+00  0.00000000E+00\n"
        "Cartesian Force Constants                  R   N=          21\n"
        f"{values}\n",
        encoding="ascii",
    )


def _verify_installed_import(python: Path, *, repository_root: Path, cwd: Path, environment: dict[str, str]) -> None:
    completed = subprocess.run(
        [str(python), "-I", "-c", "import pathlib,q2mm; print(pathlib.Path(q2mm.__file__).resolve())"],
        check=True,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    imported = Path(completed.stdout.strip()).resolve()
    source_package = (repository_root / "q2mm").resolve()
    if source_package == imported.parent or source_package in imported.parents:
        raise InstalledExampleError(f"Example proof imported q2mm from the source checkout: {imported}")
    if "site-packages" not in {part.lower() for part in imported.parts}:
        raise InstalledExampleError(f"Example proof did not import q2mm from installed site-packages: {imported}")


def run_examples(args: argparse.Namespace) -> tuple[str, str]:
    """Execute all small examples and configured publication examples."""
    python = args.python.resolve()
    examples_root = args.examples_root.resolve()
    output = args.output.resolve()
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    _verify_installed_import(python, repository_root=examples_root.parent, cwd=work, environment=environment)

    for key, expected_cases in SMALL_EXAMPLES:
        script = examples_root / key / "run.py"
        example_output = output / "small" / key
        result = _run_json(
            [
                str(python),
                "-I",
                str(script),
                "--bounded-ci",
                "--output-root",
                str(example_output),
            ],
            cwd=work,
            environment=environment,
        )
        _assert_output(result, example_output, expected_cases=expected_cases)

    fchk = output / "inputs" / "minimal.fchk"
    fchk.parent.mkdir(parents=True, exist_ok=True)
    _write_minimal_fchk(fchk)
    byo_output = output / "small" / "bring-your-own-fchk"
    byo = _run_json(
        [
            str(python),
            "-I",
            str(examples_root / "ch3f" / "run.py"),
            "--fchk",
            str(fchk),
            "--stationary-point",
            "ground_state",
            "--bounded-ci",
            "--output-root",
            str(byo_output),
        ],
        cwd=work,
        environment=environment,
    )
    _assert_output(byo, byo_output, expected_cases=1)
    small_marker = "installed-examples-small=ok"

    roots = (args.supporting_info, args.mm3_base, args.rh_enamide)
    if not any(roots):
        return small_marker, "installed-examples-publication=not-configured"
    if not all(roots):
        raise InstalledExampleError(
            "Publication example proof requires --supporting-info, --mm3-base, and --rh-enamide together."
        )
    for key, expected_cases in PUBLICATION_EXAMPLES:
        script = examples_root / "publication" / key / "run.py"
        example_output = output / "publication" / key
        result = _run_json(
            [
                str(python),
                "-I",
                str(script),
                "--supporting-info",
                str(args.supporting_info),
                "--mm3-base",
                str(args.mm3_base),
                "--rh-enamide",
                str(args.rh_enamide),
                "--bounded-ci",
                "--output-root",
                str(example_output),
            ],
            cwd=work,
            environment=environment,
        )
        _assert_output(result, example_output, expected_cases=expected_cases)
    return small_marker, "installed-examples-publication=ok"


def main() -> int:
    """Run the installed-wheel example matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--examples-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supporting-info", type=Path)
    parser.add_argument("--mm3-base", type=Path)
    parser.add_argument("--rh-enamide", type=Path)
    args = parser.parse_args()
    small, publication = run_examples(args)
    print(small)
    print(publication)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
