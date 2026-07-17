"""Freeze path-free compatibility identities for publication benchmark problems."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from q2mm._canonical import canonical_fingerprint
from q2mm.application.models import (
    molecule_fingerprint_payload,
    problem_fingerprint,
    problem_fingerprint_payload,
)
from q2mm.benchmarks.systems import load_system
from q2mm.benchmarks.systems._paths import ExternalDataRoots, natural_sort_key

_SYSTEMS = ("rh-enamide", "heck-relay", "pd-allyl", "pd-conjugate", "rh-conjugate")
_STARTING_POINTS = ("published", "qfuerza")
_PROFILE = "repository-geometry-eigenmatrix-v1"
_SCHEMA_VERSION = 1


def _fingerprint(value: Any) -> str:
    return canonical_fingerprint(value, screen_secrets=True)


def _compatibility_row(system: str, starting_point: str, roots: ExternalDataRoots) -> dict[str, Any]:
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        problem = load_system(
            system,
            data_roots=roots,
            starting_point=starting_point,
            functional_form="mm3",
        ).problem

    payload = problem_fingerprint_payload(problem)
    observations = problem.observations.values
    vector = problem.layout.vector(problem.starting_force_field)
    molecule_fingerprints = [_fingerprint(molecule_fingerprint_payload(case.molecule)) for case in problem.cases]
    observation_indices = [
        {
            "case_id": observation.case_id,
            "data_idx": int(observation.data_idx),
            "atom_indices": None
            if observation.atom_indices is None
            else [int(index) for index in observation.atom_indices],
        }
        for observation in observations
    ]
    if system == "rh-enamide":
        if roots.rh_enamide is None:
            raise ValueError("The Rh-enamide root is required to freeze its natural source order.")
        source_order = [
            path.stem
            for path in sorted(
                (roots.rh_enamide / "rh_enamide_training_set" / "jaguar_spe_freq_in_out").glob("*.in"),
                key=natural_sort_key,
            )
        ]
    else:
        source_order = [case.molecule.name for case in problem.cases]
    if len(source_order) != len(problem.cases):
        raise ValueError(f"{system} source-order length {len(source_order)} does not match {len(problem.cases)} cases.")
    return {
        "system": system,
        "starting_point": starting_point,
        "functional_form": "mm3",
        "profile": _PROFILE,
        "source_order": source_order,
        "molecule_names": [case.molecule.name for case in problem.cases],
        "case_ids": list(problem.case_ids),
        "molecule_order_fingerprint": _fingerprint(molecule_fingerprints),
        "starting_vector": {
            "count": len(vector),
            "fingerprint": _fingerprint(vector),
        },
        "layout": {
            "count": len(problem.layout),
            "fingerprint": problem.layout.fingerprint,
        },
        "active_indices": {
            "count": problem.active_space.n_active,
            "fingerprint": _fingerprint(problem.active_space.active_indices),
        },
        "observations": {
            "count": len(observations),
            "kind_histogram": dict(sorted(Counter(observation.kind for observation in observations).items())),
            "values_fingerprint": _fingerprint([float(observation.value) for observation in observations]),
            "weights_fingerprint": _fingerprint([float(observation.weight) for observation in observations]),
            "indices_fingerprint": _fingerprint(observation_indices),
            "identity_fingerprint": _fingerprint(payload["observations"]),
        },
        "stationary_points": [case.stationary_point.value for case in problem.cases],
        "problem_fingerprint": problem_fingerprint(problem),
        "baseline_evaluation": None,
        "baseline_evaluation_policy": "compare-old-new-in-process",
    }


def _write_incremental(output: Path, rows: list[dict[str, Any]]) -> None:
    document = {
        "schema_version": _SCHEMA_VERSION,
        "profile": _PROFILE,
        "incremental_per_row": True,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    sibling = output.with_name(f".{output.name}.tmp")
    sibling.write_text(f"{json.dumps(document, indent=2, sort_keys=True)}\n", encoding="utf-8")
    os.replace(sibling, output)


def main() -> int:
    """Generate the compatibility fixture from explicitly supplied external roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supporting-info", required=True, type=Path)
    parser.add_argument("--mm3-base", required=True, type=Path)
    parser.add_argument("--rh-enamide", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test/fixtures/publication_problem_compatibility.json"),
    )
    args = parser.parse_args()
    roots = ExternalDataRoots(
        supporting_info=args.supporting_info,
        mm3_base=args.mm3_base,
        rh_enamide=args.rh_enamide,
    )
    rows: list[dict[str, Any]] = []
    for system in _SYSTEMS:
        for starting_point in _STARTING_POINTS:
            rows.append(_compatibility_row(system, starting_point, roots))
            _write_incremental(args.output, rows)
            print(f"froze {system}/{starting_point}/mm3", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
