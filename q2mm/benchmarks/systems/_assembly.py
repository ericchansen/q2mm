"""Benchmark metadata and analysis assembly over generic :func:`q2mm.prepare`."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from q2mm.benchmarks.cases import BenchmarkCase
from q2mm.benchmarks.systems._paths import StartingPoint
from q2mm.models.forcefield import ForceField, FunctionalForm
from q2mm.models.hessian import hessian_to_frequencies
from q2mm.models.molecule import Molecule
from q2mm.models.problem import OptimizationProblem, StationaryPointKind
from q2mm.models.publication import PublicationMetadata
from q2mm.preparation import (
    MatchedFrequencyObservations,
    MoleculeObservations,
    QFuerzaConfig,
    prepare,
)


def _validate_starting_point(starting_point: StartingPoint) -> None:
    if starting_point not in ("published", "qfuerza"):
        raise ValueError(f"Unknown starting_point {starting_point!r}; must be one of: 'published', 'qfuerza'")


def _case_ids_for(molecules: list[Molecule], *, key: str) -> tuple[str, ...]:
    """Preserve the established repository case-ID contract."""
    return tuple(f"{molecule.name or key}-{index:03d}" for index, molecule in enumerate(molecules))


def _qm_frequencies(molecule: Molecule) -> np.ndarray:
    if molecule.hessian is None:
        raise ValueError(f"Training molecule {molecule.name!r} has no QM Hessian attached.")
    return np.asarray(hessian_to_frequencies(molecule.hessian, molecule.symbols, sort=False))


def _benchmark_audit(
    problem: OptimizationProblem,
    *,
    starting_point: StartingPoint,
    fresh: bool = False,
) -> dict[str, Any]:
    provenance = problem.preparation_provenance
    if provenance is None:
        raise RuntimeError("Generic preparation did not record its audit provenance.")
    by_type: dict[str, dict[str, int]] = {}
    for kind, raw_counts in provenance.parameter_counts.items():
        counts = dict(raw_counts)
        overwritten = counts["overwritten"]
        retained = counts["retained"]
        if fresh:
            retained += overwritten
            overwritten = 0
        by_type[kind] = {
            "qfuerza_overwritten": overwritten,
            "retained_published": retained,
            "frozen": counts["frozen"],
        }
    return {
        "starting_point": starting_point,
        "n_active": problem.active_space.n_active,
        "n_frozen": problem.active_space.n_full - problem.active_space.n_active,
        "by_type": by_type,
    }


def assemble_published_case(
    *,
    key: str,
    name: str,
    molecules: list[Molecule],
    composed_ff: ForceField,
    opt_only_ff: ForceField,
    stationary_point: StationaryPointKind,
    starting_point: StartingPoint,
    qfuerza_replace_with: float,
    functional_form: str | None,
    metadata: Mapping[str, Any],
    publication_metadata: PublicationMetadata,
    metal: str | None = None,
    normal_modes: Mapping[str, np.ndarray] | None = None,
    default_forms: tuple[str, ...] = ("mm3",),
    description: str = "",
    case_ids: tuple[str, ...] | None = None,
) -> BenchmarkCase:
    """Prepare a publication-template benchmark and retain benchmark-only data."""
    del metal
    _validate_starting_point(starting_point)
    if functional_form is not None:
        composed_ff = dataclasses.replace(composed_ff, functional_form=FunctionalForm(functional_form))
    resolved_case_ids = _case_ids_for(molecules, key=key) if case_ids is None else tuple(case_ids)
    if len(resolved_case_ids) != len(molecules):
        raise ValueError("Explicit publication case IDs must have one entry per molecule.")
    problem = prepare(
        molecules,
        stationary_point=stationary_point,
        force_field=composed_ff,
        active_parameters=opt_only_ff,
        observations=MoleculeObservations(),
        case_ids=resolved_case_ids,
        functional_form=composed_ff.functional_form,
        initialize="provided" if starting_point == "published" else "qfuerza",
        qfuerza=None if starting_point == "published" else QFuerzaConfig(replace_with=qfuerza_replace_with),
    )
    provenance = problem.preparation_provenance
    if provenance is None:
        raise RuntimeError("Generic preparation did not record its audit provenance.")
    objective_profile = publication_metadata.objective_profile.identifier
    if provenance.profile != objective_profile:
        recipe = dict(provenance.observation_recipe)
        recipe["profile"] = objective_profile
        provenance = dataclasses.replace(
            provenance,
            profile=objective_profile,
            observation_recipe=recipe,
        )
    source_ids = publication_metadata.authoritative_case_ids
    if len(source_ids) != len(problem.cases):
        raise ValueError("Publication authoritative case IDs must have one entry per prepared problem case.")
    problem = dataclasses.replace(
        problem,
        cases=tuple(
            dataclasses.replace(case, source_id=source_id)
            for case, source_id in zip(problem.cases, source_ids, strict=True)
        ),
        preparation_provenance=provenance,
        publication_metadata=publication_metadata,
    )
    qm_freqs_per_mol = tuple(
        np.asarray(sorted(value for value in _qm_frequencies(molecule) if value > 50.0)) for molecule in molecules
    )
    resolved_form = composed_ff.functional_form.value
    full_metadata: dict[str, Any] = {
        "molecule_name": name,
        "n_molecules": len(molecules),
        "n_atoms_per_mol": [len(molecule.symbols) for molecule in molecules],
        "starting_point": starting_point,
        "starting_point_audit": _benchmark_audit(problem, starting_point=starting_point),
        "objective_profile": objective_profile,
        "reproduction_status": publication_metadata.status.value,
        "publication_metadata": publication_metadata.to_dict(),
        "publication_metadata_fingerprint": publication_metadata.fingerprint,
        **dict(metadata),
        "functional_form": resolved_form,
    }
    return BenchmarkCase(
        key=key,
        name=name,
        problem=problem,
        qm_freqs_per_mol=qm_freqs_per_mol,
        metadata=full_metadata,
        normal_modes=normal_modes,
        default_forms=default_forms,
        description=description,
    )


def assemble_qfuerza_fresh_case(
    *,
    key: str,
    name: str,
    molecule: Molecule,
    stationary_point: StationaryPointKind,
    backend: Any,
    starting_point: StartingPoint,
    qfuerza_replace_with: float,
    functional_form: str,
    metadata: Mapping[str, Any],
    normal_modes_path: Callable[[Path | None], Path | None] | None = None,
    data_dir: Path | None = None,
    default_forms: tuple[str, ...] = ("harmonic", "mm3"),
    description: str = "",
) -> BenchmarkCase:
    """Prepare a fresh single-molecule benchmark and retain analysis inputs."""
    _validate_starting_point(starting_point)
    qm_frequencies = _qm_frequencies(molecule)
    problem = prepare(
        molecule,
        stationary_point=stationary_point,
        observations=MatchedFrequencyObservations(
            qm_frequencies=tuple(float(value) for value in qm_frequencies),
            backend=backend,
        ),
        case_ids=(key,),
        functional_form=functional_form,
        qfuerza=QFuerzaConfig(replace_with=qfuerza_replace_with),
    )
    qm_real = np.asarray([observation.value for observation in problem.observations.values])
    normal_modes: dict[str, np.ndarray] | None = None
    if normal_modes_path is not None:
        modes_path = normal_modes_path(data_dir)
        if modes_path is not None and modes_path.exists():
            from q2mm.io.reference import load_normal_modes

            normal_modes = load_normal_modes(modes_path)
    full_metadata: dict[str, Any] = {
        "molecule_name": name,
        "n_molecules": 1,
        "n_atoms_per_mol": [len(molecule.symbols)],
        "starting_point": starting_point,
        "starting_point_audit": _benchmark_audit(
            problem,
            starting_point=starting_point,
            fresh=True,
        ),
        **dict(metadata),
        "functional_form": functional_form,
    }
    return BenchmarkCase(
        key=key,
        name=name,
        problem=problem,
        qm_freqs_per_mol=(qm_real,),
        metadata=full_metadata,
        normal_modes=normal_modes,
        default_forms=default_forms,
        description=description,
    )
