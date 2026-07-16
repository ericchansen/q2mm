"""Case/problem assembly for :mod:`q2mm.benchmarks.systems` modules.

Turns a training-set molecule list plus a starting
:class:`~q2mm.models.forcefield.ForceField` (built by
:mod:`~q2mm.benchmarks.systems._forcefield`) into a fully-populated
immutable :class:`~q2mm.benchmarks.cases.BenchmarkCase`: the
OPT-substructure active/frozen partition, the optional QFUERZA
overwrite of active values (curvature-inverted for genuine transition
states only — see :class:`~q2mm.models.problem.StationaryPointKind`),
the honest starting-point audit, the
:class:`~q2mm.models.observations.ObservationSet` to fit against, and
the assembled :class:`~q2mm.models.problem.OptimizationProblem`.

Two entry points cover every system:

- :func:`assemble_published_case` — systems with a literature OPT
  block (rh-enamide, heck-relay, pd-allyl, pd-conjugate, rh-conjugate).
- :func:`assemble_qfuerza_fresh_case` — systems with no published FF at
  all, built entirely from one molecule's QM Hessian (CH3F, CH3F-SN2).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from q2mm.benchmarks.cases import BenchmarkCase
from q2mm.benchmarks.systems._forcefield import load_qfuerza_fresh
from q2mm.benchmarks.systems._paths import StartingPoint
from q2mm.models.forcefield import ForceField, FunctionalForm
from q2mm.models.hessian import hessian_to_frequencies
from q2mm.models.molecule import Molecule
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import (
    ActiveParameterSpace,
    OptSubstructureMembership,
    ParameterLayout,
    opt_substructure_membership,
)
from q2mm.models.problem import OptimizationProblem, StationaryPointKind, TrainingCase
from q2mm.models.seminario import qfuerza_into

# ---------------------------------------------------------------------------
# Frequency helpers (qfuerza_fresh-strategy systems: CH3F / CH3F-SN2)
# ---------------------------------------------------------------------------


def qm_frequencies_from_hessian(hessian_au: np.ndarray, symbols: Any) -> np.ndarray:
    """Compute harmonic frequencies (cm⁻¹) from a Cartesian Hessian in AU."""
    return np.array(hessian_to_frequencies(hessian_au, list(symbols), sort=False))


def build_frequency_reference(
    qm_freqs: np.ndarray,
    mm_all_freqs: np.ndarray,
    *,
    threshold: float = 50.0,
    weight: float = 0.001,
    case_id: str = "0",
) -> tuple[ObservationSet, np.ndarray]:
    """Build an ObservationSet of matched real-mode frequencies.

    Used only by the ``qfuerza_fresh``-strategy systems (CH3F, CH3F-SN2)
    whose reference is a frequency-only fit against the backend's computed
    frequencies at the starting force field — see the "Frequency-only
    refs for TS" pitfall in AGENTS.md: this is appropriate for CH3F-style
    ground-state/single-molecule benchmarks, not TS publication
    reproduction (those use :meth:`ObservationSet.from_molecules`).
    """
    qm_real = sorted(f for f in qm_freqs if f > threshold)
    mm_real_idx = sorted(i for i, f in enumerate(mm_all_freqs) if f > threshold)
    n = min(len(qm_real), len(mm_real_idx))
    ref = ObservationSet()
    for k in range(n):
        ref = ref.with_frequency(
            float(qm_real[k]),
            data_idx=mm_real_idx[k],
            weight=weight,
            case_id=case_id,
        )
    return ref, np.array(qm_real[:n])


# ---------------------------------------------------------------------------
# Starting-point audit
# ---------------------------------------------------------------------------


def audit_starting_point(
    layout: ParameterLayout,
    active_space: ActiveParameterSpace,
    force_field: ForceField,
    *,
    before_vector: np.ndarray | None,
    starting_point: StartingPoint,
) -> dict[str, Any]:
    """Classify every scalar param as qfuerza/retained_published/frozen.

    Honest accounting of where the starting values come from.  When
    *before_vector* is given (the parameter vector snapshotted before a
    QFUERZA overwrite), any active scalar whose value changed is
    ``qfuerza_overwritten``; any active scalar whose value did not
    change is ``retained_published`` (e.g. an OPT stretch-bend, or an
    active bond/angle QFUERZA could not match to any training
    molecule).  When *before_vector* is ``None`` (the ``published``
    starting point, or the ``qfuerza_fresh`` strategy where every
    parameter is already QFUERZA-derived with no separate "before"
    snapshot to diff against), every active scalar is
    ``retained_published``.

    Returns:
        A nested dict ``{starting_point, n_active, n_frozen, by_type:
        {bond_k: {qfuerza_overwritten, retained_published, frozen}, …}}``.
        ``by_type`` keys are :class:`~q2mm.models.parameters.ParameterKind`
        values (``"bond_k"``, ``"bond_eq"``, ...).

    """
    after_vector = layout.vector(force_field)
    active_mask = np.zeros(len(layout), dtype=bool)
    active_mask[active_space.active_indices] = True
    type_labels = [kind.value for kind in layout.kinds]

    reference_before = before_vector if before_vector is not None else after_vector
    by_type: dict[str, dict[str, int]] = {}
    for label, is_active, after_val, before_val in zip(
        type_labels, active_mask, after_vector, reference_before, strict=True
    ):
        bucket = by_type.setdefault(label, {"qfuerza_overwritten": 0, "retained_published": 0, "frozen": 0})
        if not is_active:
            bucket["frozen"] += 1
        elif before_vector is not None and not np.isclose(before_val, after_val, rtol=0.0, atol=1e-12):
            bucket["qfuerza_overwritten"] += 1
        else:
            bucket["retained_published"] += 1

    return {
        "starting_point": starting_point,
        "n_active": int(active_mask.sum()),
        "n_frozen": int((~active_mask).sum()),
        "by_type": by_type,
    }


# ---------------------------------------------------------------------------
# Case assembly
# ---------------------------------------------------------------------------


def _case_ids_for(molecules: list[Molecule], *, key: str) -> tuple[str, ...]:
    """Stable, deterministic case IDs for a multi-molecule training set.

    Shared by :func:`_training_cases` (which pairs each ID with its
    :class:`TrainingCase`) and :meth:`ObservationSet.from_molecules`
    (which binds each observation to the same ID) so both stay in sync.
    """
    return tuple(f"{mol.name or key}-{i:03d}" for i, mol in enumerate(molecules))


def _training_cases(
    molecules: list[Molecule], *, key: str, stationary_point: StationaryPointKind
) -> tuple[TrainingCase, ...]:
    case_ids = _case_ids_for(molecules, key=key)
    return tuple(
        TrainingCase(case_id=case_id, molecule=mol, stationary_point=stationary_point)
        for case_id, mol in zip(case_ids, molecules)
    )


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
    metal: str | None = None,
    normal_modes: Mapping[str, np.ndarray] | None = None,
    default_forms: tuple[str, ...] = ("mm3",),
    description: str = "",
) -> BenchmarkCase:
    """Assemble a :class:`BenchmarkCase` for a published-FF (OPT-substructure) system.

    Shared by every system using the ``published_opt`` /
    ``published_opt_composed`` strategies (rh-enamide, heck-relay,
    pd-allyl, pd-conjugate, rh-conjugate): identify OPT-substructure
    membership, optionally overwrite the active OPT bond/angle values
    via QFUERZA (curvature-inverted only when *stationary_point* is
    :attr:`~q2mm.models.problem.StationaryPointKind.TRANSITION_STATE`,
    per Farrugia 2025 / Limé & Norrby 2015), build the
    eigenmatrix+geometry observation set via
    :meth:`~q2mm.models.observations.ObservationSet.from_molecules`,
    and assemble the immutable :class:`~q2mm.models.problem.OptimizationProblem`.

    Args:
        key: Benchmark registry key.
        name: Human-readable system name.
        molecules: Training-set molecules (QM Hessians attached).
        composed_ff: Full force field (standard MM3 backbone + OPT
            substructure) from
            :func:`~q2mm.benchmarks.systems._forcefield.load_published_opt` or
            :func:`~q2mm.benchmarks.systems._forcefield.compose_opt_with_mm3_base`.
        opt_only_ff: The OPT-substructure-only force field used to
            identify which of *composed_ff*'s parameters are OPT.
        stationary_point: Ground-state or transition-state — decided by
            the calling system module, not this helper. Routes whether
            QFUERZA's TS-curvature inversion runs (see
            :func:`~q2mm.benchmarks.systems._forcefield.load_qfuerza_fresh`
            for why this must never be hardcoded to ``True``).
        starting_point: ``"qfuerza"`` (default) overwrites active OPT
            bond/angle values with multi-molecule QFUERZA estimates;
            ``"published"`` keeps the literature OPT values verbatim.
        qfuerza_replace_with: Replacement value (Hartree/Bohr²) for the
            most-negative TS-Hessian eigenvalue during QFUERZA
            projection.
        functional_form: Optional override (``"harmonic"`` or ``"mm3"``).
        metadata: Static metadata merged into the case's metadata
            (level of theory, publication, DOI, etc.).
        metal: Unused here (kept for call-site symmetry with
            :func:`~q2mm.benchmarks.systems._forcefield.compose_opt_with_mm3_base`);
            present for callers that pass it straight through.
        normal_modes: Optional pre-computed normal modes for PES
            distortion analysis.
        default_forms: Functional forms to benchmark by default.
        description: One-line CLI description.

    Returns:
        A fully-populated :class:`BenchmarkCase`.

    Raises:
        ValueError: If *starting_point* is not ``"published"`` or ``"qfuerza"``.

    """
    del metal  # accepted for call-site symmetry; membership already reflects it
    if starting_point not in ("published", "qfuerza"):
        raise ValueError(f"Unknown starting_point {starting_point!r}; must be one of: 'published', 'qfuerza'")

    if functional_form is not None:
        composed_ff = dataclasses.replace(composed_ff, functional_form=FunctionalForm(functional_form))

    membership: OptSubstructureMembership = opt_substructure_membership(composed_ff, opt_only_ff)
    layout = ParameterLayout.from_force_field(composed_ff)

    before_vector: np.ndarray | None = None
    if starting_point == "qfuerza":
        before_vector = layout.vector(composed_ff)
        composed_ff = qfuerza_into(
            composed_ff,
            molecules,
            active_bonds=membership.bonds,
            active_angles=membership.angles,
            active_torsions=membership.torsions,
            invert_ts_curvature=(stationary_point is StationaryPointKind.TRANSITION_STATE),
            replace_with=qfuerza_replace_with,
        )

    active_space = ActiveParameterSpace.from_membership(layout, composed_ff, membership)
    audit = audit_starting_point(
        layout, active_space, composed_ff, before_vector=before_vector, starting_point=starting_point
    )

    case_ids = _case_ids_for(molecules, key=key)
    observations = ObservationSet.from_molecules(molecules, case_ids, eigenmatrix_diagonal_only=False)
    qm_freqs_per_mol = []
    for mol in molecules:
        if mol.hessian is None:
            raise ValueError(f"Training molecule {mol.name!r} has no QM Hessian attached.")
        qm_freqs = qm_frequencies_from_hessian(mol.hessian, mol.symbols)
        qm_freqs_per_mol.append(np.array(sorted(f for f in qm_freqs if f > 50.0)))

    problem = OptimizationProblem(
        cases=tuple(
            TrainingCase(case_id=case_id, molecule=mol, stationary_point=stationary_point)
            for case_id, mol in zip(case_ids, molecules)
        ),
        starting_force_field=composed_ff,
        layout=layout,
        active_space=active_space,
        observations=observations,
    )

    resolved_form = functional_form if functional_form is not None else composed_ff.functional_form.value
    full_metadata: dict[str, Any] = {
        "molecule_name": name,
        "n_molecules": len(molecules),
        "n_atoms_per_mol": [len(m.symbols) for m in molecules],
        "starting_point": starting_point,
        "starting_point_audit": audit,
        **dict(metadata),
        "functional_form": resolved_form,
    }
    return BenchmarkCase(
        key=key,
        name=name,
        problem=problem,
        qm_freqs_per_mol=tuple(qm_freqs_per_mol),
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
    """Assemble a :class:`BenchmarkCase` for a ``qfuerza_fresh``-strategy system.

    Shared by CH3F and the CH3F-SN2 transition state: build a brand-new
    force field entirely from QFUERZA (no published OPT block to start
    from), then a frequency-only reference against the backend's computed
    frequencies at the starting force field (see the "Frequency-only
    refs for TS" pitfall in AGENTS.md — appropriate here, not for TS
    publication reproduction).

    Args:
        key: Benchmark registry key.
        name: Human-readable system name.
        molecule: The single training molecule (QM Hessian attached).
        stationary_point: Ground-state or transition-state — decided by
            the calling system module. Routes whether QFUERZA's
            TS-curvature inversion runs (see
            :func:`~q2mm.benchmarks.systems._forcefield.load_qfuerza_fresh`
            for why this must never be hardcoded to ``True``: a genuine
            ground state's Hessian routinely carries tiny spurious
            negative eigenvalues that inversion would otherwise corrupt).
        backend: MM backend used to compute frequencies at the starting
            force field.
        starting_point: Accepted for interface symmetry with the
            published-FF systems; it is a no-op here (the force field is
            already fully QFUERZA-derived) — the audit records this.
        qfuerza_replace_with: Replacement value (Hartree/Bohr²) for the
            most-negative eigenvalue during QFUERZA projection.
        functional_form: Required (``"harmonic"`` or ``"mm3"``) — CH3F
            and CH3F-SN2 genuinely support both forms (JAX/JAX-MD use
            harmonic, OpenMM/Tinker use MM3) with no scientifically
            correct single default, so the caller must always decide.
            Built directly into the QFUERZA-fresh force field (no
            post-hoc ``dataclasses.replace`` after construction).
        metadata: Static metadata merged into the case's metadata.
        normal_modes_path: Optional callable resolving a pre-computed
            normal-modes ``.npz`` path from an optional *data_dir*
            override.
        data_dir: Optional override directory forwarded to
            *normal_modes_path*.
        default_forms: Functional forms to benchmark by default.
        description: One-line CLI description.

    Returns:
        A fully-populated :class:`BenchmarkCase`.

    Raises:
        ValueError: If *starting_point* is not ``"published"`` or ``"qfuerza"``.

    """
    if starting_point not in ("published", "qfuerza"):
        raise ValueError(f"Unknown starting_point {starting_point!r}; must be one of: 'published', 'qfuerza'")

    ff = load_qfuerza_fresh(
        molecule,
        functional_form=FunctionalForm(functional_form),
        invert_ts_curvature=(stationary_point is StationaryPointKind.TRANSITION_STATE),
        replace_with=qfuerza_replace_with,
    )

    layout = ParameterLayout.from_force_field(ff)
    active_space = ActiveParameterSpace.all_active(layout, ff)
    # qfuerza_fresh strategy: every parameter is already QFUERZA-derived;
    # there is no separate "before" snapshot to diff against, so
    # starting_point is a no-op — the audit records this honestly.
    audit = audit_starting_point(layout, active_space, ff, before_vector=None, starting_point=starting_point)

    if molecule.hessian is None:
        raise ValueError(f"Training molecule {molecule.name!r} has no QM Hessian attached.")
    qm_freqs_all = qm_frequencies_from_hessian(molecule.hessian, molecule.symbols)
    from q2mm.backends.contracts import FrequencyRequest, PreparationRequest

    prepared = backend.prepare(PreparationRequest(case_id=key, molecule=molecule, force_field=ff))
    mm_all = np.asarray(prepared.frequencies(FrequencyRequest(parameters=layout.vector(ff))).frequencies)
    observations, qm_real = build_frequency_reference(qm_freqs_all, mm_all, case_id=key)

    problem = OptimizationProblem(
        cases=(TrainingCase(case_id=key, molecule=molecule, stationary_point=stationary_point),),
        starting_force_field=ff,
        layout=layout,
        active_space=active_space,
        observations=observations,
    )

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
        "starting_point_audit": audit,
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
