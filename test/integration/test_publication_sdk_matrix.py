from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import q2mm
from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    Capability,
    EnergyResult,
    EnergyUnit,
    GeometryResult,
    HessianResult,
    HessianUnit,
    LengthUnit,
    PreparationRequest,
)
from q2mm.benchmarks.publications import publication_records
from q2mm.benchmarks.systems import load_system
from q2mm.benchmarks.systems._paths import ExternalDataRoots
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.results import OptimizationResult
from q2mm.objectives.protocols import ObjectiveEvaluator

pytestmark = [pytest.mark.integration, pytest.mark.external_data]

_PROVENANCE = BackendProvenance(backend="publication-echo", role=BackendRole.MM)
_INFO = BackendInfo(
    name="Publication objective echo",
    role=BackendRole.MM,
    capabilities=frozenset({Capability.ENERGY, Capability.MINIMIZE, Capability.HESSIAN}),
    functional_forms=frozenset({"mm3"}),
    provenance=_PROVENANCE,
)
_ROWS = tuple(record for record in publication_records() if record.provisionable)


class _EchoPrepared(AbstractPreparedBackend):
    def __init__(self, request: PreparationRequest) -> None:
        super().__init__(
            info=_INFO,
            case_id=request.case_id,
            molecule=request.molecule,
            force_field=request.force_field,
            layout=ParameterLayout.from_force_field(request.force_field),
        )

    def _energy(self, request: object) -> EnergyResult:
        return EnergyResult(energy=0.0, unit=EnergyUnit.KCAL_PER_MOL, provenance=_PROVENANCE)

    def _minimize(self, request: object) -> GeometryResult:
        return GeometryResult(
            energy=0.0,
            energy_unit=EnergyUnit.KCAL_PER_MOL,
            symbols=self.molecule.symbols,
            coordinates=self.molecule.geometry,
            coordinate_unit=LengthUnit.ANGSTROM,
            provenance=_PROVENANCE,
        )

    def _hessian(self, request: object) -> HessianResult:
        if self.molecule.hessian is None:
            raise ValueError(f"Publication case {self.case_id!r} has no reference Hessian.")
        return HessianResult(
            hessian=self.molecule.hessian,
            unit=HessianUnit.HARTREE_PER_BOHR2,
            provenance=_PROVENANCE,
        )


class _EchoBackend:
    info = _INFO

    def prepare(self, request: PreparationRequest) -> _EchoPrepared:
        return _EchoPrepared(request)


class _OneEvaluationOptimizer:
    entered: bool

    def __init__(self) -> None:
        self.entered = False

    def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
        self.entered = True
        initial = np.array(space.baseline, copy=True)
        score = evaluator.value(initial)
        return OptimizationResult(
            success=True,
            message="bounded publication SDK path entered",
            initial_score=score,
            final_score=score,
            n_iterations=1,
            n_evaluations=1,
            n_params=space.n_full,
            layout_fingerprint=space.layout.fingerprint,
            initial_params=initial,
            final_params=initial,
            history=(score,),
            method="one-evaluation-smoke",
            gradient_mode="none",
        )


def _roots_or_skip() -> ExternalDataRoots:
    roots = ExternalDataRoots.from_environment()
    missing = []
    if roots.supporting_info is None or not roots.supporting_info.is_dir():
        missing.append("Q2MM_SUPPORTING_INFO")
    if roots.mm3_base is None or not roots.mm3_base.is_file():
        missing.append("Q2MM_MM3_BASE")
    if roots.rh_enamide is None or not roots.rh_enamide.is_dir():
        missing.append("Q2MM_RH_ENAMIDE")
    if missing:
        pytest.skip(f"publication SDK matrix unavailable; configure {', '.join(missing)}")
    return roots


@pytest.mark.parametrize(
    "record",
    _ROWS,
    ids=lambda record: f"{record.system}-{record.objective_profile.identifier}-{record.starting_point}",
)
def test_every_provisionable_publication_row_enters_sdk_optimizer_and_saves(
    record: Any,
    tmp_path: Path,
) -> None:
    roots = _roots_or_skip()
    case = load_system(
        record.system,
        data_roots=roots,
        starting_point=record.starting_point,
        objective_profile=record.objective_profile.identifier,
        functional_form="mm3",
    )
    problem = case.problem
    backend = _EchoBackend()

    baseline = q2mm.evaluate(problem, backend=backend, executor="python")
    assert np.isfinite(baseline.total)
    optimizer = _OneEvaluationOptimizer()
    run = q2mm.optimize(
        problem,
        backend=backend,
        recipe="explicit",
        optimizer=optimizer,
        workflow="single-stage",
        executor="python",
        n_evals=0,
    )
    assert optimizer.entered is True

    initial = problem.layout.vector(problem.starting_force_field)
    final = problem.layout.vector(run.final_force_field)
    frozen = np.ones(problem.active_space.n_full, dtype=bool)
    frozen[problem.active_space.active_indices] = False
    np.testing.assert_array_equal(final[frozen], initial[frozen])

    stem = f"{record.system}-{record.objective_profile.identifier}-{record.starting_point}"
    saved = q2mm.save(run, tmp_path / f"{stem}.fld")
    assert saved.force_field_path.is_file()
    assert saved.manifest_path is not None and saved.manifest_path.is_file()
    manifest = json.loads(saved.manifest_path.read_text(encoding="ascii"))
    publication = manifest["provenance"]["publication_metadata"]
    assert publication["status"] == record.status.value
    assert publication["objective_profile"]["identifier"] == record.objective_profile.identifier
    assert manifest["provenance"]["reproduction_status"] == record.status.value
