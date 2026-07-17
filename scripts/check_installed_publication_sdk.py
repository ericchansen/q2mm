#!/usr/bin/env python3
"""Exercise provisionable publication rows through an installed q2mm wheel."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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
from q2mm.io.mm3 import load_mm3_fld
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.results import OptimizationResult
from q2mm.objectives.protocols import ObjectiveEvaluator

_PROVENANCE = BackendProvenance(backend="publication-install-proof", role=BackendRole.MM)
_INFO = BackendInfo(
    name="Publication installed-wheel proof",
    role=BackendRole.MM,
    capabilities=frozenset({Capability.ENERGY, Capability.MINIMIZE, Capability.HESSIAN}),
    functional_forms=frozenset({"mm3"}),
    provenance=_PROVENANCE,
)


class _Prepared(AbstractPreparedBackend):
    def __init__(self, request: PreparationRequest) -> None:
        assert request.force_field is not None
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
            raise ValueError(f"Publication case {self.case_id!r} has no Hessian.")
        return HessianResult(
            hessian=self.molecule.hessian,
            unit=HessianUnit.HARTREE_PER_BOHR2,
            provenance=_PROVENANCE,
        )


class _Backend:
    info = _INFO

    def prepare(self, request: PreparationRequest) -> _Prepared:
        return _Prepared(request)


class _OneEvaluationOptimizer:
    def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
        initial = np.array(space.baseline, copy=True)
        score = evaluator.value(initial)
        return OptimizationResult(
            success=True,
            message="installed publication SDK path entered",
            initial_score=score,
            final_score=score,
            n_iterations=1,
            n_evaluations=1,
            n_params=space.n_full,
            layout_fingerprint=space.layout.fingerprint,
            initial_params=initial,
            final_params=initial,
            history=(score,),
            method="installed-publication-smoke",
            gradient_mode="none",
        )


def _run(args: argparse.Namespace) -> None:
    roots = ExternalDataRoots(
        supporting_info=args.supporting_info,
        mm3_base=args.mm3_base,
        rh_enamide=args.rh_enamide,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    records = tuple(record for record in publication_records() if record.provisionable)
    for record in records:
        case = load_system(
            record.system,
            data_roots=roots,
            starting_point=record.starting_point,
            objective_profile=record.objective_profile.identifier,
            functional_form="mm3",
        )
        problem = case.problem
        backend = _Backend()
        baseline = q2mm.evaluate(problem, backend=backend, executor="python")
        if not np.isfinite(baseline.total):
            raise RuntimeError(f"{record.system} produced a non-finite installed baseline.")
        run = q2mm.optimize(
            problem,
            backend=backend,
            recipe="explicit",
            optimizer=_OneEvaluationOptimizer(),
            workflow="single-stage",
            executor="python",
            n_evals=0,
        )
        frozen = np.ones(problem.active_space.n_full, dtype=bool)
        frozen[problem.active_space.active_indices] = False
        np.testing.assert_array_equal(run.result.final_params[frozen], problem.active_space.baseline[frozen])
        stem = f"{record.system}-{record.objective_profile.identifier}-{record.starting_point}"
        saved = q2mm.save(run, args.output / f"{stem}.fld")
        if saved.manifest_path is None or not saved.manifest_path.is_file():
            raise RuntimeError(f"{stem} wrote no installed manifest.")
        roundtrip = load_mm3_fld(saved.force_field_path)
        if roundtrip.nonbonded_excluded_atom_types != run.final_force_field.nonbonded_excluded_atom_types:
            raise RuntimeError(f"{stem} did not preserve nonbonded exclusions.")


def main() -> int:
    """Run the installed-wheel publication proof."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supporting-info", type=Path, required=True)
    parser.add_argument("--mm3-base", type=Path, required=True)
    parser.add_argument("--rh-enamide", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _run(args)
    print("installed-publication-sdk=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
