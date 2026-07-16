"""Unit tests for parameter layouts and cycling result containers."""

from __future__ import annotations

import numpy as np
import pytest

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, TorsionParam, VdwParam, FunctionalForm
from q2mm.models.parameters import ParameterKind, ParameterLayout


def _full_ff() -> ForceField:
    """Build a FF with all parameter types for testing layout ordering."""
    return ForceField(
        name="test-full",
        bonds=[
            BondParam(elements=("C", "F"), force_constant=359.7, equilibrium=1.38),
            BondParam(elements=("C", "H"), force_constant=338.1, equilibrium=1.11),
        ],
        angles=[
            AngleParam(elements=("F", "C", "F"), force_constant=71.9, equilibrium=109.5),
        ],
        torsions=[
            TorsionParam(elements=("F", "C", "C", "F"), force_constant=0.5),
            TorsionParam(elements=("H", "C", "C", "H"), force_constant=0.3),
        ],
        vdws=[
            VdwParam(atom_type="C1", radius=1.7, epsilon=0.05),
        ],
        functional_form=FunctionalForm.HARMONIC,
    )


class TestParamIndicesByType:
    def test_correct_keys(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        indices = layout.indices_by_kind
        expected_keys = {
            ParameterKind.BOND_FORCE_CONSTANT,
            ParameterKind.BOND_EQUILIBRIUM,
            ParameterKind.ANGLE_FORCE_CONSTANT,
            ParameterKind.ANGLE_EQUILIBRIUM,
            ParameterKind.TORSION_FORCE_CONSTANT,
            ParameterKind.VDW_RADIUS,
            ParameterKind.VDW_EPSILON,
        }
        assert set(indices.keys()) == expected_keys

    def test_bond_indices(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        indices = layout.indices_by_kind
        assert indices[ParameterKind.BOND_FORCE_CONSTANT] == (0, 2)
        assert indices[ParameterKind.BOND_EQUILIBRIUM] == (1, 3)

    def test_angle_indices(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        indices = layout.indices_by_kind
        assert indices[ParameterKind.ANGLE_FORCE_CONSTANT] == (4,)
        assert indices[ParameterKind.ANGLE_EQUILIBRIUM] == (5,)

    def test_torsion_indices(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        indices = layout.indices_by_kind
        assert indices[ParameterKind.TORSION_FORCE_CONSTANT] == (6, 7)

    def test_vdw_indices(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        indices = layout.indices_by_kind
        assert indices[ParameterKind.VDW_RADIUS] == (8,)
        assert indices[ParameterKind.VDW_EPSILON] == (9,)

    def test_total_indices_match_layout_length(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        indices = layout.indices_by_kind
        all_indices: list[int] = []
        for idx_list in indices.values():
            all_indices.extend(idx_list)
        assert len(all_indices) == len(layout)
        assert sorted(all_indices) == list(range(len(layout)))

    def test_empty_ff(self) -> None:
        layout = ParameterLayout.from_force_field(ForceField(name="empty", functional_form=FunctionalForm.HARMONIC))
        assert layout.indices_by_kind == {}

    def test_indices_match_param_vector_values(self) -> None:
        ff = _full_ff()
        layout = ParameterLayout.from_force_field(ff)
        vec = layout.vector(ff)
        indices = layout.indices_by_kind
        assert vec[indices[ParameterKind.BOND_FORCE_CONSTANT][0]] == 359.7
        assert vec[indices[ParameterKind.BOND_EQUILIBRIUM][0]] == 1.38
        assert vec[indices[ParameterKind.ANGLE_FORCE_CONSTANT][0]] == 71.9
        assert vec[indices[ParameterKind.TORSION_FORCE_CONSTANT][0]] == 0.5
        assert vec[indices[ParameterKind.VDW_RADIUS][0]] == 1.7


class TestParamTypeLabels:
    def test_length_matches(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        labels = [kind.value for kind in layout.kinds]
        assert len(labels) == len(layout)

    def test_label_values(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        labels = [kind.value for kind in layout.kinds]
        assert labels[0] == "bond_k"
        assert labels[1] == "bond_eq"
        assert labels[2] == "bond_k"
        assert labels[3] == "bond_eq"
        assert labels[4] == "angle_k"
        assert labels[5] == "angle_eq"
        assert labels[6] == "torsion_k"
        assert labels[7] == "torsion_k"
        assert labels[8] == "vdw_radius"
        assert labels[9] == "vdw_epsilon"


class TestStepSizes:
    def test_length_matches(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        steps = layout.steps
        assert len(steps) == len(layout)

    def test_per_type_values(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        steps = layout.steps
        assert steps[0] == pytest.approx(7.2)
        assert steps[1] == pytest.approx(0.02)
        assert steps[4] == pytest.approx(7.2)
        assert steps[5] == pytest.approx(1.0)
        assert steps[6] == pytest.approx(0.1)
        assert steps[8] == pytest.approx(0.1)
        assert steps[9] == pytest.approx(0.02)

    def test_all_positive(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        steps = layout.steps
        assert np.all(steps > 0)


class TestSensitivityResult:
    def test_dataclass_fields(self) -> None:
        from q2mm.optimizers.cycling import SensitivityResult

        sr = SensitivityResult(
            d1=np.array([1.0, 2.0]),
            d2=np.array([0.5, 1.0]),
            simp_var=np.array([0.5, 0.25]),
            ranking=np.array([1, 0]),
            metric="simp_var",
            n_evals=5,
        )
        assert sr.metric == "simp_var"
        assert sr.n_evals == 5
        assert len(sr.ranking) == 2


class TestOptimizationLoopResult:
    def test_improvement(self) -> None:
        from q2mm.models.results import OptimizationResult

        lr = OptimizationResult(
            success=True,
            message="",
            initial_score=100.0,
            final_score=10.0,
            n_iterations=3,
            n_params=0,
            layout_fingerprint="sha256:test",
            initial_params=np.zeros(0),
            final_params=np.zeros(0),
            n_evaluations=500,
        )
        assert lr.improvement == pytest.approx(0.9)

    def test_summary(self) -> None:
        from q2mm.models.results import OptimizationResult

        lr = OptimizationResult(
            success=True,
            initial_score=100.0,
            final_score=10.0,
            n_iterations=3,
            n_params=0,
            layout_fingerprint="sha256:test",
            initial_params=np.zeros(0),
            final_params=np.zeros(0),
            n_evaluations=500,
            message="converged",
        )
        s = lr.summary()
        assert "converged" in s
        assert "90.0%" in s

    def test_zero_initial_score(self) -> None:
        from q2mm.models.results import OptimizationResult

        lr = OptimizationResult(
            success=True,
            message="",
            initial_score=0.0,
            final_score=0.0,
            n_iterations=0,
            n_params=0,
            layout_fingerprint="sha256:test",
            initial_params=np.zeros(0),
            final_params=np.zeros(0),
            n_evaluations=0,
        )
        assert lr.improvement == 0.0
