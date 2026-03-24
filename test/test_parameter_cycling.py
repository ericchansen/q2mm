"""Unit tests for parameter cycling: ForceField indices, SubspaceObjective, sensitivity."""

import numpy as np
import pytest

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, TorsionParam, VdwParam


# ---- Fixtures ----


def _full_ff() -> ForceField:
    """Build a FF with all parameter types for testing indices."""
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
    )


# ---- TestParamIndices ----


class TestParamIndicesByType:
    def test_correct_keys(self) -> None:
        ff = _full_ff()
        indices = ff.get_param_indices_by_type()
        expected_keys = {"bond_k", "bond_eq", "angle_k", "angle_eq", "torsion_k", "vdw_radius", "vdw_epsilon"}
        assert set(indices.keys()) == expected_keys

    def test_bond_indices(self) -> None:
        ff = _full_ff()
        indices = ff.get_param_indices_by_type()
        # 2 bonds: [k0, eq0, k1, eq1, ...]
        assert indices["bond_k"] == [0, 2]
        assert indices["bond_eq"] == [1, 3]

    def test_angle_indices(self) -> None:
        ff = _full_ff()
        indices = ff.get_param_indices_by_type()
        # 1 angle starts after 2 bonds (index 4)
        assert indices["angle_k"] == [4]
        assert indices["angle_eq"] == [5]

    def test_torsion_indices(self) -> None:
        ff = _full_ff()
        indices = ff.get_param_indices_by_type()
        # 2 torsions start after bonds(4) + angles(2) = index 6
        assert indices["torsion_k"] == [6, 7]

    def test_vdw_indices(self) -> None:
        ff = _full_ff()
        indices = ff.get_param_indices_by_type()
        # 1 vdw starts after bonds(4) + angles(2) + torsions(2) = index 8
        assert indices["vdw_radius"] == [8]
        assert indices["vdw_epsilon"] == [9]

    def test_total_indices_match_n_params(self) -> None:
        ff = _full_ff()
        indices = ff.get_param_indices_by_type()
        all_indices = []
        for idx_list in indices.values():
            all_indices.extend(idx_list)
        assert len(all_indices) == ff.n_params
        assert sorted(all_indices) == list(range(ff.n_params))

    def test_empty_ff(self) -> None:
        ff = ForceField(name="empty")
        indices = ff.get_param_indices_by_type()
        for idx_list in indices.values():
            assert idx_list == []

    def test_indices_match_param_vector_values(self) -> None:
        """Verify that indices actually point to the right values."""
        ff = _full_ff()
        vec = ff.get_param_vector()
        indices = ff.get_param_indices_by_type()

        # bond_k[0] should be the first bond's force constant
        assert vec[indices["bond_k"][0]] == 359.7
        # bond_eq[0] should be the first bond's equilibrium
        assert vec[indices["bond_eq"][0]] == 1.38
        # angle_k[0] should be the angle force constant
        assert vec[indices["angle_k"][0]] == 71.9
        # torsion_k[0] should be the first torsion
        assert vec[indices["torsion_k"][0]] == 0.5
        # vdw_radius at index 0
        assert vec[indices["vdw_radius"][0]] == 1.7


class TestParamTypeLabels:
    def test_length_matches(self) -> None:
        ff = _full_ff()
        labels = ff.get_param_type_labels()
        assert len(labels) == ff.n_params

    def test_label_values(self) -> None:
        ff = _full_ff()
        labels = ff.get_param_type_labels()
        # 2 bonds: k, eq, k, eq
        assert labels[0] == "bond_k"
        assert labels[1] == "bond_eq"
        assert labels[2] == "bond_k"
        assert labels[3] == "bond_eq"
        # 1 angle: k, eq
        assert labels[4] == "angle_k"
        assert labels[5] == "angle_eq"
        # 2 torsions: k, k
        assert labels[6] == "torsion_k"
        assert labels[7] == "torsion_k"
        # 1 vdw: radius, epsilon
        assert labels[8] == "vdw_radius"
        assert labels[9] == "vdw_epsilon"


class TestStepSizes:
    def test_length_matches(self) -> None:
        ff = _full_ff()
        steps = ff.get_step_sizes()
        assert len(steps) == ff.n_params

    def test_per_type_values(self) -> None:
        """Step sizes should match the STEPS dict values via the mapping."""
        from q2mm.optimizers.defaults import STEPS

        ff = _full_ff()
        steps = ff.get_step_sizes()

        # bond_k → "bf" → 0.1
        assert steps[0] == STEPS["bf"]
        # bond_eq → "be" → 0.02
        assert steps[1] == STEPS["be"]
        # angle_k → "af" → 0.1
        assert steps[4] == STEPS["af"]
        # angle_eq → "ae" → 1.0
        assert steps[5] == STEPS["ae"]
        # torsion_k → "df" → 0.1
        assert steps[6] == STEPS["df"]
        # vdw_radius → "vdwr" → 0.1
        assert steps[8] == STEPS["vdwr"]
        # vdw_epsilon → "vdwfc" → 0.02
        assert steps[9] == STEPS["vdwfc"]

    def test_all_positive(self) -> None:
        ff = _full_ff()
        steps = ff.get_step_sizes()
        assert np.all(steps > 0)


# ---- TestSubspaceObjective ----
# These require OpenMM and are in test/integration/test_cycling_optimizer.py


# ---- TestSensitivity (pure unit, no engine) ----


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


class TestLoopResult:
    def test_improvement(self) -> None:
        from q2mm.optimizers.cycling import LoopResult

        lr = LoopResult(
            success=True,
            initial_score=100.0,
            final_score=10.0,
            n_cycles=3,
        )
        assert lr.improvement == pytest.approx(0.9)

    def test_summary(self) -> None:
        from q2mm.optimizers.cycling import LoopResult

        lr = LoopResult(
            success=True,
            initial_score=100.0,
            final_score=10.0,
            n_cycles=3,
            message="converged",
        )
        s = lr.summary()
        assert "converged" in s
        assert "90.00%" in s

    def test_zero_initial_score(self) -> None:
        from q2mm.optimizers.cycling import LoopResult

        lr = LoopResult(success=True, initial_score=0.0, final_score=0.0, n_cycles=0)
        assert lr.improvement == 0.0
