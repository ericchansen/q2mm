"""JAX vs OpenMM parity for improper torsions.

Regression test for the bug where ``JaxEngine`` silently dropped improper
torsion contributions because :func:`_compile_energy_fn` only iterated
``molecule.torsions`` (proper) and never ``molecule.improper_torsions``.
OpenMM (see ``q2mm/backends/mm/openmm.py``) routes both proper and improper
torsions through the same ``PeriodicTorsionForce``.

The fix appends improper-torsion matches into the same JAX
``torsion_atom_indices``/``torsion_param_map`` arrays, reusing
``_torsion_energy`` (cosine series with periodicity + phase, identical
form to OpenMM's periodic torsion).
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np
import pytest

from q2mm.models.molecule import Molecule

if TYPE_CHECKING:
    from q2mm.models.forcefield import ForceField

_HAS_JAX = importlib.util.find_spec("jax") is not None
_HAS_OPENMM = importlib.util.find_spec("openmm") is not None

pytestmark = [
    pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed"),
    pytest.mark.skipif(not _HAS_OPENMM, reason="OpenMM not installed"),
    pytest.mark.jax,
    pytest.mark.openmm,
]


def _make_formaldehyde(out_of_plane: float = 0.0) -> Molecule:
    """Build HCHO (sp2 C with 3 substituents → 1 improper at C).

    ``out_of_plane`` displaces the carbon along z so the improper dihedral
    is non-zero and the torsion energy contributes a measurable amount.
    """
    return Molecule(
        symbols=["C", "O", "H", "H"],
        geometry=np.array(
            [
                [0.0, 0.0, out_of_plane],
                [1.21, 0.0, 0.0],
                [-0.55, 0.95, 0.0],
                [-0.55, -0.95, 0.0],
            ]
        ),
        name="formaldehyde",
    )


def _make_acetaldehyde(out_of_plane: float = 0.0) -> Molecule:
    """Build CH3-CHO with both impropers and propers.

    Acetaldehyde has an sp2 carbonyl C (→ 1 improper) plus a C-C bond
    that produces real proper torsions (H_me-C2-C1-O, H_me-C2-C1-H_ald).
    ``out_of_plane`` displaces the carbonyl C along z so the improper
    dihedral is non-zero.
    """
    return Molecule(
        symbols=["C", "O", "H", "C", "H", "H", "H"],
        geometry=np.array(
            [
                [0.0, 0.0, out_of_plane],  # C1 (carbonyl)
                [1.21, 0.0, 0.0],  # O
                [-0.55, 0.95, 0.0],  # H_ald
                [-0.75, -1.30, 0.0],  # C2 (methyl)
                [-0.30, -2.20, 0.85],  # H_me1
                [-1.85, -1.30, 0.0],  # H_me2
                [-0.30, -2.20, -0.85],  # H_me3
            ]
        ),
        name="acetaldehyde",
    )


def _make_ff_with_improper(
    *,
    periodicity: int = 2,
    phase: float = 180.0,
    k_improper: float = 1.5,
    include_proper: bool = False,
) -> ForceField:
    """Tiny FF: bonds + angles + one improper (and optional proper) torsion.

    Uses the harmonic functional form so JAX and OpenMM share an identical
    energy expression for every term.
    """
    from q2mm.models.forcefield import (
        AngleParam,
        BondParam,
        ForceField,
        FunctionalForm,
        TorsionParam,
    )

    torsions = [
        TorsionParam(
            elements=("H", "H", "C", "O"),
            periodicity=periodicity,
            force_constant=k_improper,
            phase=phase,
            is_improper=True,
        )
    ]
    if include_proper:
        torsions.append(
            TorsionParam(
                elements=("H", "C", "C", "O"),
                periodicity=3,
                force_constant=0.5,
                phase=0.0,
                is_improper=False,
            )
        )

    return ForceField(
        functional_form=FunctionalForm.HARMONIC,
        bonds=[
            BondParam(elements=("C", "O"), force_constant=600.0, equilibrium=1.21),
            BondParam(elements=("C", "H"), force_constant=340.0, equilibrium=1.10),
        ],
        angles=[
            AngleParam(elements=("O", "C", "H"), force_constant=50.0, equilibrium=120.0),
            AngleParam(elements=("H", "C", "H"), force_constant=35.0, equilibrium=120.0),
        ],
        torsions=torsions,
    )


def _energy_pair(mol: Molecule, ff: ForceField) -> tuple[float, float]:
    """Return (jax_energy, openmm_energy) for a single (mol, ff).

    Forces OpenMM onto the CPU platform so parity assertions are
    deterministic across machines (GPU mixed-precision can drift the
    result by more than ``1e-6 kcal/mol``).
    """
    from q2mm.backends.mm.jax_engine import JaxEngine
    from q2mm.backends.mm.openmm import OpenMMEngine

    return float(JaxEngine().energy(mol, ff)), float(OpenMMEngine(platform_name="CPU").energy(mol, ff))


class TestJaxOpenMMImproperParity:
    def test_planar_zero_improper_at_phase_180_periodicity_2(self) -> None:
        """Planar HCHO: phase=180°, periodicity=2 → improper energy term is 0.

        For a planar geometry the signed dihedral returned by the dihedral
        calculator can be 0 *or* 180 (depending on atom ordering and normal
        direction); the chosen ``phase=180°``/``periodicity=2`` combination
        makes ``E_imp = k(1 + cos(n·φ - phase))`` evaluate to 0 in either
        case, so this asserts whole-energy parity in the trivial regime.
        """
        mol = _make_formaldehyde(out_of_plane=0.0)
        ff = _make_ff_with_improper(periodicity=2, phase=180.0, k_improper=1.5)
        e_jax, e_omm = _energy_pair(mol, ff)
        np.testing.assert_allclose(e_jax, e_omm, atol=1e-6)

    def test_pyramidalized_improper_contributes(self) -> None:
        """Out-of-plane C: improper energy is non-zero and must match OpenMM.

        Pre-fix, JAX would silently drop the improper term and disagree with
        OpenMM by exactly the improper contribution.  Post-fix, energies
        agree to within numerical precision.
        """
        mol = _make_formaldehyde(out_of_plane=0.20)
        ff = _make_ff_with_improper(periodicity=2, phase=180.0, k_improper=2.0)
        e_jax, e_omm = _energy_pair(mol, ff)
        np.testing.assert_allclose(e_jax, e_omm, atol=1e-6)

    def test_proper_and_improper_both_routed(self) -> None:
        """Acetaldehyde exercises both proper and improper routing.

        Formaldehyde alone has no proper torsions (every bond endpoint
        bar the carbonyl C has only one neighbour), so the proper FF
        param would never match.  Acetaldehyde adds H_methyl-C2-C1-O and
        related propers while keeping the sp2 improper at C1.
        """
        mol = _make_acetaldehyde(out_of_plane=0.15)
        ff = _make_ff_with_improper(periodicity=2, phase=180.0, k_improper=1.0, include_proper=True)
        e_jax, e_omm = _energy_pair(mol, ff)
        np.testing.assert_allclose(e_jax, e_omm, atol=1e-6)

    def test_pre_fix_regression_guard(self) -> None:
        """Without the fix, JAX would equal an FF with the improper removed.

        Verifies that the improper term genuinely contributes to JaxEngine
        output (not just OpenMM) by comparing against a no-improper FF.
        """
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.models.forcefield import (
            AngleParam,
            BondParam,
            ForceField,
            FunctionalForm,
        )

        mol = _make_formaldehyde(out_of_plane=0.20)
        ff_with = _make_ff_with_improper(periodicity=2, phase=180.0, k_improper=2.0)
        ff_without = ForceField(
            functional_form=FunctionalForm.HARMONIC,
            bonds=[
                BondParam(elements=("C", "O"), force_constant=600.0, equilibrium=1.21),
                BondParam(elements=("C", "H"), force_constant=340.0, equilibrium=1.10),
            ],
            angles=[
                AngleParam(elements=("O", "C", "H"), force_constant=50.0, equilibrium=120.0),
                AngleParam(elements=("H", "C", "H"), force_constant=35.0, equilibrium=120.0),
            ],
            torsions=[],
        )
        eng = JaxEngine()
        e_with = float(eng.energy(mol, ff_with))
        e_without = float(eng.energy(mol, ff_without))
        # If the improper is being routed, the two energies must differ by
        # the improper contribution (definitely non-trivial at out_of_plane=0.20).
        assert abs(e_with - e_without) > 1e-3, (
            f"JAX improper appears to be silently dropped: with-improper={e_with}, without-improper={e_without}"
        )
