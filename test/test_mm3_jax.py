"""Tests for MM3 functional forms in JaxBackend (issue #91).

Verifies:
- MM3 cubic bond, sextic angle, and Buckingham exp-6 vdW energy functions
- load_backend("jax", functional_form="mm3") produces correct energies
- jax.grad works correctly through all MM3 forms
- Parity with OpenMM MM3 implementation (when OpenMM available)
"""

from __future__ import annotations
from q2mm.backends.contracts import (
    EnergyRequest,
    HessianRequest,
    ParameterGradientRequest,
)
from test.backend_fixtures import param_vector, prepare_case
from q2mm.backends.registry import load_backend

import importlib.util

import numpy as np
import pytest

_HAS_JAX = importlib.util.find_spec("jax") is not None

try:
    import openmm  # noqa: F401

    _HAS_OPENMM = True
except ImportError:
    _HAS_OPENMM = False

pytestmark = [pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed"), pytest.mark.jax]

from test._shared import make_diatomic, make_noble_gas_pair, make_water

from q2mm.models.forcefield import (
    AngleParam,
    BondParam,
    ForceField,
    FunctionalForm,
    StretchBendParam,
    TorsionParam,
    VdwParam,
)
from q2mm.models.molecule import Molecule
from q2mm.models.parameters import ParameterLayout


def _layout(forcefield: ForceField) -> ParameterLayout:
    return ParameterLayout.from_force_field(forcefield)


def _params(forcefield: ForceField) -> np.ndarray:
    return _layout(forcefield).vector(forcefield)


def _materialize(forcefield: ForceField, vector: np.ndarray) -> ForceField:
    return _layout(forcefield).replace(forcefield, vector)


@pytest.fixture(autouse=True)
def _init_jax() -> None:
    """Lazily import JAX before each test so module collection is CUDA-free."""
    from q2mm.backends.mm._jax_common import ensure_jax

    try:
        ensure_jax()
    except ImportError as exc:
        pytest.skip(f"JAX not usable: {exc}")
    # Make jax/jnp and backend symbols available as module globals for tests.
    global jax, jnp, JaxBackend, _mm3_bond_energy, _mm3_angle_energy, _mm3_vdw_energy, _mm3_dipole_energy, _MM3_DIPOLE_CONST  # noqa: PLW0603, E501
    import jax as _jax
    import jax.numpy as _jnp

    from q2mm.backends.mm.jax_engine import (
        JaxBackend as _JaxBackend,
        _MM3_DIPOLE_CONST as _dipole_const,
        _ensure_jax as _ensure_jax_backend,
        _mm3_angle_energy as _angle,
        _mm3_bond_energy as _bond,
        _mm3_dipole_energy as _dipole,
        _mm3_vdw_energy as _vdw,
    )

    # Initialize jax_backend module-level jnp (needed for standalone functions)
    _ensure_jax_backend()

    jax = _jax
    jnp = _jnp
    JaxBackend = _JaxBackend
    _mm3_bond_energy = _bond
    _mm3_angle_energy = _angle
    _mm3_vdw_energy = _vdw
    _mm3_dipole_energy = _dipole
    _MM3_DIPOLE_CONST = _dipole_const


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _h2_ff_mm3() -> ForceField:
    """H₂ force field with MM3 functional form."""
    return ForceField(
        bonds=[BondParam(elements=("H", "H"), force_constant=5.0, equilibrium=0.74)],
        functional_form=FunctionalForm.MM3,
    )


def _water_ff_mm3() -> ForceField:
    """Water force field with MM3 functional form."""
    return ForceField(
        bonds=[BondParam(elements=("H", "O"), force_constant=8.0, equilibrium=0.96)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=0.7, equilibrium=104.5)],
        functional_form=FunctionalForm.MM3,
    )


def _he2_ff_mm3() -> ForceField:
    """He₂ force field with MM3 vdW (no bonds)."""
    return ForceField(
        vdws=[VdwParam(atom_type="He", element="He", radius=1.4, epsilon=0.056)],
        functional_form=FunctionalForm.MM3,
    )


# ---------------------------------------------------------------------------
# Unit-level tests for individual energy kernels
# ---------------------------------------------------------------------------


class TestMM3BondEnergy:
    """Test _mm3_bond_energy against known values."""

    def test_at_equilibrium_zero(self) -> None:
        """Energy is zero at equilibrium distance."""
        k = jnp.array([5.0])
        r0 = jnp.array([0.74])
        coords = jnp.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        bond_idx = jnp.array([[0, 1]])
        e = _mm3_bond_energy(k, r0, coords, bond_idx)
        assert float(e) == pytest.approx(0.0, abs=1e-10)

    def test_small_stretch(self) -> None:
        """Verify cubic correction for small displacement."""
        k = jnp.array([5.0])
        r0 = jnp.array([1.0])
        dr = 0.05  # Å
        coords = jnp.array([[0.0, 0.0, 0.0], [1.0 + dr, 0.0, 0.0]])
        bond_idx = jnp.array([[0, 1]])
        e = float(_mm3_bond_energy(k, r0, coords, bond_idx))

        # Expected: k*dr²*(1 - 2.55*dr + c4*dr²)
        c4 = (7.0 / 12.0) * 2.55**2
        expected = 5.0 * dr**2 * (1.0 - 2.55 * dr + c4 * dr**2)
        assert e == pytest.approx(expected, rel=1e-8)

    def test_differs_from_harmonic(self) -> None:
        """MM3 bond energy differs from harmonic for non-zero displacement."""
        k = jnp.array([5.0])
        r0 = jnp.array([1.0])
        coords = jnp.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
        bond_idx = jnp.array([[0, 1]])
        mm3_e = float(_mm3_bond_energy(k, r0, coords, bond_idx))
        harmonic_e = float(jnp.sum(k * (1.1 - 1.0) ** 2))
        assert mm3_e != pytest.approx(harmonic_e, rel=1e-4)

    def test_differentiable(self) -> None:
        """jax.grad works through MM3 bond energy."""
        k = jnp.array([5.0])
        r0 = jnp.array([1.0])
        bond_idx = jnp.array([[0, 1]])

        def energy_of_r(r: float) -> float:
            coords = jnp.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
            return _mm3_bond_energy(k, r0, coords, bond_idx)

        grad_fn = jax.grad(energy_of_r)
        # Gradient should be non-zero at displaced distance
        g = float(grad_fn(1.05))
        assert abs(g) > 0.01

        # Verify against finite differences
        h = 1e-6
        fd = (float(energy_of_r(1.05 + h)) - float(energy_of_r(1.05 - h))) / (2 * h)
        assert g == pytest.approx(fd, rel=1e-4)


class TestMM3AngleEnergy:
    """Test _mm3_angle_energy against known values."""

    def test_at_equilibrium_zero(self) -> None:
        """Energy is zero at equilibrium angle."""
        k = jnp.array([0.7])
        theta0 = jnp.array([np.deg2rad(109.5)])
        # Build a perfectly angled geometry
        theta = np.deg2rad(109.5)
        coords = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [np.cos(theta), np.sin(theta), 0.0]])
        angle_idx = jnp.array([[0, 1, 2]])
        e = _mm3_angle_energy(k, theta0, coords, angle_idx)
        assert float(e) == pytest.approx(0.0, abs=1e-10)

    def test_small_bend(self) -> None:
        """Verify sextic correction for small angular displacement."""
        k = jnp.array([0.7])
        theta0_deg = 109.5
        theta0_rad = np.deg2rad(theta0_deg)
        theta0 = jnp.array([theta0_rad])

        # Bent by 5 degrees
        theta_actual = np.deg2rad(114.5)
        coords = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [np.cos(theta_actual), np.sin(theta_actual), 0.0]])
        angle_idx = jnp.array([[0, 1, 2]])
        e = float(_mm3_angle_energy(k, theta0, coords, angle_idx))

        dtheta = theta_actual - theta0_rad
        dtheta_deg = dtheta * (180.0 / np.pi)
        anharmonic = (
            1.0 + (-0.014) * dtheta_deg + 5.6e-5 * dtheta_deg**2 + (-7.0e-7) * dtheta_deg**3 + 9.0e-10 * dtheta_deg**4
        )
        expected = 0.7 * dtheta**2 * anharmonic
        assert e == pytest.approx(expected, rel=1e-6)

    def test_differentiable(self) -> None:
        """jax.grad works through MM3 angle energy."""
        k = jnp.array([0.7])
        theta0 = jnp.array([np.deg2rad(109.5)])
        angle_idx = jnp.array([[0, 1, 2]])

        def energy_of_angle(theta_rad: jnp.ndarray) -> jnp.ndarray:
            coords = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [jnp.cos(theta_rad), jnp.sin(theta_rad), 0.0]])
            return _mm3_angle_energy(k, theta0, coords, angle_idx)

        grad_fn = jax.grad(energy_of_angle)
        g = float(grad_fn(jnp.deg2rad(115.0)))
        assert abs(g) > 0.0

    def test_near_collinear_gradient_matches_fd(self) -> None:
        """JAX-grad agrees with finite-difference even when θ→π (q2mm#284).

        Regression test for the gradient-correctness bug fixed by
        replacing ``arccos(clip(cos_θ, …))`` with the ``atan2``-based
        ``_angle_from_vectors``.  At near-collinear geometries the clip
        zeroed out ``∂θ/∂(cos θ)`` at the boundary, so the autodiff
        gradient went to zero while the FD gradient (and the physical
        force) stayed finite — letting the geometry minimizer drift
        through unphysical θ ≈ π configurations without resistance.

        This test fails on the pre-fix implementation (``∂E/∂atom_k =
        [0, 0, 0]``) and passes on the atan2-based implementation
        (``∂E/∂atom_k ≈ [−1.47e-4, −4.92e-1, 0]``).
        """
        # θ₀ = 109.5°; place atom k near antiparallel to atom i wrt central j
        # so cos θ ≈ -0.99999990 (the old clip boundary -1+1e-7).
        # sin θ ≈ sqrt(1 - 0.99999990²) ≈ 4.47e-4, i.e. perpendicular
        # component / bond length.
        k = jnp.array([0.3])
        theta0 = jnp.array([np.deg2rad(109.5)])
        angle_idx = jnp.array([[0, 1, 2]])
        coords = jnp.array(
            [[1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.5, 4.47e-4, 0.0]],
        )

        def energy_of_coords(c: jnp.ndarray) -> jnp.ndarray:
            return _mm3_angle_energy(k, theta0, c, angle_idx)

        g_jax = np.asarray(jax.grad(energy_of_coords)(coords))

        # FD on atom k (index 2) y-component — the perpendicular axis
        # is where the angle's restoring force should act.
        eps = 1e-7
        cp = coords.at[2, 1].add(eps)
        cm = coords.at[2, 1].add(-eps)
        g_fd_y = (float(energy_of_coords(cp)) - float(energy_of_coords(cm))) / (2 * eps)

        assert abs(g_jax[2, 1]) > 0.1, (
            f"JAX gradient on atom k y-component is suspiciously small at near-collinear "
            f"({g_jax[2, 1]:.3e}); the clip-arccos bug would zero this out."
        )
        assert g_jax[2, 1] == pytest.approx(g_fd_y, rel=1e-3), (
            f"JAX gradient {g_jax[2, 1]:.6e} disagrees with FD {g_fd_y:.6e} on atom k y-component "
            f"at near-collinear geometry (q2mm#284 regression)."
        )


class TestMM3VdwEnergy:
    """Test _mm3_vdw_energy against known values."""

    def test_zero_pairs_returns_zero(self) -> None:
        """No pairs → zero energy."""
        radius = jnp.array([1.4])
        epsilon = jnp.array([0.056])
        coords = jnp.array([[0.0, 0.0, 0.0]])
        pairs = jnp.empty((0, 2), dtype=jnp.int32)
        assert float(_mm3_vdw_energy(radius, epsilon, coords, pairs)) == 0.0

    def test_at_equilibrium_negative(self) -> None:
        """At r = rv, energy should be negative (attractive well)."""
        radius = jnp.array([1.4, 1.4])
        epsilon = jnp.array([0.056, 0.056])
        rv = 2.8  # radius1 + radius2
        coords = jnp.array([[0.0, 0.0, 0.0], [rv, 0.0, 0.0]])
        pairs = jnp.array([[0, 1]])
        e = float(_mm3_vdw_energy(radius, epsilon, coords, pairs))
        # At r=rv: ε*(184000*exp(-12) - 2.25*(1)^6) ≈ ε*(1.129 - 2.25) < 0
        assert e < 0

    def test_short_range_wall(self) -> None:
        """Below rc=0.34*rv, repulsive wall kicks in."""
        radius = jnp.array([1.4, 1.4])
        epsilon = jnp.array([0.056, 0.056])
        rv = 2.8
        rc = 0.34 * rv
        coords = jnp.array([[0.0, 0.0, 0.0], [rc * 0.5, 0.0, 0.0]])
        pairs = jnp.array([[0, 1]])
        e = float(_mm3_vdw_energy(radius, epsilon, coords, pairs))
        # Should be very repulsive
        assert e > 100.0

    def test_known_value_at_2rv(self) -> None:
        """Verify energy at r=2*rv matches analytical expression."""
        radius = jnp.array([1.5, 1.5])
        epsilon = jnp.array([0.1, 0.1])
        rv = 3.0
        r = 2.0 * rv  # r = 6.0 Å
        coords = jnp.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
        pairs = jnp.array([[0, 1]])
        e = float(_mm3_vdw_energy(radius, epsilon, coords, pairs))
        expected = 0.1 * (184000.0 * np.exp(-12.0 * r / rv) - 2.25 * (rv / r) ** 6)
        assert e == pytest.approx(expected, rel=1e-6)

    def test_differentiable(self) -> None:
        """jax.grad works through MM3 vdW energy."""
        radius = jnp.array([1.4, 1.4])
        epsilon = jnp.array([0.056, 0.056])
        pairs = jnp.array([[0, 1]])

        def energy_of_dist(d: float) -> float:
            coords = jnp.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
            return _mm3_vdw_energy(radius, epsilon, coords, pairs)

        grad_fn = jax.grad(energy_of_dist)
        g = float(grad_fn(3.0))
        h = 1e-6
        fd = (float(energy_of_dist(3.0 + h)) - float(energy_of_dist(3.0 - h))) / (2 * h)
        assert g == pytest.approx(fd, rel=1e-3)


# ---------------------------------------------------------------------------
# Integration tests: JaxBackend with functional_form="mm3"
# ---------------------------------------------------------------------------


class TestJaxBackendMM3:
    """Test JaxBackend with MM3 functional form end-to-end."""

    def test_supported_forms_includes_mm3(self) -> None:
        """JaxBackend now supports both harmonic and mm3."""
        backend = load_backend("jax")
        forms = backend.info.functional_forms
        assert "harmonic" in forms
        assert "mm3" in forms

    def test_mm3_energy_diatomic(self) -> None:
        """MM3 energy computation for H₂."""
        backend = load_backend("jax")
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff = _h2_ff_mm3()
        e = prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        assert isinstance(e, float)
        assert np.isfinite(e)
        assert e > 0  # displaced from equilibrium

    def test_mm3_energy_at_equilibrium(self) -> None:
        """MM3 energy is zero at equilibrium geometry."""
        backend = load_backend("jax")
        mol = make_diatomic(distance=0.74, bond_tolerance=2.0)
        ff = _h2_ff_mm3()
        e = prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        assert e == pytest.approx(0.0, abs=1e-8)

    def test_mm3_energy_water(self) -> None:
        """MM3 energy computation for water (bonds + angles)."""
        backend = load_backend("jax")
        mol = make_water(angle_deg=110.0, bond_length=1.0)
        ff = _water_ff_mm3()
        e = prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        assert isinstance(e, float)
        assert np.isfinite(e)

    def test_mm3_hessian(self) -> None:
        """MM3 Hessian computation produces valid matrix."""
        backend = load_backend("jax")
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff = _h2_ff_mm3()
        hess = prepare_case(backend, mol, ff).hessian(HessianRequest(parameters=param_vector(ff))).hessian
        assert hess.shape == (6, 6)
        # Hessian should be symmetric
        np.testing.assert_allclose(hess, hess.T, atol=1e-10)

    def test_mm3_differs_from_harmonic(self) -> None:
        """MM3 energy differs from harmonic for same parameters."""
        backend = load_backend("jax")
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff_mm3 = _h2_ff_mm3()
        ff_harm = ForceField(
            bonds=[BondParam(elements=("H", "H"), force_constant=5.0, equilibrium=0.74)],
            functional_form=FunctionalForm.HARMONIC,
        )
        e_mm3 = prepare_case(backend, mol, ff_mm3).energy(EnergyRequest(parameters=param_vector(ff_mm3))).energy
        e_harm = prepare_case(backend, mol, ff_harm).energy(EnergyRequest(parameters=param_vector(ff_harm))).energy
        # Both should be positive (displaced from equilibrium) but different
        assert e_mm3 > 0
        assert e_harm > 0
        assert e_mm3 != pytest.approx(e_harm, rel=1e-4)

    def test_mm3_analytical_gradient(self) -> None:
        """jax.grad through full MM3 energy matches finite differences."""
        backend = load_backend("jax")
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff = _h2_ff_mm3()
        _pg = prepare_case(backend, mol, ff).parameter_gradient(ParameterGradientRequest(parameters=param_vector(ff)))
        _e, grad_anal = _pg.energy, _pg.gradient

        # Finite difference gradient
        params = _params(ff).copy()
        grad_fd = np.zeros_like(params)
        h = 1e-5
        for i in range(len(params)):
            p_plus, p_minus = params.copy(), params.copy()
            p_plus[i] += h
            p_minus[i] -= h
            e_plus = (
                prepare_case(backend, mol, _materialize(ff, p_plus))
                .energy(EnergyRequest(parameters=param_vector(_materialize(ff, p_plus))))
                .energy
            )
            e_minus = (
                prepare_case(backend, mol, _materialize(ff, p_minus))
                .energy(EnergyRequest(parameters=param_vector(_materialize(ff, p_minus))))
                .energy
            )
            grad_fd[i] = (e_plus - e_minus) / (2 * h)

        np.testing.assert_allclose(grad_anal, grad_fd, atol=1e-4, rtol=1e-4)

    def test_mm3_vdw_energy(self) -> None:
        """MM3 vdW energy computation for He pair."""
        backend = load_backend("jax")
        mol = make_noble_gas_pair(distance=3.0, bond_tolerance=0.5)
        ff = _he2_ff_mm3()
        e = prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        assert isinstance(e, float)
        assert np.isfinite(e)


# ---------------------------------------------------------------------------
# Parity test: JAX MM3 vs OpenMM MM3
# ---------------------------------------------------------------------------


@pytest.mark.openmm
@pytest.mark.cross_backend
@pytest.mark.skipif(not _HAS_OPENMM, reason="OpenMM not installed")
class TestMM3ParityJaxVsOpenMM:
    """Verify JAX MM3 produces identical energies to OpenMM MM3."""

    def test_bond_parity(self) -> None:
        """MM3 bond energy: JAX vs OpenMM within 1e-6 kcal/mol."""
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff = _h2_ff_mm3()

        jax_e = prepare_case(load_backend("jax"), mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        omm_e = (
            prepare_case(load_backend("openmm", platform_name="CPU"), mol, ff)
            .energy(EnergyRequest(parameters=param_vector(ff)))
            .energy
        )
        assert jax_e == pytest.approx(omm_e, abs=1e-6)

    def test_water_parity(self) -> None:
        """MM3 bond+angle energy: JAX vs OpenMM within 1e-5 kcal/mol."""
        mol = make_water(angle_deg=110.0, bond_length=1.0)
        ff = _water_ff_mm3()

        jax_e = prepare_case(load_backend("jax"), mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        omm_e = (
            prepare_case(load_backend("openmm", platform_name="CPU"), mol, ff)
            .energy(EnergyRequest(parameters=param_vector(ff)))
            .energy
        )
        assert jax_e == pytest.approx(omm_e, abs=1e-5)

    def test_vdw_parity(self) -> None:
        """MM3 vdW energy: JAX vs OpenMM within 1e-6 kcal/mol."""
        mol = make_noble_gas_pair(distance=3.5, bond_tolerance=0.5)
        ff = _he2_ff_mm3()

        jax_e = prepare_case(load_backend("jax"), mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        omm_e = (
            prepare_case(load_backend("openmm", platform_name="CPU"), mol, ff)
            .energy(EnergyRequest(parameters=param_vector(ff)))
            .energy
        )
        assert jax_e == pytest.approx(omm_e, abs=1e-6)


# ---------------------------------------------------------------------------
# Near-linear torsion damping (issue #255 root cause)
# ---------------------------------------------------------------------------


def _make_four_atom_chain(
    central_angle_deg: float,
    dihedral_deg: float = 60.0,
) -> Molecule:
    """Build a 4-atom chain C-C-C-C with a controlled central angle.

    Atom 0 at origin, atom 1 along +x, atom 2 at the specified
    angle from the 0→1 bond, atom 3 rotated by dihedral_deg.
    """
    from q2mm.models.molecule import Molecule

    r = 1.5  # bond length
    theta = np.radians(central_angle_deg)
    phi = np.radians(dihedral_deg)

    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([r, 0.0, 0.0])
    # p2 at angle theta from p0-p1 bond, in the xy plane
    p2 = p1 + r * np.array([-np.cos(np.pi - theta), np.sin(np.pi - theta), 0.0])
    # p3 rotated by dihedral around the p1-p2 axis
    b1 = p2 - p1
    b1_hat = b1 / np.linalg.norm(b1)
    # perpendicular in the plane of p0,p1,p2
    v_in = p0 - p1 - np.dot(p0 - p1, b1_hat) * b1_hat
    v_in_hat = v_in / (np.linalg.norm(v_in) + 1e-30)
    v_out = np.cross(b1_hat, v_in_hat)
    p3 = p2 + r * (
        -np.cos(np.pi - theta) * b1_hat + np.sin(np.pi - theta) * (np.cos(phi) * v_in_hat + np.sin(phi) * v_out)
    )

    return Molecule(
        symbols=["C", "C", "C", "C"],
        geometry=np.array([p0, p1, p2, p3]),
    )


def _torsion_ff() -> ForceField:
    """Return a 4-atom FF with one torsion term."""
    return ForceField(
        bonds=[BondParam(elements=("C", "C"), force_constant=5.0, equilibrium=1.5)],
        angles=[AngleParam(elements=("C", "C", "C"), force_constant=0.5, equilibrium=109.5)],
        torsions=[
            TorsionParam(elements=("C", "C", "C", "C"), periodicity=3, force_constant=2.0),
        ],
        functional_form=FunctionalForm.MM3,
    )


class TestNearLinearTorsionDamping:
    """Verify the smoothstep damping for near-linear central angles."""

    def test_normal_angle_undamped(self) -> None:
        """Torsion energy at 109.5° central angle is NOT suppressed."""
        mol = _make_four_atom_chain(central_angle_deg=109.5)
        ff = _torsion_ff()
        backend = load_backend("jax")
        e = prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        # Energy should be non-trivial (torsion + angle + bond contributions)
        assert abs(e) > 0.01, f"Energy too small: {e}"

        # Compare against a manually computed torsion: at 109.5° the
        # smoothstep weight should be exactly 1.0 (sin(109.5°) >> sin(170°))
        from q2mm.backends.mm.jax_engine import _smoothstep, _SIN_LO, _SIN_HI

        sin_109 = np.sin(np.radians(109.5))
        w = float(_smoothstep(jnp.float64(sin_109), _SIN_LO, _SIN_HI))
        assert w == pytest.approx(1.0, abs=1e-10)

    def test_near_linear_suppressed(self) -> None:
        """Torsion energy at 179° central angle IS suppressed."""
        from q2mm.backends.mm.jax_engine import _smoothstep, _SIN_LO, _SIN_HI

        sin_179 = np.sin(np.radians(179.0))
        w = float(_smoothstep(jnp.float64(sin_179), _SIN_LO, _SIN_HI))
        assert w == pytest.approx(0.0, abs=1e-10)

        # Full energy check: 179° vs 120° — torsion contribution at 179°
        # should be negligible
        mol_linear = _make_four_atom_chain(central_angle_deg=179.0)
        mol_normal = _make_four_atom_chain(central_angle_deg=120.0)
        ff = _torsion_ff()
        backend = load_backend("jax")
        e_linear = prepare_case(backend, mol_linear, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        e_normal = prepare_case(backend, mol_normal, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        # Both have bond + angle energy, but 179° should have near-zero
        # torsion contribution.  The angle energy at 179° is large (far
        # from 109.5° equilibrium), so total energy is still substantial.
        # Key: the energy should be FINITE and not blown up.
        assert np.isfinite(e_linear), f"Energy at 179° is not finite: {e_linear}"
        assert np.isfinite(e_normal), f"Energy at 120° is not finite: {e_normal}"

    def test_gradient_finite_at_179(self) -> None:
        """Gradient (force) at 179° central angle is finite and bounded."""
        mol = _make_four_atom_chain(central_angle_deg=179.0)
        ff = _torsion_ff()
        backend = load_backend("jax")
        state = backend._build_state(mol, ff)
        params = jnp.array(_params(ff), dtype=jnp.float64)
        coords = jnp.array(mol.geometry, dtype=jnp.float64)

        grad = jax.grad(state._energy_fn, argnums=1)(params, coords)
        grad_np = np.array(grad)
        max_force = float(np.abs(grad_np).max())
        assert np.all(np.isfinite(grad_np)), "Gradient contains NaN/Inf"
        # Without damping, max force would be ~1000+.  With damping,
        # torsion forces are suppressed; remaining forces come from the
        # angle term (179° is 70° from the 109.5° equilibrium — the MM3
        # sextic angle produces ~350 kcal/(mol·Å) at that displacement).
        assert max_force < 500.0, (
            f"Max force {max_force:.1f} kcal/(mol·Å) too large at 179° — near-linear damping may not be working"
        )

    def test_hessian_finite_at_179(self) -> None:
        """Hessian at 179° central angle is finite (no 10⁹ frequencies)."""
        mol = _make_four_atom_chain(central_angle_deg=179.0)
        ff = _torsion_ff()
        backend = load_backend("jax")
        state = backend._build_state(mol, ff)
        params = jnp.array(_params(ff), dtype=jnp.float64)

        flat = jnp.array(mol.geometry.flatten(), dtype=jnp.float64)
        hess = jax.hessian(lambda fc: state._energy_fn(params, fc.reshape(-1, 3)))(flat)
        hess_np = np.array(hess)
        assert np.all(np.isfinite(hess_np)), "Hessian contains NaN/Inf"
        max_elem = float(np.abs(hess_np).max())
        # Without damping, Hessian elements could be 10⁶+.  With damping,
        # they should be bounded.  The MM3 sextic angle term at 179°
        # (70° from equilibrium) produces large but finite second derivatives.
        assert max_elem < 5e4, f"Hessian element {max_elem:.1f} too large"

    def test_smoothstep_transition(self) -> None:
        """Smoothstep transitions correctly between 170° and 175°."""
        from q2mm.backends.mm.jax_engine import _smoothstep, _SIN_LO, _SIN_HI

        # Below 170°: w = 1.0 (no damping)
        for angle in [90, 109.5, 120, 150, 160, 170]:
            s = np.sin(np.radians(angle))
            w = float(_smoothstep(jnp.float64(s), _SIN_LO, _SIN_HI))
            assert w == pytest.approx(1.0, abs=1e-6), f"{angle}°: w={w}"

        # Above 175°: w = 0.0 (full suppression)
        for angle in [175, 178, 179, 179.9]:
            s = np.sin(np.radians(angle))
            w = float(_smoothstep(jnp.float64(s), _SIN_LO, _SIN_HI))
            assert w == pytest.approx(0.0, abs=1e-6), f"{angle}°: w={w}"

        # Between 170-175°: 0 < w < 1 (smooth transition)
        s_172 = np.sin(np.radians(172.0))
        w_172 = float(_smoothstep(jnp.float64(s_172), _SIN_LO, _SIN_HI))
        assert 0.1 < w_172 < 0.9, f"172°: w={w_172} not in transition zone"

    def test_near_zero_angle_also_suppressed(self) -> None:
        """Torsion damping fires for near-0° too (sin is symmetric)."""
        from q2mm.backends.mm.jax_engine import _SIN_HI, _SIN_LO, _smoothstep

        for angle in [1, 3, 5]:
            s = np.sin(np.radians(angle))
            w = float(_smoothstep(jnp.float64(s), _SIN_LO, _SIN_HI))
            assert w == pytest.approx(0.0, abs=1e-6), f"{angle}°: w={w}"


class TestStretchBendEnergy:
    """Verify MM3 stretch-bend cross-term energy and differentiability."""

    def test_analytical_value(self) -> None:
        """Hand-computed SB energy for a water-like 3-atom system."""
        import math

        # Water: H-O-H, r0=0.96 Å, theta0=104.5°, k_sb=11.5 kcal/(mol·Å·rad)
        # At r_OH=1.00 Å, theta=110°:
        #   dr_sum = (1.00-0.96) + (1.00-0.96) = 0.08
        #   dtheta = (110-104.5) * pi/180 = 0.09599 rad
        #   E = 11.5 * 0.08 * 0.09599 = 0.08831 kcal/mol
        mol = make_water(angle_deg=110.0, bond_length=1.0)
        ff = ForceField(
            bonds=[BondParam(elements=("H", "O"), force_constant=8.0, equilibrium=0.96)],
            angles=[AngleParam(elements=("H", "O", "H"), force_constant=0.7, equilibrium=104.5)],
            stretch_bends=[
                StretchBendParam(elements=("H", "O", "H"), force_constant=11.5),
            ],
            functional_form=FunctionalForm.MM3,
        )
        backend = load_backend("jax")
        # Total energy includes bond + angle + SB; extract SB by subtraction
        ff_no_sb = ForceField(
            bonds=ff.bonds,
            angles=ff.angles,
            functional_form=FunctionalForm.MM3,
        )
        e_with_sb = prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        e_without_sb = (
            prepare_case(backend, mol, ff_no_sb).energy(EnergyRequest(parameters=param_vector(ff_no_sb))).energy
        )
        sb_energy = e_with_sb - e_without_sb

        dr_sum = 2 * (1.0 - 0.96)
        dtheta = math.radians(110.0 - 104.5)
        expected = 11.5 * dr_sum * dtheta
        assert sb_energy == pytest.approx(expected, abs=1e-4), f"SB energy {sb_energy:.6f} != expected {expected:.6f}"

    def test_at_equilibrium_zero(self) -> None:
        """SB energy is zero when geometry matches equilibrium."""
        mol = make_water(angle_deg=104.5, bond_length=0.96)
        ff = ForceField(
            bonds=[BondParam(elements=("H", "O"), force_constant=8.0, equilibrium=0.96)],
            angles=[AngleParam(elements=("H", "O", "H"), force_constant=0.7, equilibrium=104.5)],
            stretch_bends=[
                StretchBendParam(elements=("H", "O", "H"), force_constant=11.5),
            ],
            functional_form=FunctionalForm.MM3,
        )
        ff_no_sb = ForceField(bonds=ff.bonds, angles=ff.angles, functional_form=FunctionalForm.MM3)
        backend = load_backend("jax")
        sb_energy = (
            prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
            - prepare_case(backend, mol, ff_no_sb).energy(EnergyRequest(parameters=param_vector(ff_no_sb))).energy
        )
        assert abs(sb_energy) < 1e-10

    def test_differentiable(self) -> None:
        """jax.grad flows through SB energy without NaN."""
        mol = make_water(angle_deg=110.0, bond_length=1.0)
        ff = ForceField(
            bonds=[BondParam(elements=("H", "O"), force_constant=8.0, equilibrium=0.96)],
            angles=[AngleParam(elements=("H", "O", "H"), force_constant=0.7, equilibrium=104.5)],
            stretch_bends=[
                StretchBendParam(elements=("H", "O", "H"), force_constant=11.5),
            ],
            functional_form=FunctionalForm.MM3,
        )
        backend = load_backend("jax")
        state = backend._build_state(mol, ff)
        params = jnp.array(_params(ff), dtype=jnp.float64)
        coords = jnp.array(mol.geometry, dtype=jnp.float64)
        grad = jax.grad(state._energy_fn, argnums=1)(params, coords)
        assert np.all(np.isfinite(np.array(grad))), "SB gradient contains NaN/Inf"

    def test_unit_conversion_matches_allinger(self) -> None:
        """SB unit conversion factor matches Allinger's 2.51118 × 180/π."""
        import math

        from q2mm.models.units import MDYNRAD_TO_KCALMOLARAD

        expected = 2.51118 * (180.0 / math.pi)
        assert pytest.approx(expected, rel=1e-6) == MDYNRAD_TO_KCALMOLARAD


# ---------------------------------------------------------------------------
# MM3 Bond-Dipole Electrostatics
# ---------------------------------------------------------------------------


class TestMM3DipoleElectrostatics:
    """Tests for _mm3_dipole_energy and the electrostatics constant."""

    def test_constant_value(self) -> None:
        """MM3 dipole constant = 14.3928 / 1.5 ≈ 9.5952."""
        assert pytest.approx(14.3928 / 1.5, rel=1e-6) == _MM3_DIPOLE_CONST

    def test_zero_dipole_gives_zero_energy(self) -> None:
        """Two bonds with zero dipole moment produce zero energy."""
        # Two bonds: atoms 0-1 and 2-3, both with μ=0
        coords = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        bond_idx = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        pair_idx = jnp.array([[0, 1]], dtype=jnp.int32)
        dipoles = jnp.array([0.0, 0.0])
        E = _mm3_dipole_energy(dipoles, coords, bond_idx, pair_idx)
        assert float(E) == pytest.approx(0.0, abs=1e-15)

    def test_empty_pairs_gives_zero(self) -> None:
        """No dipole pairs → zero energy."""
        coords = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        bond_idx = jnp.array([[0, 1]], dtype=jnp.int32)
        pair_idx = jnp.empty((0, 2), dtype=jnp.int32)
        dipoles = jnp.array([1.5])
        E = _mm3_dipole_energy(dipoles, coords, bond_idx, pair_idx)
        assert float(E) == pytest.approx(0.0, abs=1e-15)

    def test_collinear_dipoles_sign(self) -> None:
        """Two collinear, head-to-tail dipoles on x-axis.

        For collinear head-to-tail: cos χ = 1, cos α_i = 1, cos α_j = 1.
        angular = 1 - 3·1·1 = -2 → energy is negative (attractive).
        """
        coords = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        bond_idx = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        pair_idx = jnp.array([[0, 1]], dtype=jnp.int32)
        dipoles = jnp.array([1.0, 1.0])
        E = float(_mm3_dipole_energy(dipoles, coords, bond_idx, pair_idx))
        assert E < 0, "Collinear head-to-tail dipoles should attract"

        # Verify exact value: const * 1 * 1 * (-2) / r³
        # r = midpoint distance = |3.5 - 0.5| = 3.0
        expected = _MM3_DIPOLE_CONST * 1.0 * 1.0 * (-2.0) / (3.0**3)
        assert pytest.approx(expected, rel=1e-10) == E

    def test_antiparallel_dipoles_repel(self) -> None:
        """Two antiparallel (opposing) dipoles on x-axis.

        Bond 0: atom0→atom1 (→), Bond 1: atom3→atom2 (←).
        Dipole vectors: d_0 = (1,0,0), d_1 = (-1,0,0).
        cos χ = -1, cos α_i = 1, cos α_j = -1.
        angular = -1 - 3·1·(-1) = -1+3 = 2 → positive (repulsive).
        """
        coords = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        bond_idx = jnp.array([[0, 1], [3, 2]], dtype=jnp.int32)
        pair_idx = jnp.array([[0, 1]], dtype=jnp.int32)
        dipoles = jnp.array([1.0, 1.0])
        E = float(_mm3_dipole_energy(dipoles, coords, bond_idx, pair_idx))
        assert E > 0, "Antiparallel dipoles should repel"

    def test_differentiable(self) -> None:
        """jax.grad flows through dipole energy without NaN."""
        coords = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [4.0, 1.0, 0.0]])
        bond_idx = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        pair_idx = jnp.array([[0, 1]], dtype=jnp.int32)
        dipoles = jnp.array([1.2, 0.8])

        def energy_of_coords(c: jnp.ndarray) -> jnp.ndarray:
            return _mm3_dipole_energy(dipoles, c, bond_idx, pair_idx)

        grad = jax.grad(energy_of_coords)(coords)
        assert np.all(np.isfinite(np.array(grad))), "Dipole gradient contains NaN/Inf"

    def test_engine_with_dipoles(self) -> None:
        """JaxBackend MM3 includes dipole energy for non-bonded bond pairs."""
        from test._shared import make_ethane

        mol = make_ethane()  # C₂H₆ has C-H bonds on different carbons
        ff = ForceField(
            bonds=[
                BondParam(elements=("C", "H"), force_constant=5.0, equilibrium=1.09, dipole_moment=0.7),
                BondParam(elements=("C", "C"), force_constant=4.4, equilibrium=1.54, dipole_moment=0.0),
            ],
            angles=[AngleParam(elements=("H", "C", "H"), force_constant=0.5, equilibrium=109.5)],
            functional_form=FunctionalForm.MM3,
        )
        backend = load_backend("jax")
        E_with = prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy

        ff_no_dipole = ForceField(
            bonds=[
                BondParam(elements=("C", "H"), force_constant=5.0, equilibrium=1.09, dipole_moment=0.0),
                BondParam(elements=("C", "C"), force_constant=4.4, equilibrium=1.54, dipole_moment=0.0),
            ],
            angles=[AngleParam(elements=("H", "C", "H"), force_constant=0.5, equilibrium=109.5)],
            functional_form=FunctionalForm.MM3,
        )
        E_without = (
            prepare_case(backend, mol, ff_no_dipole).energy(EnergyRequest(parameters=param_vector(ff_no_dipole))).energy
        )

        assert E_with != pytest.approx(E_without, abs=1e-6), "Dipole moment should contribute non-zero energy"
