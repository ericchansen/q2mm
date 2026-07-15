"""JAX-based differentiable MM backend for analytical gradients.

Provides a pure-JAX molecular mechanics engine supporting both harmonic
(OPLSAA-style) and MM3 functional forms.  All energy functions are
differentiable via ``jax.grad``, enabling analytical gradient computation
for force field parameter optimization.

**Harmonic forms** (default): harmonic bond/angle, 12-6 Lennard-Jones vdW,
cosine torsion.

**MM3 forms**: cubic bond stretch, sextic angle bend, Buckingham exp-6 vdW,
cosine torsion (shared with harmonic).  Based on Allinger et al., JACS 1989,
111, 8551.

ForceField stores parameters in canonical units (kcal/mol/Å² for bond_k,
kcal/mol/rad² for angle_k) with energy convention ``E = k·(x − x₀)²``.
The JAX energy functions use the same convention, so no unit conversion is
needed at the engine boundary.
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

from q2mm.backends.base import MMEngine, coerce_molecule
from q2mm.backends.registry import register_mm
from q2mm.models.units import KCALMOLA2_TO_HESSIAN_AU
from q2mm.constants import (
    MM3_BOND_C3,
    MM3_BOND_C4,
    MM3_ANGLE_C3,
    MM3_ANGLE_C4,
    MM3_ANGLE_C5,
    MM3_ANGLE_C6,
    MM3_VDW_A,
    MM3_VDW_B,
    MM3_VDW_C,
)
from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm
from q2mm.models.molecule import Molecule

from q2mm.backends.mm._jax_common import (
    _HAS_JAX,
    ensure_jax as _ensure_jax_common,
    layout_block_offsets,
    match_angle as _match_angle,
    match_bond as _match_bond,
    match_vdw as _match_vdw,
    params_and_coords as _params_and_coords_impl,
)

# Lazy placeholders — populated by _ensure_jax().
jax = None
jnp = None


def _ensure_jax() -> None:
    """Call the shared ensure_jax and rebind module-level jax/jnp."""
    global jax, jnp  # noqa: PLW0603
    _ensure_jax_common()
    import q2mm.backends.mm._jax_common as _jc

    jax = _jc.jax
    jnp = _jc.jnp


# ---------------------------------------------------------------------------
# Hessian unit conversion: kcal/mol/Å² → Hartree/Bohr²
# Imported from q2mm.models.units as KCALMOLA2_TO_HESSIAN_AU
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Force field param vector unit conversions
# ---------------------------------------------------------------------------
# ForceField stores bond_k in kcal/(mol·Å²) and angle_k in kcal/(mol·rad²)
# (canonical units).  The JAX energy functions use the same units.
_BOND_K_CONV = 1.0
_ANGLE_K_CONV = 1.0


# ---------------------------------------------------------------------------
# Numerically stable geometry primitives
# ---------------------------------------------------------------------------


def _safe_norm(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """Euclidean norm with gradient-safe floor to avoid NaN at zero.

    Args:
        x: Input array.
        axis: Axis along which to compute the norm.

    Returns:
        jnp.ndarray: Element-wise norm, floored at ``1e-30`` before sqrt.

    """
    return jnp.sqrt(jnp.maximum(jnp.sum(x**2, axis=axis), 1e-30))


def _angle_from_vectors(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Angle between two vectors via ``atan2(|a×b|, a·b)`` with safe VJP.

    Returns the angle (radians) in ``[0, π]`` with **correct gradients
    everywhere on that interval**, including at the endpoints (collinear
    and antiparallel).

    Why not ``arccos(a·b / |a||b|)`` clipped to ``(-1+ε, 1-ε)``?
    The clip kills ``∂θ/∂(cos θ)`` at the boundary, so when an angle is
    driven all the way to 0 or π during geometry minimization, the
    gradient pulling it back toward θ₀ vanishes and the optimizer sees
    a spurious stationary point.  The forward energy contribution is
    real (``k·dθ²`` with dθ huge) but the autodiff gradient is wrong.
    This was the root cause of the metal-TS gradient correctness bug
    documented in q2mm#284 — see
    :class:`test.test_mm3_jax.TestMM3AngleEnergy.test_near_collinear_gradient_matches_fd`
    for the reproduction.

    Why not stock ``arctan2(|a×b|, a·b)``?
    Forward value is correct, but the autodiff path through
    ``sqrt(|cross|² + ε)`` produces ill-conditioned gradients when
    ``|cross| → 0``: ``d(sqrt(x+ε))/dx = 1/(2 sqrt(x+ε))`` blows up
    as ``x → 0``, and the chain through ``d(cross)/d(atom)`` becomes
    numerically unstable.  This implementation uses a custom VJP with
    the analytic gradient ``∂θ/∂a = (cos θ · â/|a| − b̂/|a|) / sin θ``
    (likewise for b), with ``sin θ`` floored at ``1e-12`` to keep the
    direction stable when collinear.  At exact collinearity the
    gradient magnitude limits to a finite value pointing perpendicular
    to the bond axis — matching the FD answer to ~1e-7.

    Args:
        a: First vector(s), last axis is the 3D component, shape ``(..., 3)``.
        b: Second vector(s), last axis is the 3D component, shape ``(..., 3)``.

    Returns:
        Angle values in radians, in ``[0, π]``.

    """
    return _angle_from_vectors_impl()(a, b)


@functools.lru_cache(maxsize=1)
def _angle_from_vectors_impl() -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Build the jax.custom_vjp-decorated angle function lazily.

    The decorator can't be applied at module import because ``jax`` is
    populated by ``_ensure_jax()`` — applying ``@jax.custom_vjp`` at
    decoration time would crash before the first ``JaxEngine()`` call.
    Cached so the custom-VJP function is built exactly once per process.
    """
    _ensure_jax()

    @jax.custom_vjp
    def _angle(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        na = jnp.sqrt(jnp.sum(a * a, axis=-1) + 1e-30)
        nb = jnp.sqrt(jnp.sum(b * b, axis=-1) + 1e-30)
        cross = jnp.cross(a, b, axis=-1)
        sin_t = jnp.sqrt(jnp.sum(cross * cross, axis=-1) + 1e-30) / (na * nb)
        cos_t = jnp.sum(a * b, axis=-1) / (na * nb)
        return jnp.arctan2(sin_t, cos_t)

    def _fwd(a, b):  # noqa: ANN001, ANN202
        na = jnp.sqrt(jnp.sum(a * a, axis=-1) + 1e-30)
        nb = jnp.sqrt(jnp.sum(b * b, axis=-1) + 1e-30)
        cross = jnp.cross(a, b, axis=-1)
        sin_t = jnp.sqrt(jnp.sum(cross * cross, axis=-1) + 1e-30) / (na * nb)
        cos_t = jnp.sum(a * b, axis=-1) / (na * nb)
        theta = jnp.arctan2(sin_t, cos_t)
        return theta, (a, b, na, nb, sin_t, cos_t)

    def _bwd(res, g):  # noqa: ANN001, ANN202
        a, b, na, nb, sin_t, cos_t = res
        # Analytic gradients of θ = arccos(a·b / |a||b|):
        #   ∂θ/∂a = (cos θ · a/|a|² − b/(|a||b|)) / sin θ
        #   ∂θ/∂b = (cos θ · b/|b|² − a/(|a||b|)) / sin θ
        # Floor sin θ at 1e-12 to keep direction finite when collinear;
        # at exact collinearity the analytic gradient is degenerate but
        # the floor gives the physically correct perpendicular-restoring
        # direction.
        safe_sin = jnp.maximum(sin_t, 1e-12)
        inv_nanb_sin = 1.0 / (na * nb * safe_sin)
        da = (cos_t / (na * na * safe_sin))[..., None] * a - inv_nanb_sin[..., None] * b
        db = (cos_t / (nb * nb * safe_sin))[..., None] * b - inv_nanb_sin[..., None] * a
        return (g[..., None] * da, g[..., None] * db)

    _angle.defvjp(_fwd, _bwd)
    return _angle


# ---------------------------------------------------------------------------
# OPLSAA-style energy functions (pure JAX, differentiable)
# ---------------------------------------------------------------------------


def _harmonic_bond_energy(
    k: jnp.ndarray, r0: jnp.ndarray, coords: jnp.ndarray, bond_indices: jnp.ndarray
) -> jnp.ndarray:
    """Harmonic bond stretch: ``E = Σ k_i · (r_i − r0_i)²``.

    Args:
        k (jnp.ndarray): Force constants, shape ``(n_bonds,)``, in
            kcal/mol/Å².
        r0 (jnp.ndarray): Equilibrium distances, shape ``(n_bonds,)``,
            in Å.
        coords (jnp.ndarray): Cartesian coordinates, shape
            ``(n_atoms, 3)``, in Å.
        bond_indices (jnp.ndarray): Atom index pairs, shape
            ``(n_bonds, 2)``.

    Returns:
        jnp.ndarray: Scalar total bond energy in kcal/mol.

    """
    dr = coords[bond_indices[:, 0]] - coords[bond_indices[:, 1]]
    r = _safe_norm(dr, axis=-1)
    return jnp.sum(k * (r - r0) ** 2)


def _harmonic_angle_energy(
    k: jnp.ndarray, theta0: jnp.ndarray, coords: jnp.ndarray, angle_indices: jnp.ndarray
) -> jnp.ndarray:
    """Harmonic angle bend: ``E = Σ k_i · (θ_i − θ0_i)²``.

    Args:
        k (jnp.ndarray): Force constants, shape ``(n_angles,)``, in
            kcal/mol/rad².
        theta0 (jnp.ndarray): Equilibrium angles, shape ``(n_angles,)``,
            in radians.
        coords (jnp.ndarray): Cartesian coordinates, shape
            ``(n_atoms, 3)``, in Å.
        angle_indices (jnp.ndarray): Atom index triples ``(i, j, k)``,
            shape ``(n_angles, 3)``, where ``j`` is the central atom.

    Returns:
        jnp.ndarray: Scalar total angle energy in kcal/mol.

    """
    rij = coords[angle_indices[:, 0]] - coords[angle_indices[:, 1]]
    rkj = coords[angle_indices[:, 2]] - coords[angle_indices[:, 1]]
    theta = _angle_from_vectors(rij, rkj)
    return jnp.sum(k * (theta - theta0) ** 2)


def _ub_energy(
    k: jnp.ndarray,
    r0: jnp.ndarray,
    coords: jnp.ndarray,
    ub_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Urey-Bradley 1-3 interaction: ``E = Σ k_i · (r_13_i − r0_i)²``.

    Args:
        k (jnp.ndarray): Force constants, shape ``(n_ub,)``, in kcal/mol/Å².
        r0 (jnp.ndarray): Equilibrium distances, shape ``(n_ub,)``, in Å.
        coords (jnp.ndarray): Cartesian coordinates, shape ``(n_atoms, 3)``,
            in Å.
        ub_indices (jnp.ndarray): Atom index pairs ``(i, k)`` from the outer
            atoms of each UB-enabled angle, shape ``(n_ub, 2)``.

    Returns:
        jnp.ndarray: Scalar total UB energy in kcal/mol.

    """
    r_ij = coords[ub_indices[:, 0]] - coords[ub_indices[:, 1]]
    r = _safe_norm(r_ij, axis=-1)
    dr = r - r0
    return jnp.sum(k * dr**2)


def _lj_12_6_energy(
    per_atom_sigma: jnp.ndarray, per_atom_epsilon: jnp.ndarray, coords: jnp.ndarray, pair_indices: jnp.ndarray
) -> jnp.ndarray:
    """Lennard-Jones 12-6 energy with geometric combining rules.

    ``E = Σ_{i<j} 4·ε_ij · [(σ_ij/r)¹² − (σ_ij/r)⁶]`` where
    ``σ_ij = √(σ_i·σ_j)`` and ``ε_ij = √(ε_i·ε_j)``.

    Args:
        per_atom_sigma (jnp.ndarray): Per-atom sigma, shape
            ``(n_atoms,)``, in Å.
        per_atom_epsilon (jnp.ndarray): Per-atom epsilon, shape
            ``(n_atoms,)``, in kcal/mol.
        coords (jnp.ndarray): Cartesian coordinates, shape
            ``(n_atoms, 3)``, in Å.
        pair_indices (jnp.ndarray): Non-excluded atom pairs ``(i, j)``
            with ``i < j``, shape ``(n_pairs, 2)``.

    Returns:
        jnp.ndarray: Scalar total LJ energy in kcal/mol.

    """
    if pair_indices.shape[0] == 0:
        return jnp.float64(0.0)

    sig_i = per_atom_sigma[pair_indices[:, 0]]
    sig_j = per_atom_sigma[pair_indices[:, 1]]
    eps_i = per_atom_epsilon[pair_indices[:, 0]]
    eps_j = per_atom_epsilon[pair_indices[:, 1]]

    sig_ij = jnp.sqrt(sig_i * sig_j)
    eps_ij = jnp.sqrt(eps_i * eps_j)

    dr = coords[pair_indices[:, 0]] - coords[pair_indices[:, 1]]
    r = _safe_norm(dr, axis=-1)

    sr6 = (sig_ij / r) ** 6
    return jnp.sum(4.0 * eps_ij * (sr6**2 - sr6))


# ---------------------------------------------------------------------------
# MM3-style energy functions (pure JAX, differentiable)
# ---------------------------------------------------------------------------
# Allinger et al., JACS 1989, 111, 8551.

_RAD_TO_DEG = 180.0 / 3.141592653589793


def _mm3_bond_energy(k: jnp.ndarray, r0: jnp.ndarray, coords: jnp.ndarray, bond_indices: jnp.ndarray) -> jnp.ndarray:
    """MM3 cubic bond stretch: ``E = Σ k·dr²·(1 − 2.55·dr + c4·dr²)``.

    Args:
        k: Force constants, shape ``(n_bonds,)``, kcal/mol/Å².
        r0: Equilibrium distances, shape ``(n_bonds,)``, Å.
        coords: Cartesian coordinates, shape ``(n_atoms, 3)``, Å.
        bond_indices: Atom index pairs, shape ``(n_bonds, 2)``.

    Returns:
        Scalar total bond energy in kcal/mol.

    """
    dr_vec = coords[bond_indices[:, 0]] - coords[bond_indices[:, 1]]
    r = _safe_norm(dr_vec, axis=-1)
    dr = r - r0
    return jnp.sum(k * dr**2 * (1.0 - MM3_BOND_C3 * dr + MM3_BOND_C4 * dr**2))


def _mm3_angle_energy(
    k: jnp.ndarray, theta0: jnp.ndarray, coords: jnp.ndarray, angle_indices: jnp.ndarray
) -> jnp.ndarray:
    """MM3 sextic angle bend with degree-based anharmonic corrections.

    ``E = Σ k·dθ²·(1 + a3·dθ_deg + a4·dθ_deg² + a5·dθ_deg³ + a6·dθ_deg⁴)``

    where ``dθ_deg = (θ − θ₀) · (180/π)`` is the angular deviation in
    degrees, matching the MM3 convention.

    Args:
        k: Force constants, shape ``(n_angles,)``, kcal/mol/rad².
        theta0: Equilibrium angles, shape ``(n_angles,)``, radians.
        coords: Cartesian coordinates, shape ``(n_atoms, 3)``, Å.
        angle_indices: Atom index triples ``(i, j, k)``, shape
            ``(n_angles, 3)``, where ``j`` is the central atom.

    Returns:
        Scalar total angle energy in kcal/mol.

    """
    rij = coords[angle_indices[:, 0]] - coords[angle_indices[:, 1]]
    rkj = coords[angle_indices[:, 2]] - coords[angle_indices[:, 1]]
    theta = _angle_from_vectors(rij, rkj)
    dtheta = theta - theta0
    dtheta_deg = dtheta * _RAD_TO_DEG
    anharmonic = (
        1.0
        + MM3_ANGLE_C3 * dtheta_deg
        + MM3_ANGLE_C4 * dtheta_deg**2
        + MM3_ANGLE_C5 * dtheta_deg**3
        + MM3_ANGLE_C6 * dtheta_deg**4
    )
    return jnp.sum(k * dtheta**2 * anharmonic)


def _mm3_vdw_energy(
    per_atom_radius: jnp.ndarray,
    per_atom_epsilon: jnp.ndarray,
    coords: jnp.ndarray,
    pair_indices: jnp.ndarray,
) -> jnp.ndarray:
    """MM3 Buckingham exp-6 vdW with short-range repulsive wall.

    ``E = ε·(184000·exp(−12·r/rv) − 2.25·(rv/r)⁶)``

    where ``rv = r_i + r_j`` and ``ε = √(ε_i·ε_j)``.  Below ``r < 0.34·rv``
    the attractive r⁻⁶ dominates the exponential, so we switch to a hard
    r⁻¹² repulsion: ``ε·184000·exp(−12·rc/rv)·(rc/r)¹²``.

    Args:
        per_atom_radius: Per-atom vdW radius, shape ``(n_atoms,)``, Å.
        per_atom_epsilon: Per-atom well depth, shape ``(n_atoms,)``,
            kcal/mol.
        coords: Cartesian coordinates, shape ``(n_atoms, 3)``, Å.
        pair_indices: Non-excluded atom pairs ``(i, j)`` with
            ``i < j``, shape ``(n_pairs, 2)``.

    Returns:
        Scalar total vdW energy in kcal/mol.

    """
    if pair_indices.shape[0] == 0:
        return jnp.float64(0.0)

    rad_i = per_atom_radius[pair_indices[:, 0]]
    rad_j = per_atom_radius[pair_indices[:, 1]]
    eps_i = per_atom_epsilon[pair_indices[:, 0]]
    eps_j = per_atom_epsilon[pair_indices[:, 1]]

    rv = rad_i + rad_j
    eps_ij = jnp.sqrt(eps_i * eps_j)

    dr_vec = coords[pair_indices[:, 0]] - coords[pair_indices[:, 1]]
    r = _safe_norm(dr_vec, axis=-1)

    rc = 0.34 * rv

    # Normal Buckingham exp-6 region (r >= rc)
    ratio = rv / r
    e_buckingham = eps_ij * (MM3_VDW_A * jnp.exp(-MM3_VDW_B * r / rv) - MM3_VDW_C * ratio**6)

    # Short-range repulsive wall (r < rc)
    e_wall = eps_ij * MM3_VDW_A * jnp.exp(-MM3_VDW_B * rc / rv) * (rc / r) ** 12

    return jnp.sum(jnp.where(r >= rc, e_buckingham, e_wall))


# MM3 bond-dipole electrostatic constant.
# Allinger et al., JACS 1989, 111, 8551 give: E = 14.3928·μ₁·μ₂/r³·f (kcal/mol)
# with μ in Debye and r in Å.  Derivation:
#   332.05 kcal·Å/(mol·e²) × (1 D / 4.8032 e·Å)² = 14.394 kcal·Å·mol⁻¹·D⁻²
# Divided by the MM3 default dielectric D = 1.5.
_MM3_DIPOLE_CONST = 14.3928 / 1.5  # ≈ 9.595 kcal·Å³/(mol·D²)


def _mm3_dipole_energy(
    dipole_moments: jnp.ndarray,
    coords: jnp.ndarray,
    bond_indices: jnp.ndarray,
    dipole_pair_indices: jnp.ndarray,
) -> jnp.ndarray:
    """MM3 bond-dipole electrostatic energy.

    Each bond with a non-zero dipole moment contributes a point dipole
    at the bond midpoint, directed along the bond vector.  Pairs of
    dipoles interact via the MM3 point-dipole formula
    (Allinger et al., JACS 1989, 111, 8551):

    ``E = (14.394 / D) · μ_i · μ_j · (cos χ − 3 cos α_i cos α_j) / r³``

    where ``D = 1.5`` (MM3 default dielectric), ``μ`` is in Debye,
    ``r`` is the distance between bond midpoints in Å, ``χ`` is the
    angle between dipole vectors, and ``α_i``, ``α_j`` are angles
    between each dipole vector and the midpoint-midpoint vector.

    Args:
        dipole_moments: Per-bond dipole moment (Debye), shape ``(n_bonds,)``.
            Indexed by the bond's position in ``bond_indices``.
        coords: Cartesian coordinates, shape ``(n_atoms, 3)``, Å.
        bond_indices: Atom index pairs for each bond, shape ``(n_bonds, 2)``.
        dipole_pair_indices: Non-excluded bond pairs ``(bi, bj)`` with
            ``bi < bj``, shape ``(n_dipole_pairs, 2)``.

    Returns:
        Scalar total dipole energy in kcal/mol.

    """
    if dipole_pair_indices.shape[0] == 0:
        return jnp.float64(0.0)

    # Bond vectors and midpoints
    a_coords = coords[bond_indices[:, 0]]  # (n_bonds, 3)
    b_coords = coords[bond_indices[:, 1]]  # (n_bonds, 3)
    bond_vecs = b_coords - a_coords  # dipole direction: atom_0 → atom_1
    midpoints = 0.5 * (a_coords + b_coords)  # (n_bonds, 3)

    # Gather pairs
    bi = dipole_pair_indices[:, 0]
    bj = dipole_pair_indices[:, 1]

    mu_i = dipole_moments[bi]
    mu_j = dipole_moments[bj]

    d_i = bond_vecs[bi]  # dipole vector i, shape (n_pairs, 3)
    d_j = bond_vecs[bj]  # dipole vector j
    d_i_norm = d_i / (_safe_norm(d_i, axis=-1)[:, None] + 1e-30)
    d_j_norm = d_j / (_safe_norm(d_j, axis=-1)[:, None] + 1e-30)

    # Midpoint-midpoint vector
    r_vec = midpoints[bj] - midpoints[bi]  # (n_pairs, 3)
    r = _safe_norm(r_vec, axis=-1)  # (n_pairs,)
    r_hat = r_vec / (r[:, None] + 1e-30)

    # Angles: cos χ = d_i · d_j, cos α_i = d_i · r_hat, cos α_j = d_j · r_hat
    cos_chi = jnp.sum(d_i_norm * d_j_norm, axis=-1)
    cos_alpha_i = jnp.sum(d_i_norm * r_hat, axis=-1)
    cos_alpha_j = jnp.sum(d_j_norm * r_hat, axis=-1)

    # E = const * μ_i * μ_j * (cos χ - 3 cos α_i cos α_j) / r³
    angular = cos_chi - 3.0 * cos_alpha_i * cos_alpha_j
    e_pair = _MM3_DIPOLE_CONST * mu_i * mu_j * angular / (r**3 + 1e-30)

    return jnp.sum(e_pair)


def _mm3_stretch_bend_energy(
    k_sb: jnp.ndarray,
    r0_ij: jnp.ndarray,
    r0_jk: jnp.ndarray,
    theta0: jnp.ndarray,
    coords: jnp.ndarray,
    angle_indices: jnp.ndarray,
) -> jnp.ndarray:
    """MM3 stretch-bend cross-term energy.

    ``E = Σ k_sb · [(r_ij − r0_ij) + (r_jk − r0_jk)] · (θ − θ0)``

    Couples bond stretching with angle bending for each angle triple.

    Args:
        k_sb: Stretch-bend force constants, shape ``(n_sb,)``,
            kcal/(mol·Å·rad).
        r0_ij: Equilibrium i–j bond lengths, shape ``(n_sb,)``, Å.
        r0_jk: Equilibrium j–k bond lengths, shape ``(n_sb,)``, Å.
        theta0: Equilibrium angles, shape ``(n_sb,)``, radians.
        coords: Cartesian coordinates, shape ``(n_atoms, 3)``, Å.
        angle_indices: Atom index triples ``(i, j, k)``, shape
            ``(n_sb, 3)``.

    Returns:
        Scalar total stretch-bend energy in kcal/mol.

    """
    rij_vec = coords[angle_indices[:, 0]] - coords[angle_indices[:, 1]]
    rjk_vec = coords[angle_indices[:, 2]] - coords[angle_indices[:, 1]]
    r_ij = _safe_norm(rij_vec, axis=-1)
    r_jk = _safe_norm(rjk_vec, axis=-1)

    # Use atan2-based angle for correct gradients near collinear geometries
    # (see _angle_from_vectors docstring).  cos_theta is still needed for
    # the near-linear suppression switch below.
    cross = jnp.cross(rij_vec, rjk_vec, axis=-1)
    sin_theta = jnp.sqrt(jnp.sum(cross * cross, axis=-1) + 1e-30) / (r_ij * r_jk)
    cos_theta = jnp.sum(rij_vec * rjk_vec, axis=-1) / (r_ij * r_jk)
    theta = jnp.arctan2(sin_theta, cos_theta)

    # Near-linear suppression: dθ/dr diverges as 1/sin(θ) at 0° and 180°,
    # same singularity that motivated torsion damping.
    w = _smoothstep(sin_theta, _SIN_LO, _SIN_HI)

    dr_sum = (r_ij - r0_ij) + (r_jk - r0_jk)
    dtheta = theta - theta0
    return jnp.sum(w * k_sb * dr_sum * dtheta)


def _smoothstep(x: jnp.ndarray, edge0: float, edge1: float) -> jnp.ndarray:
    """Hermite smoothstep: 0 below *edge0*, 1 above *edge1*, C¹ between."""
    t = jnp.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# Near-linear suppression thresholds.
# Torsions/stretch-bends whose central bond angle is near-collinear
# (close to 0° or 180°) are physically meaningless — the dihedral is
# ill-defined and the angle gradient diverges as 1/sin(θ).  A Hermite
# smoothstep on min(sin θ_b, sin θ_c) suppresses these terms smoothly.
# sin(θ) is symmetric, so this fires for both near-0° and near-180°.
#
# sin(175°) = sin(5°) ≈ 0.087  → full suppression (w = 0)
# sin(170°) = sin(10°) ≈ 0.174  → no suppression  (w = 1)
_SIN_LO = math.sin(math.radians(175.0))  # ≈ 0.0872
_SIN_HI = math.sin(math.radians(170.0))  # ≈ 0.1736


def _torsion_energy(
    k: jnp.ndarray,
    periodicity: jnp.ndarray,
    phase: jnp.ndarray,
    coords: jnp.ndarray,
    torsion_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Cosine torsion energy: ``E = Σ w_i · k_i · (1 + cos(n_i·φ_i − γ_i))``.

    Uses an atan2-based signed dihedral angle computation.  Torsion terms
    whose central bond angle is near-collinear (within ~10° of 0° or 180°)
    are smoothly suppressed to avoid the well-known gradient singularity.

    Args:
        k (jnp.ndarray): Force constants, shape ``(n_torsions,)``, kcal/mol.
        periodicity (jnp.ndarray): Periodicity per torsion, shape
            ``(n_torsions,)``.
        phase (jnp.ndarray): Phase angles per torsion, shape
            ``(n_torsions,)``, in radians.
        coords (jnp.ndarray): Cartesian coordinates, shape
            ``(n_atoms, 3)``, in Å.
        torsion_indices (jnp.ndarray): Atom index quadruples
            ``(i, j, k, l)``, shape ``(n_torsions, 4)``.

    Returns:
        jnp.ndarray: Scalar total torsion energy in kcal/mol.

    """
    # Vectors along the torsion chain (IUPAC convention):
    # b0 = i→j, b1 = j→k, b2 = k→l
    p0 = coords[torsion_indices[:, 0]]
    p1 = coords[torsion_indices[:, 1]]
    p2 = coords[torsion_indices[:, 2]]
    p3 = coords[torsion_indices[:, 3]]
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    # Near-linear suppression: compute sin(θ) at each central atom via
    # |a × b| / (|a| |b|).  When either central angle approaches 180°,
    # sin(θ) → 0 and the torsion term is smoothly turned off.
    sin_b = _safe_norm(jnp.cross(-b0, b1)) / (_safe_norm(b0) * _safe_norm(b1) + 1e-30)
    sin_c = _safe_norm(jnp.cross(-b1, b2)) / (_safe_norm(b1) * _safe_norm(b2) + 1e-30)
    w = _smoothstep(jnp.minimum(sin_b, sin_c), _SIN_LO, _SIN_HI)

    # Normal vectors to the two planes
    n1 = jnp.cross(b0, b1)
    n2 = jnp.cross(b1, b2)
    # atan2-based signed dihedral (epsilon on x prevents NaN gradients
    # at linear geometries where both cross products vanish)
    b1_len = _safe_norm(b1, axis=-1)
    m1 = jnp.cross(n1, b1 / jnp.maximum(b1_len, 1e-10)[:, None])
    x = jnp.sum(n1 * n2, axis=-1) + 1e-20
    y = jnp.sum(m1 * n2, axis=-1)
    phi = jnp.arctan2(y, x)
    return jnp.sum(w * k * (1.0 + jnp.cos(periodicity * phi - phase)))


# ---------------------------------------------------------------------------
# Topology / handle data structures
# ---------------------------------------------------------------------------


def _build_vdw_pairs(n_atoms: int, bond_pairs: list[tuple[int, int]]) -> np.ndarray:
    """Build non-excluded vdW pair list (1-2 and 1-3 exclusions).

    Args:
        n_atoms: Total number of atoms.
        bond_pairs: List of ``(i, j)`` bonded atom index pairs.

    Returns:
        np.ndarray: Shape ``(n_pairs, 2)`` array of non-excluded pairs
            with ``i < j``, dtype ``int32``.

    """
    excluded: set[tuple[int, int]] = set()

    # 1-2 exclusions
    adj: dict[int, set[int]] = {i: set() for i in range(n_atoms)}
    for i, j in bond_pairs:
        pair = (min(i, j), max(i, j))
        excluded.add(pair)
        adj[i].add(j)
        adj[j].add(i)

    # 1-3 exclusions
    for atom in range(n_atoms):
        neighbors = list(adj[atom])
        for ni in range(len(neighbors)):
            for nj in range(ni + 1, len(neighbors)):
                pair = (min(neighbors[ni], neighbors[nj]), max(neighbors[ni], neighbors[nj]))
                excluded.add(pair)

    pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms) if (i, j) not in excluded]
    return np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)


@dataclass
class JaxHandle:
    """Cached topology and parameter mapping for a molecule.

    Created once per molecule, reused across parameter updates.

    Attributes:
        molecule: Deep copy of the input molecule.
        bond_indices: Atom index pairs, shape ``(n_matched_bonds, 2)``.
        angle_indices: Atom index triples, shape ``(n_matched_angles, 3)``.
        torsion_indices: Atom index quadruples, shape
            ``(n_matched_torsions, 4)``.
        vdw_pair_indices: Non-excluded pairs, shape ``(n_vdw_pairs, 2)``.
        ub_indices: Atom index pairs ``(i, k)`` for Urey-Bradley 1-3 terms,
            shape ``(n_ub_terms, 2)``.
        bond_param_map: Maps each matched bond → index into
            ``ForceField.bonds``.
        angle_param_map: Maps each matched angle → index into
            ``ForceField.angles``.
        torsion_param_map: Maps each matched torsion → index into
            ``ForceField.torsions``.
        atom_vdw_map: Maps each atom → index into ``ForceField.vdws``.
        ub_param_map: Maps each UB term → index into ``ForceField._ub_angles``.
        n_bond_types: Number of unique bond parameter types.
        n_angle_types: Number of unique angle parameter types.
        n_torsion_types: Number of unique torsion parameter types.
        n_vdw_types: Number of unique vdW parameter types.
        n_ub_types: Number of unique UB parameter types.

    """

    molecule: Molecule
    # Topology arrays (static, used inside JIT closures)
    bond_indices: np.ndarray  # (n_matched_bonds, 2) atom indices
    angle_indices: np.ndarray  # (n_matched_angles, 3) atom indices
    torsion_indices: np.ndarray  # (n_matched_torsions, 4) atom indices
    vdw_pair_indices: np.ndarray  # (n_vdw_pairs, 2) non-excluded pairs
    # Mappings: detected term index → ForceField param type index
    bond_param_map: np.ndarray  # (n_matched_bonds,) → index into ff.bonds
    angle_param_map: np.ndarray  # (n_matched_angles,) → index into ff.angles
    torsion_param_map: np.ndarray  # (n_matched_torsions,) → index into ff.torsions
    atom_vdw_map: np.ndarray  # (n_atoms,) → index into ff.vdws
    # Param vector layout
    n_bond_types: int
    n_angle_types: int
    n_torsion_types: int
    n_vdw_types: int
    # Urey-Bradley
    ub_indices: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int32))
    ub_param_map: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    n_ub_types: int = 0
    # Stretch-bend cross terms
    sb_angle_indices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int32))
    sb_param_map: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    sb_bond_ij_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    sb_bond_jk_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    sb_angle_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    n_sb_types: int = 0
    # Bond-dipole electrostatics
    dipole_bond_indices: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int32))
    dipole_moments: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    dipole_pair_indices: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int32))
    # Functional form used to compile the energy function
    functional_form: str = "harmonic"
    # Compiled energy function (captures topology, JIT-compiled)
    _energy_fn: Callable | None = field(default=None, repr=False)
    _grad_fn: Callable | None = field(default=None, repr=False)
    _coord_hess_fn: Callable | None = field(default=None, repr=False)
    _batched_coord_hess_fn: Callable | None = field(default=None, repr=False)
    _hess_param_jac_fn: Callable | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# JaxEngine
# ---------------------------------------------------------------------------


@register_mm("jax")
class JaxEngine(MMEngine):
    """Differentiable MM backend using JAX.

    Provides analytical gradients of the energy with respect to force field
    parameters via ``jax.grad``, eliminating the need for finite differences
    in parameter optimization.

    Supports both harmonic/LJ and MM3 (cubic bond, sextic angle, Buckingham
    exp-6 vdW) functional forms. Select the form via the required
    :attr:`ForceField.functional_form` attribute — every force field
    must state which form it uses explicitly.

    Example:
        >>> engine = JaxEngine()
        >>> energy = engine.energy(molecule, forcefield)
        >>> energy, grad = engine.energy_and_param_grad(molecule, forcefield)

    """

    def __init__(self) -> None:
        """Initialize the JAX engine.

        Raises:
            ImportError: If JAX is not installed.

        """
        _ensure_jax()

    @property
    def name(self) -> str:
        """Human-readable engine name including device type.

        Returns:
            str: e.g. ``"JAX (gpu)"`` or ``"JAX (cpu)"``.

        """
        backend = jax.default_backend()
        return f"JAX ({backend})"

    def supported_functional_forms(self) -> frozenset[str]:
        """JAX supports harmonic and MM3 functional forms.

        Returns:
            frozenset[str]: ``{"harmonic", "mm3"}``.

        """
        return frozenset({"harmonic", "mm3"})

    def is_available(self) -> bool:
        """Check if JAX is installed.

        Returns:
            bool: ``True`` if the ``jax`` package is importable.

        """
        return _HAS_JAX

    @classmethod
    def deps_available(cls) -> bool:
        """Check if JAX is importable without triggering CUDA init."""
        return _HAS_JAX

    def supports_runtime_params(self) -> bool:
        """Whether parameters can be updated without rebuilding the system.

        Returns:
            bool: Always ``True`` for JAX.

        """
        return True

    def supports_analytical_gradients(self) -> bool:
        """Whether this engine provides analytical parameter gradients.

        Returns:
            bool: Always ``True`` for JAX.

        """
        return True

    def create_context(self, structure: Molecule | JaxHandle, forcefield: ForceField | None = None) -> JaxHandle:
        """Build topology and compile energy function for a molecule.

        Args:
            structure (Molecule | JaxHandle): A :class:`Molecule` or :class:`JaxHandle`.
            forcefield: Force field to apply. Auto-generated from the
                molecule if ``None``.

        Returns:
            JaxHandle: Compiled handle for energy evaluation and gradient
                computation.

        Raises:
            ValueError: If vdW parameters are defined but not all atoms
                have matching entries.

        """
        if forcefield is not None:
            self._validate_forcefield(forcefield)
        molecule = coerce_molecule(structure, engine_name="JaxEngine")
        if forcefield is None:
            forcefield = ForceField.create_for_molecule(molecule, functional_form=FunctionalForm.HARMONIC)

        # Match bonds
        bond_atom_indices = []
        bond_param_map = []
        for bond in molecule.bonds:
            idx, param = _match_bond(
                forcefield,
                bond.elements,
                env_id=bond.env_id,
                ff_row=bond.ff_row,
                bond_order=getattr(bond, "bond_order", ""),
                bond_length=bond.length,
            )
            if param is not None:
                bond_atom_indices.append((bond.atom_i, bond.atom_j))
                bond_param_map.append(idx)

        # Match angles
        angle_atom_indices = []
        angle_param_map = []
        for angle in molecule.angles:
            idx, param = _match_angle(forcefield, angle.elements, env_id=angle.env_id, ff_row=angle.ff_row)
            if param is not None:
                angle_atom_indices.append((angle.atom_i, angle.atom_j, angle.atom_k))
                angle_param_map.append(idx)

        # Match Urey-Bradley terms (1-3 distance for CHARMM angles)
        ub_angles = forcefield._ub_angles
        ub_angle_to_ub_idx = {id(a): i for i, a in enumerate(ub_angles)}
        ub_atom_indices: list[tuple[int, int]] = []
        ub_param_map: list[int] = []
        for angle in molecule.angles:
            matched = forcefield.match_angle(angle.elements, env_id=angle.env_id, ff_row=angle.ff_row)
            if matched is not None and matched.ub_force_constant is not None and matched.ub_equilibrium is not None:
                ub_idx = ub_angle_to_ub_idx.get(id(matched))
                if ub_idx is not None:
                    ub_atom_indices.append((angle.atom_i, angle.atom_k))
                    ub_param_map.append(ub_idx)

        # Match stretch-bend cross terms
        # Use match_stretch_bend() with proper env_id/ff_row fallback.
        # Look up adjacent bonds from the already-matched bond set (not
        # re-matching with the wrong env_id).
        sb_angle_indices: list[tuple[int, int, int]] = []
        sb_param_map_list: list[int] = []
        sb_bond_ij_idx: list[int] = []
        sb_bond_jk_idx: list[int] = []
        sb_angle_idx: list[int] = []
        sb_param_index = {id(p): i for i, p in enumerate(forcefield.stretch_bends)}
        bond_ff_index = {id(p): i for i, p in enumerate(forcefield.bonds)}
        angle_ff_index = {id(p): i for i, p in enumerate(forcefield.angles)}

        # Pre-build lookup from atom pair → matched bond param (using the
        # bond's own env_id, not the angle's).
        matched_bond_by_atoms: dict[tuple[int, int], BondParam] = {}
        for bond in molecule.bonds:
            _, param = _match_bond(
                forcefield,
                bond.elements,
                env_id=bond.env_id,
                ff_row=bond.ff_row,
                bond_order=getattr(bond, "bond_order", ""),
                bond_length=bond.length,
            )
            if param is not None:
                matched_bond_by_atoms[(bond.atom_i, bond.atom_j)] = param
                matched_bond_by_atoms[(bond.atom_j, bond.atom_i)] = param

        for angle in molecule.angles:
            sb = forcefield.match_stretch_bend(angle.elements, env_id=angle.env_id, ff_row=angle.ff_row)
            if sb is None:
                continue
            bond_ij = matched_bond_by_atoms.get((angle.atom_i, angle.atom_j))
            bond_jk = matched_bond_by_atoms.get((angle.atom_j, angle.atom_k))
            angle_param = forcefield.match_angle(angle.elements, env_id=angle.env_id, ff_row=angle.ff_row)
            if bond_ij is not None and bond_jk is not None and angle_param is not None:
                sb_angle_indices.append((angle.atom_i, angle.atom_j, angle.atom_k))
                sb_param_map_list.append(sb_param_index[id(sb)])
                sb_bond_ij_idx.append(bond_ff_index[id(bond_ij)])
                sb_bond_jk_idx.append(bond_ff_index[id(bond_jk)])
                sb_angle_idx.append(angle_ff_index[id(angle_param)])

        # Match torsions — each detected torsion may match multiple FF
        # entries (one per periodicity component), each becoming a separate
        # term.  Both proper and improper torsions are routed through the
        # same cosine-series energy term (mirrors OpenMM, which puts both
        # into a single PeriodicTorsionForce; see openmm.py:1080-1132).
        torsion_atom_indices: list[tuple[int, int, int, int]] = []
        torsion_param_map: list[int] = []
        torsion_param_index = {id(p): i for i, p in enumerate(forcefield.torsions)}
        for torsion in molecule.torsions:
            matches = forcefield.match_torsion(
                torsion.element_quad, env_id=torsion.env_id, ff_row=torsion.ff_row, is_improper=False
            )
            for param in matches:
                j_ff = torsion_param_index[id(param)]
                torsion_atom_indices.append((torsion.atom_i, torsion.atom_j, torsion.atom_k, torsion.atom_l))
                torsion_param_map.append(j_ff)
        for imp in molecule.improper_torsions:
            matches = forcefield.match_torsion(imp.element_quad, env_id=imp.env_id, ff_row=imp.ff_row, is_improper=True)
            for param in matches:
                j_ff = torsion_param_index[id(param)]
                torsion_atom_indices.append((imp.atom_i, imp.atom_j, imp.atom_k, imp.atom_l))
                torsion_param_map.append(j_ff)

        # Match vdW
        atom_vdw_map = []
        for atom_index, (symbol, atom_type) in enumerate(zip(molecule.symbols, molecule.atom_types, strict=False)):
            idx, param = _match_vdw(forcefield, atom_type=atom_type, element=symbol)
            if param is not None:
                atom_vdw_map.append(idx)
            else:
                atom_vdw_map.append(-1)

        # Validate vdW mapping: if the force field defines vdW parameters,
        # disallow any unmatched atoms.  Using -1 as an index would silently
        # select the last vdW entry via JAX negative indexing, corrupting
        # the energy and gradients.  When the force field defines no vdW
        # terms at all, vdW energy is effectively disabled and unmatched
        # atoms are safe.
        unmatched = [i for i, idx in enumerate(atom_vdw_map) if idx == -1]
        if getattr(forcefield, "vdws", None) and unmatched:
            raise ValueError(
                f"Unmatched vdW parameters for atoms at indices {unmatched}. "
                "Ensure all atom types/elements have corresponding vdW "
                "parameters in the force field, or remove vdW terms from "
                "the force field if vdW interactions are not intended."
            )

        # Build vdW pair list with 1-2 and 1-3 exclusions
        vdw_pairs = _build_vdw_pairs(
            len(molecule.symbols),
            [(b.atom_i, b.atom_j) for b in molecule.bonds],
        )

        # Build bond-dipole electrostatics topology
        # Collect bonds with non-zero dipole moments and their atom indices
        dipole_bond_list: list[tuple[int, int]] = []
        dipole_moment_list: list[float] = []
        for bond_idx, bond in enumerate(molecule.bonds):
            idx, param = _match_bond(
                forcefield,
                bond.elements,
                env_id=bond.env_id,
                ff_row=bond.ff_row,
                bond_order=getattr(bond, "bond_order", ""),
                bond_length=bond.length,
            )
            if param is not None and abs(param.dipole_moment) > 1e-10:
                dipole_bond_list.append((bond.atom_i, bond.atom_j))
                dipole_moment_list.append(param.dipole_moment)

        # Build dipole pair list: exclude bond pairs that share an atom
        dipole_pairs_list: list[tuple[int, int]] = []
        if len(dipole_bond_list) > 1:
            bond_atom_sets = [set(pair) for pair in dipole_bond_list]
            for i in range(len(dipole_bond_list)):
                for j in range(i + 1, len(dipole_bond_list)):
                    # Exclude if the two bonds share any atom (1-2/1-3 at bond level)
                    if not bond_atom_sets[i] & bond_atom_sets[j]:
                        dipole_pairs_list.append((i, j))

        dipole_bond_arr = (
            np.array(dipole_bond_list, dtype=np.int32) if dipole_bond_list else np.empty((0, 2), dtype=np.int32)
        )
        dipole_moment_arr = np.array(dipole_moment_list, dtype=np.float64) if dipole_moment_list else np.empty(0)
        dipole_pair_arr = (
            np.array(dipole_pairs_list, dtype=np.int32) if dipole_pairs_list else np.empty((0, 2), dtype=np.int32)
        )

        bond_indices_arr = (
            np.array(bond_atom_indices, dtype=np.int32) if bond_atom_indices else np.empty((0, 2), dtype=np.int32)
        )
        angle_indices_arr = (
            np.array(angle_atom_indices, dtype=np.int32) if angle_atom_indices else np.empty((0, 3), dtype=np.int32)
        )
        torsion_indices_arr = (
            np.array(torsion_atom_indices, dtype=np.int32) if torsion_atom_indices else np.empty((0, 4), dtype=np.int32)
        )
        ub_indices_arr = (
            np.array(ub_atom_indices, dtype=np.int32) if ub_atom_indices else np.empty((0, 2), dtype=np.int32)
        )

        handle = JaxHandle(
            molecule=molecule,
            bond_indices=bond_indices_arr,
            angle_indices=angle_indices_arr,
            torsion_indices=torsion_indices_arr,
            vdw_pair_indices=vdw_pairs,
            bond_param_map=np.array(bond_param_map, dtype=np.int32),
            angle_param_map=np.array(angle_param_map, dtype=np.int32),
            torsion_param_map=np.array(torsion_param_map, dtype=np.int32),
            atom_vdw_map=np.array(atom_vdw_map, dtype=np.int32),
            n_bond_types=len(forcefield.bonds),
            n_angle_types=len(forcefield.angles),
            n_torsion_types=len(forcefield.torsions),
            n_vdw_types=len(forcefield.vdws),
            ub_indices=ub_indices_arr,
            ub_param_map=np.array(ub_param_map, dtype=np.int32),
            n_ub_types=len(ub_angles),
            sb_angle_indices=(
                np.array(sb_angle_indices, dtype=np.int32) if sb_angle_indices else np.empty((0, 3), dtype=np.int32)
            ),
            sb_param_map=np.array(sb_param_map_list, dtype=np.int32)
            if sb_param_map_list
            else np.empty(0, dtype=np.int32),
            sb_bond_ij_idx=np.array(sb_bond_ij_idx, dtype=np.int32),
            sb_bond_jk_idx=np.array(sb_bond_jk_idx, dtype=np.int32),
            sb_angle_idx=np.array(sb_angle_idx, dtype=np.int32),
            n_sb_types=len(forcefield.stretch_bends),
            dipole_bond_indices=dipole_bond_arr,
            dipole_moments=dipole_moment_arr,
            dipole_pair_indices=dipole_pair_arr,
            functional_form=_resolve_form(forcefield),
        )

        # Compile energy function
        handle._energy_fn = _compile_energy_fn(handle, forcefield)
        return handle

    def _get_handle(self, structure: Molecule | JaxHandle, forcefield: ForceField) -> JaxHandle:
        """Get or create a :class:`JaxHandle`.

        Args:
            structure: A :class:`Molecule` or existing :class:`JaxHandle`.
            forcefield: Force field for creating a new handle.

        Returns:
            JaxHandle: Ready-to-use handle.

        """
        if isinstance(structure, JaxHandle):
            return structure
        molecule = coerce_molecule(structure, engine_name="JaxEngine")
        return self.create_context(molecule, forcefield)

    def _params_and_coords(self, handle: JaxHandle, forcefield: ForceField) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Extract JAX arrays from force field and molecule."""
        return _params_and_coords_impl(handle.molecule.geometry, forcefield)

    def energy(self, structure: Molecule | JaxHandle, forcefield: ForceField) -> float:
        """Calculate energy in kcal/mol.

        Args:
            structure (Molecule | JaxHandle): A :class:`Molecule` or :class:`JaxHandle`.
            forcefield (ForceField): Force field parameters.

        Returns:
            float: Potential energy in kcal/mol.

        """
        handle = self._get_handle(structure, forcefield)
        params, coords = self._params_and_coords(handle, forcefield)
        return float(handle._energy_fn(params, coords))

    def energy_and_param_grad(
        self, structure: Molecule | JaxHandle, forcefield: ForceField
    ) -> tuple[float, np.ndarray]:
        """Compute energy and analytical gradient w.r.t. FF parameters.

        Args:
            structure (Molecule | JaxHandle): A :class:`Molecule` or :class:`JaxHandle`.
            forcefield (ForceField): Force field parameters.

        Returns:
            tuple[float, np.ndarray]: ``(energy, grad)`` where ``energy``
                is in kcal/mol and ``grad`` has the same shape as
                ``ParameterLayout.from_force_field(forcefield).vector(forcefield)``.

        """
        handle = self._get_handle(structure, forcefield)
        params, coords = self._params_and_coords(handle, forcefield)

        if handle._grad_fn is None:
            handle._grad_fn = jax.jit(jax.value_and_grad(handle._energy_fn, argnums=0))

        val, grad = handle._grad_fn(params, coords)
        return float(val), np.asarray(grad)

    def supports_batched_energy(self) -> bool:  # noqa: D102
        """Return True; JaxEngine supports batched energy evaluation via jax.vmap."""
        return True

    def batched_energy(
        self,
        structure: Molecule | JaxHandle,
        forcefield: ForceField,
        param_matrix: np.ndarray,
    ) -> np.ndarray:
        """Evaluate energy for a batch of parameter vectors via ``jax.vmap``.

        All ``param_matrix`` rows are evaluated in a single vectorised
        call, enabling GPU-parallel sensitivity analysis.

        Args:
            structure: Molecule or cached :class:`JaxHandle`.
            forcefield: Base force field (topology only).
            param_matrix: Shape ``(batch, n_params)`` parameter vectors.

        Returns:
            np.ndarray: Shape ``(batch,)`` energies in kcal/mol.

        """
        handle = self._get_handle(structure, forcefield)
        coords = jnp.array(handle.molecule.geometry, dtype=jnp.float64)
        batch_params = jnp.array(param_matrix, dtype=jnp.float64)

        if not hasattr(handle, "_batched_energy_fn") or handle._batched_energy_fn is None:
            handle._batched_energy_fn = jax.jit(jax.vmap(handle._energy_fn, in_axes=(0, None)))

        return np.asarray(handle._batched_energy_fn(batch_params, coords))

    def hessian(self, structure: Molecule | JaxHandle, forcefield: ForceField) -> np.ndarray:
        """Compute Hessian via ``jax.hessian`` (d²E/dcoords²) in Hartree/Bohr².

        Args:
            structure (Molecule | JaxHandle): A :class:`Molecule` or :class:`JaxHandle`.
            forcefield (ForceField): Force field parameters.

        Returns:
            np.ndarray: Shape ``(3N, 3N)`` Hessian in Hartree/Bohr².

        """
        handle = self._get_handle(structure, forcefield)
        params, coords = self._params_and_coords(handle, forcefield)

        if handle._coord_hess_fn is None:

            def _energy_of_flat_coords(flat_coords: jnp.ndarray, params_: jnp.ndarray) -> jnp.ndarray:
                return handle._energy_fn(params_, flat_coords.reshape(-1, 3))

            handle._coord_hess_fn = jax.jit(jax.hessian(_energy_of_flat_coords, argnums=0))

        flat_coords = coords.flatten()
        hess_kcal_a2 = handle._coord_hess_fn(flat_coords, params)
        return np.asarray(hess_kcal_a2) * KCALMOLA2_TO_HESSIAN_AU

    def supports_analytical_hessian_gradients(self) -> bool:  # noqa: D102
        """Return True; JaxEngine supports Hessian parameter Jacobians via jax.jacrev."""
        return True

    def hessian_and_param_jacobian(
        self, structure: Molecule | JaxHandle, forcefield: ForceField
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Hessian and its Jacobian w.r.t. FF parameters.

        Uses ``jax.jacrev(jax.hessian(E))`` to compute both the
        coordinate Hessian and its derivatives with respect to force
        field parameters in a single JIT-compiled call.

        Args:
            structure: A :class:`Molecule` or :class:`JaxHandle`.
            forcefield: Force field parameters.

        Returns:
            ``(hessian, dH_dp)`` where ``hessian`` has shape ``(3N, 3N)``
            and ``dH_dp`` has shape ``(3N, 3N, n_params)``, both in
            Hartree/Bohr².

        """
        handle = self._get_handle(structure, forcefield)
        params, coords = self._params_and_coords(handle, forcefield)
        flat_coords = coords.flatten()

        if handle._hess_param_jac_fn is None:

            def _energy_of_flat_coords(flat_coords_: jnp.ndarray, params_: jnp.ndarray) -> jnp.ndarray:
                return handle._energy_fn(params_, flat_coords_.reshape(-1, 3))

            hess_fn = jax.hessian(_energy_of_flat_coords, argnums=0)

            def _hess_and_jac(flat_coords_: jnp.ndarray, params_: jnp.ndarray) -> tuple:
                h = hess_fn(flat_coords_, params_)
                dh_dp = jax.jacrev(hess_fn, argnums=1)(flat_coords_, params_)
                return h, dh_dp

            handle._hess_param_jac_fn = jax.jit(_hess_and_jac)

        hess_kcal_a2, dhess_dp_kcal_a2 = handle._hess_param_jac_fn(flat_coords, params)
        return (
            np.asarray(hess_kcal_a2) * KCALMOLA2_TO_HESSIAN_AU,
            np.asarray(dhess_dp_kcal_a2) * KCALMOLA2_TO_HESSIAN_AU,
        )

    def minimize(
        self, structure: Molecule | JaxHandle, forcefield: ForceField, max_iterations: int = 200
    ) -> tuple[float, list[str], np.ndarray]:
        """Minimize energy w.r.t. coordinates using analytical JAX gradients.

        Uses ``scipy.optimize.minimize`` with the L-BFGS-B method.

        Args:
            structure (Molecule | JaxHandle): A :class:`Molecule` or :class:`JaxHandle`.
            forcefield (ForceField): Force field parameters.
            max_iterations (int): Maximum number of L-BFGS-B iterations.

        Returns:
            tuple[float, list[str], np.ndarray]: ``(energy, atoms, coords)``
                where energy is in kcal/mol and coords are in Å.

        """
        from scipy.optimize import minimize as scipy_minimize

        handle = self._get_handle(structure, forcefield)
        params, coords = self._params_and_coords(handle, forcefield)

        energy_fn = handle._energy_fn
        coord_grad_fn = jax.jit(jax.grad(lambda c, p: energy_fn(p, c.reshape(-1, 3)), argnums=0))

        x0 = np.asarray(coords.flatten())

        def objective(x: np.ndarray) -> float:
            """Evaluate energy at flat coordinate vector *x*."""
            return float(energy_fn(params, jnp.array(x).reshape(-1, 3)))

        def gradient(x: np.ndarray) -> np.ndarray:
            """Return energy gradient w.r.t. flat coordinate vector *x*."""
            return np.asarray(coord_grad_fn(jnp.array(x), params))

        result = scipy_minimize(
            objective,
            x0,
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": max_iterations},
        )

        opt_coords = result.x.reshape(-1, 3)
        opt_energy = float(result.fun)
        return opt_energy, list(handle.molecule.symbols), opt_coords


# ---------------------------------------------------------------------------
# Energy function compiler
# ---------------------------------------------------------------------------


def _resolve_form(forcefield: ForceField) -> str:
    """Resolve the functional form string from a ForceField."""
    return forcefield.functional_form.value


def _compile_energy_fn(handle: JaxHandle, forcefield: ForceField) -> Callable:
    """Create a JIT-compiled energy function for a specific topology.

    The returned function has signature ``(params, coords) -> energy``
    where ``params`` is the flat parameter vector from
    ``ParameterLayout.from_force_field(forcefield).vector(forcefield)`` and ``coords`` is shape
    ``(n_atoms, 3)`` in Å. Energy is returned in kcal/mol.

    Selects harmonic or MM3 energy functions based on
    ``handle.functional_form``.

    Args:
        handle: A :class:`JaxHandle` with topology arrays populated.
        forcefield: Force field used to extract static torsion
            periodicity and phase values.

    Returns:
        Callable: JIT-compiled ``energy_fn(params, coords)`` closure.

    """
    use_mm3 = handle.functional_form == "mm3"

    # Capture topology as JAX arrays in the closure
    has_bonds = handle.n_bond_types > 0 and len(handle.bond_indices) > 0
    has_angles = handle.n_angle_types > 0 and len(handle.angle_indices) > 0
    has_torsions = handle.n_torsion_types > 0 and len(handle.torsion_indices) > 0
    has_vdw = handle.n_vdw_types > 0 and len(handle.vdw_pair_indices) > 0

    _bond_indices = jnp.array(handle.bond_indices) if has_bonds else None
    _bond_map = jnp.array(handle.bond_param_map) if has_bonds else None
    _angle_indices = jnp.array(handle.angle_indices) if has_angles else None
    _angle_map = jnp.array(handle.angle_param_map) if has_angles else None
    _torsion_indices = jnp.array(handle.torsion_indices) if has_torsions else None
    _torsion_map = jnp.array(handle.torsion_param_map) if has_torsions else None
    _vdw_pairs = jnp.array(handle.vdw_pair_indices) if has_vdw else None
    _atom_vdw_map = jnp.array(handle.atom_vdw_map) if has_vdw else None

    # Capture torsion periodicity/phase as static arrays (not optimized)
    if has_torsions:
        per_term_n = []
        per_term_gamma = []
        for ff_idx in handle.torsion_param_map:
            tp = forcefield.torsions[ff_idx]
            per_term_n.append(float(tp.periodicity))
            per_term_gamma.append(math.radians(tp.phase))
        _torsion_n = jnp.array(per_term_n)
        _torsion_gamma = jnp.array(per_term_gamma)
    else:
        _torsion_n = None
        _torsion_gamma = None

    n_bt = handle.n_bond_types
    n_at = handle.n_angle_types
    n_tt = handle.n_torsion_types
    n_vt = handle.n_vdw_types
    n_ubt = handle.n_ub_types
    n_sbt = handle.n_sb_types

    # Param vector offsets
    from q2mm.models.parameters import ParameterLayout

    _offsets = layout_block_offsets(ParameterLayout.from_force_field(forcefield))
    bond_offset = _offsets["bond"]
    angle_offset = _offsets["angle"]
    torsion_offset = _offsets["torsion"]
    vdw_offset = _offsets["vdw"]
    # UB params are appended at the end of the param vector
    ub_offset = _offsets["ub"]

    # Urey-Bradley topology
    has_ub = n_ubt > 0 and len(handle.ub_indices) > 0
    _ub_indices = jnp.array(handle.ub_indices) if has_ub else None
    _ub_map = jnp.array(handle.ub_param_map) if has_ub else None

    # Stretch-bend topology
    has_sb = n_sbt > 0 and len(handle.sb_angle_indices) > 0
    _sb_angle_indices = jnp.array(handle.sb_angle_indices) if has_sb else None
    _sb_map = jnp.array(handle.sb_param_map) if has_sb else None
    # Indices into the bond/angle param blocks — used at runtime to gather
    # equilibrium values from the param vector (not frozen constants).
    _sb_bond_ij_idx = jnp.array(handle.sb_bond_ij_idx) if has_sb else None
    _sb_bond_jk_idx = jnp.array(handle.sb_bond_jk_idx) if has_sb else None
    _sb_angle_idx = jnp.array(handle.sb_angle_idx) if has_sb else None
    sb_offset = _offsets["sb"]

    # Bond-dipole electrostatics (frozen — not in the param vector)
    has_dipoles = len(handle.dipole_pair_indices) > 0
    _dipole_moments = jnp.array(handle.dipole_moments) if has_dipoles else None
    _dipole_bond_indices = jnp.array(handle.dipole_bond_indices) if has_dipoles else None
    _dipole_pair_indices = jnp.array(handle.dipole_pair_indices) if has_dipoles else None

    if use_mm3:

        @jax.jit
        def energy_fn(params: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
            """Compute total MM3 energy from parameter and coordinate arrays."""
            E = jnp.float64(0.0)

            if has_bonds:
                bond_params = params[bond_offset : bond_offset + 2 * n_bt].reshape(n_bt, 2)
                k = bond_params[_bond_map, 0] * _BOND_K_CONV
                r0 = bond_params[_bond_map, 1]
                E = E + _mm3_bond_energy(k, r0, coords, _bond_indices)

            if has_angles:
                angle_params = params[angle_offset : angle_offset + 2 * n_at].reshape(n_at, 2)
                k = angle_params[_angle_map, 0] * _ANGLE_K_CONV
                theta0_deg = angle_params[_angle_map, 1]
                theta0 = theta0_deg * (jnp.pi / 180.0)
                E = E + _mm3_angle_energy(k, theta0, coords, _angle_indices)

            if has_torsions:
                torsion_k = params[torsion_offset : torsion_offset + n_tt]
                k = torsion_k[_torsion_map]
                E = E + _torsion_energy(k, _torsion_n, _torsion_gamma, coords, _torsion_indices)

            if has_vdw:
                vdw_params = params[vdw_offset : vdw_offset + 2 * n_vt].reshape(n_vt, 2)
                per_atom_radius = vdw_params[_atom_vdw_map, 0]
                per_atom_epsilon = vdw_params[_atom_vdw_map, 1]
                E = E + _mm3_vdw_energy(per_atom_radius, per_atom_epsilon, coords, _vdw_pairs)

            if has_ub:
                ub_params = params[ub_offset : ub_offset + 2 * n_ubt].reshape(n_ubt, 2)
                k = ub_params[_ub_map, 0]
                r0 = ub_params[_ub_map, 1]
                E = E + _ub_energy(k, r0, coords, _ub_indices)

            if has_sb:
                sb_k = params[sb_offset : sb_offset + n_sbt]
                k = sb_k[_sb_map]
                # Read equilibrium values from the runtime param vector
                bond_params_sb = params[bond_offset : bond_offset + 2 * n_bt].reshape(n_bt, 2)
                angle_params_sb = params[angle_offset : angle_offset + 2 * n_at].reshape(n_at, 2)
                r0_ij = bond_params_sb[_sb_bond_ij_idx, 1]
                r0_jk = bond_params_sb[_sb_bond_jk_idx, 1]
                theta0 = angle_params_sb[_sb_angle_idx, 1] * (jnp.pi / 180.0)
                E = E + _mm3_stretch_bend_energy(k, r0_ij, r0_jk, theta0, coords, _sb_angle_indices)

            if has_dipoles:
                E = E + _mm3_dipole_energy(_dipole_moments, coords, _dipole_bond_indices, _dipole_pair_indices)

            return E

    else:

        @jax.jit
        def energy_fn(params: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
            """Compute total harmonic energy from parameter and coordinate arrays."""
            E = jnp.float64(0.0)

            if has_bonds:
                bond_params = params[bond_offset : bond_offset + 2 * n_bt].reshape(n_bt, 2)
                k = bond_params[_bond_map, 0] * _BOND_K_CONV
                r0 = bond_params[_bond_map, 1]
                E = E + _harmonic_bond_energy(k, r0, coords, _bond_indices)

            if has_angles:
                angle_params = params[angle_offset : angle_offset + 2 * n_at].reshape(n_at, 2)
                k = angle_params[_angle_map, 0] * _ANGLE_K_CONV
                theta0_deg = angle_params[_angle_map, 1]
                theta0 = theta0_deg * (jnp.pi / 180.0)
                E = E + _harmonic_angle_energy(k, theta0, coords, _angle_indices)

            if has_torsions:
                torsion_k = params[torsion_offset : torsion_offset + n_tt]
                k = torsion_k[_torsion_map]
                E = E + _torsion_energy(k, _torsion_n, _torsion_gamma, coords, _torsion_indices)

            if has_vdw:
                vdw_params = params[vdw_offset : vdw_offset + 2 * n_vt].reshape(n_vt, 2)
                per_atom_radius = vdw_params[_atom_vdw_map, 0]
                per_atom_epsilon = vdw_params[_atom_vdw_map, 1]
                per_atom_sigma = per_atom_radius * 2.0 / (2.0 ** (1.0 / 6.0))
                E = E + _lj_12_6_energy(per_atom_sigma, per_atom_epsilon, coords, _vdw_pairs)

            if has_ub:
                ub_params = params[ub_offset : ub_offset + 2 * n_ubt].reshape(n_ubt, 2)
                k = ub_params[_ub_map, 0]
                r0 = ub_params[_ub_map, 1]
                E = E + _ub_energy(k, r0, coords, _ub_indices)

            return E

    return energy_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
