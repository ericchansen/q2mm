"""Seminario/QFUERZA force constant estimation using Q2MM's clean data models.

Estimates bond and angle force constants directly from a QM Hessian matrix
using the Seminario (FUERZA) projection method. This implementation uses
Q2MM's internal models (Molecule, ForceField) instead of the legacy
MM3-specific data structures.

Reference:
    Farrugia et al., J. Chem. Theory Comput. 2025, 22, 469-476.
    Seminario, Int. J. Quantum Chem. 1996, 60, 1271-1277.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
import logging
import numpy as np
from typing import Literal

from q2mm.constants import BOHR_TO_ANG
from q2mm.models.units import (
    AU_BOND_K_TO_CANONICAL,
    AU_ANGLE_K_TO_CANONICAL,
    MDYNA_RAD2_TO_KCALMOLRAD2,
)
from q2mm.models.molecule import Molecule
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm, TorsionParam

# AU → canonical: Hartree/Bohr² → kcal/(mol·Å²) for bonds,
#                  Hartree/rad² → kcal/(mol·rad²) for angles.
HARTREE_BOHR2_TO_KCALMOLA2 = AU_BOND_K_TO_CANONICAL
HARTREE_RAD2_TO_KCALMOLRAD2 = AU_ANGLE_K_TO_CANONICAL
from q2mm.models.hessian import invert_ts_curvature as _invert_ts_curvature

logger = logging.getLogger(__name__)

# Default DFT Hessian scaling factor (B3LYP/6-31G* level).
# See: Scott & Radom, J. Phys. Chem. 1996, 100, 16502-16513.
DEFAULT_DFT_SCALING = 0.963

# QFUERZA empirical default for hydrogen angle bends (mdyn·Å/rad²).
# Farrugia et al., J. Chem. Theory Comput. 2025, 22, 469-476, Table 1.
QFUERZA_H_ANGLE_DEFAULT_MDYNA = 0.5
QFUERZA_H_ANGLE_DEFAULT_CANONICAL = QFUERZA_H_ANGLE_DEFAULT_MDYNA * MDYNA_RAD2_TO_KCALMOLRAD2


def _is_hydrogen_angle(elements: tuple[str, str, str]) -> bool:
    """Return True if either outer atom of an angle is hydrogen."""
    return elements[0] == "H" or elements[2] == "H"


def _coerce_molecules(
    molecule: Molecule | Iterable[Molecule],
) -> list[Molecule]:
    """Normalize a single molecule or iterable of molecules into a list."""
    if isinstance(molecule, Molecule):
        return [molecule]

    molecules = list(molecule)
    if not molecules:
        raise ValueError("At least one molecule is required")
    if not all(isinstance(item, Molecule) for item in molecules):
        raise TypeError("qfuerza_fresh/qfuerza_into expects Molecule instances")
    return molecules


def _match_mode(param: BondParam | AngleParam, items: list) -> str:
    """Choose the most specific available matching strategy for parameters."""
    if param.ff_row is not None and any(item.ff_row is not None for item in items):
        return "ff_row"
    if param.env_id and any(item.env_id for item in items):
        return "env_id"
    return "elements"


def _collect_matching(
    molecules: list[Molecule],
    param: BondParam | AngleParam,
    items_attr: str,
    element_key_attr: str,
) -> list[tuple[Molecule, object]]:
    """Collect all items (bonds or angles) across molecules that match a parameter."""
    all_items = [(mol, item) for mol in molecules for item in getattr(mol, items_attr)]
    match_mode = _match_mode(param, [item for _, item in all_items])
    if match_mode == "ff_row":
        return [(mol, item) for mol, item in all_items if item.ff_row == param.ff_row]
    if match_mode == "env_id":
        return [(mol, item) for mol, item in all_items if item.env_id == param.env_id]
    return [(mol, item) for mol, item in all_items if getattr(item, element_key_attr) == param.key]


def _should_keep_force_constant(value: float, invalid_policy: Literal["keep", "skip"]) -> bool:
    """Decide whether a projected force constant should contribute to an average."""
    if np.iscomplexobj(value):
        return False
    if invalid_policy == "skip":
        return value > 0.0
    return True


def _project_hessian_block(hessian: np.ndarray, atom_i: int, atom_j: int, coords: np.ndarray, au_units: bool) -> float:
    """Project a single Hessian sub-block onto the bond vector.

    Returns the projected force constant in atomic units (Hartree/Bohr^2)
    or input units if au_units=False.
    """
    if au_units:
        coords_work = coords / BOHR_TO_ANG
    else:
        coords_work = coords.copy()

    r_vec = coords_work[atom_j] - coords_work[atom_i]
    r_len = np.linalg.norm(r_vec)
    if r_len < 1e-10:
        return 0.0
    r_hat = r_vec / r_len

    i3, j3 = 3 * atom_i, 3 * atom_j
    h_sub = -hessian[i3 : i3 + 3, j3 : j3 + 3]

    # General eigenvalue decomposition (NOT eigh — sub-block is NOT symmetric)
    eigenvalues, eigenvectors = np.linalg.eig(h_sub)
    # Keep complex eigenpairs through projection — only take real at the end
    # (upstream seminario_sum uses np.abs on complex dot products)

    # Seminario projection: k = sum_n lambda_n * |e_n · r_hat|
    # np.abs handles complex dot products correctly (returns magnitude)
    k = 0.0
    for n in range(3):
        k += eigenvalues[n] * np.abs(np.dot(eigenvectors[:, n], r_hat))
    # Result should be real (imaginary parts cancel in conjugate pairs)
    return k.real


def seminario_bond_fc(
    atom_i: int,
    atom_j: int,
    coords: np.ndarray,
    hessian: np.ndarray,
    au_units: bool = True,
    dft_scaling: float = DEFAULT_DFT_SCALING,
) -> float:
    """Estimate bond stretching force constant via Seminario method.

    Averages the i->j and j->i projections (bidirectional) to match
    the original Seminario method and upstream Q2MM implementation.

    Args:
        atom_i (int): 0-based index of the first atom.
        atom_j (int): 0-based index of the second atom.
        coords (np.ndarray): Atomic coordinates, shape (N, 3) in Angstrom.
        hessian (np.ndarray): Full Cartesian Hessian, shape (3N, 3N).
        au_units (bool): If True, Hessian is in Hartree/Bohr^2 (Gaussian/Psi4 default).
        dft_scaling (float): Scaling factor for DFT Hessians (default 0.963).

    Returns:
        Force constant in kcal/(mol·Å²) (scaled)

    """
    # Bidirectional: compute i->j and j->i, then average
    f_ij = _project_hessian_block(hessian, atom_i, atom_j, coords, au_units)
    f_ji = _project_hessian_block(hessian, atom_j, atom_i, coords, au_units)
    k_bond = 0.5 * (f_ij + f_ji) * dft_scaling

    # Convert to canonical kcal/(mol·Å²)
    if au_units:
        k_bond *= HARTREE_BOHR2_TO_KCALMOLA2

    return k_bond


def seminario_angle_fc(
    atom_i: int,
    atom_j: int,
    atom_k: int,
    coords: np.ndarray,
    hessian: np.ndarray,
    au_units: bool = True,
    dft_scaling: float = DEFAULT_DFT_SCALING,
) -> float:
    """Estimate angle bending force constant via modified Seminario method.

    Computes the standard Seminario reciprocal-sum angle formula with
    |dot| projection and DFT scaling. Note that FUERZA overestimates
    H-angle FCs by ~2×; the QFUERZA substitution is applied downstream
    in ``qfuerza_into()`` (or ``qfuerza_fresh()``), not here.

    Args:
        atom_i: outer atom (0-based)
        atom_j: center atom (0-based)
        atom_k: outer atom (0-based)
        coords: Atomic coordinates, shape (N, 3) in Angstrom
        hessian: Full Cartesian Hessian, shape (3N, 3N)
        au_units: If True, Hessian is in Hartree/Bohr^2
        dft_scaling: Scaling factor for DFT Hessians (default 0.963)

    Returns:
        Force constant in kcal/(mol·rad²) (scaled)

    """
    if au_units:
        coords_work = coords / BOHR_TO_ANG
    else:
        coords_work = coords.copy()

    # Vectors from center to outer atoms
    r_ij = coords_work[atom_i] - coords_work[atom_j]
    r_kj = coords_work[atom_k] - coords_work[atom_j]
    r_ij_len = np.linalg.norm(r_ij)
    r_kj_len = np.linalg.norm(r_kj)

    if r_ij_len < 1e-10 or r_kj_len < 1e-10:
        return 0.0

    r_ij_hat = r_ij / r_ij_len
    r_kj_hat = r_kj / r_kj_len

    # Normal to the angle plane
    cross = np.cross(r_ij_hat, r_kj_hat)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-10:
        # Linear angle — use a perpendicular direction
        perp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(r_ij_hat, perp)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        cross = np.cross(r_ij_hat, perp)
        cross_norm = np.linalg.norm(cross)

    n_hat = cross / cross_norm

    # Perpendicular unit vectors in the angle plane
    u_ij = np.cross(n_hat, r_ij_hat)
    u_ij /= np.linalg.norm(u_ij)
    u_kj = np.cross(n_hat, r_kj_hat)
    u_kj /= np.linalg.norm(u_kj)

    # Sub-block Hessians — keep complex eigenpairs through projection
    i3, j3, k3 = 3 * atom_i, 3 * atom_j, 3 * atom_k

    # For i-j interaction
    h_ij = -hessian[i3 : i3 + 3, j3 : j3 + 3]
    evals_ij, evecs_ij = np.linalg.eig(h_ij)
    k_ij = 0.0
    for n in range(3):
        k_ij += evals_ij[n] * np.abs(np.dot(evecs_ij[:, n], u_ij))
    k_ij = k_ij.real

    # For k-j interaction
    h_kj = -hessian[k3 : k3 + 3, j3 : j3 + 3]
    evals_kj, evecs_kj = np.linalg.eig(h_kj)
    k_kj = 0.0
    for n in range(3):
        k_kj += evals_kj[n] * np.abs(np.dot(evecs_kj[:, n], u_kj))
    k_kj = k_kj.real

    # Combine via reciprocal sum: 1/k = 1/(k_ij·r_ij²) + 1/(k_kj·r_kj²)
    # Standard Seminario angle formula. Note: FUERZA still overestimates
    # H-angle FCs by ~2× (Allen et al. 2018); QFUERZA addresses this via
    # substitution in qfuerza_into().
    denom_ij = k_ij * r_ij_len**2
    denom_kj = k_kj * r_kj_len**2

    if abs(denom_ij) < 1e-10 or abs(denom_kj) < 1e-10:
        return 0.0

    k_angle = 1.0 / (1.0 / denom_ij + 1.0 / denom_kj)

    # Apply DFT scaling
    k_angle *= dft_scaling

    # Convert: Hartree/rad^2 -> canonical kcal/(mol·rad²)
    if au_units:
        k_angle *= HARTREE_RAD2_TO_KCALMOLRAD2

    return k_angle


def qfuerza_fresh(
    molecule: Molecule | Iterable[Molecule],
    *,
    functional_form: FunctionalForm,
    zero_torsions: bool = True,
    au_hessian: bool = True,
    invalid_policy: Literal["keep", "skip"] = "keep",
    invert_ts_curvature: bool = False,
    replace_with: float = 1.0,
    strategy: Literal["fuerza", "qfuerza"] = "qfuerza",
) -> ForceField:
    """Build a fresh force field from one molecule's QM Hessian.

    Use this when you have no published FF to start from — e.g.
    ``load_ch3f``.  All params come from the QFUERZA projection
    (Farrugia et al. *J. Chem. Theory Comput.* **2025**, *22*, 469;
    DOI 10.1021/acs.jctc.5c01751) and are returned unfrozen.

    Args:
        molecule: A single molecule with a Hessian attached, or an
            iterable containing exactly one such molecule.  Multi-molecule
            averaging requires an explicit forcefield — use
            :func:`qfuerza_into` for that case.
        functional_form: Required — every :class:`ForceField` must carry
            an explicit functional form (see
            :meth:`ForceField.create_for_molecule`). QFUERZA determines
            the initial scalar parameters; the caller separately declares
            the functional form used by its target workflow and backend.
            There is no default because the former unset value was
            interpreted differently by OpenMM and JAX.
        zero_torsions: Whether to zero out torsional parameters.  Per
            Farrugia 2025, torsions are not well-suited for FUERZA-style
            parametrization from vibrational data and are zeroed at
            initial-parameter time to be fit in a later Q2MM optimization
            stage.  Default ``True``; do not override unless you are
            sure you know what that means for your downstream pipeline.
        au_hessian: Whether the Hessian is in atomic units (Hartree/Bohr²).
        invalid_policy: ``"keep"`` retains negative force constants
            (TS reaction coordinates); ``"skip"`` mimics legacy MM3
            Seminario averaging by dropping non-positive estimates.
        invert_ts_curvature: When the molecule is a transition state,
            inverts the reaction-coordinate eigenvalue before projection
            so that the TS Hessian produces valid positive force
            constants (Limé & Norrby 2015).
        replace_with: Replacement value (Hartree/Bohr²) for the most
            negative eigenvalue when ``invert_ts_curvature=True``.
            Default ``1.0`` matches Limé & Norrby Method C.  Ignored
            when ``invert_ts_curvature=False``.
        strategy: ``"qfuerza"`` (default) applies the H-angle substitution
            for hydrogen angle bends where pure Seminario projection
            overestimates by ~2× (Farrugia 2025).  ``"fuerza"`` uses
            pure Seminario projection.

    Returns:
        A new :class:`ForceField` whose every parameter is unfrozen and
        populated from the QFUERZA projection, tagged with the
        caller-supplied *functional_form*.

    Raises:
        ValueError: If ``molecule`` is an iterable with anything other
            than one element (use :func:`qfuerza_into` for the multi-
            molecule averaging case), or if a Hessian is missing.

    """
    molecules = _coerce_molecules(molecule)
    if len(molecules) != 1:
        raise ValueError(
            f"qfuerza_fresh expects exactly one molecule, got {len(molecules)}.  "
            "Multi-molecule averaging requires an explicit forcefield — use "
            "qfuerza_into(ff, molecules, ...) instead."
        )
    ff = ForceField.create_for_molecule(
        molecules[0],
        name=f"QFUERZA FF for {molecules[0].name}",
        functional_form=functional_form,
    )
    return qfuerza_into(
        ff,
        molecules,
        zero_torsions=zero_torsions,
        au_hessian=au_hessian,
        invalid_policy=invalid_policy,
        invert_ts_curvature=invert_ts_curvature,
        replace_with=replace_with,
        strategy=strategy,
    )


def qfuerza_into(
    ff: ForceField,
    molecule: Molecule | Iterable[Molecule],
    *,
    active_bonds: frozenset[int] | None = None,
    active_angles: frozenset[int] | None = None,
    active_torsions: frozenset[int] | None = None,
    zero_torsions: bool = True,
    au_hessian: bool = True,
    invalid_policy: Literal["keep", "skip"] = "keep",
    invert_ts_curvature: bool = False,
    replace_with: float = 1.0,
    strategy: Literal["fuerza", "qfuerza"] = "qfuerza",
) -> ForceField:
    """Return a copy of *ff* with active parameter values overwritten by QFUERZA projection.

    Iterates every bond/angle/torsion in *ff* and computes new values
    from the QFUERZA projection (Farrugia et al. *J. Chem. Theory
    Comput.* **2025**, *22*, 469), but only *active* parameters are
    overwritten in the returned force field — inactive ones are carried
    over unchanged.  The caller (typically a
    ``q2mm.benchmarks.systems`` loader) decides which parameters are
    active, usually via
    :func:`q2mm.models.parameters.opt_substructure_membership` /
    :class:`~q2mm.models.parameters.ActiveParameterSpace`; silently
    overwriting a caller-fixed published OPT value was the q2mm#277
    Heck-relay bug.

    Args:
        ff: Force field whose active bond/angle/torsion values are
            re-estimated.  Not mutated; a new :class:`ForceField` is
            returned.
        molecule: Molecule(s) with Hessians; multi-molecule values are
            averaged per param.
        active_bonds: 0-based indices into ``ff.bonds`` that may be
            overwritten.  ``None`` (default) means every bond is active
            — the correct default for a freshly-built FF (e.g.
            :func:`qfuerza_fresh`) that has no frozen backbone.
        active_angles: 0-based indices into ``ff.angles`` whose
            *bending* (``force_constant``/``equilibrium``) values may
            be overwritten.  ``None`` means every angle is active.
        active_torsions: 0-based indices into ``ff.torsions`` that may
            be zeroed when *zero_torsions* is set.  ``None`` means
            every torsion is active.
        zero_torsions: See :func:`qfuerza_fresh` (default ``True`` per
            Farrugia 2025).
        au_hessian: Whether Hessians are in atomic units.
        invalid_policy: ``"keep"`` or ``"skip"`` for non-positive
            estimates.
        invert_ts_curvature: TS Hessian inversion flag (Limé & Norrby
            2015).
        replace_with: Replacement value (Hartree/Bohr²) for the most
            negative eigenvalue when ``invert_ts_curvature=True``.
            Default ``1.0`` matches Limé & Norrby Method C; smaller
            values reduce the chance of negative angle force constants
            in the QFUERZA projection but produce a softer
            reaction-coordinate mode.  Ignored when
            ``invert_ts_curvature=False``.
        strategy: ``"qfuerza"`` (with H-angle substitution) or
            ``"fuerza"`` (pure Seminario).

    Returns:
        A new :class:`ForceField` with active bond/angle/torsion values
        overwritten; inactive ones (and all other collections) are
        unchanged.

    Raises:
        ValueError: If ``strategy`` is not recognised or if any
            molecule is missing a Hessian.

    """
    if strategy not in ("fuerza", "qfuerza"):
        raise ValueError(f"Unsupported strategy {strategy!r}; expected 'fuerza' or 'qfuerza'")

    molecules = _coerce_molecules(molecule)
    if any(item.hessian is None for item in molecules):
        raise ValueError("Molecule must have a Hessian attached. Use molecule.with_hessian(hess)")

    # Invert TS curvature if this is a transition-state Hessian
    if invert_ts_curvature:
        processed_hessians: dict[int, np.ndarray] = {}
        for mol in molecules:
            processed_hessians[id(mol)] = _invert_ts_curvature(mol.hessian, replace_with=replace_with)
    else:
        processed_hessians = None

    return _estimate_into_ff(
        ff,
        molecules,
        zero_torsions=zero_torsions,
        au_hessian=au_hessian,
        invalid_policy=invalid_policy,
        processed_hessians=processed_hessians,
        strategy=strategy,
        active_bonds=active_bonds,
        active_angles=active_angles,
        active_torsions=active_torsions,
    )


def _estimate_bond(
    bond_param: BondParam,
    molecules: list[Molecule],
    *,
    au_hessian: bool,
    invalid_policy: Literal["keep", "skip"],
    processed_hessians: dict[int, np.ndarray] | None,
) -> BondParam:
    """Return *bond_param* with its value re-estimated via Seminario projection."""
    matching_bonds = _collect_matching(molecules, bond_param, "bonds", "element_pair")
    if not matching_bonds:
        logger.debug("No bonds match %s in molecule", bond_param.key)
        return bond_param

    force_constants = []
    equilibria = [bond.length for _, bond in matching_bonds]
    for molecule_item, bond in matching_bonds:
        hess = processed_hessians[id(molecule_item)] if processed_hessians else molecule_item.hessian
        k = seminario_bond_fc(
            bond.atom_i,
            bond.atom_j,
            molecule_item.geometry,
            hess,
            au_units=au_hessian,
        )
        if _should_keep_force_constant(k, invalid_policy):
            force_constants.append(float(np.real(k)))
            if k < 0:
                logger.warning(
                    f"  Bond {bond.elements} ({bond.atom_i}-{bond.atom_j}): "
                    f"negative FC = {k:.4f} (TS reaction coordinate?)"
                )
        else:
            logger.warning(f"  Bond {bond.elements} ({bond.atom_i}-{bond.atom_j}): invalid FC = {k} — skipped")

    updates: dict[str, float] = {}
    if equilibria:
        updates["equilibrium"] = float(np.mean(equilibria))
    if force_constants:
        updates["force_constant"] = float(np.mean(force_constants))
        logger.info(
            f"  Bond {bond_param.key}: k={updates['force_constant']:.4f} kcal/(mol·Å²), "
            f"r0={updates.get('equilibrium', bond_param.equilibrium):.4f} Å"
        )
    else:
        logger.warning(f"  Bond {bond_param.key}: no valid force constants found, keeping existing force constant")
    return dataclasses.replace(bond_param, **updates) if updates else bond_param


def _estimate_angle(
    angle_param: AngleParam,
    molecules: list[Molecule],
    *,
    au_hessian: bool,
    invalid_policy: Literal["keep", "skip"],
    processed_hessians: dict[int, np.ndarray] | None,
    strategy: Literal["fuerza", "qfuerza"],
) -> AngleParam:
    """Return *angle_param* with its value re-estimated via Seminario/QFUERZA projection."""
    matching_angles = _collect_matching(molecules, angle_param, "angles", "element_triple")
    if not matching_angles:
        logger.debug("No angles match %s in molecule", angle_param.key)
        return angle_param

    force_constants = []
    equilibria = [angle.value for _, angle in matching_angles]
    for molecule_item, angle in matching_angles:
        hess = processed_hessians[id(molecule_item)] if processed_hessians else molecule_item.hessian
        k = seminario_angle_fc(
            angle.atom_i,
            angle.atom_j,
            angle.atom_k,
            molecule_item.geometry,
            hess,
            au_units=au_hessian,
        )
        if _should_keep_force_constant(k, invalid_policy):
            force_constants.append(float(np.real(k)))
            if k < 0:
                logger.warning(f"  Angle {angle.elements}: negative FC = {k:.4f}")
        else:
            logger.warning(f"  Angle {angle.elements}: invalid FC = {k} — skipped")

    updates: dict[str, float] = {}
    if equilibria:
        updates["equilibrium"] = float(np.mean(equilibria))
    if force_constants:
        fuerza_value = float(np.mean(force_constants))
        if strategy == "qfuerza" and _is_hydrogen_angle(angle_param.elements):
            updates["force_constant"] = QFUERZA_H_ANGLE_DEFAULT_CANONICAL
            logger.info(
                f"  Angle {angle_param.key}: QFUERZA H-angle substitution — "
                f"{QFUERZA_H_ANGLE_DEFAULT_MDYNA} mdyn·Å/rad² "
                f"(FUERZA was {fuerza_value:.4f} kcal/(mol·rad²))"
            )
        else:
            updates["force_constant"] = fuerza_value
            logger.info(
                f"  Angle {angle_param.key}: k={updates['force_constant']:.4f}, "
                f"theta0={updates.get('equilibrium', angle_param.equilibrium):.1f} deg"
            )
    else:
        logger.warning(f"  Angle {angle_param.key}: no valid force constants found, keeping existing force constant")
    return dataclasses.replace(angle_param, **updates) if updates else angle_param


def _estimate_into_ff(
    ff: ForceField,
    molecules: list[Molecule],
    *,
    zero_torsions: bool,
    au_hessian: bool,
    invalid_policy: Literal["keep", "skip"],
    processed_hessians: dict[int, np.ndarray] | None,
    strategy: Literal["fuerza", "qfuerza"],
    active_bonds: frozenset[int] | None,
    active_angles: frozenset[int] | None,
    active_torsions: frozenset[int] | None,
) -> ForceField:
    """Return a new ForceField with active bond/angle/torsion values re-estimated.

    Parameters whose collection index is not in the corresponding
    ``active_*`` set (when given) are carried over unchanged — they
    represent caller commitments (e.g. published OPT values held fixed
    via an :class:`~q2mm.models.parameters.ActiveParameterSpace`) and
    silently overwriting them was the q2mm#277 Heck-relay bug.
    ``None`` means every parameter in that collection is active.
    """
    new_bonds: list[BondParam] = []
    for i, bond_param in enumerate(ff.bonds):
        if active_bonds is not None and i not in active_bonds:
            new_bonds.append(bond_param)
            continue
        new_bonds.append(
            _estimate_bond(
                bond_param,
                molecules,
                au_hessian=au_hessian,
                invalid_policy=invalid_policy,
                processed_hessians=processed_hessians,
            )
        )

    new_angles: list[AngleParam] = []
    for i, angle_param in enumerate(ff.angles):
        if active_angles is not None and i not in active_angles:
            new_angles.append(angle_param)
            continue
        new_angles.append(
            _estimate_angle(
                angle_param,
                molecules,
                au_hessian=au_hessian,
                invalid_policy=invalid_policy,
                processed_hessians=processed_hessians,
                strategy=strategy,
            )
        )

    new_torsions: list[TorsionParam] = list(ff.torsions)
    if zero_torsions:
        new_torsions = [
            dataclasses.replace(t, force_constant=0.0) if (active_torsions is None or i in active_torsions) else t
            for i, t in enumerate(new_torsions)
        ]

    return dataclasses.replace(ff, bonds=tuple(new_bonds), angles=tuple(new_angles), torsions=tuple(new_torsions))
