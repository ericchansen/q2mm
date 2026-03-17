#!/usr/bin/env python3
"""
Estimates bond and angle force constants, prepares a force field for Q2MM parameterization.

Script takes in the structural, hessian, and ff files to output a new forcefield with initial
parameters estimated via the Seminario/FUERZA method. Currently, only mol2, Gaussian .log, and
Amber .frcmod or MacroModel MM3* .fld filetypes are supported (respectively). If requested, it
will do this estimation after inverting the Hessian for a TSFF or zero out the values necessary
to be 0 before starting parameterization of the force field. Currently, it will always set
equilibrium bond length and angle parameter values to the average of the structural files given
as input.

Torsion parameter value estimation is not currently supported both due to lack of treatment in
Jorge Seminario's original 1998 paper, but also because we keep them at zero during the Q2MM TSFF
parameterization until all other parameter values are well-determined, otherwise the torsions will
simply be random.

This step should be done after RESP calculation, generation of a force field file, and testing that
ensures that all interactions in the structures are covered by a corresponding parameter in the
force field file (compatibility via compare.py or calculate.py usage typically).

Portions of code adapted from Samuel Genheden and can be found at
https://github.com/SGenheden/Seminario
"""

import argparse
from collections import Counter
import copy
import glob
import os
import sys
from typing import List

import numpy as np
try:
    import parmed
except ImportError:
    parmed = None

from q2mm.linear_algebra import invert_ts_curvature, reform_hessian

import logging
import logging.config
from q2mm import constants as co
from q2mm.schrod_indep_filetypes import (
    FF,
    Atom,
    GaussLog,
    MM3,
    AmberFF,
    JaguarIn,
    MacroModel,
    Mol2,
    Param,
    Structure,
    mass_weight_force_constant,
    mass_weight_hessian,
)
from q2mm import utilities


logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)

# region calculation_methods

# TODO perhaps move some of these to linear_algebra.py,
# can do when I reformat the docstrings to Google style after I've tested the method.
# Preferably, have it done before publication though.


def sub_hessian(hessian, atom1: Atom, atom2: Atom, ang_to_bohr=False) -> tuple:
    """Subsample the Hessian matrix by pulling out the terms relevant to the
    bond that is formed between atom1 and atom2 as well as calculating the
    vector from atom1 to atom2.

    Args:
        hessian (np.ndarray): Hessian matrix
        atom1 (Atom): first atom
        atom2 (Atom): second atom
        ang_to_bohr (bool, optional): whether to convert the hessian terms from
         Angstrom length units to Bohr radii units. Defaults to False.

    Returns:
        tuple: tuple of (vector from atom1 to atom2, eigenvalues of the submatrix,
         eigenvector of the submatrix)
    """
    vec12 = atom1.coords - atom2.coords
    if ang_to_bohr:
        vec12 = vec12 / co.BOHR_TO_ANG
    vec12 = vec12 / np.linalg.norm(vec12)

    submat = -hessian[
        3 * (atom1.index - 1) : 3 * (atom1.index - 1) + 3,
        3 * (atom2.index - 1) : 3 * (atom2.index - 1) + 3,
    ]
    eigval, eigvec = np.linalg.eig(submat)
    return vec12, eigval, eigvec


def get_unit_vector(atom1: Atom, atom2: Atom, ang_to_bohr=False) -> np.ndarray:
    """Get the unit vector between atom1 and atom2

    Args:
        atom1 (Atom): first atom
        atom2 (Atom): second atom
        ang_to_bohr (bool, optional): whether to convert the distance from Angstroms to Bohr radii. Defaults to False.

    Returns:
        numpy.ndarray: unit vector between atom1 and atom2
    """
    vec12 = atom1.coords - atom2.coords
    vec21 = atom2.coords - atom1.coords
    unit_vec = np.hstack((vec12, vec21))
    if ang_to_bohr:
        unit_vec = unit_vec / co.BOHR_TO_ANG
    unit_vec = unit_vec / np.linalg.norm(unit_vec)
    return unit_vec


def get_subhessian(hessian, atom1: Atom, atom2: Atom) -> np.ndarray:
    """Subsample the Hessian matrix by pulling out the terms relevant to the
    bond that is formed between atom1 and atom2 as well as calculating the
    vector from atom1 to atom2.

    Args:
        hessian (np.ndarray): Hessian matrix
        atom1 (Atom): first atom
        atom2 (Atom): second atom

    Returns:
        np.ndarray: submatrix of the Hessian relevant to atom1-atom2 interactions.
    """
    submat_11 = (
        0
        * hessian[
            3 * atom1.index - 3 : 3 * atom1.index, 3 * atom1.index - 3 : 3 * atom1.index
        ]
    )
    submat_12 = -hessian[
        3 * atom1.index - 3 : 3 * atom1.index, 3 * atom2.index - 3 : 3 * atom2.index
    ]
    submat_21 = -hessian[
        3 * atom2.index - 3 : 3 * atom2.index, 3 * atom1.index - 3 : 3 * atom1.index
    ]
    submat_22 = (
        0
        * hessian[
            3 * atom2.index - 3 : 3 * atom2.index, 3 * atom2.index - 3 : 3 * atom2.index
        ]
    )
    submat_1 = np.hstack((submat_11, submat_12))
    submat_2 = np.hstack((submat_21, submat_22))
    submat = np.vstack((submat_1, submat_2))

    return submat


def po_sum(unit_vec, eigval, eigvec):
    """
    Average the projections of the eigenvector on a specific unit vector
    according to Seminario

    Parameters
    ----------
    vec : numpy.ndarray
        the unit vector
    eigval : numpy.ndarray
        the eigenvalues of a Hessian submatrix
    eigvec : numpy.ndarray
        the eigenvectors of a Hessian submatrix

    Returns
    -------
    float :
        the average projection
    """
    ssum = 0.0
    for i in range(len(eigval)):
        ssum += eigval[i] * np.sum(
            np.abs(np.dot((np.dot(eigvec[:, i], unit_vec)), unit_vec.T))
        )
    return ssum


def seminario_sum(vec, eigval, eigvec) -> float:
    """Average the projections of the Hessian eigenvector on a specific interaction unit vector
    according to FUERZA

    Args:
        vec (np.ndarray): unit vector (atomic interaction)
        eigval (np.ndarray): eigenvalues of Hessian submatrix
        eigvec (np.ndarray): eigenvectors of Hessian submatrix

    Returns:
        float: the averaged projection to serve as a force constant estimate
    """
    ssum = 0.0
    for i in range(3):
        ssum += eigval[i] * np.abs(np.dot(eigvec[:, i], vec))
    return ssum


def po_bond_alt(bond, hessian, scaling=0.963, convert=False) -> float:
    unit_vector_ab = get_unit_vector(bond.atom1, bond.atom2)
    subhessian = get_subhessian(hessian, bond.atom1, bond.atom2)
    eigval, eigvec = np.linalg.eig(subhessian)

    ab_sum = po_sum(unit_vector_ab, eigval, eigvec)

    # 2240.87 is from Hartree/Bohr ^2 to kcal/mol/A^2
    # 418.4 is kcal/mol/A^2 to kJ/mol/nm^2

    if convert:
        return scaling * 2240.87 * 418.4 * 0.5 * po_sum
    else:
        return scaling * ab_sum * 0.5


def po_bond(atoms: list, hessian, scaling=0.963, convert=False, ang_to_bohr=False):
    unit_vector_ab = get_unit_vector(atoms[0], atoms[1], ang_to_bohr=ang_to_bohr)
    subhessian = get_subhessian(hessian, atoms[0], atoms[1])

    unit_vector_onto_subhess = np.dot(unit_vector_ab, subhessian)
    product = np.dot(unit_vector_onto_subhess, unit_vector_ab.T)

    # 2240.87 is from Hartree/Bohr ^2 to kcal/mol/A^2
    # 418.4 is kcal/mol/A^2 to kJ/mol/nm^2

    if convert:
        return scaling * 2240.87 * 418.4 * product
    else:
        return scaling * product


def seminario_bond(atoms: list, hessian, scaling=0.963, ang_to_bohr=False) -> float:
    """Estimate the bond force constant using the Seminario method, i.e. by
    analysing the Hessian submatrix. Will average over atom1->atom2 and
    atom2->atom1 force constants.

    Args:
        atoms (list[Atom]): atoms involved in the bond for which to estimate a force constant
        hessian (np.ndarray): Hessian matrix
        scaling (float, optional): Hessian scaling factor, dependent on Hessian calculation level of theory. Defaults to 0.963 for DFT.
        ang_to_bohr (bool, optional): whether to convert length units from Angstrom to Bohr radii. Defaults to False.

    Note:
        Multiplication by a scaling factor is necessary because different Quantum-level
        calculations are often off by some consistent factor. In the case of DFT, values
        are typically overestimated, so DFT calculations must be scaled down by ~0.963x,
        hence the default scaling of 0.963.

    Returns:
        float: estimated bond force constant in AU
    """

    vec12, eigval12, eigvec12 = sub_hessian(
        hessian, atoms[0], atoms[1], ang_to_bohr=ang_to_bohr
    )
    f12 = seminario_sum(vec12, eigval12, eigvec12)

    vec21, eigval21, eigvec21 = sub_hessian(
        hessian, atoms[1], atoms[0], ang_to_bohr=ang_to_bohr
    )

    f21 = seminario_sum(vec21, eigval21, eigvec21)

    # 2240.87 is from Hartree/Bohr ^2 to kcal/mol/A^2
    # 418.4 is kcal/mol/A^2 to kJ/mol/nm^2

    if f12 <= 0 or f21 <= 0:
        logger.log(
            logging.DEBUG,
            f"Estimated force constant between atoms: {atoms[0].index}, {atoms[1].index} <= 0, with raw kJ/molA estimates: {f12} {f21}!\nSetting to 0.5 instead.",
        )

    if np.iscomplexobj(f12):
        logger.log(logging.DEBUG, "Complex number in estimate for bond f12 ("+str(f12)+"): "+str(atoms))
        if not np.iscomplex(f12):
            f12 = np.real(f12)
        else:
            logger.log(logging.WARN, "WARNING: Non-zero imaginary component of bond force constant estimate for angle ("+str(f12)+"): "+str(atoms)+".")

    if np.iscomplexobj(f21):
        logger.log(logging.DEBUG, "Complex number in estimate for bond f21 ("+str(f21)+"): "+str(atoms))
        if not np.iscomplex(f21):
            f21 = np.real(f21)
        else:
            logger.log(logging.WARN, "WARNING: Non-zero imaginary component of bond force constant estimate for angle ("+str(f21)+"): "+str(atoms)+".")

    f = 0.5 * (f12 + f21)
    if f <= 0:
        logger.log(
            logging.WARNING,
            f"WARNING: Estimated force constant between atoms: {atoms[0].index}, {atoms[1].index} <= 0, with (raw, scaled) kJ/molA estimate: {f} {f*scaling}!\nPlease visualize normal modes to confirm transition states are correct.",
        )

    return scaling * f


def po_angle(
    atoms: list, hessian, scaling=0.963, convert=False, ang_to_bohr=False
) -> float:
    """
    Estimate the angle force constant using the Seminario method, i.e. by
    analysing the Hessian submatrix. Return in AU

    Parameters
    ----------
    angle : parmed.Angle
        the angle to estimate the force constant for
    hessian : numpy.ndarray
        the Hessian matrix
    scaling : float
        the Hessian scaling factor
    """
    assert len(atoms) == 3

    unit_vector_12 = get_unit_vector(atoms[0], atoms[1], ang_to_bohr=ang_to_bohr)
    unit_vector_32 = get_unit_vector(atoms[2], atoms[1], ang_to_bohr=ang_to_bohr)

    subhess_12 = get_subhessian(hessian=hessian, atom1=atoms[0], atom2=atoms[1])
    subhess_32 = get_subhessian(hessian=hessian, atom1=atoms[2], atom2=atoms[1])

    un = np.cross(unit_vector_32, unit_vector_12)
    un = un / np.linalg.norm(un)
    upa = np.cross(un, unit_vector_12)
    upc = np.cross(unit_vector_32, un)

    upa_onto_subhess_12 = np.dot(upa, subhess_12)
    product_12 = np.dot(upa_onto_subhess_12, unit_vector_12.T)
    upc_vector_onto_subhess_32 = np.dot(upc, subhess_32)
    product_32 = np.dot(upc_vector_onto_subhess_32, unit_vector_32.T)

    len12 = utilities.measure_bond(atoms[0].coords, atoms[1].coords)
    if ang_to_bohr:
        len12 = len12 / co.BOHR_TO_ANG
    len32 = utilities.measure_bond(atoms[2].coords, atoms[1].coords)
    if ang_to_bohr:
        len32 = len32 / co.BOHR_TO_ANG

    f = 1.0 / (1.0 / (product_12 * len12 * len12) + 1.0 / (product_32 * len32 * len32))

    # unit_vector_ab = create_unit_vector(angle.atom1, angle.atom2, ang_to_bohr=ang_to_bohr)
    # subhessian = get_subhessian(hessian, angle.atom1, angle.atom2)

    # unit_vector_onto_subhess = np.dot(unit_vector_ab, subhessian)
    # product = np.dot(unit_vector_onto_subhess, unit_vector_ab.T)

    # 627.5095 is Hatree to kcal/mol
    # 4.184 is kcal/mol to kJ/mol
    if convert:
        return scaling * 627.5095 * 4.184 * f
    else:
        return scaling * f


def seminario_angle(atoms:list, hessian, scaling=0.963, convert=False, ang_to_bohr=False) -> float:
    """Estimate the angle force constant using the Seminario method, i.e. by
    analysing the Hessian submatrix.

    Args:
        atoms (list[Atom]): list of Atom objects for which to estimate the angle force constant
        hessian (numpy.ndarray): Hessian matrix
        scaling (float, optional): Hessian scaling factor, dependent on Hessian calculation level of theory. Defaults to 0.963 for DFT.
        convert (bool, optional): whether to convert force constant from AU to kJ/mol. Defaults to False.
        ang_to_bohr (bool, optional): whether to convert structure length units from Angstrom to Bohr. Defaults to False.

    Note:
        Multiplication by a scaling factor is necessary because different Quantum-level
        calculations are often off by some consistent factor. In the case of DFT, values
        are typically overestimated, so DFT calculations must be scaled down by ~0.963x,
        hence the default scaling of 0.963.

    Returns:
        float: estimated angle force constant
    """

    assert len(atoms) == 3
    vec12, eigval12, eigvec12 = sub_hessian(hessian, atoms[0], atoms[1])
    vec32, eigval32, eigvec32 = sub_hessian(hessian, atoms[2], atoms[1])

    un = np.cross(vec32, vec12)
    un = un / np.linalg.norm(un)
    upa = np.cross(un, vec12)
    upc = np.cross(vec32, un)

    sum1 = seminario_sum(upa, eigval12, eigvec12)
    sum2 = seminario_sum(upc, eigval32, eigvec32)


    len12 = utilities.measure_bond(atoms[0].coords, atoms[1].coords)
    if ang_to_bohr:
        len12 = len12 / co.BOHR_TO_ANG
    len32 = utilities.measure_bond(atoms[2].coords, atoms[1].coords)
    if ang_to_bohr:
        len32 = len32 / co.BOHR_TO_ANG

    f12 = 1.0 / (sum1 * len12 * len12)
    f32 = 1.0 / (sum2 * len32 * len32)

    if f12 <= 0 or f32 <= 0:
        logger.log(
            logging.DEBUG,
            f"Estimated force constant between atoms: {atoms[0].index}, {atoms[1].index}, {atoms[2].index} <= 0, with raw kJ/molA estimates: {f12}, {f32}!",
        )

    f = 1.0 / (1.0 / (sum1 * len12 * len12) + 1.0 / (sum2 * len32 * len32))

    if f < 0:
        logger.log(
            logging.WARNING,
            f"WARNING: Estimated force constant between atoms: {atoms[0].index}, {atoms[1].index}, {atoms[2].index} < 0, with (raw, scaled) kJ/molA estimate: {f} {f*scaling}!\nPlease visualize normal modes to confirm transition states are correct.",
        )

    if np.iscomplexobj(f):
        logger.log(logging.DEBUG, "Complex number in estimate for angle("+str(f)+"): "+str(atoms))
        if not np.iscomplex(f):
            f = np.real(f)
        else:
            logger.log(logging.WARN, "WARNING: Non-zero imaginary component of angle force constant estimate for angle ("+str(f)+"): "+str(atoms))

    # 627.5095 is Hartree to kcal/mol
    # 4.184 is kcal/mol to kJ/mol
    if convert:
        return scaling * 627.5095 * 4.184 * f
    else:
        return scaling * f


# endregion calculation_methods

# region Arguments


def return_params_parser(add_help=True) -> argparse.ArgumentParser:
    """Returns an argparse.ArgumentParser object for the selection of
    parameters.

    Args:
        add_help (bool, optional): toggle acceptance of -h for descriptive help information to be output. Defaults to True.

    Returns:
        argparse.ArgumentParser: Parser for seminario.py command-line arguments
    """
    if add_help:
        description = __doc__
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter, description=description
        )
    else:
        parser = argparse.ArgumentParser(add_help=False)
    io_group = parser.add_argument_group("io")
    io_group.add_argument(
        "-o",
        "--ff-out",
        type=str,
        metavar="project.seminario.frcmod or project.seminario.fld",
        help=(
            "Use mol2 file and Gaussian Hessian to generate a new force field where\n"
            "each force constant value is replaced\n"
            "by its value estimated from the seminario calculation\n"
            "of force constants in the structure."
        ),
    )
    io_group.add_argument(
        "-i",
        "--ff-in",
        metavar="project.frcmod or project.fld",
        help="Path to input force field file.",
    )
    io_group.add_argument(
        "--mol",
        "-m",
        type=str,
        nargs="+",
        metavar="structure.mol2",
        help="Read these mol2 files, units are in Angstrom.",
    )
    io_group.add_argument(
        "--mmo",
        type=str,
        nargs="+",
        metavar="structure.mmo",
        help="Read these mmo files, units are in Angstrom.",
    )
    io_group.add_argument(
        "--log",
        "-gl",
        type=str,
        nargs="+",
        metavar="gaussian.log",
        help="Gaussian Hessian is extracted from this .log file for seminario calculations. Units are in AU.",
    )
    io_group.add_argument(
        "--jag-in",
        "-ji",
        type=str,
        nargs="+",
        metavar="jaguar.in",
        help="Jaguar Hessian is extracted from this .in file for seminario calculations. Units are in kJ/mol and Angstrom.",
    )
    # io_group.add_argument(
    #     "--fchk",
    #     "-gf",
    #     type=str,
    #     metavar="gaussian.fchk",
    #     default=None,
    #     help="Gaussian Hessian and structure are extracted from this .fchk file for seminario calculations. Units are in Bohr.",
    # ) TODO Support for this is not a priority, but it would be nice to later have the option of just an fchk (+ FF) as input.
    options_group = parser.add_argument_group("options")
    options_group.add_argument(
        "--prep",
        action="store_true",
        default=False,
        help="Flag indicating that the force field should also be prepared with zeroed out bond dipoles/charges and torsional terms (V1, V2, V3).",
    )
    options_group.add_argument(
        "--invert",
        default=False,
        action="store_true",
        help="Flag indicating that the Hessian curvature should be inverted (for TSs). Please read publications for more context.",
    )
    options_group.add_argument(
        "--po",
        action="store_true",
        default=False,
        help="Flag indicating that Per-Ola's matrix multiplications method should be used instead of Seminario/FUERZA.",
    )  # TODO finish implementation for testing out theory once Seminario/FUERZA has been tested fully and used on Rh hydrog enamides
    options_group.add_argument(
        "--individualize",
        action="store_true",
        default=False,
        help="Flag indicating that seminario-estimated FFs should be exported for each structure as opposed to averaging over all structures given.",
    )
    return parser


# endregion Arguments

# region Estimators & Averagers


def estimate_bf_param(
    param: Param,
    structs: List[Structure],
    hessians: List[np.ndarray],
    ang_to_bohr=False,
) -> float:
    """Returns estimated bond force constant for param.

     Estimates the bond force constant of the parameter based on the structures and hessians given,
     matching the structure's bonds to the force field parameter by atom type matching and then
     averaging over the force constants estimated for each bond which matches the force field parameter.

    Args:
        param (Param): bond force constant parameter to estimate
        structs (List[Structure]): structures (in Angstroms) to use as geometric basis for estimation method.
        type_dict (dict): dictionary containing the atom types associated with each DOF in the structure,
        of the form (dof:DOF, atom_types:list)
        hessians (List[np.ndarray]): Hessian matrices, 2nd derivative of potential energy surface of structure
        calculated (kJ/(mol*A^2), Cartesian, must be in order matching the structs)
        at the electronic structure level of theory or above to serve as a quantum-level reference of TS dynamics.
        ang_to_bohr (bool, optional): Whether length/position values need to be converted from Angstrom to Bohr.
        Defaults to False because Hessian units default to kJ/(mol*Angstrom^2) and structures are in Angstrom.

    Returns:
        float: estimated bond force constant
    """
    # It is necessary to average over multiple bonds in the structure which match
    # to the same atom type interactions in the force field.
    match_count = 0
    match_vals = []
    for struct, hessian in zip(structs, hessians):
        for bond in struct.bonds:
            if (
                param.ff_row == bond.ff_row
            ):  # TODO I have no clue why there is a mismatch here, it clearly comes from past IO, but literally WHY??
                # It looks like it's just an artifact of 1-basing Param ff_row and 0-basing bond ff_row when taking it from lines
                # This should be investigated and fixed or explained in full otherwise it will absolutely continue
                # to be a stumbling block when maintaining, improving, learning this code... Creates weird mismatches and I hate it
                match_count += 1
                s_bond = seminario_bond(
                    atoms=struct.get_atoms_in_DOF(bond),
                    hessian=hessian,
                    ang_to_bohr=ang_to_bohr,
                )
                if s_bond < 0 or np.iscomplex(s_bond):
                    logger.log(logging.WARN, f"Invalid estimate of param {param} in structure {struct.origin_name}")

                # The below reverse-massweighting of the force constant is only necessary if the mass-weighted
                # Hessian is used.
                # s_bond = mass_weight_force_constant(s_bond, struct.get_atoms_in_DOF(bond), reverse=True, rm = False)
                logger.log(
                    logging.DEBUG, "Seminario (KJMOLA)" + str(bond) + ": " + str(s_bond)
                )
                #TODO retest PO method p_bond = po_bond(struct.get_atoms_in_DOF(bond), hessian, ang_to_bohr)
                if s_bond > 0 and not np.iscomplex(s_bond): match_vals.append(s_bond) # only includes estimate if it is valid, warnings are triggered upon calculation. bonds where all structures have invalid estimates should simply remain unchanged from input.
    if match_count <= 0:
        logger.log(
            logging.WARN,
            "No bonds in the structures match parameter atom types for parameter: "
            + str(param),
        )
        return None
    else:
        if len(match_vals) == 0:
            logger.log(logging.WARNING, f"WARNING: All structures with interactions matching parameter {param} have invalid estimates so it will not be changed. Please review normal modes.")
            return None

        averaged_fc = sum([float(fc) for fc in match_vals]) / float(len(match_vals))
        return averaged_fc


def estimate_af_param(
    param: Param,
    structs: List[Structure],
    hessians: List[np.ndarray],
    ang_to_bohr=False,
) -> float:
    """Returns estimated angle force constant for param.

     Estimates the angle force constant of the parameter based on the structures and hessians given,
     matching the structure's angles to the force field parameter by atom type matching and then
     averaging over the force constants estimated for each angle which matches the force field parameter.

    Args:
        param (Param): bond force constant parameter to estimate
        structs (List[Structure]): structures (in Angstroms) to use as geometric basis for estimation method.
        hessians (List[np.ndarray]): Hessian matrices, 2nd derivative of potential energy surface of structure
        calculated (kJ/(mol*A^2), Cartesian, must be in order matching the structs)
        at the electronic structure level of theory or above to serve as a quantum-level reference of TS dynamics.
        ang_to_bohr (bool, optional): Whether length/position values need to be converted from Angstrom to Bohr.
        Defaults to False because Hessian units default to kJ/(mol*Angstrom^2) and structures are in Angstrom.

    Returns:
        float: estimated angle force constant
    """
    # It is necessary to average over multiple bonds in the structure which match
    # to the same atom type interactions in the force field.
    match_count = 0
    match_vals = []
    for struct, hessian in zip(structs, hessians):
        for angle in struct.angles:
            if param.ff_row == angle.ff_row:
                match_count += 1
                s_angle = seminario_angle(
                    struct.get_atoms_in_DOF(angle), hessian, ang_to_bohr=ang_to_bohr
                )
                if s_angle < 0 or np.iscomplex(s_angle):
                    logger.log(logging.WARN, f"Invalid estimate of param {param} in structure {struct.origin_name}")
                #TODO retest this method p_bond = po_angle(struct.get_atoms_in_DOF(bond), hessian, ang_to_bohr)
                logger.log(
                    logging.DEBUG,
                    "Seminario (KJMOLA)" + str(angle) + ": " + str(s_angle),
                )
                if s_angle > 0 and not np.iscomplex(s_angle): match_vals.append(s_angle)
    if match_count <= 0:
        logger.log(
            logging.WARN,
            "No angles in the structures match parameter atom types for parameter: "
            + str(param),
        )
        return None
    else:
        if len(match_vals) == 0:
            logger.log(logging.WARNING, f"WARNING: All structures with interactions matching parameter {param} have invalid estimates so it will not be changed. Please review normal modes.")
            return None

        averaged_fc = sum([float(fc) for fc in match_vals]) / match_count
        return averaged_fc


def average_ae_param(param: Param, structs: List[Structure]) -> float:
    """Returns averaged angle for param.

     Averages the equilibrium angle value of the parameter based on the structures given,
     matching the structure's angles to the force field parameter by atom type matching and then
     averaging over the angles measured for each angle which matches the force field parameter.

    Args:
        param (Param): equilibrium angle parameter to estimate
        structs (List[Structure]): structures (in Angstroms) to use in averaging.

    Returns:
        float: average angle equilibrium value (degrees)
    """
    # It is necessary to average over multiple bonds in the structure which match
    # to the same atom type interactions in the force field.
    match_count = 0
    match_vals = []
    for struct in structs:
        for angle in struct.angles:
            if param.ff_row == angle.ff_row:
                match_count += 1
                match_vals.append(angle.value)
    if match_count <= 0:
        logger.log(
            logging.WARN,
            "No angles in the structures match parameter atom types for parameter: "
            + str(param),
        )
        return param.value
    else:
        averaged_angle = sum([float(angl) for angl in match_vals]) / match_count
        return averaged_angle


def average_be_param(param: Param, structs: List[Structure]) -> float:
    """Returns averaged bond length for param.

     Averages the equilibrium bond length value of the parameter based on the structures given,
     matching the structure's bonds to the force field parameter by atom type matching and then
     averaging over the bonds measured for each bond which matches the force field parameter.

    Args:
        param (Param): equilibrium bond parameter to estimate
        structs (List[Structure]): structures (in Angstroms) to use in averaging.

    Returns:
        float: average bond length equilibrium value (Angstrom)
    """
    # It is necessary to average over multiple bonds in the structure which match
    # to the same atom type interactions in the force field.
    match_count = 0
    match_vals = []
    for struct in structs:
        for bond in struct.bonds:
            if param.ff_row == bond.ff_row:
                match_count += 1
                match_vals.append(bond.value)
    if match_count <= 0:
        logger.log(
            logging.WARN,
            "No bonds in the structure match parameter atom types for parameter: "
            + str(param),
        )
        return param.value

    else:
        averaged_bond = sum([float(length) for length in match_vals]) / match_count
        return averaged_bond


# endregion Estimators & Averagers


def seminario(
    force_field: FF,
    structures: List[Structure],
    hessians: List[np.ndarray],
    zero_out: bool,
    hessian_units=co.GAUSSIAN,
) -> FF:
    """Returns a FF with Seminario/FUERZA-estimated force constant and averaged equilibrium structure parameters.

    Args:
        force_field (FF): FF for which to estimate new parameters.
        structures (List[Structure]): structures for which to estimate new FF(s) in Angstrom, degrees
        hessians (np.ndarray): Hessian matrices (Cartesian), units assumed to be in kJ mol**-1 Ang**-2. If otherwise, specify unit
        system with hessian_units constant from constants.py. List must be in same order as structures.
        zero_out (bool): Whether to zero out parameter values for bond dipoles/charges and torsional parameters.
        hessian_units (str): Hessian matrix units, Defaults to Atomic Units (AU) with co.GAUSSIAN.

    Note:
        Hessian is the 2nd derivative of potential energy surface of structure calculated
        at the electronic structure level of theory or above to serve as a quantum-level reference of TS dynamics.

    Returns:
        FF: FF with parameters estimated via Seminario/FUERZA and averaged over structural values.
    """
    estimated_ff = copy.deepcopy(force_field)
    structs = structures

    # TODO verify if below is still an issue
    # In the case of MM3, user may have equivalent atom types such as HX referring to
    # H1, H2, or H3.  Outside of seminario, this is simply handled by MacroModel, but
    # here we need to ensure that any atom types with an equivalent generalized type
    # such as H1 --> HX are changed to their generalized type so that they can be
    # matched to parameters in the MM3 FF correctly without user having to manually
    # change all atom types just for seminario. Could also be useful in future schrod-
    # independent refactoring of Q2MM.
    # if force_field is MM3:
    #     for struct in structs:
    #         struct.generalize_to_ff_atom_types(force_field.atom_type_equivalencies, force_field.atom_types)

    ang_to_bohr = hessian_units == co.GAUSSIAN

    for param in estimated_ff.params:
        if param.ptype == "bf":
            est_bf = estimate_bf_param(
                param=param, structs=structs, hessians=hessians, ang_to_bohr=ang_to_bohr
            )
            param.convert_and_set(est_bf)

        elif param.ptype == "be":
            param.value = average_be_param(param, structs)

        elif param.ptype == "af":
            param.convert_and_set(
                estimate_af_param(param, structs, hessians, ang_to_bohr=ang_to_bohr)
            )

        elif param.ptype == "ae":
            param.value = average_ae_param(param, structs)
        # NOTE: user must make sure mol2 structure is the same as gaussian log or fchk structure (just in IRC)

        elif zero_out and (param.ptype == "df" or param.ptype == "q"):
            param.value = 0.0

    return estimated_ff


# region Stand-alone Seminario/FUERZA methods process


def main(args):
    if isinstance(args, str):
        args = args.split()
    parser = return_params_parser()
    args = parser.parse_args()

    assert (
        args.ff_in
    ), "Input force field file is required! Please also verify compatibility/sufficiency of parameters."

    assert (args.mol or args.mmo) and (
        args.log or args.jag_in
    ), "Both a mol2 structure file and a Gaussian log or Jaguar in (DFT Cartesian Hessian) file are needed!"

    if args.ff_in[-4:] == ".fld":
        ff_in = MM3(args.ff_in)
        ff_in.import_ff()
        logger.log(logging.INFO, f"mm3 ff imported: {ff_in.path}")
    elif args.ff_in[-7:] == ".frcmod":
        ff_in = AmberFF(args.ff_in)
        ff_in.import_ff()
        logger.log(logging.INFO, f"amber ff imported: {ff_in.path}")
    else:
        raise NotImplementedError()

    if args.mol and "*" in args.mol[0]:
        args.mol = sorted(glob.glob(args.mol[0]))

    if args.mmo and "*" in args.mmo[0]:
        args.mmo = sorted(glob.glob(args.mmo[0]))

    structs: List[Structure] = []

    if args.mol:
        struct_files: List[Mol2] = [Mol2(mol_path) for mol_path in args.mol]
        logger.log(
            logging.INFO,
            f"{len(struct_files)}/{len(args.mol)} Mol2 structure files imported.",
        )
    elif args.mmo:
        struct_files: List[MacroModel] = [MacroModel(mmo_path) for mmo_path in args.mmo]
        logger.log(
            logging.INFO,
            f"{len(struct_files)}/{len(args.mmo)} MMO structure files imported.",
        )

    for struct in struct_files:
        structs.extend(struct.structures)
    logger.log(
        logging.INFO, f"{len(structs)} Structures imported from mol2 files."
    )

    if args.log:
        if "*" in args.log[0]:
            args.log = sorted(glob.glob(args.log[0]))
        assert len(structs) == len(
            args.log
        ), "Gaussian log input must have 1:1 correspondence with the mol2 structures."
    elif args.jag_in:
        if "*" in args.jag_in[0]:
            args.jag_in = sorted(glob.glob(args.jag_in[0]))
        assert len(structs) == len(
            args.jag_in
        ), "Jaguar in input must have 1:1 correspondence with the mol2 structures."

    if (
        args.log
    ):  # Hess is converted from au to kJ/molA, archive and fchk H not mass-weighted.
        # NOTE: If Hessian gets mass-weighted, then the resulting force constants must be un-massweighted.
        # The code for this is left in but commented out for now in case a reason arrives to mass-weight.
        logs: List[GaussLog] = [GaussLog(log, au_hessian=False) for log in args.log]
        logger.log(
            logging.INFO,
            f"{len(logs)}/{len(args.log)} Gaussian log files imported.",
        )
        hessians: List[np.ndarray] = []
        for log in logs:
            mw_hessian = copy.deepcopy(log.structures[-1].hess)
            # evals = log.evals
            # evecs = log.evecs
            # mw_hessian = reform_hessian(evals, evecs)
            # mass_weight_hessian(mw_hessian, log.structures[-1].atoms)
            if args.invert:
                mw_hessian = invert_ts_curvature(mw_hessian)
                logger.log(
                    logging.INFO, f"Inverted Hessian from {log.filename}..."
                )
            hessians.append(copy.deepcopy(mw_hessian))
        logger.log(
            logging.INFO,
            f"{len(hessians)}/{len(args.log)} Hessians imported.",
        )
    elif args.jag_in:  # Hess should be in kJ/molA, not mass-weighted.
        # NOTE: If Hessian gets mass-weighted, then the resulting force constants must be un-massweighted.
        # The code for this is left in but commented out for now in case a reason arrives to mass-weight.
        logs: List[JaguarIn] = [JaguarIn(log) for log in args.jag_in]
        logger.log(
            logging.INFO,
            f"{len(logs)}/{len(args.jag_in)} Jaguar in files imported.",
        )
        hessians: List[np.ndarray] = []
        for i in range(len(logs)):
            log = logs[i]
            hessian = log.get_hessian(structs[i].num_atoms)
            # mass_weight_hessian(mw_hessian, log.structures[-1].atoms)
            if args.invert:
                hessian = invert_ts_curvature(hessian)
                logger.log(
                    logging.INFO, f"Inverted Hessian from {log.filename}..."
                )
            hessians.append(copy.deepcopy(hessian))
        logger.log(
            logging.INFO,
            f"{len(hessians)}/{len(args.jag_in)} Hessians imported.",
        )

    assert len(structs) == len(
        hessians
    ), "Hessians and Structures must match 1:1 in corresponding order. Uneven numbers imported, likely due to multiple entries in one file but not its corresponding file."

    if args.individualize:
        for i in range(len(structs)):
            estimated_ff = seminario(
                ff_in, [structs[i]], [hessians[i]], args.prep, hessian_units=co.KJMOLA
            )

            # Write out new FF
            estimated_ff.export_ff(
                structs[i].origin_name + "." + args.ff_out, estimated_ff.params
            )
    else:
        estimated_ff = seminario(
            ff_in, structs, hessians, args.prep, hessian_units=co.KJMOLA
        )

        # Write out new FF
        estimated_ff.export_ff(args.ff_out, estimated_ff.params)


# endregion

if __name__ == "__main__":
    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])
