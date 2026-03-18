"""True end-to-end Seminario parity tests on the Rh-enamide training set."""
from __future__ import annotations

from pathlib import Path

import pytest

from q2mm import constants as co
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.seminario import estimate_force_constants
from q2mm.schrod_indep_filetypes import JaguarIn, MM3, MacroModel
from q2mm.seminario import (
    average_ae_param,
    average_be_param,
    estimate_af_param,
    estimate_bf_param,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RH_DIR = REPO_ROOT / "examples" / "rh-enamide"
TRAINING_SET_DIR = RH_DIR / "rh_enamide_training_set"
MM3_PATH = RH_DIR / "mm3.fld"
MMO_PATH = TRAINING_SET_DIR / "rh_enamide_training_set.mmo"
JAG_DIR = TRAINING_SET_DIR / "jaguar_spe_freq_in_out"


def _legacy_bond_force_constant(value: float) -> float:
    return value * co.AU_TO_MDYNA


def _legacy_angle_force_constant(value: float) -> float:
    return value * co.AU_TO_MDYN_ANGLE


pytestmark = pytest.mark.skipif(
    not (MM3_PATH.exists() and MMO_PATH.exists() and JAG_DIR.exists()),
    reason="Rh-enamide Seminario fixtures not found",
)


@pytest.fixture(scope="module")
def rh_enamide_parity_results():
    legacy_ff = MM3(str(MM3_PATH))
    legacy_ff.import_ff()

    structures = MacroModel(str(MMO_PATH)).structures
    hessian_files = sorted(JAG_DIR.glob("*.in"))
    assert len(structures) == len(hessian_files)

    hessians = [
        JaguarIn(str(path)).get_hessian(len(structure.atoms))
        for structure, path in zip(structures, hessian_files)
    ]

    legacy_bf = {}
    legacy_be = {}
    legacy_af = {}
    legacy_ae = {}
    for param in legacy_ff.params:
        if param.ptype == "bf":
            legacy_bf[param.ff_row] = estimate_bf_param(
                param,
                structures,
                hessians,
                ang_to_bohr=True,
            )
        elif param.ptype == "be":
            legacy_be[param.ff_row] = average_be_param(param, structures)
        elif param.ptype == "af":
            legacy_af[param.ff_row] = estimate_af_param(
                param,
                structures,
                hessians,
                ang_to_bohr=True,
            )
        elif param.ptype == "ae":
            legacy_ae[param.ff_row] = average_ae_param(param, structures)

    molecules = [
        Q2MMMolecule.from_structure(
            structure,
            hessian=hessian,
            name=f"rh_enamide_{index + 1}",
        )
        for index, (structure, hessian) in enumerate(zip(structures, hessians))
    ]
    clean_start = ForceField.from_mm3_fld(MM3_PATH)
    clean_estimated = estimate_force_constants(
        molecules,
        forcefield=clean_start,
        zero_torsions=True,
        au_hessian=True,
        invalid_policy="skip",
    )

    return {
        "clean_start": clean_start,
        "clean_estimated": clean_estimated,
        "legacy_bf": legacy_bf,
        "legacy_be": legacy_be,
        "legacy_af": legacy_af,
        "legacy_ae": legacy_ae,
    }


def test_from_structure_preserves_legacy_dof_metadata():
    structures = MacroModel(str(MMO_PATH)).structures
    molecule = Q2MMMolecule.from_structure(structures[0], name="rh_enamide_1")

    assert len(molecule.bonds) == len(structures[0].bonds)
    assert len(molecule.angles) == len(structures[0].angles)
    assert molecule.bonds[0].ff_row == structures[0].bonds[0].ff_row
    assert molecule.angles[0].ff_row == structures[0].angles[0].ff_row


def test_bond_params_match_legacy_pipeline(rh_enamide_parity_results):
    clean_start = rh_enamide_parity_results["clean_start"]
    clean_estimated = rh_enamide_parity_results["clean_estimated"]
    legacy_bf = rh_enamide_parity_results["legacy_bf"]
    legacy_be = rh_enamide_parity_results["legacy_be"]
    starting_bonds = {param.ff_row: param for param in clean_start.bonds}

    assert len(clean_estimated.bonds) == len(legacy_bf)
    for bond_param in clean_estimated.bonds:
        assert bond_param.ff_row is not None
        assert bond_param.ff_row in legacy_bf
        assert bond_param.ff_row in legacy_be

        legacy_force_constant = legacy_bf[bond_param.ff_row]
        if legacy_force_constant is None:
            assert bond_param.force_constant == pytest.approx(
                starting_bonds[bond_param.ff_row].force_constant,
                abs=1e-8,
            )
        else:
            assert bond_param.force_constant == pytest.approx(
                _legacy_bond_force_constant(legacy_force_constant),
                abs=1e-8,
            )

        assert bond_param.equilibrium == pytest.approx(
            legacy_be[bond_param.ff_row],
            abs=1e-8,
        )


def test_angle_params_match_legacy_pipeline(rh_enamide_parity_results):
    clean_start = rh_enamide_parity_results["clean_start"]
    clean_estimated = rh_enamide_parity_results["clean_estimated"]
    legacy_af = rh_enamide_parity_results["legacy_af"]
    legacy_ae = rh_enamide_parity_results["legacy_ae"]
    starting_angles = {param.ff_row: param for param in clean_start.angles}

    assert len(clean_estimated.angles) == len(legacy_af)
    for angle_param in clean_estimated.angles:
        assert angle_param.ff_row is not None
        assert angle_param.ff_row in legacy_af
        assert angle_param.ff_row in legacy_ae

        legacy_force_constant = legacy_af[angle_param.ff_row]
        if legacy_force_constant is None:
            assert angle_param.force_constant == pytest.approx(
                starting_angles[angle_param.ff_row].force_constant,
                abs=1e-8,
            )
        else:
            assert angle_param.force_constant == pytest.approx(
                _legacy_angle_force_constant(legacy_force_constant),
                abs=1e-8,
            )

        assert angle_param.equilibrium == pytest.approx(
            legacy_ae[angle_param.ff_row],
            abs=1e-8,
        )
