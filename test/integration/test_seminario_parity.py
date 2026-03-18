"""Fixture-backed end-to-end Seminario parity tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.seminario import estimate_force_constants, seminario_bond_fc
from q2mm.schrod_indep_filetypes import JaguarIn, MacroModel

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "test" / "fixtures" / "seminario_parity"

RH_FIXTURE_PATH = FIXTURE_DIR / "rh_enamide_reference.json"
SN2_FIXTURE_PATH = FIXTURE_DIR / "sn2_reference.json"

RH_DIR = REPO_ROOT / "examples" / "rh-enamide"
TRAINING_SET_DIR = RH_DIR / "rh_enamide_training_set"
MM3_PATH = RH_DIR / "mm3.fld"
MMO_PATH = TRAINING_SET_DIR / "rh_enamide_training_set.mmo"
JAG_DIR = TRAINING_SET_DIR / "jaguar_spe_freq_in_out"

SN2_QM_REF = REPO_ROOT / "examples" / "sn2-test" / "qm-reference"
SN2_XYZ_PATH = SN2_QM_REF / "sn2-ts-optimized.xyz"
SN2_HESSIAN_PATH = SN2_QM_REF / "sn2-ts-hessian.npy"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _int_keyed_map(values: dict[str, float | None]) -> dict[int, float | None]:
    return {int(key): value for key, value in values.items()}


@pytest.fixture(scope="module")
def rh_enamide_fixture():
    return _load_json(RH_FIXTURE_PATH)


@pytest.fixture(scope="module")
def sn2_fixture():
    return _load_json(SN2_FIXTURE_PATH)


@pytest.fixture(scope="module")
def rh_enamide_clean_results():
    structures = MacroModel(str(MMO_PATH)).structures
    hessian_files = sorted(JAG_DIR.glob("*.in"))
    assert len(structures) == len(hessian_files)

    hessians = [
        JaguarIn(str(path)).get_hessian(len(structure.atoms)) for structure, path in zip(structures, hessian_files)
    ]

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
    }


@pytest.mark.skipif(
    not (MM3_PATH.exists() and MMO_PATH.exists() and JAG_DIR.exists() and RH_FIXTURE_PATH.exists()),
    reason="Rh-enamide parity data or fixtures not found",
)
def test_from_structure_preserves_legacy_dof_metadata():
    structures = MacroModel(str(MMO_PATH)).structures
    molecule = Q2MMMolecule.from_structure(structures[0], name="rh_enamide_1")

    assert len(molecule.bonds) == len(structures[0].bonds)
    assert len(molecule.angles) == len(structures[0].angles)
    assert molecule.bonds[0].ff_row == structures[0].bonds[0].ff_row
    assert molecule.angles[0].ff_row == structures[0].angles[0].ff_row


@pytest.mark.skipif(
    not (MM3_PATH.exists() and MMO_PATH.exists() and JAG_DIR.exists() and RH_FIXTURE_PATH.exists()),
    reason="Rh-enamide parity data or fixtures not found",
)
def test_bond_params_match_fixture(rh_enamide_clean_results, rh_enamide_fixture):
    clean_start = rh_enamide_clean_results["clean_start"]
    clean_estimated = rh_enamide_clean_results["clean_estimated"]
    fixture_bf = _int_keyed_map(rh_enamide_fixture["parameters"]["bond_force_constants_mdyn_a"])
    fixture_be = _int_keyed_map(rh_enamide_fixture["parameters"]["bond_equilibria_angstrom"])
    starting_bonds = {param.ff_row: param for param in clean_start.bonds}

    assert len(clean_estimated.bonds) == len(fixture_bf)
    for bond_param in clean_estimated.bonds:
        assert bond_param.ff_row is not None
        assert bond_param.ff_row in fixture_bf
        assert bond_param.ff_row in fixture_be

        fixture_force_constant = fixture_bf[bond_param.ff_row]
        if fixture_force_constant is None:
            assert bond_param.force_constant == pytest.approx(
                starting_bonds[bond_param.ff_row].force_constant,
                abs=1e-8,
            )
        else:
            assert bond_param.force_constant == pytest.approx(
                fixture_force_constant,
                abs=1e-8,
            )

        assert bond_param.equilibrium == pytest.approx(
            fixture_be[bond_param.ff_row],
            abs=1e-8,
        )


@pytest.mark.skipif(
    not (MM3_PATH.exists() and MMO_PATH.exists() and JAG_DIR.exists() and RH_FIXTURE_PATH.exists()),
    reason="Rh-enamide parity data or fixtures not found",
)
def test_angle_params_match_fixture(rh_enamide_clean_results, rh_enamide_fixture):
    clean_start = rh_enamide_clean_results["clean_start"]
    clean_estimated = rh_enamide_clean_results["clean_estimated"]
    fixture_af = _int_keyed_map(rh_enamide_fixture["parameters"]["angle_force_constants_mdyn_a_rad2"])
    fixture_ae = _int_keyed_map(rh_enamide_fixture["parameters"]["angle_equilibria_degrees"])
    starting_angles = {param.ff_row: param for param in clean_start.angles}

    assert len(clean_estimated.angles) == len(fixture_af)
    for angle_param in clean_estimated.angles:
        assert angle_param.ff_row is not None
        assert angle_param.ff_row in fixture_af
        assert angle_param.ff_row in fixture_ae

        fixture_force_constant = fixture_af[angle_param.ff_row]
        if fixture_force_constant is None:
            assert angle_param.force_constant == pytest.approx(
                starting_angles[angle_param.ff_row].force_constant,
                abs=1e-8,
            )
        else:
            assert angle_param.force_constant == pytest.approx(
                fixture_force_constant,
                abs=1e-8,
            )

        assert angle_param.equilibrium == pytest.approx(
            fixture_ae[angle_param.ff_row],
            abs=1e-8,
        )


@pytest.mark.skipif(
    not (SN2_FIXTURE_PATH.exists() and SN2_XYZ_PATH.exists() and SN2_HESSIAN_PATH.exists()),
    reason="SN2 parity fixtures not found",
)
def test_sn2_bond_projections_match_fixture(sn2_fixture):
    molecule = Q2MMMolecule.from_xyz(SN2_XYZ_PATH, name="sn2_ts", bond_tolerance=1.5)
    hessian = np.load(str(SN2_HESSIAN_PATH))
    scaling = float(sn2_fixture["metadata"]["dft_scaling"])

    for bond in sn2_fixture["bonds"]:
        actual = seminario_bond_fc(
            bond["atom_i"],
            bond["atom_j"],
            molecule.geometry,
            hessian,
            au_units=True,
            dft_scaling=scaling,
        )
        assert actual == pytest.approx(
            bond["legacy_force_constant_mdyn_a"],
            abs=1e-8,
        )
