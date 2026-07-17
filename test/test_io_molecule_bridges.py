from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from q2mm.io import (
    load_fchk_molecule,
    load_gaussian_molecules,
    load_jaguar_molecules,
    load_macromodel_molecules,
)
from q2mm.io.fchk import load_fchk
from q2mm.io.macromodel import MacroModel
from q2mm.models.hessian import HessianUnits
from test._shared import GS_FCHK, REPO_ROOT

_GAUSSIAN = REPO_ROOT / "test" / "fixtures" / "seminario_parity" / "cisplatin_opt_freq_m06.log"
_RH_ROOT = REPO_ROOT / "examples" / "rh-enamide" / "rh_enamide_training_set"
_JAGUAR_ROOT = _RH_ROOT / "jaguar_spe_freq_in_out"
_JAGUAR_IN = _JAGUAR_ROOT / "1ZDMPfromJCTCSI_loner1.01.in"
_JAGUAR_OUT = _JAGUAR_ROOT / "1ZDMPfromJCTCSI_loner1.out"
_MACROMODEL = _RH_ROOT / "rh_enamide_training_set.mmo"


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def test_fchk_bridge_preserves_science_and_records_path_free_hash() -> None:
    molecule = load_fchk_molecule(GS_FCHK, bond_tolerance=1.4)
    direct = load_fchk(GS_FCHK, bond_tolerance=1.4)

    assert molecule.symbols == direct.symbols
    np.testing.assert_array_equal(molecule.geometry, direct.geometry)
    assert molecule.charge == 0
    assert molecule.multiplicity == 1
    assert molecule.bond_tolerance == 1.4
    assert molecule.hessian is not None
    assert molecule.hessian_provenance is not None
    assert molecule.hessian_provenance.units is HessianUnits.ATOMIC
    assert molecule.hessian_provenance.path == GS_FCHK.name
    details = molecule.hessian_provenance.source_details
    assert details["parser"] == "q2mm.io.fchk.load_fchk"
    assert details["file_content_sha256"] == _sha256(GS_FCHK)
    assert details["source_units"] == HessianUnits.ATOMIC.value
    with pytest.raises(TypeError, match="immutable"):
        details["parser"] = "changed"


def test_gaussian_bridge_requires_explicit_index_and_canonical_hessian() -> None:
    molecule = load_gaussian_molecules([_GAUSSIAN], structure_index=-1)[0]

    assert molecule.n_atoms == 11
    assert molecule.hessian is not None
    assert molecule.hessian.shape == (33, 33)
    assert not molecule.hessian.flags.writeable
    assert molecule.hessian_provenance is not None
    assert molecule.hessian_provenance.path == _GAUSSIAN.name
    assert molecule.hessian_provenance.source_details["file_content_sha256"] == _sha256(_GAUSSIAN)
    earlier = load_gaussian_molecules([_GAUSSIAN], structure_index=0, require_hessian=False)[0]
    assert earlier.hessian is None
    np.testing.assert_array_equal(earlier.geometry, molecule.geometry)
    with pytest.raises(IndexError, match="out of range"):
        load_gaussian_molecules([_GAUSSIAN], structure_index=2)


def test_gaussian_batch_is_all_or_none(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.log"
    invalid.write_text("not a Gaussian archive", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not parse Gaussian"):
        load_gaussian_molecules([_GAUSSIAN, invalid], structure_index=-1)


def test_jaguar_input_bridge_preserves_charge_order_and_hessian_provenance() -> None:
    molecule = load_jaguar_molecules([_JAGUAR_IN], structure_index=0)[0]

    assert molecule.symbols[:4] == ("Rh", "O", "C", "N")
    assert molecule.charge == 1
    assert molecule.multiplicity == 1
    assert molecule.hessian is not None
    assert molecule.hessian.shape == (3 * molecule.n_atoms, 3 * molecule.n_atoms)
    assert molecule.hessian_provenance is not None
    assert molecule.hessian_provenance.path == _JAGUAR_IN.name
    assert molecule.hessian_provenance.source_details["parser"] == "q2mm.io.jaguar.JaguarIn"
    assert molecule.hessian_provenance.source_details["file_content_sha256"] == _sha256(_JAGUAR_IN)


def test_jaguar_output_requires_explicit_no_hessian_choice() -> None:
    with pytest.raises(ValueError, match="does not expose a full Cartesian Hessian"):
        load_jaguar_molecules([_JAGUAR_OUT], structure_index=0)

    molecule = load_jaguar_molecules([_JAGUAR_OUT], structure_index=0, require_hessian=False)[0]
    assert molecule.symbols[:4] == ("Rh", "O", "C", "N")
    assert molecule.charge == 1
    assert molecule.multiplicity == 1
    assert molecule.hessian is None


def test_jaguar_input_missing_hessian_is_not_fabricated(tmp_path: Path) -> None:
    path = tmp_path / "no-hessian.in"
    path.write_text(
        "&gen\nmolchg=0\nmultip=1\n&\n&zmat\nH1 0.0 0.0 0.0\nH2 0.0 0.0 0.74\n&\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="has no complete Hessian"):
        load_jaguar_molecules([path], structure_index=0)

    molecule = load_jaguar_molecules([path], structure_index=0, require_hessian=False)[0]
    assert molecule.symbols == ("H", "H")
    assert molecule.hessian is None
    assert molecule.hessian_provenance is None


def test_macromodel_bridge_preserves_authoritative_topology_and_atom_types() -> None:
    direct = MacroModel(str(_MACROMODEL)).molecules[0]
    molecule = load_macromodel_molecules([_MACROMODEL], structure_index=0, bond_tolerance=1.8)[0]

    assert molecule.symbols == direct.symbols
    assert molecule.atom_types == direct.atom_types
    np.testing.assert_array_equal(molecule.geometry, direct.geometry)
    assert molecule.bonds == direct.bonds
    assert molecule.angles == direct.angles
    assert molecule.torsions == direct.torsions
    assert molecule.bonds_explicit
    assert molecule.angles_explicit
    assert molecule.torsions_explicit
    assert molecule.bond_tolerance == 1.8


def test_macromodel_batch_is_all_or_none(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.mmo"
    invalid.write_text("not a MacroModel structure", encoding="utf-8")

    with pytest.raises(ValueError, match="No structures"):
        load_macromodel_molecules([_MACROMODEL, invalid], structure_index=0)
