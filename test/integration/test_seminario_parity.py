"""Fixture-backed end-to-end Seminario parity tests.

Covers issue #74: validates that the refactored code reproduces the
upstream Seminario results for both the rh-enamide and SN2 systems,
plus runtime benchmarks.

Force-constant tolerances use rel=1e-6 (not abs=1e-8) because the
refactored code derives HESSIAN_CONVERSION from base CODATA 2018
constants instead of the legacy hardcoded value.  The difference is
~5e-9 relative in the constant itself, which amplifies to ~1e-7
relative through Seminario eigenvalue decomposition — well below
any physical significance.
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import pytest

from test._shared import REPO_ROOT, SN2_QM_REF, SN2_XYZ, SN2_HESSIAN

from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.seminario import estimate_force_constants, seminario_bond_fc
from q2mm.models.units import MDYNA_TO_KCALMOLA2, MDYNA_RAD2_TO_KCALMOLRAD2
from q2mm.parsers import JaguarIn, MacroModel

FIXTURE_DIR = REPO_ROOT / "test" / "fixtures" / "seminario_parity"

RH_FIXTURE_PATH = FIXTURE_DIR / "rh_enamide_reference.json"
SN2_FIXTURE_PATH = FIXTURE_DIR / "sn2_reference.json"

RH_DIR = REPO_ROOT / "examples" / "rh-enamide"
TRAINING_SET_DIR = RH_DIR / "rh_enamide_training_set"
MM3_PATH = RH_DIR / "mm3.fld"
MMO_PATH = TRAINING_SET_DIR / "rh_enamide_training_set.mmo"
JAG_DIR = TRAINING_SET_DIR / "jaguar_spe_freq_in_out"

SN2_XYZ_PATH = SN2_XYZ
SN2_HESSIAN_PATH = SN2_HESSIAN


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _natural_sort_key(path: Path):
    """Sort key that handles numeric components correctly (e.g. 2.in < 10.in)."""
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


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
    hessian_files = sorted(JAG_DIR.glob("*.in"), key=_natural_sort_key)
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
                rel=1e-6,
            )
        else:
            # Fixture stores mdyn/Å; ForceField stores canonical kcal/(mol·Å²)
            assert bond_param.force_constant == pytest.approx(
                fixture_force_constant * MDYNA_TO_KCALMOLA2,
                rel=1e-6,
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
                rel=1e-6,
            )
        else:
            # Fixture stores mdyn·Å/rad²; ForceField stores canonical kcal/(mol·rad²)
            assert angle_param.force_constant == pytest.approx(
                fixture_force_constant * MDYNA_RAD2_TO_KCALMOLRAD2,
                rel=1e-6,
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
        # seminario_bond_fc returns canonical kcal/(mol·Å²);
        # fixture stores legacy mdyn/Å values
        assert actual == pytest.approx(
            bond["legacy_force_constant_mdyn_a"] * MDYNA_TO_KCALMOLA2,
            rel=1e-6,
        )


# ---------------------------------------------------------------------------
# Rh-enamide full pipeline stability tests (#74)
# ---------------------------------------------------------------------------
_RH_DATA_AVAILABLE = MM3_PATH.exists() and MMO_PATH.exists() and JAG_DIR.exists() and RH_FIXTURE_PATH.exists()


@pytest.mark.skipif(not _RH_DATA_AVAILABLE, reason="Rh-enamide data not found")
def test_rh_enamide_forcefield_roundtrip():
    """Loading, estimating, and re-loading FF gives consistent params."""
    structures = MacroModel(str(MMO_PATH)).structures
    hessian_files = sorted(JAG_DIR.glob("*.in"), key=_natural_sort_key)
    molecules = [
        Q2MMMolecule.from_structure(
            s,
            hessian=JaguarIn(str(h)).get_hessian(len(s.atoms)),
            name=f"rh_{i}",
        )
        for i, (s, h) in enumerate(zip(structures, hessian_files))
    ]

    ff1 = ForceField.from_mm3_fld(MM3_PATH)
    est1 = estimate_force_constants(molecules, forcefield=ff1, invalid_policy="skip")

    # Re-estimate from the same starting point — must be deterministic
    ff2 = ForceField.from_mm3_fld(MM3_PATH)
    est2 = estimate_force_constants(molecules, forcefield=ff2, invalid_policy="skip")

    for b1, b2 in zip(est1.bonds, est2.bonds):
        assert b1.force_constant == pytest.approx(b2.force_constant, abs=1e-12)
        assert b1.equilibrium == pytest.approx(b2.equilibrium, abs=1e-12)
    for a1, a2 in zip(est1.angles, est2.angles):
        assert a1.force_constant == pytest.approx(a2.force_constant, abs=1e-12)
        assert a1.equilibrium == pytest.approx(a2.equilibrium, abs=1e-12)


@pytest.mark.skipif(not _RH_DATA_AVAILABLE, reason="Rh-enamide data not found")
def test_rh_enamide_param_vector_parity(rh_enamide_clean_results, rh_enamide_fixture):
    """Parameter vector matches fixture values for all bond and angle params."""
    estimated = rh_enamide_clean_results["clean_estimated"]
    fixture_bf = _int_keyed_map(rh_enamide_fixture["parameters"]["bond_force_constants_mdyn_a"])
    fixture_be = _int_keyed_map(rh_enamide_fixture["parameters"]["bond_equilibria_angstrom"])
    fixture_af = _int_keyed_map(rh_enamide_fixture["parameters"]["angle_force_constants_mdyn_a_rad2"])
    fixture_ae = _int_keyed_map(rh_enamide_fixture["parameters"]["angle_equilibria_degrees"])

    # Collect max deviations for reporting
    max_bond_k_diff = 0.0
    max_bond_eq_diff = 0.0
    max_angle_k_diff = 0.0
    max_angle_eq_diff = 0.0

    starting = rh_enamide_clean_results["clean_start"]
    starting_bonds = {p.ff_row: p for p in starting.bonds}
    starting_angles = {p.ff_row: p for p in starting.angles}

    for b in estimated.bonds:
        expected_k = fixture_bf.get(b.ff_row)
        if expected_k is None:
            expected_k = starting_bonds[b.ff_row].force_constant
        else:
            expected_k *= MDYNA_TO_KCALMOLA2  # fixture mdyn/Å → canonical
        max_bond_k_diff = max(max_bond_k_diff, abs(b.force_constant - expected_k))
        max_bond_eq_diff = max(max_bond_eq_diff, abs(b.equilibrium - fixture_be[b.ff_row]))

    for a in estimated.angles:
        expected_k = fixture_af.get(a.ff_row)
        if expected_k is None:
            expected_k = starting_angles[a.ff_row].force_constant
        else:
            expected_k *= MDYNA_RAD2_TO_KCALMOLRAD2  # fixture mdyn·Å/rad² → canonical
        max_angle_k_diff = max(max_angle_k_diff, abs(a.force_constant - expected_k))
        max_angle_eq_diff = max(max_angle_eq_diff, abs(a.equilibrium - fixture_ae[a.ff_row]))

    bond_rel = max_bond_k_diff / max(abs(b.force_constant) for b in estimated.bonds)
    assert bond_rel < 1e-6, f"Bond FC max diff: abs={max_bond_k_diff}, rel={bond_rel}"
    assert max_bond_eq_diff < 1e-8, f"Bond eq max diff: {max_bond_eq_diff}"
    angle_rel = max_angle_k_diff / max(abs(a.force_constant) for a in estimated.angles)
    assert angle_rel < 1e-6, f"Angle FC max diff: abs={max_angle_k_diff}, rel={angle_rel}"
    assert max_angle_eq_diff < 1e-8, f"Angle eq max diff: {max_angle_eq_diff}"


# ---------------------------------------------------------------------------
# Runtime benchmarks (informational, never fail)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _RH_DATA_AVAILABLE, reason="Rh-enamide data not found")
@pytest.mark.slow
def test_rh_enamide_seminario_benchmark(rh_enamide_clean_results, capsys):
    """Benchmark: time the full rh-enamide Seminario pipeline (informational)."""
    structures = MacroModel(str(MMO_PATH)).structures
    hessian_files = sorted(JAG_DIR.glob("*.in"), key=_natural_sort_key)

    # Time parsing
    t0 = time.perf_counter()
    hessians = [JaguarIn(str(p)).get_hessian(len(s.atoms)) for s, p in zip(structures, hessian_files)]
    t_parse = time.perf_counter() - t0

    # Time molecule creation
    t0 = time.perf_counter()
    molecules = [
        Q2MMMolecule.from_structure(s, hessian=h, name=f"rh_{i}") for i, (s, h) in enumerate(zip(structures, hessians))
    ]
    t_mol = time.perf_counter() - t0

    # Time Seminario estimation (10 iterations for stable timing)
    ff_template = ForceField.from_mm3_fld(MM3_PATH)
    times = []
    for _ in range(10):
        ff = ff_template.copy()
        t0 = time.perf_counter()
        estimate_force_constants(molecules, forcefield=ff, invalid_policy="skip")
        times.append(time.perf_counter() - t0)

    t_est_mean = np.mean(times)
    t_est_std = np.std(times)

    with capsys.disabled():
        print(f"\n{'=' * 60}")
        print(f"Rh-enamide Seminario benchmark ({len(structures)} structures)")
        print(f"{'=' * 60}")
        print(f"  Jaguar parsing:     {t_parse:.3f}s")
        print(f"  Molecule creation:  {t_mol:.3f}s")
        print(f"  Seminario estimate: {t_est_mean:.4f}s ± {t_est_std:.4f}s (10 runs)")
        print(f"  Total (single run): {t_parse + t_mol + t_est_mean:.3f}s")
        print(f"{'=' * 60}")
