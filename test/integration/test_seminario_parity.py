"""Fixture-backed end-to-end QFUERZA estimation parity tests.

Covers issue #74: validates that the refactored code reproduces the
expected Seminario/QFUERZA results for both the rh-enamide and SN2
systems, plus runtime benchmarks.

Also includes Zenodo-validated tests (cisplatin) that verify QFUERZA
rules against the paper authors' own force field files:
  Farrugia et al., J. Chem. Theory Comput. 2025, 22, 469-476
  Zenodo DOI: 10.5281/zenodo.17386006

Force-constant tolerances use rel=1e-6 (not abs=1e-8) because the
refactored code derives HESSIAN_AU_TO_KJMOLA2 from base CODATA 2018
constants instead of the legacy hardcoded value.  The difference is
~5e-9 relative in the constant itself, which amplifies to ~1e-7
relative through Seminario eigenvalue decomposition — well below
any physical significance.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from test._shared import REPO_ROOT, SN2_XYZ, SN2_HESSIAN

from q2mm.models.forcefield import ForceField, FunctionalForm
from q2mm.models.molecule import Molecule
from q2mm.models.seminario import (
    qfuerza_fresh,
    qfuerza_into,
    seminario_bond_fc,
    QFUERZA_H_ANGLE_DEFAULT_MDYNA,
    _is_hydrogen_angle,
)
from q2mm.models.units import (
    MDYNA_TO_KCALMOLA2,
    MDYNA_RAD2_TO_KCALMOLRAD2,
    KCALMOLRAD2_TO_MDYNA_RAD2,
)
from q2mm.io import JaguarIn, MacroModel, load_mm3_fld
from q2mm.io.gaussian import GaussLog
from q2mm.io.xyz import load_xyz

FIXTURE_DIR = REPO_ROOT / "test" / "fixtures" / "seminario_parity"

CISPLATIN_ZENODO_PATH = FIXTURE_DIR / "cisplatin_zenodo_reference.json"
CISPLATIN_GAUSSIAN_LOG = FIXTURE_DIR / "cisplatin_opt_freq_m06.log"

RH_FIXTURE_PATH = FIXTURE_DIR / "rh_enamide_reference.json"
SN2_FIXTURE_PATH = FIXTURE_DIR / "sn2_reference.json"

RH_DIR = REPO_ROOT / "examples" / "rh-enamide"
TRAINING_SET_DIR = RH_DIR / "rh_enamide_training_set"
MM3_PATH = RH_DIR / "mm3.fld"
MMO_PATH = TRAINING_SET_DIR / "rh_enamide_training_set.mmo"
JAG_DIR = TRAINING_SET_DIR / "jaguar_spe_freq_in_out"

SN2_XYZ_PATH = SN2_XYZ
SN2_HESSIAN_PATH = SN2_HESSIAN

# All fixtures referenced by this module are tracked in-repo
# (see AGENTS.md §2 rule 5).  Fail loudly at collection time if any
# are missing — that means the working copy is corrupt, not that the
# test should silently skip.
_REQUIRED_PARITY_FIXTURES: dict[str, Path] = {
    "CISPLATIN_ZENODO_PATH": CISPLATIN_ZENODO_PATH,
    "CISPLATIN_GAUSSIAN_LOG": CISPLATIN_GAUSSIAN_LOG,
    "RH_FIXTURE_PATH": RH_FIXTURE_PATH,
    "SN2_FIXTURE_PATH": SN2_FIXTURE_PATH,
    "MM3_PATH": MM3_PATH,
    "MMO_PATH": MMO_PATH,
    "JAG_DIR": JAG_DIR,
}
_missing_parity = sorted(name for name, p in _REQUIRED_PARITY_FIXTURES.items() if not p.exists())
if _missing_parity:
    raise RuntimeError(
        "Missing in-repo fixtures for test_seminario_parity: "
        + ", ".join(f"{n}={_REQUIRED_PARITY_FIXTURES[n]}" for n in _missing_parity)
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _natural_sort_key(path: Path) -> list[int | str]:
    """Sort key that handles numeric components correctly (e.g. 2.in < 10.in)."""
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


def _int_keyed_map(values: dict[str, float | None]) -> dict[int, float | None]:
    return {int(key): value for key, value in values.items()}


@pytest.fixture(scope="module")
def rh_enamide_fixture() -> dict[str, Any]:
    return _load_json(RH_FIXTURE_PATH)


@pytest.fixture(scope="module")
def sn2_fixture() -> dict[str, Any]:
    return _load_json(SN2_FIXTURE_PATH)


@pytest.fixture(scope="module")
def rh_enamide_clean_results() -> dict[str, ForceField]:
    base_molecules = MacroModel(str(MMO_PATH)).molecules
    hessian_files = sorted(JAG_DIR.glob("*.in"), key=_natural_sort_key)
    assert len(base_molecules) == len(hessian_files)

    molecules = [
        JaguarIn(str(path)).attach_hessian(molecule).with_overrides(name=f"rh_enamide_{index + 1}")
        for index, (molecule, path) in enumerate(zip(base_molecules, hessian_files))
    ]
    clean_start = load_mm3_fld(MM3_PATH, include_standard=False)
    clean_estimated = qfuerza_into(clean_start, molecules, zero_torsions=True, au_hessian=True, invalid_policy="skip")

    return {
        "clean_start": clean_start,
        "clean_estimated": clean_estimated,
    }


def test_from_structure_preserves_legacy_dof_metadata() -> None:
    structures = MacroModel(str(MMO_PATH)).molecules
    molecule = structures[0].with_overrides(name="rh_enamide_1")

    assert len(molecule.bonds) == len(structures[0].bonds)
    assert len(molecule.angles) == len(structures[0].angles)
    assert molecule.bonds[0].ff_row == structures[0].bonds[0].ff_row
    assert molecule.angles[0].ff_row == structures[0].angles[0].ff_row


def test_bond_params_match_fixture(
    rh_enamide_clean_results: dict[str, ForceField], rh_enamide_fixture: dict[str, Any]
) -> None:
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


def test_angle_params_match_fixture(
    rh_enamide_clean_results: dict[str, ForceField], rh_enamide_fixture: dict[str, Any]
) -> None:
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


def test_sn2_bond_projections_match_fixture(sn2_fixture: dict[str, Any]) -> None:
    molecule = load_xyz(SN2_XYZ_PATH, name="sn2_ts", bond_tolerance=1.5)
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
def test_rh_enamide_forcefield_roundtrip() -> None:
    """Loading, estimating, and re-loading FF gives consistent params."""
    structures = MacroModel(str(MMO_PATH)).molecules
    hessian_files = sorted(JAG_DIR.glob("*.in"), key=_natural_sort_key)
    molecules = [
        JaguarIn(str(h)).attach_hessian(s).with_overrides(name=f"rh_{i}")
        for i, (s, h) in enumerate(zip(structures, hessian_files))
    ]

    ff1 = load_mm3_fld(MM3_PATH)
    est1 = qfuerza_into(ff1, molecules, invalid_policy="skip")

    # Re-estimate from the same starting point — must be deterministic
    ff2 = load_mm3_fld(MM3_PATH)
    est2 = qfuerza_into(ff2, molecules, invalid_policy="skip")

    for b1, b2 in zip(est1.bonds, est2.bonds):
        assert b1.force_constant == pytest.approx(b2.force_constant, abs=1e-12)
        assert b1.equilibrium == pytest.approx(b2.equilibrium, abs=1e-12)
    for a1, a2 in zip(est1.angles, est2.angles):
        assert a1.force_constant == pytest.approx(a2.force_constant, abs=1e-12)
        assert a1.equilibrium == pytest.approx(a2.equilibrium, abs=1e-12)


def test_rh_enamide_param_vector_parity(
    rh_enamide_clean_results: dict[str, ForceField], rh_enamide_fixture: dict[str, Any]
) -> None:
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
@pytest.mark.validation
def test_rh_enamide_seminario_benchmark(
    rh_enamide_clean_results: dict[str, ForceField], capsys: pytest.CaptureFixture[str]
) -> None:
    """Benchmark: time the full rh-enamide Seminario pipeline (informational)."""
    structures = MacroModel(str(MMO_PATH)).molecules
    hessian_files = sorted(JAG_DIR.glob("*.in"), key=_natural_sort_key)

    # Time parsing
    t0 = time.perf_counter()
    hessians = [JaguarIn(str(p)).get_hessian(s.n_atoms) for s, p in zip(structures, hessian_files)]
    t_parse = time.perf_counter() - t0

    # Time molecule creation
    t0 = time.perf_counter()
    molecules = [s.with_hessian(h).with_overrides(name=f"rh_{i}") for i, (s, h) in enumerate(zip(structures, hessians))]
    t_mol = time.perf_counter() - t0

    # Time Seminario estimation (10 iterations for stable timing)
    ff_template = load_mm3_fld(MM3_PATH)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = qfuerza_into(ff_template, molecules, invalid_policy="skip")
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


# ---------------------------------------------------------------------------
# Zenodo-validated cisplatin tests (externally grounded)
#
# These tests verify QFUERZA rules against force field files published by
# the paper authors in Zenodo (DOI: 10.5281/zenodo.17386006).  Unlike the
# self-referential golden fixtures above, these reference values come from
# an independent source.
# ---------------------------------------------------------------------------
# Atom labels used in the .fld files that correspond to hydrogen
_CISPLATIN_H_LABELS = {"H3"}


@pytest.fixture(scope="module")
def cisplatin_zenodo() -> dict[str, Any]:
    return _load_json(CISPLATIN_ZENODO_PATH)


def _h_angle_in_cisplatin(atoms: str) -> bool:
    """Return True if the angle involves hydrogen as an outer atom."""
    parts = atoms.split("-")
    return parts[0] in _CISPLATIN_H_LABELS or parts[2] in _CISPLATIN_H_LABELS


class TestCisplatinZenodoQFUERZARules:
    """Verify QFUERZA definition rules against the paper's own force field files."""

    def test_qfuerza_bonds_equal_fuerza(self, cisplatin_zenodo: dict[str, Any]) -> None:
        """QFUERZA must not modify bond force constants (same as FUERZA)."""
        fuerza = cisplatin_zenodo["methods"]["fuerza"]
        qfuerza = cisplatin_zenodo["methods"]["qfuerza"]

        assert len(qfuerza["bonds"]) == len(fuerza["bonds"])
        for qb, fb in zip(qfuerza["bonds"], fuerza["bonds"]):
            assert qb["atoms"] == fb["atoms"]
            assert qb["force_constant"] == pytest.approx(fb["force_constant"], abs=1e-6), (
                f"Bond {qb['atoms']}: QFUERZA={qb['force_constant']}, FUERZA={fb['force_constant']}"
            )
            assert qb["equilibrium"] == pytest.approx(fb["equilibrium"], abs=1e-6)

    def test_qfuerza_nonhydrogen_angles_equal_fuerza(self, cisplatin_zenodo: dict[str, Any]) -> None:
        """Non-hydrogen angle FCs must be unchanged from FUERZA."""
        fuerza_angles = cisplatin_zenodo["methods"]["fuerza"]["angles"]
        qfuerza_angles = cisplatin_zenodo["methods"]["qfuerza"]["angles"]

        assert len(qfuerza_angles) == len(fuerza_angles)
        for qa, fa in zip(qfuerza_angles, fuerza_angles):
            assert qa["atoms"] == fa["atoms"]
            if not _h_angle_in_cisplatin(qa["atoms"]):
                assert qa["force_constant"] == pytest.approx(fa["force_constant"], abs=1e-6), (
                    f"Non-H angle {qa['atoms']}: QFUERZA={qa['force_constant']}, FUERZA={fa['force_constant']}"
                )

    def test_qfuerza_hydrogen_angles_substituted(self, cisplatin_zenodo: dict[str, Any]) -> None:
        """H-angle FCs must be exactly 0.5 mdyn·Å/rad²."""
        qfuerza_angles = cisplatin_zenodo["methods"]["qfuerza"]["angles"]
        h_angles = [a for a in qfuerza_angles if _h_angle_in_cisplatin(a["atoms"])]

        assert len(h_angles) >= 2, "Expected at least 2 H-angles (H-N-Pt and H-N-H)"
        for a in h_angles:
            assert a["force_constant"] == pytest.approx(QFUERZA_H_ANGLE_DEFAULT_MDYNA, abs=1e-6), (
                f"H-angle {a['atoms']}: expected {QFUERZA_H_ANGLE_DEFAULT_MDYNA}, got {a['force_constant']}"
            )

    def test_fuerza_overestimates_hydrogen_angles(self, cisplatin_zenodo: dict[str, Any]) -> None:
        """FUERZA H-angle FCs must be larger than 0.5 (the known overestimation)."""
        fuerza_angles = cisplatin_zenodo["methods"]["fuerza"]["angles"]
        h_angles = [a for a in fuerza_angles if _h_angle_in_cisplatin(a["atoms"])]

        for a in h_angles:
            ratio = a["force_constant"] / QFUERZA_H_ANGLE_DEFAULT_MDYNA
            assert ratio > 1.5, (
                f"H-angle {a['atoms']}: FUERZA={a['force_constant']}, "
                f"ratio to QFUERZA default={ratio:.2f}× (expected >1.5×)"
            )

    def test_gamma_fuerza_bonds_equal_fuerza(self, cisplatin_zenodo: dict[str, Any]) -> None:
        """γ-FUERZA does not modify bonds (same as FUERZA)."""
        fuerza = cisplatin_zenodo["methods"]["fuerza"]
        gamma = cisplatin_zenodo["methods"]["gamma_fuerza"]

        assert len(gamma["bonds"]) == len(fuerza["bonds"])
        for gb, fb in zip(gamma["bonds"], fuerza["bonds"]):
            assert gb["atoms"] == fb["atoms"]
            assert gb["force_constant"] == pytest.approx(fb["force_constant"], abs=1e-6)

    def test_gamma_fuerza_scales_angles(self, cisplatin_zenodo: dict[str, Any]) -> None:
        """γ-FUERZA angle FCs should be FUERZA × γ where γ ≈ 0.68."""
        fuerza_angles = cisplatin_zenodo["methods"]["fuerza"]["angles"]
        gamma_angles = cisplatin_zenodo["methods"]["gamma_fuerza"]["angles"]

        assert len(gamma_angles) == len(fuerza_angles)
        ratios = []
        for ga, fa in zip(gamma_angles, fuerza_angles):
            assert ga["atoms"] == fa["atoms"]
            if fa["force_constant"] > 0.01:
                ratios.append(ga["force_constant"] / fa["force_constant"])

        assert len(ratios) >= 4
        mean_gamma = sum(ratios) / len(ratios)
        assert mean_gamma == pytest.approx(0.68, abs=0.01), f"Mean γ={mean_gamma:.4f}, expected ~0.68"
        # All ratios should be the same γ
        for r in ratios:
            assert r == pytest.approx(mean_gamma, rel=1e-3)

    def test_optimized_methods_converge(self, cisplatin_zenodo: dict[str, Any]) -> None:
        """After optimization, QFUERZA and FUERZA bond FCs converge.

        Angle FCs can diverge significantly (up to 49% for Cl-Pt-Cl)
        because different initializations land in different local minima.
        The paper documents this behavior, so we only assert bond convergence.
        """
        qopt = cisplatin_zenodo["methods"]["qfuerza_optimized"]
        fopt = cisplatin_zenodo["methods"]["fuerza_optimized"]

        assert len(qopt["bonds"]) == len(fopt["bonds"])
        for qb, fb in zip(qopt["bonds"], fopt["bonds"]):
            assert qb["atoms"] == fb["atoms"]
            assert qb["force_constant"] == pytest.approx(fb["force_constant"], rel=0.01), (
                f"Optimized bond {qb['atoms']}: QFUERZA={qb['force_constant']}, FUERZA={fb['force_constant']}"
            )

    def test_is_hydrogen_angle_matches_paper_labels(self) -> None:
        """Our _is_hydrogen_angle logic must agree with the paper's H-angle classification."""
        # Cisplatin angles from the .fld files, mapped to element tuples
        # In cisplatin: atoms 1=N(NH3), 2=Pt, 3=Cl, H3=H
        cisplatin_angles = [
            (("N", "Pt", "Cl"), False),  # N-Pt-Cl
            (("N", "Pt", "N"), False),  # N-Pt-N
            (("Cl", "Pt", "Cl"), False),  # Cl-Pt-Cl
            (("H", "N", "Pt"), True),  # H-N-Pt
            (("H", "N", "H"), True),  # H-N-H
        ]
        for elements, expected in cisplatin_angles:
            assert _is_hydrogen_angle(elements) == expected, f"_is_hydrogen_angle({elements}) should be {expected}"

    def test_approximation_uses_fixed_defaults(self, cisplatin_zenodo: dict[str, Any]) -> None:
        """Approximation method: bonds=5.0, angles=0.5 (fixed defaults, no Hessian)."""
        approx = cisplatin_zenodo["methods"]["approximation"]
        for b in approx["bonds"]:
            assert b["force_constant"] == pytest.approx(5.0, abs=1e-6)
        for a in approx["angles"]:
            assert a["force_constant"] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
#   Cisplatin Hessian → Force Constant Parity Tests (Issues #233 / #234)
# ---------------------------------------------------------------------------
#
# These tests parse the actual Gaussian log from the QFUERZA paper's Zenodo
# archive (DOI: 10.5281/zenodo.17386006), run our Seminario projection,
# and verify that:
#   1. Structure parsing produces the expected cisplatin geometry.
#   2. Bond/angle auto-detection finds the correct connectivity.
#   3. FUERZA bond FCs are reproducible from the Hessian.
#   4. The N-Pt bond FC matches the paper's published value exactly¹.
#   5. QFUERZA correctly substitutes H-angle defaults (0.5 mdyne·Å/rad²).
#
# ¹ Pt-Cl and N-H bond FCs diverge ~17% from the paper. Investigation
#   (session checkpoint 007) traced this to unreproducible code modifications
#   in the reference implementation (missing `convert_and_set` method,
#   `au_hessian` parameter not accepted by GaussLog). See issue #236.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cisplatin_molecule() -> Molecule:
    """Parse the cisplatin Gaussian log and build a Molecule."""
    log = GaussLog(str(CISPLATIN_GAUSSIAN_LOG), au_hessian=True)
    return log.molecules[-1]


@pytest.mark.integration
class TestCisplatinHessianParity:
    """Reproduce FUERZA/QFUERZA force constants from the cisplatin QM Hessian.

    Reference: Farrugia et al., JCTC 2025, 22, 469-476.
    Zenodo archive: 10.5281/zenodo.17386006
    """

    # ── Structure parsing ─────────────────────────────────────────────

    def test_structure_atom_count(self, cisplatin_molecule: Molecule) -> None:
        """Cisplatin has 11 atoms: Pt + 2 Cl + 2 N + 6 H."""
        assert len(cisplatin_molecule.symbols) == 11

    def test_structure_elements(self, cisplatin_molecule: Molecule) -> None:
        """Element composition: 1 Pt, 2 Cl, 2 N, 6 H."""
        from collections import Counter

        counts = Counter(cisplatin_molecule.symbols)
        assert counts == {"Pt": 1, "Cl": 2, "N": 2, "H": 6}

    def test_hessian_shape(self, cisplatin_molecule: Molecule) -> None:
        """Hessian must be 33×33 (3N × 3N for 11 atoms)."""
        assert cisplatin_molecule.hessian.shape == (33, 33)

    def test_hessian_symmetric(self, cisplatin_molecule: Molecule) -> None:
        """Full Hessian must be symmetric."""
        np.testing.assert_allclose(
            cisplatin_molecule.hessian,
            cisplatin_molecule.hessian.T,
            atol=1e-12,
        )

    # ── Connectivity detection ────────────────────────────────────────

    def test_bond_count(self, cisplatin_molecule: Molecule) -> None:
        """Auto-detection must find 10 bonds: 2 Pt-Cl + 2 Pt-N + 6 N-H."""
        assert len(cisplatin_molecule.bonds) == 10

    def test_bond_types(self, cisplatin_molecule: Molecule) -> None:
        """Three bond types: Cl-Pt, H-N, N-Pt."""
        from collections import Counter

        types = Counter(tuple(sorted(b.elements)) for b in cisplatin_molecule.bonds)
        assert types == {("Cl", "Pt"): 2, ("N", "Pt"): 2, ("H", "N"): 6}

    def test_angle_count(self, cisplatin_molecule: Molecule) -> None:
        """Auto-detection must find 18 angles."""
        assert len(cisplatin_molecule.angles) == 18

    def test_angle_types(self, cisplatin_molecule: Molecule) -> None:
        """Five angle types matching cisplatin square-planar + NH3 geometry."""
        from collections import Counter

        types = Counter(a.element_triple for a in cisplatin_molecule.angles)
        expected = {
            ("Cl", "Pt", "Cl"): 1,
            ("Cl", "Pt", "N"): 4,
            ("N", "Pt", "N"): 1,
            ("H", "N", "Pt"): 6,
            ("H", "N", "H"): 6,
        }
        assert types == expected

    # ── FUERZA bond force constants ──────────────────────────────────

    def test_fuerza_n_pt_matches_paper(self, cisplatin_molecule: Molecule) -> None:
        """N-Pt bond FC must match the paper's FUERZA value (1.1687 mdyne/Å).

        This is a direct validation that our Seminario eigenvalue projection
        reproduces the published result.  Uses dft_scaling=1.0 because the
        paper does not apply DFT frequency scaling to FUERZA estimates.
        """
        # Average over both N-Pt bonds
        n_pt_bonds = [b for b in cisplatin_molecule.bonds if tuple(sorted(b.elements)) == ("N", "Pt")]
        assert len(n_pt_bonds) == 2

        fcs = []
        for bond in n_pt_bonds:
            k = seminario_bond_fc(
                bond.atom_i,
                bond.atom_j,
                cisplatin_molecule.geometry,
                cisplatin_molecule.hessian,
                au_units=True,
                dft_scaling=1.0,
            )
            fcs.append(k / MDYNA_TO_KCALMOLA2)

        avg_fc = np.mean(fcs)
        # Paper: N-Pt = 1.1687 mdyne/Å (FUERZA, no DFT scaling)
        assert avg_fc == pytest.approx(1.1687, abs=0.001), f"N-Pt avg FC = {avg_fc:.4f} mdyne/Å, expected ~1.1687"

    def test_fuerza_bond_fcs_self_consistent(self, cisplatin_molecule: Molecule) -> None:
        """All FUERZA bond FCs must be reproducible from the Hessian.

        Tests self-consistency: our code produces stable values from the
        Gaussian log.  Uses default dft_scaling=0.963.
        """
        ff = qfuerza_fresh(cisplatin_molecule, functional_form=FunctionalForm.HARMONIC, strategy="fuerza")

        # Expected values from our pipeline (mdyne/Å, with default DFT scaling)
        expected = {
            ("Cl", "Pt"): 1.5676,
            ("N", "Pt"): 1.1253,
            ("H", "N"): 6.4537,
        }

        for bp in ff.bonds:
            fc_mdyna = bp.force_constant / MDYNA_TO_KCALMOLA2
            exp = expected[bp.key]
            assert fc_mdyna == pytest.approx(exp, abs=0.01), (
                f"Bond {bp.key}: got {fc_mdyna:.4f}, expected ~{exp:.4f} mdyne/Å"
            )

    def test_fuerza_bond_equilibria(self, cisplatin_molecule: Molecule) -> None:
        """Bond equilibrium lengths must match the optimized QM geometry."""
        ff = qfuerza_fresh(cisplatin_molecule, functional_form=FunctionalForm.HARMONIC, strategy="fuerza")

        expected_eq = {
            ("Cl", "Pt"): 2.32,
            ("N", "Pt"): 2.12,
            ("H", "N"): 1.02,
        }

        for bp in ff.bonds:
            exp = expected_eq[bp.key]
            assert bp.equilibrium == pytest.approx(exp, abs=0.02), (
                f"Bond {bp.key}: r0={bp.equilibrium:.3f}, expected ~{exp:.2f} Å"
            )

    # ── FUERZA angle force constants ─────────────────────────────────

    def test_fuerza_angle_fcs_self_consistent(self, cisplatin_molecule: Molecule) -> None:
        """FUERZA angle FCs must be reproducible from the Hessian."""
        ff = qfuerza_fresh(cisplatin_molecule, functional_form=FunctionalForm.HARMONIC, strategy="fuerza")

        # Expected values (kcal/(mol·rad²), with default DFT scaling)
        expected = {
            ("Cl", "Pt", "Cl"): 89.76,
            ("Cl", "Pt", "N"): 106.11,
            ("N", "Pt", "N"): 129.50,
            ("H", "N", "Pt"): 54.38,
            ("H", "N", "H"): 46.82,
        }

        for ap in ff.angles:
            exp = expected[ap.key]
            assert ap.force_constant == pytest.approx(exp, abs=0.1), (
                f"Angle {ap.key}: k={ap.force_constant:.2f}, expected ~{exp:.2f} kcal/(mol·rad²)"
            )

    # ── QFUERZA tests ────────────────────────────────────────────────

    def test_qfuerza_bonds_same_as_fuerza(self, cisplatin_molecule: Molecule) -> None:
        """QFUERZA must not modify bond force constants."""
        ff_f = qfuerza_fresh(cisplatin_molecule, functional_form=FunctionalForm.HARMONIC, strategy="fuerza")
        ff_q = qfuerza_fresh(cisplatin_molecule, functional_form=FunctionalForm.HARMONIC, strategy="qfuerza")

        ff_f_bonds = {bp.key: bp for bp in ff_f.bonds}
        ff_q_bonds = {bp.key: bp for bp in ff_q.bonds}

        assert ff_q_bonds.keys() == ff_f_bonds.keys()
        for key in ff_f_bonds:
            assert ff_q_bonds[key].force_constant == pytest.approx(ff_f_bonds[key].force_constant, rel=1e-10)

    def test_qfuerza_h_angle_substitution(self, cisplatin_molecule: Molecule) -> None:
        """QFUERZA must substitute H-angle FCs with 0.5 mdyne·Å/rad².

        This matches the paper's QFUERZA definition exactly.
        """
        ff = qfuerza_fresh(cisplatin_molecule, functional_form=FunctionalForm.HARMONIC, strategy="qfuerza")

        h_angle_keys = [("H", "N", "Pt"), ("H", "N", "H")]
        for ap in ff.angles:
            if ap.key in h_angle_keys:
                fc_mdyna = ap.force_constant * KCALMOLRAD2_TO_MDYNA_RAD2
                assert fc_mdyna == pytest.approx(QFUERZA_H_ANGLE_DEFAULT_MDYNA, abs=1e-6), (
                    f"QFUERZA angle {ap.key}: got {fc_mdyna:.4f} mdyne·Å/rad², expected {QFUERZA_H_ANGLE_DEFAULT_MDYNA}"
                )

    def test_qfuerza_heavy_angles_unchanged(self, cisplatin_molecule: Molecule) -> None:
        """QFUERZA must not modify non-hydrogen angle FCs."""
        ff_f = qfuerza_fresh(cisplatin_molecule, functional_form=FunctionalForm.HARMONIC, strategy="fuerza")
        ff_q = qfuerza_fresh(cisplatin_molecule, functional_form=FunctionalForm.HARMONIC, strategy="qfuerza")

        ff_f_angles = {ap.key: ap for ap in ff_f.angles}
        ff_q_angles = {ap.key: ap for ap in ff_q.angles}

        heavy_keys = [("Cl", "Pt", "Cl"), ("Cl", "Pt", "N"), ("N", "Pt", "N")]
        for key in heavy_keys:
            af = ff_f_angles[key]
            aq = ff_q_angles[key]
            assert aq.force_constant == pytest.approx(af.force_constant, rel=1e-10), (
                f"Heavy angle {key}: QFUERZA={aq.force_constant:.4f} ≠ FUERZA={af.force_constant:.4f}"
            )

    def test_qfuerza_h_angles_reduced_from_fuerza(self, cisplatin_molecule: Molecule) -> None:
        """QFUERZA H-angle FCs must be smaller than FUERZA (corrects overestimation)."""
        ff_f = qfuerza_fresh(cisplatin_molecule, functional_form=FunctionalForm.HARMONIC, strategy="fuerza")
        ff_q = qfuerza_fresh(cisplatin_molecule, functional_form=FunctionalForm.HARMONIC, strategy="qfuerza")

        ff_f_angles = {ap.key: ap for ap in ff_f.angles}
        ff_q_angles = {ap.key: ap for ap in ff_q.angles}

        h_angle_keys = [("H", "N", "Pt"), ("H", "N", "H")]
        for key in h_angle_keys:
            af = ff_f_angles[key]
            aq = ff_q_angles[key]
            assert aq.force_constant < af.force_constant, (
                f"Angle {key}: QFUERZA ({aq.force_constant:.2f}) should be less than FUERZA ({af.force_constant:.2f})"
            )
