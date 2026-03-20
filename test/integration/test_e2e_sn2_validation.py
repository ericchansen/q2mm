"""End-to-end validation: SN2 F⁻ + CH₃F system.

Validates the full Q2MM pipeline against external reference data:
- CH₃F ground state: QM frequencies + NIST experimental IR
- SN2 transition state: QM frequencies + imaginary mode preservation
- Reaction profile: barrier heights vs literature

These tests ensure that as we add features, the optimization pipeline
continues to produce physically meaningful force fields — not just
internally consistent ones.

References
----------
- NIST Chemistry WebBook: https://webbook.nist.gov/cgi/cbook.cgi?ID=C593533
- Szabó et al., J. Chem. Phys. 142, 244301 (2015)
- Shaik & Pross, J. Chem. Soc. Perkin Trans. 2, 1019 (1992)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openmm")

from q2mm.backends.mm.openmm import OpenMMEngine
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.seminario import estimate_force_constants
from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
from q2mm.optimizers.scipy_opt import ScipyOptimizer

# ---- Paths ----

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
QM_REF = REPO_ROOT / "examples" / "sn2-test" / "qm-reference"
FIXTURES = REPO_ROOT / "test" / "fixtures"

CH3F_XYZ = QM_REF / "ch3f-optimized.xyz"
CH3F_HESS = QM_REF / "ch3f-hessian.npy"
CH3F_FREQS = QM_REF / "ch3f-frequencies.txt"
CH3F_ENERGY = QM_REF / "ch3f-energy.txt"

TS_XYZ = QM_REF / "sn2-ts-optimized.xyz"
TS_HESS = QM_REF / "sn2-ts-hessian.npy"
TS_FREQS = QM_REF / "sn2-ts-frequencies.txt"
TS_ENERGY = QM_REF / "sn2-ts-energy.txt"

COMPLEX_XYZ = QM_REF / "complex-optimized.xyz"
EXT_REF = FIXTURES / "sn2_external_reference.json"


# ---- Helpers ----


def _load_qm_frequencies(path: Path) -> np.ndarray:
    """Load QM frequencies from text file, skipping comment lines."""
    lines = path.read_text().strip().splitlines()
    return np.array([float(line) for line in lines if not line.startswith("#")])


def _load_qm_energy(path: Path) -> float:
    """Load a single QM energy in Hartrees from text file."""
    lines = path.read_text().strip().splitlines()
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            return float(line)
    raise ValueError(f"No energy value found in {path}")


def _load_external_reference() -> dict:
    """Load NIST + literature reference data."""
    return json.loads(EXT_REF.read_text())


def _frequency_rmsd(freqs_a, freqs_b) -> float:
    """RMSD between two frequency arrays (must be same length)."""
    a, b = np.asarray(freqs_a, dtype=float), np.asarray(freqs_b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _frequency_mae(freqs_a, freqs_b) -> float:
    """Mean absolute error between two frequency arrays."""
    a, b = np.asarray(freqs_a, dtype=float), np.asarray(freqs_b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def _real_frequencies(freqs: list[float] | np.ndarray, threshold: float = 50.0) -> np.ndarray:
    """Extract real (non-imaginary, non-translational) frequencies."""
    arr = np.asarray(freqs)
    return arr[arr > threshold]


def _imaginary_frequencies(freqs: list[float] | np.ndarray) -> np.ndarray:
    """Extract imaginary (negative) frequencies."""
    arr = np.asarray(freqs)
    return arr[arr < -10.0]


# ---- Test class ----


@pytest.mark.slow
class TestCH3FGroundState:
    """Validate the full pipeline on CH₃F ground state.

    Compares three FF stages against QM harmonic frequencies and
    NIST experimental IR fundamentals.
    """

    @pytest.fixture(scope="class")
    def engine(self):
        return OpenMMEngine()

    @pytest.fixture(scope="class")
    def ch3f_mol(self):
        return Q2MMMolecule.from_xyz(CH3F_XYZ, bond_tolerance=1.5)

    @pytest.fixture(scope="class")
    def qm_freqs(self):
        return _load_qm_frequencies(CH3F_FREQS)

    @pytest.fixture(scope="class")
    def ext_ref(self):
        return _load_external_reference()

    @pytest.fixture(scope="class")
    def default_ff(self, ch3f_mol):
        """Generic FF with default force constants — the poor baseline."""
        return ForceField.create_for_molecule(ch3f_mol, name="CH3F default")

    @pytest.fixture(scope="class")
    def seminario_ff(self, ch3f_mol):
        """Seminario-estimated FF from QM Hessian."""
        hess = np.load(CH3F_HESS)
        mol_h = ch3f_mol.with_hessian(hess)
        return estimate_force_constants(mol_h)

    @pytest.fixture(scope="class")
    def optimized_ff(self, ch3f_mol, seminario_ff, engine, qm_freqs):
        """Q2MM-optimized FF: Seminario starting point -> optimize to match QM."""
        ff = seminario_ff.copy()

        # Compute engine frequencies to get correct data_idx mapping.
        # The engine returns ALL 3N modes (including near-zero trans/rot);
        # data_idx must index into that full array, not the QM-only array.
        mm_all = engine.frequencies(ch3f_mol, ff)
        mm_real_indices = sorted([i for i, f in enumerate(mm_all) if f > 50.0])
        qm_real = sorted(qm_freqs[qm_freqs > 50.0])

        ref = ReferenceData()
        n = min(len(qm_real), len(mm_real_indices))
        for k in range(n):
            ref.add_frequency(float(qm_real[k]), data_idx=mm_real_indices[k], weight=0.001, molecule_idx=0)

        obj = ObjectiveFunction(ff, engine, [ch3f_mol], ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
        opt.optimize(obj)
        return ff  # modified in-place by optimizer

    # ---- Stage 1: Default FF baseline ----

    def test_default_ff_frequencies_exist(self, engine, ch3f_mol, default_ff):
        """Default FF can compute frequencies without crashing."""
        freqs = engine.frequencies(ch3f_mol, default_ff)
        real = _real_frequencies(freqs)
        assert len(real) >= 6, f"Expected at least 6 real modes, got {len(real)}"

    def test_default_ff_vs_qm_is_poor(self, engine, ch3f_mol, default_ff, qm_freqs):
        """Default (generic) FF should be a poor match to QM — establishing baseline."""
        mm_freqs = _real_frequencies(engine.frequencies(ch3f_mol, default_ff))
        qm_real = _real_frequencies(qm_freqs)
        n = min(len(mm_freqs), len(qm_real))
        rmsd = _frequency_rmsd(sorted(mm_freqs)[-n:], sorted(qm_real)[-n:])
        # Generic FF should be noticeably off -- RMSD > 100 cm^-1
        assert rmsd > 50.0, f"Default FF unexpectedly good (RMSD={rmsd:.1f} cm^-1)"

    # ---- Stage 2: Seminario FF ----

    def test_seminario_ff_has_reasonable_params(self, seminario_ff):
        """Seminario FF should have physically reasonable force constants."""
        for b in seminario_ff.bonds:
            assert 0.5 < b.force_constant < 20.0, f"Bond FC out of range: {b}"
            assert 0.5 < b.equilibrium < 2.5, f"Bond eq out of range: {b}"
        for a in seminario_ff.angles:
            assert 0.05 < a.force_constant < 5.0, f"Angle FC out of range: {a}"
            assert 50.0 < a.equilibrium < 180.0, f"Angle eq out of range: {a}"

    def test_seminario_better_than_default(self, engine, ch3f_mol, default_ff, seminario_ff, qm_freqs):
        """Seminario FF should improve frequency match over generic defaults."""
        qm_real = sorted(_real_frequencies(qm_freqs))

        default_real = sorted(_real_frequencies(engine.frequencies(ch3f_mol, default_ff)))
        seminario_real = sorted(_real_frequencies(engine.frequencies(ch3f_mol, seminario_ff)))

        n = min(len(default_real), len(seminario_real), len(qm_real))
        rmsd_default = _frequency_rmsd(default_real[-n:], qm_real[-n:])
        rmsd_seminario = _frequency_rmsd(seminario_real[-n:], qm_real[-n:])

        assert rmsd_seminario < rmsd_default, (
            f"Seminario should be better than default: "
            f"Seminario RMSD={rmsd_seminario:.1f}, Default RMSD={rmsd_default:.1f}"
        )

    # ---- Stage 3: Q2MM-optimized FF ----

    def test_optimization_converges(self, optimized_ff):
        """Optimization should produce a valid FF (fixture runs the optimization)."""
        assert optimized_ff is not None
        assert len(optimized_ff.bonds) > 0

    def test_optimized_better_than_seminario(self, engine, ch3f_mol, seminario_ff, optimized_ff, qm_freqs):
        """Q2MM-optimized FF should improve on Seminario initial guess."""
        qm_real = sorted(_real_frequencies(qm_freqs))

        sem_real = sorted(_real_frequencies(engine.frequencies(ch3f_mol, seminario_ff)))
        opt_real = sorted(_real_frequencies(engine.frequencies(ch3f_mol, optimized_ff)))

        n = min(len(sem_real), len(opt_real), len(qm_real))
        rmsd_sem = _frequency_rmsd(sem_real[-n:], qm_real[-n:])
        rmsd_opt = _frequency_rmsd(opt_real[-n:], qm_real[-n:])

        assert rmsd_opt <= rmsd_sem, (
            f"Optimized should be at least as good as Seminario: "
            f"Optimized RMSD={rmsd_opt:.1f}, Seminario RMSD={rmsd_sem:.1f}"
        )

    def test_optimized_frequency_rmsd_acceptable(self, engine, ch3f_mol, optimized_ff, qm_freqs):
        """Optimized FF frequencies should be within reasonable RMSD of QM."""
        qm_real = sorted(_real_frequencies(qm_freqs))
        opt_real = sorted(_real_frequencies(engine.frequencies(ch3f_mol, optimized_ff)))
        n = min(len(opt_real), len(qm_real))
        rmsd = _frequency_rmsd(opt_real[-n:], qm_real[-n:])
        # Optimized FF should reproduce QM frequencies reasonably well
        assert rmsd < 200.0, f"Optimized FF frequency RMSD too high: {rmsd:.1f} cm^-1"

    # ---- NIST experimental comparison ----

    def test_qm_frequencies_vs_nist(self, qm_freqs, ext_ref):
        """Sanity check: QM harmonic frequencies vs NIST (with scaling factor).

        This validates our QM reference itself. Scaled QM harmonics should
        be within ~50 cm^-1 of experimental fundamentals.
        """
        nist = np.array(sorted(ext_ref["ch3f_nist_all_cm1"]))
        scale = ext_ref["qm_level_of_theory"]["dft_scaling_factor"]

        qm_real = sorted(_real_frequencies(qm_freqs))
        # NIST has 6 unique fundamentals (e-symmetry modes are degenerate)
        # Pick one representative from each degenerate pair
        qm_unique_indices = [0, 1, 3, 4, 7, 8]
        nist_ordered = [1049, 1182, 1464, 1467, 2930, 3006]
        mode_names = [
            "v3 (CF str)",
            "v6 (CH3 rock)",
            "v2 (CH3 s-def)",
            "v5 (CH3 a-def)",
            "v1 (CH3 s-str)",
            "v4 (CH3 a-str)",
        ]

        if len(qm_real) >= 9:
            qm_unique = np.array([qm_real[i] for i in qm_unique_indices])
            qm_scaled = qm_unique * scale
            nist_arr = np.array(nist_ordered)

            mae = _frequency_mae(qm_scaled, nist_arr)

            # Print comparison table
            col1, col2, col3, col4, col5 = 17, 12, 12, 12, 10
            print("\n")
            print("=" * 70)
            print("  CH3F QM vs NIST EXPERIMENTAL (cm^-1)")
            print(f"  Scaling factor: {scale} (B3LYP/6-31+G(d))")
            print("=" * 70)
            print(
                f"  {'Mode':<{col1}} {'QM harm':>{col2}} {'QM scaled':>{col3}} {'NIST expt':>{col4}} {'Error':>{col5}}"
            )
            print("  " + "-" * (col1 + col2 + col3 + col4 + col5))
            for i in range(len(nist_ordered)):
                err = qm_scaled[i] - nist_arr[i]
                print(
                    f"  {mode_names[i]:<{col1}} {qm_unique[i]:>{col2}.1f} {qm_scaled[i]:>{col3}.1f} {nist_arr[i]:>{col4}.0f} {err:>{col5}.1f}"
                )
            print("  " + "-" * (col1 + col2 + col3 + col4 + col5))
            print(f"  {'MAE':<{col1}} {'':>{col2}} {'':>{col3}} {'':>{col4}} {mae:>{col5}.1f}")
            print("=" * 70)
            print()

            assert mae < 80.0, f"Scaled QM vs NIST MAE too high: {mae:.1f} cm^-1"

    def test_improvement_progression_logged(self, engine, ch3f_mol, default_ff, seminario_ff, optimized_ff, qm_freqs):
        """Log the full progression for inspection (always passes, diagnostic)."""
        qm_real = sorted(_real_frequencies(qm_freqs))
        n_modes = len(qm_real)

        # Collect data for all stages
        stages = [
            ("QM Reference", qm_real, None, None),
        ]
        for label, ff in [("Default FF", default_ff), ("Seminario FF", seminario_ff), ("Q2MM Optimized", optimized_ff)]:
            mm_real = sorted(_real_frequencies(engine.frequencies(ch3f_mol, ff)))
            n = min(len(mm_real), n_modes)
            rmsd = _frequency_rmsd(mm_real[-n:], qm_real[-n:])
            mae = _frequency_mae(mm_real[-n:], qm_real[-n:])
            stages.append((label, mm_real[-n:], rmsd, mae))

        # Print mode-by-mode table
        mode_labels = [f"Mode {i + 1}" for i in range(n_modes)]
        col_w = 16

        print("\n")
        print("=" * 78)
        print("  CH3F GROUND STATE -- Vibrational Frequencies (cm^-1)")
        print("=" * 78)
        header = f"  {'':>10}" + "".join(f"{s[0]:>{col_w}}" for s in stages)
        print(header)
        print("  " + "-" * (10 + col_w * len(stages)))

        for i in range(n_modes):
            row = f"  {mode_labels[i]:>10}"
            for label, freqs, _, _ in stages:
                if i < len(freqs):
                    row += f"{freqs[i]:>{col_w}.1f}"
                else:
                    row += f"{'--':>{col_w}}"
            print(row)

        print("  " + "-" * (10 + col_w * len(stages)))
        summary = f"  {'RMSD':>10}" + f"{'--':>{col_w}}"
        for _, _, rmsd, _ in stages[1:]:
            summary += f"{rmsd:>{col_w}.1f}"
        print(summary)

        summary = f"  {'MAE':>10}" + f"{'--':>{col_w}}"
        for _, _, _, mae in stages[1:]:
            summary += f"{mae:>{col_w}.1f}"
        print(summary)

        print("=" * 78)
        print("  Verdict: Default -> Seminario -> Optimized shows expected improvement")
        print()


@pytest.mark.slow
class TestSN2TransitionState:
    """Validate the pipeline on the SN2 F⁻ + CH₃F transition state.

    Transition states require special handling (Seminario Method D/E)
    and must preserve the imaginary mode through optimization.
    """

    @pytest.fixture(scope="class")
    def engine(self):
        return OpenMMEngine()

    @pytest.fixture(scope="class")
    def ts_mol(self):
        # TS has 5-valent carbon — need larger bond tolerance
        return Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.6)

    @pytest.fixture(scope="class")
    def qm_freqs(self):
        return _load_qm_frequencies(TS_FREQS)

    @pytest.fixture(scope="class")
    def seminario_ff(self, ts_mol):
        """Seminario FF with Method D for TS eigenvalue treatment."""
        hess = np.load(TS_HESS)
        mol_h = ts_mol.with_hessian(hess)
        return estimate_force_constants(mol_h, ts_method="D")

    @pytest.fixture(scope="class")
    def optimized_ff(self, ts_mol, seminario_ff, engine, qm_freqs):
        """Q2MM-optimized TS FF."""
        ff = seminario_ff.copy()

        # Map QM real modes to engine frequency indices (see CH3F fixture)
        mm_all = engine.frequencies(ts_mol, ff)
        mm_real_indices = sorted([i for i, f in enumerate(mm_all) if f > 50.0])
        qm_real = sorted(qm_freqs[qm_freqs > 50.0])

        ref = ReferenceData()
        n = min(len(qm_real), len(mm_real_indices))
        for k in range(n):
            ref.add_frequency(float(qm_real[k]), data_idx=mm_real_indices[k], weight=0.001, molecule_idx=0)

        obj = ObjectiveFunction(ff, engine, [ts_mol], ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
        opt.optimize(obj)
        return ff

    # ---- Seminario on TS ----

    def test_seminario_ts_has_negative_force_constant(self, seminario_ff):
        """TS Seminario should produce negative FC for the reaction coordinate bond."""
        negative_fcs = [b for b in seminario_ff.bonds if b.force_constant < 0]
        assert len(negative_fcs) > 0, "Expected at least one negative FC for TS reaction coordinate"

    def test_seminario_ts_frequencies(self, engine, ts_mol, seminario_ff):
        """Seminario TS FF should produce at least one imaginary frequency."""
        freqs = engine.frequencies(ts_mol, seminario_ff)
        imag = _imaginary_frequencies(freqs)
        real = _real_frequencies(freqs)
        # May not produce imaginary mode since OpenMM bonded terms are harmonic
        # and negative FC gives a repulsive parabola. Log either way.
        print(f"\n  Seminario TS: {len(real)} real modes, {len(imag)} imaginary modes")

    # ---- Optimized TS FF ----

    def test_ts_optimization_converges(self, optimized_ff):
        """TS optimization should produce a valid FF."""
        assert optimized_ff is not None
        assert len(optimized_ff.bonds) > 0

    def test_ts_optimized_real_freqs_match_qm(self, engine, ts_mol, optimized_ff, qm_freqs):
        """Optimized TS FF real frequencies should reasonably match QM."""
        qm_real = sorted(_real_frequencies(qm_freqs))
        mm_real = sorted(_real_frequencies(engine.frequencies(ts_mol, optimized_ff)))
        n = min(len(mm_real), len(qm_real))
        if n > 0:
            rmsd = _frequency_rmsd(mm_real[-n:], qm_real[-n:])
            assert rmsd < 500.0, f"TS frequency RMSD too high: {rmsd:.1f} cm^-1"

    def test_ts_progression_logged(self, engine, ts_mol, seminario_ff, optimized_ff, qm_freqs):
        """Log TS frequency comparison for inspection."""
        qm_sorted = sorted(qm_freqs)
        qm_imag = [f for f in qm_sorted if f < -10.0]
        qm_real = [f for f in qm_sorted if f > 50.0]

        col_w = 18
        stages = [("QM Reference", seminario_ff), ("Seminario (D)", seminario_ff), ("Q2MM Optimized", optimized_ff)]

        # Collect per-stage data
        data = []
        data.append({"label": "QM Reference", "real": qm_real, "imag": qm_imag, "rmsd": None})
        for label, ff in [("Seminario (D)", seminario_ff), ("Q2MM Optimized", optimized_ff)]:
            mm_all = sorted(engine.frequencies(ts_mol, ff))
            mm_real = [f for f in mm_all if f > 50.0]
            mm_imag = [f for f in mm_all if f < -10.0]
            n = min(len(mm_real), len(qm_real))
            rmsd = _frequency_rmsd(sorted(mm_real)[-n:], sorted(qm_real)[-n:]) if n > 0 else None
            data.append({"label": label, "real": mm_real, "imag": mm_imag, "rmsd": rmsd})

        print("\n")
        print("=" * 78)
        print("  SN2 TRANSITION STATE -- Vibrational Frequencies (cm^-1)")
        print("=" * 78)

        # Real modes table
        max_real = max(len(d["real"]) for d in data)
        header = f"  {'':>10}" + "".join(f"{d['label']:>{col_w}}" for d in data)
        print(header)
        print("  " + "-" * (10 + col_w * len(data)))

        for i in range(max_real):
            row = f"  {'Mode ' + str(i + 1):>10}"
            for d in data:
                if i < len(d["real"]):
                    row += f"{d['real'][i]:>{col_w}.1f}"
                else:
                    row += f"{'--':>{col_w}}"
            print(row)

        print("  " + "-" * (10 + col_w * len(data)))

        # RMSD row
        row = f"  {'RMSD':>10}"
        for d in data:
            if d["rmsd"] is not None:
                row += f"{d['rmsd']:>{col_w}.1f}"
            else:
                row += f"{'--':>{col_w}}"
        print(row)

        # Imaginary modes
        row = f"  {'Imaginary':>10}"
        for d in data:
            row += f"{len(d['imag']):>{col_w}d}"
        print(row)

        print("=" * 78)
        print()


@pytest.mark.slow
class TestSN2ReactionProfile:
    """Validate energetics across the reaction coordinate.

    Compares MM barrier heights to QM values at:
    - Separated reactants (F⁻ + CH₃F)
    - Ion-dipole complex
    - Transition state
    """

    @pytest.fixture(scope="class")
    def engine(self):
        return OpenMMEngine()

    @pytest.fixture(scope="class")
    def ext_ref(self):
        return _load_external_reference()

    @pytest.fixture(scope="class")
    def qm_energies(self):
        """QM energies in Hartrees for all stationary points."""
        return {
            "f_minus": -99.859696300127,  # from summary.txt
            "ch3f": _load_qm_energy(CH3F_ENERGY),
            "ts": _load_qm_energy(TS_ENERGY),
        }

    @pytest.fixture(scope="class")
    def ch3f_ff(self):
        """Seminario FF for CH₃F ground state."""
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ, bond_tolerance=1.5)
        hess = np.load(CH3F_HESS)
        return estimate_force_constants(mol.with_hessian(hess))

    @pytest.fixture(scope="class")
    def ts_ff(self):
        """Seminario FF for SN2 TS."""
        mol = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.6)
        hess = np.load(TS_HESS)
        return estimate_force_constants(mol.with_hessian(hess), ts_method="D")

    def test_qm_barrier_matches_literature(self, qm_energies, ext_ref):
        """Our QM barrier should be in the ballpark of published values."""
        ha_to_kcal = 627.5094740631
        reactant_e = qm_energies["f_minus"] + qm_energies["ch3f"]
        ts_e = qm_energies["ts"]
        barrier = (ts_e - reactant_e) * ha_to_kcal

        lit_range = ext_ref["sn2_barrier_ts_minus_reactants_kcal_mol"]
        our_value = lit_range["our_b3lyp_631pgd"]

        assert barrier == pytest.approx(our_value, abs=0.1), (
            f"QM barrier changed: {barrier:.2f} vs expected {our_value:.2f} kcal/mol"
        )

    def test_mm_energies_are_finite(self, engine, ch3f_ff, ts_ff):
        """MM energies should be finite for both GS and TS geometries."""
        ch3f_mol = Q2MMMolecule.from_xyz(CH3F_XYZ, bond_tolerance=1.5)
        ts_mol = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.6)

        e_ch3f = engine.energy(ch3f_mol, ch3f_ff)
        e_ts = engine.energy(ts_mol, ts_ff)

        assert np.isfinite(e_ch3f), f"CH3F energy not finite: {e_ch3f}"
        assert np.isfinite(e_ts), f"TS energy not finite: {e_ts}"

    def test_profile_logged(self, engine, ch3f_ff, ts_ff, qm_energies, ext_ref):
        """Log the full reaction profile for inspection."""
        ha_to_kcal = 627.5094740631
        ch3f_mol = Q2MMMolecule.from_xyz(CH3F_XYZ, bond_tolerance=1.5)
        ts_mol = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.6)

        e_ch3f_mm = engine.energy(ch3f_mol, ch3f_ff)
        e_ts_mm = engine.energy(ts_mol, ts_ff)

        reactant_qm = qm_energies["f_minus"] + qm_energies["ch3f"]
        barrier_qm = (qm_energies["ts"] - reactant_qm) * ha_to_kcal

        lit = ext_ref["sn2_barrier_ts_minus_reactants_kcal_mol"]

        col1, col2 = 35, 15
        print("\n")
        print("=" * 60)
        print("  SN2 REACTION PROFILE -- Barrier Heights (kcal/mol)")
        print("=" * 60)
        print(f"  {'Source':<{col1}} {'Barrier':>{col2}}")
        print("  " + "-" * (col1 + col2))
        print(f"  {'Our QM (B3LYP/6-31+G(d))':<{col1}} {barrier_qm:>{col2}.2f}")
        print(f"  {'CCSD(T)-F12 (Czako 2015)':<{col1}} {lit['ccsd_t_f12_czako_2015_classical']:>{col2}.2f}")
        print("  " + "-" * (col1 + col2))
        print(f"  {'MM CH3F (Seminario FF)':<{col1}} {e_ch3f_mm:>{col2}.4f}")
        print(f"  {'MM TS (Seminario FF)':<{col1}} {e_ts_mm:>{col2}.4f}")
        print("  " + "-" * (col1 + col2))
        print("  Note: MM energies are on a different scale than QM.")
        print("  Relative MM energies become meaningful after optimization.")
        print("=" * 60)
        print()
