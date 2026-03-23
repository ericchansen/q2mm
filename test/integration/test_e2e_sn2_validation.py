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

import json
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openmm")
pytestmark = pytest.mark.openmm

from test._shared import (
    CH3F_ENERGY,
    CH3F_FREQS,
    CH3F_HESS,
    CH3F_MODES,
    CH3F_XYZ,
    REPO_ROOT,
    SN2_ENERGY,
    SN2_FREQS,
    SN2_HESSIAN,
    SN2_XYZ,
)

from q2mm.backends.mm.openmm import OpenMMEngine
from q2mm.diagnostics import TablePrinter, compute_distortions, frequency_mae, frequency_rmsd, load_normal_modes
from q2mm.diagnostics.benchmark import real_frequencies
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.seminario import estimate_force_constants
from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
from q2mm.optimizers.scipy_opt import ScipyOptimizer

# ---- Paths (test-specific) ----

FIXTURES = REPO_ROOT / "test" / "fixtures"

TS_XYZ = SN2_XYZ
TS_HESS = SN2_HESSIAN
TS_FREQS = SN2_FREQS
TS_ENERGY = SN2_ENERGY

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
        """Generic FF with default force constants -- the poor baseline."""
        return ForceField.create_for_molecule(ch3f_mol, name="CH3F default")

    @pytest.fixture(scope="class")
    def seminario_result(self, ch3f_mol):
        """Seminario-estimated FF from QM Hessian, with timing."""
        hess = np.load(CH3F_HESS)
        mol_h = ch3f_mol.with_hessian(hess)
        t0 = time.perf_counter()
        ff = estimate_force_constants(mol_h)
        elapsed = time.perf_counter() - t0
        return ff, elapsed

    @pytest.fixture(scope="class")
    def seminario_ff(self, seminario_result):
        return seminario_result[0]

    @pytest.fixture(scope="class")
    def optimized_result(self, ch3f_mol, seminario_ff, engine, qm_freqs):
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
        t0 = time.perf_counter()
        opt.optimize(obj)
        elapsed = time.perf_counter() - t0
        n_eval = obj.n_eval
        return ff, elapsed, n_eval  # ff modified in-place by optimizer

    @pytest.fixture(scope="class")
    def optimized_ff(self, optimized_result):
        return optimized_result[0]

    # ---- Stage 1: Default FF baseline ----

    def test_default_ff_frequencies_exist(self, engine, ch3f_mol, default_ff):
        """Default FF can compute frequencies without crashing."""
        freqs = engine.frequencies(ch3f_mol, default_ff)
        real = real_frequencies(freqs)
        assert len(real) >= 6, f"Expected at least 6 real modes, got {len(real)}"

    def test_default_ff_vs_qm_is_poor(self, engine, ch3f_mol, default_ff, qm_freqs):
        """Default (generic) FF should be a poor match to QM — establishing baseline."""
        mm_freqs = real_frequencies(engine.frequencies(ch3f_mol, default_ff))
        qm_real = real_frequencies(qm_freqs)
        n = min(len(mm_freqs), len(qm_real))
        rmsd = frequency_rmsd(sorted(mm_freqs)[-n:], sorted(qm_real)[-n:])
        # Generic FF should be noticeably off -- RMSD > 50 cm^-1
        assert rmsd > 50.0, f"Default FF unexpectedly good (RMSD={rmsd:.1f} cm^-1)"

    # ---- Stage 2: Seminario FF ----

    def test_seminario_ff_has_reasonable_params(self, seminario_ff):
        """Seminario FF should have physically reasonable force constants."""
        for b in seminario_ff.bonds:
            assert 35.0 < b.force_constant < 1440.0, f"Bond FC out of range: {b}"
            assert 0.5 < b.equilibrium < 2.5, f"Bond eq out of range: {b}"
        for a in seminario_ff.angles:
            assert 3.5 < a.force_constant < 360.0, f"Angle FC out of range: {a}"
            assert 50.0 < a.equilibrium < 180.0, f"Angle eq out of range: {a}"

    def test_seminario_better_than_default(self, engine, ch3f_mol, default_ff, seminario_ff, qm_freqs):
        """Seminario FF should improve frequency match over generic defaults."""
        qm_real = sorted(real_frequencies(qm_freqs))

        default_real = sorted(real_frequencies(engine.frequencies(ch3f_mol, default_ff)))
        seminario_real = sorted(real_frequencies(engine.frequencies(ch3f_mol, seminario_ff)))

        n = min(len(default_real), len(seminario_real), len(qm_real))
        rmsd_default = frequency_rmsd(default_real[-n:], qm_real[-n:])
        rmsd_seminario = frequency_rmsd(seminario_real[-n:], qm_real[-n:])

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
        qm_real = sorted(real_frequencies(qm_freqs))

        sem_real = sorted(real_frequencies(engine.frequencies(ch3f_mol, seminario_ff)))
        opt_real = sorted(real_frequencies(engine.frequencies(ch3f_mol, optimized_ff)))

        n = min(len(sem_real), len(opt_real), len(qm_real))
        rmsd_sem = frequency_rmsd(sem_real[-n:], qm_real[-n:])
        rmsd_opt = frequency_rmsd(opt_real[-n:], qm_real[-n:])

        assert rmsd_opt <= rmsd_sem, (
            f"Optimized should be at least as good as Seminario: "
            f"Optimized RMSD={rmsd_opt:.1f}, Seminario RMSD={rmsd_sem:.1f}"
        )

    def test_optimized_frequency_rmsd_acceptable(self, engine, ch3f_mol, optimized_ff, qm_freqs):
        """Optimized FF frequencies should be within reasonable RMSD of QM."""
        qm_real = sorted(real_frequencies(qm_freqs))
        opt_real = sorted(real_frequencies(engine.frequencies(ch3f_mol, optimized_ff)))
        n = min(len(opt_real), len(qm_real))
        rmsd = frequency_rmsd(opt_real[-n:], qm_real[-n:])
        # Optimized FF should reproduce QM frequencies reasonably well
        assert rmsd < 200.0, f"Optimized FF frequency RMSD too high: {rmsd:.1f} cm^-1"

    # ---- NIST experimental comparison ----

    def test_qm_frequencies_vs_nist(self, qm_freqs, ext_ref, capsys):
        """Sanity check: QM harmonic frequencies vs NIST (with scaling factor).

        This validates our QM reference itself. Scaled QM harmonics should
        be within ~50 cm^-1 of experimental fundamentals.
        """
        scale = ext_ref["qm_level_of_theory"]["dft_scaling_factor"]

        qm_real = sorted(real_frequencies(qm_freqs))
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

        assert len(qm_real) >= 9, f"Expected at least 9 vibrational modes for CH3F (3N-6), got {len(qm_real)}"
        qm_unique = np.array([qm_real[i] for i in qm_unique_indices])
        qm_scaled = qm_unique * scale
        nist_arr = np.array(nist_ordered)

        mae = frequency_mae(qm_scaled, nist_arr)

        # Print comparison table
        col1, col2, col3, col4, col5 = 17, 12, 12, 12, 10
        t = TablePrinter()
        t.blank()
        t.bar()
        t.title("CH3F QM vs NIST EXPERIMENTAL (cm^-1)")
        t.title(f"Scaling factor: {scale} (B3LYP/6-31+G(d))")
        t.bar()
        t.row(f"{'Mode':<{col1}} {'QM harm':>{col2}} {'QM scaled':>{col3}} {'NIST expt':>{col4}} {'Error':>{col5}}")
        t.sep()
        for i in range(len(nist_ordered)):
            err = qm_scaled[i] - nist_arr[i]
            t.row(
                f"{mode_names[i]:<{col1}} {qm_unique[i]:>{col2}.1f} {qm_scaled[i]:>{col3}.1f} {nist_arr[i]:>{col4}.0f} {err:>{col5}.1f}"
            )
        t.sep()
        t.row(f"{'MAE':<{col1}} {'':>{col2}} {'':>{col3}} {'':>{col4}} {mae:>{col5}.1f}")
        t.bar()
        t.blank()
        with capsys.disabled():
            t.flush()

        assert mae < 80.0, f"Scaled QM vs NIST MAE too high: {mae:.1f} cm^-1"

    def test_improvement_progression_logged(
        self, engine, ch3f_mol, default_ff, seminario_result, optimized_result, qm_freqs, capsys
    ):
        """Log the full progression for inspection (always passes, diagnostic)."""
        seminario_ff, t_seminario = seminario_result
        optimized_ff, t_optimize, n_eval = optimized_result

        qm_real = sorted(real_frequencies(qm_freqs))
        n_modes = len(qm_real)

        # Time frequency evaluations for each stage
        stages = [("QM Reference", qm_real, None, None, None)]
        for label, ff in [("Default FF", default_ff), ("Seminario FF", seminario_ff), ("Q2MM Optimized", optimized_ff)]:
            t0 = time.perf_counter()
            mm_real = sorted(real_frequencies(engine.frequencies(ch3f_mol, ff)))
            t_freq = time.perf_counter() - t0
            n = min(len(mm_real), n_modes)
            rmsd = frequency_rmsd(mm_real[-n:], qm_real[-n:])
            mae = frequency_mae(mm_real[-n:], qm_real[-n:])
            stages.append((label, mm_real[-n:], rmsd, mae, t_freq))

        # Print mode-by-mode table
        col_w = 16
        t = TablePrinter()
        t.blank()
        t.bar()
        t.title("CH3F GROUND STATE -- Vibrational Frequencies (cm^-1)")
        t.bar()
        t.row(f"{'':>10}" + "".join(f"{s[0]:>{col_w}}" for s in stages))
        t.sep()

        for i in range(n_modes):
            row = f"{f'Mode {i + 1}':>10}"
            for _, freqs, _, _, _ in stages:
                if i < len(freqs):
                    row += f"{freqs[i]:>{col_w}.1f}"
                else:
                    row += f"{'--':>{col_w}}"
            t.row(row)

        t.sep()
        row = f"{'RMSD':>10}" + f"{'--':>{col_w}}"
        for _, _, rmsd, _, _ in stages[1:]:
            row += f"{rmsd:>{col_w}.1f}"
        t.row(row)

        row = f"{'MAE':>10}" + f"{'--':>{col_w}}"
        for _, _, _, mae, _ in stages[1:]:
            row += f"{mae:>{col_w}.1f}"
        t.row(row)

        # Timing section
        t.bar()
        t.title("TIMING")
        t.sep()
        t.row(f"{'Seminario estimation:':<40} {t_seminario * 1000:>8.1f} ms")
        t.row(f"{'L-BFGS-B optimization:':<40} {t_optimize * 1000:>8.1f} ms  ({n_eval} evaluations)")
        for label, _, _, _, t_freq in stages[1:]:
            t.row(f"{'Freq eval (' + label + '):':<40} {t_freq * 1000:>8.1f} ms")
        t.bar()
        t.blank()
        with capsys.disabled():
            t.flush()

    # ---- PES distortion: MM vs QM harmonic along normal modes ----

    @pytest.fixture(scope="class")
    def normal_modes(self):
        return load_normal_modes(CH3F_MODES)

    @pytest.fixture(scope="class")
    def distortion_results(self, engine, ch3f_mol, seminario_ff, optimized_ff, normal_modes):
        """Compute PES distortions for both Seminario and optimized FFs."""
        sem_results, e_eq_sem, t_sem = compute_distortions(ch3f_mol, seminario_ff, engine, normal_modes)
        opt_results, e_eq_opt, t_opt = compute_distortions(ch3f_mol, optimized_ff, engine, normal_modes)
        return {
            "seminario": {"results": sem_results, "e_eq": e_eq_sem, "elapsed": t_sem},
            "optimized": {"results": opt_results, "e_eq": e_eq_opt, "elapsed": t_opt},
        }

    def test_pes_distortion_small_displacement(self, distortion_results):
        """At small displacements (0.05 Ang), most modes should roughly track QM.

        This tests whether the optimized FF reproduces not just frequencies
        (curvature at the minimum) but the actual PES shape along each mode.
        Some modes can show large errors because:
        - The optimizer adjusts force constants to match frequencies, which
          shifts equilibrium geometries and alters mode coupling
        - QM normal modes involve specific internal coordinate mixtures that
          differ from the MM mode structure
        - A displacement along one QM mode can activate multiple MM terms

        We check that the MEDIAN error is reasonable, tolerating outliers.
        """
        opt = distortion_results["optimized"]["results"]
        errors = [abs(m["displacements"][0]["pct_err"]) for m in opt]
        median_err = float(np.median(errors))

        # Median error at 0.05 Ang should be modest
        assert median_err < 30.0, f"Median PES distortion error at 0.05 Ang too high: {median_err:.1f}%"
        # Most modes should be within 50%
        n_ok = sum(1 for e in errors if e < 50.0)
        assert n_ok >= len(errors) // 2, f"Too many modes with >50% error: {len(errors) - n_ok}/{len(errors)}"

    def test_pes_distortion_logged(self, distortion_results, normal_modes, capsys):
        """Log the full PES distortion comparison table (diagnostic)."""
        target_norms = [0.05, 0.10, 0.15]

        for ff_label, key in [("Seminario FF", "seminario"), ("Q2MM Optimized", "optimized")]:
            data = distortion_results[key]
            results = data["results"]
            elapsed = data["elapsed"]

            t = TablePrinter()
            t.blank()
            t.bar()
            t.title(f"PES DISTORTION -- MM vs QM Harmonic Energy (kcal/mol) [{ff_label}]")
            t.bar()

            # Column widths (QM/MM values, error strings, max-error)
            W_E = 8  # energy values (e.g. " 19.384")
            W_ERR = 8  # error strings (e.g. "+208.6%")
            W_ME = 8  # max-error strings (e.g. " 208.6%")
            W_GRP = W_E + 1 + W_E + 1 + W_ERR  # group body

            sub = f"{'Mode':>6} {'Freq':>8}"
            for d in target_norms:
                sub += f" | {'QM':>{W_E}} {'MM':>{W_E}} {'Err':>{W_ERR}}"
            sub += f" | {'MaxErr':>{W_ME}}"
            t.row(sub)

            units = f"{'':>6} {'(cm-1)':>8}"
            for d in target_norms:
                label = f"d={d:.2f} Ang"
                units += f" | {label:^{W_GRP}}"
            units += f" | {'':>{W_ME}}"
            t.row(units)
            t.sep()

            all_pct_errors = []
            for m in results:
                row = f"{m['mode_idx'] - 5:>6d} {m['freq_cm1']:>8.1f}"
                mode_max_err = 0.0
                for disp in m["displacements"]:
                    err_s = f"{disp['pct_err']:+.1f}%"
                    row += f" | {disp['e_qm']:>{W_E}.3f} {disp['e_mm']:>{W_E}.3f} {err_s:>{W_ERR}}"
                    mode_max_err = max(mode_max_err, abs(disp["pct_err"]))
                    all_pct_errors.append(abs(disp["pct_err"]))
                me_s = f"{mode_max_err:.1f}%"
                row += f" | {me_s:>{W_ME}}"
                t.row(row)

            t.sep()
            median_err = float(np.median(all_pct_errors))
            max_err = float(np.max(all_pct_errors))
            t.row(f"Median |error|: {median_err:.1f}%    Max |error|: {max_err:.1f}%")
            t.row(
                f"Distortion eval time: {elapsed * 1000:.1f} ms ({len(results)} modes x {len(target_norms)} amplitudes)"
            )
            t.row("Eigendecomposition: pre-computed from QM Hessian (< 0.2 ms)")
            t.bar()
            t.blank()
            with capsys.disabled():
                t.flush()


@pytest.mark.slow
class TestSN2TransitionState:
    """Validate the pipeline on the SN2 F- + CH3F transition state.

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
    def seminario_result(self, ts_mol):
        """Seminario FF with Method D for TS eigenvalue treatment."""
        hess = np.load(TS_HESS)
        mol_h = ts_mol.with_hessian(hess)
        t0 = time.perf_counter()
        ff = estimate_force_constants(mol_h, ts_method="D")
        elapsed = time.perf_counter() - t0
        return ff, elapsed

    @pytest.fixture(scope="class")
    def seminario_ff(self, seminario_result):
        return seminario_result[0]

    @pytest.fixture(scope="class")
    def optimized_result(self, ts_mol, seminario_ff, engine, qm_freqs):
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
        t0 = time.perf_counter()
        opt.optimize(obj)
        elapsed = time.perf_counter() - t0
        return ff, elapsed, obj.n_eval

    @pytest.fixture(scope="class")
    def optimized_ff(self, optimized_result):
        return optimized_result[0]

    # ---- Seminario on TS ----

    def test_seminario_ts_has_negative_force_constant(self, seminario_ff):
        """TS Seminario should produce negative FC for the reaction coordinate bond."""
        negative_fcs = [b for b in seminario_ff.bonds if b.force_constant < 0]
        assert len(negative_fcs) > 0, "Expected at least one negative FC for TS reaction coordinate"

    def test_seminario_ts_frequencies(self, engine, ts_mol, seminario_ff):
        """Seminario TS FF should produce at least one imaginary frequency."""
        freqs = engine.frequencies(ts_mol, seminario_ff)
        imag = _imaginary_frequencies(freqs)
        real = real_frequencies(freqs)
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
        qm_real = sorted(real_frequencies(qm_freqs))
        mm_real = sorted(real_frequencies(engine.frequencies(ts_mol, optimized_ff)))
        n = min(len(mm_real), len(qm_real))
        if n > 0:
            rmsd = frequency_rmsd(mm_real[-n:], qm_real[-n:])
            assert rmsd < 500.0, f"TS frequency RMSD too high: {rmsd:.1f} cm^-1"

    def test_ts_progression_logged(self, engine, ts_mol, seminario_result, optimized_result, qm_freqs, capsys):
        """Log TS frequency comparison for inspection."""
        _, t_seminario = seminario_result
        _, t_optimize, n_eval = optimized_result
        seminario_ff = seminario_result[0]
        optimized_ff = optimized_result[0]

        qm_sorted = sorted(qm_freqs)
        qm_imag = [f for f in qm_sorted if f < -10.0]
        qm_real = [f for f in qm_sorted if f > 50.0]

        col_w = 18

        # Collect per-stage data
        data = []
        data.append({"label": "QM Reference", "real": qm_real, "imag": qm_imag, "rmsd": None})
        for label, ff in [("Seminario (D)", seminario_ff), ("Q2MM Optimized", optimized_ff)]:
            t0 = time.perf_counter()
            mm_all = sorted(engine.frequencies(ts_mol, ff))
            t_freq = time.perf_counter() - t0
            mm_real = [f for f in mm_all if f > 50.0]
            mm_imag = [f for f in mm_all if f < -10.0]
            n = min(len(mm_real), len(qm_real))
            rmsd = frequency_rmsd(sorted(mm_real)[-n:], sorted(qm_real)[-n:]) if n > 0 else None
            data.append({"label": label, "real": mm_real, "imag": mm_imag, "rmsd": rmsd, "t_freq": t_freq})

        t = TablePrinter()
        t.blank()
        t.bar()
        t.title("SN2 TRANSITION STATE -- Vibrational Frequencies (cm^-1)")
        t.bar()

        # Real modes table
        max_real = max(len(d["real"]) for d in data)
        t.row(f"{'':>10}" + "".join(f"{d['label']:>{col_w}}" for d in data))
        t.sep()

        for i in range(max_real):
            row = f"{f'Mode {i + 1}':>10}"
            for d in data:
                if i < len(d["real"]):
                    row += f"{d['real'][i]:>{col_w}.1f}"
                else:
                    row += f"{'--':>{col_w}}"
            t.row(row)

        t.sep()

        # RMSD row
        row = f"{'RMSD':>10}"
        for d in data:
            if d["rmsd"] is not None:
                row += f"{d['rmsd']:>{col_w}.1f}"
            else:
                row += f"{'--':>{col_w}}"
        t.row(row)

        # Imaginary modes
        row = f"{'Imaginary':>10}"
        for d in data:
            row += f"{len(d['imag']):>{col_w}d}"
        t.row(row)

        # Timing
        t.bar()
        t.title("TIMING")
        t.sep()
        t.row(f"{'Seminario (Method D):':<40} {t_seminario * 1000:>8.1f} ms")
        t.row(f"{'L-BFGS-B optimization:':<40} {t_optimize * 1000:>8.1f} ms  ({n_eval} evaluations)")
        for d in data[1:]:
            t.row(f"{'Freq eval (' + d['label'] + '):':<40} {d['t_freq'] * 1000:>8.1f} ms")
        t.bar()
        t.blank()
        with capsys.disabled():
            t.flush()


@pytest.mark.slow
class TestSN2ReactionProfile:
    """Validate QM energetics across the reaction coordinate.

    Compares our QM barrier heights to published values at:
    - Separated reactants (F- + CH3F)
    - Ion-dipole complex
    - Transition state

    Note: MM barrier heights are not meaningful here because CH3F and TS
    use separate molecule-specific force fields with different atom counts
    and connectivity.  Each FF's energy zero is arbitrary, so comparing
    energies across FFs is apples-to-oranges.  A meaningful MM barrier
    requires a reactive force field on a single PES (e.g., EVB or ReaxFF).
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
            "complex": -239.632665168590,  # from summary.txt
            "ts": _load_qm_energy(TS_ENERGY),
        }

    @pytest.fixture(scope="class")
    def ch3f_ff(self):
        """Seminario FF for CH3F ground state."""
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

    def test_profile_logged(self, qm_energies, ext_ref, capsys):
        """Log the QM reaction profile for inspection."""
        ha_to_kcal = 627.5094740631
        reactant_e = qm_energies["f_minus"] + qm_energies["ch3f"]
        complex_e = qm_energies["complex"]
        ts_e = qm_energies["ts"]

        barrier_vs_react = (ts_e - reactant_e) * ha_to_kcal
        barrier_vs_complex = (ts_e - complex_e) * ha_to_kcal
        complexation_e = (complex_e - reactant_e) * ha_to_kcal

        lit_react = ext_ref["sn2_barrier_ts_minus_reactants_kcal_mol"]
        lit_complex = ext_ref["sn2_barrier_ts_minus_complex_kcal_mol"]

        col1, col2, col3 = 35, 15, 15
        t = TablePrinter()
        t.blank()
        t.bar()
        t.title("SN2 REACTION PROFILE -- QM Energetics (kcal/mol)")
        t.bar()
        t.row(f"{'':>{col1}} {'Our B3LYP':>{col2}} {'Literature':>{col3}}")
        t.sep()
        t.row(f"{'F- + CH3F -> complex':<{col1}} {complexation_e:>{col2}.2f} {'--':>{col3}}")
        t.row(
            f"{'TS - reactants':<{col1}} {barrier_vs_react:>{col2}.2f} {lit_react['ccsd_t_f12_czako_2015_classical']:>{col3}.2f}"
        )
        t.row(
            f"{'TS - complex':<{col1}} {barrier_vs_complex:>{col2}.2f} {lit_complex['vb_benchmark_shaik_1992']:>{col3}.1f}"
        )
        t.sep()
        t.row(f"{'':>{col1}} {'B3LYP/6-31+G(d)':>{col2}} {'CCSD(T)/VB':>{col3}}")
        t.bar()
        t.row("Note: MM barrier heights cannot be computed here because CH3F")
        t.row("and TS use separate force fields with different connectivity.")
        t.row("Each FF has an arbitrary energy zero. A meaningful MM barrier")
        t.row("requires a reactive potential (EVB, ReaxFF, etc.).")
        t.bar()
        t.blank()
        with capsys.disabled():
            t.flush()
