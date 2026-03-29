"""Reference data analysis tools for Hessians and eigenmatrices.

Diagnostic utilities for sanity-checking QM reference data before
optimization.  Complements the benchmark and PES distortion tools by
focusing on the *input* Hessian quality rather than the *output* fit.

Implements the tools proposed in GitHub issue #122:

- **Eigenvalue spectrum analysis**: count/magnitude of negatives,
  condition number, real-mode filtering.
- **Symmetry check**: verify Hessian is symmetric within tolerance.
- **Frequency comparison**: QM vs MM RMSD, MAE, max deviation,
  per-mode percent error.
- **Mode coupling analysis**: off-diagonal magnitude in eigenmatrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from q2mm.constants import REAL_FREQUENCY_THRESHOLD
from q2mm.models.hessian import decompose, hessian_to_frequencies, transform_to_eigenmatrix

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class EigenvalueAnalysis:
    """Results from eigenvalue spectrum analysis.

    Attributes:
        eigenvalues: All eigenvalues sorted ascending.
        n_negative: Number of negative eigenvalues.
        negative_values: The negative eigenvalues themselves.
        n_zero: Number of near-zero eigenvalues (below *zero_tol*).
        condition_number: Ratio of largest to smallest positive eigenvalue.
            ``inf`` when no positive eigenvalues exist.
        expected_negatives: Expected count (1 for TS, 0 for minimum).
        is_consistent: Whether negative count matches expectation.

    """

    eigenvalues: np.ndarray
    n_negative: int
    negative_values: np.ndarray
    n_zero: int
    condition_number: float
    expected_negatives: int
    is_consistent: bool


@dataclass
class SymmetryCheck:
    """Results from Hessian symmetry verification.

    Attributes:
        is_symmetric: Whether ``max_deviation <= tolerance``.
        max_deviation: Largest absolute difference between ``H[i,j]``
            and ``H[j,i]``.
        mean_deviation: Mean absolute asymmetry.
        tolerance: Threshold used for the check.

    """

    is_symmetric: bool
    max_deviation: float
    mean_deviation: float
    tolerance: float


@dataclass
class FrequencyComparison:
    """QM vs MM (or other) frequency comparison metrics.

    Attributes:
        qm_frequencies: QM reference frequencies (real modes, cm⁻¹).
        other_frequencies: Comparison frequencies (real modes, cm⁻¹).
        n_modes: Number of modes compared.
        rmsd: Root-mean-square deviation (cm⁻¹).
        mae: Mean absolute error (cm⁻¹).
        max_deviation: Largest absolute difference (cm⁻¹).
        per_mode: Per-mode detail as list of dicts with keys ``qm``,
            ``other``, ``diff``, ``pct_err``.

    """

    qm_frequencies: np.ndarray
    other_frequencies: np.ndarray
    n_modes: int
    rmsd: float
    mae: float
    max_deviation: float
    per_mode: list[dict[str, float]] = field(default_factory=list)


@dataclass
class ModeCouplingAnalysis:
    """Results from eigenmatrix off-diagonal analysis.

    Attributes:
        coupling_matrix: Absolute off-diagonal elements normalized by
            the geometric mean of the corresponding diagonal elements.
        max_coupling: Largest normalized off-diagonal value.
        mean_coupling: Mean normalized off-diagonal value.
        strongly_coupled_pairs: List of ``(i, j, value)`` tuples for
            pairs exceeding *coupling_threshold*.

    """

    coupling_matrix: np.ndarray
    max_coupling: float
    mean_coupling: float
    strongly_coupled_pairs: list[tuple[int, int, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def analyze_eigenvalues(
    hessian: np.ndarray,
    *,
    is_transition_state: bool = True,
    zero_tol: float = 1e-6,
) -> EigenvalueAnalysis:
    """Analyze the eigenvalue spectrum of a Hessian matrix.

    Args:
        hessian: ``(3N, 3N)`` Hessian matrix (any consistent unit system).
        is_transition_state: If ``True``, expect exactly 1 negative
            eigenvalue; if ``False``, expect 0.
        zero_tol: Eigenvalues with absolute value below this are
            counted as "near-zero".

    Returns:
        EigenvalueAnalysis with spectrum statistics.

    Raises:
        ValueError: If *hessian* is not square.

    """
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError(f"Hessian must be square, got shape {hessian.shape}")

    eigenvalues, _ = decompose(hessian)

    neg_mask = eigenvalues < -zero_tol
    zero_mask = np.abs(eigenvalues) <= zero_tol
    pos_mask = eigenvalues > zero_tol

    negative_values = eigenvalues[neg_mask]
    n_negative = int(neg_mask.sum())
    n_zero = int(zero_mask.sum())

    positive_evals = eigenvalues[pos_mask]
    if len(positive_evals) >= 2:
        condition_number = float(positive_evals[-1] / positive_evals[0])
    elif len(positive_evals) == 1:
        condition_number = 1.0
    else:
        condition_number = float("inf")

    expected = 1 if is_transition_state else 0
    is_consistent = n_negative == expected

    return EigenvalueAnalysis(
        eigenvalues=eigenvalues,
        n_negative=n_negative,
        negative_values=negative_values,
        n_zero=n_zero,
        condition_number=condition_number,
        expected_negatives=expected,
        is_consistent=is_consistent,
    )


def check_symmetry(
    hessian: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> SymmetryCheck:
    """Check whether a Hessian matrix is symmetric.

    Args:
        hessian: ``(3N, 3N)`` Hessian matrix.
        tolerance: Maximum acceptable deviation between ``H[i,j]``
            and ``H[j,i]``.

    Returns:
        SymmetryCheck with deviation statistics.

    Raises:
        ValueError: If *hessian* is not square.

    """
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError(f"Hessian must be square, got shape {hessian.shape}")

    diff = np.abs(hessian - hessian.T)
    max_dev = float(diff.max())
    # Mean of upper-triangular off-diagonal elements
    n = hessian.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    mean_dev = float(diff[triu_indices].mean()) if len(triu_indices[0]) > 0 else 0.0

    return SymmetryCheck(
        is_symmetric=max_dev <= tolerance,
        max_deviation=max_dev,
        mean_deviation=mean_dev,
        tolerance=tolerance,
    )


def compare_frequencies(
    qm_freqs: Sequence[float] | np.ndarray,
    other_freqs: Sequence[float] | np.ndarray,
    *,
    threshold: float = REAL_FREQUENCY_THRESHOLD,
) -> FrequencyComparison:
    """Compare two sets of vibrational frequencies.

    Both input arrays are filtered to real modes (above *threshold*)
    and sorted.  The shorter array is used to determine the number of
    modes compared.

    Args:
        qm_freqs: QM reference frequencies (cm⁻¹).
        other_freqs: Comparison frequencies (cm⁻¹), e.g. from an MM
            engine.
        threshold: Minimum frequency to include (cm⁻¹).

    Returns:
        FrequencyComparison with RMSD, MAE, max deviation, and
        per-mode detail.

    """
    qm = np.sort(np.asarray(qm_freqs, dtype=float))
    other = np.sort(np.asarray(other_freqs, dtype=float))

    # Filter to real modes
    qm = qm[qm > threshold]
    other = other[other > threshold]

    n = min(len(qm), len(other))
    if n == 0:
        return FrequencyComparison(
            qm_frequencies=qm,
            other_frequencies=other,
            n_modes=0,
            rmsd=0.0,
            mae=0.0,
            max_deviation=0.0,
        )

    qm_matched = qm[:n]
    other_matched = other[:n]
    diffs = other_matched - qm_matched

    rmsd = float(np.sqrt(np.mean(diffs**2)))
    mae = float(np.mean(np.abs(diffs)))
    max_dev = float(np.max(np.abs(diffs)))

    per_mode = []
    for i in range(n):
        pct_err = 100.0 * diffs[i] / qm_matched[i] if abs(qm_matched[i]) > 1e-10 else 0.0
        per_mode.append(
            {
                "mode": i,
                "qm": float(qm_matched[i]),
                "other": float(other_matched[i]),
                "diff": float(diffs[i]),
                "pct_err": float(pct_err),
            }
        )

    return FrequencyComparison(
        qm_frequencies=qm_matched,
        other_frequencies=other_matched,
        n_modes=n,
        rmsd=rmsd,
        mae=mae,
        max_deviation=max_dev,
        per_mode=per_mode,
    )


def analyze_mode_coupling(
    hessian: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    coupling_threshold: float = 0.1,
    skip_modes: int = 0,
) -> ModeCouplingAnalysis:
    """Analyze off-diagonal mode coupling in an eigenmatrix.

    Projects *hessian* onto the basis defined by *eigenvectors* and
    examines how large the off-diagonal elements are relative to the
    diagonal.  Large values indicate that modes in the projection basis
    are coupled in *hessian* — common when comparing an MM Hessian
    against QM eigenvectors.

    Args:
        hessian: ``(3N, 3N)`` Hessian matrix.
        eigenvectors: ``(3N, 3N)`` eigenvector matrix (columns are
            eigenvectors, ``np.linalg.eigh`` convention).
        coupling_threshold: Normalized coupling above this is flagged
            as "strongly coupled".
        skip_modes: Number of lowest modes to skip (e.g. 6 for
            translations/rotations).

    Returns:
        ModeCouplingAnalysis with coupling matrix and flagged pairs.

    """
    eigenmatrix = transform_to_eigenmatrix(hessian, eigenvectors)

    n = eigenmatrix.shape[0]
    start = skip_modes

    diag = np.abs(np.diag(eigenmatrix))

    # Normalized coupling: |E[i,j]| / sqrt(|E[i,i]| * |E[j,j]|)
    coupling = np.zeros((n, n))
    for i in range(start, n):
        for j in range(start, i):
            denom = np.sqrt(diag[i] * diag[j]) if (diag[i] > 1e-30 and diag[j] > 1e-30) else 1.0
            coupling[i, j] = abs(eigenmatrix[i, j]) / denom
            coupling[j, i] = coupling[i, j]

    # Collect strongly coupled pairs
    pairs = []
    for i in range(start, n):
        for j in range(start, i):
            if coupling[i, j] > coupling_threshold:
                pairs.append((i, j, float(coupling[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=True)

    # Statistics over upper triangle (excluding skipped modes)
    triu = []
    for i in range(start, n):
        for j in range(start, i):
            triu.append(coupling[i, j])

    max_coupling = float(max(triu)) if triu else 0.0
    mean_coupling = float(np.mean(triu)) if triu else 0.0

    return ModeCouplingAnalysis(
        coupling_matrix=coupling,
        max_coupling=max_coupling,
        mean_coupling=mean_coupling,
        strongly_coupled_pairs=pairs,
    )


# ---------------------------------------------------------------------------
# Text report helper
# ---------------------------------------------------------------------------


def format_hessian_report(
    hessian: np.ndarray,
    *,
    symbols: Sequence[str] | None = None,
    is_transition_state: bool = True,
    mm_frequencies: Sequence[float] | np.ndarray | None = None,
    mm_hessian: np.ndarray | None = None,
) -> str:
    """Generate a human-readable diagnostic report for a Hessian.

    Runs all available analyses and formats the results as a multi-
    section text report suitable for terminal output or logging.

    Args:
        hessian: ``(3N, 3N)`` QM Hessian matrix (Hartree/Bohr²).
        symbols: Element symbols (length N) for frequency conversion.
            If provided, frequencies are computed and included.
        is_transition_state: Expected saddle-point order.
        mm_frequencies: Optional MM frequencies for comparison.
        mm_hessian: Optional MM Hessian for mode coupling analysis.

    Returns:
        Multi-line report string.

    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Hessian Reference Data Analysis")
    lines.append("=" * 60)

    # --- Symmetry ---
    sym = check_symmetry(hessian)
    lines.append("")
    lines.append("Symmetry Check")
    lines.append("-" * 40)
    lines.append(f"  Symmetric:       {'YES' if sym.is_symmetric else 'NO'}")
    lines.append(f"  Max deviation:   {sym.max_deviation:.2e}")
    lines.append(f"  Mean deviation:  {sym.mean_deviation:.2e}")
    lines.append(f"  Tolerance:       {sym.tolerance:.2e}")

    # --- Eigenvalue spectrum ---
    eig = analyze_eigenvalues(hessian, is_transition_state=is_transition_state)
    lines.append("")
    lines.append("Eigenvalue Spectrum")
    lines.append("-" * 40)
    lines.append(f"  Matrix size:     {hessian.shape[0]}×{hessian.shape[1]}")
    lines.append(f"  Negative:        {eig.n_negative} (expected {eig.expected_negatives})")
    if eig.n_negative > 0:
        neg_strs = [f"{v:.4f}" for v in eig.negative_values]
        lines.append(f"  Negative values: [{', '.join(neg_strs)}]")
    lines.append(f"  Near-zero:       {eig.n_zero}")
    lines.append(f"  Condition #:     {eig.condition_number:.2e}")
    status = "OK" if eig.is_consistent else "WARNING — unexpected negative count"
    lines.append(f"  Status:          {status}")

    # --- Frequencies ---
    if symbols is not None:
        qm_all = hessian_to_frequencies(hessian, list(symbols))
        qm_real = sorted(f for f in qm_all if f > REAL_FREQUENCY_THRESHOLD)
        lines.append("")
        lines.append("QM Frequencies (cm⁻¹)")
        lines.append("-" * 40)
        lines.append(f"  Total modes:     {len(qm_all)}")
        lines.append(f"  Real modes:      {len(qm_real)}")
        if qm_real:
            lines.append(f"  Range:           {qm_real[0]:.1f} — {qm_real[-1]:.1f}")

        # --- Comparison ---
        if mm_frequencies is not None:
            comp = compare_frequencies(qm_all, mm_frequencies)
            lines.append("")
            lines.append("Frequency Comparison (QM vs MM)")
            lines.append("-" * 40)
            lines.append(f"  Modes compared:  {comp.n_modes}")
            lines.append(f"  RMSD:            {comp.rmsd:.2f} cm⁻¹")
            lines.append(f"  MAE:             {comp.mae:.2f} cm⁻¹")
            lines.append(f"  Max deviation:   {comp.max_deviation:.2f} cm⁻¹")
            if comp.per_mode:
                lines.append("")
                lines.append(f"  {'Mode':>4}  {'QM':>10}  {'MM':>10}  {'Δ':>10}  {'%err':>8}")
                for m in comp.per_mode:
                    lines.append(
                        f"  {m['mode']:4d}  {m['qm']:10.2f}  {m['other']:10.2f}"
                        f"  {m['diff']:+10.2f}  {m['pct_err']:+8.2f}%"
                    )

    # --- Mode coupling ---
    if mm_hessian is not None:
        _, qm_evecs = decompose(hessian)
        coupling = analyze_mode_coupling(mm_hessian, qm_evecs, skip_modes=6)
        lines.append("")
        lines.append("Mode Coupling (MM projected onto QM eigenvectors)")
        lines.append("-" * 40)
        lines.append(f"  Max coupling:    {coupling.max_coupling:.4f}")
        lines.append(f"  Mean coupling:   {coupling.mean_coupling:.4f}")
        if coupling.strongly_coupled_pairs:
            lines.append(f"  Strongly coupled pairs (>{coupling.strongly_coupled_pairs[0][2]:.2f}):")
            for i, j, val in coupling.strongly_coupled_pairs[:10]:
                lines.append(f"    modes {i:3d}–{j:3d}: {val:.4f}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
