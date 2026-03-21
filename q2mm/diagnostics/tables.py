"""Reusable table formatting for Q2MM diagnostics.

Provides ``TablePrinter`` for building dynamically-sized ASCII tables,
plus convenience functions for common benchmark table layouts.

Color support
-------------
Tables use ANSI 256-color codes for a green→yellow→red spectrum on
error percentages, RMSD values, and status labels.  Color is auto-
detected (TTY check + NO_COLOR env var) and can be forced via
``TablePrinter(color=True/False)`` or ``Q2MM_COLOR=1/0``.
"""

from __future__ import annotations

import io
import os
import re
import sys
from typing import Any


# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*."""
    return _ANSI_RE.sub("", text)


def _visible_len(text: str) -> int:
    """Length of *text* excluding ANSI escape sequences."""
    return len(_strip_ansi(text))


def _color_enabled() -> bool:
    """Check if color output should be used."""
    env = os.environ.get("Q2MM_COLOR", "").lower()
    if env in ("0", "false", "no"):
        return False
    if env in ("1", "true", "yes"):
        return True
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _ansi(code: int | str, text: str) -> str:
    """Wrap *text* in an ANSI escape sequence."""
    return f"\033[{code}m{text}\033[0m"


def _lerp_color(value: float, lo: float, hi: float) -> str | None:
    """Return an ANSI 256-color code for a green→yellow→red gradient.

    *value* is clamped to [lo, hi].  Returns None if color is not meaningful.
    """
    if hi <= lo:
        return None
    t = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    # Green (46) → Yellow (226) → Red (196) in 256-color palette
    if t < 0.5:
        # Green → Yellow: 46 → 82 → 118 → 154 → 190 → 226
        idx = int(t * 2 * 5)
        colors = [46, 82, 118, 154, 190, 226]
    else:
        # Yellow → Red: 226 → 220 → 214 → 208 → 202 → 196
        idx = int((t - 0.5) * 2 * 5)
        colors = [226, 220, 214, 208, 202, 196]
    return f"38;5;{colors[min(idx, len(colors) - 1)]}"


def colorize_pct(text: str, pct_err: float, *, lo: float = 0.0, hi: float = 50.0) -> str:
    """Colorize a percentage error string on a green→red spectrum."""
    code = _lerp_color(abs(pct_err), lo, hi)
    return _ansi(code, text) if code else text


def colorize_status(text: str, status: str) -> str:
    """Colorize a status label."""
    status_colors = {
        "converged": "32",  # green
        "maxiter": "33",  # yellow
        "not converged": "33",  # yellow
        "error": "31",  # red
    }
    code = status_colors.get(status)
    return _ansi(code, text) if code else text


def colorize_rmsd(text: str, rmsd: float, *, good: float = 50.0, bad: float = 500.0) -> str:
    """Colorize an RMSD value on a green→red spectrum."""
    code = _lerp_color(rmsd, good, bad)
    return _ansi(code, text) if code else text


class TablePrinter:
    """Build ASCII tables with dynamic bar sizing.

    Collects all lines, measures the widest, then sizes ``=`` and ``-``
    bars to match on ``flush()``.

    Example::

        t = TablePrinter()
        t.bar()
        t.title("My Table")
        t.bar()
        t.row("  col1   col2   col3")
        t.sep()
        t.row("  1.23   4.56   7.89")
        t.bar()
        t.flush()
    """

    def __init__(self, *, color: bool | None = None):
        self._entries: list[tuple[str, str | None]] = []
        self.color = _color_enabled() if color is None else color

    def title(self, text: str):
        self._entries.append(("text", f"  {text}"))

    def row(self, text: str):
        self._entries.append(("text", f"  {text}"))

    def bar(self):
        """Full-width ``=====`` section delimiter."""
        self._entries.append(("bar", None))

    def sep(self):
        """Content-width ``-----`` sub-separator (indented 2 spaces)."""
        self._entries.append(("sep", None))

    def blank(self):
        self._entries.append(("blank", None))

    def flush(self, file=None):
        """Print everything with bars dynamically sized to content.

        Parameters
        ----------
        file : file-like, optional
            Write to this stream instead of stdout.
        """
        text_lines = [text for _, text in self._entries if text is not None]
        w = max(_visible_len(line) for line in text_lines) if text_lines else 60
        for kind, text in self._entries:
            if kind == "bar":
                print("=" * w, file=file)
            elif kind == "sep":
                print("  " + "-" * (w - 2), file=file)
            elif kind == "blank":
                print(file=file)
            else:
                print(text, file=file)
        self._entries.clear()

    def to_string(self) -> str:
        """Render to a string instead of printing."""
        buf = io.StringIO()
        self.flush(file=buf)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Convenience table builders
# ---------------------------------------------------------------------------


def frequency_progression_table(
    qm_freqs: list[float],
    stages: list[tuple[str, list[float]]],
    *,
    title: str = "FREQUENCY PROGRESSION",
    mode_names: list[str] | None = None,
) -> TablePrinter:
    """Build a mode-by-mode frequency comparison table.

    Parameters
    ----------
    qm_freqs : list[float]
        QM reference frequencies (cm-1).
    stages : list of (label, freqs) tuples
        Each entry is (stage_label, mm_frequencies). E.g.
        [("Default FF", [...]), ("Seminario", [...]), ("Optimized", [...])].
    title : str
        Table title.
    mode_names : list[str], optional
        Names for each mode (e.g. "CF str", "CH3 def"). If None, uses
        integer indices.
    """
    import numpy as np

    n_modes = len(qm_freqs)
    qm = np.asarray(qm_freqs)

    # Column widths
    W_NAME = 6 if mode_names is None else max(len(n) for n in mode_names) + 1
    W_FREQ = 8
    W_STAGE = max(max(len(s[0]) for s in stages), W_FREQ)

    t = TablePrinter()
    t.blank()
    t.bar()
    t.title(title)
    t.bar()

    # Header
    hdr = f"{'Mode':>{W_NAME}} {'QM ref':>{W_FREQ}}"
    for label, _ in stages:
        hdr += f" | {label:>{W_STAGE}}"
    t.row(hdr)
    t.sep()

    # Data rows
    for i in range(n_modes):
        name = mode_names[i] if mode_names else str(i + 1)
        row = f"{name:>{W_NAME}} {qm[i]:>{W_FREQ}.1f}"
        for _, freqs in stages:
            row += f" | {freqs[i]:>{W_STAGE}.1f}"
        t.row(row)

    # Summary row
    t.sep()
    use_color = _color_enabled()
    summary = f"{'RMSD':>{W_NAME}} {'':>{W_FREQ}}"
    for _, freqs in stages:
        mm = np.asarray(freqs[:n_modes])
        rmsd = float(np.sqrt(np.mean((qm - mm) ** 2)))
        rmsd_s = f"{rmsd:>{W_STAGE}.1f}"
        if use_color:
            rmsd_s = colorize_rmsd(rmsd_s, rmsd)
        summary += f" | {rmsd_s}"
    t.row(summary)

    mae_row = f"{'MAE':>{W_NAME}} {'':>{W_FREQ}}"
    for _, freqs in stages:
        mm = np.asarray(freqs[:n_modes])
        mae = float(np.mean(np.abs(qm - mm)))
        mae_row += f" | {mae:>{W_STAGE}.1f}"
    t.row(mae_row)

    t.bar()
    t.blank()
    return t


def pes_distortion_table(
    distortion_results: list[dict],
    *,
    title: str = "PES DISTORTION -- MM vs QM Harmonic Energy (kcal/mol)",
    elapsed_s: float | None = None,
) -> TablePrinter:
    """Build a PES distortion comparison table.

    Parameters
    ----------
    distortion_results : list[dict]
        Output from ``compute_distortions()``. Each entry has keys:
        mode_idx, freq_cm1, displacements (list of {d_ang, e_qm, e_mm, pct_err}).
    title : str
        Table title.
    elapsed_s : float, optional
        Wall clock time for the distortion evaluation.
    """
    import numpy as np

    if not distortion_results:
        t = TablePrinter()
        t.row(f"{title}: no data (Hessian not available for this backend)")
        return t

    # Discover amplitudes from first mode
    target_norms = [d["d_ang"] for d in distortion_results[0]["displacements"]]

    W_E = 8  # energy value columns
    W_ERR = 8  # error string columns
    W_ME = 8  # max-error column
    W_GRP = W_E + 1 + W_E + 1 + W_ERR  # group body

    t = TablePrinter()
    t.blank()
    t.bar()
    t.title(title)
    t.row("  Displace molecule along each QM normal mode; compare QM vs MM energy.")
    t.row("  E(QM) = harmonic energy from QM Hessian.  E(MM) = MM single-point energy.")
    t.bar()

    # Header — use group label to indicate units
    sub = f"{'Mode':>6} {'Freq':>8}"
    for d in target_norms:
        sub += f" | {'E(QM)':>{W_E}} {'E(MM)':>{W_E}} {'Err':>{W_ERR}}"
    sub += f" | {'MaxErr':>{W_ME}}"
    t.row(sub)

    units = f"{'':>6} {'(cm-1)':>8}"
    for d in target_norms:
        label = f"d={d:.2f} Å"
        grp = f"{label} (kcal/mol)"
        units += f" | {grp:^{W_GRP}}"
    units += f" | {'':>{W_ME}}"
    t.row(units)
    t.sep()

    use_color = _color_enabled()

    all_pct_errors: list[float] = []
    for i, m in enumerate(distortion_results, 1):
        row = f"{i:>6d} {m['freq_cm1']:>8.1f}"
        mode_max_err = 0.0
        for disp in m["displacements"]:
            err_s = f"{disp['pct_err']:+.1f}%"
            if use_color:
                err_s = colorize_pct(f"{err_s:>{W_ERR}}", disp["pct_err"])
            else:
                err_s = f"{err_s:>{W_ERR}}"
            row += f" | {disp['e_qm']:>{W_E}.3f} {disp['e_mm']:>{W_E}.3f} {err_s}"
            mode_max_err = max(mode_max_err, abs(disp["pct_err"]))
            all_pct_errors.append(abs(disp["pct_err"]))
        me_s = f"{mode_max_err:.1f}%"
        if use_color:
            me_s = colorize_pct(f"{me_s:>{W_ME}}", mode_max_err)
        else:
            me_s = f"{me_s:>{W_ME}}"
        row += f" | {me_s}"
        t.row(row)

    t.sep()
    if all_pct_errors:
        median_err = float(np.median(all_pct_errors))
        max_err = float(np.max(all_pct_errors))
        t.row(f"Median |error|: {median_err:.1f}%    Max |error|: {max_err:.1f}%")
    if elapsed_s is not None:
        n_modes = len(distortion_results)
        n_amps = len(target_norms)
        t.row(f"Distortion eval time: {elapsed_s * 1000:.1f} ms ({n_modes} modes x {n_amps} amplitudes)")
    t.bar()
    t.blank()
    return t


def timing_table(
    timings: dict[str, Any],
    *,
    title: str = "TIMING",
) -> TablePrinter:
    """Build a timing breakdown table.

    Parameters
    ----------
    timings : dict
        Keys like "seminario_s", "optimization_s", "n_eval",
        "per_eval_ms", etc.
    """
    t = TablePrinter()
    t.blank()
    t.bar()
    t.title(title)
    t.bar()

    W_LABEL = 24
    W_VAL = 12

    for key, val in timings.items():
        label = key.replace("_", " ").title()
        if isinstance(val, float):
            if "ms" in key or val < 1.0:
                val_s = f"{val * 1000:.1f} ms"
            else:
                val_s = f"{val:.2f} s"
        else:
            val_s = str(val)
        t.row(f"{label:<{W_LABEL}} {val_s:>{W_VAL}}")

    t.bar()
    t.blank()
    return t


def parameter_table(
    param_names: list[str],
    default_values: list[float] | None,
    seminario_values: list[float] | None,
    optimized_values: list[float],
    *,
    title: str = "OPTIMIZED PARAMETERS",
) -> TablePrinter:
    """Build a parameter comparison table showing Default → Seminario → Optimized."""
    t = TablePrinter()
    t.blank()
    t.bar()
    t.title(title)
    t.bar()

    W_NAME = max(len(n) for n in param_names) + 1 if param_names else 12
    W_VAL = 10

    has_default = default_values and len(default_values) == len(param_names)
    has_seminario = seminario_values and len(seminario_values) == len(param_names)

    hdr = f"{'Parameter':>{W_NAME}}"
    if has_default:
        hdr += f" {'Default':>{W_VAL}}"
    if has_seminario:
        hdr += f" {'Seminario':>{W_VAL}}"
    hdr += f" {'Optimized':>{W_VAL}} {'Change%':>8}"
    t.row(hdr)
    t.sep()

    for i, name in enumerate(param_names):
        # Change% is relative to the optimizer starting point
        if has_seminario:
            base = seminario_values[i]
        elif has_default:
            base = default_values[i]
        else:
            base = optimized_values[i]

        opt = optimized_values[i]
        pct = ((opt - base) / abs(base) * 100) if abs(base) > 1e-12 else 0.0
        pct_s = f"{pct:+.1f}%"

        row = f"{name:>{W_NAME}}"
        if has_default:
            row += f" {default_values[i]:>{W_VAL}.4f}"
        if has_seminario:
            row += f" {seminario_values[i]:>{W_VAL}.4f}"
        row += f" {opt:>{W_VAL}.4f} {pct_s:>8}"
        t.row(row)

    t.bar()
    t.blank()
    return t


def convergence_table(
    initial_score: float,
    final_score: float,
    n_eval: int,
    converged: bool,
    message: str = "",
    *,
    title: str = "CONVERGENCE",
) -> TablePrinter:
    """Build a convergence summary table."""
    t = TablePrinter()
    t.blank()
    t.bar()
    t.title(title)
    t.bar()

    improvement = (1.0 - final_score / initial_score) * 100 if initial_score > 0 else 0.0
    status = "converged" if converged else "NOT converged"

    t.row(f"Initial score:     {initial_score:.6f}")
    t.row(f"Final score:       {final_score:.6f}")
    t.row(f"Improvement:       {improvement:.1f}%")
    t.row(f"Function evals:    {n_eval}")
    t.row(f"Status:            {status}")
    if message:
        t.row(f"Message:           {message}")

    t.bar()
    t.blank()
    return t


def leaderboard_table(
    rows: list[dict],
    *,
    title: str = "BACKEND x OPTIMIZER LEADERBOARD",
) -> TablePrinter:
    """Build a summary leaderboard from multiple benchmark results.

    Parameters
    ----------
    rows : list[dict]
        Each dict has keys: backend, optimizer, rmsd, mae, time_s,
        n_eval, final_score, converged, message, error.
    """
    t = TablePrinter()
    t.blank()
    t.bar()
    t.title(title)
    t.bar()

    W_BE = max((len(r["backend"]) for r in rows), default=8)
    W_OPT = max((len(r["optimizer"]) for r in rows), default=10)

    hdr = f"{'Backend':<{W_BE}}  {'Optimizer':<{W_OPT}}  {'RMSD':>6}  {'MAE':>6}  {'Time':>7}  {'Evals':>5}  {'Score':>8}  {'Status':>12}"
    t.row(hdr)
    t.sep()

    use_color = _color_enabled()

    for r in rows:
        if r.get("error"):
            status = "error"
        elif r["converged"]:
            status = "converged"
        else:
            msg = (r.get("message") or "").lower()
            if "iteration" in msg or "maxiter" in msg or "limit" in msg:
                status = "maxiter"
            else:
                status = "not converged"
        time_s = f"{r['time_s']:.1f}s"
        rmsd_s = f"{r['rmsd']:>6.1f}"
        status_s = f"{status:>12}"
        if use_color:
            import math

            if not math.isnan(r["rmsd"]):
                rmsd_s = colorize_rmsd(rmsd_s, r["rmsd"])
            status_s = colorize_status(status_s, status)
        t.row(
            f"{r['backend']:<{W_BE}}  {r['optimizer']:<{W_OPT}}  "
            f"{rmsd_s}  {r['mae']:>6.1f}  {time_s:>7}  "
            f"{r['n_eval']:>5d}  {r['final_score']:>8.4f}  {status_s}"
        )

    t.bar()
    t.blank()
    return t
