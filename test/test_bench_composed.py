"""Regression tests for scripts/bench_composed.py.

F0 (audit 2026-07-02): ``run_workflow_b`` Phase 2 rebuilt ``SystemData`` with a
nonexistent ``freq_ref`` keyword.  ``SystemData`` is a frozen dataclass whose
reference field is ``reference`` and which has no ``freq_ref`` attribute, so
every composed "multi-start -> optax refinement" run aborted with a
``TypeError`` the moment Phase 1 finished, making Phase 2 permanently dead.

These tests lock the contract at the source level so a future re-typo of any
``SystemData(...)`` keyword in the script fails loudly, without needing to run
the full (GPU/backend) benchmark.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

from q2mm.diagnostics.systems import SystemData

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_COMPOSED = REPO_ROOT / "scripts" / "bench_composed.py"


def _systemdata_keyword_sets() -> list[set[str]]:
    """Return the keyword names of every ``SystemData(...)`` call in the script."""
    source = BENCH_COMPOSED.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(BENCH_COMPOSED))
    calls: list[set[str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "SystemData":
            calls.append({kw.arg for kw in node.keywords if kw.arg is not None})
    return calls


def test_systemdata_has_no_freq_ref_field() -> None:
    """The dataclass exposes ``reference`` and never ``freq_ref`` (F0 root cause)."""
    field_names = {f.name for f in dataclasses.fields(SystemData)}
    assert "reference" in field_names
    assert "freq_ref" not in field_names


def test_bench_composed_systemdata_calls_use_valid_fields() -> None:
    """Every SystemData construction in the script uses only real fields (F0)."""
    valid_fields = {f.name for f in dataclasses.fields(SystemData)}
    calls = _systemdata_keyword_sets()

    # The script must actually build a SystemData (Phase 2 reconstruction).
    assert calls, "expected at least one SystemData(...) call in bench_composed.py"

    for kwargs in calls:
        unknown = kwargs - valid_fields
        assert not unknown, f"bench_composed.py builds SystemData with unknown field(s): {sorted(unknown)}"
        # Guard the specific F0 regression explicitly.
        assert "freq_ref" not in kwargs
