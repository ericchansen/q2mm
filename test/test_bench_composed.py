"""Regression tests for ``scripts/bench_composed.py``.

Phase 2 now rebuilds an immutable ``BenchmarkCase`` / ``OptimizationProblem``
pair around the phase-1 optimized force field. These tests lock the source-
level contract so a future typo in the ``dataclasses.replace(...)`` keywords or
an accidental reintroduction of the deleted legacy system-bundle API fails loudly.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

from q2mm.benchmarks.cases import BenchmarkCase
from q2mm.models.problem import OptimizationProblem

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_COMPOSED = REPO_ROOT / "scripts" / "bench_composed.py"


def _replace_keyword_sets() -> dict[str, set[str]]:
    """Return ``dataclasses.replace`` keyword sets keyed by the target expression."""
    source = BENCH_COMPOSED.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(BENCH_COMPOSED))
    calls: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "dataclasses":
            continue
        if node.func.attr != "replace" or not node.args:
            continue
        calls[ast.unparse(node.args[0])] = {kw.arg for kw in node.keywords if kw.arg is not None}
    return calls


def test_benchmark_case_and_problem_fields_used_by_phase2_exist() -> None:
    """Phase 2 rebuild keywords must map to real immutable dataclass fields."""
    problem_fields = {field.name for field in dataclasses.fields(OptimizationProblem)}
    case_fields = {field.name for field in dataclasses.fields(BenchmarkCase)}

    assert {"starting_force_field", "active_space"} <= problem_fields
    assert "problem" in case_fields


def test_bench_composed_phase2_replace_calls_use_valid_fields() -> None:
    """Phase 2 reconstructs ``OptimizationProblem`` and ``BenchmarkCase`` via valid fields."""
    calls = _replace_keyword_sets()

    problem_kwargs = calls.get("sys_data.problem")
    case_kwargs = calls.get("sys_data")
    assert problem_kwargs is not None, "expected dataclasses.replace(sys_data.problem, ...) in bench_composed.py"
    assert case_kwargs is not None, "expected dataclasses.replace(sys_data, ...) in bench_composed.py"

    valid_problem_fields = {field.name for field in dataclasses.fields(OptimizationProblem)}
    valid_case_fields = {field.name for field in dataclasses.fields(BenchmarkCase)}

    assert problem_kwargs <= valid_problem_fields
    assert {"starting_force_field", "active_space"} <= problem_kwargs
    assert case_kwargs <= valid_case_fields
    assert case_kwargs == {"problem"}


def test_bench_composed_uses_benchmark_case_api_only() -> None:
    """The script must use the new benchmark/problem API, not deleted legacy names."""
    source = BENCH_COMPOSED.read_text(encoding="utf-8")

    assert "from q2mm.benchmarks.systems import load_system" in source
    assert "with_baseline(" in source
    legacy_type_name = "SystemData"
    legacy_module_name = "q2mm" + ".systems"
    assert legacy_type_name not in source
    assert legacy_module_name not in source
    assert "freq_ref" not in source
