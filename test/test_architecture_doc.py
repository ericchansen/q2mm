"""Guard test: keep the architecture doc's module tree in sync with the code.

``docs/how-it-works/architecture.md`` contains a ``## Module organization``
tree that new contributors and AI agents treat as authoritative.  It has
drifted before (it listed a deleted ``optimizers/scoring.py`` and omitted
whole packages such as ``workflows/``).  These tests fail loudly the next
time the tree and the real package diverge, so the fix lands in the same PR
as the module change.

The module tree is checked by full package-relative path, so same-named
modules in different layers cannot conceal a missing or misplaced entry.
Import guards resolve relative imports as well as absolute imports, and
retired APIs are checked as executable imports rather than prose substrings.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from importlib.util import resolve_name
from pathlib import Path

import pytest

from test._shared import REPO_ROOT

ARCH_DOC = REPO_ROOT / "docs" / "how-it-works" / "architecture.md"
PACKAGE_ROOT = REPO_ROOT / "q2mm"

_TREE_ENTRY = re.compile(r"^(?P<indent>(?:│   |    )*)(?:├── |└── )(?P<name>\S+)")
_RETIRED_MODEL_PATHS = (
    PACKAGE_ROOT / "systems.py",
    PACKAGE_ROOT / "models" / "loaders.py",
    PACKAGE_ROOT / "models" / "datum.py",
    PACKAGE_ROOT / "optimizers" / "reference.py",
    PACKAGE_ROOT / "optimizers" / "defaults.py",
    PACKAGE_ROOT / "optimizers" / "objective.py",
    PACKAGE_ROOT / "optimizers" / "spec.py",
    PACKAGE_ROOT / "optimizers" / "jaxloss.py",
    PACKAGE_ROOT / "optimizers" / "evaluators",
)
_FOUNDATION_PATHS = (
    PACKAGE_ROOT / "benchmarks" / "cases.py",
    PACKAGE_ROOT / "benchmarks" / "systems" / "ch3f.py",
    PACKAGE_ROOT / "benchmarks" / "systems" / "ch3f_sn2.py",
    PACKAGE_ROOT / "benchmarks" / "systems" / "heck_relay.py",
    PACKAGE_ROOT / "benchmarks" / "systems" / "pd_allyl.py",
    PACKAGE_ROOT / "benchmarks" / "systems" / "pd_conjugate.py",
    PACKAGE_ROOT / "benchmarks" / "systems" / "rh_conjugate.py",
    PACKAGE_ROOT / "benchmarks" / "systems" / "rh_enamide.py",
    PACKAGE_ROOT / "models" / "observations.py",
    PACKAGE_ROOT / "models" / "parameters.py",
    PACKAGE_ROOT / "models" / "problem.py",
    PACKAGE_ROOT / "models" / "results.py",
    PACKAGE_ROOT / "objectives" / "plan.py",
    PACKAGE_ROOT / "objectives" / "protocols.py",
    PACKAGE_ROOT / "objectives" / "python.py",
    PACKAGE_ROOT / "objectives" / "jax.py",
    PACKAGE_ROOT / "objectives" / "metrics.py",
)
_BENCHMARK_PATHS = (
    PACKAGE_ROOT / "benchmarks" / "profiles.py",
    PACKAGE_ROOT / "benchmarks" / "acceptance.py",
    PACKAGE_ROOT / "benchmarks" / "analysis.py",
    PACKAGE_ROOT / "benchmarks" / "records.py",
    PACKAGE_ROOT / "benchmarks" / "artifacts.py",
    PACKAGE_ROOT / "benchmarks" / "runner.py",
    PACKAGE_ROOT / "benchmarks" / "cli.py",
)
_RETIRED_BENCHMARK_PATHS = (
    PACKAGE_ROOT / "benchmark_runner.py",
    PACKAGE_ROOT / "diagnostics",
    PACKAGE_ROOT / "diagnostics" / "benchmark.py",
    PACKAGE_ROOT / "diagnostics" / "cli.py",
    PACKAGE_ROOT / "diagnostics" / "report.py",
    PACKAGE_ROOT / "diagnostics" / "tables.py",
    PACKAGE_ROOT / "diagnostics" / "pes_distortion.py",
    REPO_ROOT / "scripts" / "benchmark.py",
)


def _documented_module_counts() -> Counter[str]:
    """Count package-relative Python paths in the indented module tree."""
    lines = ARCH_DOC.read_text(encoding="utf-8").splitlines()
    heading = next(i for i, line in enumerate(lines) if line.strip() == "## Module organization")
    open_fence = next(i for i in range(heading + 1, len(lines)) if lines[i].startswith("```"))
    close_fence = next(i for i in range(open_fence + 1, len(lines)) if lines[i].startswith("```"))

    counts: Counter[str] = Counter()
    directories: list[str] = []
    for line in lines[open_fence + 1 : close_fence]:
        code = line.split("#", 1)[0]
        entry = _TREE_ENTRY.match(code)
        if entry is None:
            assert ".py" not in code, f"Unrecognized module-tree entry: {line!r}"
            continue
        depth = len(entry["indent"]) // 4
        assert depth <= len(directories), f"Module-tree indentation has no parent: {line!r}"
        name = entry["name"]
        if name.endswith("/"):
            directories = [*directories[:depth], name.rstrip("/")]
        elif name.endswith(".py"):
            counts["/".join([*directories[:depth], name])] += 1
    return counts


def _real_module_counts() -> Counter[str]:
    """Count package-relative Python paths, excluding caches."""
    return Counter(
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_module_tree_records_full_parent_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    document = tmp_path / "architecture.md"
    document.write_text(
        "## Module organization\n\n```\nq2mm/\n├── first/\n│   └── shared.py\n└── second/\n    └── shared.py\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "ARCH_DOC", document)
    assert _documented_module_counts() == Counter({"first/shared.py": 1, "second/shared.py": 1})


def test_wrong_parent_directory_does_not_pass_module_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "q2mm"
    (package / "correct").mkdir(parents=True)
    (package / "correct" / "shared.py").write_text("", encoding="utf-8")
    document = tmp_path / "architecture.md"
    document.write_text("## Module organization\n\n```\nq2mm/\n└── wrong/\n    └── shared.py\n```\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "ARCH_DOC", document)
    monkeypatch.setitem(globals(), "PACKAGE_ROOT", package)
    with pytest.raises(AssertionError):
        test_documented_modules_exist()


def test_import_analysis_resolves_relative_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "q2mm"
    (package / "models").mkdir(parents=True)
    module = package / "models" / "probe.py"
    module.write_text(
        "from .. import benchmarks\n\ndef probe():\n    from ..backends.mm import openmm\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "PACKAGE_ROOT", package)
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    imported = _imported_dotted_modules(module)
    assert "q2mm.benchmarks" in imported
    assert "q2mm.backends.mm.openmm" in imported


def test_retired_import_detection_ignores_prose_but_checks_literal_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "q2mm"
    package.mkdir()
    module = package / "probe.py"
    module.write_text(
        "# Historical name: q2mm.diagnostics\nclass BenchmarkResult:\n    pass\n"
        'import importlib\nimportlib.import_module("q2mm.benchmark_runner")\n',
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "PACKAGE_ROOT", package)
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    imported = _imported_dotted_modules(module)
    assert "q2mm.diagnostics" not in imported
    assert "BenchmarkResult" not in imported
    assert "q2mm.benchmark_runner" in imported
    with pytest.raises(AssertionError, match="Retired API imports"):
        test_no_retired_api_imports_in_current_code()
    module.write_text(
        "# Historical name: q2mm.diagnostics\nclass BenchmarkResult:\n    pass\n",
        encoding="utf-8",
    )
    test_no_retired_api_imports_in_current_code()


def test_relative_import_cannot_bypass_the_layer_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "q2mm"
    (package / "io").mkdir(parents=True)
    (package / "io" / "probe.py").write_text("from ..optimizers import catalog\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "PACKAGE_ROOT", package)
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    with pytest.raises(AssertionError, match="import-direction violations"):
        test_layer_import_direction()


@pytest.mark.parametrize(
    "source",
    [
        "import importlib\nimportlib.import_module({absolute})",
        "import importlib as loader\nloader.import_module(name={absolute})",
        "from importlib import import_module\nimport_module({absolute})",
        "from importlib import import_module as load\nload(name={absolute})",
        "import importlib\nimportlib.import_module({relative}, 'q2mm')",
        "from importlib import import_module as load\nload(name={relative}, package='q2mm')",
        "__import__(name={absolute})",
        "import builtins as loader\nloader.__import__({absolute})",
        "from builtins import __import__ as load\nload(name={absolute})",
    ],
)
@pytest.mark.parametrize("target", ["q2mm.benchmark_runner", "q2mm.optimizers.catalog"])
def test_literal_dynamic_import_forms_cannot_bypass_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, target: str
) -> None:
    package = tmp_path / "q2mm"
    (package / "io").mkdir(parents=True)
    module = package / "io" / "probe.py"
    module.write_text(
        source.format(absolute=repr(target), relative=repr(target.removeprefix("q2mm"))), encoding="utf-8"
    )
    monkeypatch.setitem(globals(), "PACKAGE_ROOT", package)
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    assert target in _imported_dotted_modules(module)
    if target == "q2mm.benchmark_runner":
        with pytest.raises(AssertionError, match="Retired API imports"):
            test_no_retired_api_imports_in_current_code()
    else:
        with pytest.raises(AssertionError, match="import-direction violations"):
            test_layer_import_direction()


@pytest.mark.parametrize(
    "source",
    [
        "import importlib as loader\ndef ordinary(loader):\n    loader.import_module({target})",
        "from importlib import import_module as load\ndef ordinary(load):\n    load({target})",
        "def ordinary(__import__):\n    __import__({target})",
        "def ordinary(importlib):\n    importlib.import_module({target})",
        "from importlib import import_module as load\nload = lambda name: None\nload({target})",
        "def first():\n    from importlib import import_module as load\n    load('collections')\n"
        "def second(load):\n    load({target})",
    ],
)
def test_loader_aliases_do_not_escape_lexical_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    package = tmp_path / "q2mm"
    package.mkdir()
    module = package / "probe.py"
    target = "q2mm.benchmark_runner"
    module.write_text(source.format(target=repr(target)), encoding="utf-8")
    monkeypatch.setitem(globals(), "PACKAGE_ROOT", package)
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    assert target not in _imported_dotted_modules(module)


def test_literal_loader_alias_resolves_in_closures_and_late_module_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "q2mm"
    package.mkdir()
    module = package / "probe.py"
    module.write_text(
        "def outer():\n"
        "    from importlib import import_module as load\n"
        "    def inner():\n"
        "        load('.benchmark_runner', package='q2mm')\n"
        "def later():\n"
        "    library.import_module(name='q2mm.diagnostics')\n"
        "import importlib as library\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "PACKAGE_ROOT", package)
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    assert {"q2mm.benchmark_runner", "q2mm.diagnostics"} <= _imported_dotted_modules(module)


@pytest.mark.parametrize(
    "comprehension",
    [
        "[loader for loader in values]",
        "{loader for loader in values}",
        "{loader: loader for loader in values}",
        "(loader for loader in values)",
    ],
)
@pytest.mark.parametrize("function_scope", [False, True])
def test_comprehension_target_does_not_hide_outer_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, comprehension: str, function_scope: bool
) -> None:
    package = tmp_path / "q2mm"
    package.mkdir()
    module = package / "probe.py"
    statements = [comprehension, "loader.import_module('q2mm.benchmark_runner')"]
    body = "\n".join("    " + line for line in statements) if function_scope else "\n".join(statements)
    module.write_text(
        "import importlib as loader\n" + ("def probe():\n" if function_scope else "") + body + "\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "PACKAGE_ROOT", package)
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    assert "q2mm.benchmark_runner" in _imported_dotted_modules(module)
    with pytest.raises(AssertionError, match="Retired API imports"):
        test_no_retired_api_imports_in_current_code()


@pytest.mark.parametrize(
    ("expression", "detected"),
    [
        ("[loader('q2mm.benchmark_runner') for loader in values]", False),
        ("{loader('q2mm.benchmark_runner') for loader in values}", False),
        ("{loader: loader('q2mm.benchmark_runner') for loader in values}", False),
        ("(loader('q2mm.benchmark_runner') for loader in values)", False),
        ("[item for loader in loader('q2mm.benchmark_runner') for item in values]", True),
        ("[item for loader in values for item in loader('q2mm.benchmark_runner')]", False),
        ("[(loader := replacement) for item in values]\nloader('q2mm.benchmark_runner')", False),
    ],
)
def test_comprehension_loader_scope_matches_expression_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expression: str, detected: bool
) -> None:
    package = tmp_path / "q2mm"
    package.mkdir()
    module = package / "probe.py"
    module.write_text("from importlib import import_module as loader\n" + expression + "\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "PACKAGE_ROOT", package)
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    assert ("q2mm.benchmark_runner" in _imported_dotted_modules(module)) is detected


@pytest.mark.parametrize("outer_iterable", [False, True])
def test_class_comprehension_only_uses_class_bindings_in_outer_iterable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outer_iterable: bool
) -> None:
    package = tmp_path / "q2mm"
    package.mkdir()
    module = package / "probe.py"
    expression = (
        "[item for item in loader('q2mm.benchmark_runner')]"
        if outer_iterable
        else "[loader('q2mm.benchmark_runner') for item in values]"
    )
    module.write_text(
        "class Example:\n    from importlib import import_module as loader\n    result = " + expression + "\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "PACKAGE_ROOT", package)
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    assert ("q2mm.benchmark_runner" in _imported_dotted_modules(module)) is outer_iterable


@pytest.mark.parametrize(
    "pattern",
    [
        "loader",
        "{'item': item} as loader",
        "[*loader]",
        "{'item': item, **loader}",
        "{'item': [*loader]}",
    ],
)
@pytest.mark.parametrize("function_scope", [False, True])
def test_match_pattern_capture_is_not_an_import_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pattern: str, function_scope: bool
) -> None:
    package = tmp_path / "q2mm"
    package.mkdir()
    module = package / "probe.py"
    statements = f"match value:\n    case {pattern}:\n        loader('q2mm.benchmark_runner')\n"
    if function_scope:
        statements = "def probe(value):\n" + "".join("    " + line + "\n" for line in statements.splitlines())
    module.write_text("from importlib import import_module as loader\n" + statements, encoding="utf-8")
    monkeypatch.setitem(globals(), "PACKAGE_ROOT", package)
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    assert "q2mm.benchmark_runner" not in _imported_dotted_modules(module)


@pytest.mark.parametrize("pattern", ["_", "captured", "[*captured]", "{'item': item, **rest}"])
def test_unrelated_pattern_captures_keep_literal_loader_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pattern: str
) -> None:
    package = tmp_path / "q2mm"
    package.mkdir()
    module = package / "probe.py"
    module.write_text(
        f"from importlib import import_module as loader\n"
        f"match value:\n    case {pattern}:\n        loader('q2mm.benchmark_runner')\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "PACKAGE_ROOT", package)
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    assert "q2mm.benchmark_runner" in _imported_dotted_modules(module)


@pytest.mark.parametrize(
    ("source", "detected"),
    [
        (
            "def loader(name): pass\nclass Outer:\n    from importlib import import_module as loader\n"
            "    class Inner:\n        loader('q2mm.benchmark_runner')\n",
            False,
        ),
        (
            "from importlib import import_module as loader\nclass Outer:\n    loader = None\n"
            "    class Inner:\n        loader('q2mm.benchmark_runner')\n",
            True,
        ),
        (
            "class Outer:\n    from importlib import import_module as loader\n"
            "    class Inner(loader('q2mm.benchmark_runner')):\n        pass\n",
            True,
        ),
        (
            "class Outer:\n    class Inner:\n        from importlib import import_module as loader\n"
            "        loader('q2mm.benchmark_runner')\n",
            True,
        ),
    ],
)
def test_nested_class_loader_resolution_uses_lexical_not_outer_class_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, detected: bool
) -> None:
    package = tmp_path / "q2mm"
    package.mkdir()
    module = package / "probe.py"
    module.write_text(source, encoding="utf-8")
    monkeypatch.setitem(globals(), "PACKAGE_ROOT", package)
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    assert ("q2mm.benchmark_runner" in _imported_dotted_modules(module)) is detected


@pytest.mark.parametrize(
    "call",
    [
        "__import__('q2mm', fromlist=[{child}])",
        "__import__('q2mm', None, None, ({child},))",
        "__import__(name='q2mm', fromlist=['*', {child}])",
        "builtins.__import__('q2mm', fromlist=[{child}])",
        "builtin_import('q2mm', fromlist=[{child}])",
        "__import__('q2mm', fromlist={{{child}}})",
        "__import__('q2mm', fromlist={{{child}: None}})",
        "__import__('q2mm', fromlist=[*({child},)])",
        "__import__('q2mm', fromlist={{**{{{child}: None}}}})",
        "__import__('q2mm', fromlist={{{child}: computed_value}})",
    ],
)
@pytest.mark.parametrize("child", ["benchmark_runner", "optimizers"])
def test_literal_builtin_fromlist_cannot_bypass_retired_or_layer_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call: str, child: str
) -> None:
    package = tmp_path / "q2mm"
    (package / "io").mkdir(parents=True)
    module = package / "io" / "probe.py"
    module.write_text(
        "import builtins\nfrom builtins import __import__ as builtin_import\n" + call.format(child=repr(child)),
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "PACKAGE_ROOT", package)
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    assert f"q2mm.{child}" in _imported_dotted_modules(module)
    if child == "benchmark_runner":
        with pytest.raises(AssertionError, match="Retired API imports"):
            test_no_retired_api_imports_in_current_code()
    else:
        with pytest.raises(AssertionError, match="import-direction violations"):
            test_layer_import_direction()


@pytest.mark.parametrize(
    "fromlist",
    [
        "'benchmark_runner'",
        "[('benchmark_runner',)]",
        "{'safe': 'benchmark_runner'}",
        "computed_names",
        "b'benchmark_runner'",
    ],
)
def test_fromlist_analysis_does_not_invent_nonmember_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fromlist: str
) -> None:
    package = tmp_path / "q2mm"
    package.mkdir()
    module = package / "probe.py"
    module.write_text(f"__import__('q2mm', fromlist={fromlist})\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "PACKAGE_ROOT", package)
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    assert "q2mm.benchmark_runner" not in _imported_dotted_modules(module)


def test_package_initializer_relative_imports_use_its_own_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "q2mm"
    (package / "models").mkdir(parents=True)
    module = package / "models" / "__init__.py"
    module.write_text("from . import parameters\nfrom ..backends import contracts\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "PACKAGE_ROOT", package)
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    imported = _imported_dotted_modules(module)
    assert "q2mm.models.parameters" in imported
    assert "q2mm.backends.contracts" in imported


def test_documented_modules_exist() -> None:
    """Every module named in the architecture doc must exist on disk.

    Occurrences are counted, so the doc listing a basename *more* times
    than it appears on disk (a phantom or stale duplicate) also fails.
    """
    documented = _documented_module_counts()
    real = _real_module_counts()
    missing = sorted(name for name, count in documented.items() if count > real[name])
    assert not missing, (
        "architecture.md 'Module organization' lists modules that no longer "
        f"exist under q2mm/ (or lists more copies than exist on disk): {missing}. "
        "Update the doc tree in the same change that removed/renamed them."
    )


def test_real_modules_are_documented() -> None:
    """Every real public module must appear in the architecture doc tree.

    ``__init__.py`` files and private ``_``-prefixed helper modules are
    implementation details and are not required to be listed.

    Full paths distinguish same-named modules in different layers without
    forbidding legitimate names or hiding entries behind migration allowlists.
    """
    documented = _documented_module_counts()
    real = _real_module_counts()
    undocumented = sorted(
        name
        for name, count in real.items()
        if (
            name.rsplit("/", 1)[-1] != "__init__.py"
            and not name.rsplit("/", 1)[-1].startswith("_")
            and documented[name] < count
        )
    )
    assert not undocumented, (
        "These q2mm modules are missing (or under-listed) in architecture.md's "
        f"'Module organization' tree: {undocumented}. Add every copy to the doc so "
        "the module map stays complete."
    )


def test_foundational_module_surface_is_present() -> None:
    """Canonical foundational owners exist and their retired counterparts stay absent."""
    for path in _RETIRED_MODEL_PATHS:
        assert not path.exists(), f"{path.relative_to(REPO_ROOT)} should remain retired."
    for path in _FOUNDATION_PATHS:
        assert path.exists(), f"{path.relative_to(REPO_ROOT)} is a canonical foundational owner."


def test_benchmark_module_surface_is_present() -> None:
    """Canonical benchmark owners exist and superseded modules stay absent."""
    for path in _RETIRED_BENCHMARK_PATHS:
        assert not path.exists(), (
            f"{path.relative_to(REPO_ROOT)} was superseded by the q2mm.benchmarks package and should stay deleted."
        )
    for path in _BENCHMARK_PATHS:
        assert path.exists(), f"{path.relative_to(REPO_ROOT)} is a canonical benchmark owner."


_RETIRED_IMPORT_PREFIXES = (
    "q2mm.benchmark_runner",
    "q2mm.diagnostics",
    "q2mm.systems",
    "q2mm.models.loaders",
    "q2mm.models.datum",
    "q2mm.optimizers.reference",
    "q2mm.optimizers.defaults",
    "q2mm.optimizers.objective",
    "q2mm.optimizers.spec",
    "q2mm.optimizers.jaxloss",
    "q2mm.optimizers.evaluators",
    "q2mm.benchmarks.runner.TablePrinter",
    "q2mm.benchmarks.runner.BenchmarkResult",
    "q2mm.benchmarks.runner.run_combo",
)


def test_no_retired_api_imports_in_current_code() -> None:
    """Reject executable retired imports without banning words in prose or unrelated names."""
    offenders: dict[str, list[str]] = {}
    for root in ("q2mm", "scripts", "examples"):
        for path in (REPO_ROOT / root).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            hits = sorted(
                module
                for module in _imported_dotted_modules(path)
                if any(module == prefix or module.startswith(prefix + ".") for prefix in _RETIRED_IMPORT_PREFIXES)
            )
            if hits:
                offenders[path.relative_to(REPO_ROOT).as_posix()] = hits
    assert not offenders, f"Retired API imports remain in current code: {offenders}"


def test_canonical_benchmark_metric_and_result_paths() -> None:
    """Canonical owners exist at their declared paths, with retired paths absent."""
    real = _real_module_counts()
    assert real["benchmarks/runner.py"] == 1
    assert real["objectives/metrics.py"] == 1
    assert real["models/results.py"] == 1
    # The old parallel benchmark/diagnostics stack must be fully gone.
    assert real["benchmark_runner.py"] == 0
    assert real["diagnostics/tables.py"] == 0
    assert real["diagnostics/report.py"] == 0


def test_module_tree_sanity() -> None:
    """The parser found a non-trivial tree (guards against a silent no-op)."""
    documented = _documented_module_counts()
    assert documented["models/forcefield.py"] == 1
    assert len(documented) > 30


# ---------------------------------------------------------------------------
# Import-direction guard: q2mm.models is the foundational layer
# ---------------------------------------------------------------------------

_MODELS_ROOT = PACKAGE_ROOT / "models"
# Every one of these depends on q2mm.models (parsers/serializers, MM
# backends, optimizers, workflows, benchmark registry) — never the other
# way around. A models/*.py file importing any of them would create a
# layering violation (and a real risk of import cycles).
_FORBIDDEN_OUTER_LAYERS = (
    "q2mm.io",
    "q2mm.backends",
    "q2mm.optimizers",
    "q2mm.workflows",
    "q2mm.benchmarks",
)


def _scope_loader_bindings(
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda,
) -> dict[str, str | None]:
    """Resolve import bindings; ambiguous/reassigned names are not known loaders."""
    bindings: dict[str, str | None] = {}

    def bind(name: str, target: str | None) -> None:
        bindings[name] = target if name not in bindings or bindings[name] == target else None

    def collect(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bind(node.name, None)
            return
        if isinstance(node, ast.Lambda):
            return
        if isinstance(node, ast.comprehension):
            collect(node.iter)
            for condition in node.ifs:
                collect(condition)
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                bind(alias.asname or alias.name.split(".")[0], alias.name if alias.asname else alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    target = f"{node.module}.{alias.name}" if not node.level and node.module else None
                    bind(alias.asname or alias.name, target)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bind(node.id, None)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            bind(node.name, None)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bind(node.rest, None)
        for child in ast.iter_child_nodes(node):
            collect(child)

    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        for argument in (*scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs):
            bind(argument.arg, None)
        for argument in (scope.args.vararg, scope.args.kwarg):
            if argument is not None:
                bind(argument.arg, None)
    body = scope.body
    for node in body if isinstance(body, list) else [body]:
        collect(node)
    return bindings


def _literal_dynamic_imports(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    scopes: list[tuple[ast.AST, dict[str, str | None]]] = [(tree, _scope_loader_bindings(tree))]

    def loader_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            for _scope, bindings in reversed(scopes):
                if node.id in bindings:
                    return bindings[node.id]
            return "builtins.__import__" if node.id == "__import__" else None
        if isinstance(node, ast.Attribute):
            parent = loader_name(node.value)
            return f"{parent}.{node.attr}" if parent else None
        return None

    def call_argument(node: ast.Call, index: int, name: str) -> ast.expr | None:
        return (
            node.args[index]
            if len(node.args) > index
            else next((keyword.value for keyword in node.keywords if keyword.arg == name), None)
        )

    def literal_argument(node: ast.Call, index: int, name: str) -> str | None:
        value = call_argument(node, index, name)
        return value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else None

    def literal_fromlist_names(node: ast.expr | None) -> set[str]:
        names: set[str] = set()
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for item in node.elts:
                if isinstance(item, ast.Starred):
                    names.update(literal_fromlist_names(item.value))
                elif isinstance(item, ast.Constant) and isinstance(item.value, str):
                    names.add(item.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if key is None and isinstance(value, ast.Dict):
                    names.update(literal_fromlist_names(value))
                elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                    names.add(key.value)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.update(node.value)
        return names

    def walk(node: ast.AST) -> None:
        nonlocal scopes
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # The first iterable runs outside the implicit comprehension scope.
            walk(node.generators[0].iter)
            outer = scopes
            bindings: dict[str, str | None] = {
                target.id: None
                for generator in node.generators
                for target in ast.walk(generator.target)
                if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store)
            }
            scopes = [
                *[entry for entry in outer if not isinstance(entry[0], ast.ClassDef)],
                (node, bindings),
            ]
            for index, generator in enumerate(node.generators):
                if index:
                    walk(generator.iter)
                walk(generator.target)
                for condition in generator.ifs:
                    walk(condition)
            if isinstance(node, ast.DictComp):
                walk(node.key)
                walk(node.value)
            else:
                walk(node.elt)
            scopes = outer
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            body = node.body
            body_nodes = body if isinstance(body, list) else [body]
            for child in ast.iter_child_nodes(node):
                if child not in body_nodes:
                    walk(child)
            outer = scopes
            inherited = [entry for entry in outer if not isinstance(entry[0], ast.ClassDef)]
            scopes = [*inherited, (node, _scope_loader_bindings(node))]
            for child in body_nodes:
                walk(child)
            scopes = outer
            return
        if isinstance(node, ast.Call):
            loader = loader_name(node.func)
            if loader in ("importlib.import_module", "builtins.__import__"):
                name = literal_argument(node, 0, "name")
                if name and not name.startswith("."):
                    modules.add(name)
                    if loader == "builtins.__import__":
                        fromlist = call_argument(node, 3, "fromlist")
                        modules.update(
                            f"{name}.{child}" for child in literal_fromlist_names(fromlist) if child and child != "*"
                        )
                elif name and loader == "importlib.import_module":
                    package = literal_argument(node, 1, "package")
                    if package:
                        modules.add(resolve_name(name, package))
        for child in ast.iter_child_nodes(node):
            walk(child)

    walk(tree)
    return modules


def _imported_dotted_modules(path: Path) -> set[str]:
    """Resolve static and literal dynamic import targets throughout a module.

    Relative imports use the source package, not a basename approximation.
    Function-scoped imports and lexical loader aliases are included. Literal
    keyword names and importlib package-relative names resolve without
    executing imports. Reassigned/computed loaders or names require separate
    review; comments and unrelated strings are not imports.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if path.is_relative_to(PACKAGE_ROOT):
        package = ("q2mm", *path.relative_to(PACKAGE_ROOT).parent.parts)
    else:
        package = path.relative_to(REPO_ROOT).parent.parts
    modules = _literal_dynamic_imports(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                assert node.level <= len(package), f"Relative import escapes its package in {path}"
                prefix = package[: len(package) - node.level + 1]
                target = ".".join((*prefix, *((node.module or "").split(".") if node.module else ())))
            else:
                target = node.module or ""
            if target:
                modules.add(target)
                modules.update(f"{target}.{alias.name}" for alias in node.names if alias.name != "*")
    return modules


def test_models_package_never_imports_outer_layers() -> None:
    """``q2mm.models`` is the foundational layer; it must never import outer layers.

    Outer layers include ``q2mm.io``, ``q2mm.backends``, ``q2mm.optimizers``,
    ``q2mm.workflows``, and ``q2mm.benchmarks``, all of which depend on
    ``q2mm.models`` and not the reverse. This check walks the entire AST of
    every ``q2mm/models/*.py`` file, including nested function bodies, not
    just module-top-level statements, so a lazy, function-scoped import of
    an outer layer is caught too.

    Format-specific I/O belongs at the boundary. Dependency-free helpers
    such as ``q2mm._jax_support`` provide shared low-level support without
    making the foundational models depend on a concrete backend.
    """
    violations: dict[str, list[str]] = {}
    for path in sorted(_MODELS_ROOT.glob("*.py")):
        imported = _imported_dotted_modules(path)
        bad = sorted(
            mod
            for mod in imported
            if any(mod == layer or mod.startswith(layer + ".") for layer in _FORBIDDEN_OUTER_LAYERS)
        )
        if bad:
            violations[path.name] = bad
    assert not violations, (
        f"q2mm/models/*.py files import from outer layers, violating the "
        f"foundational-layer contract: {violations}. Move the dependency the "
        f"other way (the outer-layer module should import q2mm.models, not "
        f"the reverse), or introduce a dependency-free top-level helper (see "
        f"q2mm/_jax_support.py) for optional heavy-dependency imports."
    )


# ---------------------------------------------------------------------------
# Full layer-graph import-direction guard
# ---------------------------------------------------------------------------

# The dependency direction is "a package may import layers to its left":
#
#   constants/elements <- models <- backends <- objectives <- optimizers
#       <- workflows <- benchmarks/CLI
#
# q2mm.io is a model-dependent boundary composed by apps/benchmarks. Each
# entry maps a package directory to the dotted-module prefixes it must NOT
# import. Concrete backend *engines* (jax_engine/openmm/tinker/jax_md_engine)
# are called out explicitly for optimizers/workflows, which receive an
# evaluator + parameter space and never construct an engine.
_ENGINE_PREFIXES = (
    "q2mm.backends.mm.jax_engine",
    "q2mm.backends.mm.jax_md_engine",
    "q2mm.backends.mm.openmm",
    "q2mm.backends.mm.tinker",
    "q2mm.backends.qm",
)
_LAYER_FORBIDDEN: dict[str, tuple[str, ...]] = {
    "io": ("q2mm.objectives", "q2mm.optimizers", "q2mm.workflows", "q2mm.benchmarks"),
    "objectives": ("q2mm.optimizers", "q2mm.workflows", "q2mm.benchmarks"),
    "optimizers": (*_ENGINE_PREFIXES, "q2mm.workflows", "q2mm.benchmarks"),
    "workflows": (*_ENGINE_PREFIXES, "q2mm.benchmarks"),
}


def test_layer_import_direction() -> None:
    """Enforce the final left-to-right dependency direction across packages.

    Walks the entire AST of every module in each package (lazy and eager
    imports alike) and fails if any package imports a forbidden higher
    layer. This is the composition-root guarantee: benchmarks compose the
    lower layers, and no lower layer reaches back up into optimizers,
    workflows, benchmarks, or a concrete backend engine.
    """
    violations: dict[str, list[str]] = {}
    for package, forbidden in _LAYER_FORBIDDEN.items():
        for path in (PACKAGE_ROOT / package).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            bad = sorted(
                mod
                for mod in _imported_dotted_modules(path)
                if any(mod == layer or mod.startswith(layer + ".") for layer in forbidden)
            )
            if bad:
                violations[str(path.relative_to(PACKAGE_ROOT))] = bad
    assert not violations, f"import-direction violations (a package imports a higher layer): {violations}"


def test_benchmarks_is_the_composition_root() -> None:
    """No module outside q2mm.benchmarks may import q2mm.benchmarks."""
    offenders: dict[str, list[str]] = {}
    for path in PACKAGE_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(PACKAGE_ROOT)
        if rel.parts and rel.parts[0] == "benchmarks":
            continue
        bad = sorted(
            m for m in _imported_dotted_modules(path) if m == "q2mm.benchmarks" or m.startswith("q2mm.benchmarks.")
        )
        if bad:
            offenders[str(rel)] = bad
    assert not offenders, f"lower layers must not import the benchmarks composition root: {offenders}"


# ---------------------------------------------------------------------------
# Benchmark system registry: one concrete module per scientific system
# ---------------------------------------------------------------------------

_EXPECTED_SYSTEM_KEYS = frozenset(
    {"ch3f", "ch3f-sn2", "rh-enamide", "heck-relay", "pd-allyl", "pd-conjugate", "rh-conjugate", "ferrocene"}
)


def test_registry_maps_every_key_to_one_concrete_module() -> None:
    """Every registry key resolves to exactly one importable system module."""
    from q2mm.benchmarks.systems import SYSTEM_KEYS
    from q2mm.benchmarks.systems import _REGISTRY  # type: ignore[attr-defined]

    assert set(SYSTEM_KEYS) == _EXPECTED_SYSTEM_KEYS
    assert set(_REGISTRY) == _EXPECTED_SYSTEM_KEYS
    seen_modules: set[str] = set()
    systems_dir = PACKAGE_ROOT / "benchmarks" / "systems"
    for key, module_path in _REGISTRY.items():
        assert module_path not in seen_modules, f"registry key {key!r} shares a module with another key"
        seen_modules.add(module_path)
        rel = module_path.removeprefix("q2mm.benchmarks.systems.").replace(".", "/") + ".py"
        assert (systems_dir / rel).is_file(), f"registry key {key!r} points at missing module {module_path!r}"


def test_no_monolithic_systems_module() -> None:
    """The one-module-per-system split leaves no monolithic systems.py."""
    assert not (PACKAGE_ROOT / "benchmarks" / "systems.py").exists()
    assert not (PACKAGE_ROOT / "systems.py").exists()
    # CH3F ground state and CH3F-SN2 remain distinct modules.
    assert (PACKAGE_ROOT / "benchmarks" / "systems" / "ch3f.py").is_file()
    assert (PACKAGE_ROOT / "benchmarks" / "systems" / "ch3f_sn2.py").is_file()
