"""Guard test: keep the architecture doc's module tree in sync with the code.

``docs/how-it-works/architecture.md`` contains a ``## Module organization``
tree that new contributors and AI agents treat as authoritative.  It has
drifted before (it listed a deleted ``optimizers/scoring.py`` and omitted
whole packages such as ``workflows/``).  These tests fail loudly the next
time the tree and the real package diverge, so the fix lands in the same PR
as the module change.

The checks compare **basenames** (e.g. ``forcefield.py``), not full paths,
so moving a module between directories without updating the doc is not
caught — but the high-value cases (a documented module that no longer
exists, or a real module that is undocumented) are.

Basename *occurrences* are counted rather than compared as sets: several
public modules legitimately share a basename (e.g. ``openmm.py`` lives
under both ``q2mm/io/`` and ``q2mm/backends/mm/``).  A set-membership
check would pass as long as *one* copy were documented, silently hiding a
missing duplicate; counting requires every copy to be listed.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

from test._shared import REPO_ROOT

ARCH_DOC = REPO_ROOT / "docs" / "how-it-works" / "architecture.md"
PACKAGE_ROOT = REPO_ROOT / "q2mm"

_PY_TOKEN = re.compile(r"([A-Za-z0-9_./-]+\.py)\b")
_DELETED_PHASE1_PATHS = (
    PACKAGE_ROOT / "systems.py",
    PACKAGE_ROOT / "models" / "loaders.py",
    PACKAGE_ROOT / "models" / "datum.py",
    PACKAGE_ROOT / "optimizers" / "reference.py",
    PACKAGE_ROOT / "optimizers" / "defaults.py",
)
_REQUIRED_PHASE2_PATHS = (
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
)
_DELETED_PHASE1_MODULES = frozenset(
    {
        "datum.py",
        "defaults.py",
        "loaders.py",
        "reference.py",
        "systems.py",
    }
)
_UNDOCUMENTED_PHASE2_MODULES = frozenset(
    {
        "cases.py",
        "ch3f.py",
        "ch3f_sn2.py",
        "heck_relay.py",
        "observations.py",
        "parameters.py",
        "pd_allyl.py",
        "pd_conjugate.py",
        "problem.py",
        "rh_conjugate.py",
        "rh_enamide.py",
    }
)


def _documented_module_counts() -> Counter[str]:
    """Count each ``*.py`` basename named in the doc's module tree."""
    lines = ARCH_DOC.read_text(encoding="utf-8").splitlines()
    heading = next(i for i, line in enumerate(lines) if line.strip() == "## Module organization")
    open_fence = next(i for i in range(heading + 1, len(lines)) if lines[i].startswith("```"))
    close_fence = next(i for i in range(open_fence + 1, len(lines)) if lines[i].startswith("```"))

    counts: Counter[str] = Counter()
    for line in lines[open_fence + 1 : close_fence]:
        code = line.split("#", 1)[0]  # ignore the inline description comment
        for token in _PY_TOKEN.findall(code):
            counts[token.rsplit("/", 1)[-1]] += 1
    return counts


def _real_module_counts() -> Counter[str]:
    """Count every ``*.py`` basename under ``q2mm/`` (excluding caches)."""
    return Counter(path.name for path in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def test_documented_modules_exist() -> None:
    """Every module named in the architecture doc must exist on disk.

    Occurrences are counted, so the doc listing a basename *more* times
    than it appears on disk (a phantom or stale duplicate) also fails.
    """
    documented = _documented_module_counts()
    real = _real_module_counts()
    missing = sorted(
        name for name, count in documented.items() if name not in _DELETED_PHASE1_MODULES and count > real[name]
    )
    assert not missing, (
        "architecture.md 'Module organization' lists modules that no longer "
        f"exist under q2mm/ (or lists more copies than exist on disk): {missing}. "
        "Update the doc tree in the same change that removed/renamed them."
    )


def test_real_modules_are_documented() -> None:
    """Every real public module must appear in the architecture doc tree.

    ``__init__.py`` files and private ``_``-prefixed helper modules are
    implementation details and are not required to be listed.

    Occurrences are counted, so a public module that appears N times on
    disk must be listed N times in the doc.  This catches the case where
    two public modules share a basename (e.g. ``openmm.py`` under both
    ``io/`` and ``backends/mm/``) but only one copy is documented — a
    set-membership check would miss it.
    """
    documented = _documented_module_counts()
    real = _real_module_counts()
    undocumented = sorted(
        name
        for name, count in real.items()
        if (
            name != "__init__.py"
            and not name.startswith("_")
            and name not in _UNDOCUMENTED_PHASE2_MODULES
            and documented[name] < count
        )
    )
    assert not undocumented, (
        "These q2mm modules are missing (or under-listed) in architecture.md's "
        f"'Module organization' tree: {undocumented}. Add every copy to the doc so "
        "the module map stays complete."
    )


def test_phase2_module_surface_is_present() -> None:
    """The phase-2 module split exists, and the removed phase-1 modules stay gone."""
    for path in _DELETED_PHASE1_PATHS:
        assert not path.exists(), f"{path.relative_to(REPO_ROOT)} should remain deleted in the phase-2 module surface."
    for path in _REQUIRED_PHASE2_PATHS:
        assert path.exists(), f"{path.relative_to(REPO_ROOT)} is part of the phase-2 surface and should exist on disk."


def test_module_tree_sanity() -> None:
    """The parser found a non-trivial tree (guards against a silent no-op)."""
    documented = _documented_module_counts()
    assert documented["forcefield.py"] == 1
    assert len(documented) > 30


# ---------------------------------------------------------------------------
# Import-direction guard: q2mm.models is the foundational layer
# ---------------------------------------------------------------------------

_MODELS_ROOT = PACKAGE_ROOT / "models"
# Every one of these depends on q2mm.models (parsers/serializers, MM
# backends, optimizers, workflows, benchmark registry, diagnostics) —
# never the other way around. A models/*.py file importing any of them
# would create a layering violation (and a real risk of import cycles).
_FORBIDDEN_OUTER_LAYERS = (
    "q2mm.io",
    "q2mm.backends",
    "q2mm.optimizers",
    "q2mm.workflows",
    "q2mm.benchmarks",
    "q2mm.diagnostics",
)


def _imported_dotted_modules(path: Path) -> set[str]:
    """Return every dotted module target imported anywhere in *path*.

    Walks the *entire* AST (``ast.walk``, not just ``tree.body``), so a
    lazy, function-scoped import is caught exactly the same as an eager,
    module-top-level one. ``q2mm.models`` must never reach an outer
    layer at all, not even lazily inside a function body to dodge an
    eager/circular-import cost, because the foundational layer must be
    importable (and reasoned about) without pulling in backends, I/O,
    optimizers, workflows, benchmarks, or diagnostics. A module that
    genuinely needs an optional heavy dependency (e.g. JAX) at call time
    should depend on a dependency-free foundational helper (see
    ``q2mm/_jax_support.py``, a top-level sibling of ``q2mm/models/``
    and ``q2mm/backends/`` — not "outer" to either), not a backend/io/etc.
    module.

    Regression guard for the specific violation this closed:
    ``q2mm/models/hessian.py`` used to lazily import
    ``q2mm.backends.mm._jax_common`` inside three function bodies, which
    a top-level-only AST scan could never see.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_models_package_never_imports_outer_layers() -> None:
    """``q2mm.models`` is the foundational layer; it must never import outer layers.

    Outer layers include ``q2mm.io``, ``q2mm.backends``, ``q2mm.optimizers``,
    ``q2mm.workflows``, ``q2mm.benchmarks``, and ``q2mm.diagnostics``, all of
    which depend on ``q2mm.models`` and not the reverse. This check walks
    the entire AST of every ``q2mm/models/*.py`` file, including nested
    function bodies, not just module-top-level statements, so a lazy,
    function-scoped import of an outer layer is caught too.

    Regression guards for two real phase-2 layering violations:

    1. ``q2mm/models/__init__.py`` re-exported ``q2mm.io.load_mm3_fld``
       and friends: format-specific loaders/savers belong to callers
       importing ``q2mm.io`` directly, never to ``q2mm.models`` itself.
    2. ``q2mm/models/hessian.py`` lazily imported
       ``q2mm.backends.mm._jax_common`` inside three function bodies to
       reach JAX. A genuinely lower-level, dependency-free helper
       (``q2mm/_jax_support.py``, a top-level sibling of both
       ``q2mm/models/`` and ``q2mm/backends/``, imported by both) now
       provides the one canonical JAX import guard, so neither layer
       re-implements it and neither depends on the other for it.
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
