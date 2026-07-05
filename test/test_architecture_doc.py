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

import re
from collections import Counter

from test._shared import REPO_ROOT

ARCH_DOC = REPO_ROOT / "docs" / "how-it-works" / "architecture.md"
PACKAGE_ROOT = REPO_ROOT / "q2mm"

_PY_TOKEN = re.compile(r"([A-Za-z0-9_./-]+\.py)\b")


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
        if name != "__init__.py" and not name.startswith("_") and documented[name] < count
    )
    assert not undocumented, (
        "These q2mm modules are missing (or under-listed) in architecture.md's "
        f"'Module organization' tree: {undocumented}. Add every copy to the doc so "
        "the module map stays complete."
    )


def test_module_tree_sanity() -> None:
    """The parser found a non-trivial tree (guards against a silent no-op)."""
    documented = _documented_module_counts()
    assert documented["forcefield.py"] == 1
    assert len(documented) > 30
