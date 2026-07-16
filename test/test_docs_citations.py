"""Citation audit: methodology pages must cite the primary QFUERZA/TS DOIs.

Every rewritten QFUERZA / transition-state methodology page must carry a
clickable link to the Zotero-verified primary sources, and the superseded
JCTC DOI must never stand in for the transition-state reference:

- QFUERZA: Farrugia et al. 2025 — https://doi.org/10.1021/acs.jctc.5c01751
- TS Hessian inversion / Method E2: Limé & Norrby (J. Comput. Chem. 36, 244;
  Zotero item UATW7GLJ) — https://doi.org/10.1002/jcc.23797
"""

from __future__ import annotations

from pathlib import Path

import pytest

from test._shared import REPO_ROOT

_FARRUGIA = "https://doi.org/10.1021/acs.jctc.5c01751"
_LIME_NORRBY = "https://doi.org/10.1002/jcc.23797"
_WRONG_TS_DOI = "10.1021/acs.jctc.5b00461"

# Methodology pages whose scientific claims rest on QFUERZA + TS inversion.
_QFUERZA_PAGES = (
    REPO_ROOT / "docs" / "how-it-works" / "theory.md",
    REPO_ROOT / "docs" / "benchmarks" / "qfuerza-recovery.md",
)

# Current docs + agent instructions that must not cite the wrong TS DOI.
_CURRENT_TEXT_FILES = (
    *(REPO_ROOT / "docs").rglob("*.md"),
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "README.md",
)


@pytest.mark.parametrize("page", _QFUERZA_PAGES, ids=lambda p: p.name)
def test_methodology_page_cites_both_primary_dois(page: Path) -> None:
    text = page.read_text(encoding="utf-8")
    assert _FARRUGIA in text, f"{page.name} is missing the QFUERZA (Farrugia 2025) DOI link {_FARRUGIA}"
    assert _LIME_NORRBY in text, f"{page.name} is missing the TS-inversion (Limé & Norrby) DOI link {_LIME_NORRBY}"


def test_agents_cites_correct_ts_doi() -> None:
    text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert _LIME_NORRBY in text, f"AGENTS.md must cite the TS reference DOI {_LIME_NORRBY}"


def test_wrong_ts_doi_absent_from_current_docs_and_instructions() -> None:
    offenders = [
        f.relative_to(REPO_ROOT).as_posix()
        for f in _CURRENT_TEXT_FILES
        if f.is_file() and _WRONG_TS_DOI in f.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"the superseded TS DOI {_WRONG_TS_DOI} must not appear in current docs/instructions: {offenders}"
    )
